"""CPU contracts for native API evidence, token parity and SSE framing."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "validate_native_v2_api", Path(__file__).resolve().parents[1] / "scripts/validate_native_v2_api.py")
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)
from longspark_vllm.gsm8k import canonical_prompt


def example():
    question = "What is one plus one?"
    return dict(source_id="gsm8k-test-0000", dataset_index=0, question=question,
                reference_answer="#### 2", gold_answer="2", prompt=canonical_prompt(question),
                input_ids=[11, 12])


def response(chat=False):
    choice = dict(index=0, token_ids=[21, 151645], finish_reason="stop")
    result = dict(id="req-1", model="test", choices=[choice],
                  usage=dict(prompt_tokens=2, completion_tokens=2, total_tokens=4))
    if chat:
        choice["message"] = dict(role="assistant", content="#### 2")
        result["prompt_token_ids"] = [11, 12]
    else:
        choice.update(text="#### 2", prompt_token_ids=[11, 12])
    return result


def events(chat=False):
    result = []
    for index, (token, text) in enumerate(((21, "#### 2"), (151645, ""))):
        choice = dict(index=0, token_ids=[token], finish_reason="stop" if index else None)
        event = dict(id="req-1", model="test", choices=[choice])
        if chat:
            choice["delta"] = dict(content=text)
            if not index:
                event["prompt_token_ids"] = [11, 12]
        else:
            choice["text"] = text
            if not index:
                choice["prompt_token_ids"] = [11, 12]
        result.append(event)
    result.append(dict(id="req-1", model="test", choices=[],
                       usage=dict(prompt_tokens=2, completion_tokens=2, total_tokens=4)))
    return result


def sse(items):
    return "".join("data: " + json.dumps(item) + "\n\n" for item in items) + "data: [DONE]\n\n"


class SSETests(unittest.TestCase):
    def test_json_and_sse_match_for_both_endpoints(self):
        for chat in (False, True):
            with self.subTest(chat=chat):
                plain = api.normalize_response(json.dumps(response(chat)), chat=chat, stream=False)
                streamed = api.normalize_response(sse(events(chat)), chat=chat, stream=True)
                for key in ("output_ids", "output_text", "input_ids", "finish_reason", "usage"):
                    self.assertEqual(plain[key], streamed[key])

    def test_crlf_comments_and_multiline_data(self):
        raw = ': ping\r\nevent: message\r\ndata: {"x":\r\ndata: 1}\r\n\r\ndata: [DONE]\r\n\r\n'
        self.assertEqual(api.parse_sse(raw.encode()), [{"x": 1}])

    def test_missing_duplicate_or_out_of_order_done(self):
        for raw in ('data: {}\n\n', 'data: [DONE]\n\ndata: [DONE]\n\n',
                    'data: [DONE]\n\ndata: {}\n\n'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                api.parse_sse(raw)

    def test_unterminated_frame(self):
        for suffix in ("", "\n"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "incomplete frame"):
                api.parse_sse('data: {}\n\ndata: [DONE]' + suffix)

    def test_server_error_inside_successful_http_stream(self):
        with self.assertRaisesRegex(ValueError, "error"):
            api.parse_sse(sse([dict(error=dict(message="engine died"))]))

    def test_wrong_usage_missing_ids_and_truncation_fail(self):
        for change in (lambda row: row["usage"].update(completion_tokens=3),
                       lambda row: row["choices"][0].pop("token_ids"),
                       lambda row: row["choices"][0].update(finish_reason="length"),
                       lambda row: row["choices"][0].update(token_ids=[True, 2]),
                       lambda row: row["choices"][0].update(index=1)):
            row = response()
            change(row)
            with self.subTest(response=row), self.assertRaises(ValueError):
                api.normalize_response(json.dumps(row), chat=False, stream=False)

    def test_mixed_request_ids_and_models_fail(self):
        for key in ("id", "model"):
            rows = events()
            rows[1][key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                api.normalize_response(sse(rows), chat=False, stream=True)

    def test_duplicate_finish_and_changed_prompt_fail(self):
        for change in (lambda rows: rows[0]["choices"][0].update(finish_reason="stop"),
                       lambda rows: rows[1]["choices"][0].update(prompt_token_ids=[999])):
            rows = events()
            change(rows)
            with self.assertRaises(ValueError):
                api.normalize_response(sse(rows), chat=False, stream=True)

    def test_thinking_output_is_not_silently_ignored(self):
        row = response(True)
        row["choices"][0]["message"]["reasoning"] = "hidden thought"
        with self.assertRaisesRegex(ValueError, "thinking"):
            api.normalize_response(json.dumps(row), chat=True, stream=False)


class ValidationTests(unittest.TestCase):
    def test_tokenized_completions_and_chat_messages_preserve_fixture(self):
        fixture = example()
        for chat in (False, True):
            body = api.request_payload(fixture, "test", chat=chat, stream=True, max_tokens=128)
            self.assertTrue(body["return_token_ids"])
            self.assertTrue(body["stream_options"]["include_usage"])
            self.assertEqual(body["temperature"], 0)
            if chat:
                self.assertEqual(body["messages"], [dict(role="user", content=fixture["prompt"])])
                self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
            else:
                self.assertEqual(body["prompt"], fixture["input_ids"])

    def test_minimum_length_large_stop_list_regression_case(self):
        cases = dict(api.unsupported_cases())
        self.assertEqual(cases["min_tokens_with_many_stop_ids"],
                         {"min_tokens": 1, "stop_token_ids": list(range(129))})

    def test_exact_baseline_tokens_and_text(self):
        normalized = api.normalize_response(json.dumps(response()), chat=False, stream=False)
        record = api.validate_record(example(), normalized, normalized, max_tokens=10, eos_token_ids=[151645])
        self.assertTrue(record["correct"])
        self.assertTrue(record["baseline_exact_token_parity"])
        for key, replacement in (("output_ids", [22, 151645]), ("output_text", "#### 3")):
            baseline = dict(normalized, **{key: replacement})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "baseline"):
                api.validate_record(example(), normalized, baseline, max_tokens=10, eos_token_ids=[151645])

    def test_retokenized_prompt_cannot_count_as_parity(self):
        normalized = api.normalize_response(json.dumps(response()), chat=False, stream=False)
        normalized["input_ids"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "prompt token IDs"):
            api.validate_record(example(), normalized, None, max_tokens=10, eos_token_ids=[151645])

    def test_baseline_duplicate_or_missing_source_rejected(self):
        row = dict(source_id="gsm8k-test-0000", output_ids=[21], output_text="hi", prompt_tokens=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                api.load_baseline(path, [example()])
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "cover every"):
                api.load_baseline(path, [example()])

    def test_existing_output_cannot_be_modified_even_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text('{"existing":true}')
            args = api.parser().parse_args(["--output", str(root)])
            with patch.object(api, "parser") as parser:
                parser.return_value.parse_args.return_value = args
                with self.assertRaises(FileExistsError):
                    api.main()
            self.assertEqual((root / "plan.json").read_text(), '{"existing":true}')
            self.assertFalse((root / "failure.json").exists())


class MetricsTests(unittest.TestCase):
    def test_counters_aggregate_engines_but_ignore_histograms(self):
        raw = '\n'.join([
            '# HELP vllm:num_requests_running running requests',
            'vllm:spec_decode_num_drafts_total{model_name="test",engine="0"} 10',
            'vllm:spec_decode_num_drafts_total{model_name="test",engine="1"} 5',
            'vllm:spec_decode_num_draft_tokens_total{model_name="test"} 105',
            'vllm:spec_decode_num_accepted_tokens_total{model_name="test"} 60',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 15',
            'vllm:num_requests_running{model_name="test"} 4',
        ])
        result = api.metric_values(raw)
        self.assertEqual(result[api.METRICS["rounds"]], 15)
        before = {key: 0 for key in result}
        delta = api.acceptance_delta(before, result)
        self.assertEqual(delta["tau"], 5)
        self.assertEqual(delta["accepted_per_round"], 4)
        self.assertEqual(result[api.METRICS["running"]], 4)

    def test_invalid_or_reset_counters_fail(self):
        for number in ("NaN", "Inf", "-1"):
            with self.subTest(number=number), self.assertRaises(ValueError):
                api.metric_values(f'{api.METRICS["rounds"]} {number}')
        after = {api.METRICS[key]: value for key, value in (("rounds", 10), ("proposed", 70), ("accepted", 20))}
        with self.assertRaises(ValueError):
            api.acceptance_delta(after, {key: value - 1 for key, value in after.items()})

    def test_absent_metrics_report_unavailable(self):
        self.assertEqual(api.acceptance_delta({}, {}), {"available": False})

    def test_overlap_does_not_count_adjacent_requests_as_concurrent(self):
        self.assertEqual(api.maximum_overlap([(1, 2), (2, 3)]), 1)
        self.assertEqual(api.maximum_overlap([(1, 4), (2, 5), (3, 6)]), 3)
        with self.assertRaises(ValueError):
            api.maximum_overlap([(2, 1)])


class ClientIntegrationTests(unittest.TestCase):
    def test_real_http_transport_full_smoke_and_recovery(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, status, raw, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def authorized(self):
                if self.headers.get("Authorization") == "Bearer test-key":
                    return True
                self.send(401, b'{"error":{"message":"key required"}}')
                return False

            def do_GET(self):
                if not self.authorized():
                    return
                if self.path == "/health":
                    self.send(200, b"")
                elif self.path == "/v1/models":
                    self.send(200, b'{"data":[{"id":"test"}]}')
                elif self.path == "/metrics":
                    self.send(200, b"vllm:num_requests_running 4\n", "text/plain")
                else:
                    self.send(404, b"{}")

            def do_POST(self):
                if not self.authorized():
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if (body.get("temperature") != 0 or body.get("top_p") != 1 or
                        body.get("repetition_penalty", 1) != 1 or body.get("n") != 1 or
                        body.get("min_tokens", 0) > 0):
                    self.send(400, b'{"error":{"message":"greedy only"}}')
                    return
                time.sleep(0.02)  # CPU-only fake work makes true client overlap observable.
                chat = self.path == "/v1/chat/completions"
                raw = sse(events(chat)) if body.get("stream") else json.dumps(response(chat))
                self.send(200, raw.encode(), "text/event-stream" if body.get("stream") else "application/json")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = root / "input.jsonl"
                fixture.write_text(json.dumps(example()) + "\n")
                baseline = root / "baseline.jsonl"
                baseline.write_text(json.dumps(dict(source_id="gsm8k-test-0000", output_ids=[21, 151645],
                                                     output_text="#### 2", prompt_tokens=2)) + "\n")
                args = api.parser().parse_args([
                    "--base-url", f"http://127.0.0.1:{server.server_port}", "--model", "test",
                    "--api-key", "test-key", "--input", str(fixture), "--baseline", str(baseline),
                    "--output", str(root / "run"), "--limit", "1", "--metrics-settle-seconds", "0",
                    "--check-unsupported"])
                summary = api.run(args)
                self.assertTrue(summary["passed"])
                self.assertTrue(summary["baseline_exact_token_parity"])
                self.assertEqual(summary["successful_generations"], 9)
                self.assertEqual(summary["max_client_overlap"], 4)
                self.assertEqual(len(summary["rejection_cases"]), 10)
                self.assertTrue((root / "run/recovery-after-rejections/response.sse").is_file())
                self.assertNotIn("test-key", (root / "run/plan.json").read_text())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
