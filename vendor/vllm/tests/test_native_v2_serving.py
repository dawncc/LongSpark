"""API boundary contracts; stdlib-only and executable without CUDA/vLLM."""

import asyncio
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longspark_vllm.serving import (
    LongSparkGreedyMiddleware, UnsupportedRequest, make_plan, normalize_request, parser,
)


class RequestPolicyTests(unittest.TestCase):
    def test_omitted_or_null_temperature_is_greedy_without_mutating_caller(self):
        for request in ({"prompt": [1, 2]}, {"temperature": None}, {"temperature": 0}):
            before = dict(request)
            result = normalize_request(request)
            self.assertEqual(result["temperature"], 0.0)
            self.assertEqual(request, before)

    def test_sampling_modifiers_are_rejected_including_nonfinite_and_bool(self):
        examples = {"temperature": [0.01, -1, "0", True, float("nan"), float("inf"), 10**400],
                    "top_p": [0.9], "top_k": [1, True], "min_p": [0.1],
                    "presence_penalty": [1], "frequency_penalty": [-1],
                    "repetition_penalty": [1.1], "min_tokens": [1], "n": [2], "best_of": [2],
                    "use_beam_search": [True], "logit_bias": [{"2": 4}],
                    "allowed_token_ids": [[2]], "structured_outputs": [{"regex": ".*"}],
                    "response_format": [{"type": "json_object"}], "tools": [[{}]],
                    "tool_choice": ["auto"], "modalities": [["audio"]]}
        for key, values in examples.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(UnsupportedRequest) as caught:
                    normalize_request({key: value})
                self.assertEqual(caught.exception.param, key)

    def test_min_tokens_with_many_stops_is_rejected_before_native_worker_limit(self):
        # vLLM 0.30 frontend accepts this; V2 LogitBiasState.add_request then
        # raises for its fixed 128-entry stop buffer inside the GPU worker.
        with self.assertRaises(UnsupportedRequest) as caught:
            normalize_request(dict(prompt="hello", min_tokens=1, max_tokens=16,
                                   stop_token_ids=list(range(129))))
        self.assertEqual(caught.exception.param, "min_tokens")

    def test_text_messages_and_common_openai_controls_are_preserved(self):
        request = dict(messages=[dict(role="user", content=[dict(type="text", text="hi")])],
                       stream=True, stream_options=dict(include_usage=True), max_tokens=10,
                       stop=["done"], seed=0, return_token_ids=True, response_format=dict(type="text"))
        self.assertEqual(normalize_request(request), dict(request, temperature=0.0))
        request["messages"][0]["content"] = [dict(type="image_url", image_url=dict(url="https://example.test/image"))]
        with self.assertRaises(UnsupportedRequest):
            normalize_request(request)

    def test_native_protocol_still_validates_non_object_json(self):
        for body in (None, 1, "text", []):
            with self.subTest(body=body), self.assertRaises(UnsupportedRequest):
                normalize_request(body)


class LaunchPlanTests(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(["--target", "/target", "--draft", "/draft",
                                   "--audit-dir", "/audit", *extra])

    def test_plan_pins_native_runner_async_graphs_and_model_defaults(self):
        plan = make_plan(self.args())
        argv = plan["argv"]
        self.assertEqual(plan["entrypoint"], "vllm.entrypoints.cli.main")
        self.assertEqual(argv[:2], ["serve", "/target"])
        self.assertIn("--async-scheduling", argv)
        self.assertIn("--no-enable-prefix-caching", argv)
        self.assertNotIn("--enforce-eager", argv)
        self.assertEqual(argv[argv.index("--kv-cache-memory-bytes") + 1], str(4 * 1024**3))
        self.assertEqual(argv[argv.index("--generation-config") + 1], "auto")
        self.assertEqual(json.loads(argv[argv.index("--override-generation-config") + 1]),
                         dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                              repetition_penalty=1.0, max_new_tokens=1024))
        self.assertEqual(plan["environment"]["VLLM_USE_V2_MODEL_RUNNER"], "1")
        self.assertEqual(plan["environment"]["VLLM_USE_RUST_FRONTEND"], "0")
        self.assertEqual(plan["environment"]["LONGSPARK_SERVE_PREFILL_WIDTH"], "512")

    def test_bad_limits_fail_before_loading_engine(self):
        for flags in (["--port", "0"], ["--gpu", "-1"], ["--kv-cache-gib", "nan"],
                      ["--max-num-seqs", "65"], ["--max-model-len", "64"],
                      ["--max-new-tokens", "0"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                make_plan(self.args(*flags))

    def test_preview_needs_no_vllm_and_does_not_create_audit_directory(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "not-created"
            completed = subprocess.run([sys.executable, str(root / "vendor/vllm/scripts/serve_native_v2.py"),
                                        "--target", "/target", "--draft", "/draft", "--audit-dir", str(audit)],
                                       capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout)["base_url"], "http://127.0.0.1:18080/v1")
            self.assertFalse(audit.exists())


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, body=b'{}', path="/v1/completions", *, method="POST", app=None, headers=()):
        output, seen = [], []

        async def native(scope, receive, send):
            seen.append((scope, await receive()))
            await send(dict(type="http.response.start", status=200, headers=[]))
            await send(dict(type="http.response.body", body=b"data: first\n\n", more_body=True))
            await send(dict(type="http.response.body", body=b"data: [DONE]\n\n", more_body=False))

        events = iter([dict(type="http.request", body=body[:2], more_body=True),
                       dict(type="http.request", body=body[2:], more_body=False),
                       dict(type="http.disconnect")])

        async def receive():
            return next(events)

        async def send(message):
            output.append(message)

        await LongSparkGreedyMiddleware(app or native)(
            dict(type="http", method=method, path=path, headers=list(headers)), receive, send)
        return output, seen

    async def test_valid_request_rewrites_length_and_preserves_stream_chunks(self):
        output, seen = await self.request(b'{"stream":true}', headers=[(b"content-length", b"15")])
        self.assertEqual(len(seen), 1)
        scope, event = seen[0]
        self.assertEqual(json.loads(event["body"]), dict(stream=True, temperature=0.0))
        self.assertEqual(dict(scope["headers"])[b"content-length"], str(len(event["body"])).encode())
        self.assertEqual([item.get("body") for item in output[1:]], [b"data: first\n\n", b"data: [DONE]\n\n"])

    async def test_bad_requests_never_reach_native_engine(self):
        for body in (b'{"temperature":0.7}', b'{"top_p":0.9}', b'[]', b'{broken', b'{"temperature":NaN}'):
            with self.subTest(body=body):
                output, seen = await self.request(body)
                self.assertEqual(output[0]["status"], 400)
                self.assertEqual(seen, [])
                self.assertIn("error", json.loads(output[1]["body"]))

    async def test_other_generation_and_management_endpoints_cannot_bypass_guard(self):
        for path in ("/v1/responses", "/v1/messages", "/inference/v1/generate", "/collective_rpc", "/sleep"):
            output, seen = await self.request(b'{}', path)
            self.assertEqual(output[0]["status"], 404)
            self.assertEqual(seen, [])

    async def test_health_models_and_metrics_are_native(self):
        for path in ("/health", "/v1/models", "/metrics"):
            output, seen = await self.request(b'{}', path, method="GET")
            self.assertEqual(output[0]["status"], 200)
            self.assertEqual(len(seen), 1)

    async def test_body_limit_and_compression_fail_before_engine(self):
        with patch("longspark_vllm.serving.BODY_LIMIT", 5):
            output, seen = await self.request(b'{"prompt":"too long"}')
            self.assertEqual(output[0]["status"], 413)
            self.assertEqual(seen, [])
        output, seen = await self.request(b'{}', headers=[(b"content-encoding", b"gzip")])
        self.assertEqual(output[0]["status"], 415)
        self.assertEqual(seen, [])

    async def test_disconnect_remains_available_to_native_stream_cancellation(self):
        observed = []

        async def native(scope, receive, send):
            observed.extend([await receive(), await receive()])

        await self.request(b'{}', app=native)
        self.assertEqual(observed[1]["type"], "http.disconnect")

    async def test_concurrent_requests_have_separate_replayed_bodies(self):
        seen = []

        async def native(scope, receive, send):
            await asyncio.sleep(0)
            seen.append(json.loads((await receive())["body"])["prompt"])

        await asyncio.gather(*(self.request(json.dumps(dict(prompt=index)).encode(), app=native)
                               for index in range(4)))
        self.assertEqual(sorted(seen), list(range(4)))

    async def test_lifespan_warms_before_ready_and_audits_before_native_cleanup(self):
        events = []

        class Engine:
            async def collective_rpc(self, name, **kwargs):
                events.append(name)
                return [dict(method=name, **kwargs)]

            def shutdown(self, timeout=None):
                events.append("engine_shutdown")

        scope = dict(type="lifespan", app=SimpleNamespace(state=SimpleNamespace(engine_client=Engine())))

        async def native(scope, receive, send):
            self.assertEqual((await receive())["type"], "lifespan.startup")
            await send(dict(type="lifespan.startup.complete"))
            self.assertEqual((await receive())["type"], "lifespan.shutdown")
            events.append("native_cleanup")
            await send(dict(type="lifespan.shutdown.complete"))

        incoming = iter([dict(type="lifespan.startup"), dict(type="lifespan.shutdown")])

        async def receive():
            return next(incoming)

        async def send(event):
            events.append(event["type"])

        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                "LONGSPARK_SERVE_AUDIT_DIR": temporary, "LONGSPARK_SERVE_PREFILL_WIDTH": "64"}):
            await LongSparkGreedyMiddleware(native)(scope, receive, send)
            self.assertEqual(events, ["native_v2_warm_prefill_shapes", "native_v2_audit_start",
                                      "lifespan.startup.complete", "native_v2_audit_end",
                                      "native_cleanup", "lifespan.shutdown.complete"])
            self.assertEqual(json.loads((Path(temporary) / "prefill-warmup.json").read_text())[0]["args"], [64])
            self.assertTrue((Path(temporary) / "shutdown-audit.json").is_file())

    async def shutdown_case(self, *, rpc_error=False, same_thread=False, rpc_delay=0,
                            max_wait=30):
        events = []

        class Engine:
            async def collective_rpc(self, name, **kwargs):
                events.append(name)
                if rpc_delay:
                    await asyncio.sleep(rpc_delay)
                if rpc_error:
                    raise RuntimeError("worker unavailable")
                return [dict(method=name)]

            def shutdown(self, timeout=None):
                events.append(("engine_shutdown", timeout))
                return "native-return"

        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                "LONGSPARK_SERVE_AUDIT_DIR": temporary}), patch("longspark_vllm.serving.AUDIT_TIMEOUT", max_wait):
            middleware = LongSparkGreedyMiddleware(None)
            middleware.audit_started = True
            engine = Engine()
            middleware._install_shutdown_hook(engine)
            self.assertFalse(inspect.iscoroutinefunction(engine.shutdown))
            result = engine.shutdown(timeout=9) if same_thread else await asyncio.to_thread(engine.shutdown, timeout=9)
            self.assertEqual(result, "native-return")
            # vLLM's context-manager cleanup calls again from the loop thread.
            self.assertEqual(engine.shutdown(timeout=0), "native-return")
            # Uvicorn's ASGI shutdown is later still; no RPC after worker close.
            await middleware._shutdown(engine)
            self.assertTrue(middleware.stopping)
            files = {path.name: json.loads(path.read_text()) for path in Path(temporary).glob("*.json")}
            return events, files

    async def test_native_sync_shutdown_thread_collects_before_engine_exit_once(self):
        events, files = await self.shutdown_case()
        self.assertEqual(events, ["native_v2_audit_end", ("engine_shutdown", 9), ("engine_shutdown", 0)])
        self.assertIn("shutdown-audit.json", files)
        self.assertNotIn("shutdown-audit-error.json", files)

    async def test_rpc_failure_still_calls_original_shutdown_and_marks_incomplete(self):
        events, files = await self.shutdown_case(rpc_error=True)
        self.assertEqual(events[-2:], [("engine_shutdown", 9), ("engine_shutdown", 0)])
        self.assertFalse(files["shutdown-audit-error.json"]["graph_audit_complete"])
        self.assertNotIn("shutdown-audit.json", files)

    async def test_same_event_loop_shutdown_never_deadlocks_or_retries_after_close(self):
        events, files = await self.shutdown_case(same_thread=True)
        self.assertEqual(events, [("engine_shutdown", 9), ("engine_shutdown", 0)])
        self.assertIn("event-loop thread", files["shutdown-audit-error.json"]["error"])

    async def test_rpc_timeout_does_not_prevent_native_shutdown(self):
        events, files = await self.shutdown_case(rpc_delay=60, max_wait=0.01)
        self.assertEqual(events[-2:], [("engine_shutdown", 9), ("engine_shutdown", 0)])
        self.assertFalse(files["shutdown-audit-error.json"]["graph_audit_complete"])
        self.assertNotIn("shutdown-audit.json", files)

    async def test_shutdown_rejects_new_generation_without_calling_native(self):
        async def native(*args):
            self.fail("A stopping server must reject new generation")

        middleware = LongSparkGreedyMiddleware(native)
        middleware.stopping = True
        output, _ = await self.request(b'{}', app=middleware)
        self.assertEqual(output[0]["status"], 503)


if __name__ == "__main__":
    unittest.main()
