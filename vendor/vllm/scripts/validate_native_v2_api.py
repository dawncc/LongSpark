#!/usr/bin/env python3
"""Validate native vLLM HTTP JSON/SSE parity, concurrency and rejection safety.

This stdlib-only client runs against an already started server. It never starts
an engine and refuses to overwrite an existing evidence directory. HTTP timings
include the frontend and network; they are not engine-only decode timings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vendor/vllm"))
from longspark_vllm.gsm8k import load_inputs, score_record

METRICS = {
    "rounds": "vllm:spec_decode_num_drafts_total",
    "proposed": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "running": "vllm:num_requests_running",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def token_ids(value, name):
    if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
        raise ValueError(f"{name} must be a list of nonnegative integer token IDs")
    return value


def parse_sse(raw):
    """Parse complete SSE frames, including CRLF and multi-line data fields."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    frames, fields = [], []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if not line:
            if fields:
                frames.append("\n".join(fields))
                fields = []
        elif line.startswith("data:"):
            fields.append(line[5:].removeprefix(" "))
        elif not line.startswith((":", "event:", "id:", "retry:")):
            raise ValueError(f"Unexpected SSE line: {line[:80]}")
    if fields:
        raise ValueError("SSE response ended with an incomplete frame")
    if not frames or frames[-1] != "[DONE]" or frames.count("[DONE]") != 1:
        raise ValueError("SSE must end with exactly one [DONE] frame")
    events = [json.loads(frame) for frame in frames[:-1]]
    if any(not isinstance(event, dict) or "error" in event for event in events):
        raise ValueError("SSE includes an error or non-object event")
    return events


def normalize_response(payload, *, chat, stream):
    """Require native token tracing and collect exactly one complete choice."""
    events = parse_sse(payload) if stream else [json.loads(payload)]
    ids, texts, prompts, finishes, usages = [], [], [], [], []
    request_ids, models = set(), set()
    native_metrics = None
    for event in events:
        if not isinstance(event, dict) or "error" in event:
            raise ValueError("Response contains an API error")
        if event.get("id"):
            request_ids.add(event["id"])
        if event.get("model"):
            models.add(event["model"])
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected a single choice or a streaming usage event")
        if event.get("usage") is not None:
            usages.append(event["usage"])
        if event.get("metrics") is not None:
            native_metrics = event["metrics"]
        if chat and event.get("prompt_token_ids") is not None:
            prompts.append(token_ids(event["prompt_token_ids"], "prompt_token_ids"))
        for choice in choices:
            if choice.get("index") != 0:
                raise ValueError("Expected choice index 0")
            if choice.get("token_ids") is not None:
                ids.extend(token_ids(choice["token_ids"], "token_ids"))
            if not chat and choice.get("prompt_token_ids") is not None:
                prompts.append(token_ids(choice["prompt_token_ids"], "prompt_token_ids"))
            if chat:
                message = choice.get("delta" if stream else "message")
                if not isinstance(message, dict):
                    raise ValueError("Chat response is missing its message")
                if message.get("reasoning") or message.get("reasoning_content"):
                    raise ValueError("Unexpected thinking output for enable_thinking=False")
                text = message.get("content") or ""
            else:
                text = choice.get("text")
            if not isinstance(text, str):
                raise ValueError("Response content must be text")
            texts.append(text)
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    if len(request_ids) != 1 or len(models) != 1:
        raise ValueError("Response request ID or model is missing/inconsistent")
    if not ids or not prompts or any(prompt != prompts[0] for prompt in prompts):
        raise ValueError("Response is missing token IDs or has inconsistent prompt IDs")
    if len(finishes) != 1 or finishes[0] != "stop":
        raise ValueError("Expected one natural stop; truncated outputs are not parity evidence")
    if not usages:
        raise ValueError("Response is missing final token usage")
    usage = usages[-1]
    if (usage.get("prompt_tokens") != len(prompts[0]) or
            usage.get("completion_tokens") != len(ids) or
            usage.get("total_tokens") != len(prompts[0]) + len(ids)):
        raise ValueError("Usage token counts differ from returned token IDs")
    return dict(request_id=next(iter(request_ids)), model=next(iter(models)),
                output_ids=ids, output_text="".join(texts), input_ids=prompts[0],
                finish_reason=finishes[0], usage=usage, event_count=len(events),
                native_metrics=native_metrics)


def request_payload(example, model, *, chat, stream, max_tokens):
    body = dict(model=model, temperature=0, top_p=1, n=1, max_tokens=max_tokens,
                return_token_ids=True, seed=0, stream=stream)
    if chat:
        body.update(messages=[dict(role="user", content=example["prompt"])],
                    chat_template_kwargs=dict(enable_thinking=False))
    else:
        body["prompt"] = example["input_ids"]
    if stream:
        body["stream_options"] = dict(include_usage=True)
    return body


def unsupported_cases():
    # Combined minimum-length and many stop IDs previously reached a fixed-size
    # native worker buffer. The guard must reject before engine dispatch.
    return [("temperature", {"temperature": 0.7}),
            ("top_p", {"top_p": 0.9}),
            ("repetition_penalty", {"repetition_penalty": 1.1}),
            ("n", {"n": 2}),
            ("min_tokens_with_many_stop_ids", {"min_tokens": 1, "stop_token_ids": list(range(129))})]


def validate_record(example, normalized, baseline, *, max_tokens, eos_token_ids):
    if normalized["input_ids"] != example["input_ids"]:
        raise ValueError("API prompt token IDs differ from the shared GSM8K fixture")
    record = score_record(example, dict(source_id=example["source_id"], **normalized),
                          eos_token_ids=eos_token_ids, max_new_tokens=max_tokens)
    if not record["eos_at_end"] or record["tokens_after_eos"]:
        raise ValueError("API completion did not terminate at a natural EOS token")
    if baseline is not None:
        if normalized["output_ids"] != baseline["output_ids"]:
            raise ValueError(f"{example['source_id']}: API output IDs differ from baseline")
        if normalized["output_text"] != baseline["output_text"]:
            raise ValueError(f"{example['source_id']}: API output text differs from baseline")
        record["baseline_exact_token_parity"] = True
    else:
        record["baseline_exact_token_parity"] = None
    return record


def metric_values(raw):
    values = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?', line)
        if match and match[1] in METRICS.values():
            number = float(match[2])
            if not math.isfinite(number) or number < 0:
                raise ValueError("Metrics contain a non-finite or negative counter")
            values[match[1]] = values.get(match[1], 0.0) + number
    return values


def acceptance_delta(before, after):
    keys = [METRICS[name] for name in ("rounds", "proposed", "accepted")]
    if not all(key in before and key in after for key in keys):
        return dict(available=False)
    rounds, proposed, accepted = [after[key] - before[key] for key in keys]
    if min(rounds, proposed, accepted) < 0 or accepted > proposed:
        raise ValueError("Speculative counter delta is inconsistent")
    return dict(available=True, verification_rounds=rounds, proposed_tokens=proposed,
                accepted_tokens=accepted,
                accepted_per_round=accepted / rounds if rounds else None,
                tau=1 + accepted / rounds if rounds else None,
                acceptance_rate=accepted / proposed if proposed else None,
                definition="Server-global raw counter delta for this entire validation run; final EOS clipping is not applied")


def maximum_overlap(intervals):
    if any(end < start for start, end in intervals):
        raise ValueError("Request interval runs backwards")
    events = sorted((time_, delta) for start, end in intervals
                    for time_, delta in ((start, 1), (end, -1)))
    count = highest = 0
    for _, delta in events:
        count += delta
        highest = max(highest, count)
    return highest


class Client:
    def __init__(self, base_url, *, api_key=None, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.api_key, self.timeout = api_key, timeout

    def request(self, route, body=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.base_url + route, data=json.dumps(body).encode() if body is not None else None,
                          headers=headers)
        start = time.perf_counter()
        try:
            response = urlopen(request, timeout=self.timeout)
        except HTTPError as error:
            response = error
        with response:
            chunks, line_timings = [], []
            stream = bool(body and body.get("stream")) and response.status == 200
            if stream:
                for line in response:
                    chunks.append(line)
                    line_timings.append(dict(seconds=time.perf_counter() - start, bytes=len(line)))
            else:
                chunks.append(response.read())
            end = time.perf_counter()
            return dict(status=response.status, headers=dict(response.headers),
                        raw=b"".join(chunks), start=start, end=end, seconds=end - start,
                        line_timings=line_timings)


def save_exchange(directory, body, response):
    directory.mkdir()
    write_json(directory / "request.json", body)
    (directory / ("response.sse" if body and body.get("stream") and response["status"] == 200
                  else "response.json")).write_bytes(response["raw"])
    write_json(directory / "transport.json", {key: value for key, value in response.items() if key != "raw"})


def load_baseline(path, examples):
    if path is None:
        return {}
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        source_id = row["source_id"]
        if source_id in records:
            raise ValueError("Duplicate source ID in baseline")
        records[source_id] = row
    for example in examples:
        if example["source_id"] not in records:
            raise ValueError("Baseline does not cover every selected example")
        row = records[example["source_id"]]
        token_ids(row["output_ids"], "baseline output_ids")
        if row.get("prompt_tokens") != len(example["input_ids"]):
            raise ValueError("Baseline prompt token count differs from fixture")
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:18080")
    p.add_argument("--model", default="longspark-qwen3-4b")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    p.add_argument("--input", type=Path, default=ROOT / "vendor/vllm/validation/gsm8k100/inputs.jsonl")
    p.add_argument("--baseline", type=Path, help="Offline vanilla/records.jsonl to require exact token/text parity")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--metrics-settle-seconds", type=float, default=6,
                   help="Wait before counter snapshots; must exceed server stats interval (default 5s)")
    p.add_argument("--eos-token-id", type=int, action="append", dest="eos_token_ids")
    p.add_argument("--check-unsupported", action="store_true",
                   help="Only enable after LongSpark request guard is installed; require HTTP 4xx before engine dispatch")
    return p


def run(args):
    if args.concurrency < 2 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Require concurrency >=2 and finite timeout >0")
    if not math.isfinite(args.metrics_settle_seconds) or not 0 <= args.metrics_settle_seconds <= 60:
        raise ValueError("metrics-settle-seconds must be finite and between 0 and 60")
    examples, digest = load_inputs(args.input, limit=args.limit,
                                  max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len)
    baseline = load_baseline(args.baseline, examples)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            if key != "api_key"}
    plan.update(input_sha256=digest, api_key_configured=bool(args.api_key),
                baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest() if args.baseline else None,
                timing_definition="Client wall time including HTTP serialization, scheduling and network",
                metrics_scope="Dedicated server required: /metrics counters are server-global",
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output / "plan.json", plan)
    client = Client(args.base_url, api_key=args.api_key, timeout=args.timeout)
    eos = args.eos_token_ids or [151643, 151645]
    records, negative = [], []

    def get(route, name):
        response = client.request(route)
        save_exchange(args.output / name, None, response)
        if response["status"] != 200:
            raise ValueError(f"{route} returned HTTP {response['status']}")
        return response["raw"]

    get("/health", "health-before")
    models = json.loads(get("/v1/models", "models"))
    if args.model not in [item.get("id") for item in models.get("data", [])]:
        raise ValueError("Requested model is absent from /v1/models")
    time.sleep(args.metrics_settle_seconds)
    before_response = client.request("/metrics")
    save_exchange(args.output / "metrics-before", None, before_response)
    before = metric_values(before_response["raw"].decode()) if before_response["status"] == 200 else {}

    def one(example, *, chat, stream, name, barrier=None):
        body = request_payload(example, args.model, chat=chat, stream=stream, max_tokens=args.max_new_tokens)
        route = "/v1/chat/completions" if chat else "/v1/completions"
        if barrier is not None:
            barrier.wait(timeout=args.timeout)
        response = client.request(route, body)
        save_exchange(args.output / name, body, response)
        if response["status"] != 200:
            raise ValueError(f"{name}: HTTP {response['status']}")
        normalized = normalize_response(response["raw"], chat=chat, stream=stream)
        if normalized["model"] != args.model:
            raise ValueError("Response model differs from the requested model")
        record = validate_record(example, normalized, baseline.get(example["source_id"]),
                                 max_tokens=args.max_new_tokens, eos_token_ids=eos)
        record.update(case=name, chat=chat, stream=stream, client_start=response["start"],
                      client_end=response["end"], client_seconds=response["seconds"],
                      client_output_tps=len(record["output_ids"]) / response["seconds"])
        write_json(args.output / name / "validated.json", record)
        return record

    for index, example in enumerate(examples):
        for chat in (False, True):
            for stream in (False, True):
                name = f"{'chat' if chat else 'completion'}-{'sse' if stream else 'json'}-{index:03d}"
                records.append(one(example, chat=chat, stream=stream, name=name))
    # Each of four endpoint/transport combinations must independently match.
    for example in examples:
        group = [record for record in records if record["source_id"] == example["source_id"]]
        if any(row["output_ids"] != group[0]["output_ids"] or row["output_text"] != group[0]["output_text"] for row in group[1:]):
            raise ValueError("HTTP endpoint/streaming variants differ from one another")

    samples, monitoring_errors = [], []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                response = client.request("/metrics")
                if response["status"] == 200:
                    samples.append(dict(monotonic=time.perf_counter(), values=metric_values(response["raw"].decode())))
            except Exception as error:
                monitoring_errors.append(str(error))
            stop.wait(0.05)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    barrier = threading.Barrier(args.concurrency)
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(one, examples[index % len(examples)], chat=bool(index % 2),
                                   stream=bool((index // 2) % 2), name=f"concurrent-{index:03d}", barrier=barrier)
                       for index in range(args.concurrency)]
            concurrent = [future.result() for future in futures]
    finally:
        stop.set()
        monitor_thread.join(timeout=args.timeout + 1)
        write_json(args.output / "concurrency-metrics.json", dict(samples=samples, errors=monitoring_errors))
    overlap = maximum_overlap([(record["client_start"], record["client_end"]) for record in concurrent])
    if overlap < 2:
        raise ValueError("Concurrent HTTP requests did not actually overlap")
    sequential_by_id = {row["source_id"]: row for row in records}
    if any(row["output_ids"] != sequential_by_id[row["source_id"]]["output_ids"] for row in concurrent):
        raise ValueError("Concurrent output token IDs differ from the sequential result")
    records.extend(concurrent)

    if args.check_unsupported:
        for chat in (False, True):
            for field, overrides in unsupported_cases():
                body = request_payload(examples[0], args.model, chat=chat, stream=False, max_tokens=8)
                body.update(overrides)
                name = f"reject-{'chat' if chat else 'completion'}-{field}"
                response = client.request("/v1/chat/completions" if chat else "/v1/completions", body)
                save_exchange(args.output / name, body, response)
                if not 400 <= response["status"] < 500:
                    raise ValueError(f"{name}: expected HTTP 4xx, got {response['status']}")
                error = json.loads(response["raw"])
                if not isinstance(error, dict) or "error" not in error:
                    raise ValueError(f"{name}: missing OpenAI-compatible error body")
                negative.append(dict(case=name, status=response["status"], body=error))
        get("/health", "health-after-rejections")
        records.append(one(examples[0], chat=False, stream=True, name="recovery-after-rejections"))
    get("/health", "health-after")
    time.sleep(args.metrics_settle_seconds)
    after_response = client.request("/metrics")
    save_exchange(args.output / "metrics-after", None, after_response)
    after = metric_values(after_response["raw"].decode()) if after_response["status"] == 200 else {}
    summary = dict(passed=True, input_sha256=digest, unique_examples=len(examples),
                   successful_generations=len(records), correct_generations=sum(bool(row["correct"]) for row in records),
                   baseline_exact_token_parity=all(row["baseline_exact_token_parity"] is True for row in records) if baseline else None,
                   json_sse_chat_completion_exact_parity=True, concurrent_requests=args.concurrency,
                   max_client_overlap=overlap,
                   max_observed_server_running=max((sample["values"].get(METRICS["running"], 0) for sample in samples), default=0),
                   concurrency_note="Client overlap is proven; short-lived server metrics may lag. Use native worker max_decode_batch_size audit to prove GPU batching.",
                   rejection_cases=negative, acceptance=acceptance_delta(before, after),
                   records=[{key: row[key] for key in ("case", "source_id", "correct", "client_seconds", "client_output_tps")} for row in records])
    write_json(args.output / "summary.json", summary)
    return summary


def main():
    args = parser().parse_args()
    output_existed = args.output.exists()
    try:
        summary = run(args)
    except Exception as error:
        # Never append to an old run when the output-directory precondition fails.
        if not output_existed and args.output.is_dir() and (args.output / "plan.json").exists() and not (args.output / "failure.json").exists() and not (args.output / "summary.json").exists():
            write_json(args.output / "failure.json", dict(passed=False, error=repr(error)))
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
