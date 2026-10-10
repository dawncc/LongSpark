#!/usr/bin/env python3
"""Evaluate GSM8K through native vLLM V2 LLM sessions and worker graph audits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "vendor/vllm"), str(ROOT / "vendor")]
from longspark_vllm.gsm8k import (
    METHODS, compare_outputs, load_inputs, resolve_eos, score_record, summarize_records,
)
SPECULATOR = "longspark_vllm.native_v2.speculator.LongSparkSpeculator"
EXTENSION = "longspark_vllm.native_v2.diagnostics.NativeV2Diagnostics"


def integration_source_hashes(*, root=ROOT):
    paths = set((root / "vendor/vllm/longspark_vllm/native_v2").rglob("*.py"))
    draft_dir = root / "vendor/specforge/modeling/draft"
    paths.update(draft_dir.glob("draftfreekv*.py"))
    paths.update(draft_dir.glob("global16_raw256*.py"))
    paths.update(root / relative for relative in (
        "vendor/vllm/longspark_vllm/gsm8k.py",
        "vendor/vllm/scripts/evaluate_native_v2_gsm8k.py",
        "vendor/vllm/patches/native-v2-custom-class.manifest.json",
    ))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def verify_integration_sources(expected, *, root=ROOT):
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Plan has no integration source hashes")
    actual = integration_source_hashes(root=root)
    changed = sorted(path for path in set(expected) | set(actual)
                     if expected.get(path) != actual.get(path))
    if changed:
        raise ValueError("Integration sources changed since plan creation: " + ", ".join(changed))
    return actual


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--warmup-count", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--kv-cache-gib", type=float, default=2.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kernel-policy", choices=("invariant", "default"), default="invariant")
    p.add_argument("--native-optimizations", choices=("reference", "fused"), default="fused",
                   help="Select reference or fused state stores/direct gather for controlled comparisons.")
    p.add_argument("--execute", action="store_true")
    return p


def llm_options(plan, method):
    options = dict(
        model=plan["target"], dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=plan["max_model_len"], max_num_seqs=plan["max_num_seqs"],
        max_num_batched_tokens=plan["max_num_batched_tokens"],
        kv_cache_memory_bytes=int(plan["kv_cache_gib"] * 1024 ** 3),
        enforce_eager=False, enable_prefix_caching=False, enable_chunked_prefill=True,
        async_scheduling=True, disable_log_stats=False, seed=plan["seed"],
        generation_config="auto", trust_remote_code=False,
        attention_config=dict(backend="FLASH_ATTN"),
        cudagraph_metrics=True,
        worker_extension_cls=EXTENSION,
        compilation_config=dict(cudagraph_mode="FULL_AND_PIECEWISE"),
    )
    if method == "longspark":
        options["per_request_spec_decode_metrics"] = "detailed"
        options["speculative_config"] = dict(
            method="custom_class", model=SPECULATOR, num_speculative_tokens=7,
            draft_sample_method="greedy",
        )
        options["additional_config"] = dict(
            native_v2_custom_class=True, longspark=dict(
                draft_model=plan["draft"],
                use_fused_state_commit=plan.get("native_optimizations", "fused") == "fused",
                use_direct_gather=plan.get("native_optimizations", "fused") == "fused",
            ),
        )
    return options


def make_plan(args):
    if args.limit < 1 or not 0 <= args.warmup_count <= args.limit:
        raise ValueError("Require limit > 0 and 0 <= warmup-count <= limit")
    if min(args.max_num_seqs, args.max_model_len, args.max_num_batched_tokens) < 1:
        raise ValueError("Batch and context limits must be positive")
    if args.max_num_batched_tokens < args.max_num_seqs * 8:
        raise ValueError("max-num-batched-tokens must cover max-num-seqs * 8 verify tokens")
    if args.max_new_tokens < 2 or args.gpu < 0:
        raise ValueError("Require max-new-tokens >= 2 and a nonnegative GPU index")
    if not math.isfinite(args.kv_cache_gib) or args.kv_cache_gib <= 0:
        raise ValueError("KV cache size must be finite and positive")
    examples, digest = load_inputs(args.input, limit=args.limit,
                                  max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len)
    candidates = [args.input.with_suffix(args.input.suffix + ".metadata.json"),
                  args.input.with_suffix(".metadata.json")]
    provenance_path = args.provenance or next((path for path in candidates if path.is_file()), None)
    provenance = json.loads(provenance_path.read_text()) if provenance_path else None
    if provenance is not None and (provenance.get("split") != "test" or
                                   provenance.get("input_sha256", digest) != digest):
        raise ValueError("Provenance must describe the same test fixture SHA256")
    plan = dict(
        schema_version=1, backend="vllm_native_v2", methods=list(METHODS),
        input=str(args.input.resolve()), input_sha256=digest, provenance=provenance,
        target=str(args.target.resolve()), draft=str(args.draft.resolve()), output=str(args.output.resolve()),
        limit=args.limit, source_ids=[row["source_id"] for row in examples],
        prompt_tokens=[len(row["input_ids"]) for row in examples],
        warmup_count=args.warmup_count, debug_without_warmup=args.warmup_count == 0,
        max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_gib=args.kv_cache_gib, gpu=args.gpu, seed=args.seed,
        kernel_policy=args.kernel_policy, native_optimizations=args.native_optimizations,
        temperature=0.0, ignore_eos=False,
        async_scheduling=True, target_tp=1, dtype="bfloat16", retokenize=False,
        flashinfer_sampler=False,
        sampler_policy="VLLM_USE_FLASHINFER_SAMPLER=0 is fixed for this greedy evaluation, including startup warmup; FLASH_ATTN attention is unchanged",
        method_isolation="Each method has a fresh subprocess and native LLM engine",
        timing_definition="service=last_token_ts-scheduled_ts; decode=last_token_ts-first_token_ts; engine monotonic timestamps",
        wall_timing_definition="One generate call per batch, including frontend work; never copied to each request as service time",
        acceptance_definition="Native raw detailed metrics retained; emitted acceptance clips the final verify step to actual output IDs",
        integration_source_hashes=integration_source_hashes(),
        prefill_shape_warmup=dict(
            max_width=min(args.max_num_batched_tokens, args.max_model_len,
                          max(len(row["input_ids"]) for row in examples)
                          if args.max_num_seqs == 1 else args.max_num_batched_tokens),
            phase="After engine load, before request warmup and measurement",
            operation="Zero-count state.update only; all widths 9..max_width and all supported batch sizes",
        ),
    )
    plan["llm_options"] = {method: llm_options(plan, method) for method in METHODS}
    return plan


def acceptance_metrics(raw, output_tokens, *, method):
    if method == "vanilla":
        if raw is not None:
            raise ValueError("AR unexpectedly returned speculative metrics")
        return dict(verification_rounds=0, accepted_draft_tokens_raw=0,
                    accepted_draft_tokens_emitted=0, proposed_tokens=0, raw_spec_decode_metrics=None)
    if raw is None:
        if output_tokens == 1:
            return dict(verification_rounds=0, accepted_draft_tokens_raw=0,
                        accepted_draft_tokens_emitted=0, proposed_tokens=0, raw_spec_decode_metrics=None)
        raise ValueError("LongSpark returned no native detailed speculative metrics")
    accepted, drafted = raw.get("per_step_accepted"), raw.get("per_step_drafted")
    if not isinstance(accepted, list) or not isinstance(drafted, list) or len(accepted) != len(drafted):
        raise ValueError("Native per-step accepted/drafted metrics are required")
    if any(type(a) is not int or type(d) is not int or not 0 <= a <= d <= 7
           for a, d in zip(accepted, drafted)):
        raise ValueError("Invalid native seven-token acceptance counts")
    rounds = len(accepted)
    if (raw.get("num_spec_steps") != rounds or raw.get("num_accepted_draft_tokens") != sum(accepted)
            or raw.get("num_draft_tokens") != sum(drafted)):
        raise ValueError("Native detailed counts disagree with their summary")
    # The first output comes from prefill. Verify emits accepted draft tokens
    # first, followed by a correction/bonus. Native stats precede stop handling.
    remaining = max(output_tokens - 1, 0)
    emitted, emitted_steps = 0, []
    for index, count in enumerate(accepted):
        if remaining == 0:
            raise ValueError("Native verify metrics continue after the emitted sequence ended")
        used = min(count, remaining)
        emitted += used
        emitted_steps.append(used)
        remaining -= min(count + 1, remaining)
        if remaining == 0 and index != rounds - 1:
            raise ValueError("Only the final native verify step may be truncated by stopping")
    if remaining:
        raise ValueError("Native verify metrics do not account for all emitted tokens")
    return dict(verification_rounds=rounds, accepted_draft_tokens_raw=sum(accepted),
                accepted_draft_tokens_emitted=emitted, proposed_tokens=sum(drafted),
                per_step_accepted_emitted=emitted_steps, raw_spec_decode_metrics=raw)


def record_output(example, output, *, method, eos, max_new_tokens, batch_index):
    if not output.finished or len(output.outputs) != 1:
        raise ValueError("Expected one finished native completion per request")
    if list(output.prompt_token_ids) != example["input_ids"]:
        raise ValueError("Native engine input IDs differ from the shared fixture")
    completion = output.outputs[0]
    ids = list(completion.token_ids)
    if not ids:
        raise ValueError("Native completion has no output tokens")
    metrics = output.metrics
    if metrics is None:
        raise ValueError("Native request timing metrics are missing")
    times = {name: float(getattr(metrics, name))
             for name in ("scheduled_ts", "first_token_ts", "last_token_ts")}
    if not all(math.isfinite(value) and value > 0 for value in times.values()):
        raise ValueError("Native engine timestamps must be finite and positive")
    service = times["last_token_ts"] - times["scheduled_ts"]
    decode = times["last_token_ts"] - times["first_token_ts"]
    if service <= 0 or decode < 0 or service < decode:
        raise ValueError("Native request timing intervals are inconsistent")
    if len(ids) > 1 and decode <= 0:
        raise ValueError("Multi-token native completion has no decode interval")
    raw = completion.spec_decode_metrics
    raw = raw.to_dict() if raw is not None else None
    native_metrics = asdict(metrics) if is_dataclass(metrics) else dict(vars(metrics))
    result = dict(
        method=method, source_id=example["source_id"], batch_index=batch_index,
        request_id=output.request_id, prompt_tokens=len(example["input_ids"]),
        output_ids=ids, output_text=completion.text, native_finish_reason=completion.finish_reason,
        stop_reason=completion.stop_reason,
        finish_reason="eos" if ids[-1] in eos else completion.finish_reason,
        service_seconds=service, total_seconds=service, decode_seconds=decode,
        prefill_seconds=times["first_token_ts"] - times["scheduled_ts"],
        native_request_metrics=native_metrics,
        **acceptance_metrics(raw, len(ids), method=method),
    )
    scored = score_record(example, result, eos_token_ids=eos, max_new_tokens=max_new_tokens)
    if scored["tokens_after_eos"]:
        raise ValueError("Native natural-EOS generation emitted tokens after EOS")
    if not scored["eos_at_end"] and not scored["truncated"]:
        raise ValueError("Native generation stopped without EOS or token cap")
    return scored


def performance_summary(records):
    def ratio(n, d):
        return n / d if d else None
    tokens = sum(row["output_tokens"] for row in records)
    decode_tokens = sum(max(0, row["output_tokens"] - 1) for row in records)
    service = sum(row["service_seconds"] for row in records)
    decode = sum(row["decode_seconds"] for row in records)
    rounds = sum(row["verification_rounds"] for row in records)
    raw = sum(row["accepted_draft_tokens_raw"] for row in records)
    emitted = sum(row["accepted_draft_tokens_emitted"] for row in records)
    proposed = sum(row["proposed_tokens"] for row in records)
    return dict(requests=len(records), output_tokens=tokens, decode_tokens=decode_tokens,
                service_seconds=service, decode_seconds=decode,
                aggregate_service_tps=ratio(tokens, service), aggregate_decode_tps=ratio(decode_tokens, decode),
                verification_rounds=rounds, proposed_tokens=proposed,
                accepted_draft_tokens_raw=raw, accepted_draft_tokens_emitted=emitted,
                mean_accepted_draft_tokens=ratio(emitted, rounds),
                tau=1 + emitted / rounds if rounds else None,
                tau_raw=1 + raw / rounds if rounds else None,
                acceptance_rate=ratio(emitted, proposed), acceptance_rate_raw=ratio(raw, proposed),
                effective_advance=ratio(decode_tokens, rounds))


def require_audit(audits, *, method, require_replays):
    if len(audits) != 1:
        raise RuntimeError("TP1 evaluation expects one worker audit")
    for audit in audits:
        before, after = audit["before"], audit["after"]
        if (after["runner_class"] != "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
                or not after["use_v2_model_runner"] or not after["async_scheduling"]):
            raise RuntimeError("Worker did not execute the required async native V2 runner")
        target = audit["target"]
        if after["target"]["captures"] < 1 or target["captures_delta"] or target["manager_replaced"]:
            raise RuntimeError("Target graphs missing, captured during measurement, or manager replaced")
        if require_replays and target["replays"] < 1:
            raise RuntimeError("Target has no measured actual CUDA graph replay")
        if method == "longspark":
            if after["speculator_class"] != SPECULATOR:
                raise RuntimeError("Worker did not load the native LongSpark speculator")
            spec_before, spec_after = before["speculator"], after["speculator"]
            if any(stats.get("startup_complete") is not True for stats in (spec_before, spec_after)):
                raise RuntimeError("LongSpark startup was incomplete during the audit")
            startup_counts = [stats.get("startup_eager_calls") for stats in (spec_before, spec_after)]
            if (any(type(count) is not int or count < 0 for count in startup_counts)
                    or startup_counts[0] != startup_counts[1]):
                raise RuntimeError("LongSpark startup eager calls changed during the audit")
            delta = audit["speculator_delta"]
            if delta is None or any(key not in delta for key in ("captures", "replays", "fallbacks")):
                raise RuntimeError("LongSpark runtime graph counters are missing")
            if after["speculator"]["captures"] < 1 or delta["captures"] or delta["fallbacks"]:
                raise RuntimeError("LongSpark graph capture/fallback audit failed")
            if require_replays and delta["replays"] < 1:
                raise RuntimeError("LongSpark has no measured actual graph replay")
        elif after["speculator_class"] is not None or before["speculator"] is not None:
            raise RuntimeError("Native AR baseline unexpectedly has a speculator")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def append_jsonl(stream, payload):
    stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def run_worker(plan, method):
    verified_sources = verify_integration_sources(plan.get("integration_source_hashes"))
    # The parent also supplies these to spawned engine processes before imports.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if plan["kernel_policy"] == "invariant" else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan["gpu"])
    from vllm import LLM, SamplingParams

    examples, digest = load_inputs(plan["input"], limit=plan["limit"],
                                  max_new_tokens=plan["max_new_tokens"], max_model_len=plan["max_model_len"])
    if digest != plan["input_sha256"]:
        raise ValueError("Input fixture changed after plan creation")
    folder = Path(plan["output"]) / method
    folder.mkdir(exist_ok=False)
    state = dict(status="loading", method=method, llm_options=plan["llm_options"][method],
                 integration_source_hashes=verified_sources, integration_sources_verified_before_load=True)
    write_json(folder / "manifest.json", state)
    llm = None
    audit_active = False
    records, batches = [], []
    try:
        llm = LLM(**plan["llm_options"][method])
        config = json.loads((Path(plan["target"]) / "config.json").read_text())
        eos = resolve_eos(plan["target"], llm.get_tokenizer(), config)
        state.update(eos=eos, environment=dict(
            python=sys.version, executable=sys.executable, vllm=importlib.metadata.version("vllm"),
            torch=importlib.metadata.version("torch"), pid=os.getpid(),
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            vllm_use_v2_model_runner=os.environ["VLLM_USE_V2_MODEL_RUNNER"],
            vllm_use_flashinfer_sampler=os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
            vllm_batch_invariant=os.environ["VLLM_BATCH_INVARIANT"],
        ))
        state["after_load"] = llm.collective_rpc("native_v2_audit_snapshot")
        write_json(folder / "manifest.json", state)
        state["status"] = "prefill_shape_warmup"
        write_json(folder / "manifest.json", state)
        print(f"{method}: prefill state shape warmup through width "
              f"{plan['prefill_shape_warmup']['max_width']}", flush=True)
        prefill_warmup = llm.collective_rpc(
            "native_v2_warm_prefill_shapes", kwargs=dict(max_width=plan["prefill_shape_warmup"]["max_width"]))
        write_json(folder / "prefill-warmup.json", prefill_warmup)
        if len(prefill_warmup) != 1 or prefill_warmup[0].get("state_unchanged") is not True:
            raise RuntimeError("Prefill shape warmup did not prove unchanged worker state")
        for phase, selected in (("warmup", examples[:plan["warmup_count"]]), ("measurement", examples)):
            state["status"] = phase
            write_json(folder / "manifest.json", state)
            before_metrics = [asdict(item) for item in llm.get_metrics()]
            llm.collective_rpc("native_v2_audit_start", kwargs=dict(label=phase))
            audit_active = True
            phase_records = []
            filename = "warmup.jsonl" if phase == "warmup" else "records.jsonl"
            with (folder / filename).open("x") as stream:
                for start in range(0, len(selected), plan["max_num_seqs"]):
                    chunk = selected[start:start + plan["max_num_seqs"]]
                    params = [SamplingParams(temperature=0.0, max_tokens=plan["max_new_tokens"],
                                             seed=plan["seed"] + row["dataset_index"], ignore_eos=False,
                                             stop_token_ids=eos["effective_token_ids"])
                              for row in chunk]
                    started = time.perf_counter()
                    outputs = llm.generate([dict(prompt_token_ids=row["input_ids"]) for row in chunk],
                                           params, use_tqdm=False)
                    wall = time.perf_counter() - started
                    if len(outputs) != len(chunk):
                        raise ValueError("Native generate returned the wrong number of requests")
                    batch_index = start // plan["max_num_seqs"]
                    batch = dict(phase=phase, batch_index=batch_index, wall_seconds=wall,
                                 source_ids=[row["source_id"] for row in chunk], submitted_concurrently=len(chunk),
                                 output_tokens=sum(len(out.outputs[0].token_ids) for out in outputs))
                    batches.append(batch)
                    for example, output in zip(chunk, outputs):
                        record = record_output(example, output, method=method,
                                               eos=eos["effective_token_ids"], max_new_tokens=plan["max_new_tokens"],
                                               batch_index=batch_index)
                        record["excluded_from_metrics"] = phase == "warmup"
                        append_jsonl(stream, record)
                        phase_records.append(record)
                        print(f"{method}/{phase}: {len(phase_records)}/{len(selected)} "
                              f"{example['source_id']} tokens={record['output_tokens']} "
                              f"correct={record['correct']} service={record['service_seconds']:.3f}s", flush=True)
            audits = llm.collective_rpc("native_v2_audit_end")
            audit_active = False
            write_json(folder / f"{phase}-audit.json", audits)
            write_json(folder / f"{phase}-metrics.json", dict(
                before=before_metrics, after=[asdict(item) for item in llm.get_metrics()]))
            require_audit(audits, method=method, require_replays=any(row["output_tokens"] > 1 for row in phase_records))
            if phase == "measurement":
                records = phase_records
        write_json(folder / "batches.json", batches)
        measured_batches = [batch for batch in batches if batch["phase"] == "measurement"]
        performance = performance_summary(records)
        wall = sum(batch["wall_seconds"] for batch in measured_batches)
        performance.update(generate_wall_seconds=wall, generate_wall_tps=performance["output_tokens"] / wall,
                           max_submitted_concurrency=max(batch["submitted_concurrently"] for batch in measured_batches))
        write_json(folder / "summary.json", dict(scoring=summarize_records(records), performance=performance))
        state.update(status="completed", measured_requests=len(records))
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if llm is not None:
            if audit_active:
                try:
                    write_json(folder / "interrupted-audit.json", llm.collective_rpc("native_v2_audit_end"))
                except Exception as error:
                    state["audit_cleanup_error"] = repr(error)
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as error:
                state["shutdown_error"] = repr(error)
                state["status"] = "failed"
                write_json(folder / "manifest.json", state)
                raise
        write_json(folder / "manifest.json", state)


def run_method_process(plan, method, plan_path):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(plan["gpu"]), VLLM_USE_V2_MODEL_RUNNER="1",
               VLLM_USE_FLASHINFER_SAMPLER="0",
               VLLM_BATCH_INVARIANT="1" if plan["kernel_policy"] == "invariant" else "0",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "vendor/vllm"), str(ROOT / "vendor"), env.get("PYTHONPATH"))))
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(plan_path),
               "--worker-method", method]
    with (Path(plan["output"]) / f"{method}.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise RuntimeError(f"{method} native LLM subprocess exited {code}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def execute(plan):
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    state = dict(status="running", schema_version=1, methods={},
                 integration_source_hashes=plan["integration_source_hashes"])
    write_json(output / "manifest.json", state)
    try:
        records, summaries = [], {}
        for method in METHODS:
            state["methods"][method] = "running"
            write_json(output / "manifest.json", state)
            run_method_process(plan, method, output / "plan.json")
            records.extend(json.loads(line) for line in (output / method / "records.jsonl").read_text().splitlines())
            summaries[method] = json.loads((output / method / "summary.json").read_text())
            state["methods"][method] = "completed"
        parity = compare_outputs(records)
        performance = {method: summary["performance"] for method, summary in summaries.items()}
        completed = dict(schema_version=1, status="completed", backend="vllm_native_v2",
                         input_sha256=plan["input_sha256"], requests=len(records),
                         integration_source_hashes=plan["integration_source_hashes"],
                         greedy_parity_passed=parity["passed"], parity=parity,
                         correctness_status="passed" if parity["passed"] else "failed",
                         scoring=summarize_records(records), performance=performance,
                         aggregate_service_speedup=performance["longspark"]["aggregate_service_tps"] / performance["vanilla"]["aggregate_service_tps"],
                         definitions=dict(tau="1 + emitted accepted draft tokens / verified steps",
                                          tau_raw="1 + native pre-stop accepted draft tokens / verified steps",
                                          service_tps="sum(output_tokens) / sum(last_token_ts - scheduled_ts)",
                                          decode_tps="sum(output_tokens - 1) / sum(last_token_ts - first_token_ts)",
                                          wall_tps="sum(output_tokens) / sum(generate batch wall time)"))
        write_json(output / "completed.json", completed)
        state.update(status="completed", greedy_parity_passed=parity["passed"])
        print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "manifest.json", state)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker-plan" in argv:
        p = argparse.ArgumentParser()
        p.add_argument("--worker-plan", type=Path, required=True)
        p.add_argument("--worker-method", choices=METHODS, required=True)
        args = p.parse_args(argv)
        run_worker(json.loads(args.worker_plan.read_text()), args.worker_method)
        return
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = make_plan(args)
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    if args.execute:
        execute(plan)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
