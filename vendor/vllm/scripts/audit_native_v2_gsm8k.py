#!/usr/bin/env python3
"""Independently audit saved native V2 GSM8K evidence using only the stdlib.

No runner, scoring, model, GPU, or network code is imported. Inputs are never
modified. JSON is written to stdout; --output can save a new report exclusively.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sys


METHODS = ("vanilla", "longspark")
RUNNER = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
TARGET_MANAGER = "vllm.v1.worker.gpu.cudagraph_utils.ModelCudaGraphManager"
DRAFT_MANAGER = "vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager"
SPECULATOR = "longspark_vllm.native_v2.speculator.LongSparkSpeculator"
NUMBER = re.compile(r"(?P<sign>[+\-]?\s*\$?|\$\s*[+\-]?)\s*"
                    r"(?P<value>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name}: expected integer >= {minimum}")
    return value


def real(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), f"{name}: invalid finite time/count")
    return float(value)


def compare(actual, expected, name):
    """Compare every independently derived field; extra descriptive fields are OK."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{name}: expected object")
        for key, value in expected.items():
            require(key in actual, f"{name}.{key}: missing evidence")
            compare(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{name}: list length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{name}[{index}]")
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
                f"{name}: stored {actual!r}, recomputed {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"{name}: stored {actual!r}, recomputed {expected!r}")


def read_json(path):
    def reject_constant(value):
        raise AuditError(f"{path}: non-finite JSON value {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def read_jsonl(path):
    data = Path(path).read_bytes()
    rows = []
    for line_no, line in enumerate(data.decode().splitlines(), 1):
        require(bool(line.strip()), f"{path}:{line_no}: blank record")
        row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            AuditError(f"{path}:{line_no}: non-finite JSON value {value}")))
        require(isinstance(row, dict), f"{path}:{line_no}: record must be an object")
        rows.append(row)
    return rows, hashlib.sha256(data).hexdigest()


def parse_answer(text):
    require(isinstance(text, str), "Answer text must be a string")
    if "####" not in text:
        return None, None, "missing_marker"
    tail = text.rsplit("####", 1)[1].lstrip()
    line = tail.splitlines()[0].strip() if tail else ""
    match = NUMBER.fullmatch(line.replace("\N{MINUS SIGN}", "-"))
    if match is None:
        return None, line, "invalid_final_answer"
    value = Decimal(("-" if "-" in match["sign"] else "") + match["value"].replace(",", ""))
    result = "0" if value == 0 else format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result, line, "parsed"


def token_ids(values, name, *, nonempty=True):
    require(isinstance(values, list) and (bool(values) or not nonempty), f"{name}: missing token IDs")
    for value in values:
        integer(value, name)
    return values


def recompute_acceptance(raw, output_tokens, method):
    zero = dict(verification_rounds=0, accepted_draft_tokens_raw=0,
                accepted_draft_tokens_emitted=0, proposed_tokens=0)
    if method == "vanilla":
        require(raw is None, "AR must not contain speculative metrics")
        return zero
    if raw is None:
        require(output_tokens == 1, "LongSpark decode is missing native acceptance metrics")
        return zero
    require(isinstance(raw, dict), "Native acceptance metrics must be an object")
    accepted, drafted = raw["per_step_accepted"], raw["per_step_drafted"]
    require(isinstance(accepted, list) and isinstance(drafted, list) and len(accepted) == len(drafted),
            "Accepted/drafted step arrays must have equal lengths")
    for accepted_count, proposed_count in zip(accepted, drafted):
        integer(accepted_count, "accepted count")
        integer(proposed_count, "proposed count")
        require(accepted_count <= proposed_count <= 7, "Invalid seven-token verification counts")
    compare(raw, dict(num_spec_steps=len(accepted), num_accepted_draft_tokens=sum(accepted),
                      num_draft_tokens=sum(drafted)), "raw acceptance")
    # Verification emits accepted draft IDs followed by a correction/bonus.
    # Only the final step may be cut short, including EOS inside accepted IDs.
    emitted_steps = []
    remaining = output_tokens - 1
    for index, count in enumerate(accepted):
        require(remaining > 0, "Verification metrics continue after emitted output ended")
        emitted_steps.append(min(count, remaining))
        produced = min(count + 1, remaining)
        remaining -= produced
        require(remaining > 0 or index == len(accepted) - 1,
                "Only the final verification step may be clipped")
    require(remaining == 0, "Verification metrics do not account for the emitted output")
    return dict(verification_rounds=len(accepted), accepted_draft_tokens_raw=sum(accepted),
                accepted_draft_tokens_emitted=sum(emitted_steps), proposed_tokens=sum(drafted),
                per_step_accepted_emitted=emitted_steps)


def audit_record(row, example, method, eos, cap, batch_index):
    name = f"{method}/{example['source_id']}"
    ids = token_ids(row["output_ids"], name)
    require(len(ids) <= cap, f"{name}: output exceeds token cap")
    reference_answer, _, _ = parse_answer(example["reference_answer"])
    require(reference_answer is not None, f"{name}: fixture reference answer is unparseable")
    compare(example["gold_answer"], reference_answer, f"{name}.fixture gold")
    answer, final_line, parse_status = parse_answer(row["output_text"])
    eos_positions = [i for i, value in enumerate(ids) if value in eos]
    first_eos = eos_positions[0] if eos_positions else None
    seen = first_eos is not None
    require(not seen or first_eos == len(ids) - 1, f"{name}: tokens emitted after EOS")
    at_cap = len(ids) == cap
    require(seen or at_cap, f"{name}: stopped before EOS or token cap")
    native_finish = row["native_finish_reason"]
    require(native_finish in ({"stop", "length"} if seen and at_cap else {"stop"} if seen else {"length"}),
            f"{name}: native finish reason contradicts actual IDs")
    stop_reason = row["stop_reason"]
    require(stop_reason is None or (type(stop_reason) is int and stop_reason == ids[-1] and seen),
            f"{name}: unexpected native stop reason")
    computed = dict(method=method, source_id=example["source_id"], dataset_index=example["dataset_index"],
                    question=example["question"], reference_answer=example["reference_answer"],
                    gold_answer=reference_answer, prompt_tokens=len(example["input_ids"]), batch_index=batch_index,
                    parsed_answer=answer, final_answer_line=final_line, parse_status=parse_status,
                    correct=answer == reference_answer, output_tokens=len(ids), eos_token_ids=sorted(eos),
                    eos_seen=seen, eos_at_end=seen, first_eos_index=first_eos, tokens_after_eos=0,
                    max_new_tokens=cap, token_cap_reached=at_cap, truncated=not seen and at_cap,
                    finish_reason="eos" if seen else native_finish, excluded_from_metrics=False)
    metrics = row["native_request_metrics"]
    scheduled = real(metrics["scheduled_ts"], name + ".scheduled_ts", positive=True)
    first = real(metrics["first_token_ts"], name + ".first_token_ts", positive=True)
    last = real(metrics["last_token_ts"], name + ".last_token_ts", positive=True)
    require(scheduled <= first <= last and last > scheduled and (len(ids) == 1 or last > first),
            f"{name}: inconsistent native timing intervals")
    computed.update(service_seconds=last - scheduled, total_seconds=last - scheduled,
                    decode_seconds=last - first, prefill_seconds=first - scheduled)
    computed.update(recompute_acceptance(row["raw_spec_decode_metrics"], len(ids), method))
    compare(row, computed, name)
    return dict(row, **computed)


def scoring(rows):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    unparsed = sum(row["parsed_answer"] is None for row in rows)
    tokens = sum(row["output_tokens"] for row in rows)
    return dict(total=total, correct=correct, incorrect=total - correct, unparsed=unparsed,
                accuracy=correct / total if total else None,
                parsed_accuracy=correct / (total - unparsed) if total > unparsed else None,
                parse_status_counts=dict(sorted(Counter(row["parse_status"] for row in rows).items())),
                eos_count=sum(row["eos_seen"] for row in rows), truncated_count=sum(row["truncated"] for row in rows),
                token_cap_count=sum(row["token_cap_reached"] for row in rows), outputs_with_tokens_after_eos=0,
                output_tokens=tokens, mean_output_tokens=tokens / total if total else None)


def performance(rows, batches):
    tokens = sum(row["output_tokens"] for row in rows)
    decode_tokens = tokens - len(rows)
    service = sum(row["service_seconds"] for row in rows)
    decode = sum(row["decode_seconds"] for row in rows)
    rounds = sum(row["verification_rounds"] for row in rows)
    raw = sum(row["accepted_draft_tokens_raw"] for row in rows)
    emitted = sum(row["accepted_draft_tokens_emitted"] for row in rows)
    proposed = sum(row["proposed_tokens"] for row in rows)
    wall = sum(real(batch["wall_seconds"], "batch wall time", positive=True) for batch in batches)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return dict(requests=len(rows), output_tokens=tokens, decode_tokens=decode_tokens,
                service_seconds=service, decode_seconds=decode,
                aggregate_service_tps=ratio(tokens, service), aggregate_decode_tps=ratio(decode_tokens, decode),
                verification_rounds=rounds, proposed_tokens=proposed,
                accepted_draft_tokens_raw=raw, accepted_draft_tokens_emitted=emitted,
                mean_accepted_draft_tokens=ratio(emitted, rounds),
                tau=1 + emitted / rounds if rounds else None, tau_raw=1 + raw / rounds if rounds else None,
                acceptance_rate=ratio(emitted, proposed), acceptance_rate_raw=ratio(raw, proposed),
                effective_advance=ratio(decode_tokens, rounds), generate_wall_seconds=wall,
                generate_wall_tps=ratio(tokens, wall),
                max_submitted_concurrency=max(batch["submitted_concurrently"] for batch in batches))


def audit_snapshot(snapshot, method):
    compare(snapshot, dict(runner_class=RUNNER, use_v2_model_runner=True, async_scheduling=True), method + ".worker")
    target = snapshot["target"]
    compare(target["manager_class"], TARGET_MANAGER, method + ".target manager")
    captures = integer(target["captures"], "target captures", 1)
    require(isinstance(target["captured_shapes"], list) and len(target["captured_shapes"]) == captures,
            f"{method}: target captured-shape evidence differs from count")
    if method == "longspark":
        compare(snapshot["speculator_class"], SPECULATOR, "native speculator class")
        speculator = snapshot["speculator"]
        compare(speculator, dict(implementation="longspark_native_model_runner_v2", cuda_graph_manager=DRAFT_MANAGER,
                                 target_forward_owned_by=RUNNER, draft_tokens=7), "native speculator")
        integer(speculator["captures"], "draft captures", 1)
        integer(speculator["replays"], "draft replays")
        compare(speculator["fallbacks"], 0, "draft fallbacks")
    else:
        compare(snapshot["speculator_class"], None, "AR speculator class")
        compare(snapshot["speculator"], None, "AR speculator")


def audit_graphs(items, method, *, need_replays, max_num_seqs):
    require(isinstance(items, list) and len(items) == 1, "Expected exactly one TP1 worker audit")
    audit = items[0]
    compare(audit["label"], "measurement", "graph phase")
    before, after, target = audit["before"], audit["after"], audit["target"]
    audit_snapshot(before, method)
    audit_snapshot(after, method)
    compare(after["target"], before["target"], method + ".unchanged target graphs")
    compare(target, dict(captures_delta=0, manager_replaced=False), method + ".target graph changes")
    replays = integer(target["replays"], "target replays", 1 if need_replays else 0)
    shapes = target["replay_shapes"]
    require(isinstance(shapes, dict), "Missing target replay shapes")
    require(sum(integer(count, "replay shape count", 1) for count in shapes.values()) == replays,
            "Target replay-shape sum disagrees with total")
    require(set(shapes).issubset(before["target"]["captured_shapes"]), "Target replay shape was not captured")
    dispatches = target["dispatches"]
    require(isinstance(dispatches, dict), "Missing target dispatch evidence")
    full_dispatches = 0
    for key, count in dispatches.items():
        integer(count, "dispatch count")
        require(key.startswith(("uniform_decode:", "other:")), "Unknown target dispatch kind")
        if key.endswith(":FULL"):
            full_dispatches += count
        require(not key.startswith("uniform_decode:") or key.endswith(":FULL") or count == 0,
                "Target uniform decode used an eager/piecewise fallback")
    require(full_dispatches >= replays, "Recorded target replays exceed full-graph dispatches")
    for key in ("max_dispatched_requests", "max_decode_requests"):
        require(integer(target[key], key) <= max_num_seqs, f"{key}: exceeds configured concurrency")
    draft_replays = 0
    if method == "longspark":
        left, right = before["speculator"], after["speculator"]
        delta = {key: right[key] - left[key] for key in ("captures", "replays", "fallbacks")}
        compare(audit["speculator_delta"], delta, "native draft graph counter differences")
        compare(delta["captures"], 0, "measurement-time draft captures")
        compare(delta["fallbacks"], 0, "measurement-time draft fallbacks")
        draft_replays = integer(delta["replays"], "measured draft replays", 1 if need_replays else 0)
    else:
        compare(audit["speculator_delta"], None, "AR speculator delta")
    return dict(target_replays=replays, draft_replays=draft_replays,
                captures_during_measurement=0, decode_fallbacks=0,
                max_dispatched_requests=target["max_dispatched_requests"],
                max_decode_requests=target["max_decode_requests"])


def audit_directory(directory, *, input_path=None, expect_limit=None):
    directory = Path(directory)
    plan = read_json(directory / "plan.json")
    compare(plan, dict(schema_version=1, backend="vllm_native_v2", methods=list(METHODS),
                       temperature=0.0, ignore_eos=False, async_scheduling=True,
                       target_tp=1, dtype="bfloat16", retokenize=False), "plan")
    limit = integer(plan["limit"], "plan limit", 1)
    if expect_limit is not None:
        compare(limit, expect_limit, "requested evaluation limit")
    batch_size = integer(plan["max_num_seqs"], "max_num_seqs", 1)
    cap = integer(plan["max_new_tokens"], "max_new_tokens", 2)
    require(plan["kernel_policy"] in ("invariant", "default"), "Unknown kernel policy")
    integer(plan["max_model_len"], "max_model_len", cap + 1)
    integer(plan["max_num_batched_tokens"], "max_num_batched_tokens", 8 * batch_size)
    fixture, input_sha = read_jsonl(input_path or plan["input"])
    compare(plan["input_sha256"], input_sha, "fixture SHA256")
    require(len(fixture) >= limit, "Fixture contains fewer examples than the plan requires")
    fixture = fixture[:limit]
    source_ids = [f"gsm8k-test-{index:04d}" for index in range(limit)]
    compare(plan["source_ids"], source_ids, "ordered first-N plan source IDs")
    for index, example in enumerate(fixture):
        compare(example, dict(source_id=source_ids[index], dataset_index=index), "fixture order")
        token_ids(example["input_ids"], "fixture input IDs")
    compare(plan["prompt_tokens"], [len(example["input_ids"]) for example in fixture], "prompt lengths")
    provenance = plan.get("provenance")
    if provenance is not None:
        compare(provenance["split"], "test", "fixture split")
        if "input_sha256" in provenance:
            compare(provenance["input_sha256"], input_sha, "provenance input SHA256")
    root_manifest = read_json(directory / "manifest.json")
    compare(root_manifest, dict(status="completed", methods={method: "completed" for method in METHODS}), "run manifest")
    completed = read_json(directory / "completed.json")
    compare(completed, dict(schema_version=1, status="completed", backend="vllm_native_v2",
                            input_sha256=input_sha, requests=2 * limit), "completed run")
    all_records, scores, performances, graphs, eos_by_method = {}, {}, {}, {}, {}
    for method in METHODS:
        folder = directory / method
        manifest = read_json(folder / "manifest.json")
        compare(manifest, dict(status="completed", method=method, measured_requests=limit,
                               llm_options=plan["llm_options"][method]), method + ".manifest")
        options = manifest["llm_options"]
        compare(options, dict(model=plan["target"], dtype="bfloat16", tensor_parallel_size=1,
                              max_model_len=plan["max_model_len"], max_num_seqs=batch_size,
                              max_num_batched_tokens=plan["max_num_batched_tokens"],
                              kv_cache_memory_bytes=int(plan["kv_cache_gib"] * 1024 ** 3),
                              seed=plan["seed"], enforce_eager=False, enable_prefix_caching=False,
                              enable_chunked_prefill=True, async_scheduling=True,
                              attention_config=dict(backend="FLASH_ATTN"),
                              compilation_config=dict(cudagraph_mode="FULL_AND_PIECEWISE")), method + ".engine options")
        if method == "longspark":
            compare(options, dict(per_request_spec_decode_metrics="detailed",
                                  speculative_config=dict(method="custom_class", model=SPECULATOR,
                                                          num_speculative_tokens=7, draft_sample_method="greedy"),
                                  additional_config=dict(native_v2_custom_class=True,
                                                         longspark=dict(draft_model=plan["draft"]))),
                    "LongSpark native configuration")
        else:
            require(not options.get("speculative_config"), "AR options unexpectedly enable speculative decoding")
        environment = manifest["environment"]
        compare(environment["vllm_use_v2_model_runner"], "1", method + ".V2 environment")
        compare(environment["vllm_batch_invariant"], "1" if plan["kernel_policy"] == "invariant" else "0",
                method + ".kernel policy")
        require(environment["vllm"] in ("0.30.0", "0.30.0+cu129"), "Unexpected audited vLLM release")
        if plan.get("flashinfer_sampler") is False:
            compare(environment["vllm_use_flashinfer_sampler"], "0", method + ".sampler policy")
        loaded = manifest["after_load"]
        require(isinstance(loaded, list) and len(loaded) == 1, "Missing TP1 after-load snapshot")
        audit_snapshot(loaded[0], method)
        eos = set(token_ids(manifest["eos"]["effective_token_ids"], "effective EOS IDs"))
        eos_by_method[method] = sorted(eos)
        rows, _ = read_jsonl(folder / "records.jsonl")
        require(len(rows) == limit, f"{method}: expected {limit} completed records, found {len(rows)}")
        compare([row["source_id"] for row in rows], source_ids, method + ".ordered complete coverage")
        require(len({row["request_id"] for row in rows}) == limit, f"{method}: duplicate request IDs")
        checked = [audit_record(row, example, method, eos, cap, index // batch_size)
                   for index, (row, example) in enumerate(zip(rows, fixture))]
        all_records[method] = checked
        batches = [item for item in read_json(folder / "batches.json") if item["phase"] == "measurement"]
        require(len(batches) == math.ceil(limit / batch_size), f"{method}: missing measured batch wall times")
        for index, batch in enumerate(batches):
            batch_rows = checked[index * batch_size:(index + 1) * batch_size]
            compare(batch, dict(batch_index=index, source_ids=[row["source_id"] for row in batch_rows],
                                submitted_concurrently=len(batch_rows),
                                output_tokens=sum(row["output_tokens"] for row in batch_rows)), method + ".batch")
        scores[method] = scoring(checked)
        performances[method] = performance(checked, batches)
        summary = read_json(folder / "summary.json")
        compare(summary["performance"], performances[method], method + ".performance summary")
        compare(summary["scoring"], dict(scores[method], by_method={method: scores[method]}), method + ".scoring summary")
        graphs[method] = audit_graphs(read_json(folder / "measurement-audit.json"), method,
                                     need_replays=any(row["output_tokens"] > 1 for row in checked),
                                     max_num_seqs=batch_size)
    compare(eos_by_method["longspark"], eos_by_method["vanilla"], "shared natural EOS IDs")
    parity = []
    for left, right in zip(all_records["vanilla"], all_records["longspark"]):
        lhs, rhs = left["output_ids"], right["output_ids"]
        first = next((i for i in range(min(len(lhs), len(rhs))) if lhs[i] != rhs[i]), None)
        if first is None and len(lhs) != len(rhs):
            first = min(len(lhs), len(rhs))
        parity.append(dict(source_id=left["source_id"], exact_match=lhs == rhs, first_mismatch_index=first,
                           vanilla_tokens=len(lhs), longspark_tokens=len(rhs)))
    parity_result = dict(passed=all(row["exact_match"] for row in parity),
                         matches=sum(row["exact_match"] for row in parity), comparisons=limit, checks=parity)
    compare(completed["parity"], parity_result, "recorded token parity")
    require(parity_result["passed"], "Exact AR/LongSpark token parity failed: " +
            str([row for row in parity if not row["exact_match"]][:3]))
    compare(completed["greedy_parity_passed"], True, "completed parity flag")
    compare(completed["correctness_status"], "passed", "completed correctness status")
    compare(root_manifest["greedy_parity_passed"], True, "root parity flag")
    compare(completed["performance"], performances, "completed performance")
    combined_scores = dict(scoring(all_records["vanilla"] + all_records["longspark"]), by_method=scores)
    compare(completed["scoring"], combined_scores, "completed scoring")
    speedup = performances["longspark"]["aggregate_service_tps"] / performances["vanilla"]["aggregate_service_tps"]
    compare(completed["aggregate_service_speedup"], speedup, "service speedup")
    return dict(schema_version=1, status="passed", backend="vllm_native_v2", directory=str(directory.resolve()),
                input_sha256=input_sha, required_requests_per_method=limit,
                verified_requests_per_method={method: len(rows) for method, rows in all_records.items()},
                token_parity=dict(matches=limit, comparisons=limit, passed=True),
                scoring={method: {key: scores[method][key] for key in
                                  ("correct", "incorrect", "unparsed", "accuracy", "eos_count", "truncated_count")}
                         for method in METHODS},
                performance=performances, graphs=graphs, aggregate_service_speedup=speedup,
                scope="Offline evidence consistency and exact token parity; no GPU execution or checkpoint authenticity certification")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Native runner output directory")
    parser.add_argument("--input", type=Path, help="Local copy of the exact same input fixture; SHA256 must match")
    parser.add_argument("--expect-limit", type=int, help="Require a specific plan limit, e.g. 100")
    parser.add_argument("--output", type=Path, help="Save a new audit JSON; never overwrite an existing file")
    args = parser.parse_args(argv)
    try:
        report = audit_directory(args.directory, input_path=args.input, expect_limit=args.expect_limit)
        code = 0
    except (AuditError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report = dict(schema_version=1, status="failed", directory=str(args.directory.resolve()),
                      error=f"{type(error).__name__}: {error}")
        code = 1
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x") as stream:
                stream.write(encoded)
        except OSError as error:
            print(json.dumps(dict(status="failed", error=f"Cannot save audit report: {error}")), file=sys.stderr)
            return 1
    print(encoded, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
