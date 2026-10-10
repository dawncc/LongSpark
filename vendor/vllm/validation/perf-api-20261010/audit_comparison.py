#!/usr/bin/env python3
"""Audit the saved baseline/C1/C4 evidence, without model or GPU imports.

All numeric results are recomputed from raw records through the independent
native auditor. Source hashes are checked against the baseline Git blobs or
current local source files. The output is created exclusively, never replaced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


BASELINE_COMMIT = "2542b709249ea155e1c8b4442fa1088cb4e1d84f"
RUNS = {"baseline-c1": 1, "optimized-c1": 1, "optimized-c4": 4}
METHODS = ("vanilla", "longspark")
FIXED_SOURCES = (
    "vendor/vllm/longspark_vllm/gsm8k.py",
    "vendor/vllm/scripts/evaluate_native_v2_gsm8k.py",
    "vendor/vllm/patches/native-v2-custom-class.manifest.json",
)
SOURCE_DIRECTORIES = ("vendor/vllm/longspark_vllm/native_v2", "vendor/specforge/modeling/draft")
HISTORICAL_COMPOSITION = (
    "vendor/vllm/validation/native-v2-gsm8k100-20260930/"
    "native-c1-full-v2/provenance/audit-composition.py"
)
MATCHED_PLAN_FIELDS = (
    "backend", "methods", "input_sha256", "provenance", "target", "draft", "limit",
    "source_ids", "prompt_tokens", "warmup_count", "debug_without_warmup",
    "max_new_tokens", "max_model_len", "max_num_batched_tokens",
    "seed", "kernel_policy", "temperature", "ignore_eos", "async_scheduling", "target_tp",
    "dtype", "retokenize", "flashinfer_sampler", "sampler_policy", "method_isolation",
    "timing_definition", "wall_timing_definition", "acceptance_definition",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def records(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    require(len(rows) == 100, f"{path}: expected exactly 100 records")
    require([row["source_id"] for row in rows] == [f"gsm8k-test-{i:04d}" for i in range(100)],
            f"{path}: ordered first-100 source IDs differ")
    return rows


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_selected(path):
    part = Path(path)
    return ((path.startswith(SOURCE_DIRECTORIES[0] + "/") and part.suffix == ".py") or
            (path.startswith(SOURCE_DIRECTORIES[1] + "/") and
             part.parent.as_posix() == SOURCE_DIRECTORIES[1] and part.suffix == ".py" and
             part.name.startswith(("draftfreekv", "global16_raw256"))))


def expected_sources(root, baseline):
    """Independently enumerate the runner's documented source coverage."""
    if baseline:
        resolved = subprocess.check_output(
            ["git", "rev-parse", BASELINE_COMMIT + "^{commit}"], cwd=root, text=True).strip()
        require(resolved == BASELINE_COMMIT, "Baseline commit did not resolve exactly")
        tree = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", BASELINE_COMMIT, *SOURCE_DIRECTORIES],
            cwd=root, text=True).splitlines()
        paths = set(FIXED_SOURCES) | {path for path in tree if source_selected(path)}
        values = {path: hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{BASELINE_COMMIT}:{path}"], cwd=root)).hexdigest()
                  for path in sorted(paths)}
    else:
        paths = set(FIXED_SOURCES)
        for directory in SOURCE_DIRECTORIES:
            paths.update(path.relative_to(root).as_posix() for path in (root / directory).rglob("*.py")
                         if source_selected(path.relative_to(root).as_posix()))
        for path in paths:
            require((root / path).resolve().is_relative_to(root), "Source escapes repository: " + path)
        values = {path: sha(root / path) for path in sorted(paths)}
    require(values and not any(Path(path).name.startswith("._") for path in values),
            "Source set is empty or contains AppleDouble sidecars")
    return values


def audit_sources(root, directory, plan, baseline):
    expected = expected_sources(root, baseline)
    require(plan["integration_source_hashes"] == expected,
            directory.name + ": plan source map differs from " +
            ("baseline Git blobs" if baseline else "current local files") + ": " +
            repr(sorted(path for path in set(expected) | set(plan["integration_source_hashes"])
                        if expected.get(path) != plan["integration_source_hashes"].get(path))))
    checks = {}
    for name in ("manifest.json", "completed.json", "vanilla/manifest.json", "longspark/manifest.json"):
        path = directory / name
        manifest = read(path)
        require(manifest["integration_source_hashes"] == expected, directory.name + "/" + name + ": source map mismatch")
        checks[name] = dict(sha256=sha(path), source_map_matches=True)
        if name.endswith("/manifest.json"):
            require(manifest["integration_sources_verified_before_load"] is True,
                    directory.name + "/" + name + ": missing pre-load source verification")
            checks[name]["verified_before_model_load"] = True
    return dict(status="passed", authority="git_blobs" if baseline else "current_local_files",
                baseline_commit=BASELINE_COMMIT if baseline else None, source_count=len(expected),
                files=expected, manifest_checks=checks)


def evidence_hashes(directory):
    names = ["plan.json", "manifest.json", "completed.json"]
    for method in METHODS:
        names.append(method + ".log")
        names.extend(method + "/" + name for name in (
            "manifest.json", "records.jsonl", "batches.json", "summary.json",
            "measurement-audit.json", "prefill-warmup.json"))
    return {name: sha(directory / name) for name in names}


def audit_run(root, directory, input_path, base_auditor, composition, expected_concurrency):
    plan = read(directory / "plan.json")
    require(plan["max_num_seqs"] == expected_concurrency, directory.name + ": wrong concurrency")
    require(plan["gpu"] == (1 if directory.name == "optimized-c4" else 0), directory.name + ": wrong GPU")
    baseline = directory.name == "baseline-c1"
    if not baseline:
        require(plan["native_optimizations"] == "fused", directory.name + ": optimizations are disabled")
        options = plan["llm_options"]["longspark"]["additional_config"]["longspark"]
        require(options["use_fused_state_commit"] is True and options["use_direct_gather"] is True,
                directory.name + ": both native optimization flags must be true")
    result = base_auditor.audit_directory(directory, input_path=input_path, expect_limit=100)
    fixture = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    # Only these two checks are reused; the historical source audit's fixed
    # source count and remote-path assumptions do not apply to this experiment.
    result["prefill_shape_warmup_audit"] = composition.audit_warmup(directory, plan, fixture)
    result["compilation_log_audit"] = composition.audit_logs(directory, plan)
    result["source_integrity"] = audit_sources(root, directory, plan, baseline)
    for method in METHODS:
        measurement = read(directory / method / "measurement-audit.json")[0]
        before, after = measurement["before"], measurement["after"]
        if method == "longspark":
            require(before["speculator"]["startup_eager_calls"] == after["speculator"]["startup_eager_calls"],
                    directory.name + ": startup eager execution occurred during measurement")
            if not baseline:
                for snapshot in (before, after):
                    require(snapshot["speculator"]["direct_gather"] is True and
                            snapshot["speculator"]["fused_state_commit"] is True,
                            directory.name + ": worker optimization settings differ from plan")
        scores = result["scoring"][method]
        require(scores["eos_count"] == 100 and scores["truncated_count"] == 0 and scores["unparsed"] == 0,
                directory.name + "/" + method + ": expected complete natural-EOS parseable outputs")
    result["evidence_sha256"] = evidence_hashes(directory)
    return result, plan


def compare_records(directories):
    rows = {name: {method: records(folder / method / "records.jsonl") for method in METHODS}
            for name, folder in directories.items()}
    baseline = rows["baseline-c1"]
    token_checks, acceptance_checks = {}, {}
    for name, methods in rows.items():
        token_checks[name] = {}
        for method, data in methods.items():
            mismatches = [row["source_id"] for ref, row in zip(baseline["vanilla"], data)
                          if row["output_ids"] != ref["output_ids"]]
            require(not mismatches, f"{name}/{method}: cross-run output token mismatch: {mismatches}")
            token_checks[name][method] = dict(matches=100, comparisons=100, exact=True)
        differences = []
        for ref, row in zip(baseline["longspark"], methods["longspark"]):
            if (row["raw_spec_decode_metrics"] != ref["raw_spec_decode_metrics"] or
                    row["per_step_accepted_emitted"] != ref["per_step_accepted_emitted"]):
                differences.append(dict(source_id=row["source_id"],
                    baseline_raw=ref["raw_spec_decode_metrics"], actual_raw=row["raw_spec_decode_metrics"],
                    baseline_accepted_emitted=ref["per_step_accepted_emitted"],
                    actual_accepted_emitted=row["per_step_accepted_emitted"]))
        # C1 before/after must preserve proposals as well as target output.
        # A different draft batch shape may legitimately alter draft proposals
        # while native target verification preserves the same greedy output.
        required_exact = name != "optimized-c4"
        require(not required_exact or not differences,
                f"{name}: same-C1 LongSpark accepted/drafted sequence mismatch: " +
                repr([row["source_id"] for row in differences]))
        acceptance_checks[name] = dict(matches=100 - len(differences), comparisons=100,
            required_exact=required_exact, raw_and_emitted_sequences_exact=not differences,
            different_requests=differences,
            interpretation=("Same-concurrency before/after parity is required." if required_exact else
                            "C4 draft acceptance may differ with batch shape; exact target output remains required."))
    return dict(status="passed", reference="baseline-c1/vanilla", output_tokens=token_checks,
                longspark_acceptance_reference="baseline-c1/longspark", acceptance_sequences=acceptance_checks)


def summarize(runs):
    old = runs["baseline-c1"]["performance"]
    new = runs["optimized-c1"]["performance"]
    c4 = runs["optimized-c4"]["performance"]
    metric_names = ("aggregate_decode_tps", "aggregate_service_tps", "generate_wall_tps")
    c1 = {}
    for method in METHODS:
        c1[method] = {metric: dict(before=old[method][metric], after=new[method][metric],
                                  ratio=new[method][metric] / old[method][metric],
                                  percent_change=100 * (new[method][metric] / old[method][metric] - 1))
                      for metric in metric_names}
    c1["longspark_over_ar"] = {
        metric: dict(before=old["longspark"][metric] / old["vanilla"][metric],
                     after=new["longspark"][metric] / new["vanilla"][metric],
                     ratio_of_speedups=(new["longspark"][metric] / new["vanilla"][metric]) /
                                       (old["longspark"][metric] / old["vanilla"][metric]))
        for metric in metric_names}
    system = {method: dict(output_tokens=c4[method]["output_tokens"],
                           generate_wall_seconds=c4[method]["generate_wall_seconds"],
                           system_output_tps=c4[method]["generate_wall_tps"],
                           submitted_concurrency=c4[method]["max_submitted_concurrency"],
                           versus_optimized_c1_wall_ratio=c4[method]["generate_wall_tps"] /
                                                        new[method]["generate_wall_tps"])
              for method in METHODS}
    return dict(before_after_c1=c1,
                optimized_c4=dict(methods=system,
                    longspark_over_ar_wall_speedup=c4["longspark"]["generate_wall_tps"] / c4["vanilla"]["generate_wall_tps"],
                    throughput_definition="total emitted output tokens / sum of measured generate-call wall times",
                    sum_request_decode_tps_is_system_throughput=False,
                    caveat="C4 uses GPU 1 and a 4 GiB KV cache while C1 uses GPU 0 and 2 GiB; this measures batched throughput, not a same-GPU, same-configuration isolated native-code speedup."),
                scoring={name: run["scoring"] for name, run in runs.items()},
                acceptance={name: {key: run["performance"]["longspark"][key] for key in (
                    "verification_rounds", "proposed_tokens", "accepted_draft_tokens_raw",
                    "accepted_draft_tokens_emitted", "mean_accepted_draft_tokens", "tau", "tau_raw",
                    "acceptance_rate", "acceptance_rate_raw", "effective_advance")}
                            for name, run in runs.items()},
                graphs={name: run["graphs"] for name, run in runs.items()})


def audit_comparison(root, directory, input_path, *, baseline_only=False):
    root, directory, input_path = root.resolve(), directory.resolve(), input_path.resolve()
    audit_path = root / "vendor/vllm/scripts/audit_native_v2_gsm8k.py"
    historical_path = root / HISTORICAL_COMPOSITION
    base_auditor = load_module("comparison_native_auditor", audit_path)
    composition = load_module("comparison_warmup_log_checks", historical_path)
    names = ("baseline-c1",) if baseline_only else tuple(RUNS)
    directories = {name: directory / name for name in names}
    results, plans = {}, {}
    for name, folder in directories.items():
        results[name], plans[name] = audit_run(root, folder, input_path, base_auditor, composition, RUNS[name])
    for name in names:
        for key in MATCHED_PLAN_FIELDS:
            require(plans[name][key] == plans["baseline-c1"][key], name + ": unmatched plan field " + key)
        if name != "optimized-c4":
            require(plans[name]["kv_cache_gib"] == plans["baseline-c1"]["kv_cache_gib"],
                    name + ": C1 KV cache budget differs")
    if not baseline_only:
        require(plans["optimized-c1"]["integration_source_hashes"] == plans["optimized-c4"]["integration_source_hashes"],
                "Optimized C1/C4 ran different integration sources")
    parity = compare_records(directories)
    metadata_path = input_path.with_name(input_path.stem + ".metadata.json")
    input_metadata = read(metadata_path)
    require(input_metadata == plans["baseline-c1"]["provenance"],
            "Input fixture metadata differs from recorded dataset provenance")
    report = dict(schema_version=1, status="baseline_passed" if baseline_only else "passed",
                  audited_at_utc=datetime.now(timezone.utc).isoformat(), root=str(root), directory=str(directory),
                  baseline_commit=BASELINE_COMMIT, input_path=str(input_path), input_sha256=sha(input_path),
                  input_metadata=dict(path=str(metadata_path), sha256=sha(metadata_path),
                                      matches_recorded_provenance=True),
                  tools_sha256={"comparison": sha(__file__), "native_auditor": sha(audit_path),
                                "historical_warmup_logs": sha(historical_path)},
                  matched_plan_fields=list(MATCHED_PLAN_FIELDS),
                  run_configuration={name: {key: plan[key] for key in
                      ("gpu", "max_num_seqs", "kv_cache_gib", "prefill_shape_warmup")}
                      for name, plan in plans.items()},
                  runs=results, cross_run_parity=parity,
                  scope="Offline independent recomputation, source provenance, log/warmup/graph checks; no model checkpoint authenticity certification or new GPU execution.")
    if not baseline_only:
        report["comparison"] = summarize(results)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, help="Exclusive output file; defaults to comparison-audit.json for full audit")
    parser.add_argument("--baseline-only", action="store_true", help="Validate downloaded baseline without requiring or claiming the final comparison")
    args = parser.parse_args(argv)
    try:
        report = audit_comparison(args.root, args.directory,
                                  args.input or args.root / "vendor/vllm/validation/gsm8k100/inputs.jsonl",
                                  baseline_only=args.baseline_only)
        output = args.output or (None if args.baseline_only else args.directory / "comparison-audit.json")
        encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if output is not None:
            with output.open("x") as stream:
                stream.write(encoded)
        print(json.dumps(dict(status=report["status"], output=str(output) if output else None,
                              input_sha256=report["input_sha256"],
                              source_counts={name: run["source_integrity"]["source_count"]
                                             for name, run in report["runs"].items()},
                              comparison=report.get("comparison")), indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.CalledProcessError) as error:
        print(json.dumps(dict(status="failed", error=f"{type(error).__name__}: {error}"), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
