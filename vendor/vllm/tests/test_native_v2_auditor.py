"""Independent offline native-result audits: arithmetic, evidence and failures."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_script("audit_native_v2_gsm8k")
runner = load_script("evaluate_native_v2_gsm8k")
from longspark_vllm.gsm8k import canonical_prompt, summarize_records


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def change(path, function):
    value = json.loads(path.read_text())
    function(value)
    write(path, value)


def raw_counts(accepted):
    return dict(per_step_accepted=accepted, per_step_drafted=[7] * len(accepted),
                num_spec_steps=len(accepted), num_draft_tokens=7 * len(accepted),
                num_accepted_draft_tokens=sum(accepted))


def snapshot(method, replay_count=0):
    result = dict(runner_class=audit.RUNNER, use_v2_model_runner=True, async_scheduling=True,
                  target=dict(manager_class=audit.TARGET_MANAGER, captures=2,
                              captured_shapes=["batch1", "batch2"]),
                  speculator_class=None, speculator=None)
    if method == "longspark":
        result.update(speculator_class=audit.SPECULATOR,
                      speculator=dict(implementation="longspark_native_model_runner_v2",
                                      cuda_graph_manager=audit.DRAFT_MANAGER, target_forward_owned_by=audit.RUNNER,
                                      draft_tokens=7, captures=2, replays=replay_count, fallbacks=0))
    return result


@pytest.fixture
def evidence(tmp_path):
    fixture = tmp_path / "input.jsonl"
    examples = [dict(source_id=f"gsm8k-test-{i:04d}", dataset_index=i,
                     question=f"What is {i} plus one?", reference_answer=f"#### {i + 1}",
                     gold_answer=str(i + 1), prompt=canonical_prompt(f"What is {i} plus one?"),
                     input_ids=[i + 1, 42]) for i in range(3)]
    fixture.write_text("".join(json.dumps(row) + "\n" for row in examples))
    directory = tmp_path / "native"
    directory.mkdir()
    args = runner.parser().parse_args([
        "--input", str(fixture), "--target", str(tmp_path / "target"), "--draft", str(tmp_path / "draft"),
        "--output", str(directory), "--limit", "3", "--max-new-tokens", "6", "--max-num-seqs", "2",
    ])
    plan = runner.make_plan(args)
    write(directory / "plan.json", plan)
    all_rows, performances = [], {}
    ids = [[5, 6, 99], [7, 8, 9, 10, 11, 99], [4, 5, 6, 7, 8, 9]]
    # Includes EOS within an accepted span, EOS at cap, and truncation at cap.
    accepted = [[6], [1, 6], [2, 1]]
    for method in audit.METHODS:
        folder = directory / method
        folder.mkdir()
        rows = []
        for index, example in enumerate(examples):
            raw = raw_counts(accepted[index]) if method == "longspark" else None
            completion = SimpleNamespace(token_ids=ids[index], text=f"Reasoning.\n#### {12 if index == 1 else index + 1}",
                                         finish_reason="length" if index == 2 else "stop",
                                         stop_reason=None if index == 2 else 99,
                                         spec_decode_metrics=SimpleNamespace(to_dict=lambda raw=raw: raw) if raw else None)
            output = SimpleNamespace(finished=True, outputs=[completion], prompt_token_ids=example["input_ids"],
                                     request_id=f"{method}-{index}",
                                     metrics=SimpleNamespace(scheduled_ts=100.0 + index * 10,
                                                             first_token_ts=101.0 + index * 10,
                                                             last_token_ts=103.0 + index * 11))
            row = runner.record_output(example, output, method=method, eos=[99],
                                       max_new_tokens=6, batch_index=index // 2)
            row["excluded_from_metrics"] = False
            rows.append(row)
        (folder / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        batches = [dict(phase="measurement", batch_index=i,
                        source_ids=[row["source_id"] for row in rows[i * 2:(i + 1) * 2]],
                        submitted_concurrently=len(rows[i * 2:(i + 1) * 2]),
                        output_tokens=sum(row["output_tokens"] for row in rows[i * 2:(i + 1) * 2]),
                        wall_seconds=9.0 if i == 0 else 6.0) for i in range(2)]
        write(folder / "batches.json", batches)
        manifest = dict(status="completed", method=method, measured_requests=3,
                        llm_options=plan["llm_options"][method],
                        environment=dict(vllm_use_v2_model_runner="1", vllm_batch_invariant="1",
                                         vllm_use_flashinfer_sampler="0", vllm="0.30.0+cu129"),
                        eos=dict(effective_token_ids=[99]), after_load=[snapshot(method)])
        write(folder / "manifest.json", manifest)
        graph = dict(label="measurement", before=snapshot(method, 1), after=snapshot(method, 6),
                     target=dict(replays=5, replay_shapes={"batch1": 2, "batch2": 3},
                                 dispatches={"uniform_decode:FULL": 5, "other:NONE": 2},
                                 max_dispatched_requests=2, max_decode_requests=2,
                                 captures_delta=0, manager_replaced=False),
                     speculator_delta=dict(captures=0, replays=5, fallbacks=0) if method == "longspark" else None)
        write(folder / "measurement-audit.json", [graph])
        perf = runner.performance_summary(rows)
        perf.update(generate_wall_seconds=15.0, generate_wall_tps=1.0, max_submitted_concurrency=2)
        write(folder / "summary.json", dict(scoring=summarize_records(rows), performance=perf))
        all_rows.extend(rows)
        performances[method] = perf
    write(directory / "manifest.json", dict(status="completed", methods={name: "completed" for name in audit.METHODS},
                                            greedy_parity_passed=True))
    write(directory / "completed.json", dict(schema_version=1, status="completed", backend="vllm_native_v2",
                                             input_sha256=plan["input_sha256"], requests=6,
                                             greedy_parity_passed=True, correctness_status="passed",
                                             parity=runner.compare_outputs(all_rows),
                                             scoring=summarize_records(all_rows), performance=performances,
                                             aggregate_service_speedup=1.0))
    return directory


def test_independent_full_audit_recomputes_raw_emitted_time_and_accuracy(evidence):
    result = audit.audit_directory(evidence, expect_limit=3)
    assert result["status"] == "passed"
    assert result["token_parity"] == dict(matches=3, comparisons=3, passed=True)
    assert result["scoring"]["longspark"] == dict(correct=2, incorrect=1, unparsed=0, accuracy=2 / 3,
                                                 eos_count=2, truncated_count=1)
    performance = result["performance"]["longspark"]
    assert performance["accepted_draft_tokens_raw"] == 16
    assert performance["accepted_draft_tokens_emitted"] == 9
    assert performance["tau"] == 2.8
    assert performance["tau_raw"] == 4.2
    assert performance["generate_wall_tps"] == 1.0
    assert performance["aggregate_service_tps"] == 15 / 12
    assert performance["aggregate_decode_tps"] == 12 / 9
    assert result["graphs"]["longspark"]["draft_replays"] == 5


@pytest.mark.parametrize("raw,tokens", [
    (None, 2), (raw_counts([2]), 5), (raw_counts([2, 2]), 2),
    (raw_counts([8]), 2), (dict(raw_counts([2]), num_spec_steps=2), 2),
])
def test_invalid_acceptance_is_rejected(raw, tokens):
    with pytest.raises(audit.AuditError):
        audit.recompute_acceptance(raw, tokens, "longspark")


def test_eos_clipping_and_single_token_edges():
    assert audit.recompute_acceptance(raw_counts([2, 6]), 6, "longspark")["per_step_accepted_emitted"] == [2, 2]
    assert audit.recompute_acceptance(None, 1, "longspark")["verification_rounds"] == 0
    assert audit.parse_answer("#### -$1,200.000\nignored") == ("-1200", "-$1,200.000", "parsed")
    assert audit.parse_answer("#### 12 apples")[0] is None
    assert audit.parse_answer("#### ١,٢٣٤")[0] is None
    assert audit.parse_answer("Reasoning says 42")[2] == "missing_marker"


@pytest.mark.parametrize("path,mutate,match", [
    ("longspark/summary.json", lambda x: x["performance"].update(tau=99.0), "tau"),
    ("vanilla/summary.json", lambda x: x["performance"].update(generate_wall_tps=99.0), "wall_tps"),
    ("completed.json", lambda x: x.update(requests=100), "requests"),
    ("longspark/manifest.json", lambda x: x.update(measured_requests=2), "measured_requests"),
    ("longspark/measurement-audit.json", lambda x: x[0]["after"].update(async_scheduling=False), "async"),
    ("longspark/measurement-audit.json", lambda x: x[0]["after"].update(runner_class="ManualTarget"), "runner_class"),
    ("longspark/measurement-audit.json", lambda x: x[0]["after"]["target"].update(manager_class=audit.DRAFT_MANAGER), "target manager"),
    ("longspark/measurement-audit.json", lambda x: x[0]["after"]["speculator"].update(cuda_graph_manager=audit.TARGET_MANAGER), "cuda_graph_manager"),
    ("longspark/measurement-audit.json", lambda x: x[0]["target"].update(replays=0), "replays"),
    ("longspark/measurement-audit.json", lambda x: x[0]["target"].update(captures_delta=1), "captures_delta"),
    ("longspark/measurement-audit.json", lambda x: x[0]["speculator_delta"].update(fallbacks=1), "fallbacks"),
    ("longspark/measurement-audit.json", lambda x: x[0]["target"]["dispatches"].update({"uniform_decode:NONE": 1}), "fallback"),
    ("longspark/measurement-audit.json", lambda x: x[0]["target"].update(replay_shapes={"unknown": 5}), "not captured"),
    ("vanilla/batches.json", lambda x: x[0].update(output_tokens=999), "output_tokens"),
])
def test_bad_summaries_and_graph_evidence_fail(evidence, path, mutate, match):
    change(evidence / path, mutate)
    with pytest.raises(audit.AuditError, match=match):
        audit.audit_directory(evidence)


@pytest.mark.parametrize("mutate,match", [
    (lambda rows: rows.pop(), "completed records"),
    (lambda rows: rows.reverse(), "coverage"),
    (lambda rows: rows[0].update(correct=False), "correct"),
    (lambda rows: rows[0].update(decode_seconds=99.0), "decode_seconds"),
    (lambda rows: rows[0].update(output_ids=[5, 99, 6]), "after EOS"),
    (lambda rows: rows[0].update(excluded_from_metrics=True), "excluded_from_metrics"),
    (lambda rows: rows[0].update(accepted_draft_tokens_emitted=6), "accepted_draft_tokens_emitted"),
    (lambda rows: rows[0].update(eos_token_ids=[6]), "eos_token_ids"),
    (lambda rows: rows[0].update(output_ids=[7, 6, 99]), "parity"),
])
def test_bad_records_fail_even_when_completion_claims_success(evidence, mutate, match):
    path = evidence / "longspark/records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(audit.AuditError, match=match):
        audit.audit_directory(evidence)


def test_fixture_digest_and_expected_limit_are_required(evidence, tmp_path):
    original = Path(json.loads((evidence / "plan.json").read_text())["input"])
    copied = tmp_path / "relocated.jsonl"
    copied.write_bytes(original.read_bytes())
    original.unlink()
    assert audit.audit_directory(evidence, input_path=copied)["status"] == "passed"
    copied.write_text(copied.read_text() + " ")
    with pytest.raises(audit.AuditError, match="SHA256|blank record"):
        audit.audit_directory(evidence, input_path=copied)
    with pytest.raises(audit.AuditError, match="limit"):
        audit.audit_directory(evidence, expect_limit=100)


def test_cli_is_gpu_free_read_only_and_never_overwrites_report(evidence, tmp_path):
    report = tmp_path / "audit.json"
    before = {path: path.read_bytes() for path in evidence.rglob("*") if path.is_file()}
    block_gpu_imports = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'transformers', 'specforge', 'sglang'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    cmd = [sys.executable, "-S", "-c", block_gpu_imports, str(SCRIPTS / "audit_native_v2_gsm8k.py"),
           str(evidence), "--expect-limit", "3", "--output", str(report)]
    process = subprocess.run(cmd, text=True, capture_output=True, check=True)
    assert json.loads(process.stdout)["status"] == "passed"
    assert before == {path: path.read_bytes() for path in before}
    saved = report.read_bytes()
    again = subprocess.run(cmd, text=True, capture_output=True)
    assert again.returncode == 1 and report.read_bytes() == saved
    (evidence / "completed.json").unlink()
    failed = subprocess.run([sys.executable, str(SCRIPTS / "audit_native_v2_gsm8k.py"), str(evidence)],
                            text=True, capture_output=True)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["status"] == "failed"
