"""CPU contracts for native engine evidence, timing, and stop-aware statistics."""

from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[1] / "scripts/evaluate_native_v2_gsm8k.py"
SPEC = importlib.util.spec_from_file_location("native_v2_gsm8k_runner", ENTRY)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
from longspark_vllm.gsm8k import canonical_prompt
from longspark_vllm.native_v2.diagnostics import NativeV2Diagnostics


def raw_metrics(accepted, drafted=None):
    drafted = drafted or [7] * len(accepted)
    return dict(num_spec_steps=len(accepted), num_accepted_draft_tokens=sum(accepted),
                num_draft_tokens=sum(drafted), per_step_accepted=accepted, per_step_drafted=drafted)


class FakeManager:
    def __init__(self):
        self.graphs = {"capture": object()}
        self.calls = 0

    def dispatch(self, *args, **kwargs):
        return SimpleNamespace(cg_mode=SimpleNamespace(name="FULL"))

    def run_fullgraph(self, desc):
        self.calls += 1
        return "native output"


class FakeSpeculator:
    def __init__(self):
        self.replays = 0

    def runtime_stats(self):
        return dict(captures=1, replays=self.replays, fallbacks=0,
                    startup_complete=True, startup_eager_calls=3)


FakeSpeculator.__module__ = "longspark_vllm.native_v2.speculator"
FakeSpeculator.__qualname__ = "LongSparkSpeculator"
FakeRunner = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"})


class FakeWorker(NativeV2Diagnostics):
    def __init__(self, speculative=False):
        self.model_runner = FakeRunner()
        self.model_runner.cudagraph_manager = FakeManager()
        self.model_runner.speculator = FakeSpeculator() if speculative else None
        self.model_runner.decode_query_len = 8 if speculative else 1
        self.vllm_config = SimpleNamespace(use_v2_model_runner=True,
                                         scheduler_config=SimpleNamespace(async_scheduling=True))


@dataclass
class Timing:
    scheduled_ts: float = 100.0
    first_token_ts: float = 101.0
    last_token_ts: float = 103.0
    arrival_time: float = 1900000000.0


class FakeTensor:
    def __init__(self, shape, *, dtype="bfloat16", device=None, value=0):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype, self.device, self.value = dtype, device, value
    def clone(self):
        return FakeTensor(self.shape, dtype=self.dtype, device=self.device, value=self.value)
    def view(self, dtype):
        return self


def fake_torch():
    module = ModuleType("torch")
    module.bfloat16, module.long, module.uint8 = "bfloat16", "long", "uint8"
    module.inference_mode = nullcontext
    module.cuda = SimpleNamespace(synchronize=lambda device: None)
    module.zeros = lambda shape, **kwargs: FakeTensor(shape, **kwargs)
    module.ones = lambda shape, **kwargs: FakeTensor(shape, value=1, **kwargs)
    module.zeros_like = lambda tensor: FakeTensor(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    module.arange = lambda start, end, **kwargs: FakeTensor((end - start,), value=list(range(start, end)), **kwargs)
    module.equal = lambda left, right: left.shape == right.shape and left.value == right.value
    return module


class NativeRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="native-v2-test ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixture = self.root / "input.jsonl"
        self.examples = [dict(source_id=f"gsm8k-test-{i:04d}", dataset_index=i,
                              question=f"What is {i} plus one?", reference_answer=f"#### {i + 1}",
                              gold_answer=str(i + 1), prompt=canonical_prompt(f"What is {i} plus one?"),
                              input_ids=[i + 1, 42]) for i in range(3)]
        self.fixture.write_text("".join(json.dumps(row) + "\n" for row in self.examples))
        self.target = self.root / "target"
        self.target.mkdir()
        (self.target / "config.json").write_text(json.dumps(dict(eos_token_id=99, vocab_size=200)))
        self.args = ["--input", str(self.fixture), "--target", str(self.target),
                     "--draft", str(self.root / "draft"), "--output", str(self.root / "output"),
                     "--limit", "3", "--max-new-tokens", "10", "--max-num-seqs", "2"]

    def plan(self, *extra):
        return runner.make_plan(runner.parser().parse_args(self.args + list(extra)))

    def test_preview_requires_no_inference_packages_and_keeps_engine_contract(self):
        code = """
import importlib.abc, os, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'transformers', 'specforge', 'sglang'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = 'unchanged'
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
assert os.environ['VLLM_USE_V2_MODEL_RUNNER'] == 'unchanged'
"""
        result = subprocess.run([sys.executable, "-S", "-c", code, str(ENTRY), *self.args],
                                text=True, capture_output=True, check=True)
        plan = json.loads(result.stdout)
        spec = plan["llm_options"]["longspark"]
        ar = plan["llm_options"]["vanilla"]
        self.assertEqual(spec["speculative_config"]["model"], runner.SPECULATOR)
        self.assertTrue(spec["async_scheduling"])
        self.assertFalse(spec["enable_prefix_caching"])
        self.assertTrue(spec["additional_config"]["native_v2_custom_class"])
        self.assertEqual(spec["per_request_spec_decode_metrics"], "detailed")
        self.assertEqual(spec["attention_config"], {"backend": "FLASH_ATTN"})
        self.assertEqual(spec["max_num_batched_tokens"], 512)
        self.assertFalse(plan["flashinfer_sampler"])
        self.assertIn("vendor/vllm/longspark_vllm/native_v2/diagnostics.py", plan["integration_source_hashes"])
        self.assertIn("vendor/vllm/longspark_vllm/native_v2/target_modules.py", plan["integration_source_hashes"])
        self.assertNotIn("vendor/vllm/longspark_vllm/drafter.py", plan["integration_source_hashes"])
        self.assertNotIn("vendor/vllm/scripts/evaluate_vllm_gsm8k.py", plan["integration_source_hashes"])
        self.assertNotIn("speculative_config", ar)
        self.assertNotIn("per_request_spec_decode_metrics", ar)
        self.assertFalse((self.root / "output").exists())

    def test_emitted_metrics_clip_eos_inside_final_accepted_span(self):
        # First ordinary token + (2 accepted and bonus) + 2 final accepted tokens.
        result = runner.acceptance_metrics(raw_metrics([2, 6]), 6, method="longspark")
        self.assertEqual(result["accepted_draft_tokens_raw"], 8)
        self.assertEqual(result["accepted_draft_tokens_emitted"], 4)
        self.assertEqual(result["per_step_accepted_emitted"], [2, 2])
        self.assertEqual(result["verification_rounds"], 2)
        self.assertEqual(runner.acceptance_metrics(raw_metrics([2]), 4, method="longspark")
                         ["accepted_draft_tokens_emitted"], 2)

    def test_inconsistent_native_metrics_fail_instead_of_inventing_statistics(self):
        for raw, tokens in ((None, 2), (raw_metrics([2]), 5), (raw_metrics([2, 2]), 2),
                            (raw_metrics([8]), 2), (dict(raw_metrics([2]), num_spec_steps=2), 2)):
            with self.subTest(raw=raw, tokens=tokens), self.assertRaises(ValueError):
                runner.acceptance_metrics(raw, tokens, method="longspark")
        self.assertEqual(runner.acceptance_metrics(None, 1, method="longspark")["verification_rounds"], 0)

    def output(self, *, speculative=False):
        raw = raw_metrics([2]) if speculative else None
        completion = SimpleNamespace(token_ids=[5, 99], text="#### 1", finish_reason="stop", stop_reason=99,
                                     spec_decode_metrics=SimpleNamespace(to_dict=lambda: raw) if raw else None)
        return SimpleNamespace(finished=True, outputs=[completion], prompt_token_ids=[1, 42],
                               request_id="request-1", metrics=Timing())

    def test_request_service_uses_same_clock_engine_intervals(self):
        row = runner.record_output(self.examples[0], self.output(speculative=True), method="longspark",
                                   eos=[99], max_new_tokens=10, batch_index=0)
        self.assertEqual(row["service_seconds"], 3)
        self.assertEqual(row["decode_seconds"], 2)
        self.assertEqual(row["prefill_seconds"], 1)
        self.assertEqual(row["accepted_draft_tokens_emitted"], 1)
        self.assertTrue(row["correct"])
        self.assertEqual(row["native_request_metrics"]["arrival_time"], 1900000000.0)

    def test_worker_audit_calls_native_graph_and_restores_original_methods(self):
        worker = FakeWorker(speculative=True)
        manager = worker.model_runner.cudagraph_manager
        original_replay, original_dispatch = manager.run_fullgraph, manager.dispatch
        worker.native_v2_audit_start("warmup")
        desc = manager.dispatch(1, 8, 8, 0)
        self.assertEqual(manager.run_fullgraph(desc), "native output")
        worker.model_runner.speculator.replays += 1
        audit = worker.native_v2_audit_end()
        self.assertEqual(audit["target"]["replays"], 1)
        self.assertEqual(manager.calls, 1)
        self.assertEqual(audit["target"]["dispatches"], {"uniform_decode:FULL": 1})
        self.assertEqual(manager.run_fullgraph, original_replay)
        self.assertEqual(manager.dispatch, original_dispatch)
        runner.require_audit([audit], method="longspark", require_replays=True)
        worker.native_v2_audit_start("measurement")
        clean = worker.native_v2_audit_end()
        self.assertEqual(clean["target"]["replays"], 0)
        self.assertEqual(clean["speculator_delta"]["replays"], 0)
        with self.assertRaisesRegex(RuntimeError, "no measured actual"):
            runner.require_audit([clean], method="longspark", require_replays=True)

    def test_graph_failure_is_not_counted_and_wrong_runner_is_rejected(self):
        worker = FakeWorker()
        def fail(_):
            raise RuntimeError("native failure")
        worker.model_runner.cudagraph_manager.run_fullgraph = fail
        worker.native_v2_audit_start()
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            worker.model_runner.cudagraph_manager.run_fullgraph(None)
        self.assertEqual(worker.native_v2_audit_end()["target"]["replays"], 0)
        worker.model_runner = SimpleNamespace(cudagraph_manager=FakeManager(), speculator=None)
        with self.assertRaisesRegex(RuntimeError, "Expected native V2"):
            worker.native_v2_audit_start()

    def test_audit_rejects_incomplete_startup_and_new_startup_eager_calls(self):
        worker = FakeWorker(speculative=True)
        worker.native_v2_audit_start()
        audit = worker.native_v2_audit_end()
        runner.require_audit([audit], method="longspark", require_replays=False)
        audit["before"]["speculator"]["startup_complete"] = False
        with self.assertRaisesRegex(RuntimeError, "startup was incomplete"):
            runner.require_audit([audit], method="longspark", require_replays=False)
        audit["before"]["speculator"]["startup_complete"] = True
        audit["after"]["speculator"]["startup_eager_calls"] += 1
        with self.assertRaisesRegex(RuntimeError, "startup eager calls changed"):
            runner.require_audit([audit], method="longspark", require_replays=False)

    def test_source_hashes_detect_modified_added_and_missing_native_files(self):
        package = self.root / "vendor/vllm/longspark_vllm/native_v2"
        package.mkdir(parents=True)
        source = package / "speculator.py"
        source.write_text("version = 1\n")
        hashes = runner.integration_source_hashes(root=self.root)
        self.assertEqual(runner.verify_integration_sources(hashes, root=self.root), hashes)
        source.write_text("version = 2\n")
        with self.assertRaisesRegex(ValueError, "sources changed"):
            runner.verify_integration_sources(hashes, root=self.root)
        source.write_text("version = 1\n")
        extra = package / "new.py"
        extra.write_text("new = True\n")
        with self.assertRaisesRegex(ValueError, "new.py"):
            runner.verify_integration_sources(hashes, root=self.root)
        extra.unlink()
        source.unlink()
        with self.assertRaisesRegex(ValueError, "speculator.py"):
            runner.verify_integration_sources(hashes, root=self.root)

    def test_worker_rejects_source_drift_before_importing_or_loading_vllm(self):
        plan = self.plan()
        path = "vendor/vllm/scripts/evaluate_native_v2_gsm8k.py"
        plan["integration_source_hashes"][path] = "wrong"
        with patch.dict(sys.modules, vllm=None), self.assertRaisesRegex(ValueError, "sources changed"):
            runner.run_worker(plan, "vanilla")
        self.assertFalse(Path(plan["output"]).exists())

    def test_prefill_shape_rpc_covers_every_width_batch_and_keeps_state_unchanged(self):
        worker = FakeWorker(speculative=True)
        spec = worker.model_runner.speculator
        spec.max_num_reqs, spec.max_num_tokens, spec.max_model_len = 2, 16, 32
        tensor = FakeTensor((5,), value=17)
        calls = []
        state = SimpleNamespace(device=SimpleNamespace(type="cuda"), dtype="bfloat16",
                                max_num_reqs=4, num_layers=3, hidden_size=8, kv_heads=2, head_dim=4,
                                mutable_state=lambda: (tensor,), update=lambda *args: calls.append(args))
        spec.state = state
        with patch.dict(sys.modules, torch=fake_torch()):
            evidence = worker.native_v2_warm_prefill_shapes(12)
        self.assertEqual(evidence["widths"], [9, 10, 11, 12])
        self.assertEqual(evidence["batches"], [1, 2])
        self.assertEqual(evidence["updates"], 8)
        self.assertTrue(evidence["state_unchanged"])
        self.assertEqual([(args[1].shape[0], args[1].shape[2]) for args in calls],
                         [(batch, width) for batch in (1, 2) for width in range(9, 13)])
        for slots, keys, values, hidden, starts, counts in calls:
            batch = keys.shape[0]
            self.assertEqual(slots.value, list(range(2, 2 + batch)))
            self.assertEqual(keys.shape, values.shape)
            self.assertEqual(hidden.shape, (batch, 3, 8))
            self.assertEqual((starts.value, counts.value), (1, 0))
            self.assertEqual(keys.dtype, "bfloat16")
        self.assertEqual(worker.model_runner.cudagraph_manager.calls, 0)
        self.assertEqual(spec.replays, 0)
        state.update = lambda *args: setattr(tensor, "value", 18)
        with patch.dict(sys.modules, torch=fake_torch()), self.assertRaisesRegex(RuntimeError, "changed LongSpark"):
            worker.native_v2_warm_prefill_shapes(9)

    def test_prefill_shape_rpc_ar_is_noop_without_torch_import(self):
        worker = FakeWorker()
        with patch.dict(sys.modules, torch=None):
            evidence = worker.native_v2_warm_prefill_shapes(16)
        self.assertEqual(evidence["status"], "not_applicable")
        self.assertEqual(evidence["updates"], 0)

    def test_native_worker_batches_requests_excludes_warmup_and_shuts_down(self):
        instances = []
        class FakeLLM:
            def __init__(this, **options):
                this.options = options
                this.worker = FakeWorker(speculative="speculative_config" in options)
                this.calls, this.closed, this.rpc_calls = [], False, []
                this.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=this.close))
                instances.append(this)
            def close(this):
                this.closed = True
            def get_tokenizer(this):
                return SimpleNamespace(eos_token_id=99)
            def get_metrics(this):
                return []
            def collective_rpc(this, name, kwargs=None):
                this.rpc_calls.append(name)
                if name == "native_v2_warm_prefill_shapes":
                    return [dict(status="completed", state_unchanged=True)]
                return [getattr(this.worker, name)(**(kwargs or {}))]
            def generate(this, prompts, params, use_tqdm):
                this.calls.append((prompts, params))
                manager = this.worker.model_runner.cudagraph_manager
                width = this.worker.model_runner.decode_query_len
                manager.run_fullgraph(manager.dispatch(len(prompts), width * len(prompts), width, 0))
                this.worker.model_runner.speculator.replays += 1
                outputs = []
                for prompt in prompts:
                    ids = prompt["prompt_token_ids"]
                    raw = raw_metrics([2])
                    outputs.append(SimpleNamespace(
                        finished=True, prompt_token_ids=ids, request_id=str(ids[0]), metrics=Timing(),
                        outputs=[SimpleNamespace(token_ids=[5, 99], text=f"#### {ids[0]}",
                                                 finish_reason="stop", stop_reason=99,
                                                 spec_decode_metrics=SimpleNamespace(to_dict=lambda: raw))]))
                return outputs
        module = ModuleType("vllm")
        module.LLM = FakeLLM
        module.SamplingParams = lambda **options: options
        plan = self.plan()
        Path(plan["output"]).mkdir()
        with patch.dict(sys.modules, vllm=module), patch.dict("os.environ"), \
                patch("importlib.metadata.version", return_value="fake"), redirect_stdout(io.StringIO()):
            runner.run_worker(plan, "longspark")
        llm = instances[0]
        self.assertTrue(llm.closed)
        self.assertEqual([len(prompts) for prompts, _ in llm.calls], [2, 1, 2, 1])
        self.assertEqual(llm.rpc_calls[:3], ["native_v2_audit_snapshot", "native_v2_warm_prefill_shapes",
                                           "native_v2_audit_start"])
        self.assertTrue(all(not param["ignore_eos"] and param["temperature"] == 0
                            for _, params in llm.calls for param in params))
        folder = Path(plan["output"]) / "longspark"
        rows = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not row["excluded_from_metrics"] for row in rows))
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(summary["performance"]["service_seconds"], 9)
        self.assertEqual(summary["performance"]["max_submitted_concurrency"], 2)
        audit = json.loads((folder / "measurement-audit.json").read_text())[0]
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["environment"]["vllm_use_flashinfer_sampler"], "0")
        self.assertEqual(manifest["integration_source_hashes"], plan["integration_source_hashes"])
        self.assertTrue(manifest["integration_sources_verified_before_load"])
        self.assertEqual(audit["target"]["replays"], 2)
        self.assertEqual(audit["target"]["max_decode_requests"], 2)
        self.assertEqual(audit["speculator_delta"]["replays"], 2)

    def test_parent_uses_distinct_sessions_and_rejects_existing_evidence(self):
        plan = self.plan()
        processes = []
        class Process:
            pid = 123456789
            stdout = iter(["worker output\n"])
            def wait(this, **kwargs):
                return 0
            def poll(this):
                return 0
        def create(command, **options):
            processes.append((command, options))
            return Process()
        Path(plan["output"]).mkdir()
        with patch.object(runner.subprocess, "Popen", side_effect=create), redirect_stdout(io.StringIO()):
            for method in runner.METHODS:
                runner.run_method_process(plan, method, self.root / "plan.json")
        self.assertEqual([command[-1] for command, _ in processes], list(runner.METHODS))
        self.assertTrue(all(options["start_new_session"] for _, options in processes))
        self.assertTrue(all(options["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "1" for _, options in processes))
        self.assertTrue(all(options["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0" for _, options in processes))
        with self.assertRaises(FileExistsError):
            runner.execute(plan)

    def test_interrupted_parent_terminates_entire_worker_process_group(self):
        plan = self.plan()
        Path(plan["output"]).mkdir()
        class Process:
            pid = 123456789
            @property
            def stdout(this):
                raise KeyboardInterrupt()
            def poll(this):
                return None
            def wait(this, **kwargs):
                return -15
        with patch.object(runner.subprocess, "Popen", return_value=Process()), \
                patch.object(runner.os, "killpg") as kill, self.assertRaises(KeyboardInterrupt):
            runner.run_method_process(plan, "longspark", self.root / "plan.json")
        kill.assert_called_once_with(Process.pid, runner.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
