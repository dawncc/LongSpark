"""Worker-side evidence for native V2 graph execution, with no CUDA imports."""

from __future__ import annotations

from collections import Counter
from functools import wraps
import time


def _qualified_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _snapshot(worker):
    runner = worker.model_runner
    manager = runner.cudagraph_manager
    if manager is None:
        raise RuntimeError("Native target CUDA graph manager is missing")
    speculator = getattr(runner, "speculator", None)
    return dict(
        runner_class=_qualified_name(runner),
        use_v2_model_runner=worker.vllm_config.use_v2_model_runner,
        async_scheduling=worker.vllm_config.scheduler_config.async_scheduling,
        target=dict(manager_class=_qualified_name(manager), captures=len(manager.graphs),
                    captured_shapes=[repr(desc) for desc in manager.graphs]),
        speculator_class=_qualified_name(speculator) if speculator is not None else None,
        speculator=speculator.runtime_stats() if speculator is not None else None,
    )


class NativeV2Diagnostics:
    """Mixed into vLLM's worker through ``worker_extension_cls``."""

    def native_v2_audit_snapshot(self):
        return _snapshot(self)

    def native_v2_warm_prefill_shapes(self, max_width):
        """Compile state-update shapes on private slots before request timing."""
        if getattr(self, "_longspark_native_audit", None) is not None:
            raise RuntimeError("Prefill shape warmup cannot run inside a request audit")
        speculator = getattr(self.model_runner, "speculator", None)
        if speculator is None:
            return dict(status="not_applicable", reason="AR has no LongSpark state",
                        widths=[], batches=[], updates=0, state_unchanged=True, wall_seconds=0.0)
        if type(max_width) is not int or not 1 <= max_width <= min(
                speculator.max_num_tokens, speculator.max_model_len):
            raise ValueError("Prefill warmup width must fit the native token and context limits")
        if speculator.runtime_stats().get("startup_complete") is not True:
            raise RuntimeError("Prefill shape warmup requires completed graph startup")
        import torch

        state = speculator.state
        capacity = speculator.max_num_reqs
        if state.device.type != "cuda" or state.dtype != torch.bfloat16:
            raise RuntimeError("Prefill shape warmup requires CUDA BF16 state")
        if state.max_num_reqs < 2 * capacity:
            raise RuntimeError("Prefill shape warmup requires separate positive padding slots")
        widths, batches = list(range(9, max_width + 1)), list(range(1, capacity + 1))
        updates = 0
        with torch.inference_mode():
            before = [tensor.clone() for tensor in state.mutable_state()]
            torch.cuda.synchronize(state.device)
            started = time.perf_counter()
            for batch in batches:
                slots = torch.arange(capacity, capacity + batch, dtype=torch.long, device=state.device)
                starts = torch.ones(batch, dtype=torch.long, device=state.device)
                counts = torch.zeros(batch, dtype=torch.long, device=state.device)
                hidden = torch.zeros((batch, state.num_layers, state.hidden_size),
                                     dtype=state.dtype, device=state.device)
                for width in widths:
                    # A sliced maximum-width buffer would retain different strides
                    # than real contiguous gathered K/V and miss JIT signatures.
                    shape = (batch, state.num_layers, width, state.kv_heads, state.head_dim)
                    keys = torch.zeros(shape, dtype=state.dtype, device=state.device)
                    values = torch.zeros_like(keys)
                    state.update(slots, keys, values, hidden, starts, counts)
                    updates += 1
            torch.cuda.synchronize(state.device)
            elapsed = time.perf_counter() - started
            after = state.mutable_state()
            unchanged = len(before) == len(after) and all(
                torch.equal(old.view(torch.uint8), new.view(torch.uint8))
                for old, new in zip(before, after))
        if not unchanged:
            raise RuntimeError("Prefill shape warmup changed LongSpark mutable state")
        return dict(status="completed", max_width=max_width, widths=widths, batches=batches,
                    updates=updates, state_unchanged=True, comparison="bitwise torch.equal(uint8 views)",
                    wall_seconds=elapsed, dtype=str(state.dtype), device=str(state.device),
                    slot_policy="private positive slots: max_num_reqs + arange(batch)",
                    input_layout="contiguous [B,L,width,KVH,D]; boundary hidden [B,L,H]",
                    start_position=1, valid_count=0)

    def native_v2_audit_start(self, label="measurement"):
        if getattr(self, "_longspark_native_audit", None) is not None:
            raise RuntimeError("An audit is already active; end it before starting another")
        before = _snapshot(self)
        if before["runner_class"] != "vllm.v1.worker.gpu.model_runner.GPUModelRunner":
            raise RuntimeError(f"Expected native V2 GPUModelRunner, got {before['runner_class']}")
        if not before["use_v2_model_runner"] or not before["async_scheduling"]:
            raise RuntimeError("Native evaluation requires V2 with async scheduling")
        runner = self.model_runner
        manager = runner.cudagraph_manager
        original_replay = manager.run_fullgraph
        original_dispatch = manager.dispatch
        state = dict(label=label, before=before, manager=manager,
                     original_replay=original_replay, original_dispatch=original_dispatch,
                     replays=0, replay_shapes=Counter(), dispatches=Counter(),
                     max_dispatched_requests=0, max_decode_requests=0)

        @wraps(original_replay)
        def audited_replay(desc):
            result = original_replay(desc)
            # Count only after the original method successfully submitted its replay.
            state["replays"] += 1
            state["replay_shapes"][repr(desc)] += 1
            return result

        @wraps(original_dispatch)
        def audited_dispatch(*args, **kwargs):
            desc = original_dispatch(*args, **kwargs)
            uniform = kwargs.get("uniform_token_count", args[2] if len(args) > 2 else None)
            requests = kwargs.get("num_reqs", args[0] if args else 0)
            kind = "uniform_decode" if uniform == runner.decode_query_len else "other"
            mode = getattr(desc.cg_mode, "name", str(desc.cg_mode))
            state["dispatches"][f"{kind}:{mode}"] += 1
            state["max_dispatched_requests"] = max(state["max_dispatched_requests"], requests)
            if kind == "uniform_decode":
                state["max_decode_requests"] = max(state["max_decode_requests"], requests)
            return desc

        manager.run_fullgraph = audited_replay
        manager.dispatch = audited_dispatch
        self._longspark_native_audit = state
        return dict(label=label, before=before)

    def native_v2_audit_end(self):
        state = getattr(self, "_longspark_native_audit", None)
        if state is None:
            raise RuntimeError("No native graph audit is active")
        manager = state["manager"]
        manager.run_fullgraph = state["original_replay"]
        manager.dispatch = state["original_dispatch"]
        self._longspark_native_audit = None
        after = _snapshot(self)
        before = state["before"]
        spec_before, spec_after = before["speculator"], after["speculator"]
        spec_delta = None
        if spec_before is not None and spec_after is not None:
            spec_delta = {
                key: spec_after[key] - spec_before[key]
                for key in ("captures", "replays", "fallbacks")
                if isinstance(spec_before.get(key), int) and isinstance(spec_after.get(key), int)
            }
        return dict(
            label=state["label"], before=before, after=after,
            target=dict(replays=state["replays"], replay_shapes=dict(state["replay_shapes"]),
                        dispatches=dict(state["dispatches"]),
                        max_dispatched_requests=state["max_dispatched_requests"],
                        max_decode_requests=state["max_decode_requests"],
                        captures_delta=after["target"]["captures"] - before["target"]["captures"],
                        manager_replaced=self.model_runner.cudagraph_manager is not manager),
            speculator_delta=spec_delta,
        )
