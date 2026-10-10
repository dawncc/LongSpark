"""Exact state equivalence for fused stores, including real Qwen3-4B geometry."""

import copy
import os

import pytest
import torch

from longspark_vllm.native_v2.state import LongSparkState

pytestmark = pytest.mark.skipif(os.environ.get("LONGSPARK_TEST_CUDA") != "1",
                                reason="Set LONGSPARK_TEST_CUDA=1 on an idle GPU")


def make_state():
    # No target/draft weights are needed to validate the committed state math.
    state = LongSparkState.__new__(LongSparkState)
    state.device = torch.device("cuda:0")
    state.dtype = torch.bfloat16
    state.max_num_reqs, state.num_layers = 6, 5
    state.kv_heads, state.head_dim, state.hidden_size = 8, 128, 2560
    state.query = torch.randn(1, 5, 32, 16, 128, device="cuda", dtype=torch.bfloat16)
    state.keys = torch.randn(7, 5, 256, 8, 128, device="cuda", dtype=torch.bfloat16)
    state.values = torch.randn_like(state.keys)
    state.global_output = torch.randn(7, 5, 32, 16, 128, device="cuda")
    state.global_lse = torch.randn(7, 5, 32, 16, device="cuda")
    state.hidden = torch.randn(7, 5, 2560, device="cuda", dtype=torch.bfloat16)
    state.lengths = torch.zeros(7, device="cuda", dtype=torch.long)
    state.raw_offsets = torch.arange(256, device="cuda")
    state.layer_offsets = torch.arange(5, device="cuda")
    return state


def assert_bitwise(reference, actual):
    for old, new in zip(reference.mutable_state(), actual.mutable_state()):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    return graph


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("width", [8, 512])
@pytest.mark.parametrize("boundary,strided", [(True, False), (False, True)])
def test_full_update_is_bitwise_equal_to_reference(batch, width, boundary, strided):
    torch.manual_seed(325)
    reference = make_state()
    fast = copy.deepcopy(reference)
    reference.use_fused_commit = False
    step = 2 if strided else 1
    keys = torch.randn(batch, 5, width, 8, 128 * step,
                       device="cuda", dtype=torch.bfloat16)[..., ::step]
    values = torch.randn_like(keys)
    hshape = (batch, 5, 2560 * step) if boundary else (batch, width, 5, 2560 * step)
    hidden = torch.randn(hshape, device="cuda", dtype=torch.bfloat16)[..., ::step]
    slots = torch.tensor([2, 0, 4, 1][:batch], device="cuda")
    starts = torch.zeros(batch, dtype=torch.long, device="cuda")
    counts = torch.tensor([width, min(3, width), 0, 1][:batch], device="cuda")
    calls = [lambda s=s: s.update(slots, keys, values, hidden, starts, counts)
             for s in (reference, fast)]
    with torch.inference_mode():
        for call in calls:
            call()
        assert_bitwise(reference, fast)
        # Extend, wrap Raw256, reorder slots, and reset reused request state.
        for iteration in range(4):
            accepted = [3, 0, 7, 1][:batch]
            for row, count in enumerate(accepted):
                keys[row, :, count:] = float("nan")
                values[row, :, count:] = float("nan")
            counts.copy_(torch.tensor(accepted, device="cuda"))
            starts.copy_(reference.lengths.index_select(0, slots))
            if iteration == 2:
                slots.copy_(slots.flip(0))
                starts.zero_()
            for call in calls:
                call()
            assert_bitwise(reference, fast)
        snapshot = tuple(t.clone() for t in fast.mutable_state())
        counts.zero_()
        starts.fill_(987)
        for call in calls:
            call()
        assert_bitwise(reference, fast)
        for before, after in zip(snapshot, fast.mutable_state()):
            assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
        # Dynamic accepted counts/slot ordering remain device inputs to one graph.
        starts.zero_()
        counts.fill_(1)
        graphs = [capture(call) for call in calls]
        for iteration in range(3):
            starts.copy_(reference.lengths.index_select(0, slots))
            if iteration == 1:
                slots.copy_(slots.flip(0))
                starts.zero_()
            if iteration == 2:
                slots.fill_(-1)  # multiple padding rows must issue no stores
                counts.fill_(width)
                starts.fill_(777)
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert_bitwise(reference, fast)


def test_unseen_prefill_widths_reuse_the_same_compiled_store_kernel():
    from longspark_vllm.native_v2.commit_kernel import fused_commit_state, _commit_state_kernel

    state = make_state()
    slots = torch.tensor([0], device="cuda")
    starts = torch.ones(1, device="cuda", dtype=torch.long)
    counts = torch.zeros_like(starts)
    hidden = torch.zeros(1, 5, 2560, device="cuda", dtype=torch.bfloat16)
    output = torch.zeros(1, 5, 32, 16, 128, device="cuda")
    lse = torch.zeros(1, 5, 32, 16, device="cuda")
    variants = []
    for width in (8, 9, 10, 11, 17, 31, 63, 127, 190, 255, 256, 257, 512, 509):
        keys = torch.zeros(1, 5, width, 8, 128, device="cuda", dtype=torch.bfloat16)
        fused_commit_state(state, slots, keys, torch.zeros_like(keys), hidden,
                           starts, counts, output, lse)
        variants.append(len(_commit_state_kernel.device_caches[torch.cuda.current_device()][0]))
    assert len(set(variants)) == 1
