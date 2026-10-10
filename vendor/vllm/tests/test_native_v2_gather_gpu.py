"""Opt-in CUDA equivalence checks for the standalone fused native KV gather."""

import os
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("LONGSPARK_TEST_CUDA") != "1",
                                reason="Set LONGSPARK_TEST_CUDA=1 on an idle GPU")


def make_case(widths=(8, 3), rejected=(5, 1), *, strided=False, dim=8, heads=3, hidden=73):
    from longspark_vllm.native_v2.inputs import gather_target_context
    from longspark_vllm.native_v2.gather_kernel import fused_gather_target_context

    torch.manual_seed(934)
    batch, rows = len(widths), sum(widths) + 3
    device = "cuda"
    step = 2 if strided else 1

    def vector(values):
        tensor = torch.empty(len(values) * step, dtype=torch.int64, device=device)[::step]
        tensor.copy_(torch.tensor(values, dtype=torch.int64, device=device))
        return tensor

    starts = [0, *np.cumsum(widths).tolist()]
    positions = vector([50 * b + 13 + j for b, count in enumerate(widths)
                        for j in range(count)] + [0, 0, 0])
    caches, slots, hidden_states = [], {}, []
    for layer in range(5):
        page_size = (4, 8, 16, 4, 8)[layer]
        pages = (rows + page_size - 1) // page_size + 3
        cache = torch.randn(pages * step, heads * step, page_size * step, 2 * dim * step,
                            device=device, dtype=torch.bfloat16)[::step, ::step, ::step, ::step]
        mapping = torch.randperm(pages * page_size, device=device)[:rows].to(torch.int64)
        for b, reject in enumerate(rejected):
            for row in range(starts[b + 1] - reject, starts[b + 1]):
                slot = int(mapping[row].item())  # Fixture setup only.
                cache[slot // page_size, :, slot % page_size, :] = float("nan")
                if row % 2:
                    mapping[row] = -1
        slot_tensor = torch.empty(rows * step, dtype=torch.int64, device=device)[::step]
        slot_tensor.copy_(mapping)
        slots[str(layer)] = slot_tensor
        caches.append(NS(layer_name=str(layer), kv_cache=cache))
        states = torch.randn(rows * step, hidden * step, device=device,
                             dtype=torch.bfloat16)[::step, ::step] + layer
        if strided:
            backing = torch.empty(rows * 2, hidden * 2, dtype=states.dtype, device=device)
            view = backing[::2, ::2]
            view.copy_(states)
            states = view
        hidden_states.append(states)
    owner = NS(layer_ids=[0, 2, 4, 6, 8], aux_layer_ids=[0, 2, 4, 6], target_depth=9,
               head_dim=dim, _target_attention=caches, _token_offsets=torch.arange(max(widths), device=device))
    input_batch = NS(num_reqs=batch, num_scheduled_tokens=np.asarray(widths),
                     query_start_loc=vector(starts), positions=positions)
    args = (owner, input_batch, slots, hidden_states[-1], hidden_states[:-1], vector(rejected))
    return args, gather_target_context, fused_gather_target_context


@pytest.mark.parametrize("widths,rejected,strided", [
    ((1,), (0,), True),
    ((8,), (0,), False),
    ((8, 3), (5, 1), True),
    ((17, 33, 1), (2, 0, 0), True),
    ((512, 8), (7, 5), False),
    ((1, 2), (1, 2), True),
])
def test_fused_matches_reference_for_pages_rejection_masks_and_strides(widths, rejected, strided):
    args, reference, fused = make_case(widths, rejected, strided=strided)
    expected, actual = reference(*args), fused(*args)
    torch.cuda.synchronize()
    assert expected[-1] == actual[-1]
    for old, new in zip(expected[:-1], actual[:-1]):
        torch.testing.assert_close(new, old, atol=0, rtol=0, equal_nan=True)
    assert torch.isfinite(actual[0]).all() and torch.isfinite(actual[1]).all()


@pytest.mark.parametrize("width", [1, 8])
def test_real_qwen_geometry_and_outputs_own_storage(width):
    # Keep all underlying cache rows resident; only vary which ones are accepted.
    args, reference, fused = make_case((width,), (0,), dim=128, heads=8, hidden=2560)
    args[-1].fill_(min(3, width - 1))
    first = fused(*args)
    saved = [tensor.clone() for tensor in first[:-1]]
    args[-1].fill_(0)
    second = fused(*args)
    torch.cuda.synchronize()
    for before, tensor in zip(saved, first[:-1]):
        torch.testing.assert_close(tensor, before, atol=0, rtol=0, equal_nan=True)
    expected = reference(*args)
    for old, new in zip(expected[:-1], second[:-1]):
        torch.testing.assert_close(new, old, atol=0, rtol=0, equal_nan=True)


def test_kernel_can_capture_and_replay_dynamic_gpu_rejection_counts():
    args, reference, fused = make_case((8, 3), (0, 0), strided=True)
    # Compile/warm on a side stream before capture, as required by torch.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fused(*args)
    args[-1].copy_(torch.tensor([6, 1], dtype=torch.int64, device="cuda"))
    graph.replay()
    expected = reference(*args)
    torch.cuda.synchronize()
    for old, new in zip(expected[:-1], output[:-1]):
        torch.testing.assert_close(new, old, atol=0, rtol=0, equal_nan=True)


def test_device_slot_assert_is_enabled_without_triton_debug():
    # A failed device assertion poisons its CUDA context; isolate it from the
    # remaining tests, and prove debug=True is effective without an env flag.
    script = """
import torch
from test_native_v2_gather_gpu import make_case
args, _, fused = make_case((8,), (0,))
args[2]['0'][0] = -1
fused(*args)
torch.cuda.synchronize()
"""
    env = dict(os.environ, TRITON_DEBUG="0")
    env["PYTHONPATH"] = os.pathsep.join([str(__import__("pathlib").Path(__file__).parent),
                                       env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "resident target KV slot" in result.stderr or "device-side assert" in result.stderr
