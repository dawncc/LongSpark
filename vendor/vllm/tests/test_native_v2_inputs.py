from types import SimpleNamespace

import numpy as np
import pytest
import torch

from longspark_vllm.native_v2.inputs import gather_target_context


def context_case():
    # Two requests have unrelated physical page mappings and distinct prefix lengths.
    caches = [torch.arange(3 * 2 * 4 * 6).reshape(3, 2, 4, 6).float() + 1000 * i for i in range(2)]
    owner = SimpleNamespace(layer_ids=[0, 2], aux_layer_ids=[0], target_depth=3,
                            head_dim=3, _token_offsets=torch.arange(16),
                            _target_attention=[SimpleNamespace(layer_name=str(i), kv_cache=k)
                                               for i, k in enumerate(caches)])
    batch = SimpleNamespace(num_reqs=2, num_scheduled_tokens=np.array([3, 2]),
                            query_start_loc=torch.tensor([0, 3, 5]),
                            positions=torch.tensor([9, 10, 11, 20, 21]))
    slots = {str(i): torch.tensor([6, 7, 8, 10, 11]) for i in range(2)}
    final = torch.arange(20).reshape(5, 4).float() + 9000
    aux = [final - 8000]
    return owner, batch, slots, final, aux, torch.tensor([1, 1])


def test_native_paged_gather_excludes_rejected_suffix_and_uses_final_norm():
    args = context_case()
    keys, values, hidden, starts, counts, width = gather_target_context(*args)
    assert width == 3
    assert keys.shape == values.shape == (2, 2, 3, 2, 3)
    assert starts.tolist() == [9, 20]
    assert counts.tolist() == [2, 1]
    for b, (physical, boundary) in enumerate([([6, 7], 1), ([10], 3)]):
        for layer, attention in enumerate(args[0]._target_attention):
            for col, slot in enumerate(physical):
                expected = attention.kv_cache[slot // 4, :, slot % 4]
                torch.testing.assert_close(keys[b, layer, col], expected[:, :3])
                torch.testing.assert_close(values[b, layer, col], expected[:, 3:])
            assert not keys[b, layer, len(physical):].any()
            assert not values[b, layer, len(physical):].any()
        torch.testing.assert_close(hidden[b, 0], args[4][0][boundary])
        torch.testing.assert_close(hidden[b, 1], args[3][boundary])


def test_native_gather_rejects_missing_accepted_slot():
    args = list(context_case())
    args[2]['0'][0] = -1
    with pytest.raises(RuntimeError, match='resident target KV'):
        gather_target_context(*args)


def test_native_gather_masks_nonresident_rejected_slot():
    args = list(context_case())
    for slots in args[2].values():
        slots[2] = -1
        slots[4] = -1
    keys, *_ = gather_target_context(*args)
    assert torch.isfinite(keys).all()


def test_native_gather_requires_all_aux_layers():
    args = list(context_case())
    args[4] = None
    with pytest.raises(RuntimeError, match='auxiliary'):
        gather_target_context(*args)
