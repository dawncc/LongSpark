"""Fused native target KV/hidden gather; no target execution or state mutation.

The eager implementation in inputs.py remains the reference. This optional
path performs the same gather with one Triton launch, including GPU rejection
masking and boundary hidden selection. It deliberately does not read accepted
counts or positions back to the host.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_context_kernel(
    cache_ptrs, slot_ptrs, hidden_ptrs,
    query_starts, rejected, positions,
    keys_out, values_out, hidden_out, starts_out, counts_out,
    CACHE_STRIDES: tl.constexpr, CACHE_PAGES: tl.constexpr,
    PAGE_SIZES: tl.constexpr, SLOT_STRIDES: tl.constexpr,
    SLOT_ROWS: tl.constexpr, HIDDEN_STRIDES: tl.constexpr,
    QUERY_STRIDE: tl.constexpr, REJECTED_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr, POSITION_ROWS: tl.constexpr,
    HIDDEN_ROWS: tl.constexpr,
    LAYERS: tl.constexpr, WIDTH: tl.constexpr, INPUT_WIDTH: tl.constexpr, KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, HIDDEN_SIZE: tl.constexpr, BLOCK: tl.constexpr,
):
    tile, batch, layer_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    begin = tl.load(query_starts + batch * QUERY_STRIDE).to(tl.int64)
    end = tl.load(query_starts + (batch + 1) * QUERY_STRIDE).to(tl.int64)
    reject_count = tl.load(rejected + batch * REJECTED_STRIDE).to(tl.int64)
    count = end - begin - reject_count
    valid_range = ((begin >= 0) & (begin < POSITION_ROWS)
                   & (end >= begin) & (end <= POSITION_ROWS)
                   & (reject_count >= 0) & (count >= 0) & (count <= INPUT_WIDTH))
    tl.device_assert(valid_range, "Invalid native LongSpark query/rejection range")
    start = tl.load(positions + begin * POSITION_STRIDE, mask=valid_range, other=0)
    if (tile == 0) & (layer_id == 0):
        tl.store(starts_out + batch, start)
        tl.store(counts_out + batch, count)

    flat = tile * BLOCK + tl.arange(0, BLOCK)
    dimension = flat % HEAD_DIM
    head = (flat // HEAD_DIM) % KV_HEADS
    token = flat // (KV_HEADS * HEAD_DIM)
    in_output = flat < WIDTH * KV_HEADS * HEAD_DIM
    accepted = in_output & (token < count) & valid_range
    row = begin + token
    boundary = tl.minimum(tl.maximum(begin + count - 1, 0), HIDDEN_ROWS - 1)

    # Each layer has its own storage and physical slot mapping. Compile-time
    # tuple indexing preserves arbitrary strides without stacking/copying cache.
    for layer in tl.static_range(LAYERS):
        if layer_id == layer:
            slot_row_valid = row < SLOT_ROWS[layer]
            tl.device_assert((~accepted) | slot_row_valid,
                             "Accepted LongSpark context has no slot-mapping row")
            slot = tl.load(slot_ptrs[layer] + row * SLOT_STRIDES[layer],
                           mask=accepted & slot_row_valid, other=-1).to(tl.int64)
            resident = ((slot >= 0) & (slot < CACHE_PAGES[layer] * PAGE_SIZES[layer]))
            tl.device_assert((~accepted) | resident,
                             "Accepted LongSpark context has no resident target KV slot")
            page = slot // PAGE_SIZES[layer]
            page_offset = slot % PAGE_SIZES[layer]
            offset = (page * CACHE_STRIDES[layer][0]
                      + head * CACHE_STRIDES[layer][1]
                      + page_offset * CACHE_STRIDES[layer][2]
                      + dimension * CACHE_STRIDES[layer][3])
            key = tl.load(cache_ptrs[layer] + offset,
                          mask=accepted & resident, other=0)
            value = tl.load(cache_ptrs[layer] + offset
                            + HEAD_DIM * CACHE_STRIDES[layer][3],
                            mask=accepted & resident, other=0)
            destination = ((batch * LAYERS + layer) * WIDTH * KV_HEADS * HEAD_DIM + flat)
            tl.store(keys_out + destination, key, mask=in_output)
            tl.store(values_out + destination, value, mask=in_output)
            # Extra tiles are launched when hidden width exceeds the KV width.
            hidden = tl.load(hidden_ptrs[layer]
                             + boundary * HIDDEN_STRIDES[layer][0]
                             + flat * HIDDEN_STRIDES[layer][1],
                             mask=flat < HIDDEN_SIZE, other=0)
            tl.store(hidden_out + (batch * LAYERS + layer) * HIDDEN_SIZE + flat,
                     hidden, mask=flat < HIDDEN_SIZE)


def fused_gather_target_context(owner, input_batch, slot_mappings,
                                last_hidden_states, aux_hidden_states, num_rejected, *, output_buffers=None):
    """Return ``keys, values, hidden, starts, counts, width`` as in inputs.py.

    Without output_buffers, outputs own their storage. Optional validated decode
    buffers write directly into the fixed-width CUDA-graph inputs. All input
    cache/hidden/index strides are passed to the
    kernel; no contiguous copy or CPU accepted-length synchronization occurs.
    ``width`` alone comes from the native scheduler's CPU upper bound.
    Device assertions are explicitly enabled, independent of TRITON_DEBUG.
    """
    batch = input_batch.num_reqs
    width = int(input_batch.num_scheduled_tokens.max())
    if batch < 1 or width < 1:
        raise ValueError("Native LongSpark gather requires a nonempty scheduled batch")
    if aux_hidden_states is None or len(aux_hidden_states) != len(owner.aux_layer_ids):
        raise RuntimeError("Native target did not provide the requested LongSpark auxiliary layers")
    selected = dict(zip(owner.aux_layer_ids, aux_hidden_states))
    selected[owner.target_depth - 1] = last_hidden_states
    hidden_inputs = tuple(selected[layer] for layer in owner.layer_ids)
    caches = tuple(attention.kv_cache for attention in owner._target_attention)
    slots = tuple(slot_mappings[attention.layer_name] for attention in owner._target_attention)
    if not caches or len(caches) != len(hidden_inputs):
        raise ValueError("Selected target layers and KV caches do not align")
    if any(cache.ndim != 4 or cache.shape[-1] != 2 * owner.head_dim for cache in caches):
        raise ValueError("Native LongSpark requires FlashAttention packed KV layout")
    device, dtype = caches[0].device, caches[0].dtype
    kv_heads, hidden_size = caches[0].shape[1], last_hidden_states.shape[1]
    if device.type != "cuda":
        raise ValueError("Fused native LongSpark gather requires CUDA")
    if any(cache.device != device or cache.dtype != dtype or cache.shape[1] != kv_heads
           or cache.shape[0] < 1 or cache.shape[2] < 1 for cache in caches):
        raise ValueError("Selected native KV caches must have matching heads, dtype and device")
    if any(hidden.device != device or hidden.dtype != last_hidden_states.dtype
           or hidden.ndim != 2 or hidden.shape[1] != hidden_size
           or hidden.shape[0] < last_hidden_states.shape[0] for hidden in hidden_inputs):
        raise ValueError("Selected native hidden states must have matching rows, dtype and device")
    indexes = (input_batch.query_start_loc, input_batch.positions, num_rejected, *slots)
    if any(index.device != device or index.ndim != 1
           or index.dtype not in (torch.int32, torch.int64) for index in indexes):
        raise ValueError("Native gather indexes must be CUDA int32/int64 vectors")
    if (input_batch.query_start_loc.numel() < batch + 1 or num_rejected.numel() < batch
            or input_batch.positions.numel() < 1 or last_hidden_states.shape[0] < 1):
        raise ValueError("Native gather index buffers are too short")
    layers = len(caches)
    output_width = width
    if output_buffers is None:
        shape = (batch, layers, width, kv_heads, owner.head_dim)
        keys = torch.empty(shape, dtype=dtype, device=device)
        values = torch.empty_like(keys)
        hidden = torch.empty((batch, layers, hidden_size),
                             dtype=last_hidden_states.dtype, device=device)
        starts = torch.empty(batch, dtype=torch.int64, device=device)
        counts = torch.empty_like(starts)
    else:
        if width > 8 or len(output_buffers) != 5:
            raise ValueError("Direct LongSpark gather requires five fixed-width decode buffers")
        keys, values, hidden, starts, counts = output_buffers
        output_width = 8
        shapes = ((batch, layers, 8, kv_heads, owner.head_dim),) * 2 + (
            (batch, layers, hidden_size), (batch,), (batch,))
        dtypes = (dtype, dtype, last_hidden_states.dtype, torch.int64, torch.int64)
        if any(t.shape != shape or t.dtype != expected_dtype or t.device != device
               or not t.is_contiguous()
               for t, shape, expected_dtype in zip(output_buffers, shapes, dtypes)):
            raise ValueError("Direct LongSpark gather buffer shape, dtype, device or layout differs")
    block = 256
    tiles = triton.cdiv(max(output_width * kv_heads * owner.head_dim, hidden_size), block)
    _gather_context_kernel[(tiles, batch, layers)](
        caches, slots, hidden_inputs,
        input_batch.query_start_loc, num_rejected, input_batch.positions,
        keys, values, hidden, starts, counts,
        CACHE_STRIDES=tuple(cache.stride() for cache in caches),
        CACHE_PAGES=tuple(cache.shape[0] for cache in caches),
        PAGE_SIZES=tuple(cache.shape[2] for cache in caches),
        SLOT_STRIDES=tuple(slot.stride(0) for slot in slots),
        SLOT_ROWS=tuple(slot.numel() for slot in slots),
        HIDDEN_STRIDES=tuple(state.stride() for state in hidden_inputs),
        QUERY_STRIDE=input_batch.query_start_loc.stride(0),
        REJECTED_STRIDE=num_rejected.stride(0),
        POSITION_STRIDE=input_batch.positions.stride(0),
        POSITION_ROWS=input_batch.positions.numel(),
        HIDDEN_ROWS=last_hidden_states.shape[0], LAYERS=layers, WIDTH=output_width, INPUT_WIDTH=width,
        KV_HEADS=kv_heads, HEAD_DIM=owner.head_dim, HIDDEN_SIZE=hidden_size,
        BLOCK=block, num_warps=4, debug=True,
    )
    return keys, values, hidden, starts, counts, width
