"""Exact fused state stores after the unchanged reference attention arithmetic.

The kernel only selects and copies values. It never computes attention or
changes accumulator precision. Zero-count and padding rows issue no stores.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["KEY_STRIDES", "VALUE_STRIDES", "HIDDEN_STRIDES", "KEEP"])
def _commit_state_kernel(
    slots, starts, counts, keys, values, hidden, output, lse,
    state_keys, state_values, state_hidden, state_output, state_lse, state_lengths,
    KEY_STRIDES, VALUE_STRIDES, HIDDEN_STRIDES, OUTPUT_STRIDES: tl.constexpr,
    LSE_STRIDES: tl.constexpr, SLOT_STRIDE: tl.constexpr,
    START_STRIDE: tl.constexpr, COUNT_STRIDE: tl.constexpr,
    LAYERS: tl.constexpr, KEEP,
    KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, HIDDEN_SIZE: tl.constexpr,
    QUERY_HEADS: tl.constexpr, BOUNDARY_HIDDEN: tl.constexpr, BLOCK: tl.constexpr,
):
    tile, batch, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(slots + batch * SLOT_STRIDE).to(tl.int64)
    count = tl.load(counts + batch * COUNT_STRIDE).to(tl.int64)
    start = tl.load(starts + batch * START_STRIDE).to(tl.int64)
    changed = count > 0
    flat = tile * BLOCK + tl.arange(0, BLOCK)

    # Each live row has a unique physical ring destination, even for T > 256.
    row = flat // (KV_HEADS * HEAD_DIM)
    head = (flat // HEAD_DIM) % KV_HEADS
    dim = flat % HEAD_DIM
    source_row = tl.maximum(count - KEEP, 0) + row
    valid_raw = changed & (row < KEEP) & (source_row < count)
    destination_row = (start + source_row) % 256
    source_key = (batch * KEY_STRIDES[0] + layer * KEY_STRIDES[1]
                  + source_row * KEY_STRIDES[2] + head * KEY_STRIDES[3]
                  + dim * KEY_STRIDES[4])
    source_value = (batch * VALUE_STRIDES[0] + layer * VALUE_STRIDES[1]
                    + source_row * VALUE_STRIDES[2] + head * VALUE_STRIDES[3]
                    + dim * VALUE_STRIDES[4])
    destination = (((slot * LAYERS + layer) * 256 + destination_row)
                   * KV_HEADS + head) * HEAD_DIM + dim
    key = tl.load(keys + source_key, mask=valid_raw, other=0)
    value = tl.load(values + source_value, mask=valid_raw, other=0)
    tl.store(state_keys + destination, key, mask=valid_raw)
    tl.store(state_values + destination, value, mask=valid_raw)

    valid_hidden = changed & (flat < HIDDEN_SIZE)
    if BOUNDARY_HIDDEN:
        hidden_offset = (batch * HIDDEN_STRIDES[0] + layer * HIDDEN_STRIDES[1]
                         + flat * HIDDEN_STRIDES[2])
    else:
        hidden_offset = (batch * HIDDEN_STRIDES[0] + (count - 1) * HIDDEN_STRIDES[1]
                         + layer * HIDDEN_STRIDES[2] + flat * HIDDEN_STRIDES[3])
    boundary = tl.load(hidden + hidden_offset, mask=valid_hidden, other=0)
    tl.store(state_hidden + (slot * LAYERS + layer) * HIDDEN_SIZE + flat,
             boundary, mask=valid_hidden)

    # These are already-normalized FP32 results of the original PyTorch math.
    query = (flat // HEAD_DIM) % 16
    query_head = flat // (16 * HEAD_DIM)
    valid_output = changed & (flat < QUERY_HEADS * 16 * HEAD_DIM)
    output_offset = (batch * OUTPUT_STRIDES[0] + layer * OUTPUT_STRIDES[1]
                     + query_head * OUTPUT_STRIDES[2] + query * OUTPUT_STRIDES[3]
                     + dim * OUTPUT_STRIDES[4])
    merged = tl.load(output + output_offset, mask=valid_output, other=0)
    tl.store(state_output + (slot * LAYERS + layer) * QUERY_HEADS * 16 * HEAD_DIM + flat,
             merged, mask=valid_output)
    valid_lse = changed & (flat < QUERY_HEADS * 16)
    lse_offset = (batch * LSE_STRIDES[0] + layer * LSE_STRIDES[1]
                  + (flat // 16) * LSE_STRIDES[2] + (flat % 16) * LSE_STRIDES[3])
    merged_lse = tl.load(lse + lse_offset, mask=valid_lse, other=0)
    tl.store(state_lse + (slot * LAYERS + layer) * QUERY_HEADS * 16 + flat,
             merged_lse, mask=valid_lse)
    if (tile == 0) & (layer == 0):
        tl.store(state_lengths + slot, start + count, mask=changed)


def fused_commit_state(state, slots, keys, values, hidden, starts, counts, output, lse):
    """Commit validated inputs; mutable state buffers are contiguous by construction.

    Prefill width-dependent strides and KEEP stay runtime scalars, so unseen
    prompt lengths reuse the same compiled kernel instead of triggering JIT.
    """
    tokens = keys.shape[2]
    keep = min(tokens, 256)
    query_heads = output.shape[2]
    block = 256
    extent = max(keep * state.kv_heads * state.head_dim,
                 state.hidden_size, query_heads * 16 * state.head_dim)
    _commit_state_kernel[(triton.cdiv(extent, block), slots.shape[0], state.num_layers)](
        slots, starts, counts, keys, values, hidden, output, lse,
        state.keys, state.values, state.hidden, state.global_output,
        state.global_lse, state.lengths,
        KEY_STRIDES=keys.stride(), VALUE_STRIDES=values.stride(),
        HIDDEN_STRIDES=hidden.stride(), OUTPUT_STRIDES=output.stride(),
        LSE_STRIDES=lse.stride(), SLOT_STRIDE=slots.stride(0),
        START_STRIDE=starts.stride(0), COUNT_STRIDE=counts.stride(0),
        LAYERS=state.num_layers, KEEP=keep,
        KV_HEADS=state.kv_heads, HEAD_DIM=state.head_dim,
        HIDDEN_SIZE=state.hidden_size, QUERY_HEADS=query_heads,
        BOUNDARY_HIDDEN=hidden.ndim == 3, BLOCK=block, num_warps=4,
    )
