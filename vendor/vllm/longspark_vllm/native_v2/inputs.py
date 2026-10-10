"""GPU-only gather from native ModelRunner V2 input and paged-KV buffers."""
import torch


def gather_target_context(owner, input_batch, slot_mappings, last_hidden_states, aux_hidden_states, num_rejected):
    """Gather only from native buffers; accepted counts never leave the GPU."""
    batch = input_batch.num_reqs
    width = int(input_batch.num_scheduled_tokens.max())
    # Upper-bound query width is scheduler-owned CPU metadata. The actual
    # number accepted remains GPU-resident, including under async scheduling.
    begin = input_batch.query_start_loc[:batch].long()
    counts = input_batch.query_start_loc[1:batch + 1].long() - begin - num_rejected.long()
    offsets = owner._token_offsets[:width]
    rows = begin[:, None] + offsets[None, :]
    safe_rows = rows.clamp(0, input_batch.positions.shape[0] - 1)
    valid = offsets[None, :] < counts[:, None]
    starts = input_batch.positions[begin]
    boundary = (begin + counts - 1).clamp(0, last_hidden_states.shape[0] - 1)
    if aux_hidden_states is None or len(aux_hidden_states) != len(owner.aux_layer_ids):
        raise RuntimeError("Native target did not provide the requested LongSpark auxiliary layers")
    selected = dict(zip(owner.aux_layer_ids, aux_hidden_states))
    selected[owner.target_depth - 1] = last_hidden_states
    hidden = torch.stack([selected[i][boundary] for i in owner.layer_ids], dim=1)
    key_parts, value_parts = [], []
    for attention in owner._target_attention:
        cache = attention.kv_cache
        if cache.ndim != 4 or cache.shape[-1] != 2 * owner.head_dim:
            raise ValueError("Native LongSpark requires FlashAttention packed KV layout")
        slots = slot_mappings[attention.layer_name][safe_rows].long()
        torch._assert_async(((~valid) | (slots >= 0)).all(),
                            "Accepted LongSpark context has no resident target KV slot")
        resident = valid & (slots >= 0)
        safe_slots = slots.clamp_min(0)
        page_size = cache.shape[2]
        packed = cache[safe_slots // page_size, :, safe_slots % page_size, :]
        packed = packed.masked_fill(~resident[..., None, None], 0)
        keys, values = packed.split(owner.head_dim, dim=-1)
        key_parts.append(keys)
        value_parts.append(values)
    return (torch.stack(key_parts, dim=1), torch.stack(value_parts, dim=1),
            hidden, starts, counts, width)

