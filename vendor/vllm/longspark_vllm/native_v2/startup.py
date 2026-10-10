"""CPU-only startup checks for native graph coverage and prefill scratch."""

from __future__ import annotations


def require_graph_coverage(manager, max_num_reqs, *, full_mode, query_len=7):
    """Fail at startup if any supported request count would run eagerly."""
    if type(max_num_reqs) is not int or max_num_reqs < 1:
        raise ValueError("max_num_reqs must be a positive integer")
    missing = []
    for batch in range(1, max_num_reqs + 1):
        desc = manager.dispatch(batch, batch * query_len,
                                uniform_token_count=query_len, num_active_loras=0)
        if desc.cg_mode != full_mode or desc not in manager.graphs:
            missing.append(batch)
    if not manager.graphs or missing:
        sizes = manager.compilation_config.cudagraph_capture_sizes
        raise RuntimeError(
            "Native LongSpark CUDA graph capture has incomplete batch coverage: "
            f"missing_batches={missing}, captures={len(manager.graphs)}, "
            f"cudagraph_capture_sizes={sizes}, draft_query_len={query_len}. "
            "Include capture sizes that cover every request count after rounding "
            "to the draft query length; eager fallback is forbidden."
        )
    return list(range(1, max_num_reqs + 1))


def prefill_workspace_budget(*, max_num_reqs, max_num_tokens, max_model_len,
                             num_layers, kv_heads, query_heads, head_dim,
                             hidden_size):
    """Conservative BF16 gather/update scratch, excluding persistent state.

    The gathered rectangle is B * max(query_length), not the total scheduled
    tokens. With T scheduled tokens and nonempty requests, its largest width
    is min(max_model_len, T-B+1). Check every legal B, including unbalanced
    mixed prefill/decode batches.

    Tensor liveness follows inputs.gather_target_context, LongSparkState.update,
    and SpecForge's _attention_statistics / merge_normalized_attention:
      * Gather: packed K/V views plus both stacks (4 single-KV BF16 buffers),
        boundary hidden list/stack, and row/slot/index/mask intermediates.
      * Update: input K/V plus sanitized K/V (4), four FP32 score-sized
        intermediates, six FP32 Global16 output-sized intermediates during
        normalization/merge, three Raw256 gather/where temporaries, four
        boundary hidden buffers and scalar/index/mask scratch.
    Some of these peaks are sequential; retaining their sum is conservative.
    A separately reported 32 MiB margin covers allocator rounding and library
    scratch. The real draft network is profiled while this reservation is live.
    This is a startup budget for this implementation, not a general guarantee
    for arbitrary attention kernels or external allocations.
    """
    dimensions = dict(max_num_reqs=max_num_reqs, max_num_tokens=max_num_tokens,
                      max_model_len=max_model_len, num_layers=num_layers,
                      kv_heads=kv_heads, query_heads=query_heads,
                      head_dim=head_dim, hidden_size=hidden_size)
    if any(type(value) is not int or value <= 0 for value in dimensions.values()):
        raise ValueError("Prefill budget dimensions must be positive integers")
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    candidates = []
    for batch in range(1, min(max_num_reqs, max_num_tokens) + 1):
        width = min(max_model_len, max_num_tokens - batch + 1)
        kv = batch * num_layers * width * kv_heads * head_dim * 2
        hidden = batch * num_layers * hidden_size * 2
        score = batch * num_layers * query_heads * 16 * width * 4
        global_output = batch * num_layers * query_heads * 16 * head_dim * 4
        scalar = batch * num_layers * query_heads * 16 * 4
        raw = batch * num_layers * min(width, 256) * kv_heads * head_dim * 2
        # Six int64 row/slot tensors and four masks, plus request vectors.
        gather_indices = batch * width * (6 * 8 + 4) + batch * 8 * 8
        gather = 4 * kv + 2 * hidden + gather_indices
        # Include row offsets, gather addresses and both raw-ring masks.
        update_indices = (batch * width * (8 * 4 + 4)
                          + batch * num_layers * min(width, 256) * (8 * 4 + 2)
                          + batch * 8 * 12)
        update_terms = dict(input_and_clean_kv=4 * kv, attention_scores=4 * score,
                            global_outputs=6 * global_output,
                            scalar_statistics=16 * scalar, raw_ring=3 * raw,
                            boundary_hidden=4 * hidden, indices_masks=update_indices)
        update = sum(update_terms.values())
        margin = 32 * 1024 ** 2
        candidates.append(dict(
            bytes=max(gather, update) + margin, max_batch=batch, padded_width=width,
            gather_bytes=gather, update_bytes=update, update_terms=update_terms,
            allocator_library_margin_bytes=margin,
        ))
    result = max(candidates, key=lambda item: item["bytes"])
    result["policy"] = "bf16_rectangular_gather_and_fp32_global16_update_v1"
    return result
