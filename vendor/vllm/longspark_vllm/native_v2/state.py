"""Batched, graph-safe LongSpark state indexed by persistent V2 request slots.

Only confirmed target rows are committed. Proposal does not modify state, so
rejected draft suffixes never require undoing an online-softmax update. The
caller supplies post-RoPE target K, target V, and the selected target layer
outputs (the last target layer uses the *final-normalized* hidden state).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from .target_modules import _normalized, _target_modules


class _BatchContext:
    position_ids_are_scheduler_owned = True
    cuda_graph_static_inputs = True

    def __init__(self, owner, slots):
        self.owner = owner
        self.lengths = owner.lengths.index_select(0, slots)
        self.keys = owner.keys.index_select(0, slots)
        self.values = owner.values.index_select(0, slots)
        self.global_output = owner.global_output.index_select(0, slots)
        self.batch_size = slots.shape[0]

    def attend(self, *, layer_index, query, local_keys, local_values,
               local_attention_mask):
        from specforge.modeling.draft.global16_raw256 import (
            attention_statistics_from_4d, merge_normalized_attention,
        )

        owner = self.owner
        global_keys = owner.query[:, layer_index].expand(self.batch_size, -1, -1, -1)
        raw_mask = (owner.raw_offsets[None, :] < self.lengths[:, None])[:, None, :]
        global_mask = (self.lengths > 0)[:, None, None].expand(-1, 7, 16)
        sources = (
            (global_keys, self.global_output[:, layer_index].to(owner.dtype), global_mask),
            (self.keys[:, layer_index].transpose(1, 2),
             self.values[:, layer_index].transpose(1, 2), raw_mask.expand(-1, 7, -1)),
            (local_keys, local_values, local_attention_mask),
        )
        parts = [_normalized(attention_statistics_from_4d(
            query=query, keys=keys, values=values, attention_mask=mask,
        )) for keys, values, mask in sources]
        output, _ = merge_normalized_attention(parts)
        return output.to(query.dtype)


class LongSparkState:
    """BF16, TP=1, native-RoPE Qwen3 LongSpark execution; greedy Local7 only.

    ``idx_mapping`` is a device int32/int64 vector of distinct active request
    slots. ``-1`` is a graph-padding row: its updates are inert and its proposal
    is all zero. Padding uses a private sentinel, never a real request slot.
    No operation branches on a tensor value or transfers it to the host.
    """

    raw_rows = 256
    global_rows = 16
    block_size = 7

    def __init__(self, draft_path, target_model, hf_config, max_num_reqs, device):
        from specforge.modeling.draft import DraftFreeKVModel

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if not isinstance(max_num_reqs, int) or isinstance(max_num_reqs, bool) or max_num_reqs < 1:
            raise ValueError("max_num_reqs must be a positive integer")
        self.max_num_reqs = max_num_reqs
        if getattr(hf_config, "model_type", None) != "qwen3":
            raise ValueError("Native LongSpark supports dense Qwen3 targets only")
        if getattr(hf_config, "quantization_config", None):
            raise ValueError("Native LongSpark requires an unquantized BF16 target")
        if getattr(hf_config, "use_sliding_window", False):
            raise ValueError("Native LongSpark requires full target attention")
        for rope in (getattr(hf_config, "rope_scaling", None),
                     getattr(hf_config, "rope_parameters", None)):
            if rope and (not isinstance(rope, dict) or
                         rope.get("rope_type", rope.get("type", "default")) != "default" or
                         any(name in rope for name in ("factor", "long_factor", "short_factor"))):
                raise ValueError("Native LongSpark supports native, unscaled RoPE only")
        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
        self.rope_theta = float(rope_parameters.get(
            "rope_theta", getattr(hf_config, "rope_theta", 1_000_000.0)))
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("Target native rotary theta must be positive and finite")
        self.embedding, self.head, self.norm = _target_modules(
            SimpleNamespace(model=target_model, hf_config=hf_config), self.device)
        self.dtype = self.embedding.weight.dtype
        if any(module.weight.dtype != torch.bfloat16
               for module in (self.embedding, self.head, self.norm)):
            raise ValueError("Native LongSpark requires BF16 target weights")
        # Weight shape validation above rejects sharded embeddings/heads. Also
        # check attention head partitioning when given a real vLLM target.
        for layer in getattr(target_model.model, "layers", ()):
            attention = layer.self_attn
            if (attention.num_heads != hf_config.num_attention_heads or
                    attention.num_kv_heads != hf_config.num_key_value_heads):
                raise ValueError("Native LongSpark supports tensor_parallel_size=1 only")
            if not math.isclose(float(getattr(attention.rotary_emb, "base", self.rope_theta)),
                                self.rope_theta, rel_tol=1e-9):
                raise ValueError("Target attention must use the configured native RoPE")

        self.model = DraftFreeKVModel.from_pretrained(
            draft_path, torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        config = self.model.config
        self.reference = self.model.global16_raw256_reference
        if (self.reference is None or not config.global16_raw256_enabled or
                config.draft_mode != "parallel_shifted_qwen" or config.block_size != 7):
            raise ValueError("Native LongSpark requires the Global16/Raw256/Local7 checkpoint")
        ref = self.reference.config
        if (ref.global_memory_slots, ref.raw_target_rows, ref.local_draft_rows,
                ref.max_accepted_tokens) != (16, 256, 7, 8):
            raise ValueError("Native LongSpark requires fixed Global16/Raw256/Local7 geometry")
        dim = int(getattr(hf_config, "head_dim", None) or
                  hf_config.hidden_size // hf_config.num_attention_heads)
        if (config.hidden_size != hf_config.hidden_size or
                config.vocab_size != hf_config.vocab_size or
                config.num_key_value_heads != hf_config.num_key_value_heads or
                config.head_dim != dim or ref.num_attention_heads != hf_config.num_attention_heads):
            raise ValueError("LongSpark checkpoint geometry differs from its Qwen3 target")
        if not math.isclose(float(config.qwen_rope_theta), self.rope_theta, rel_tol=1e-9):
            raise ValueError("LongSpark checkpoint and native target rotary theta differ")
        self.layer_ids = list(config.target_layer_ids)
        self.num_layers = len(self.layer_ids)
        if (len(self.layer_ids) != len(set(self.layer_ids)) or
                any(layer < 0 or layer >= hf_config.num_hidden_layers for layer in self.layer_ids)):
            raise ValueError("LongSpark selected layers are outside the target")
        self.hidden_size = int(config.hidden_size)
        self.kv_heads, self.head_dim = int(config.num_key_value_heads), dim
        for layer in self.model.parallel_qwen_layers:
            attention = layer.self_attn
            attention.inv_freq = 1.0 / (self.rope_theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim))
        self.query = self.reference.global_queries(
            batch_size=1, device=self.device, dtype=self.dtype).detach()
        slots = self.max_num_reqs + 1  # final slot is reserved for graph padding
        self.keys = torch.zeros(slots, self.num_layers, 256, self.kv_heads, dim,
                                dtype=self.dtype, device=self.device)
        self.values = torch.zeros_like(self.keys)
        self.global_output = torch.zeros((slots, *self.query.shape[1:]),
                                         dtype=torch.float32, device=self.device)
        self.global_lse = torch.full(self.global_output.shape[:-1], float("-inf"),
                                     dtype=torch.float32, device=self.device)
        self.hidden = torch.zeros(slots, self.num_layers, self.hidden_size,
                                  dtype=self.dtype, device=self.device)
        self.lengths = torch.zeros(slots, dtype=torch.long, device=self.device)
        self.raw_offsets = torch.arange(256, dtype=torch.long, device=self.device)
        self.layer_offsets = torch.arange(self.num_layers, dtype=torch.long, device=self.device)

    def _vector(self, tensor, size, name):
        if (tensor.shape != (size,) or tensor.dtype not in (torch.int32, torch.int64) or
                tensor.device != self.device):
            raise ValueError(f"{name} must be a device int32/int64 vector of length {size}")

    def _slots(self, idx_mapping):
        if idx_mapping.ndim != 1:
            raise ValueError("idx_mapping must be one-dimensional")
        self._vector(idx_mapping, idx_mapping.shape[0], "idx_mapping")
        torch._assert_async(((idx_mapping >= -1) & (idx_mapping < self.max_num_reqs)).all(),
                            "LongSpark request slot is out of range")
        active = idx_mapping >= 0
        slots = torch.where(active, idx_mapping, self.max_num_reqs).long()
        return slots, active

    @property
    def state_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.mutable_state())

    def mutable_state(self):
        return self.keys, self.values, self.global_output, self.global_lse, self.hidden, self.lengths

    @torch.inference_mode()
    def reset(self):
        """Clear state, including the private padding slot, before graph warmup."""
        for tensor in (self.keys, self.values, self.global_output, self.hidden, self.lengths):
            tensor.zero_()
        self.global_lse.fill_(float("-inf"))

    @torch.inference_mode()
    def update(self, idx_mapping, keys, values, hidden, start_positions, valid_counts):
        """Commit a padded batch of accepted deltas, including chunked prefill.

        K/V: [B,L,T,KVH,D], hidden: [B,T,L,H] or boundary [B,L,H].
        A nonempty update with ``start_positions==0`` resets that request slot.
        Otherwise start must equal its committed length. Zero-count rows are
        exact no-ops even when their start position is zero or a placeholder.
        Rejected rows may contain arbitrary values,
        including NaN: they are zeroed before any attention contraction.
        """
        from specforge.modeling.draft.global16_raw256 import (
            _attention_statistics, merge_normalized_attention,
        )

        slots, active = self._slots(idx_mapping)
        batch = slots.shape[0]
        if (keys.ndim != 5 or keys.shape != values.shape or keys.shape[:2] != (batch, self.num_layers)
                or keys.shape[3:] != (self.kv_heads, self.head_dim) or keys.shape[2] < 1):
            raise ValueError("Target K/V must be [B,selected_layers,T,kv_heads,head_dim], T>=1")
        tokens = keys.shape[2]
        for name, tensor in (("keys", keys), ("values", values), ("hidden", hidden)):
            if tensor.dtype != self.dtype or tensor.device != self.device:
                raise ValueError(f"{name} must use the state's BF16 dtype and device")
        if hidden.shape not in ((batch, tokens, self.num_layers, self.hidden_size),
                                (batch, self.num_layers, self.hidden_size)):
            raise ValueError("Hidden must be [B,T,L,H] or gathered boundary [B,L,H]")
        self._vector(start_positions, batch, "start_positions")
        self._vector(valid_counts, batch, "valid_counts")
        torch._assert_async(((valid_counts >= 0) & (valid_counts <= tokens)).all(),
                            "LongSpark valid_counts must lie within padded input width")
        starts = torch.where(active, start_positions, 0).long()
        counts = torch.where(active, valid_counts, 0).long()
        previous_length = self.lengths.index_select(0, slots)
        changed = counts > 0
        torch._assert_async(((~changed) | (starts == 0) | (starts == previous_length)).all(),
                            "LongSpark update must extend its confirmed prefix or reset at zero")
        reset = (starts == 0) & changed
        valid = torch.arange(tokens, device=self.device)[None, :] < counts[:, None]
        keep_shape = valid[:, None, :, None, None]
        clean_keys = torch.where(keep_shape, keys, 0)
        clean_values = torch.where(keep_shape, values, 0)
        statistics = _attention_statistics(
            query=self.query.expand(batch, -1, -1, -1, -1),
            keys=clean_keys.transpose(2, 3), values=clean_values.transpose(2, 3),
            attention_mask=valid[:, None, :].expand(-1, 16, -1),
        )
        old_output = self.global_output.index_select(0, slots)
        old_lse = self.global_lse.index_select(0, slots)
        old_output = torch.where(reset[:, None, None, None, None], 0, old_output)
        old_lse = torch.where(reset[:, None, None, None], float("-inf"), old_lse)
        output, lse = merge_normalized_attention(((old_output, old_lse), _normalized(statistics)))
        # Keep the attention contractions and normalization above unchanged.
        # Fuse only the final indexed stores; this removes staging allocations
        # and preserves every committed BF16/FP32 bit. The reference switch is
        # useful for GPU regression tests and paired performance measurements.
        if self.device.type == "cuda" and getattr(self, "use_fused_commit", True):
            from .commit_kernel import fused_commit_state
            fused_commit_state(self, slots, keys, values, hidden, starts, counts, output, lse)
            return
        self.global_output.index_copy_(0, slots, torch.where(changed[:, None, None, None, None], output, old_output))
        self.global_lse.index_copy_(0, slots, torch.where(changed[:, None, None, None], lse, old_lse))

        # Exactly min(T,256) distinct physical destinations per request. Select
        # the last accepted rows *before* wrapping, so T>256 cannot introduce
        # duplicate scatter indices or let a rejected suffix erase live rows.
        keep = min(tokens, self.raw_rows)
        offsets = (counts - keep).clamp_min(0)[:, None] + self.raw_offsets[None, :keep]
        raw_valid = offsets < counts[:, None]
        physical = (starts[:, None] + offsets).remainder(self.raw_rows)
        source = offsets.clamp_max(tokens - 1)[:, None, :, None, None].expand(
            -1, self.num_layers, -1, self.kv_heads, self.head_dim)
        address = (slots[:, None, None], self.layer_offsets[None, :, None], physical[:, None, :])
        raw_mask = raw_valid[:, None, :, None, None]
        self.keys[address] = torch.where(raw_mask, keys.gather(2, source), self.keys[address])
        self.values[address] = torch.where(raw_mask, values.gather(2, source), self.values[address])
        if hidden.ndim == 4:
            boundary_index = (counts - 1).clamp_min(0)[:, None, None, None].expand(
                -1, 1, self.num_layers, self.hidden_size)
            boundary = hidden.gather(1, boundary_index).squeeze(1)
        else:
            boundary = hidden
        old_hidden = torch.where(reset[:, None, None], 0, self.hidden.index_select(0, slots))
        self.hidden.index_copy_(0, slots, torch.where((counts > 0)[:, None, None], boundary, old_hidden))
        self.lengths.index_copy_(0, slots, torch.where(changed, starts + counts, previous_length))

    @torch.inference_mode()
    def propose(self, idx_mapping, anchors):
        """Return device [B,7] greedy proposals without mutating committed state.

        Empty positive slots are also legal for native profiling/capture. Their
        Global/Raw sources are masked; the caller keeps those dummy slots
        separate from live requests whose prefix must already be committed.
        """
        slots, active = self._slots(idx_mapping)
        self._vector(anchors, slots.shape[0], "anchors")
        anchors = torch.where(active, anchors, 0).long()
        lengths = self.lengths.index_select(0, slots)
        torch._assert_async(((anchors >= 0) & (anchors < self.head.weight.shape[0])).all(),
                            "LongSpark anchor must be a valid vocabulary token")
        logits = self.model(
            current_hidden_states=self.hidden.index_select(0, slots),
            current_token_ids=anchors, current_position_ids=lengths,
            target_final_norm=self.norm, target_embed_tokens=self.embedding,
            target_lm_head=self.head, global16_paged_context=_BatchContext(self, slots),
            apply_markov_correction=False,
        )
        markov = self.model.parallel_markov_head
        previous = anchors
        sampled = []
        for step in range(self.block_size):
            corrected = logits[:, step]
            if markov is not None:
                corrected = corrected + markov.compute_bias(previous).to(corrected.dtype)
            previous = corrected.argmax(-1)
            sampled.append(previous)
        tokens = torch.stack(sampled, dim=1)
        return torch.where(active[:, None], tokens, 0)
