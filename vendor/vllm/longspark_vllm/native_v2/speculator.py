"""Native ModelRunner V2 LongSpark proposer, following DSpark's lifecycle.

The vLLM engine owns scheduling, target forward, paged KV, GPU rejection,
request lifetimes and async output. This extension only consumes committed
context and proposes the next seven tokens; it never executes the target.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from .state import LongSparkState
from .inputs import gather_target_context
from .gather_kernel import fused_gather_target_context
from .startup import prefill_workspace_budget, require_graph_coverage

logger = init_logger(__name__)


class LongSparkSpeculator(DraftModelSpeculator):
    """Seven-slot anchor-first drafting with device-side accepted-prefix state."""

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        options = vllm_config.additional_config.get("longspark", {})
        self.draft_path = Path(options["draft_model"])
        checkpoint = json.loads((self.draft_path / "config.json").read_text())
        self.layer_ids = list(checkpoint["target_layer_ids"])
        if self.layer_ids != sorted(set(self.layer_ids)):
            raise ValueError("LongSpark target layers must be unique and ascending")
        self.hf_config = vllm_config.model_config.hf_config
        self.target_depth = int(self.hf_config.num_hidden_layers)
        self.head_dim = int(getattr(self.hf_config, "head_dim", None) or
                            self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        self.aux_layer_ids = [i for i in self.layer_ids if i != self.target_depth - 1]
        parallel = vllm_config.parallel_config
        if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1):
            raise ValueError("Native LongSpark currently requires TP=PP=DP=CP=1")
        if self.hf_config.model_type != "qwen3" or self.dtype != torch.bfloat16:
            raise ValueError("Native LongSpark requires dense Qwen3 BF16")
        if vllm_config.model_config.quantization or vllm_config.cache_config.cache_dtype != "auto":
            raise ValueError("Native LongSpark requires unquantized BF16 target weights and KV")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Disable prefix caching: LongSpark must observe every prompt KV row")
        if (self.num_speculative_steps != 7 or self.draft_logits is not None
                or self.enable_adaptive_verification or self.draft_watermarker is not None
                or self.speculative_config.use_local_argmax_reduction):
            raise ValueError("Native LongSpark currently requires fixed seven-token greedy drafting")
        if vllm_config.lora_config is not None:
            raise ValueError("Native LongSpark does not support LoRA")
        self._graph_manager = None
        self._replays = self._fallbacks = self._prefill_calls = self._decode_calls = 0
        self._dummy_calls = 0
        self._startup_complete = False
        self._startup_eager_calls = 0
        self._fused_gather_calls = self._reference_gather_calls = 0
        self._target_attention = []
        # Dedicated extra slots prevent padded/capture rows from touching live state.
        self._padding_slots = torch.arange(self.max_num_reqs, device=device, dtype=torch.long) + self.max_num_reqs
        self._idx = self._padding_slots.clone()
        self._anchors = torch.zeros(self.max_num_reqs, dtype=torch.long, device=device)
        self._starts = torch.ones(self.max_num_reqs, dtype=torch.long, device=device)
        self._counts = torch.zeros_like(self._starts)
        self._token_offsets = torch.arange(self.max_num_tokens, device=device, dtype=torch.long)
        self._prefill_workspace_budget = prefill_workspace_budget(
            max_num_reqs=self.max_num_reqs, max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len, num_layers=len(self.layer_ids),
            kv_heads=int(self.hf_config.num_key_value_heads),
            query_heads=int(self.hf_config.num_attention_heads), head_dim=self.head_dim,
            hidden_size=self.hidden_size,
        )
        self._profile_budget_checks = self._post_capture_budget_checks = 0
        self._graph_batch_coverage = []

    def get_aux_hidden_state_layers(self):
        # Native Qwen3 captures residual sums after each layer. Its final
        # normalized hidden state is supplied separately by ModelRunner V2.
        return tuple(layer + 1 for layer in self.aux_layer_ids)

    def load_draft_model(self, target_model, target_attn_layer_names):
        del target_attn_layer_names
        self.state = LongSparkState(
            str(self.draft_path), target_model, self.hf_config,
            max_num_reqs=2 * self.max_num_reqs, device=self.device,
        )
        self._target_attention = [target_model.model.layers[i].self_attn.attn for i in self.layer_ids]
        layers = len(self.layer_ids)
        kv_heads = int(self.hf_config.num_key_value_heads)
        head_dim = self.head_dim
        shape = (self.max_num_reqs, layers, 8, kv_heads, head_dim)
        self._keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self._values = torch.zeros_like(self._keys)
        self._hidden = torch.zeros(self.max_num_reqs, layers, self.hidden_size,
                                   dtype=self.dtype, device=self.device)
        return self.state.model

    def set_attn(self, model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups):
        # No additional draft KV cache: the bounded state reads the real target KV.
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups
        self.attn_groups = []

    def init_cudagraph_manager(self, cudagraph_mode):
        mode = (CUDAGraphMode.FULL_DECODE_ONLY
                if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL else CUDAGraphMode.NONE)
        if mode == CUDAGraphMode.NONE and not self.vllm_config.model_config.enforce_eager:
            raise ValueError("Native LongSpark requires FULL decode CUDA graphs; eager fallback is disabled")
        # Same native manager and seven-query layout used by DSpark.
        self._graph_manager = CudaGraphManager(
            self.vllm_config, self.device, mode, decode_query_len=7,
        )

    @torch.inference_mode()
    def _generate_draft(self, batch):
        self.state.update(
            self._idx[:batch], self._keys[:batch], self._values[:batch],
            self._hidden[:batch], self._starts[:batch], self._counts[:batch],
        )
        self.draft_tokens[:batch].copy_(self.state.propose(self._idx[:batch], self._anchors[:batch]))

    def capture(self):
        assert self._graph_manager is not None
        self._idx.copy_(self._padding_slots)
        self._starts.fill_(1)
        self._counts.zero_()
        self._anchors.zero_()

        def prepare(desc, warmup):
            del warmup
            batch = desc.num_reqs or desc.num_tokens // 7
            return lambda _mode: self._generate_draft(batch)

        self._graph_manager.capture(prepare, "Capturing native LongSpark CUDA graphs")
        if not self.vllm_config.model_config.enforce_eager:
            self._graph_batch_coverage = require_graph_coverage(
                self._graph_manager, self.max_num_reqs, full_mode=CUDAGraphMode.FULL,
            )
            # Explicit kv_cache_memory_bytes skips automatic cache downsizing.
            # Check again after native KV and both graph pools are allocated;
            # this tensor is temporary and does not touch any request state.
            scratch = torch.empty(self._prefill_workspace_budget["bytes"],
                                  dtype=torch.uint8, device=self.device)
            del scratch
            self._post_capture_budget_checks += 1
        self._startup_complete = True
        logger.info("LongSpark native V2 captured %d draft graphs", len(self._graph_manager.graphs))

    @torch.inference_mode()
    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync=None, dummy_run=False,
                skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
        del attn_metadata, seeds, dp_sync, mm_inputs
        batch = input_batch.num_reqs
        self._idx.copy_(self._padding_slots)
        self._starts.fill_(1)
        self._counts.zero_()
        self._anchors.zero_()
        if dummy_run or skip_attn_for_dummy_run:
            # Profile the actual seven-position network without writing request
            # state or dereferencing a target KV cache that is not allocated yet.
            self._dummy_calls += 1
            # Keep the large-prefill gather/update estimate live across the real
            # draft network profile. No target cache or live state is accessed.
            scratch = None
            if is_profile:
                scratch = torch.empty(self._prefill_workspace_budget["bytes"],
                                      dtype=torch.uint8, device=self.device)
                self._profile_budget_checks += 1
            self._generate_draft(batch)
            del scratch
            return self.draft_tokens[:batch]
        idx = input_batch.idx_mapping.long()
        # V2's built-in warmup runs real scheduler batches with synthetic random
        # sampling parameters before capture(). They exercise kernels only.
        # Serving begins after capture; its supported temperature remains zero.
        startup = not self._startup_complete and not self.vllm_config.model_config.enforce_eager
        if not startup:
            torch._assert_async((temperature[idx] == 0).all(), "Native LongSpark currently requires temperature zero")
        # Fuse the repeatedly used decode shapes. Keep variable-size prefill
        # on the reference path to avoid a first-time Triton compile per prompt.
        gather = gather_target_context
        if int(input_batch.num_scheduled_tokens.max()) <= 8:
            gather = fused_gather_target_context
            self._fused_gather_calls += 1
        else:
            self._reference_gather_calls += 1
        keys, values, hidden, starts, counts, width = gather(
            self, input_batch, slot_mappings, last_hidden_states, aux_hidden_states, num_rejected,
        )
        self._idx[:batch].copy_(idx)
        anchors = last_sampled.reshape(self.max_num_reqs, -1)[idx, 0]
        # A partial prefill proposal is discarded by the scheduler. Unlike
        # DSpark, custom_class does not request a prefill lookahead token.
        self._anchors[:batch].copy_(torch.where(num_sampled > 0, anchors, 0))
        if width > 8:
            # DSpark also ingests prefill context outside its fixed-width query
            # graph. Chunked prefills and mixed batches preserve the same state.
            self._prefill_calls += 1
            self.state.update(idx, keys, values, hidden, starts, counts)
        else:
            self._decode_calls += 1
            self._keys[:batch].zero_()
            self._values[:batch].zero_()
            self._keys[:batch, :, :width].copy_(keys)
            self._values[:batch, :, :width].copy_(values)
            self._hidden[:batch].copy_(hidden)
            self._starts[:batch].copy_(starts)
            self._counts[:batch].copy_(counts)
        manager = self._graph_manager
        if manager is None:
            raise RuntimeError("Native LongSpark graph manager was not initialized")
        desc = manager.dispatch(batch, batch * 7, uniform_token_count=7, num_active_loras=0)
        if startup:
            self._generate_draft(batch)
            self._startup_eager_calls += 1
        elif desc.cg_mode == CUDAGraphMode.FULL and not is_profile:
            manager.run_fullgraph(desc)
            self._replays += 1
        elif self.vllm_config.model_config.enforce_eager or is_profile:
            self._generate_draft(batch)
        else:
            self._fallbacks += 1
            raise RuntimeError("Native LongSpark has no captured graph for this batch; fallback forbidden")
        return self.draft_tokens[:batch]

    def runtime_stats(self):
        manager = self._graph_manager
        return dict(
            implementation="longspark_native_model_runner_v2", captures=len(manager.graphs) if manager else 0,
            replays=self._replays, fallbacks=self._fallbacks,
            prefill_calls=self._prefill_calls, decode_calls=self._decode_calls, dummy_calls=self._dummy_calls,
            startup_complete=self._startup_complete, startup_eager_calls=self._startup_eager_calls,
            context_gather="triton_decode_reference_prefill",
            fused_gather_calls=self._fused_gather_calls, reference_gather_calls=self._reference_gather_calls,
            cuda_graph_manager=type(manager).__module__ + "." + type(manager).__name__ if manager else None,
            selected_target_layers=self.layer_ids, draft_tokens=7,
            target_forward_owned_by="vllm.v1.worker.gpu.model_runner.GPUModelRunner",
            graph_batch_sizes=sorted({d.num_reqs for d in manager.graphs}) if manager else [],
            graph_batch_coverage=self._graph_batch_coverage,
            prefill_workspace_budget=self._prefill_workspace_budget,
            profile_budget_checks=self._profile_budget_checks,
            post_capture_budget_checks=self._post_capture_budget_checks,
        )
