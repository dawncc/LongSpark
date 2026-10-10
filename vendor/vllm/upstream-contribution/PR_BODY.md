<!-- markdownlint-disable -->
## Overview

Add opt-in `method="longspark"` speculative decoding for Model Runner V2. LongSpark predicts seven tokens in parallel, applies sequential Markov correction, and maintains Global16/Raw256 memory from confirmed target KV. It uses the standard checkpoint loader and `vllm serve` API.

The initial scope is native dense Qwen3, BF16 weights/KV, FlashAttention, one CUDA GPU, and greedy sampling. Prefix caching, LoRA, KV transfer, quantization, adaptive verification, structured output, custom logits processing, watermarking, and input streaming are rejected explicitly. Output streaming is supported.

## Claims

- Existing methods retain their selection and validation paths. Common changes consist of registration, method-scoped capability/request checks, and one additional auxiliary-hidden-state method entry in Model Runner V2. The scheduler and rejection sampler are unchanged.
- The native V2 graph manager captures the Local7 forward and sequential sampling. Decode gather writes directly into persistent graph inputs; fused state stores preserve the original BF16/FP32 arithmetic.
- In one paired GSM8K first-100 run on an L20X at concurrency 4, all output token sequences matched AR and both scored 96%. Observed generation wall throughput was 984.96 versus 217.46 token/s (4.529x); accepted draft tokens per verification round averaged 4.7254 (5.7254 including the bonus/correction token). These are vLLM counters before final EOS trimming; eight requests dropped the last bonus, giving 5.7233 delivered decode tokens/round. This measures LongSpark versus AR, not a gain over the prior out-of-tree integration, and is not a repeated-trial estimate.

## Validation

Base: `10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf`.

- Relevant configuration, sampling, MRV2, n-gram, DSpark top-k and EAGLE kernel regressions: **279 passed**. A final capability/input-stream guard rerun: **34 passed**, overlapping the preceding group. The gated `Meta-Llama-3-8B-Instruct` configuration test failed with HTTP 403 on unmodified main and was deselected in the modified run.
- New model/state tests: **18 passed, 2 skipped** (CUDA Graph cases on CPU). Four real-checkpoint comparisons against the original implementation matched all six state buffers, full FP32 logits, and draft tokens bit-for-bit, including long prefill, rejected rows, padding, and slot reuse.
- Native API tests: **2 passed**: completions/chat, JSON/SSE equality, four concurrent requests, unsupported sampling returning HTTP 400, and successful generation afterward.
- Contributed GSM8K tests: **2 passed**: graph mode first 100 and eager mode first 4, with exact token equality to AR and nonzero draft acceptance. These tests use upstream zero-shot chat prompts: 85% accuracy and acceptance length 6.161 for the 100-question case. They are a different prompt format from the fixed 96% fixture above.
- Existing model registry aliases: **2 passed**; existing CUDA model initialization: **1 passed**. An additional n-gram GPU engine smoke matched AR prefixes on 4/4 requests.
- All affected generation tests used `VLLM_GPU_SYNC_CHECK=error`; no synchronization error occurred. Target and LongSpark FULL graph capture completed. Pre-commit and whitespace checks passed.

Representative commands (from the vLLM root; activate its environment and select an idle CUDA GPU):

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=0
export VLLM_GPU_SYNC_CHECK=error
.venv/bin/python -m pytest tests/test_config.py -k 'v2_model_runner or speculative or dspark or eagle or longspark' --deselect=tests/test_config.py::test_eagle_draft_model_config -q
.venv/bin/python -m pytest tests/config/test_longspark.py tests/test_sampling_params.py tests/config/test_speculative_draft_hf_overrides.py tests/config/test_speculative_draft_max_position_embeddings.py tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_gpu_sampler_flags.py tests/v1/spec_decode/test_dspark_topk.py tests/v1/spec_decode/test_eagle_step_kernel.py tests/v1/spec_decode/test_ngram_gpu.py tests/v1/streaming_input/test_async_llm_streaming.py -q
.venv/bin/python -m pytest tests/v1/spec_decode/test_longspark_model.py tests/v1/spec_decode/test_longspark_state.py -q
.venv/bin/python -m pytest tests/v1/e2e/spec_decode/longspark/ -v -s
.venv/bin/pre-commit run
```

The E2E runs used local copies of `Qwen/Qwen3-4B` and `Hehy/LongSpark-Qwen3-4B`. The public draft config was fetched at revision `241060f1fdefdfe9b49d1a8a2c578670ac5e9d10` and matches the local checkpoint config byte-for-byte.

Environment: Python 3.12, Torch 2.13.0+cu129, Transformers 5.19.0. No exact-base cu129 wheel was available. Current-main Python code was tested with the cu129 extension wheel from `8cbd5d03006c33185f402249ff2b448efd594986`; comparison of compiled-source paths found only a deletion inside `#ifdef USE_ROCM`, leaving the CUDA compilation path unchanged. This is **not** an exact-commit full source build. The Python frontend was selected; Rust frontend, ROCm, full DSpark model E2E, and full EAGLE model E2E were not validated.

## Details

GitHub searches on 2026-10-10 found no LongSpark issue or open PR in `vllm-project/vllm`. DSpark supplies the existing native lifecycle pattern; LongSpark differs by maintaining bounded confirmed-target memory rather than a draft paged KV cache. No generic custom-proposer interface changes, monkeypatches, diagnostic RPCs, experiment launchers, or new runtime dependencies are included.

Inference code is extracted from LongSpark/SpecForge with original MIT notices preserved. Model geometry and unsupported features are checked explicitly. A dedicated CUDA Buildkite job runs the new GSM8K and API E2E tests; unit tests use the existing spec-decode/config suites.

AI assistance: OpenAI Codex assisted with implementation, code review, testing, and this description. The contribution is prepared for human review; the submitting human must review all changed lines, run the relevant tests, and agree to the DCO before publishing. No human review or sign-off is claimed here.

---
<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill; its findings and evidence are recorded in the preparation bundle.
- [x] AI assistance was used during creation and is disclosed above.
- [x] Design Fit: the feature is opt-in, method-scoped, and targets Model Runner V2.
- [x] Testing and Validation: affected behavior, model outputs, acceptance, serving, and regressions were exercised; environment limits are explicit.
- [x] Code Quality and Style: pre-commit and whitespace checks pass; source attribution is retained.
- [x] Pull Request Contents: scope, claims, evidence, limitations, and duplicate-work checks are included.
- [ ] Human submitter has reviewed every changed line, rerun relevant tests, and agreed to/signs the DCO.

</details>

**BEFORE SUBMITTING, PLEASE READ https://docs.vllm.ai/en/latest/contributing**
