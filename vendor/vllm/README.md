# LongSpark on vLLM ModelRunner V2

This bundle integrates the original LongSpark checkpoint with **vLLM 0.30.0's
native ModelRunner V2**, following DSpark's speculator lifecycle. The native
engine owns scheduling, target execution, paged KV, GPU rejection and output
delivery. LongSpark supplies bounded Global16/Raw256/Local7 state and the next
seven draft tokens. All vLLM integration code lives under `vendor/vllm/`.

The latest [performance and API validation](validation/perf-api-20261010/README.md)
matched all GSM8K first-100 output tokens with AR, with 96% accuracy. Fused
state stores and direct graph-input gathering improved C1 decode from
334.210 to 335.982 token/s in the recorded before/after run (+0.53%), with
unchanged acceptance length 4.727. At C4, offline wall throughput was
980.625 token/s versus AR's 215.502 (4.550x). These C4 figures compare methods
within that concurrency setting, not the isolated effect of the code change.
The native HTTP API separately passed ordinary, streaming and four-request
concurrency validation with actual target/draft graph replay.

The retained earlier GSM8K first-100 benchmark, run before the code cleanup, matched
AR on **100/100 complete token sequences**, with **96% accuracy for both** and
natural EOS throughout. On one L20X GPU, LongSpark reached **334.022 decode
token/s versus 72.782 for AR (4.589x)**, accepting **4.727 draft tokens/round
(tau 5.727)**. Native target/draft graph replay was verified, with no measured
capture, fallback or JIT-monitor compilation warnings. The
[validation report](validation/native-v2-gsm8k100-20260930/README.md) retains
the original evidence and source hashes. The
[cleanup regression record](validation/v2-only-cleanup/README.md) separately
records verification of the V2-only tree.

Run commands below from the LongSpark repository root. The package name is
`longspark_vllm`; it does not shadow the installed upstream `vllm` package.

## Upstream contribution

The [main-branch contribution bundle](upstream-contribution/README.md) provides
an independent `method="longspark"` implementation for vLLM Model Runner V2,
a cleanly applicable patch, native API/GSM8K regression evidence, and a PR
description. Its isolated checkout is `vendor/vllm/upstream/`; the base is
`10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf`. Human review, relevant test reruns
and DCO sign-off are required before publishing to vLLM.

## Installation

Use an isolated Linux environment. The existing SGLang launchers use a separate
PyTorch/CUDA dependency set. The installer pins vLLM 0.30.0, Transformers
5.12.1 and safetensors 0.8.0, creates `vendor/vllm/.venv`, and checks dependencies
and CUDA. Apply the version-checked native V2 patch separately before use.

```bash
PYTHON_BIN=python3.12 bash vendor/vllm/scripts/install_vllm.sh
```

An existing environment is never overwritten; set `VLLM_ENV_DIR` to select a
new location. The pinned release supports Python 3.10–3.14; Python 3.12 was used
for the recorded GPU evaluation. The default PyPI wheel uses CUDA 13.0 and
requires a suitable driver (580 or newer for CUDA 13.x).

For Linux x86_64 hosts using CUDA 12.9:

```bash
VLLM_CUDA_VARIANT=cu129 PYTHON_BIN=python3.12 \
  bash vendor/vllm/scripts/install_vllm.sh
```

The `cu129` path installs the official `vllm==0.30.0+cu129` wheel with its pinned
SHA256 and selects matching PyTorch libraries using uv's
`--torch-backend=cu129`. It verifies the installed vLLM version and PyTorch
CUDA build. `VLLM_CUDA_VARIANT=default` selects the normal PyPI path.

On supported NVIDIA datacenter GPUs with an older driver, such as driver 570,
CUDA 12.9 PTX compilation may require the `cuda-compat-12-9` user-space
libraries. If provided on the host, prepend their actual directory for both
installation and evaluation:

```bash
export LD_LIBRARY_PATH="/path/to/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Check the [NVIDIA forward-compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
for the hardware/driver combination. The installer inherits this variable; it
does not install compatibility libraries or modify the system driver.
Release and dependency references:
[vLLM 0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0),
[pinned requirements](https://github.com/vllm-project/vllm/tree/v0.30.0/requirements).

After installation, check and apply the native patch. Stop any workers in
that environment before applying or rolling back, and restart them afterwards:

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --check
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --apply
# Restore the verified original upstream files when needed:
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --rollback
```

The installer accepts only exact original/patched source hashes for `0.30.0`
and `0.30.0+cu129`, backs up all five upstream files, and refuses conflicting
local edits. See the [patch guide](patches/README.md) for the contract and
rollback location.

## Run GSM8K

Download a pinned target/draft pair using [WEIGHTS.md](../../docs/WEIGHTS.md).
The retained fixture contains the official GSM8K test set's first 100 questions
in source order, tokenized once using a Qwen3 non-thinking prompt. Both methods
consume those same token IDs.

Preview a three-question comparison without initializing CUDA:

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/evaluate_native_v2_gsm8k.py \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl \
  --target models/4B/target --draft models/4B/longspark --gpu 0 \
  --limit 3 --warmup-count 2 --max-num-seqs 2 --max-num-batched-tokens 512 \
  --max-model-len 4096 --kv-cache-gib 2 --max-new-tokens 1024 \
  --kernel-policy invariant --output results/gsm8k-native-v2-smoke
```

Add `--execute` to run on the selected idle GPU. Existing output directories
are never overwritten. For the recorded first-100 configuration, use a new
output directory, `--limit 100 --warmup-count 3 --max-num-seqs 1` and the other
settings above. The token budget must cover at least `8 * max_num_seqs`.

The launcher enables `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling and native
CUDA graphs before engine imports. It starts AR and LongSpark in separate
fresh processes, warms them independently, and excludes loading, capture and
warmup from measurement. `invariant` enables batch-invariant kernels for strict
greedy token comparisons. `default` is a separate experimental condition.

## Native HTTP API

Use `scripts/serve_native_v2.py` to launch the native vLLM OpenAI-compatible
server with the same V2 speculator, async scheduling and target/draft CUDA
graphs. It supports `/v1/chat/completions`, `/v1/completions`, SSE streaming,
token-ID input/output and concurrent requests. Temperature defaults to zero;
unsupported sampling options receive a client error before engine submission.

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/serve_native_v2.py \
  --target models/4B/target --draft models/4B/longspark --gpu 0 \
  --max-num-seqs 4 --port 18080 \
  --audit-dir results/longspark-api --execute
```

The default address is `127.0.0.1`. The [serving guide](SERVING.md) covers API
calls, SSH forwarding, authentication, supported parameters and graph audits.
Omit `--execute` for a plan preview without loading vLLM or CUDA.

Supported scope is dense unquantized Qwen3 BF16, native unscaled RoPE,
FlashAttention's packed BF16 KV layout, one GPU with TP=PP=DP=CP=1, text input,
and greedy seven-token proposals. Prefix caching, LoRA, fixed-NTK, temperature
sampling, quantization and multi-GPU drafting are unsupported. The retained
first-100 figures above measure offline native engine execution; HTTP API
validation and any API throughput measurement are recorded separately.

## Implementation

| Path | Responsibility |
| --- | --- |
| `longspark_vllm/native_v2/speculator.py` | V2 lifecycle, native graph manager, accepted-prefix updates and proposal scheduling |
| `longspark_vllm/native_v2/state.py` | Per-request Global16/Raw256/Local7 state and checkpoint execution |
| `longspark_vllm/native_v2/commit_kernel.py` | Fused final state stores, preserving the reference attention arithmetic |
| `longspark_vllm/native_v2/target_modules.py` | Shared target modules and attention-statistics helpers |
| `longspark_vllm/native_v2/inputs.py`, `gather_kernel.py` | Reference prefill gather and fused decode KV/hidden gather |
| `longspark_vllm/native_v2/startup.py`, `diagnostics.py` | Memory/graph coverage checks, prefill warmup and actual replay audits |
| `longspark_vllm/gsm8k.py` | Fixture loading, answer scoring and shared evaluation utilities |
| `longspark_vllm/serving.py`, `scripts/serve_native_v2.py` | Native vLLM HTTP launch configuration, greedy request validation and lifecycle graph audits |
| `patches/` | Exact vLLM 0.30.0 extension patch and source-hash manifest |
| `scripts/` | Installation, native evaluation and independent audit entry points |
| `tests/` | CPU contracts and opt-in GPU checks |
| `validation/` | Shared fixture, retained V2 benchmark and cleanup regression evidence |

See the [implementation guide](longspark_vllm/native_v2/README.md) for the
DSpark relationship, checkpoint semantics, request-slot state and graph
boundaries. Decode accepted-state updates and the draft backbone share a
native graph; larger prefills update state outside that graph. Decode gather
is one fused Triton kernel. Reported graph replay does not imply every
operation is compiled or captured.

## Validation

Run the independent standard-library audit on a completed result:

```bash
python vendor/vllm/scripts/audit_native_v2_gsm8k.py \
  results/gsm8k100-native-v2 --expect-limit 100 \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl
```

The audit recomputes parity, scores, acceptance and timing aggregates, and
requires actual native V2 execution and target/draft graph replay. Read
`greedy_parity_passed` and `correctness_status`; process completion alone is
not a parity pass. The retained report distinguishes request service time,
decode time and batch wall time.

Run CPU checks in an environment with PyTorch, Transformers 5.12.1,
safetensors and pytest:

```bash
PYTHONPATH=vendor/vllm:vendor python -m pytest -q vendor/vllm/tests
```

Run the state, input and fused-gather checks on an idle GPU in the vLLM
environment:

```bash
CUDA_VISIBLE_DEVICES=0 LONGSPARK_TEST_CUDA=1 VLLM_BATCH_INVARIANT=1 \
  PYTHONPATH=vendor/vllm:vendor vendor/vllm/.venv/bin/python -m pytest -q \
  vendor/vllm/tests/test_native_v2_state.py \
  vendor/vllm/tests/test_native_v2_inputs.py \
  vendor/vllm/tests/test_native_v2_gather_gpu.py
```

GPU checks are skipped unless explicitly enabled; after opting in, unavailable
or incompatible CUDA is a failure. The remote checkout for recorded runs is
`/data/chenchen/codex/LongSpark` on rl10. Historical records retain their original
execution paths and source digests.
