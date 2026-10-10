# Native ModelRunner V2 LongSpark

This extension puts LongSpark drafting inside vLLM 0.30.0's native GPU
ModelRunner V2. The engine owns request scheduling, target execution, paged KV,
verification, stopping and output delivery. `LongSparkSpeculator` consumes the
native target results and returns the next seven draft token IDs. It never
executes the target itself.

Run the commands here from the LongSpark repository root. Install the isolated
environment using [the bundle guide](../../README.md#installation).

## Relationship to DSpark

The integration follows the upstream DSpark execution structure, but loads
the original LongSpark checkpoint and retains its mathematical definition.

| Component | Upstream Qwen3 DSpark | This LongSpark extension |
| --- | --- | --- |
| Native interface | V2 `DraftModelSpeculator`, through DFlash | V2 `DraftModelSpeculator`, through the explicitly opted-in custom-class extension |
| Query layout | Anchor plus six masked slots for seven next-token predictions | Same seven-slot, anchor-first layout |
| Token dependency | Left-to-right low-rank Markov correction | The checkpoint's Markov correction, conditioned on each actual previous greedy token |
| Context | Target hidden features projected into a separate draft KV cache | Actual selected target layers' post-RoPE K and V, summarized into Global16 plus the latest Raw256 |
| Draft inputs | DFlash-derived model's feature projection and embeddings | LongSpark `DraftStateEncoder`, learned position embeddings, and `ParallelShiftedInput` |
| Draft attention | DFlash attention over its context/query cache | Separate Global16, Raw256 and fully visible Local7 sources, combined by online-softmax statistics |
| Persistent state | Draft attention cache managed through native KV groups | Bounded per-request FP32 global summary, BF16 Raw256 and boundary hidden; no full-prefix draft cache |
| Graphs | Native graph manager covering parallel backbone and sequential sampling | Native graph manager covering the small accepted update, LongSpark backbone and sequential sampling |

Consequently, the checkpoint is **not** registered as `Qwen3DSparkModel` or
converted to DFlash weights. Their context projection and input networks are
different. The extension loads `DraftFreeKVModel` from bundled SpecForge and
shares the colocated target embedding, output head and final norm weights.
The target remains the native vLLM Qwen3 model. Checkpoint files are unchanged.

The upstream reference implementations are
[`DSparkSpeculator`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py),
[`Qwen3DSparkForCausalLM`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/qwen3_dspark.py)
and the associated DFlash model/speculator. These are architectural references;
the LongSpark implementation resides in [speculator.py](speculator.py),
[inputs.py](inputs.py) and [state.py](state.py).

## Install and roll back the patch

The pinned release does not expose this custom-class V2 contract without the
bundled patch. Stop workers in the isolated environment, then run:

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --check
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --apply
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --check
```

The installer accepts only `0.30.0` and `0.30.0+cu129`, verifies every original
or patched source hash, and saves original files before modifying the five
upstream files. It does not import CUDA, install dependencies, restart workers
or modify the system driver. Unknown or mixed source states are rejected;
there is no force or fuzzy-patch option. Restart workers after applying it.

To restore the original verified files, stop workers and run:

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --rollback
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --check
```

Rollback verifies the backup and refuses to overwrite later source edits.
Backups are retained under
`site-packages/.longspark-native-v2-patch/longspark-native-v2-custom-class-v1/`.
See [the patch manifest and contract](../../patches/README.md) for exact changes.
The bundle installation script installs dependencies; apply the patch
explicitly using the commands above before running the native extension.

The engine selects the extension through `method="custom_class"`, class path
`longspark_vllm.native_v2.speculator.LongSparkSpeculator`, and
`additional_config.native_v2_custom_class=True`. The checkpoint path is supplied
in `additional_config.longspark.draft_model`. The patched path preserves V1's
existing custom proposer behavior unless this V2 opt-in is explicit.

## Accepted context and batch state

`gather_target_context` reads native GPU slot mappings and target KV pages.
For request `r`, the confirmed target interval ends at
`query_start_loc[r + 1] - num_rejected[r]`. Accepted lengths remain GPU tensors;
the CPU only knows the scheduler's padded query-width bound.

The input to `LongSparkState.update` is:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `idx_mapping` | `[B]` | Persistent request slot, independent of current packed batch order |
| `keys`, `values` | `[B,L,T,KVH,D]` | Padded target delta for selected layers |
| `hidden` | `[B,L,H]` or `[B,T,L,H]` | Confirmed boundary hidden, or rows from which to gather it |
| `start_positions` | `[B]` | Absolute first position of each delta |
| `valid_counts` | `[B]` | Confirmed rows per request; rejected suffix rows are excluded |

Every nonempty update either extends the committed prefix or resets a reused
slot when its start position is zero. A zero-count update is an exact no-op,
including for graph padding and after an eager prefill update. Dedicated dummy
slots keep profiling/capture separate from live request state. Rejected NaN
padding is cleared before contractions; Raw256 writes have unique physical
destinations even when a prefill chunk exceeds 256 rows.

The selected final target layer uses `last_hidden_states` after the target's
final norm. Other selected layers use native auxiliary residual-sum outputs.
Treating the final auxiliary capture as already normalized would change the
checkpoint's inputs. The published 4B checkpoint selects layers
`[0, 9, 18, 26, 35]`.

Proposal reads the confirmed state without modifying it, so rejection never
needs to subtract from the Global16 summary. Target cache rollback and
overwriting rejected suffix slots remain native engine responsibilities.
The extension currently disables prefix caching: a skipped prompt prefix
would omit rows required to construct its online state. Preempted requests
must therefore rebuild state from their re-executed prefix starting at zero.

## Native graphs and performance boundaries

The runner requests target `FULL_AND_PIECEWISE` CUDA graph mode with
`enforce_eager=False`; it leaves the upstream compilation mode at its default.
The drafter uses the native `CudaGraphManager` with `decode_query_len=7` and
full decode graphs. Captures are bucketed by padded batch size.

Native kernel warmup precedes graph capture and uses synthetic requests with
random sampling parameters. The extension counts its eager startup calls
separately. Once capture completes, serving requires temperature zero and
complete graph coverage; those startup calls are excluded from measurements.

For target deltas of up to eight rows, the accepted-state update and proposal
are captured together. Per-request acceptance counts, lengths, slot mapping
and Markov feedback remain device data. Larger prefill deltas update bounded
state outside the fixed-width proposal graph, then run that same proposal
path with a zero-count update. Decode context gathering uses one fused Triton
kernel for all selected KV and boundary-hidden layers, outside the draft
graph. It writes directly to the graph input buffers, including zeroing the
unused tail, avoiding intermediate allocation and copy launches. Variable-size
prefill gathering retains the reference implementation to avoid a new JIT
compilation for every prompt length. Missing decode graph coverage
raises an error; the evaluation path does not silently switch to eager.

After the unchanged BF16 contractions and FP32 attention normalization,
`commit_kernel.py` fuses only the final Global16, Raw256, hidden and length
stores. Runtime prefill strides keep different prompt lengths on the same
compiled store kernel. CPU execution retains the reference implementation.
Both optimizations default on; set
`additional_config.longspark.use_fused_state_commit=false` and
`use_direct_gather=false` to restore the previous paths. The offline runner's
`--native-optimizations reference` selects both reference paths for A/B runs.
See the [component measurements](../../validation/perf-api-20261010/components/README.md)
and [end-to-end/API validation](../../validation/perf-api-20261010/README.md).

The three-source draft attention still uses SpecForge's PyTorch statistics
implementation. This extension does not introduce a fused LongSpark attention
kernel or claim every operation is compiled. Native scheduling and actual
graph replay establish the execution path; they do not establish optimal
performance by themselves.

## Run the native engine

For the native vLLM OpenAI-compatible HTTP service, see the
[serving guide](../../SERVING.md). It uses this same speculator and graph path.

Prepare the pinned target and draft according to [WEIGHTS.md](../../../../docs/WEIGHTS.md).
The existing shared GSM8K JSONL supplies token IDs directly. Preview a small
two-request comparison without importing vLLM or initializing CUDA:

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/evaluate_native_v2_gsm8k.py \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl \
  --target models/4B/target --draft models/4B/longspark --gpu 0 \
  --limit 3 --warmup-count 2 --max-num-seqs 2 --max-num-batched-tokens 512 \
  --max-model-len 4096 --kv-cache-gib 2 --max-new-tokens 1024 \
  --kernel-policy invariant --output results/gsm8k-native-v2-smoke
```

Add `--execute` after reviewing the plan. For a first-100 run, use a new output
directory and increase the limit:

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/evaluate_native_v2_gsm8k.py \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl \
  --target models/4B/target --draft models/4B/longspark --gpu 0 \
  --limit 100 --warmup-count 3 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --max-model-len 4096 --kv-cache-gib 2 --max-new-tokens 1024 \
  --kernel-policy invariant --execute --output results/gsm8k100-native-v2
```

`--gpu` selects the physical GPU through the subprocess's `CUDA_VISIBLE_DEVICES`.
The launcher sets `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling and the selected
batch-invariant policy before engine imports. Use the host's required CUDA
compatibility library path from the installation guide when applicable.
It fixes `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid requiring a CUDA compiler for
the built-in random-sampler warmup on rl10. Measured requests use greedy
sampling; the target attention backend remains FlashAttention 3.
`--max-num-seqs 1` gives serial submission; values above one submit batches of
that size. The token budget must cover at least `8 * max_num_seqs`. Increase
context/KV budgets explicitly when needed. Existing result directories are
never overwritten.

Each method starts a fresh native `LLM`, warms up separately, then measures
the same unmodified tokenized inputs at temperature zero with natural EOS.
Before request warmup, the LongSpark worker precompiles bounded-state prefill
updates for the possible widths and batch sizes on private padding slots.
Zero accepted counts make these updates inert; a byte-for-byte comparison of
all mutable state must pass. `prefill-warmup.json` records this excluded phase.
This covers shape-specific matrix kernels that three sample prompts alone
may not warm. The final report also inspects JIT monitor warnings between the
last warmup result and the last measured result.
The runner evaluates AR and LongSpark; it does not run SGLang. `invariant` and
`default` kernel policies are separate experimental conditions. Compare exact
output IDs in both policies and report mismatches as such.

## Read the evidence

- `plan.json` records both native engine configurations and the input digest.
- Each method's `manifest.json` records environment and actual worker classes.
- `measurement-audit.json` records successful native target replay calls,
  dispatched shapes/concurrency and LongSpark graph counter differences.
- `records.jsonl` retains per-question outputs, natural stopping, scores,
  native timestamps and raw detailed speculative metrics.
- `batches.json` records wall time and submitted concurrency; `summary.json`
  retains service, decode and batch-wall throughput separately.
- `completed.json` records exact AR/LongSpark token parity and aggregate results.

The plan and manifests include source hashes. Each engine verifies them
before loading; changing inference sources between AR and LongSpark fails the
run. The independent standard-library audit recomputes the recorded results:

```bash
python vendor/vllm/scripts/audit_native_v2_gsm8k.py \
  results/gsm8k100-native-v2 --expect-limit 100 \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl
```

The audit requires the actual V2 runner and async scheduling, existing target
and draft captures, positive measured replays when decoding occurred, no
measurement-time capture, and no draft fallback. Successful completion alone
does not prove exact parity: check `greedy_parity_passed` and
`correctness_status` explicitly.

Per-request service time is `last_token_ts - scheduled_ts`; decode time is
`last_token_ts - first_token_ts`. Summed request times can overlap under
batching. Therefore `generate_wall_tps` measures batch throughput separately
and must not be confused with tokens divided by summed request service times.
Native raw acceptance counts precede EOS/output-cap truncation; emitted counts
clip the final verification step to tokens actually returned.

## Supported scope and validation

Current execution is limited to dense unquantized Qwen3 BF16, native unscaled
RoPE, FlashAttention's expected packed BF16 KV layout, one GPU with
TP=PP=DP=CP=1, text input and greedy seven-token proposals. The published
LongSpark Global16/Raw256/Local7 geometry must match the target. Native
temperature sampling, fixed-NTK, quantization, LoRA, prefix caching, adaptive
verification and multi-GPU drafting are unsupported. Serving API integration,
distributed execution and production load are not established by this runner.

The [native GSM8K first-100 report](../../validation/native-v2-gsm8k100-20260930/README.md)
records the V2 benchmark before the code cleanup: 100/100 exact AR/LongSpark
token matches, 96% accuracy for both, and 334.022 versus 72.782 decode token/s
(4.589x) on one L20X. Actual native target/draft graph replay was verified.
The retained raw records, original source hashes and independent audit describe
that measured revision. The
[cleanup regression record](../../validation/v2-only-cleanup/README.md)
separately records verification of the V2-only tree.

```bash
CUDA_VISIBLE_DEVICES=0 LONGSPARK_TEST_CUDA=1 VLLM_BATCH_INVARIANT=1 \
  PYTHONPATH=vendor/vllm:vendor vendor/vllm/.venv/bin/python -m pytest -q \
  vendor/vllm/tests/test_native_v2_state.py \
  vendor/vllm/tests/test_native_v2_inputs.py
```

CPU checks also cover patch install/rollback and runner audits:

```bash
PYTHONPATH=vendor/vllm:vendor python -m pytest -q \
  vendor/vllm/tests/test_native_v2_state.py \
  vendor/vllm/tests/test_native_v2_inputs.py \
  vendor/vllm/tests/test_native_v2_patch_installer.py \
  vendor/vllm/tests/test_native_v2_gsm8k_runner.py
```
