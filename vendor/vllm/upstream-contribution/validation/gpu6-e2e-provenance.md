# Native API and model E2E validation

Executed on rl10, GPU 6, on 2026-10-10. The remote checkout was
`/data/chenchen/codex/vllm-longspark-upstream`, with the contribution applied to
upstream base commit `10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf`.
Both successful invocations exited with status 0. No additional source changes
were made to obtain these results.

## Source identity and checkpoint substitution

The public test source is
`tests/v1/e2e/spec_decode/longspark/test_longspark.py`.
Its SHA-256 was identical in the local contribution checkout and the remote
executed checkout:

```text
a9629371e341e07d1b754248cd771db74a6cc670dfb206e3fc935330704fe06f
```

`local-checkpoints-pytest.py` is a copy of the exact validation-only runner at
remote `artifacts/api-test-runner.py`; SHA-256:

```text
1ecbac03ace55e24b8af6fbb147f1172ec4cf83efbb9c863858a5fb1ead4058f
```

That runner uses a pytest collection plugin to substitute the available local
target `/data/lpf/models/Qwen3-4B` and draft
`/data/chenchen/codex/LongSpark/models/4B/longspark` for the public source's
`Qwen/Qwen3-4B` and `Hehy/LongSpark-Qwen3-4B`, and selects the Python native
frontend with `VLLM_USE_RUST_FRONTEND=0`. It does not change prompts, assertions,
sampling parameters, or engine settings in the public test.

The public test sets `VLLM_USE_V2_MODEL_RUNNER=1`,
`VLLM_GPU_SYNC_CHECK=error`, `VLLM_BATCH_INVARIANT=1`, and
`VLLM_USE_FLASHINFER_SAMPLER=0`. Both graph paths use
`CompilationConfig(cudagraph_mode="FULL_AND_PIECEWISE")`;
logs confirm V2, FlashAttention 3, and LongSpark FULL graph capture.

## Commands actually executed

These commands ran through SSH on rl10. The `.venv` is the isolated uv-managed
environment in the checkout. Adding its `bin` directory to PATH is necessary
because `RemoteOpenAIServer` launches the `vllm` executable.

```sh
cd /data/chenchen/codex/vllm-longspark-upstream &&
PATH=/data/chenchen/codex/vllm-longspark-upstream/.venv/bin:$PATH \
CUDA_VISIBLE_DEVICES=6 VLLM_USE_RUST_FRONTEND=0 \
.venv/bin/python artifacts/api-test-runner.py \
> artifacts/api-e2e/test.log 2>&1
```

```sh
cd /data/chenchen/codex/vllm-longspark-upstream &&
PATH=/data/chenchen/codex/vllm-longspark-upstream/.venv/bin:$PATH \
CUDA_VISIBLE_DEVICES=6 VLLM_USE_RUST_FRONTEND=0 \
.venv/bin/python artifacts/api-test-runner.py greedy_target artifacts/model-e2e \
> artifacts/model-e2e/test.log 2>&1
```

For a repeat run using this downloaded runner, copy
`local-checkpoints-pytest.py` to `artifacts/api-test-runner.py` in the remote
checkout first. The public tests themselves remain unchanged.

## Results and evidence

| Run | Result | Evidence |
| --- | --- | --- |
| Native API, completion and chat | 2 passed, 2 deselected, 87.99 s | `api-e2e/test.log`, `api-e2e/junit.xml` |
| Model, graph first 100 and eager first 4 | 2 passed, 2 deselected, 211.75 s | `model-e2e/test.log`, `model-e2e/junit.xml` |

Native API tests cover JSON/SSE equality, four concurrent requests per endpoint,
and HTTP 400 for unsupported nonzero temperature and presence penalty, followed
by successful requests proving that rejection did not terminate the worker.
The server uses the ordinary `RemoteOpenAIServer` fixture and native routes.

The public model test uses the existing GSM8K helper with `num_shots=0` and a
single-user nonthinking chat prompt. Its graph run produced **100/100 exact
token-sequence matches to AR**, **85% accuracy**, and **6.161 mean acceptance
length**. The eager smoke produced **4/4 exact matches**, **100% accuracy**, and
**6.636 mean acceptance length**. Acceptance length includes the bonus token.
No GPU synchronization checker error occurred.

The separate user-fixed-prompt benchmark, recorded by the coordinating task,
reported **100/100 exact matches**, **96% accuracy**, and **5.725 mean acceptance
length**. It uses a different prompt fixture; its accuracy and acceptance must
not be presented as the result of this public zero-shot E2E test or combined
with this test's metrics. Both validations evaluate the first 100 GSM8K items.

The API server fixture terminated its server normally, and the GPU 6 model
test processes exited. The logs and JUnit files were copied directly from
the remote artifact directories after completion.
