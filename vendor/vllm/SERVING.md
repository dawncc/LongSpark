# LongSpark V2 native vLLM API

The launcher calls the pinned **vLLM 0.30.0 native `vllm serve` CLI**.
vLLM owns the OpenAI-compatible HTTP handlers, chat templates, continuous
batching, cancellation, metrics and streaming. LongSpark uses the same native
ModelRunner V2 speculator as the offline evaluator. A small ASGI middleware
rejects unsupported generation options before they can reach the worker.
There is no custom inference loop or second HTTP proxy.

## Start the server

Install and apply the version-checked patch as described in [README.md](README.md).
From the repository root on rl10:

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/serve_native_v2.py \
  --target /data/lpf/models/Qwen3-4B \
  --draft /data/chenchen/codex/LongSpark/models/4B/longspark \
  --gpu 0 --host 127.0.0.1 --port 18080 \
  --served-model-name longspark-qwen3-4b \
  --max-num-seqs 4 --max-num-batched-tokens 512 \
  --max-model-len 4096 --max-new-tokens 1024 --kv-cache-gib 4 \
  --audit-dir /data/chenchen/codex/LongSpark/artifacts/api-serving/example \
  --execute
```

Choose an idle GPU and a new audit directory. Omit `--execute` to print the
native CLI arguments and environment without importing the inference runtime.
The launcher checks the installed patch hashes; it never applies a patch to
a running installation. Existing audit directories are not overwritten.

The default is BF16, one GPU, seven draft tokens, async V2 scheduling,
`FULL_AND_PIECEWISE` CUDA graphs, batch-invariant kernels, chunked prefill and
no prefix caching. The server reserves 4 GiB for target KV to accommodate
four 4096-token requests in the tested Qwen3-4B configuration. Check native
startup KV capacity when changing the model, context or concurrency.
`--max-num-seqs` controls the native scheduler's concurrent
request limit. Its token budget must be at least eight times that limit.
The default chat template sets `enable_thinking=false`. `--max-new-tokens`
sets the server generation default and maximum output limit; prompts and
outputs must also fit `--max-model-len`.
The model's `generation_config.json` retains its EOS token IDs; all sampling
defaults are explicitly overridden to greedy, unfiltered decoding. In the
validated Qwen3-4B checkpoint this preserves both EOS IDs 151645 and 151643.

The native server honors `VLLM_API_KEY` if authentication is desired; export
it before starting the launcher. The key is not written to the launch plan.
For access from your local computer, forward the loopback port:

```bash
ssh -N -L 18080:127.0.0.1:18080 rl10
```

The client base URL is then `http://127.0.0.1:18080/v1`.

## Call the API

Chat completion (add an Authorization bearer header if `VLLM_API_KEY` was set):

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"longspark-qwen3-4b","messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":128,"temperature":0}'
```

Streaming text completion:

```bash
curl -N http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"longspark-qwen3-4b","prompt":"The answer is","max_tokens":32,"stream":true,"stream_options":{"include_usage":true}}'
```

The Python OpenAI client works with these native endpoints:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="longspark-qwen3-4b",
    messages=[{"role": "user", "content": "Explain 17 times 23."}],
    temperature=0,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

Use the configured key instead of `EMPTY` when authentication is enabled.
Both endpoints support native `return_token_ids=true`; completion requests
also accept a token-ID list as `prompt`. SSE responses retain native chunking,
usage and `[DONE]` framing. Client disconnects pass through to native request
cancellation. Independent requests can run concurrently; `n=1` is required
per prompt. A list of prompts in the completions API uses native batching.

## Supported request parameters

Omitted or null temperature becomes `0`. Explicit nonzero temperature,
`top_p != 1`, `top_k` other than `0`/`-1`, nonzero `min_p`, non-neutral
penalties, beam search, `n != 1`, `best_of != 1`, logit bias, constrained
decoding, custom logits processors, tool calls and multimodal content return
an OpenAI-shaped HTTP 400 error. These limits describe the validated LongSpark
scope. They prevent unsupported requests from terminating the shared GPU
worker. Ordinary stop strings, token limits, seeds, streaming, usage and
token-ID return options pass to vLLM's own request validation.
`min_tokens` must be zero: minimum-length stop masking is outside this
validated scope, and vLLM 0.30 also has a fixed 128-token stop-mask buffer in
its V2 worker. Rejecting it before submission avoids a worker failure from
otherwise valid requests with too many stop IDs.

Generation is exposed only through `/v1/chat/completions` and
`/v1/completions`. Other write endpoints, including Responses and native
management RPC routes, are disabled by this launcher. Native `/health`,
`/v1/models`, `/metrics`, `/tokenize` and `/detokenize` remain available.
Request bodies are limited to 16 MiB and must be uncompressed JSON.

## Warmup and graph evidence

Before the server reports ready, its lifespan hook calls native
`AsyncLLM.collective_rpc` to prewarm LongSpark state-update widths up to the
configured prefill token budget, then starts a worker graph audit. Warmup
uses private slots and verifies mutable state is unchanged. The audit spans
HTTP requests. A wrapper around native `AsyncLLM.shutdown` collects the final
audit before the engine closes. The method stays synchronous; vLLM's native
signal handler runs it in an executor thread, and the wrapper submits the
RPC to the API event loop with a 30-second bound. ASGI lifespan shutdown
alone cannot do this in vLLM 0.30 because it follows engine shutdown.

Some native sampler kernels and the first concurrent gather shape still JIT
on their first real requests. Send representative single and concurrent
requests before measuring HTTP latency. The [recorded API validation](validation/perf-api-20261010/README.md)
keeps these cold-request warnings separate from graph replay evidence.

| File in `--audit-dir` | Evidence |
| --- | --- |
| `launch.json`, `patch-check.json` | Exact native options and verified installed patch |
| `prefill-warmup.json` | Shape coverage and unchanged-state check |
| `startup-audit.json` | Actual native V2 runner, async mode and captured graphs |
| `shutdown-audit.json` | Actual target/draft replays, capture deltas and fallbacks |

Stop the server with SIGINT/SIGTERM **after requests finish** to retain the
final audit. If requests are still running, the audit ends at the pre-shutdown
RPC snapshot; it does not claim to cover subsequent drain/abort work.
New generation requests receive HTTP 503 once the shutdown wrapper begins.
A forced kill cannot produce a shutdown audit. A shutdown RPC failure,
timeout, or exceptional teardown on the API event-loop thread is saved as
`shutdown-audit-error.json`; it is not a successful graph audit. The native
shutdown still runs, and later teardown calls do not repeat the RPC.
No HTTP diagnostics or control endpoint is added. Offline benchmark speeds
are not HTTP throughput measurements; report API client wall time separately.

Upstream references:
[native CLI](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/cli/serve.py),
[native API launcher](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/launchers/api_server/entry.py),
[middleware registration](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/serve/middleware/register.py).
