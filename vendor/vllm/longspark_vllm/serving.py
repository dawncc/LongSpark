"""Native vLLM API launch configuration and a greedy-only ASGI boundary.

The native vLLM server owns HTTP routing, tokenization, scheduling, cancellation
and SSE output. This module imports neither vLLM nor CUDA until execution.
"""

from __future__ import annotations

import argparse
import asyncio
from functools import wraps
import inspect
import json
import logging
import math
import os
from pathlib import Path
import threading

SPECULATOR = "longspark_vllm.native_v2.speculator.LongSparkSpeculator"
EXTENSION = "longspark_vllm.native_v2.diagnostics.NativeV2Diagnostics"
MIDDLEWARE = "longspark_vllm.serving.LongSparkGreedyMiddleware"
COMPLETIONS = {"/v1/completions", "/v1/chat/completions"}
READ_ONLY_POSTS = {"/tokenize", "/detokenize"}
BODY_LIMIT = 16 * 1024 * 1024
AUDIT_TIMEOUT = 30.0


class UnsupportedRequest(ValueError):
    def __init__(self, message, param=None):
        super().__init__(message)
        self.param = param


def normalize_request(body):
    """Reject unsupported generation *before* any request reaches the engine."""
    if not isinstance(body, dict):
        raise UnsupportedRequest("The request must be a JSON object.")
    body = dict(body)
    fixed = {"temperature": 0.0, "top_p": 1.0, "min_p": 0.0,
             "presence_penalty": 0.0, "frequency_penalty": 0.0,
             "repetition_penalty": 1.0, "min_tokens": 0, "n": 1, "best_of": 1}
    for key, expected in fixed.items():
        value = body.get(key)
        if value is not None and (type(value) not in (int, float) or value != expected):
            raise UnsupportedRequest(f"LongSpark V2 requires {key}={expected}.", key)
    value = body.get("top_k")
    if value is not None and (type(value) is not int or value not in (-1, 0)):
        raise UnsupportedRequest("LongSpark V2 requires top_k=0 or -1.", "top_k")
    if body.get("use_beam_search") not in (None, False):
        raise UnsupportedRequest("LongSpark V2 does not support beam search.", "use_beam_search")
    for key in ("logit_bias", "allowed_token_ids", "bad_words", "logits_processors",
                "structured_outputs", "guided_json", "guided_regex", "guided_choice",
                "guided_grammar", "guided_decoding_backend", "guided_whitespace_pattern",
                "tools", "functions", "documents", "mm_processor_kwargs", "multi_modal_data"):
        if body.get(key) not in (None, [], {}):
            raise UnsupportedRequest(f"LongSpark V2 does not support {key}.", key)
    for key in ("tool_choice", "function_call"):
        if body.get(key) not in (None, "none"):
            raise UnsupportedRequest(f"LongSpark V2 does not support {key}.", key)
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise UnsupportedRequest("LongSpark V2 supports only text response_format.", "response_format")
    modalities = body.get("modalities")
    if modalities is not None and modalities != ["text"]:
        raise UnsupportedRequest("LongSpark V2 supports text input/output only.", "modalities")
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list) and any(
                        not isinstance(part, dict) or part.get("type") != "text"
                        for part in content):
                    raise UnsupportedRequest("LongSpark V2 supports text messages only.", "messages")
    # An explicit default also covers null and prevents model generation_config
    # defaults from accidentally enabling stochastic sampling.
    body["temperature"] = 0.0
    return body


def parser():
    p = argparse.ArgumentParser(description="Serve LongSpark with the native vLLM V2 OpenAI API.")
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--served-model-name", default="longspark-qwen3-4b")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="Server default, and maximum allowed output tokens per request.")
    p.add_argument("--kv-cache-gib", type=float, default=4.0)
    p.add_argument("--kernel-policy", choices=("invariant", "default"), default="invariant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--audit-dir", type=Path, required=True,
                   help="New directory for launch, prefill warmup and startup/shutdown graph evidence.")
    p.add_argument("--execute", action="store_true", help="Start the server; otherwise print a CUDA-free plan.")
    return p


def make_plan(args):
    if not 1 <= args.port <= 65535 or args.gpu < 0:
        raise ValueError("Require port 1..65535 and a nonnegative GPU index")
    if min(args.max_num_seqs, args.max_num_batched_tokens, args.max_model_len,
           args.max_new_tokens) < 1:
        raise ValueError("Batch, context and output token limits must be positive")
    if args.max_num_batched_tokens < args.max_num_seqs * 8:
        raise ValueError("max-num-batched-tokens must cover max-num-seqs * 8 verify tokens")
    if args.max_new_tokens > args.max_model_len:
        raise ValueError("max-new-tokens must not exceed max-model-len")
    if not math.isfinite(args.kv_cache_gib) or args.kv_cache_gib <= 0:
        raise ValueError("KV cache size must be finite and positive")
    if not args.host or not args.served_model_name:
        raise ValueError("Host and served model name must not be empty")
    options = ["serve", str(args.target.resolve()), "--served-model-name", args.served_model_name,
               "--host", args.host, "--port", str(args.port), "--api-server-count", "1",
               "--dtype", "bfloat16", "--tensor-parallel-size", "1",
               "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
               "--max-num-batched-tokens", str(args.max_num_batched_tokens),
               "--kv-cache-memory-bytes", str(int(args.kv_cache_gib * 1024**3)),
               "--no-enable-prefix-caching", "--enable-chunked-prefill", "--async-scheduling",
               "--seed", str(args.seed), "--generation-config", "auto",
               # Preserve model EOS IDs while overriding all HF sampling defaults.
               "--override-generation-config", json.dumps(dict(temperature=0.0, top_p=1.0,
                    top_k=0, min_p=0.0, repetition_penalty=1.0, max_new_tokens=args.max_new_tokens)),
               "--default-chat-template-kwargs", json.dumps(dict(enable_thinking=False)),
               "--attention-config", json.dumps(dict(backend="FLASH_ATTN")),
               "--compilation-config", json.dumps(dict(cudagraph_mode="FULL_AND_PIECEWISE")),
               "--cudagraph-metrics", "--worker-extension-cls", EXTENSION,
               "--speculative-config", json.dumps(dict(method="custom_class", model=SPECULATOR,
                                                        num_speculative_tokens=7, draft_sample_method="greedy")),
               "--additional-config", json.dumps(dict(native_v2_custom_class=True,
                                                     longspark=dict(draft_model=str(args.draft.resolve())))),
               "--middleware", MIDDLEWARE]
    env = dict(VLLM_USE_V2_MODEL_RUNNER="1", VLLM_USE_FLASHINFER_SAMPLER="0",
               VLLM_USE_RUST_FRONTEND="0",
               VLLM_BATCH_INVARIANT="1" if args.kernel_policy == "invariant" else "0",
               CUDA_VISIBLE_DEVICES=str(args.gpu),
               LONGSPARK_SERVE_AUDIT_DIR=str(args.audit_dir.resolve()),
               LONGSPARK_SERVE_PREFILL_WIDTH=str(min(args.max_num_batched_tokens, args.max_model_len)))
    return dict(schema_version=1, entrypoint="vllm.entrypoints.cli.main", argv=options,
                environment=env, api_key_source="VLLM_API_KEY environment (not recorded)",
                supported_generation_endpoints=sorted(COMPLETIONS),
                served_model_name=args.served_model_name,
                base_url=f"http://{args.host}:{args.port}/v1", kernel_policy=args.kernel_policy,
                audit_dir=str(args.audit_dir.resolve()))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class LongSparkGreedyMiddleware:
    """Guard requests without buffering native streaming responses.

    Startup runs before readiness. The native synchronous shutdown hook takes
    the final audit before workers exit; ASGI lifespan shutdown is too late in
    vLLM 0.30. No HTTP control or diagnostics route is added.
    """

    def __init__(self, app):
        self.app = app
        self.audit_dir = os.environ.get("LONGSPARK_SERVE_AUDIT_DIR")
        self.prefill_width = int(os.environ.get("LONGSPARK_SERVE_PREFILL_WIDTH", "512"))
        self.audit_started = False
        self.stopping = False

    async def _startup(self, scope):
        if not self.audit_dir:
            raise RuntimeError("Use serve_native_v2.py to configure LongSpark serving diagnostics")
        engine = scope["app"].state.engine_client
        warmup = await engine.collective_rpc("native_v2_warm_prefill_shapes", args=(self.prefill_width,))
        write_json(Path(self.audit_dir) / "prefill-warmup.json", warmup)
        audit = await engine.collective_rpc("native_v2_audit_start", kwargs={"label": "http_serving"})
        self.audit_started = True
        write_json(Path(self.audit_dir) / "startup-audit.json", audit)
        self._install_shutdown_hook(engine)

    async def _shutdown(self, engine):
        if self.audit_started:
            result = await engine.collective_rpc("native_v2_audit_end", timeout=AUDIT_TIMEOUT)
            write_json(Path(self.audit_dir) / "shutdown-audit.json", result)
            self.audit_started = False

    def _shutdown_error(self, error):
        # Do not attempt a second RPC after the native engine has exited.
        self.audit_started = False
        try:
            write_json(Path(self.audit_dir) / "shutdown-audit-error.json",
                       dict(error=f"{type(error).__name__}: {error}", graph_audit_complete=False))
        except Exception:
            logging.getLogger(__name__).exception("Could not save incomplete LongSpark shutdown audit")

    def _install_shutdown_hook(self, engine):
        original = engine.shutdown
        if inspect.iscoroutinefunction(original):
            raise RuntimeError("Expected vLLM 0.30 synchronous AsyncLLM.shutdown")
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()

        @wraps(original)
        def audited_shutdown(timeout=None):
            self.stopping = True
            if self.audit_started:
                try:
                    if threading.get_ident() == loop_thread:
                        # The native signal path calls from an executor thread.
                        # An exceptional same-thread teardown must never block
                        # its own event loop or pretend an audit was completed.
                        raise RuntimeError("Shutdown called on the API event-loop thread; final graph audit incomplete")
                    if loop.is_closed() or not loop.is_running():
                        raise RuntimeError("API event loop is unavailable; final graph audit incomplete")
                    future = asyncio.run_coroutine_threadsafe(self._shutdown(engine), loop)
                    try:
                        future.result(timeout=AUDIT_TIMEOUT)
                    except Exception:
                        future.cancel()
                        raise
                except Exception as error:
                    self._shutdown_error(error)
            # Preserve the native sync signature, timeout, return and errors.
            # Context-manager cleanup may call this again after the signal path.
            return original(timeout=timeout)

        engine.shutdown = audited_shutdown

    async def _lifespan(self, scope, receive, send):
        async def guarded_send(message):
            if message["type"] == "lifespan.startup.complete":
                await self._startup(scope)
            await send(message)

        async def guarded_receive():
            message = await receive()
            if message["type"] == "lifespan.shutdown":
                try:
                    await self._shutdown(scope["app"].state.engine_client)
                except Exception as error:
                    self._shutdown_error(error)
            return message

        await self.app(scope, guarded_receive, guarded_send)

    @staticmethod
    async def _error(send, message, *, param=None, status=400):
        body = json.dumps(dict(error=dict(message=message, type="invalid_request_error",
                                         param=param, code="unsupported_longspark_request"))).encode()
        await send(dict(type="http.response.start", status=status,
                        headers=[(b"content-type", b"application/json"),
                                 (b"content-length", str(len(body)).encode())]))
        await send(dict(type="http.response.body", body=body))

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self._lifespan(scope, receive, send)
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        path = scope["path"].rstrip("/")
        if scope["method"] == "POST" and path in READ_ONLY_POSTS:
            return await self.app(scope, receive, send)
        if scope["method"] != "POST" or path not in COMPLETIONS:
            return await self._error(send, "This LongSpark server supports /v1/chat/completions and /v1/completions.",
                                     status=404)
        if self.stopping:
            return await self._error(send, "The LongSpark server is shutting down.", status=503)
        headers = dict(scope.get("headers", []))
        if headers.get(b"content-encoding", b"identity").lower() != b"identity":
            return await self._error(send, "Compressed request bodies are unsupported.", status=415)
        chunks, size = [], 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            part = event.get("body", b"")
            size += len(part)
            if size > BODY_LIMIT:
                return await self._error(send, "Request body exceeds 16 MiB.", status=413)
            chunks.append(part)
            if not event.get("more_body", False):
                break
        try:
            body = normalize_request(json.loads(b"".join(chunks)))
            encoded = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
        except UnsupportedRequest as error:
            return await self._error(send, str(error), param=error.param)
        except (ValueError, UnicodeError):
            return await self._error(send, "Request body must contain valid finite JSON.")
        updated_scope = dict(scope)
        updated_scope["headers"] = [(key, value) for key, value in scope.get("headers", [])
                                    if key not in (b"content-length", b"transfer-encoding")]
        updated_scope["headers"].append((b"content-length", str(len(encoded)).encode()))
        delivered = False

        async def replay_body():
            nonlocal delivered
            if not delivered:
                delivered = True
                return dict(type="http.request", body=encoded, more_body=False)
            return await receive()

        if self.stopping:
            return await self._error(send, "The LongSpark server is shutting down.", status=503)
        return await self.app(updated_scope, replay_body, send)
