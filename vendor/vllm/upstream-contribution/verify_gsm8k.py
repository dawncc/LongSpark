"""Reproduce the user-owned GSM8K-100 comparison against the upstream branch.

Run each method in a fresh process with upstream/.venv/bin/python. This helper
belongs to the contribution evidence, not the proposed vLLM source patch.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("ar", "longspark", "ngram_gpu"), required=True
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    import vllm

    rows = [json.loads(line) for line in args.input.read_text().splitlines()][
        : args.limit
    ]
    assert len(rows) == args.limit
    options = dict(
        model=args.target,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=args.batch,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=True,
        disable_log_stats=False,
        seed=0,
        kv_cache_memory_bytes=2 * 1024**3,
        enforce_eager=args.eager,
        attention_config={"backend": "FLASH_ATTN"},
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
    )
    if args.method == "longspark":
        options["speculative_config"] = dict(
            method="longspark",
            model=args.draft,
            num_speculative_tokens=7,
        )
    elif args.method == "ngram_gpu":
        options["speculative_config"] = dict(
            method="ngram_gpu",
            num_speculative_tokens=3,
            prompt_lookup_min=2,
            prompt_lookup_max=5,
        )
    if args.method != "ar":
        options["per_request_spec_decode_metrics"] = "detailed"
    llm = LLM(**options)
    assert llm.llm_engine.vllm_config.use_v2_model_runner
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    llm.generate(
        [{"prompt_token_ids": row["input_ids"]} for row in rows[: min(3, len(rows))]],
        params,
        use_tqdm=False,
    )
    records = []
    elapsed = 0.0
    for start in range(0, len(rows), args.batch):
        batch = rows[start : start + args.batch]
        begin = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": row["input_ids"]} for row in batch],
            params,
            use_tqdm=False,
        )
        wall_seconds = time.perf_counter() - begin
        elapsed += wall_seconds
        for row, output in zip(batch, outputs):
            assert list(output.prompt_token_ids) == row["input_ids"]
            candidate = output.outputs[0]
            metrics = output.metrics
            stats = getattr(candidate, "spec_decode_metrics", None)
            if stats is not None:
                stats = stats.to_dict()
            records.append(
                dict(
                    source_id=row["source_id"],
                    gold_answer=row["gold_answer"],
                    output_ids=list(candidate.token_ids),
                    output_text=candidate.text,
                    finish_reason=candidate.finish_reason,
                    stop_reason=candidate.stop_reason,
                    timing=dataclasses.asdict(metrics)
                    if dataclasses.is_dataclass(metrics)
                    else vars(metrics)
                    if metrics
                    else None,
                    spec_decode_metrics=stats,
                    batch_index=start // args.batch,
                    batch_wall_seconds=wall_seconds,
                )
            )
        print(f"{args.method}: {len(records)}/{len(rows)}", flush=True)
    source = Path(vllm.__file__).resolve().parents[1]
    payload = dict(
        method=args.method,
        options=options,
        upstream_base=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        vllm_file=vllm.__file__,
        vllm_version=vllm.__version__,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        env={
            name: os.environ.get(name)
            for name in (
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_GPU_SYNC_CHECK",
                "VLLM_BATCH_INVARIANT",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_USE_RUST_FRONTEND",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        wall_seconds=elapsed,
        output_tokens=sum(len(record["output_ids"]) for record in records),
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in ("records", "options")
            }
        )
    )


if __name__ == "__main__":
    main()
