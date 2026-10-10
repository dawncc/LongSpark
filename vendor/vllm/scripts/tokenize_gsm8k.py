#!/usr/bin/env python3
"""Create a shared, nonthinking Qwen3 fixture without exposing reference answers."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".metadata.json")
    if args.output.exists() or metadata_path.exists():
        parser.error("Refusing to overwrite existing fixture evidence")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    options = dict(add_generation_prompt=True, enable_thinking=False)
    rows = [json.loads(line) for line in args.examples.read_text().splitlines()]
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **options)["input_ids"]
        assert ids == tokenizer.encode(rendered, add_special_tokens=False)
        assert rendered.endswith("<think>\n\n</think>\n\n")
        row.update(rendered_prompt=rendered, input_ids=ids)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    metadata = json.loads(args.examples.with_suffix(args.examples.suffix + ".metadata.json").read_text())
    lengths = [len(row["input_ids"]) for row in rows]
    metadata.update(
        tokenizer=str(args.target.resolve()),
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        chat_template_options={**options, "return_dict": True},
        input_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        prompt_tokens=dict(min=min(lengths), max=max(lengths), mean=sum(lengths) / len(lengths)),
        target_metadata_sha256={name: hashlib.sha256((args.target / name).read_bytes()).hexdigest()
                                for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json")},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "input_sha256": metadata["input_sha256"], "prompt_tokens": metadata["prompt_tokens"]}))


if __name__ == "__main__":
    main()
