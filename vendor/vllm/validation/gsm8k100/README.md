# GSM8K test first 100

`inputs.jsonl` is the unchanged fixture used by the native ModelRunner V2
benchmark. It supplies token IDs directly; evaluation does not retokenize.

- Dataset: `openai/grade-school-math`, test split, original indices 0–99.
- Source revision: `3101c7d5072418e28b9008a6636bde82a006892c`.
- Input SHA256: `ba6199428e423c9ff69b18bfe64efb4bdeb09509797d3b3d2c4738b8aebd89ef`.
- Qwen3-4B tokenizer, non-thinking chat template, step-by-step prompt ending
  with a `#### numeric-answer` instruction; no demonstrations.
- `examples.jsonl` retains the source questions and expected answers.
- Metadata records selection, source digest, prompt and tokenizer details;
  `LICENSE.txt` retains the dataset license.

The fixture moved here during V2-only cleanup. Original benchmark manifests
retain the path used at execution time, with the same content digest.
