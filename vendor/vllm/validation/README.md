# Native ModelRunner V2 validation

This directory contains only the native LongSpark V2 input fixture, final
GSM8K first-100 evidence, and regression checks for the V2-only cleanup.

- [GSM8K fixture](gsm8k100/README.md): the unchanged official test first 100,
  tokenized with the target Qwen3 tokenizer and non-thinking chat template.
- [Native V2 benchmark](native-v2-gsm8k100-20260930/README.md): 100/100 exact
  AR/LongSpark output matches, 96% accuracy for both, acceptance and graph audits.
- [Cleanup validation](v2-only-cleanup/README.md): checks for the final code layout.

Raw benchmark plans and manifests preserve the original paths and source hashes
recorded when they ran. Historical hashes are not rewritten to claim that a
later refactor was benchmarked. Use the current fixture path when rerunning:

```bash
python vendor/vllm/scripts/audit_native_v2_gsm8k.py \
  vendor/vllm/validation/native-v2-gsm8k100-20260930/native-c1-full-v2 \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl --expect-limit 100
```
