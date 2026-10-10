# Standalone fused target-context gather

On rl10's physical GPU 1 (NVIDIA L20X), the independent Triton gather passed
**10 GPU tests** in **9.23 s**. The reference is
`longspark_vllm.native_v2.inputs.gather_target_context`. Comparisons require
exact equality, including all five selected layers and the final-normalized
hidden state. Coverage includes widths 1/8 and longer prefills, multiple
requests, different page sizes and physical page mappings, noncontiguous cache
and input strides, rejected NaNs/nonresident slots, zero accepted rows, output
storage isolation, actual CUDA graph replay with changed GPU rejection counts,
and a subprocess proving invalid-slot assertions work with `TRITON_DEBUG=0`.

Reproduce using the isolated vLLM environment:

```bash
CUDA_VISIBLE_DEVICES=1 LONGSPARK_TEST_CUDA=1 PYTHONPATH=vendor/vllm:vendor \
  vendor/vllm/.venv/bin/python -m pytest \
  vendor/vllm/tests/test_native_v2_gather_gpu.py -q
```

`microbenchmark-v1.json` records a **standalone gather microbenchmark**, not
end-to-end decoding. It used `triton.testing.do_bench(warmup=10, rep=40)` on the
test fixture's five BF16 caches, eight KV heads, head dimension 128, and hidden
size 2560. Batches 1/2 and widths 1/8 were each checked for exact equivalence
before timing. The measured reference was 0.700–0.753 ms and fused gather
0.0443–0.0456 ms, approximately 15.7–16.9x faster for this component.

The native speculator uses this fused kernel for decode context gathering;
variable prefill shapes retain the reference path. Current cleanup regression
results are recorded in [the V2-only report](../../v2-only-cleanup/README.md).
