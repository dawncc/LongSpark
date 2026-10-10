# Native ModelRunner V2 extension

This small, version-specific source patch lets an explicitly opted-in
`custom_class` speculator use vLLM 0.30.0's native GPU ModelRunner V2. The existing
V1 custom proposer contract remains the default. LongSpark's model, bounded
state, and draft sampling code live in `longspark_vllm.native_v2`; they are not
implemented in this patch.

The patch changes five upstream files:

| File | Change |
| --- | --- |
| `vllm/config/vllm.py` | Allow explicitly opted-in V2 custom speculators, parallel drafting and async scheduling; reject use of that opt-in with V1 or pipeline parallelism. |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | Import the configured class and require a V2 `DraftModelSpeculator` subclass constructed with `(vllm_config, device)`. |
| `vllm/v1/worker/gpu/spec_decode/speculator.py` | Add optional `get_aux_hidden_state_layers()` declaration. |
| `vllm/v1/worker/gpu/model_runner.py` | Read that declaration before loading the target and enable native auxiliary outputs. |
| `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` | Accept explicit capture indices without changing the target or draft Hugging Face config. |

The native scheduler, target model execution, paged KV allocation, target CUDA
graphs, GPU rejection sampler and engine output path are retained. The
LongSpark implementation is exclusively the native V2 extension described in
[the implementation guide](../longspark_vllm/native_v2/README.md).

## Install and rollback

Stop the isolated vLLM environment's workers before applying or rolling back.
Restart workers after changing the files; an existing Python process can retain
the old imported classes. Use the Python executable from that environment:

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --check
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --apply
# Restore the original, verified source files when needed:
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --rollback
```

The default operation is `--check`. The installer uses package metadata without
importing vLLM or initializing CUDA. It accepts only versions `0.30.0` and
`0.30.0+cu129` and the exact original or patched file hashes in
`native-v2-custom-class.manifest.json`. A different source build, mixed patch
state or later local edit stops the operation before any write. There is no
force mode or fuzzy matching. A complete rollback copy is retained in
`site-packages/.longspark-native-v2-patch/longspark-native-v2-custom-class-v1/`.
Rollback also checks those backups and refuses to overwrite later user edits.

For local source review, `--root /path/to/checkout --version 0.30.0 --check`
checks a directory containing `vllm/`. `--version` in this mode does not bypass
the exact source hashes. No dependencies are installed by this tool.

## Extension contract

The intended native engine configuration includes:

```python
speculative_config = {
    "method": "custom_class",
    "model": "longspark_vllm.native_v2.speculator.LongSparkSpeculator",
    "num_speculative_tokens": 7,
    "draft_sample_method": "greedy",
}
additional_config = {
    "native_v2_custom_class": True,
    "longspark": {"draft_model": "/path/to/longspark/checkpoint"},
}
```

The drafter predicts seven tokens in parallel internally. The LongSpark runner
leaves `parallel_drafting` unset because its bounded state does not require
additional target query slots.

Set `VLLM_USE_V2_MODEL_RUNNER=1` before creating the engine. Native graph execution
must be configured and verified through the native runner; `enforce_eager=True`
would disable that native graph path. The custom extension can use async
scheduling, subject to the existing executor and padded-batch checks. LongSpark
must independently validate the model, attention backend, precision, parallel
configuration and features its implementation actually supports.

`get_aux_hidden_state_layers() -> tuple[int, ...] | None` is called before target
loading. It must not use CUDA. `None` preserves the upstream built-in policy;
an empty tuple requests no auxiliary states. Other tuples use the target's
`SupportsEagle3` capture numbering: decoder layer `i` is capture index `i + 1`.
The already normalized final target hidden state is passed separately to
`propose`, so LongSpark must use that value for its last selected layer, rather
than the pre-normalization auxiliary capture.

The native `propose` contract supplies a packed `InputBatch`, native attention
metadata, per-layer physical slot mappings, target hidden states, and GPU
accepted/rejected counts. Persistent state must follow request slot identity,
handle reuse/preemption, and commit only accepted target rows. LongSpark
disables prefix caching because it needs every prompt row to construct its
hidden-state-dependent summary.
Natural EOS, output caps, target cache rollback and proposal scheduling remain
the engine's responsibility.

## Verification scope

The patch installer has CPU tests for exact context and hash checks, version
rejection, interrupted-install rollback, idempotency, backup tampering and
preservation of subsequent user edits. Local review also applied all five hunks
to an isolated copy of v0.30.0, parsed the resulting Python files, checked the
legacy/opt-in configuration gates and verified byte-identical rollback.
These checks do not establish GPU correctness or performance of a custom
speculator; that requires separate native engine tests and reports.
