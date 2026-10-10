"""Batched LongSpark contracts and opt-in real capture without target loading."""

import os
from types import SimpleNamespace

import pytest
import torch

from torch import nn

from specforge.modeling.draft import DraftFreeKVConfig, DraftFreeKVModel
from specforge.modeling.draft.global16_raw256_training import DenseGlobal16Raw256DraftContext
from longspark_vllm.native_v2.state import LongSparkState


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(32)
    config = DraftFreeKVConfig(
        target_layer_ids=[0, 2], num_target_layers=3,
        block_size=7, hidden_size=16, num_key_value_heads=1, head_dim=8,
        vocab_size=32, adapter_rank=8, draft_mode="parallel_shifted_qwen",
        mask_token_id=0, parallel_mixer_layers=2, parallel_markov_rank=4,
        qwen_intermediate_size=32, qwen_num_attention_heads=2,
        qwen_gradient_checkpointing=False, global16_raw256_enabled=True,
    )
    model = DraftFreeKVModel(config).eval()
    # Nonzero correction makes this fixture exercise actual Markov feedback.
    with torch.no_grad():
        model.parallel_markov_head.markov_w1.weight.normal_(std=0.5)
        model.parallel_markov_head.markov_w2.weight.normal_(std=0.5)
    path = tmp_path_factory.mktemp("longspark-draft")
    model.save_pretrained(path)
    return path


@pytest.fixture
def target():
    return SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=nn.Embedding(40, 16), norm=nn.RMSNorm(16)),
            lm_head=nn.Linear(16, 40, bias=False),
        ),
        hf_config=SimpleNamespace(
            hidden_size=16, vocab_size=32, num_key_value_heads=1,
            num_attention_heads=2, head_dim=8, rms_norm_eps=1e-6,
        ),
        layer_ids=[0, 2], rope_theta=1_000_000.0, position_variant="native",
    )


@pytest.fixture
def native_target(target):
    target.hf_config.model_type = "qwen3"
    target.hf_config.num_hidden_layers = 3
    target.hf_config.rope_theta = 1_000_000.0
    target.model.model.embed_tokens.bfloat16()
    target.model.model.norm.bfloat16()
    target.model.lm_head.bfloat16()
    return target


@pytest.fixture
def state(checkpoint, native_target):
    return LongSparkState(checkpoint, native_target.model, native_target.hf_config, 4, "cpu")


def batch_rows(tokens, batch=2, device="cpu"):
    torch.manual_seed(181 + tokens)
    keys = torch.randn(batch, 2, tokens, 1, 8, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    hidden = torch.randn(batch, tokens, 2, 16, dtype=torch.bfloat16, device=device)
    return keys, values, hidden


def assert_request(state, slot, keys, values, hidden):
    length = keys.shape[1]
    assert state.lengths[slot].item() == length
    positions = torch.arange(max(0, length - 256), length, device=state.device)
    torch.testing.assert_close(state.keys[slot, :, positions % 256], keys[:, positions], atol=0, rtol=0)
    torch.testing.assert_close(state.values[slot, :, positions % 256], values[:, positions], atol=0, rtol=0)
    torch.testing.assert_close(state.hidden[slot], hidden[-1], atol=0, rtol=0)
    dense = state.reference.global_state_from_full_prefix(
        query=state.query,
        target_keys=keys.transpose(1, 2)[None],
        target_values=values.transpose(1, 2)[None],
        prefix_mask=torch.ones(1, length, dtype=torch.bool, device=state.device),
    )
    # BF16 contractions round before the FP32 online merge; changing chunk
    # boundaries legitimately adds BF16-level rounding, not FP32 state loss.
    torch.testing.assert_close(state.global_output[slot], dense.values()[0], atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(state.global_lse[slot],
                               (dense.max_logits + dense.normalizer.log())[0],
                               atol=2e-3, rtol=2e-3)
    assert state.global_output.dtype == state.global_lse.dtype == torch.float32


@pytest.mark.parametrize("counts", [(1, 3), (255, 256), (257, 511), (512, 512)])
def test_batched_prefill_masks_rejected_nan_and_keeps_only_recent_rows(state, counts):
    data = batch_rows(512)
    for row, count in enumerate(counts):
        data[0][row, :, count:] = float("nan")
        data[1][row, :, count:] = float("nan")
        data[2][row, count:] = float("nan")
    before_bytes = state.state_bytes
    state.update(torch.tensor([2, 0]), *data, torch.tensor([0, 0]), torch.tensor(counts))
    for row, (slot, count) in enumerate(zip((2, 0), counts)):
        assert_request(state, slot, data[0][row, :, :count], data[1][row, :, :count], data[2][row, :count])
    assert state.lengths.tolist() == [counts[1], 0, counts[0], 0, 0]
    assert state.state_bytes == before_bytes


def test_mixed_decode_acceptance_reordered_slots_and_reset(state):
    prefix = batch_rows(255)
    state.update(torch.tensor([0, 2]), *prefix, torch.tensor([0, 0]), torch.tensor([254, 255]))
    delta = batch_rows(8)
    for row, accepted in enumerate((3, 1)):
        delta[0][row, :, accepted:] = float("nan")
        delta[1][row, :, accepted:] = float("nan")
        delta[2][row, accepted:] = float("nan")
    state.update(torch.tensor([2, 0]), *delta, torch.tensor([255, 254]), torch.tensor([3, 1]))
    for row, (slot, before_row, before_count, accepted) in enumerate(((2, 1, 255, 3), (0, 0, 254, 1))):
        keys = torch.cat((prefix[0][before_row, :, :before_count], delta[0][row, :, :accepted]), dim=1)
        values = torch.cat((prefix[1][before_row, :, :before_count], delta[1][row, :, :accepted]), dim=1)
        hidden = torch.cat((prefix[2][before_row, :before_count], delta[2][row, :accepted]))
        assert_request(state, slot, keys, values, hidden)
    untouched = tuple(tensor[0].clone() for tensor in state.mutable_state())
    replacement = batch_rows(8, batch=1)
    state.update(torch.tensor([2]), *replacement, torch.tensor([0]), torch.tensor([2]))
    assert_request(state, 2, replacement[0][0, :, :2], replacement[1][0, :, :2], replacement[2][0, :2])
    for tensor, old in zip(state.mutable_state(), untouched):
        torch.testing.assert_close(tensor[0], old, atol=0, rtol=0)


def test_boundary_hidden_zero_count_and_padding_are_inert(state):
    data = batch_rows(8, batch=4)
    slots = torch.tensor([0, 2, -1, -1])
    starts = torch.zeros(4, dtype=torch.long)
    state.update(slots, *data, starts, torch.tensor([8, 3, 0, 0]))
    snapshot = tuple(tensor.clone() for tensor in state.mutable_state())
    boundary = data[2][:, 0]
    for starts in (torch.tensor([8, 3, 0, 0]), torch.zeros(4, dtype=torch.long),
                   torch.ones(4, dtype=torch.long)):
        state.update(slots, data[0], data[1], boundary, starts,
                     torch.zeros(4, dtype=torch.long))
        for tensor, old in zip(state.mutable_state(), snapshot):
            torch.testing.assert_close(tensor, old, atol=0, rtol=0)
    tokens = state.propose(slots, torch.tensor([1, 2, -99, -99]))
    assert tokens.shape == (4, 7)
    assert torch.equal(tokens[2:], torch.zeros_like(tokens[2:]))


def test_positive_dummy_slots_profile_without_mutating_live_state(state):
    data = batch_rows(8)
    state.update(torch.tensor([0, 1]), *data, torch.tensor([0, 0]), torch.tensor([3, 8]))
    before = tuple(tensor.clone() for tensor in state.mutable_state())
    state.update(torch.tensor([2, 3]), *data, torch.tensor([1, 1]), torch.tensor([0, 0]))
    tokens = state.propose(torch.tensor([2, 3]), torch.tensor([0, 0]))
    assert tokens.shape == (2, 7)
    for tensor, old in zip(state.mutable_state(), before):
        torch.testing.assert_close(tensor, old, atol=0, rtol=0)


def test_boundary_hidden_interface_updates_correct_request(state):
    data = batch_rows(8)
    boundary = torch.stack([data[2][0, 2], data[2][1, 7]])
    state.update(torch.tensor([3, 1]), data[0], data[1], boundary,
                 torch.tensor([0, 0]), torch.tensor([3, 8]))
    torch.testing.assert_close(state.hidden[3], boundary[0], atol=0, rtol=0)
    torch.testing.assert_close(state.hidden[1], boundary[1], atol=0, rtol=0)


def test_batched_proposals_match_full_prefix_training_reference(state):
    data = batch_rows(270)
    state.update(torch.tensor([3, 1]), *data, torch.tensor([0, 0]), torch.tensor([270, 257]))
    before = tuple(tensor.clone() for tensor in state.mutable_state())
    actual = state.propose(torch.tensor([1, 3]), torch.tensor([2, 1]))
    # The training reference attends directly to all confirmed rows and selects
    # Raw256 by a logical prefix mask, independently of the serving slot/ring.
    for out_row, (data_row, count, anchor) in enumerate(((1, 257, 2), (0, 270, 1))):
        context = DenseGlobal16Raw256DraftContext(
            reference=state.reference,
            keys=data[0][data_row:data_row + 1, :, :count].transpose(2, 3),
            values=data[1][data_row:data_row + 1, :, :count].transpose(2, 3),
            anchor_attention_mask=torch.ones(1, count, dtype=torch.bool),
            source_indices=torch.tensor([0]),
        )
        with torch.inference_mode():
            logits = state.model(
                current_hidden_states=data[2][data_row:data_row + 1, count - 1],
                current_token_ids=torch.tensor([anchor]),
                current_position_ids=torch.tensor([count]),
                target_final_norm=state.norm, target_embed_tokens=state.embedding,
                target_lm_head=state.head, global16_paged_context=context,
                apply_markov_correction=False,
            )
            expected = [torch.tensor([anchor])]
            for step in range(7):
                bias = state.model.parallel_markov_head.compute_bias(expected[-1])
                expected.append((logits[:, step] + bias.to(logits.dtype)).argmax(-1))
        assert torch.equal(actual[out_row], torch.cat(expected[1:]))
    for tensor, old in zip(state.mutable_state(), before):
        torch.testing.assert_close(tensor, old, atol=0, rtol=0)


def test_markov_correction_uses_previous_realized_token(state):
    data = batch_rows(1)
    state.update(torch.tensor([0, 1]), *data, torch.tensor([0, 0]), torch.tensor([1, 1]))
    calls = []

    class Markov:
        def compute_bias(self, previous):
            calls.append(previous.clone())
            return torch.nn.functional.one_hot((previous + 1) % 32, num_classes=32).float() * 100

    class Model:
        parallel_markov_head = Markov()

        def __call__(self, **kwargs):
            return torch.zeros(2, 7, 32)

    state.model = Model()
    result = state.propose(torch.tensor([0, 1]), torch.tensor([1, 11]))
    expected = torch.tensor([[2, 3, 4, 5, 6, 7, 8], [12, 13, 14, 15, 16, 17, 18]])
    assert torch.equal(result, expected)
    assert torch.equal(torch.stack(calls[1:], dim=1), result[:, :-1])


def test_tensor_programs_do_not_read_host_scalars(state, monkeypatch):
    data = batch_rows(8)

    def reject(*args, **kwargs):
        raise AssertionError("captured tensor program attempted a host read")

    with monkeypatch.context() as patch:
        for name in ("item", "tolist", "cpu", "__bool__"):
            patch.setattr(torch.Tensor, name, reject)
        slots, starts, counts = torch.tensor([0, 2]), torch.tensor([0, 0]), torch.tensor([1, 3])
        state.update(slots, *data, starts, counts)
        state.propose(slots, torch.tensor([1, 2]))


@pytest.mark.parametrize("setting,value,match", [
    ("model_type", "qwen2", "dense Qwen3"),
    ("rope_scaling", {"rope_type": "linear", "factor": 2}, "native, unscaled"),
    ("quantization_config", {"quant_method": "awq"}, "unquantized"),
])
def test_rejects_unsupported_targets(checkpoint, native_target, setting, value, match):
    setattr(native_target.hf_config, setting, value)
    with pytest.raises(ValueError, match=match):
        LongSparkState(checkpoint, native_target.model, native_target.hf_config, 2, "cpu")


def test_rejects_fp32_target(checkpoint, target):
    target.hf_config.model_type = "qwen3"
    with pytest.raises(ValueError, match="BF16"):
        LongSparkState(checkpoint, target.model, target.hf_config, 2, "cpu")


def test_rejects_discontinuous_commit_and_invalid_input_shape(state):
    data = batch_rows(8)
    with pytest.raises(RuntimeError, match="extend its confirmed prefix"):
        state.update(torch.tensor([0, 1]), *data, torch.tensor([2, 2]), torch.tensor([1, 1]))
    with pytest.raises(ValueError, match="int32/int64"):
        state.update(torch.tensor([0., 1.]), *data, torch.tensor([0, 0]), torch.tensor([1, 1]))


@pytest.mark.skipif(os.environ.get("LONGSPARK_TEST_CUDA") != "1",
                    reason="Set LONGSPARK_TEST_CUDA=1 for actual batched CUDA capture/replay")
@pytest.mark.parametrize("width", [8, 512])
def test_real_batched_cuda_graph_dynamic_counts_and_reordered_slots(checkpoint, native_target, monkeypatch, width):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance
    init_batch_invariance()
    native_target.model.model.embed_tokens.cuda()
    native_target.model.model.norm.cuda()
    native_target.model.lm_head.cuda()
    state = LongSparkState(checkpoint, native_target.model, native_target.hf_config, 4, "cuda:0")
    slots = torch.tensor([0, 2], device="cuda")
    starts = torch.zeros(2, dtype=torch.long, device="cuda")
    counts = torch.tensor([1, 3], device="cuda")
    anchors = torch.tensor([1, 2], device="cuda")
    data = batch_rows(width, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            state.update(slots, *data, starts, counts)
            state.propose(slots, anchors)
        state.reset()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        state.update(slots, *data, starts, counts)
        result = state.propose(slots, anchors)
    state.reset()
    graph.replay()
    torch.cuda.synchronize()
    assert state.lengths.tolist() == [1, 0, 3, 0, 0]
    for row, (slot, count) in enumerate(((0, 1), (2, 3))):
        assert_request(state, slot, data[0][row, :, :count], data[1][row, :, :count], data[2][row, :count])
    slots.copy_(torch.tensor([2, 0], device="cuda"))
    starts.copy_(torch.tensor([3, 1], device="cuda"))
    counts.copy_(torch.tensor([8, 1], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert state.lengths.tolist() == [2, 0, 11, 0, 0]
    expected = state.propose(slots, anchors)
    assert torch.equal(result, expected)
    # Same captured graph must honor zero-start request reuse and changed count.
    starts.zero_()
    counts.copy_(torch.tensor([3, 1], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    for row, (slot, count) in enumerate(((2, 3), (0, 1))):
        assert_request(state, slot, data[0][row, :, :count], data[1][row, :, :count], data[2][row, :count])
    before = tuple(tensor.clone() for tensor in state.mutable_state())
    counts.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for tensor, old in zip(state.mutable_state(), before):
        torch.testing.assert_close(tensor, old, atol=0, rtol=0)


def test_target_weights_shared_without_padded_vocabulary(state, native_target):
    assert state.embedding.weight.data_ptr() == native_target.model.model.embed_tokens.weight.data_ptr()
    assert state.head.weight.data_ptr() == native_target.model.lm_head.weight.data_ptr()
    assert state.norm.weight.data_ptr() == native_target.model.model.norm.weight.data_ptr()
    assert state.embedding.num_embeddings == state.head.weight.shape[0] == 32


def test_native_rotary_frequencies_remain_fp32(state):
    expected = 1.0 / (state.rope_theta ** (torch.arange(0, 8, 2).float() / 8))
    for layer in state.model.parallel_qwen_layers:
        assert layer.self_attn.inv_freq.dtype == torch.float32
        torch.testing.assert_close(layer.self_attn.inv_freq, expected, rtol=0, atol=0)
