"""Target weight sharing and attention normalization for ModelRunner V2."""

import torch
from torch import nn


def _normalized(statistics):
    """Convert SpecForge online-softmax statistics to output and logsumexp."""
    visible = statistics.normalizer > 0
    lse = statistics.max_logits + statistics.normalizer.clamp_min(
        torch.finfo(statistics.normalizer.dtype).tiny
    ).log()
    return statistics.output(), lse.masked_fill(~visible, float("-inf"))


class _RMSNorm(nn.Module):
    def __init__(self, weight, epsilon):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = float(epsilon)

    def forward(self, hidden):
        dtype = hidden.dtype
        values = hidden.float()
        values = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return values.to(dtype=dtype) * self.weight


def _target_modules(target, device):
    """Share TP=1 weights locally; copy only three modules across devices.

    PyTorch wrappers deliberately avoid calling vLLM's distributed embedding
    implementation from the draft device.  Padded vocabulary rows are excluded
    from both sampling and the rejection-sampling distribution.
    """
    model = target.model
    hf = target.hf_config
    vocab_size = int(hf.vocab_size)
    hidden_size = int(hf.hidden_size)

    def weight(module, shape):
        value = module.weight.detach()
        if not value.is_floating_point() or value.ndim != len(shape):
            raise ValueError("LongSpark requires unquantized target module weights")
        if tuple(value.shape[1:]) != tuple(shape[1:]) or value.shape[0] < shape[0]:
            raise ValueError("LongSpark requires complete, unsharded target weights (TP=1)")
        # Tensor.to aliases storage on the same device and copies across devices.
        return value[: shape[0]].to(device=device)

    embed_weight = weight(model.model.embed_tokens, (vocab_size, hidden_size))
    head_weight = weight(model.lm_head, (vocab_size, hidden_size))
    norm_weight = weight(model.model.norm, (hidden_size,))
    embedding = nn.Embedding.from_pretrained(embed_weight, freeze=True)
    head = nn.Linear(hidden_size, vocab_size, bias=False, device="meta")
    head.weight = nn.Parameter(head_weight, requires_grad=False)
    norm = _RMSNorm(norm_weight, hf.rms_norm_eps)
    return embedding, head, norm
