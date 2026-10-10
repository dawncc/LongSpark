import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from safetensors import safe_open

UPSTREAM = Path('/data/chenchen/codex/vllm-longspark-upstream')
LEGACY = Path('/data/chenchen/codex/LongSpark')
sys.path.insert(0, str(LEGACY / 'vendor'))
sys.path.insert(0, str(LEGACY / 'vendor/vllm'))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

from longspark_vllm.native_v2.state import LongSparkState as LegacyState, _BatchContext as LegacyContext

NewModel = module('upstream_model_under_test', UPSTREAM / 'vllm/model_executor/models/longspark.py').LongSparkDraftModel
from vllm.v1.worker.gpu.spec_decode.longspark.state import LongSparkState as NewState, _BatchContext as NewContext

device = torch.device('cuda:0')
target_path = Path('/data/lpf/models/Qwen3-4B')
hf = SimpleNamespace(**json.loads((target_path / 'config.json').read_text()))
index = json.loads((target_path / 'model.safetensors.index.json').read_text())['weight_map']
weights = {}
for name in ['model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight']:
    if name == 'lm_head.weight' and name not in index and hf.tie_word_embeddings:
        weights[name] = weights['model.embed_tokens.weight']
        continue
    with safe_open(str(target_path / index[name]), framework='pt', device='cpu') as source:
        weights[name] = source.get_tensor(name).to(device=device, dtype=torch.bfloat16)
embed = nn.Embedding.from_pretrained(weights['model.embed_tokens.weight'], freeze=True)
head = nn.Linear(hf.hidden_size, hf.vocab_size, bias=False, device='meta')
head.weight = nn.Parameter(weights['lm_head.weight'], requires_grad=False)
norm = nn.RMSNorm(hf.hidden_size, eps=hf.rms_norm_eps, device='meta')
norm.weight = nn.Parameter(weights['model.norm.weight'], requires_grad=False)
target = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed, norm=norm), lm_head=head)

old = LegacyState(str(LEGACY / 'models/4B/longspark'), target, hf, 8, device)
cfg = SimpleNamespace(speculative_config=SimpleNamespace(method='longspark', draft_model_config=SimpleNamespace(hf_config=old.model.config)))
with torch.device(device):
    new_model = NewModel(vllm_config=cfg).to(dtype=torch.bfloat16).eval()
new_model.load_weights(old.model.state_dict().items())
new = NewState(new_model, target, hf, 8, device)

results = []
torch.manual_seed(4242)

def exact_state(label):
    equal = [torch.equal(left.view(torch.uint8), right.view(torch.uint8)) for left, right in zip(old.mutable_state(), new.mutable_state())]
    assert all(equal), (label, equal)
    return equal

@torch.inference_mode()
def compare(label, slots, counts, width, starts=None):
    batch = len(slots)
    idx = torch.tensor(slots, device=device)
    counts = torch.tensor(counts, device=device)
    if starts is None:
        starts = old.lengths.index_select(0, idx.clamp_min(0))
    else:
        starts = torch.tensor(starts, device=device)
    shape = (batch, old.num_layers, width, old.kv_heads, old.head_dim)
    keys = torch.randn(shape, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    hidden = torch.randn(batch, width, old.num_layers, old.hidden_size, device=device, dtype=torch.bfloat16)
    invalid = torch.arange(width, device=device)[None] >= counts[:, None]
    keys.masked_fill_(invalid[:, None, :, None, None], float('nan'))
    values.masked_fill_(invalid[:, None, :, None, None], float('nan'))
    hidden.masked_fill_(invalid[:, :, None, None], float('nan'))
    for state in [old, new]:
        state.update(idx, keys, values, hidden, starts, counts)
    state_equal = exact_state(label)
    anchors = torch.arange(10, 10 + batch, device=device)
    mapped, active = old._slots(idx)
    kwargs = dict(current_hidden_states=old.hidden.index_select(0, mapped), current_token_ids=anchors,
                  current_position_ids=old.lengths.index_select(0, mapped), target_final_norm=old.norm,
                  target_embed_tokens=old.embedding, target_lm_head=old.head, apply_markov_correction=False)
    old_logits = old.model(**kwargs, global16_paged_context=LegacyContext(old, mapped))
    new_logits = new.model(**kwargs, global16_paged_context=NewContext(new, mapped))
    logits_equal = torch.equal(old_logits.view(torch.uint8), new_logits.view(torch.uint8))
    old_tokens, new_tokens = old.propose(idx, anchors), new.propose(idx, anchors)
    tokens_equal = torch.equal(old_tokens, new_tokens)
    max_error = float((old_logits - new_logits).abs().max().item())
    results.append(dict(label=label, batch=batch, width=width, counts=counts.tolist(), state_equal=state_equal,
                        logits_bitwise_equal=logits_equal, max_abs_logit_error=max_error, proposal_equal=tokens_equal))
    print(json.dumps(results[-1]), flush=True)
    assert logits_equal and tokens_equal, results[-1]

compare('prefill-b1-8', [0], [8], 8, starts=[0])
compare('prefill-b4-512-mixed', [0, 1, 2, 3], [512, 257, 1, 0], 512, starts=[0, 0, 0, 0])
compare('decode-reorder-noop-padding', [2, 0, 1, -1], [1, 8, 0, 0], 8)
compare('reset-reuse', [1, 3, 0, 2], [7, 8, 0, 5], 8, starts=[0, 0, 512+8, 2])
files = [UPSTREAM / 'vllm/model_executor/models/longspark.py'] + sorted((UPSTREAM / 'vllm/v1/worker/gpu/spec_decode/longspark').glob('*.py'))
report = dict(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(), results=results,
              checkpoint=str(LEGACY / 'models/4B/longspark'), source_sha256={str(p.relative_to(UPSTREAM)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(UPSTREAM / 'artifacts/model-parity/result.json').write_text(json.dumps(report, indent=2)+'\n')
