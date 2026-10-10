# Reproduction copy: import the current repository instead of the isolated profiling package.
import json,torch,types,time
from safetensors import safe_open
from torch import nn
from longspark_vllm.native_v2.state import LongSparkState
root='/data/lpf/models/Qwen3-4B'
idx=json.load(open(root+'/model.safetensors.index.json'))['weight_map']
weights={}
for name in ['model.embed_tokens.weight','model.norm.weight','lm_head.weight']:
 source=name if name in idx else 'model.embed_tokens.weight'
 with safe_open(root+'/'+idx[source],framework='pt',device='cuda') as f:weights[name]=f.get_tensor(source)
 def mod(w):
  m=nn.Module();m.weight=nn.Parameter(w,requires_grad=False);return m
hf=types.SimpleNamespace(**json.load(open(root+'/config.json')))
target=types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=mod(weights['model.embed_tokens.weight']),norm=mod(weights['model.norm.weight'])),lm_head=mod(weights['lm_head.weight']))
s=LongSparkState('/data/chenchen/codex/LongSpark/models/4B/longspark',target,hf,4,'cuda:0')
slot=torch.tensor([0],device='cuda');start=torch.tensor([0],device='cuda');count=torch.tensor([8],device='cuda');anchor=torch.tensor([1],device='cuda')
k=torch.randn(1,5,8,8,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);h=torch.randn(1,5,2560,device='cuda',dtype=torch.bfloat16)
s.update(slot,k,v,h,start,count)
def run():return s.propose(slot,anchor)
with torch.inference_mode():
 for _ in range(20):run()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  for _ in range(10):run()
 torch.cuda.synchronize()
 print(p.key_averages().table(sort_by='self_device_time_total',row_limit=40))
 p.export_chrome_trace('/data/chenchen/codex/LongSpark/artifacts/perf-api-20261010/perf-stage/propose-profile.json')
 st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(st):
  for _ in range(3):run()
 torch.cuda.current_stream().wait_stream(st);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=st):run()
 for _ in range(50):g.replay()
 a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record()
 for _ in range(500):g.replay()
 b.record();b.synchronize()
 print(json.dumps({'propose_graph_ms':a.elapsed_time(b)/500}))
