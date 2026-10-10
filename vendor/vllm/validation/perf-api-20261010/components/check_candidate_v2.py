# Reproduction copy: import the current repository instead of the isolated profiling package.
import json,torch,copy
from longspark_vllm.native_v2.state import LongSparkState

def make(B):
 s=LongSparkState.__new__(LongSparkState);L,KV,HQ,D,H=5,8,32,128,2560
 s.device=torch.device('cuda:0');s.dtype=torch.bfloat16;s.max_num_reqs=6;s.num_layers=L;s.kv_heads=KV;s.head_dim=D;s.hidden_size=H
 s.query=torch.randn(1,L,HQ,16,D,device='cuda',dtype=torch.bfloat16)
 s.keys=torch.randn(7,L,256,KV,D,device='cuda',dtype=torch.bfloat16);s.values=torch.randn_like(s.keys)
 s.global_output=torch.randn(7,L,HQ,16,D,device='cuda');s.global_lse=torch.randn(7,L,HQ,16,device='cuda')
 s.hidden=torch.randn(7,L,H,device='cuda',dtype=torch.bfloat16);s.lengths=torch.zeros(7,device='cuda',dtype=torch.long)
 s.raw_offsets=torch.arange(256,device='cuda');s.layer_offsets=torch.arange(L,device='cuda')
 return s

def equal(a,b):
 for i,(x,y) in enumerate(zip(a.mutable_state(),b.mutable_state())):
  assert torch.equal(x.view(torch.uint8),y.view(torch.uint8)),(i,(x-y).abs().max().item())

def graph(run):
 st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(st):
  for _ in range(3):run()
 torch.cuda.current_stream().wait_stream(st)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=st):run()
 return g

def bench(g):
 for _ in range(100):g.replay()
 a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record()
 for _ in range(1000):g.replay()
 b.record();b.synchronize()
 return a.elapsed_time(b)/1000

results=[]
with torch.inference_mode():
 for B in (1,4):
  for width in (8,512):
   for boundary in (True,False):
    torch.manual_seed(321)
    ref=make(B);fast=copy.deepcopy(ref);ref.use_fused_commit=False
    k=torch.randn(B,5,width,8,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    h=torch.randn((B,5,2560) if boundary else (B,width,5,2560),device='cuda',dtype=torch.bfloat16)
    slots=torch.tensor([2,0,4,1][:B],device='cuda');starts=torch.zeros(B,dtype=torch.long,device='cuda');counts=torch.tensor([width,min(3,width),0,1][:B],device='cuda')
    funcs=[lambda s=s:s.update(slots,k,v,h,starts,counts) for s in (ref,fast)]
    for f in funcs:f()
    equal(ref,fast)
    # rejected NaNs, wraparound, reused slots, and zero-count rows.
    for step in range(4):
     curcounts=[3,0,7,1][:B]
     for row,c in enumerate(curcounts): k[row,:,c:]=float('nan');v[row,:,c:]=float('nan')
     counts.copy_(torch.tensor(curcounts,device='cuda'))
     starts.copy_(ref.lengths.index_select(0,slots))
     if step == 2:starts.zero_()
     for f in funcs:f()
     equal(ref,fast)
    counts.zero_();starts.fill_(987)
    old=tuple(t.clone() for t in fast.mutable_state())
    for f in funcs:f()
    equal(ref,fast)
    assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(old,fast.mutable_state()))
    # Warm capture uses resets, then real replays consume dynamic starts/counts.
    starts.zero_();counts.fill_(1)
    graphs=[graph(f) for f in funcs]
    for _ in range(3):
     starts.copy_(ref.lengths.index_select(0,slots))
     for g in graphs:g.replay()
     torch.cuda.synchronize();equal(ref,fast)
    starts.zero_()
    timings=[bench(g) for g in graphs]
    results.append({'batch':B,'width':width,'boundary_hidden':boundary,'bitwise_equal':True,'reference_graph_ms':timings[0],'fused_graph_ms':timings[1],'speedup':timings[0]/timings[1]})
    print(json.dumps(results[-1]),flush=True)
    del graphs,ref,fast,k,v,h,funcs
json.dump(results,open('/data/chenchen/codex/LongSpark/artifacts/perf-api-20261010/perf-stage/state-candidate-bench-v2.json','w'),indent=2)
