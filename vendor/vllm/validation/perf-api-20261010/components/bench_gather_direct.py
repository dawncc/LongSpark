import torch,json,time
from test_native_v2_gather_gpu import make_case
results=[]
for widths in ((1,),(8,),(8,3,1,7)):
 args,ref,fused=make_case(widths,tuple(0 for _ in widths),dim=128,heads=8,hidden=2560)
 B=len(widths);buffers=(torch.empty(B,5,8,8,128,device='cuda',dtype=torch.bfloat16),torch.empty(B,5,8,8,128,device='cuda',dtype=torch.bfloat16),torch.empty(B,5,2560,device='cuda',dtype=torch.bfloat16),torch.empty(B,device='cuda',dtype=torch.long),torch.empty(B,device='cuda',dtype=torch.long))
 def old():
  k,v,h,s,c,w=fused(*args)
  buffers[0].zero_();buffers[1].zero_();buffers[0][:,:,:w].copy_(k);buffers[1][:,:,:w].copy_(v)
  for dst,src in zip(buffers[2:],(h,s,c)):dst.copy_(src)
 def new():fused(*args,output_buffers=buffers)
 timings={}
 for name,fn in [('staged',old),('direct',new)]:
  for _ in range(20):fn()
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):fn()
  torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):fn()
  for _ in range(100):g.replay()
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  a.record()
  for _ in range(1000):g.replay()
  b.record();b.synchronize();timings[name+'_gpu_graph_ms']=a.elapsed_time(b)/1000
  torch.cuda.synchronize();t=time.perf_counter()
  for _ in range(1000):fn()
  torch.cuda.synchronize();timings[name+'_eager_wall_ms']=(time.perf_counter()-t)*1000/1000
 results.append({'widths':widths,**timings});print(json.dumps(results[-1]),flush=True)
json.dump(results,open('/data/chenchen/codex/LongSpark/artifacts/perf-api-20261010/perf-stage/direct-gather-bench.json','w'),indent=2)
