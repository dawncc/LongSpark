import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
server=root/'api-server-v3'
startup=json.loads((server/'startup-audit.json').read_text())
shutdown=json.loads((server/'shutdown-audit.json').read_text())
smoke=json.loads((root/'api-smoke-v3/summary.json').read_text())
assert len(startup)==len(shutdown)==1
entry=shutdown[0]
before,after,target,delta=(entry[key] for key in ('before','after','target','speculator_delta'))
assert before==startup[0]['before']
assert before['runner_class']=='vllm.v1.worker.gpu.model_runner.GPUModelRunner'
assert before['use_v2_model_runner'] and before['async_scheduling']
assert before['target']['manager_class']=='vllm.v1.worker.gpu.cudagraph_utils.ModelCudaGraphManager'
assert before['speculator']['cuda_graph_manager']=='vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager'
assert before['target']['captures']>0 and before['speculator']['captures']>0
assert target['replays']>0 and target['captures_delta']==0 and not target['manager_replaced']
assert target['max_decode_requests']>=2
assert sum(target['replay_shapes'].values())==target['replays']
assert all(name=='uniform_decode:FULL' for name,count in target['dispatches'].items() if name.startswith('uniform_decode:') and count)
assert delta['replays']>0 and delta['captures']==delta['fallbacks']==0
assert after['speculator']['startup_eager_calls']==before['speculator']['startup_eager_calls']
assert after['speculator']['replays']-before['speculator']['replays']==delta['replays']
assert not (server/'shutdown-audit-error.json').exists()
assert smoke['passed'] and smoke['baseline_exact_token_parity'] and smoke['json_sse_chat_completion_exact_parity']
assert smoke['successful_generations']==17 and smoke['correct_generations']==17
assert smoke['max_client_overlap']==4 and smoke['max_observed_server_running']>=2
assert len(smoke['rejection_cases'])==10 and all(case['status']==400 for case in smoke['rejection_cases'])
logpath=root/'api-server-v3.log'
lines=logpath.read_text().splitlines()
start=next(i for i,line in enumerate(lines) if 'Application startup complete.' in line)
end=next(i for i,line in enumerate(lines[start:],start) if '[shutdown] API server: shutdown triggered' in line)
jit=[dict(line=i+1,text=line) for i,line in enumerate(lines[start:end+1],start)
     if 'jit_monitor.py' in line and 'JIT compilation during inference' in line]
result=dict(passed=True,api_smoke='api-smoke-v3/summary.json',
 target=dict(captured_before=before['target']['captures'],captured_after=after['target']['captures'],
             captures_delta=target['captures_delta'],actual_replays=target['replays'],
             max_decode_requests=target['max_decode_requests'],dispatches=target['dispatches'],
             replay_shapes=target['replay_shapes']),
 draft=dict(captured_before=before['speculator']['captures'],captured_after=after['speculator']['captures'],
            captures_delta=delta['captures'],actual_replays=delta['replays'],fallbacks_delta=delta['fallbacks'],
            startup_eager_calls_before=before['speculator']['startup_eager_calls'],
            startup_eager_calls_after=after['speculator']['startup_eager_calls']),
 smoke=dict(generations=17,exact_token_text_parity=True,negative_http400_cases=10,client_concurrency=4,
            server_running_max=smoke['max_observed_server_running']),
 inference_jit=dict(window='Application startup complete through API shutdown triggered; includes first request compilation',
                    first_line=start+1,last_line=end+1,count=len(jit),entries=jit,
                    zero_inference_jit=len(jit)==0,
                    note='API smoke validates functionality; client wall times include HTTP and any observed compilation.'),
 proof_files_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (server/'startup-audit.json',server/'shutdown-audit.json',
                                  root/'api-smoke-v3/summary.json',logpath)})
output=server/'verified-audit.json'
assert not output.exists()
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
