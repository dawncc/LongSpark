# LongSpark V2 性能优化与原生 HTTP API 验证

2026-10-10，在 rl10 实测。代码和证据均位于 `vendor/vllm/`；远端仓库为 `/data/chenchen/codex/LongSpark`。本轮保持 vLLM 0.30.0 原生 ModelRunner V2、LongSpark checkpoint 和投机数学定义不变。

## 结果

使用官方 GSM8K test 前 100 题、既有固定 token fixture。每组分别启动 AR 和 LongSpark，完成加载、图捕获、prefill 形状预热和 3 题 warmup 后计时。所有 6 个方法/配置组合均输出 21,339 tokens，100/100 完整输出 token 序列一致，96/100 正确，全部自然 EOS，无截断。

### 同配置单请求前后对比

两组均在 GPU 0，C1、2 GiB KV、max model len 4096、max batch tokens 512、max new tokens 1024、BF16、七个 greedy draft tokens、batch-invariant kernels、async scheduling、FULL_AND_PIECEWISE。

| 指标 | 修改前 `2542b70` | 本轮优化后 | 变化 |
|---|---:|---:|---:|
| AR decode token/s | 72.814 | 72.820 | +0.008% |
| LongSpark decode token/s | 334.210 | 335.982 | +0.530% |
| LongSpark service token/s | 327.496 | 329.499 | +0.612% |
| LongSpark wall token/s | 319.290 | 321.240 | +0.611% |
| LongSpark decode / AR decode | 4.590× | 4.614× | |
| 接受 draft tokens / verification round | 4.726954 | 4.726954 | 完全相同 |
| τ = 1 + 接受长度 | 5.726954 | 5.726954 | 完全相同 |

同 C1 前后每题输出及每轮接受序列都一致：17,537 accepted / 3,710 rounds。这里是一次完整百题前后测量，端到端收益约 0.5%–0.6%，不是多轮统计置信结论，也不能将组件的约 30% 降时解释为整体提速。

### 四请求并发配置

GPU 1，C4、4 GiB KV，其余参数相同。这里比较该配置内的 AR 与 LongSpark；不同并发、GPU、KV 容量的结果不作为纯代码改动的前后加速。

| 方法 | 总 generate wall time | 端到端输出吞吐 |
|---|---:|---:|
| AR | 99.020 s | 215.502 token/s |
| LongSpark | 21.761 s | 980.625 token/s |

LongSpark wall 吞吐是同配置 AR 的 **4.550×**。每批最多四个请求，批内所有请求完成后再提交下一批；不是持续满负载的 HTTP 吞吐测试。`sum(request decode time)` 存在请求重叠，不能用它作为并发吞吐分母。

C4 的平均接受长度为 4.725411、τ 为 5.725411（17,536 accepted / 3,711 rounds）。四题的 draft 接受序列与 C1 有变化，目标模型输出仍全部一致；不同 batch shape 下不声称 draft 接受序列逐位不变。

## 优化内容与图证据

- `commit_kernel.py` 融合 Global16、Raw256、hidden、length 的最终提交。BF16 contractions 和 FP32 attention normalization 保持原实现。
- Decode gather 直接写入已有 CUDA Graph 输入缓冲，去掉中间分配、清零和拷贝。
- 两项默认开启，可用 `--native-optimizations reference` 在同一源码版本关闭，方便重复 A/B。

真实 Qwen3-4B 几何的组件测试中，B1/width8 状态更新从 0.26232 ms 降到 0.18163 ms；B4 从 0.34877 ms 降到 0.24430 ms。详见 [组件证据](components/README.md)。14 种 prefill 宽度共用一个 commit kernel 变体。

| 配置 | Target 实际图回放 | Draft 实际图回放 | 最大 decode 请求数 |
|---|---:|---:|---:|
| 修改前 C1 | 3,810 | 3,910 | 1 |
| 优化后 C1 | 3,810 | 3,910 | 1 |
| 优化后 C4 | 1,291 | 1,325 | 4 |

这三组测量窗口内 target/draft 新增捕获为 0、draft fallback 为 0；已启用的 JIT monitor 未报告测量期编译。Prefill gather 和部分输入处理仍在 draft 图外，不声称整个服务全部入图。

[独立汇总审计](comparison-audit.json) 从原始 records、native timing、接受数组、图计数、warmup 和日志重新计算结果。基线的 15 个源码哈希核对 Git `2542b70`；优化后两组的 16 个源码哈希核对当前文件及各 manifest。原始目录为 [baseline-c1](baseline-c1)、[optimized-c1](optimized-c1)、[optimized-c4](optimized-c4)。

## 原生 vLLM HTTP API

[启动和调用说明](../../SERVING.md)。使用原生 `vllm serve`，没有额外代理或自建推理循环。默认 loopback、C4、4 GiB KV、BF16、V2 async、CUDA graphs、greedy；保留模型的两个 EOS token IDs。

最终验证：[api-smoke-v3/summary.json](api-smoke-v3/summary.json) 和 [api-server-v3/verified-audit.json](api-server-v3/verified-audit.json)。

- 三道固定 GSM8K 题，覆盖 `/v1/completions` 与 `/v1/chat/completions` 的 JSON/SSE，加四请求并发和错误后的恢复请求，共 17 次生成；17/17 答案正确，prompt/output token 和文本与 AR 基线完全一致。
- 10 个不支持参数的请求均返回 HTTP 400，随后仍正常生成；包括 `min_tokens=1` 加 129 个 stop IDs 的 worker 容量回归用例。
- Worker 审计确认实际 decode batch 达到 4，target 图回放 372 次、draft 387 次；捕获增量与 fallback 均为 0。
- 正常 SIGTERM 前成功收集最终审计。修复前的 `api-*-v2` 仅保留调试证据，其关停审计失败，不作为最终图证明。

HTTP 首次生成/首次多请求形状仍会触发 4 个原生 sampler/gather kernel 的 JIT 警告；启动完成不等于所有请求路径已预热。上述 HTTP 检查验证功能和图执行，不声称冷启动请求零 JIT，也不将其短样本速度当作 GSM8K100 API 吞吐。交付的常驻服务另行执行了单请求和四请求预热。

交付时服务运行于 rl10 GPU 2，PID `3275906`，地址 `http://127.0.0.1:18080/v1`，模型名 `longspark-qwen3-4b`。通过 `ssh -N -L 18080:127.0.0.1:18080 rl10` 从本机访问。服务审计/日志在远端 `artifacts/perf-api-20261010/api-service*`。这是交付时状态记录，重启后 PID 会变化。

[常驻服务预热](api-service-warmup/summary.json) 再次通过 17 次生成与 10 个拒绝检查；随后使用 OpenAI Python SDK 3.22.1 完成普通/流式调用（[SDK 结果](api-service/client-smoke.json)，[日志快照](api-service/log-snapshot.txt)），这两次预热后调用未出现 JIT 警告。常驻服务尚未关闭，因此其最终 shutdown audit 会在正常停止后写出；完整已闭合的图验证使用上面的 v3 服务记录。

当前范围为 dense Qwen3 BF16、单卡、七 token greedy speculation、文本输入。随机采样、非中性 penalties、`min_tokens>0`、工具调用、多模态、结构化解码和多返回候选会在入口拒绝。

## 环境与复现

rl10：NVIDIA L20X，driver 570.172.08，Python 3.12.3，vLLM 0.30.0+cu129，Torch 2.13.0+cu129，Triton 3.7.1，Transformers 5.12.1。每个任务占用独立 GPU，未声称独占整台主机。安装后的五文件扩展 patch 已按哈希复核，见 [deployed-patch-check.json](deployed-patch-check.json)。

[最终运行环境](runtime-final.json) 同时记录 API、native 实现、脚本及测试源码 SHA256，已逐项核对远端与本地一致。`runtime-before-api-fix.json` 对应修复前 API 版本，仅保留溯源，不作为最终版本证明。

仓库根目录执行，使用新的 output 路径：

```bash
vendor/vllm/.venv/bin/python vendor/vllm/scripts/evaluate_native_v2_gsm8k.py \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl \
  --target /data/lpf/models/Qwen3-4B --draft models/4B/longspark \
  --output artifacts/repeat-optimized-c1 --limit 100 --warmup-count 3 \
  --gpu 0 --native-optimizations fused --execute
```

C4 增加 `--max-num-seqs 4 --kv-cache-gib 4` 并选择空闲 GPU；同源码参考路径改为 `--native-optimizations reference`。每次运行都核对载入前源码哈希，拒绝覆盖已有证据。

```bash
CUDA_VISIBLE_DEVICES=1 LONGSPARK_TEST_CUDA=1 PYTHONPATH=vendor/vllm:vendor \
  vendor/vllm/.venv/bin/python -m pytest -q vendor/vllm/tests
python3 vendor/vllm/validation/perf-api-20261010/audit_comparison.py \
  --output /tmp/longspark-comparison-recheck.json
```

最终混合 CPU/CUDA 测试 **174 passed，120 subtests passed**，见 [完整日志](full-tests-final.txt)。源码和真实 GPU 测试覆盖逐位状态、NaN 拒绝后缀、动态图输入、slots 重排/复用、padding、直接输出缓冲以及 native API 的真实关闭顺序。
