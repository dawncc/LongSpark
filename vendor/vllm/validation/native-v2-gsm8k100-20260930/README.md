# 原生 vLLM ModelRunner V2：GSM8K 前 100 题

2026-09-30 在 rl10 单张 L20X 完成。AR 与 LongSpark **100/100 完整 token 序列一致**，均为 **96/100 正确**；两组各 100 题自然 EOS，0 截断、0 未解析。

本报告保留整理代码前的最终原生 V2 百题测量。V2-only 整理只抽取必要辅助函数并删除旧实现；原始结果和源码哈希未改写，整理后的检查见 [回归报告](../v2-only-cleanup/README.md)。

## 性能与接受长度

| 方法 | service token/s | decode token/s | generate wall token/s | 正确/总数 |
|---|---:|---:|---:|---:|
| 原生 V2 AR | 72.780 | 72.782 | 72.450 | 96/100 |
| 原生 V2 LongSpark | 327.576 | 334.022 | 319.400 | 96/100 |
| LongSpark / AR | 4.501× | 4.589× | 4.409× | — |

每方法输出 21339 token，包含 EOS。LongSpark 验证 3710 轮，提出 25970 个草稿 token，实际接受 17537 个；平均接受 **4.726954/轮**，**τ=5.726954**，接受率 **67.528%**。raw 与 emitted 接受数相同；计入最后 EOS 后，实际 decode 推进量/轮为 5.724798。

service 使用原生 `last_token_ts - scheduled_ts`；decode 使用 `last_token_ts - first_token_ts`，分子为输出 token 数减 1；两者均加权聚合。generate wall 覆盖客户端完整批次调用及异步收尾。加载、预编译、图捕获和前三题预热均不计入。该单次百题结果不代表完整 GSM8K 或生产并发吞吐。

## 原生 V2 与实际图回放

运行时类为 `vllm.v1.worker.gpu.model_runner.GPUModelRunner`，`use_v2_model_runner=true`、`async_scheduling=true`。日志中的“V1 LLM engine”是引擎系列名称，worker 类和审计确认实际使用 ModelRunner V2。

| 方法 | target FULL replay | draft replay | 计时内新捕获 | draft fallback |
|---|---:|---:|---:|---:|
| AR | 21339 | — | 0 | — |
| LongSpark | 3810 | 3910 | 0 | 0 |

Target 使用原生 `ModelCudaGraphManager`，LongSpark 使用原生 `CudaGraphManager`。Draft 图包括接受状态更新、并行草稿骨干和顺序 Markov 修正；融合 context 读取调用 3810 次。startup eager 计数在正式阶段保持 3→3。异步引擎可在 EOS 被前端处理前发射下一步，因此回放数不等于返回的验证轮数。

Prefill 的可变长度读取/状态更新在 decode 图外。正式测量前覆盖 B=1、width=9..190 的 **182 种**状态更新形状，用时 **0.272 秒**；独立 padding slots 上的零接受数更新通过全部可变状态逐字节比较。两组从最后一次请求预热到最后一次正式结果均 **0 次 JIT monitor 编译告警**。

## 配置

- 单张 NVIDIA L20X，Qwen3-4B target 与官方 LongSpark 4B checkpoint。
- vLLM `0.30.0+cu129`、PyTorch `2.13.0+cu129`、Python 3.12.3。
- FlashAttention 3、BF16、TP=PP=DP=CP=1，原生异步调度与 `enforce_eager=False`。
- greedy T=0、7 个草稿 token、batch invariant，C1、最大上下文 4096、批 token 512、KV 2 GiB。
- 关闭 prefix cache，开启 chunked prefill；自然 EOS，上限 1024。
- `VLLM_USE_FLASHINFER_SAMPLER=0` 避免随机采样器启动预热要求 nvcc；测量采用 greedy，FlashAttention 仍启用。
- 原生 `DraftModelSpeculator` 接口参考 DSpark；LongSpark 保留实际 target KV、Global16/Raw256/Local7 和原 checkpoint 网络。

## 证据与复现

- [独立百题审计](native-c1-full-v2/audit-full.json)、[汇总](native-c1-full-v2/completed.json)、[逐题 CSV](per-question.csv)。
- [AR 原始记录](native-c1-full-v2/vanilla/records.jsonl)、[LongSpark 原始记录](native-c1-full-v2/longspark/records.jsonl)。
- [AR 图审计](native-c1-full-v2/vanilla/measurement-audit.json)、[LongSpark 图审计](native-c1-full-v2/longspark/measurement-audit.json)、[prefill 预热](native-c1-full-v2/longspark/prefill-warmup.json)。
- [原始运行计划](native-c1-full-v2/plan.json)、[运行时源码与补丁复核](native-c1-full-v2/provenance/remote-source-patch-check.json)、[部署补丁检查](patch-verified-final.json)。
- [融合读取 GPU 测试](fused-gather/README.md)、[测试后硬件](hardware-after-final.txt)。

原始计划记录 16 项执行源码/清单 SHA，运行时全部一致，5 个部署补丁 SHA 也通过核验。整理后部分辅助函数已迁移、旧文件已删除，因此这些历史 SHA 只描述当时运行的源码，不能用于断言当前文件逐字节一致。数据移动到 [gsm8k100](../gsm8k100/README.md)，内容 SHA 未变。

```bash
cd /data/chenchen/codex/LongSpark
vendor/vllm/.venv/bin/python vendor/vllm/scripts/install_native_v2_patch.py --check
PYTHONPATH=vendor/vllm:vendor vendor/vllm/.venv/bin/python \
  vendor/vllm/scripts/evaluate_native_v2_gsm8k.py \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl \
  --target /data/lpf/models/Qwen3-4B --draft models/4B/longspark \
  --output artifacts/gsm8k100-native-v2-rerun \
  --limit 100 --warmup-count 3 --execute
```

使用新的输出目录。实现范围与安装方式见 [V2 指南](../../longspark_vllm/native_v2/README.md) 和 [补丁说明](../../patches/README.md)。
