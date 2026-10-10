# V2-only 清理与回归

2026-09-30 将 `vendor/vllm/` 整理为只包含原生 ModelRunner V2 的实现、安装工具、GSM8K 工具、测试和验证证据。旧 manual target/drafter/decoder/graph/CLI、SGLang 比较入口及旧实验输出已移除。原仓库的其他实现未改动。

## 实现迁移

- V2 权重共享和归一化辅助函数独立到 `native_v2/target_modules.py`。
- 数据加载、EOS 解析、输出对比函数移入 `gsm8k.py`，V2 runner 不再导入旧评测脚本。
- 六个迁移符号 AST 完全一致；`state.py` 只修改 import，其余六个原生核心模块保持原始字节。详见 [迁移证据](extraction-equivalence.json)。
- 状态测试改用 SpecForge 全前缀参考，不再依赖已删除的 manual drafter。
- 数据移动到 [gsm8k100](../gsm8k100/README.md)，输入 SHA256 未变；当前运行源码清单包含 15 项文件。

## 检查结果

| 检查 | 结果 |
|---|---|
| [CPU 全套](cpu-tests.txt) | 108 passed、61 subtests passed；12 项 CUDA 测试在本机跳过 |
| [rl10 CUDA 状态/输入/融合读取](gpu-tests.txt) | 34 passed，包含真实图回放 |
| [C1 真实模型回归](c1/audit.json) | 前 3 题 AR/LongSpark token 完全一致，均 3/3 正确、自然 EOS |
| [C2、64-token 分块回归](c2/audit.json) | 前 3 题 AR/LongSpark token 完全一致，均 3/3 正确、自然 EOS |
| [整理前后输出对照](pre-cleanup-output-comparison.json) | 两种配置的 AR 与 LongSpark 均与各自原始输出逐 token 一致 |
| [保留百题结果的离线复核](historical-result-audit.json) | 100/100 parity、96% accuracy；重新计算全部统计通过 |
| [五个 vLLM 补丁文件](patch-check.json) | 版本与 SHA 检查通过，状态 patched |

两次真实模型回归均使用 rl10 GPU0、Qwen3-4B、原 LongSpark 4B checkpoint、BF16、greedy、batch invariant；每方法独立加载，前三题预热后再测量前三题。C1 的 `max_num_seqs=1`、批 token=512；C2 为 2 和 64。数据、启动配置、源码哈希、原始输出和日志均在对应目录。

C1 的 LongSpark target/draft 实际图回放为 **78/81**，C2 为 **66/72**，C2 实测 decode 并发为 2。两组测量期间均没有新图捕获、draft 回退或 JIT monitor 编译告警；prefill 状态形状预热的逐字节不变核验通过。当前 15 项源码哈希与回归运行计划全部一致。

原有 [百题性能报告](../native-v2-gsm8k100-20260930/README.md) 保留整理前最终 V2 结果；本次三题回归用于验证清理没有改变行为，不用于宣称新的性能提升或重新测得百题性能。

## 复现

```bash
PYTHONPATH=vendor/vllm:vendor python -m pytest -q vendor/vllm/tests
CUDA_VISIBLE_DEVICES=1 LONGSPARK_TEST_CUDA=1 VLLM_BATCH_INVARIANT=1 \
  PYTHONPATH=vendor/vllm:vendor vendor/vllm/.venv/bin/python -m pytest -q \
  vendor/vllm/tests/test_native_v2_state.py \
  vendor/vllm/tests/test_native_v2_inputs.py \
  vendor/vllm/tests/test_native_v2_gather_gpu.py

vendor/vllm/.venv/bin/python vendor/vllm/scripts/evaluate_native_v2_gsm8k.py \
  --input vendor/vllm/validation/gsm8k100/inputs.jsonl \
  --target /data/lpf/models/Qwen3-4B --draft models/4B/longspark \
  --limit 3 --warmup-count 3 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --gpu 0 --execute --output artifacts/v2-only-c1-rerun
```

C2 改为 `--max-num-seqs 2 --max-num-batched-tokens 64` 并使用新的输出目录。
