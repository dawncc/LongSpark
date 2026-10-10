# Native V2 组件性能检查（2026-10-10）

环境：rl10，独占 GPU 1 / NVIDIA L20X，vLLM 0.30.0 + cu129，Torch 2.13，BF16。
这里是组件微基准，不能替代完整 GSM8K / API 端到端吞吐测试。

## 改动与数值语义

- `commit_kernel.py` 将 Global16 结果、Raw256 环形缓存、最后 hidden 和长度的最终提交合并为一次 GPU kernel。全部 attention/BF16 contractions/FP32 normalization 保持原函数不变；kernel 仅索引与拷贝。
- `gather_kernel.py` 可直接写入既有 CUDA Graph 输入缓冲，去掉中间输出分配、两次清零和五次拷贝。无 `output_buffers` 时保留原来输出独占存储的接口。
- `additional_config.longspark.use_fused_state_commit=false` 与 `use_direct_gather=false` 可恢复两个优化前路径，支持同一版本 A/B。
- 与 prefill 宽度相关的 strides / KEEP 是 runtime scalar。8、9、10、11、17、31、63、127、190、255、256、257、512、509 十四个宽度仅产生一个 commit kernel 变体，避免每个新 prompt 长度触发编译。

## 组件时延

下表均为真实 Qwen3-4B 维度：5 层、32 个 query heads、8 个 KV heads、head dim 128、hidden 2560。状态使用合成输入；CUDA Graph 预热后连续回放 1000 次，CUDA Event 计时。参考路径和融合路径逐位比较全部六个状态缓冲。

| 状态更新 | 原实现 / ms | 融合 / ms | 组件加速 |
|---|---:|---:|---:|
| B=1，width=8，boundary hidden | 0.26232 | 0.18163 | 1.444× |
| B=4，width=8，boundary hidden | 0.34877 | 0.24430 | 1.428× |
| B=1，width=512，boundary hidden | 0.40008 | 0.27563 | 1.452× |
| B=4，width=512，boundary hidden | 0.80087 | 0.52543 | 1.524× |

Direct gather（包含旧路径向静态输入缓冲的 staging）：

| 输入宽度 | 旧 GPU 图 / ms | Direct GPU 图 / ms | 旧 eager wall / ms | Direct eager wall / ms |
|---|---:|---:|---:|---:|
| B=1，width=1 | 0.01328 | 0.00295 | 0.12472 | 0.06645 |
| B=1，width=8 | 0.01201 | 0.00305 | 0.12203 | 0.06554 |
| B=4，width=[8,3,1,7] | 0.01219 | 0.00316 | 0.12506 | 0.06767 |

Eager wall 是 1000 次完整 Python 调用、最后同步的总时间均值，包含 Python/launch 成本，不等同于单 kernel GPU 时延。

## 验证与后续瓶颈

`candidate-tests-final.log`：**42 passed**，包含新状态提交/缓存复用测试、原状态测试、gather 测试。覆盖真实几何、B1/B4、width8/512、跨环形边界、重排 slots、reset 复用、zero-count 精确 no-op、多个 padding 行、拒绝后缀 NaN、非连续输入、真实 CUDA Graph 重放、直接输出存储与 tail 清零。

`profile-state.log`：原状态更新约 0.263 ms/轮，两次 BMM 合计约 6 μs，主要时间在 where/index/gather/copy。`profile-propose.log` 使用真实 LongSpark checkpoint 与 Qwen3 权重（合成已提交状态），proposal 图约 3.090 ms；MM 约占记录的 GPU 时间 27.75%，剩余包含拷贝、逐元素计算及归约。本轮保持这些浮点运算不变，未声称其得到优化。

原始完整 Chrome profiler trace 保留在远端 `artifacts/perf-api-20261010/perf-stage/update-reference-profile.json` 与 `propose-profile.json`，仓库收录聚合日志。

## 重现

仓库内三个 probe 脚本是运行脚本的重现副本，只将隔离测试包的 state 导入改为当前仓库导入。原始 probe 位于远端 `artifacts/perf-api-20261010/perf-stage/`，隔离包避免影响 GPU 0 同时执行的旧提交基线。执行前确认 GPU 1 空闲。以下命令在 rl10 仓库根目录使用：

```bash
CUDA_VISIBLE_DEVICES=1 LONGSPARK_TEST_CUDA=1 \
PYTHONPATH=vendor/vllm:vendor:vendor/vllm/tests \
vendor/vllm/.venv/bin/python -m pytest -q \
  vendor/vllm/tests/test_native_v2_commit_gpu.py \
  vendor/vllm/tests/test_native_v2_gather_gpu.py \
  vendor/vllm/tests/test_native_v2_state.py

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=vendor/vllm:vendor:vendor/vllm/tests \
vendor/vllm/.venv/bin/python \
  vendor/vllm/validation/perf-api-20261010/components/check_candidate_v2.py

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=vendor/vllm:vendor:vendor/vllm/tests \
vendor/vllm/.venv/bin/python \
  vendor/vllm/validation/perf-api-20261010/components/bench_gather_direct.py
```

输入和测量配置见脚本；结果写回远端 `artifacts/perf-api-20261010/perf-stage/`。`source-sha256.json` 记录本轮组件测试所用实现/测试文件哈希。
