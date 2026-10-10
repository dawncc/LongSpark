# LongSpark → vLLM 主干贡献包

已按 vLLM 主干 `10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf` 整理为独立、显式启用的 `method="longspark"`。本地工作分支为 `feat/longspark-v2`，35 个修改文件已暂存；尚未创建上游提交、签署 DCO、推送 fork 或发布 PR。

- 源码：`../upstream/`，仍位于 LongSpark 的 `vendor/vllm/` 下。
- 远端源码及独立环境：`rl10:/data/chenchen/codex/vllm-longspark-upstream/`。
- [完整补丁](longspark-v2.patch)；[基线和文件 SHA256](manifest.json)。补丁已经在干净基线上验证应用，35 个文件的 Git tree 完全一致。
- [PR 标题](PR_TITLE.txt)、[英文 PR 正文](PR_BODY.md)、[提交说明](COMMIT_MESSAGE.txt)、[上游检查清单](CHECKLIST.md)。

## 修改范围

LongSpark 模型、状态、attention 和 Triton gather/commit 内核独立放在模型文件和 `vllm/v1/worker/gpu/spec_decode/longspark/`。标准 loader 加载 checkpoint；标准 `vllm serve` 提供 API。公共文件仅加入必要的模型/方法注册、LongSpark 范围的能力和请求校验，以及 ModelRunner 的一行辅助隐状态方法选择。scheduler、rejection sampler、既有 `custom_class` 语义保持原实现。

初始范围是 CUDA、原生 dense Qwen3、BF16、FlashAttention、单卡、固定 7-token greedy。明确拒绝 prefix caching、LoRA、KV transfer、量化、自适应验证、结构化输出、采样惩罚和输入流；支持普通 HTTP 响应和输出 SSE。保留 eager 调试路径。原有 vLLM 0.30 集成和远端 GPU2 API 服务没有切换或替换。

## 验证结果

| 验证 | 结果 |
| --- | --- |
| 既有配置、采样、V2、n-gram、DSpark、EAGLE 回归 | 279 passed |
| 最终能力限制/输入流复测 | 34 passed，与上行有重叠 |
| 新模型/状态测试 | 18 passed，2 个 CPU CUDA Graph 用例跳过 |
| 真实 checkpoint 新旧对照 | 4 组，全部状态/logits/草稿逐 bit 相同 |
| 标准 API | 2 passed：补全/聊天、JSON/SSE、4 并发、400 后服务正常 |
| 上游格式 GSM8K E2E | graph 前100条、eager 前4条，2 passed，全部 token 对齐 AR |
| 模型注册 / 原生 GPU 初始化 | 2 passed / 1 passed |
| n-gram GPU 实际推理 | 4/4 与 AR 前缀 token 一致 |
| pre-commit / diff 检查 / 干净基线应用 | 通过 |

用户固定的 GSM8K 前 100 条，C4、BF16、最多生成 1024 tokens：

| 指标 | AR | LongSpark |
| --- | ---: | ---: |
| 准确率 | 96% | 96% |
| 输出 tokens | 21,339 | 21,339 |
| 完整 token 序列一致 | — | 100/100 |
| 生成耗时（秒） | 98.128 | 21.665 |
| 生成 wall token/s | 217.460 | 984.958 |

单次配对测量的吞吐比为 **4.529x**。共 3,711 轮验证，接受 17,536 / 草拟 25,977 tokens，平均接受草稿 **4.7254**，vLLM 含 bonus 的统计长度 **5.7254**。8 条请求在 EOS 后裁掉最后的 bonus，实际交付 decode tokens/round 为 5.7233。全部自然 EOS，没有达到 token 上限。

计时为各 batch 的 `generate()` wall time 之和，不含模型加载和 3 条预热。没有重复试验置信区间，不代表相对旧版集成的加速幅度。另一个上游 E2E 使用 zero-shot chat prompt，准确率为 85%、含 bonus 接受长度 6.161；该结果与上述固定提示词实验分开记录。

原始输出、逐轮接受计数、命令、日志和来源核验见 [validation](validation/)。推理均开启 `VLLM_GPU_SYNC_CHECK=error`；target 和 draft 的 FULL CUDA Graph 捕获完成。

环境限制：目标主干没有同提交的 cu129 wheel，使用经源码核对 CUDA 编译路径相同的 `8cbd5d030…` 扩展，运行当前主干 Python 源码，并关闭 Rust frontend。这不是同提交的完整源码构建。ROCm、Rust frontend、完整 DSpark/EAGLE 模型 E2E 未验证。另有一个 Meta-Llama gated 配置测试在未修改的主干上就返回 403，已单独记录并排除。[环境来源](validation/setup/environment-provenance.json) 包含详细依据。

## 复现与提交

当前独立主干工作树已应用补丁。需要重新构建工作树时，在一个干净的 vLLM clone 中执行：

```bash
git switch -c feat/longspark-v2 10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf
git apply --check /path/to/LongSpark/vendor/vllm/upstream-contribution/longspark-v2.patch
git apply --index /path/to/LongSpark/vendor/vllm/upstream-contribution/longspark-v2.patch
```

按 vLLM 的 `AGENTS.md` 使用 uv 建立环境。正常机器优先按官方 editable/precompiled 安装；此次 rl10 的特殊二进制来源仅作为可复现记录，不应默认为其他机器的安装方式。测试命令在 [PR 正文](PR_BODY.md) 和 [回归 manifest](validation/regression-modified/manifest.json) 中。用户固定数据的评测入口为 [verify_gsm8k.py](verify_gsm8k.py)，分别运行 `--method ar` 和 `--method longspark`，输入仍使用原 `validation/gsm8k100/inputs.jsonl`。

公开 PR（GitHub 对 MR 的称呼）之前还需要你逐行审阅、亲自运行相关测试并同意 DCO。上游 `AGENTS.md` 明确要求：**“The submitting human must review every changed line and run relevant tests.”** 上游的 commit hook 会自动写入 `Signed-off-by`，因此这里没有替你进行该签署。

完成上述审阅后，配置自己的 Git 姓名/邮箱，在主干工作树中提交并使用 GitHub CLI 创建 fork/PR：

```bash
# 在 vendor/vllm/upstream 中，完成审阅及 DCO 同意后运行。
git commit -s -F ../upstream-contribution/COMMIT_MESSAGE.txt
gh repo fork vllm-project/vllm --clone=false --remote --remote-name fork
git push -u fork feat/longspark-v2
gh pr create --repo vllm-project/vllm --base main \
  --head dawncc:feat/longspark-v2 \
  --title "[Spec Decode] Add LongSpark for Model Runner V2" \
  --body-file ../upstream-contribution/PR_BODY.md
```

发布前重新核查重复 PR，并更新正文中人类审阅与 DCO 的勾选项及最终基线。此次检索未发现已有 LongSpark issue 或开放 PR；没有向维护者发送任何消息。
