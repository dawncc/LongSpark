# vLLM pr-checklist：实现完成，等待人类审阅和 DCO

依据：上游 `.agents/skills/pr-checklist/SKILL.md`、`AGENTS.md`、模型实现/测试指南与 Model Runner V2 设计文档。

## 1. Design Fit

- LongSpark 是显式启用的新方法；复杂逻辑放在独立模型/状态模块。
- 复用标准模型加载、V2 scheduler/rejection、CUDA Graph manager、API 和测试 fixtures；不修改通用自定义 proposer 合约。
- 公共 ModelRunner 只有一行新增方法；其他共享改动是注册和方法范围的 validation。输入 streaming 会提前拒绝，避免复用 slot 时丢失摘要状态。
- 本次新增代码超过 250 行，属于高复杂度新功能。主要负担为独立的 Global16/Raw256 attention、状态提交和原 checkpoint 数值兼容。保留这部分代码的依据是原 checkpoint 完整数值对齐和 GSM8K 实测；未带入训练、SpecForge/SGLang 依赖或实验诊断。
- GPU_SYNC_CHECK=error 实际推理通过；仅 scheduler 的 CPU metadata 决定 gather 宽度，accepted lengths 不回传 CPU。
- 能力限制均明确且只作用于 LongSpark，ROCm/其他模型/多卡等不会被误称支持。

## 2. Testing and Validation

- 修改前 baseline：168 passed，1 个 gated HF HTTP403。修改后主回归279 passed，最终 gate 复测34 passed（重叠）。
- 新模型/状态18 passed、2 CPU图测试skip；真实 checkpoint 四组完整 logits/状态逐bit一致。
- 固定 GSM8K100：100/100 token相同、双方96%；吞吐/接受长度保留逐条和逐轮原始记录。
- 上游零样本 chat GSM8K100/eager4：2 passed；这组100题85%，不能与固定提示词96%混用。
- 原生 API JSON/SSE/并发/非法请求恢复：2 passed。n-gram GPU额外4条与AR前缀一致。
- 模型registry2、GPU初始化1通过。没有用未改路径或零接受冒充新增功能验证。
- 新 CUDA Buildkite job 运行 contributed GSM8K/API；unit suites 已覆盖新状态/模型。CI配置本身已lint，但尚未触发远端 Buildkite。
- 证据边界：非同提交完整源码build；已核对wheel与main的CUDA编译路径一致。Rust/ROCm/full DSpark/full EAGLE模型E2E未测。单次吞吐比不声称跨硬件稳定收益或零冷启动JIT。

## 3. Code Quality and Style

- pre-commit、diff --check 通过，源码原始 MIT notices 保留。
- 模型使用标准 weight loader，严格拒绝多余、缺失、重复和错误形状的 checkpoint 权重。
- 文档覆盖模型选择、CLI、Python、API、性能配置、支持范围和内存预算。
- 新生产代码不增加 runtime dependency；无 monkeypatch、HTTP middleware、进程关闭拦截、诊断 RPC。

## 4. Pull Request Contents

- PR正文遵循 Overview / Claims / Validation / Details 和 checklist 格式。
- 明确 AI assistance、测试命令与结果、模型评测、二进制来源和未验证范围。
- GitHub LongSpark issue/开放PR检索无重复，结果保存。DSpark是现有实现参考，LongSpark的有界 target-state 算法不同。
- 标题、提交说明、补丁、文件hash和干净基线应用检查准备完成。实验材料保留在LongSpark贡献包，不进入上游补丁。

## 剩余人类步骤

上游 AGENTS 原文：**“The submitting human must review every changed line and run relevant tests.”**

上游贡献指南要求提交者同意 DCO，并用 Signed-off-by 认证。尚未声称你已审阅、复测或同意；尚未创建上游commit、fork push或PR。请完成这些步骤后再发布。当前代码、补丁和说明可直接审阅，无需再次实施重构。
