# Assignment 任务清单

依据：`week4coursework-combined.pdf`、ASR/RAG README、四份 task guides 和报告模板。以下为任务整理，未执行实现或实验。

## 1. 环境与准备

- [ ] 掌握 CUDA tutorial 1–5 涉及的线程索引、内存、同步与 tiled GEMM。
- [ ] 配置集群、ASR 环境、模型和 CUDA 编译环境，验证 PyTorch ASR baseline。
- [ ] 配置独立 RAG 环境、课程数据及自己的 ELM API key；密钥不提交或分享。
- [ ] 每人同时只使用一个 GPU allocation；记录 GPU 型号，比较实验使用相同 GPU 类型。

## 2. ASR：完成四项 CUDA 任务

- [ ] **Task 1 — RoPE**：完成 `hw1-asr/csrc/task1_rope.cu` 的索引与 q/k 旋转。
- [ ] **Task 2 — Reductions**：完成 `task2_reduce.cuh`、`task2_reduce.cu` 的归约、RMSNorm 和 online softmax。
- [ ] **Task 3 — Tiled GEMM**：完成 `task3_gemm.cu` 的共享内存分块矩阵乘法、bias 与 GELU。
- [ ] **Task 4 — Attention**：完成 `task4_attention.cu` 的 streaming attention、online softmax、GQA 与 causal mask。
- [ ] 依次通过 `make test1` 至 `make test4`，逐步接入真实模型验证转录，最终通过完整 `make test` 和转录检查。

约束：任务计算使用自己的 CUDA 实现，不调用 cuBLAS、cuDNN、CUTLASS、Thrust 或 PyTorch 代做；保留接口，提交时提供的模型、封装、benchmark、其他 kernels 与测试保持原样。

## 3. ASR：性能实验与报告问题

- [ ] 比较自己的 kernels 与 PyTorch：转录耗时、每 token 耗时、峰值显存、各操作耗时。
- [ ] 比较 tiled GEMM、naive GEMM 与 cuBLAS：记录各案例 GFLOP/s，解释差距。
- [ ] 分析 RoPE 和 reductions 的 GB/s 及其占 GPU 峰值带宽的比例。
- [ ] 每个 task 至少完成一个 “Try this” 实验，共至少 4 项；记录改动、前后数字与原因。
- [ ] 回答四份指南末尾全部 16 道报告问题：RoPE 3 道、Reductions 4 道、GEMM 4 道、Attention 5 道。
- [ ] 保存同一 GPU 节点上的 `make test`、`./run.sh --profile` 和 `nvidia-smi -L` 输出。

## 4. 完整 RAG baseline

- [ ] 理解并运行：音频 → ASR → enc_a 编码 → 精确检索 → 证据组装 → ELM 回答。
- [ ] 检查转录、检索文档、证据、答案和各阶段耗时；先用少量查询验证。
- [ ] 在 CUDA kernels 通过后，将其接入同一 RAG pipeline。
- [ ] 评估检索质量：Recall@5/10/20/100、MRR@10、NDCG@10。
- [ ] 生成候选答案并评估 correctness、grounding、completeness；人工检查代表性输出。
- [ ] 记录查询覆盖、judge 设置、缓存和计时边界；说明 LLM judge 及其以 ASR transcript 为问题的局限。

注意：enc_b 未提供 query encoder，同维向量不代表空间兼容；使用 enc_b 时需提供兼容编码或有依据的映射。材料未明确要求每个实验跑完所有 split。

## 5. 自选系统研究

- [ ] 提出一个明确的 systems question，写清 baseline、技术改动与预期取舍。
- [ ] 选择一个方向深入研究：检索深度、ANN、batching、context selection、reranking、动态更新、表示迁移，或课件允许的 ASR 优化；无需全部做。
- [ ] 实现改动，在控制输入和无关设置的条件下与 baseline 比较。
- [ ] 同时测量质量／正确性与相关系统成本，例如延迟、吞吐、内存、上下文大小或索引构建成本。
- [ ] 记录硬件、软件、重复次数、缓存与计时范围；研究下游组件时复用相同转录。
- [ ] 解释结果、机制、负结果、取舍与局限，保留可复现实验的代码和记录。

仅切换语言模型不足以构成系统研究；task 的小实验也不能自动替代完整研究的证据链。

## 6. 报告与交付

- [ ] 使用提供的 LaTeX 模板，整合 ASR 与系统研究；**正文总计最多 6 页**。
- [ ] 正文包含问题、技术方法、实验控制、质量与成本结果、分析和局限；关键内容不只放附录。
- [ ] References 和可选 Appendix 不计正文页数；保留模板格式、review 模式、匿名作者与课程页眉，abstract 保持注释。
- [ ] 填写 AI declaration、Group ID、组员姓名和 UUN。
- [ ] 编译最终 PDF，运行提供的格式检查器并核对警告。
- [ ] 整理五个 task 源文件、ASR 测试／profile／GPU 信息、系统研究代码与实验记录、报告 PDF；按正式 brief 要求提交。

## 7. 提交前核实

- [ ] 截止日期、提交平台、命名与打包规则。
- [ ] 分值、权重与详细评分标准。
- [ ] 组队人数上限、注册办法及其他考核环节。
- [ ] 声明页等最终页数政策（课件标注待确认）。
