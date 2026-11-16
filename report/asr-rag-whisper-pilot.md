# Codex prompt for the Whisper ASR retrieval pilot

请在 `erertertet/edin-mls-26-autumn` 里运行一个 ASR × RAG 小样本实验，遵循仓库的 `AGENTS.md`，在 Edinburgh Slurm 计算节点上执行转录和检索。

**Research question**

> How does replacing GLM-ASR-Nano with Whisper large-v3 affect evidence recall for fixed top-5 and similarity-cutoff retrieval on the same spoken queries?

1. 从课程 dev split 里选有音频和至少一条正相关标注（qrels）的查询。按 query ID 排序，用 Python `random.Random(26)` 无放回抽取 8 条，在看转录或检索结果前记录所选 ID 和音频哈希。
2. 对同一批音频分别得到 GLM-ASR-Nano 和 `openai/whisper-large-v3` 的转录。能确认缓存与同一份音频匹配时，可直接复用 GLM-ASR 转录，否则用仓库已有 ASR adapter 补跑。Whisper 使用转录模式，记录模型 revision 和解码设置。两种转录都是实验输入。
3. 两种转录都用同一个 `enc_a` 编码，对同一批 24,000 条课程文档完整计算余弦相似度。每份排序分别取 **fixed top-5** 和 **绝对 cutoff ≥ 0.8925**。cutoff 无命中时返回 top-1，排序并列沿用 corpus row order。沿用报告参数，四组共用同一份 qrels。首轮做到检索即可。
4. 用每条查询找回的相关文档数除以该查询的正相关标注文档总数，然后对相同的有效配对查询取平均。给出下面两行表，并用 **Whisper − GLM-ASR** 报变化的百分点。正值表示换成 Whisper 后召回提高。

| 检索方式 | GLM-ASR 平均召回率 | Whisper 平均召回率 | 变化（百分点） |
| --- | ---: | ---: | ---: |
| Fixed top-5 | | | |
| Absolute cutoff 0.8925 | | | |

把两份转录、每条查询的最高相似度、返回文档数、召回率、运行命令和 Slurm job ID 保存在集群实验目录。失败或缺失的查询单独报告，汇总注明完成的配对数。最后说哪种策略的变化幅度更大。若两份转录有数字或专名分歧，可以用个例解释检索变化。这个 pilot 的结论限于抽到的音频、两款模型和固定检索参数。两份转录的分歧只表明模型输出不同，判断哪份转录正确需要独立参考。

参照 [RAG 设置](../rag/README.md)、[全库排序实现](../rag/extensions/prepare_uncapped_rankings.py)、[当前 cutoff 实验记录](rag-threshold-controls-results.md) 和 [Whisper large-v3 模型卡](https://huggingface.co/openai/whisper-large-v3)。
