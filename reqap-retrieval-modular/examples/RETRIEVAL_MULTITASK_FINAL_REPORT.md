# 检索系统多任务评测与融合实现 — 最终说明（与代码一致）

本文档与当前仓库实现、`RETRIEVAL_BENCHMARKS.md` 及评测脚本行为对齐，用于论文/技术报告引用。**不包含**「将 Router 改为可学习模型」等未实现项；**不包含**将某次 Grid 结果固化为线上默认权重的结论（Grid 仅作为可选超参搜索工具）。

---

## 一、问题与目标

在多种查询形态（短意图 / 长结构化事件 / 自然语言问句）下，构建**可复现的三任务检索评测**，并在此基础上比较 **BM25、SPLADE、Dense** 及 **融合管线**，避免单一任务或错误标签构造导致的结论失真。

---

## 二、三任务定义（与脚本一一对应）

| 任务 | 脚本与数据 | Query | 标签 | 主指标（聚焦表） |
|------|------------|-------|------|------------------|
| **Task A — Exact Match** | `prepare_retrieve_dev_eval.py` → `obs.csv` + `queries.jsonl`；`eval_three_pipelines.py --benchmark-profile exact` | 完整结构化事件（`input[0]` + `input[1]`） | 单相关文档（自匹配） | Hit@1、MRR、Hit@5（辅） |
| **Task B — Short Query** | `build_short_query_eval.py`；`eval_three_pipelines.py --benchmark-profile short` | 仅 `input[0]`（短意图） | **同一 `query_key` 下多正例** `relevant_ids` | Recall@10/50、NDCG@10、P@10（**不以 Hit@1 为主**） |
| **Task C — PerQA** | `eval_perqa_retrieval_export.py` + `queries_dev_p*.jsonl` | 自然语言问题 | 多相关 `obs` id | Recall@10/50、NDCG@10 |

三任务**统一**的是数据字段形态（`query` + `relevant_ids`）与「多路检索 + 标准 IR 指标」的评测方式；**不**强行统一「主指标名称」（因任务标签结构不同）。

---

## 三、关键修正（与对话/代码一致）

### 3.1 Task B：从「自检索」改为多正例召回

**原问题**：若短任务仍用「整段事件 = query + 单一 doc id」，易出现 query 与文档高度重叠、自匹配占优，**低估**真实检索难度（尤其 SPLADE/Dense）。

**实现**：`build_short_query_eval.py` 按 `input[0]`（`query_key`）分组，**同一 key 对应多个 `relevant_ids`**，与 Task A 共用同一 `obs.csv` 语料 schema，仅 `queries.jsonl` 不同，便于索引可比。

### 3.2 Task C：`queries.jsonl` 与 `obs.csv` 必须对齐

**原问题**：`relevant_ids` 来自大范围 id，而索引仅覆盖较小 `obs.csv` 时，Recall 会**接近 0**（评测失效）。

**实现**：PerQA 必须使用与官方 benchmark 一致的 **`dev_persona_*_obs.csv`**；`build_eval_indexes.py` 在构建后写入 **`index_meta.json`**（语料哈希、模型路径、索引文档数等），`eval_three_pipelines.py` / `eval_perqa_retrieval_export.py` 在评测前做 **一致性校验**（`relevant_ids ⊆ obs`、`index_meta` 指纹等）。

### 3.3 SPLADE 工程修复（ReQAP 核心库）

- **`sparse_retrieval.py`**：单非零维等边界情况下的张量索引，避免 `IndexError`。
- **`index_construction.py` / `CollectionDataset`**：`resolve_splade_doc_ref`，将 posting 中的全局 `obs id` 与行号正确对应，避免 `KeyError`（PerQA 大语料必备）。

---

## 四、三类检索器角色（实验现象归纳）

| 检索器 | 机制 | 典型任务倾向（文献 + 本项目观测） |
|--------|------|-----------------------------------|
| **BM25** | 词法稀疏 | Task A/B 上字面匹配强；Task C 纯 BM25 多标签语义召回往往弱 |
| **SPLADE** | 神经稀疏扩展 | 介于词法与共现语义之间；训练方式会显著改变 A/B/C 折衷 |
| **Dense** | 双塔语义向量 | Task C 语义匹配通常占优；极短 query 或强 ID 级任务上未必最优 |

> 具体数值以各工作区 `results_table*.md` / `*_results_models.json` 为准。

---

## 五、SPLADE 训练：分桶（Bucket）选项

**脚本**：`train_splade_retriever.py`（`--bucket-training`）。

**设计要点**（与实现一致）：按启发式将样本分为 **short / long** query（如 short 对应 `input[0]`，long 对应拼接），共享编码器、多损失头 + 稀疏正则（见脚本内 `train_splade_bucket_contrastive` 与参数说明）。

**已知折衷**（需在论文中如实报告）：分桶训练可在 **PerQA 类语义任务**上提升 recall/排序，但可能削弱 **强词法/精确匹配**任务上的表现；需与「未微调 Hub SPLADE」对照，避免单一任务过拟合解读。

---

## 六、融合方法：代码中的真实行为

### 6.1 与「RRF」的关系（避免概念混淆）

- **`fusion/rrf.py`** 仍存在，用于 **排序级 RRF**（例如 `SpladeThenDenseRerank` 内 **SPLADE 列表与 Dense 重排** 的 `rrf_fuse` 等路径）。
- **当前主线 `DynamicFusionOurs`**（`pipelines/dynamic_fusion.py`）对 **BM25 / SPLADE / Dense 三路并行检索**采用的是：
  - 各文档上的 **`bm25_score` / `splade_score` / `dense_score`**
  - **min–max 归一化**后按权重 **加权求和**（`fusion/weighted_sum.py`），**不是** \(\sum 1/(k+\mathrm{rank})\) 的 RRF。

因此：若报告中写「Phase 1 全面 RRF」，应限定为 **历史方案或子模块**；**当前 Dynamic Fusion 的默认实现是分数加权融合**。

### 6.2 Query-aware 权重（默认，未 Grid 时）

**实现**：`routing/query_router.py` 中 `route_query_fusion_weights(query)`：

- 短查询 / 结构化 / 默认三类 **规则权重**（对 `bm25_score`、`splade_score`、`dense_score` 通道归一化后使用）。
- 与 `query_router.py` 文档一致：用 **加权分数融合** 替代「仅排序的 RRF」作为 Dynamic 的主融合语义。

### 6.3 Grid Search（可选，用于 PerQA 调参）

**脚本**：`eval_perqa_retrieval_export.py --grid-search-dynamic --grid-metric <指标>`。

**当前网格**（代码内写死，可改源码扩展）：

- `w_bm25 ∈ {0.1, 0.2, 0.3}`
- `w_splade ∈ {0.3, 0.4, 0.5, 0.6}`
- `w_dense ∈ {0.2, 0.3, 0.4, 0.5}`  

共 **48** 组；构造 `DynamicFusionOurs(..., w1_bm25=..., w2_dense=..., w3_splade=...)` 时 **绕过** 上述路由，对**所有 query 使用同一组全局系数**（内部仍会归一化）。

**输出**：`{queries_stem}_dynamic_grid_search.json` / `.csv`；终端打印 `[Grid Best]`。

**说明**：Grid 用于 **在 dev 上搜索固定融合系数**，不等同于「生产系统最终权重」；与「可学习 Router」无关，也未在主线评测中默认启用。

---

## 七、端到端命令流（便于复现）

1. **Task A**：`prepare_retrieve_dev_eval.py` → `build_eval_indexes.py`（`RETRIEVE_EVAL_WORKSPACE`）→ `eval_three_pipelines.py --benchmark-profile exact`  
2. **Task B**：`build_short_query_eval.py` → 同上建索引（换 short 工作区）→ `eval_three_pipelines.py --benchmark-profile short`  
3. **Task C**：设置 `PERQA_OBS_CSV`、各 `PERQA_*_INDEX` 与模型环境变量 → `build_eval_indexes.py` → `eval_perqa_retrieval_export.py`  
4. **一键**：`run_all_retrieval_benchmarks.py`（参数见脚本说明）

**环境变量摘要**：见 `RETRIEVAL_BENCHMARKS.md` 表格。

---

## 八、与「用户草稿报告」的对照说明

| 草稿表述 | 与代码一致的处理方式 |
|----------|----------------------|
| 三任务「统一为同一套 Recall/MRR/NDCG」 | **部分一致**：三任务均支持这些指标，但 **Task A 聚焦表以 Hit@1/MRR 为主**，Task B 强调 Recall/NDCG，与 `README`/脚本设计一致 |
| Phase 1 Dynamic = RRF | **需修正**：当前 **`DynamicFusionOurs` 为分数加权**；RRF 存在于其他融合子路径，非 Dynamic 主线的默认行为 |
| Grid 得到「稳定区域」与角色分工 | **属结果解读**：需附 **完整 `dynamic_grid_search.csv`** 与多随机种子/多 split 再下结论；本文档不写入固定「最优区间」为系统默认值 |
| Query Router 升级为可学习模型 | **未在代码中实现**，本文档不展开 |

---

## 九、核心结论（可写入论文「小结」、与实现一致）

1. **多任务检索不存在单一「永远最优」的孤立模型**；BM25、SPLADE、Dense 在不同标签结构与查询长度下各有优势，需分任务报告。  
2. **评测协议修正**（Task B 多正例、Task C 语料对齐 + `index_meta`）是得到可信结论的前提。  
3. **当前融合主线**为 **三路分数 min–max 归一化 + 加权求和**；默认权重来自 **规则式 Query-aware 路由**；**Grid** 为可选的 **全局固定权重**搜索工具。  
4. **SPLADE 分桶训练**为可选训练范式，需与未微调及任务 A/B/C 联合报告，避免单任务夸大。

---

## 十、文件索引（代码侧）

| 内容 | 路径 |
|------|------|
| 三任务说明 | `examples/RETRIEVAL_BENCHMARKS.md` |
| Task A/B 评测 | `examples/eval_three_pipelines.py` |
| Task C 评测 / Grid | `examples/eval_perqa_retrieval_export.py` |
| Dynamic Fusion | `reqap_modular_retrieval/pipelines/dynamic_fusion.py` |
| 路由权重 | `reqap_modular_retrieval/routing/query_router.py` |
| 加权融合 | `reqap_modular_retrieval/fusion/weighted_sum.py`、`normalize.py` |
| SPLADE 训练（含 bucket） | `examples/train_splade_retriever.py` |
| 索引构建与指纹 | `examples/build_eval_indexes.py` |
| 一键三任务 | `examples/run_all_retrieval_benchmarks.py` |

---

*文档生成依据：仓库当前实现与 `RETRIEVAL_BENCHMARKS.md`；若脚本与本文冲突，以代码为准。*
