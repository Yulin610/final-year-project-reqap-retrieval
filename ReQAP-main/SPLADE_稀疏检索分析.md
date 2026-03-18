# SPLADE 稀疏检索分析

## 一、SPLADE 概述

**SPLADE** (Sparse Lexical And Expansion) 是一种基于神经网络的稀疏检索方法，它将查询和文档编码为高维稀疏向量（维度 = 词汇表大小），然后使用倒排索引进行高效检索。

### 1.1 核心思想

- **稀疏表示**：每个文档/查询被编码为词汇表大小的稀疏向量
- **可解释性**：可以知道哪些词匹配，匹配的权重是多少
- **倒排索引**：使用传统信息检索的倒排索引结构
- **神经扩展**：使用 MLM 模型自动扩展查询和文档

## 二、SPLADE 架构设计

### 2.1 整体架构

```
查询文本 (Query Text)
    ↓
SPLADE 模型编码
    ↓
稀疏向量 (Sparse Vector, 维度 = vocab_size)
    ↓
提取非零元素 (Non-zero tokens)
    ↓
倒排索引查找 (Inverted Index Lookup)
    ↓
分数计算 (Score Calculation)
    ↓
排序返回 (Ranked Results)
```

### 2.2 关键组件

1. **Splade 模型** (`models.py`)
   - 基于 MLM (Masked Language Model)
   - 输出维度 = 词汇表大小（如 BERT: 30522）
   - 使用 ReLU + Log 激活函数

2. **倒排索引** (`inverted_index.py`)
   - 存储每个词对应的文档列表
   - 存储每个词在文档中的权重

3. **检索器** (`sparse_retrieval.py`)
   - 查询编码
   - 索引查找
   - 分数计算

## 三、SPLADE 模型详解

### 3.1 模型结构

```python
class Splade(SiameseBase):
    def __init__(self, model_type_or_dir, agg="max", ...):
        # 使用 MLM 模型（如 BERT）
        # output_dim = vocab_size (30522 for BERT)
    
    def encode(self, tokens, is_q):
        # 1. 获取 MLM logits: (bs, seq_len, vocab_size)
        out = self.encode_(tokens, is_q)["logits"]
        
        # 2. 应用 ReLU + Log 激活
        activated = torch.log(1 + torch.relu(out))
        
        # 3. 聚合（max 或 sum）
        if self.agg == "max":
            return torch.max(activated * attention_mask, dim=1)
        else:
            return torch.sum(activated * attention_mask, dim=1)
```

### 3.2 编码过程

**输入**：文本序列（tokenized）
```
"I went running" → [101, 1045, 2097, 2734, 102]  # BERT token IDs
```

**处理**：
1. **MLM 前向传播**：得到每个位置的词汇表 logits
   ```
   Shape: (batch_size, seq_len, vocab_size)
   Example: (1, 5, 30522)
   ```

2. **激活函数**：`log(1 + ReLU(logits))`
   - ReLU：只保留正激活
   - Log：平滑权重分布
   - 结果：稀疏的权重向量

3. **聚合**：
   - **Max 聚合**：取每个词的最大激活值
   - **Sum 聚合**：累加所有位置的激活值
   - 结果：`(batch_size, vocab_size)` 稀疏向量

**输出**：稀疏向量
```
[0, 0, 0, 0.5, 0, ..., 0.3, 0, 0.8, ...]
 ↑  ↑  ↑   ↑   ↑        ↑   ↑   ↑
 大部分为 0，只有相关词有非零值
```

### 3.3 为什么使用稀疏表示？

**优势**：
1. **可解释性**：可以看到匹配的具体词汇
2. **高效索引**：只存储非零元素，节省空间
3. **快速检索**：倒排索引查找速度快
4. **查询扩展**：自动扩展相关词汇

**示例**：
```python
查询："running"
SPLADE 扩展后可能包含：
- "running" (0.8)
- "jogging" (0.5)  # 同义词
- "exercise" (0.3)  # 相关词
- "workout" (0.2)   # 相关词
```

## 四、倒排索引结构

### 4.1 索引格式

```python
class IndexDictOfArray:
    # 倒排索引：token_id → [doc_ids]
    index_doc_id = {
        1234: [doc_1, doc_2, doc_5, ...],  # token "running" 出现在这些文档
        5678: [doc_2, doc_3, ...],         # token "jogging" 出现在这些文档
        ...
    }
    
    # 权重：token_id → [weights]
    index_doc_value = {
        1234: [0.8, 0.6, 0.4, ...],  # 每个文档中该词的权重
        5678: [0.5, 0.3, ...],
        ...
    }
```

### 4.2 索引构建流程

```python
def run(self, collection_loader):
    for batch in collection_loader:
        # 1. 编码文档
        batch_documents = model(d_kwargs=inputs)["d_rep"]
        # Shape: (batch_size, vocab_size)
        
        # 2. 提取非零元素
        row, col = torch.nonzero(batch_documents, as_tuple=True)
        # row: 文档 ID
        # col: token ID (词汇表索引)
        # data: 权重值
        
        # 3. 添加到倒排索引
        for doc_id, token_id, weight in zip(row, col, data):
            index_doc_id[token_id].append(doc_id)
            index_doc_value[token_id].append(weight)
```

### 4.3 存储格式

使用 **HDF5** 格式存储索引：
- `index_doc_id_{token_id}`: 文档 ID 数组
- `index_doc_value_{token_id}`: 权重数组
- `doc_ids.pkl`: 文档 ID 映射

**优势**：
- 高效压缩
- 快速加载
- 支持大规模数据

## 五、检索流程详解

### 5.1 检索步骤

```python
def retrieve(self, query, involve_model=True, top_k=10, threshold=0):
    # 步骤 1: 编码查询
    if involve_model:
        query_rep = model(q_kwargs=processed_query)["q_rep"]
        # Shape: (vocab_size,) 稀疏向量
    else:
        query_rep = create_query_rep(input_ids, dim_voc)
        # 简单的 one-hot 编码
    
    # 步骤 2: 提取非零元素
    query_rep_nonzero = torch.nonzero(query_rep)
    # 只处理有激活的 token
    
    # 步骤 3: 倒排索引查找和分数计算
    scores, derivations = score_float(
        index_doc_ids,      # 倒排索引：token → docs
        index_doc_values,   # 权重：token → weights
        query_rep_nonzero,  # 查询的非零 token
        query_values,       # 查询的权重
        threshold,          # 阈值
        size_collection     # 文档总数
    )
    
    # 步骤 4: Top-K 选择
    if top_k > 0:
        filtered_indexes, scores = select_topk(...)
    
    # 步骤 5: 格式化结果
    return query_result, bow_rep
```

### 5.2 分数计算算法

```python
def score_float(inverted_index_ids, inverted_index_floats, 
                indexes_to_retrieve, query_values, threshold, size_collection):
    # 初始化分数数组
    scores = np.zeros(size_collection, dtype=np.float32)
    derivations = [[] for _ in range(size_collection)]
    
    # 对查询中的每个非零 token
    for token_id, query_weight in zip(indexes_to_retrieve, query_values):
        # 获取包含该 token 的所有文档
        doc_ids = inverted_index_ids[token_id]
        doc_weights = inverted_index_floats[token_id]
        
        # 计算分数：query_weight * doc_weight
        for doc_id, doc_weight in zip(doc_ids, doc_weights):
            score = query_weight * doc_weight
            scores[doc_id] += score  # 累加多个 token 的贡献
            derivations[doc_id].append((token_id, score))
    
    # 过滤阈值
    filtered_indexes = np.argwhere(scores > threshold)[:, 0]
    return filtered_indexes, scores, derivations
```

**分数公式**：
```
Score(doc, query) = Σ (query_weight[token] * doc_weight[token])
                    token in query ∩ doc
```

### 5.3 可解释性

SPLADE 的一个重要特性是**可解释性**：

```python
derivation = [
    {"token": "running", "score": 0.8},
    {"token": "jogging", "score": 0.5},
    {"token": "exercise", "score": 0.3}
]
```

可以看到：
- 哪些词匹配了
- 每个词的贡献分数
- 为什么这个文档被检索到

## 六、与 Dense 检索的对比

### 6.1 表示方式对比

| 特性 | SPLADE (稀疏) | Dense (密集) |
|------|--------------|--------------|
| **维度** | 词汇表大小 (~30K) | 嵌入维度 (384/768) |
| **稀疏性** | 高度稀疏（大部分为 0） | 密集（所有维度都有值） |
| **语义** | 词级别匹配 | 语义级别匹配 |
| **可解释性** | ✅ 高（可见匹配的词） | ❌ 低（只有相似度分数） |

### 6.2 索引结构对比

**SPLADE - 倒排索引**：
```
Token "running" → [doc_1, doc_2, doc_5, ...]
Token "jogging" → [doc_2, doc_3, ...]
```

**Dense - FAISS 向量索引**：
```
所有文档的嵌入向量 → FAISS Index
查询向量 → 相似度搜索
```

### 6.3 检索流程对比

**SPLADE**：
1. 查询编码为稀疏向量
2. 提取非零 token
3. 倒排索引查找
4. 累加分数
5. 排序

**Dense**：
1. 查询编码为密集向量
2. FAISS 相似度搜索
3. 返回 Top-K

### 6.4 优势对比

**SPLADE 优势**：
- ✅ **精确匹配**：擅长关键词匹配
- ✅ **可解释性**：可以看到匹配的词
- ✅ **查询扩展**：自动扩展相关词汇
- ✅ **内存效率**：只存储非零元素

**Dense 优势**：
- ✅ **语义匹配**：理解同义词和语义相似性
- ✅ **简洁表示**：低维向量（384 vs 30K）
- ✅ **快速搜索**：FAISS 优化
- ✅ **跨语言**：多语言模型支持

### 6.5 互补性

两者互补，混合检索可以结合优势：

```
查询："I went running"

SPLADE 匹配：
- "running" (精确匹配)
- "jogging" (扩展词)
- "exercise" (扩展词)

Dense 匹配：
- "I did a workout" (语义相似)
- "I exercised" (语义相似)
- "I went for a jog" (语义相似)

混合结果：覆盖更全面
```

## 七、性能特性

### 7.1 索引构建

**SPLADE**：
- 需要编码所有文档
- 构建倒排索引
- 存储格式：HDF5
- 时间：取决于文档数量和模型大小

**存储空间**：
- 只存储非零元素
- 对于稀疏表示，空间效率高

### 7.2 检索速度

**SPLADE**：
- 只处理查询中的非零 token
- 倒排索引查找：O(相关文档数)
- 通常很快，除非查询扩展出很多词

**优化**：
- 使用阈值过滤
- Top-K 提前终止
- 批量处理

### 7.3 内存使用

**SPLADE**：
- 倒排索引：只存储非零元素
- 对于稀疏数据，内存效率高
- 但词汇表维度大（~30K）

## 八、设计模式分析

### 8.1 适配器模式

```python
class Splade(SiameseBase):
    # 适配不同的 MLM 模型
    # 统一接口，支持 BERT、RoBERTa 等
```

### 8.2 策略模式

```python
# 聚合策略
if self.agg == "max":
    return torch.max(...)
else:
    return torch.sum(...)
```

### 8.3 模板方法模式

```python
def retrieve(self, query, ...):
    # 1. 编码查询（模板方法）
    query_rep = self._encode_query(query)
    
    # 2. 索引查找（固定流程）
    scores = self._lookup_index(query_rep)
    
    # 3. 格式化结果（固定流程）
    return self._format_results(scores)
```

## 九、关键设计决策

### 9.1 为什么使用 MLM？

**设计决策**：使用 Masked Language Model 而不是普通的编码器

**原因**：
1. **词汇表输出**：MLM 输出维度 = 词汇表大小，适合稀疏表示
2. **查询扩展**：MLM 可以预测相关词汇
3. **预训练优势**：利用大规模预训练的 MLM 模型

### 9.2 为什么使用 ReLU + Log？

**设计决策**：`log(1 + ReLU(logits))`

**原因**：
1. **ReLU**：只保留正激活，产生稀疏性
2. **Log**：平滑权重分布，避免极端值
3. **稀疏性**：大部分元素为 0，只有相关词有激活

### 9.3 为什么使用倒排索引？

**设计决策**：使用倒排索引而不是直接计算相似度

**原因**：
1. **效率**：只处理非零元素
2. **可扩展性**：支持大规模文档集合
3. **传统优势**：利用成熟的 IR 技术

### 9.4 为什么支持两种聚合方式？

**设计决策**：支持 Max 和 Sum 聚合

**Max 聚合**：
- 取每个词的最大激活
- 更稀疏，更关注最强信号

**Sum 聚合**：
- 累加所有位置的激活
- 更密集，考虑所有信号

## 十、使用场景

### 10.1 适合 SPLADE 的场景

✅ **关键词检索**：需要精确匹配关键词  
✅ **可解释性要求**：需要知道为什么检索到某个文档  
✅ **查询扩展**：需要自动扩展相关词汇  
✅ **结构化数据**：事件、属性等结构化信息  

### 10.2 不适合的场景

❌ **语义相似性**：同义词、语义转换  
❌ **跨语言检索**：多语言场景  
❌ **长文档**：文档很长时，稀疏向量可能很密集  

## 十一、总结

### 11.1 核心特点

1. **稀疏表示**：高维稀疏向量（维度 = 词汇表大小）
2. **可解释性**：可以看到匹配的具体词汇
3. **查询扩展**：自动扩展相关词汇
4. **倒排索引**：高效的检索结构

### 11.2 设计优势

✅ **可解释性强** - 可以看到匹配的词  
✅ **精确匹配** - 擅长关键词检索  
✅ **自动扩展** - 神经查询扩展  
✅ **高效索引** - 只存储非零元素  

### 11.3 与 Dense 的互补

- **SPLADE**：精确匹配 + 可解释性
- **Dense**：语义匹配 + 简洁表示
- **混合**：结合两者优势，覆盖更全面

---

**总结**：SPLADE 稀疏检索是一个设计精良的神经检索系统，它结合了传统信息检索的倒排索引和现代神经网络的表示学习，在可解释性和精确匹配方面具有独特优势。









