下面给你一个**更偏面试/落地的 MiniOneRec 优化方向总结**。核心思路是：不要只说“加一个 loss”“换个 reward”，而是把 MiniOneRec 包装成一个更完整的 **industrial-style generative recommender pipeline**：从 **SID 构建 → SID 质量控制 → 生成式召回/重排 → RL/偏好对齐 → 低成本推理 → 反事实评估** 全链路优化。

---

## 一、最近几年相关工作的主线

### 1. Generative Retrieval / SID 是主线，不是简单“LLM 推荐”

Google/TIGER 提出的核心范式是：不再把推荐看成对 item embedding 做 ANN 检索，而是让模型**自回归生成目标 item 的 Semantic ID**。TIGER 用 RQ-VAE 从 item 文本 embedding 构造 SID，然后用 Transformer 根据用户历史 SID 序列预测下一个 SID，并且强调 SID 能提升冷启动和泛化能力。([arXiv](https://arxiv.org/abs/2305.05065 "[2305.05065] Recommender Systems with Generative Retrieval"))

MiniOneRec 本质上就是沿着这条路线：用 RQ-VAE 构造 SID，再做 SFT 和推荐导向 RL。论文中也明确说 MiniOneRec 是一个开源 generative recommendation 框架，包含 SID construction、SFT、recommendation-oriented RL，并使用 Qwen 0.5B 到 7B 做后训练。([arXiv](https://arxiv.org/abs/2510.24431 "[2510.24431] MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation"))

**面试表达可以说：**

> MiniOneRec 不是把 LLM 当成 reranker，而是把推荐目标空间离散化成 SID token space，让推荐问题变成 constrained sequence generation。我的优化重点不是简单换 backbone，而是围绕 SID 表示质量、生成约束、偏好对齐和推理成本做系统级改造。

---

### 2. 只用文本 SID 不够，工业界更看重 semantic + collaborative + multimodal

LC-Rec 指出 LLM 的语言语义和推荐系统的协同语义之间存在 gap，因此提出用 learning-based vector quantization 和 alignment tuning，把 language semantics 与 collaborative semantics 融合进 item index。([arXiv](https://arxiv.org/abs/2311.09049 "[2311.09049] Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation"))

TokenRec 更直接地把用户/物品的协同过滤表示量化成离散 token，目标是把高阶 collaborative knowledge 注入 LLM-compatible token，同时减少 beam search 这类自回归推理成本。([arXiv](https://arxiv.org/abs/2406.10450 "[2406.10450] TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation"))

Spotify 2025 的 search + recommendation 工作进一步说明：**搜索任务偏内容语义，推荐任务偏协同语义**，单独为搜索或推荐训练的 SID 在另一个任务上可能退化；他们用 multi-task bi-encoder 同时学习 query-item 搜索对和 item-item 共现对，再用 RQ-KMeans 生成更均衡的跨任务 SID。([Spotify Research](https://research.atspotify.com/2025/9/semantic-ids-for-generative-search-and-recommendation "Semantic IDs for Generative Search and Recommendation | Spotify Research"))

**对 MiniOneRec 的启发：**

MiniOneRec 当前如果主要从 item 文本 embedding 出发，容易导致 SID 更像“语义类别编码”，而不是“推荐空间中的位置编码”。实际落地应该让 SID 同时回答四个问题：

1. 这个 item 是什么；
    
2. 它和哪些 item 语义相似；
    
3. 它和哪些 item 被同一类用户共同喜欢；
    
4. 它在业务推荐空间里属于头部、长尾、冷启动还是潜在兴趣迁移点。
    

---

### 3. 工业落地里 SID 最大问题不是“能不能生成”，而是 codebook collapse、SID collision 和 SID-to-item resolution

Snapchat 2026 的实践非常有参考价值。他们总结 SID 应用到工业推荐时遇到两个关键问题：一是 RQ-VAE codebook collapse，部分 code 被过度使用，很多 code 死掉；二是 SID-to-item resolution，因为多个 item 可能共享同一个 SID，生成 SID 后还需要把 SID 映射回具体 item。([arXiv](https://arxiv.org/html/2604.03949v1 "Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices"))

他们的实用做法包括：用 STE 让梯度影响整个 codebook、融合多源 embedding 缓解 collapse、对同一个 SID bucket 内的 item 做轻量二阶段排序，并且发现“优先从 top SID 里取更多 item”比“从更多低分 SID 里各取一点”效果更好。([arXiv](https://arxiv.org/html/2604.03949v1 "Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices"))

还有一个重要结论：**SID uniqueness 不是越高越好**。Snapchat 的实验发现，当 uniqueness 超过一定阈值后，继续追求更高唯一性对 GR 性能提升很小，uniqueness 更适合作为避免 collapse 的 sanity check，而不是最终质量指标。([arXiv](https://arxiv.org/html/2604.03949v1 "Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices"))

**对 MiniOneRec 的启发：**

MiniOneRec 的 SID 优化不能只汇报“codebook 使用率提升了”，还要证明 SID 对下游推荐有效。更好的实验指标应该包括：

- code usage entropy；
    
- code perplexity；
    
- SID collision rate；
    
- semantic cohesion；
    
- preference discrimination；
    
- long-tail Recall/NDCG；
    
- cold-start Recall/NDCG；
    
- SID bucket 内 rerank 后的最终 HR@K/NDCG@K。
    

---

### 4. 生成式检索里 prefix-level optimization 很关键

NCI 把检索看成 seq2seq 生成 docid，并提出 prefix-aware decoder、query generation、semantic document identifiers 和 consistency regularization。([NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a46156bd3579c3b268108ea6aca71d13-Abstract-Conference.html "A Neural Corpus Indexer for Document Retrieval"))

RIPOR 进一步指出，生成式检索不能只在完整 docid 序列上打分，因为自回归生成过程中每一个 prefix 都会影响最终候选空间，因此提出 prefix-oriented ranking optimization，并且认为初始 ID 应该基于 query-document relevance，而不只是文档语义。([arXiv](https://arxiv.org/abs/2311.09134 "[2311.09134] Scalable and Effective Generative Information Retrieval"))

**对 MiniOneRec 的启发：**

MiniOneRec 现在如果只是让模型生成完整 SID，然后看最终 item 是否命中，训练信号太粗。可以把 SID 预测拆成：

```text
level-1 coarse semantic cluster
level-2 collaborative sub-cluster
level-3 fine-grained item bucket
level-4 optional disambiguation / duplicate token
```

然后在每一层 prefix 上都加 ranking/contrastive supervision。这样模型不是只学“最后一个 item token”，而是学“从粗到细定位 item 空间”。

---

### 5. 真实工业推荐已经从 next-item 走向 session/list generation + preference alignment

OneRec 很有参考价值，因为它明确不是只做 retrieval selector，而是试图统一 retrieve 和 rank。它采用 encoder-decoder 结构、session-wise generation、sparse MoE 扩容，以及 DPO-style iterative preference alignment，并在快手主场景部署，报告 watch-time 提升。([arXiv](https://arxiv.org/abs/2502.18965 "[2502.18965] OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment"))

这对 MiniOneRec 的启发很直接：MiniOneRec 当前 next-item prediction 更像学术 benchmark；如果想面试更像落地项目，应该把目标从“预测下一个 item”扩展成“生成一组候选推荐列表”，然后在列表层面优化 NDCG、diversity、novelty、coverage 和 latency。

---

## 二、基于 MiniOneRec 的优化方向，按面试价值排序

## 方向 1：多信号融合的 Collaborative-Semantic SID 构建

这是最值得做的方向，和作者未来方向 11.2、11.3、11.6 都有关。

### 核心问题

MiniOneRec 当前 SID 主要来自 item 文本 embedding。问题是：

```text
文本相似 ≠ 推荐相似
语义相似 ≠ 用户会共同点击
同类商品 ≠ 同一用户偏好
```

比如两个商品标题很像，但一个是高端款，一个是低价替代款，它们在推荐空间里的用户群可能完全不同。反过来，两个 item 文本不相似，但经常被同一类用户连续购买，它们在推荐空间里应该更接近。

### 可以怎么改

构造一个 multi-signal item representation：

```text
z_item = Fuse(
    text_embedding,
    collaborative_embedding,
    category_embedding,
    brand_embedding,
    price_bucket_embedding,
    review_sentiment_embedding,
    optional_image_embedding
)
```

其中：

- text embedding：来自 Qwen/BGE/Sentence-T5；
    
- collaborative embedding：来自 SASRec、LightGCN、BPR、item2vec 或共现图；
    
- category/brand/price：来自 metadata；
    
- review sentiment：从 review 文本提取情感、质量、场景标签；
    
- image embedding：如果是 Amazon 商品，可以用 CLIP/BLIP 取图像向量。
    

然后把融合后的 `z_item` 输入 RQ-VAE / RQ-KMeans / MM-RQ-VAE 生成 SID。类似思路在 LC-Rec、TokenRec、Spotify multi-task SID、MME-SID 中都能找到支撑。([arXiv](https://arxiv.org/abs/2311.09049 "[2311.09049] Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation"))

### 推荐的技术实现

可以设计三种版本，便于做 ablation：

```text
Baseline SID:
text embedding → RQ-VAE → SID

Version A:
text embedding + metadata embedding → RQ-VAE → SID

Version B:
text embedding + collaborative embedding → RQ-VAE → SID

Version C:
text + collaborative + category/brand/price/review/image → gated fusion → RQ-VAE → SID
```

融合模块不要太复杂，面试时可以说：

```text
h_text = W_t e_text
h_cf   = W_c e_cf
h_meta = W_m e_meta

gate = softmax(W_g [h_text; h_cf; h_meta])
z = gate_t h_text + gate_c h_cf + gate_m h_meta
```

### 训练 loss

建议不要只用 reconstruction loss，可以做成：

```text
L = L_recon
  + λ1 L_commit
  + λ2 L_code_usage
  + λ3 L_item_item_contrastive
  + λ4 L_category_hierarchy
  + λ5 L_popularity_balance
```

其中：

- `L_item_item_contrastive`：让共现 item / 同用户序列中相邻 item 更近；
    
- `L_category_hierarchy`：让 SID 前几层能预测类目层级；
    
- `L_popularity_balance`：避免头部 item 抢占过多 code；
    
- `L_code_usage`：避免 codebook collapse。
    

### 面试亮点说法

> 我没有简单把更多特征 concat 进去，而是把 SID 从 text-only semantic tokenizer 改成 collaborative-semantic tokenizer。这样 SID 既保留 item 语义，又编码用户行为图中的邻近关系，能同时改善冷启动、长尾推荐和跨域迁移。

---

## 方向 2：Codebook 使用均衡 + SID 质量评估体系

这个方向非常适合面试，因为能体现你理解 RQ-VAE 的实际训练问题。

### 核心问题

RQ-VAE 很容易出现：

```text
少数 code 高频使用
大量 code 死亡
不同 item 映射到同一个 SID
SID collision 后无法定位具体 item
```

Snapchat 的工业实践明确把 codebook collapse 和 SID-to-item resolution 作为 SID 落地的关键挑战。([arXiv](https://arxiv.org/html/2604.03949v1 "Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices"))

### 可以怎么改

第一层是训练期平衡：

```text
1. usage entropy regularization
2. KL-to-uniform code usage loss
3. EMA codebook update
4. dead code re-initialization
5. Gumbel-softmax exploration
6. STE through full codebook
7. non-uniform quantization
```

CARD 2026 提出 non-uniform quantization，把 skewed semantic embedding distribution 映射到更平衡的 latent space，以改善 codebook utilization 和 quantization accuracy。([arXiv](https://arxiv.org/abs/2604.26427 "[2604.26427] CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation"))

R3-VAE 2026 则从训练稳定性和 SID 质量评估角度出发，引入 reference vector、dot-product rating mechanism，以及 semantic cohesion / preference discrimination 两个 SID 评估指标。([arXiv](https://arxiv.org/abs/2604.11440 "[2604.11440] R3-VAE: Reference Vector-Guided Rating Residual Quantization VAE for Generative Recommendation"))

第二层是推理期 disambiguation：

```text
generated SID → candidate bucket → lightweight reranker → final item
```

也就是不强求每个 item 都有唯一 SID，而是承认 SID 是一个 semantic bucket，再用轻量特征做 bucket 内排序。

### 推荐实验指标

不要只报 HR@K/NDCG@K，建议加：

```text
SID uniqueness
code usage entropy
dead code ratio
average bucket size
max bucket size
long-tail bucket coverage
semantic cohesion
preference discrimination
cold-start Recall@K
long-tail Recall@K
```

### 面试亮点说法

> 我把 SID tokenizer 当成推荐系统里的“可学习索引结构”，所以不仅看 reconstruction loss，也看它对下游 retrieval 的可区分性、bucket 分布、长尾覆盖和冲突消解成本。这样比只优化 RQ-VAE loss 更接近工业落地。

---

## 方向 3：Prefix-aware SID 训练与 Trie-constrained decoding

这个方向和生成式检索联系最紧，适合把 MiniOneRec 从“能跑通”提升到“更像 generative retrieval system”。

### 核心问题

SID 是一个序列，例如：

```text
item → [c1, c2, c3, c4]
```

传统训练通常只看完整 SID 是否生成对，但在自回归生成里：

```text
第 1 个 token 决定粗粒度候选空间
第 2 个 token 决定子簇
第 3/4 个 token 决定具体 bucket 或 item
```

如果前缀错了，后面再对也没用。

RIPOR 的核心启发就是：生成式检索应该优化 prefix-level ranking，而不只是完整 ID 序列打分。([arXiv](https://arxiv.org/abs/2311.09134 "[2311.09134] Scalable and Effective Generative Information Retrieval"))

### 可以怎么改 MiniOneRec

在 SFT 阶段增加 prefix supervision：

```text
target SID = [c1, c2, c3, c4]

loss = CE(c1)
     + CE(c1,c2)
     + CE(c1,c2,c3)
     + CE(c1,c2,c3,c4)
```

更推荐的表达是：

```text
L = L_token_CE + λ L_prefix_rank + μ L_valid_prefix
```

其中：

- `L_token_CE`：普通 token-level 交叉熵；
    
- `L_prefix_rank`：让正样本 SID prefix 分数高于 hard negative prefix；
    
- `L_valid_prefix`：约束模型不要生成训练集中不存在的 prefix。
    

推理时构建一个 SID trie：

```text
root
 ├── c1
 │    ├── c2
 │    │    ├── c3
 │    │    │    └── c4
```

解码时只允许生成 trie 中存在的合法 token。这样可以减少 invalid SID，提高生成效率，也能降低 beam search 的无效分支。

### 面试亮点说法

> 我把 SID 解码看成层次化检索，而不是普通文本生成。因此我加了 prefix-level supervision 和 trie-constrained decoding，使模型在每个前缀层级都尽量朝正确候选簇收缩，减少无效 SID 和 beam search 成本。

---

## 方向 4：从 next-item prediction 扩展到 session/list generation

这个方向最像工业推荐，因为真实系统不是只推荐一个 item，而是一次返回一个候选列表。

### 核心问题

MiniOneRec 当前目标更接近：

```text
history → next item SID
```

但真实推荐一般是：

```text
history + context → top-K candidate list
```

而且需要考虑列表内多样性、去重、覆盖、业务约束、曝光疲劳等。

OneRec 的 session-wise generation 就是这个方向：它不再逐点生成单个 item，而是生成一个更连贯的 session-level 推荐结果，并用 DPO 做 preference alignment。([arXiv](https://arxiv.org/abs/2502.18965 "[2502.18965] OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment"))

### 可以怎么改

把训练样本从：

```text
[user history] → next item
```

改成：

```text
[user history] → next K interacted items / session items
```

例如：

```text
Input:
用户过去 20 个 item SID

Output:
未来 session 中点击/购买/高停留的 5 个 item SID
```

SFT 阶段训练 list generation，RL 阶段优化 list-level reward：

```text
R_list = α NDCG@K
       + β HR@K
       + γ Diversity@K
       + δ Novelty@K
       - λ RepeatPenalty
       - η PopularityBias
```

### 关键注意点

不要直接让 LLM 自由生成一大串 item，否则会有重复、非法 SID、曝光不可控。建议：

```text
1. trie-constrained decoding 保证 SID 合法；
2. generated SID bucket 内做 lightweight rerank；
3. list-level post-processing 做去重、多样性和业务规则过滤；
4. 记录每一步 latency。
```

### 面试亮点说法

> 我把 MiniOneRec 从 next-item benchmark 改成 list-wise generative recommendation。这样优化目标从单点命中变成列表质量，更接近真实推荐系统的 candidate generation 阶段。

---

## 方向 5：更真实的 RL reward 与 preference alignment

这个方向对应作者的 11.4、11.5、11.7，也很适合面试。

### 当前问题

MiniOneRec 采用 ACC reward + rank reward 是合理的 baseline，但真实推荐里：

```text
未点击 ≠ 不喜欢
未出现 ≠ 负样本
点击 ≠ 长期满意
短期准确率 ≠ 推荐系统最终目标
```

因此 RL reward 如果只惩罚“没命中真实 next item”，会误伤很多 potential positive items。

### 可以怎么改

设计 soft positive reward：

```text
R = R_exact
  + R_semantic
  + R_collaborative
  + R_rank
  + R_diversity
  + R_novelty
  - R_popularity_bias
  - R_latency
```

具体解释：

```text
R_exact:
生成 item 等于真实交互 item，给最高奖励。

R_semantic:
生成 item 与真实 item 文本/类目/属性相似，给部分奖励。

R_collaborative:
生成 item 与真实 item 在 item-item co-click graph 中接近，给部分奖励。

R_rank:
真实 item 或同类 positive item 排名越靠前奖励越高。

R_diversity:
列表内 item 不要全部来自同一个类目或同一个 SID prefix。

R_novelty:
适当奖励非头部但相关的长尾 item。

R_popularity_bias:
过度推荐热门 item 扣分。

R_latency:
beam size、生成 token 数、rerank 候选数过大扣分。
```

LLM + RL 优化 novelty 的相关工作也指出，novelty/top-k 目标因为排序不可导、候选空间巨大，很难直接优化，可以把 list reward 拆成 item-wise reward 来降低复杂度。([arXiv](https://arxiv.org/abs/2406.14169 "[2406.14169] Optimizing Novelty of Top-k Recommendations using Large Language Models and Reinforcement Learning"))

### Preference alignment 怎么做

不要只用 PPO/GRPO，也可以加入 DPO-style preference pair：

```text
chosen:
真实点击 item / 高评分 item / 高停留 item / 多正样本 item

rejected:
曝光未点击 item / 随机负样本 / 高流行但不相关 item / 低相似 item
```

如果没有曝光日志，可以构造弱偏好：

```text
chosen = ground-truth item
rejected = popularity-matched negative
rejected = same-category hard negative
rejected = generated invalid SID
```

OneRec 也采用了 DPO-style iterative preference alignment，并且针对推荐系统“一次展示机会，很难同时获得正负样本”的问题设计 reward model 和采样策略。([arXiv](https://arxiv.org/abs/2502.18965 "[2502.18965] OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment"))

### 面试亮点说法

> 我没有把未点击 item 粗暴当负样本，而是设计 soft-positive reward：语义相似、协同相似、多正样本和曝光信息都能提供部分奖励。这样可以缓解 implicit feedback 里的 false negative 问题，让 RL 更符合真实推荐反馈。

---

## 方向 6：两阶段 generative retrieval + lightweight reranking，降低推理成本

这个方向非常落地。

### 核心问题

LLM-based generative recommendation 的主要瓶颈是：

```text
自回归生成慢
beam search 成本高
长历史输入导致 KV cache 大
生成 SID 后还要映射 item
7B 模型在线部署成本高
```

Meta 的 HSTU 工作强调推荐场景需要针对高基数、非平稳、流式行为数据重新设计架构；他们报告 HSTU 在长序列上比 FlashAttention2-based Transformer 更快，并在生产 A/B 中带来收益。([arXiv](https://arxiv.org/abs/2402.17152 "[2402.17152] Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"))

TokenRec 也明确提出要减少自回归 decoding 和 beam search 的时间成本。([arXiv](https://arxiv.org/abs/2406.10450 "[2406.10450] TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation"))

### 可以怎么改 MiniOneRec

推荐改成：

```text
Stage 1: MiniOneRec 生成 top-M SID buckets
Stage 2: bucket 内 lightweight reranker 选 top-K item
```

Stage 2 可以用：

```text
DIN / DCN / LightGBM / small MLP / SASRec score / popularity + freshness heuristic
```

实际线上部署时，MiniOneRec 不一定直接输出最终 item，而是作为新的 candidate generator，与传统召回并联：

```text
传统召回：itemCF / ANN / graph recall / popular recall
生成式召回：MiniOneRec top SID buckets
融合召回：merge + dedup + ranker
```

### 可做的工程优化

```text
1. LoRA / QLoRA 替代全参微调；
2. distill 7B → 1.5B / 0.5B；
3. prefix KV cache 复用用户长期历史；
4. SID trie constrained decoding；
5. beam search 改成 top-p sampling + validity filter；
6. top SID depth fetching 替代大范围 breadth fetching；
7. 缓存用户历史 representation；
8. 记录 tokens/request、latency、GPU memory。
```

Snapchat 的实践也支持“top SID depth fetching”和“bucket 内 relevance-guided mapping”这种二阶段策略。([arXiv](https://arxiv.org/html/2604.03949v1 "Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices"))

### 面试亮点说法

> 我不会把 LLM 直接放在线上全量排序，而是把它定位成 generative candidate generator。它生成少量高置信 SID bucket，再由轻量 reranker 做 item-level disambiguation。这样既保留生成式推荐的泛化能力，又能控制延迟和 GPU 成本。

---

## 方向 7：搜索 + 推荐统一建模，做成跨任务泛化亮点

这个方向适合把项目讲得更有“生成式搜索推荐”味道。

### 核心问题

很多平台同时有：

```text
搜索：query → item
推荐：user history → item
```

传统系统通常是两套模型。生成式检索的一个重要趋势是用统一模型处理 query understanding、retrieval、recommendation、reranking、explanation。Spotify 的研究就指出，联合生成式模型有机会统一 search 和 recommendation，但 SID 构建必须兼顾搜索语义与推荐协同语义。([arXiv](https://arxiv.org/abs/2410.16823 "[2410.16823] Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other?"))

### 可以怎么改 MiniOneRec

把训练数据扩展成多任务 instruction 格式：

```text
Task 1: recommendation
Input: user history SID
Output: next item SID

Task 2: search
Input: query text
Output: clicked item SID

Task 3: item-to-item
Input: source item SID
Output: co-click item SID

Task 4: explanation
Input: user history + target SID
Output: short reason
```

训练目标：

```text
L = L_rec + λ L_search + μ L_item2item + ν L_explain
```

这样可以形成统一模型：

```text
query / user history / item context → generated SID / explanation
```

### 面试亮点说法

> 我把 MiniOneRec 从单任务序列推荐扩展成 search-recommendation unified generative retrieval。搜索提供 content relevance，推荐提供 collaborative preference，两者共同正则化 item representation，可以缓解 popularity bias 和冷启动问题。

---

## 方向 8：反事实评估与曝光日志建模

这个方向最能体现你知道 offline recommendation 的坑。

### 核心问题

MiniOneRec 如果只用 Amazon Review 的 next-item split，实际评估是：

```text
看模型能不能预测用户真实交互过的 item
```

但真实系统里还有曝光集合：

```text
展示了哪些 item？
用户点击了哪些？
哪些展示了但没点击？
哪些根本没曝光？
```

如果没有曝光日志，把所有 unobserved item 当负样本是不合理的。

### 可以怎么改

如果有曝光日志：

```text
positive = clicked / purchased / long dwell
negative = exposed but skipped
unobserved = unknown，不直接当负样本
```

如果没有曝光日志：

```text
1. popularity-matched negative sampling
2. same-category hard negative
3. item-item graph distant negative
4. soft negative instead of hard negative
5. IPS/SNIPS-style offline correction as optional extension
```

反事实推荐评估领域常用 IPS/SNIPS 来校正曝光偏差，近期也有工作继续强调 IPS-weighted training 和 SNIPS evaluation 对处理 biased exposure 的价值。([arXiv](https://arxiv.org/html/2509.00333v1?utm_source=chatgpt.com "Counterfactual Risk Minimization with IPS-Weighted BPR ..."))

### 面试亮点说法

> 我不会把未交互样本直接当负样本，因为推荐日志是 missing-not-at-random。我的评估会区分 exposed negative 和 unobserved item，并用 propensity-aware 或 soft-negative reward 减少曝光偏差。

---

## 三、最终推荐你优先做的 4 个项目亮点

如果你要把 MiniOneRec 做成一个面试项目，我建议优先选这四个，不要八个都做。

### P0：Collaborative-Semantic SID Tokenizer

**目标：** 替换 text-only RQ-VAE SID。

**做法：**

```text
text embedding
+ collaborative item embedding
+ category / brand / price
+ review sentiment
→ gated fusion
→ balanced RQ-VAE
→ SID
```

**实验：**

```text
Text-only SID
vs Text+Meta SID
vs Text+CF SID
vs Text+CF+Meta SID
```

**指标：**

```text
HR@10
NDCG@10
long-tail HR@10
cold-start HR@10
SID entropy
collision rate
code usage perplexity
```

---

### P0：Balanced Codebook + SID Quality Dashboard

**目标：** 解决 codebook collapse，不只看 RQ-VAE reconstruction loss。

**做法：**

```text
entropy regularization
dead code re-initialization
EMA update
STE full-codebook update
popularity-aware balancing
```

**展示：**

```text
训练前后 code usage histogram
不同 SID 层级的 entropy
bucket size distribution
SID collision heatmap
long-tail item coverage
```

这部分非常适合面试，因为图一出来很直观。

---

### P1：Prefix-aware Training + Constrained Decoding

**目标：** 减少 invalid SID，提高生成召回质量和推理效率。

**做法：**

```text
prefix-level CE / ranking loss
SID trie
valid token mask
top SID bucket retrieval
bucket-level rerank
```

**指标：**

```text
invalid SID ratio
beam search latency
tokens/request
Recall@K
NDCG@K
```

---

### P1：List-wise RL Reward / DPO Alignment

**目标：** 从 next-item accuracy 扩展到更真实的推荐列表优化。

**做法：**

```text
R = accuracy
  + rank
  + semantic similarity
  + collaborative similarity
  + diversity
  + novelty
  - popularity bias
  - latency
```

**偏好样本构造：**

```text
chosen: clicked / purchased / high-rating / co-click positive
rejected: exposed non-click / popularity-matched negative / invalid SID / hard negative
```

**指标：**

```text
HR@K
NDCG@K
ILD diversity
coverage
novelty
long-tail ratio
latency
```

---

## 四、可以在面试中这样总结你的 MiniOneRec 优化项目

你可以组织成下面这段：

> MiniOneRec 原始框架已经打通了 SID construction、SFT 和 recommendation-oriented RL，但它更像一个 open-source reproduction pipeline。我的优化重点是把它往工业生成式推荐系统靠近。第一，我把 text-only SID 改成 collaborative-semantic SID，引入用户交互图、类目、品牌、价格、review 和可选图像信息，使 SID 同时表达 item 语义和推荐空间位置。第二，我针对 RQ-VAE 的 codebook collapse 和 SID collision 设计了 code usage balancing、dead code reinitialization 和 SID quality dashboard，不只看 reconstruction loss，而是评估 entropy、collision、long-tail coverage 和 downstream NDCG。第三，我把 SID 解码改成 prefix-aware 和 trie-constrained decoding，减少 invalid SID 和 beam search 成本。第四，我把 RL reward 从 ACC/rank 扩展成 list-wise multi-objective reward，加入 diversity、novelty、popularity debias 和 latency penalty，使优化目标更接近真实推荐系统。

---

## 五、最适合写进简历的一句话

> 基于 MiniOneRec 构建工业化生成式推荐优化框架：设计 collaborative-semantic SID tokenizer，缓解 RQ-VAE codebook collapse 与 SID collision；引入 prefix-aware constrained decoding 和 list-wise multi-objective RL reward，在 HR/NDCG、长尾覆盖、多样性和推理延迟之间实现更可控的推荐效果。