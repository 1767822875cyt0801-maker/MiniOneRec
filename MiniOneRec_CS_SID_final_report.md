# MiniOneRec CS-SID Final Report

## 1. 项目背景

MiniOneRec 是一个生成式推荐系统复现项目。与传统推荐模型直接输出 item score 不同，MiniOneRec 的模型输出是语义 ID：

```text
Semantic ID, SID
例如：<a_12><b_34><c_56>
```

在这种范式下，SID 不是普通标签，而是连接 item 表示、训练数据、constrained decoding、候选生成和最终 item-level 推荐指标的核心结构。

因此 SID 质量会直接影响：

- 生成模型是否容易学习；
- constrained decoding 是否稳定；
- beam 中是否包含正确 SID；
- SID prefix 是否能形成有意义的候选邻域；
- SID-level 命中能否转化成 item-level HR/NDCG。

本项目的优化目标是：在不把 RL 作为主线的情况下，系统提升 SID 结构质量，并建立从 SID-level 到 item-level 的完整诊断评估闭环。

## 2. 原始问题

官方 Text-SID baseline 主要依赖文本语义 embedding。它是一个可用的强 baseline，但存在几个问题：

1. SID collision。
   多个 item 可能共享同一个 SID，导致 SID-level hit 不等于 item-level hit。

2. Prefix 组织未必符合推荐行为。
   文本相似 item 未必是用户行为上可替代的 item。

3. Head / mid / tail 表现差异明显。
   只看 overall HR/NDCG 无法解释收益来源。

4. SID-level 指标不足以代表最终推荐质量。
   生成模型输出 SID，但线上推荐最终需要 item candidates。

因此，本项目将优化目标从“继续堆 SFT/RL”转向：

```text
诊断 SID -> 构建 CS embedding -> 生成 CS-SID -> 重写训练数据
-> sid-only SFT 消融 -> candidate expansion -> lightweight rerank
```

## 3. 方法

### 3.1 SID 契约与 Manifest

首先建立统一 SID 契约：

```text
item2sid.json
sid2items.json
valid_sid_set.json
item_mapping.json
experiment_manifest.json
```

这样可以保证：

- train/valid/test CSV 使用的 SID；
- SFT tokenizer 扩展的 SID token；
- evaluate constrained decoding 使用的 SID trie；
- calc/calc_plus 使用的 valid SID set；

都来自同一个 SID version。

### 3.2 SID 静态诊断

通过 `analyze_sid.py` 统计：

- collision rate；
- collided item rate；
- SID length distribution；
- prefix fanout；
- token entropy；
- head/mid/tail collision；
- train-only interaction weighted collision；
- CSV parse/coverage/history length consistency。

这一步的目的不是提升指标，而是定位 SID 结构问题。

### 3.3 Train-only CF Embedding

只使用 train CSV 构建协同表示：

```text
history_item_id + target_item_id
  -> item-item co-occurrence
  -> PPMI
  -> TruncatedSVD
  -> CF embedding
```

不使用 valid/test，避免泄漏。

### 3.4 CS Embedding

将文本 embedding 和 CF embedding 融合：

```text
cs_emb = normalize(alpha * text_emb + (1 - alpha) * cf_emb)
```

其中：

- alpha 越大，越偏文本语义；
- alpha 越小，越偏协同行为。

本项目测试了：

```text
alpha = 0.2 / 0.5 / 0.7
```

### 3.5 RQ-KMeans 生成 CS-SID

将 text/CS embedding 输入 SID 生成脚本：

```text
run_rqkmeans_with_emb.py
```

生成：

```text
index.json
info.txt
item2sid.json
sid2items.json
valid_sid_set.json
item_mapping.json
```

### 3.6 Append Dedup Suffix

为解决 SID collision，引入 append dedup：

```text
singleton:
  <a_x><b_y><c_z>

collision bucket:
  <a_x><b_y><c_z><d_0>
  <a_x><b_y><c_z><d_1>
```

这样：

- 前 3 层仍保留 semantic / KMeans prefix；
- 第 4 层只做 collision disambiguation；
- post-dedup collision = 0。

### 3.7 Rewrite CSV

用新 `item2sid.json` 重写：

```text
history_item_sid
item_sid
```

生成 versioned train/valid/test CSV。这样 SFT 真正训练新 SID，而不是只生成离线 SID 文件。

### 3.8 Sid-only SFT

统一配置：

```text
base model = Qwen2.5-0.5B
sample = 30000
epoch = 1
beam = 20
task = sid-only
no RL
no auxiliary title2sid / seq-title2sid
```

这样可以隔离 SID 结构本身的影响。

### 3.9 Candidate Expansion + Lightweight Rerank

从预测 SID beams 展开 item candidates：

```text
exact SID bucket
optional prefix@3 bucket
optional prefix@2 bucket
```

然后使用轻量非泄漏 rerank：

- SID rank；
- exact/prefix source；
- train-only popularity；
- bucket size penalty；
- history embedding cosine；
- recent history embedding cosine。

target item 只用于最终 metrics，不参与 rerank 特征。

## 4. 工程实现

核心脚本：

| Script | Role |
|---|---|
| `utils_sid.py` | SID parse/normalize/prefix/json 工具 |
| `build_sid_mapping.py` | 构建 Text-SID baseline mapping |
| `check_sid_stage0.py` | SID 契约检查 |
| `analyze_sid.py` | 静态 SID 质量诊断 |
| `calc_plus.py` | 增强预测评估 |
| `build_cs_embeddings.py` | 构建 CF/CS embedding |
| `run_rqkmeans_with_emb.py` | 从 embedding 生成 SID |
| `rewrite_sid_csv.py` | 重写 CSV SID 字段 |
| `update_sid_manifest.py` | 更新 manifest |
| `evaluate_candidates.py` | 生成 item candidates |
| `rerank.py` | 轻量 history-aware rerank |
| `scripts/run_sid_30k_validation.sh` | 30k sid-only 训练/评估 |
| `scripts/run_stage6_candidate_rerank.sh` | Stage 6 candidate/rerank 一键脚本 |

完整链路：

```text
item_id
  -> row_index
  -> text / CF / CS embedding
  -> SID generation
  -> dedup
  -> rewrite CSV
  -> SFT
  -> evaluate
  -> calc_plus
  -> evaluate_candidates
  -> rerank
  -> final HR/NDCG
```

避免 row index 错位：

```text
item_order[i] == item_id
row_index[item_id] == i
len(item_order) == embedding.shape[0]
```

避免数据泄漏：

- CF 只用 train_csv；
- popularity 只用 train_csv；
- rerank 只用 history_item_id；
- target item 只用于 metrics；
- 不使用 valid/test 构建 embedding 或 popularity。

## 5. 实验设置

```text
Categories:
  Industrial_and_Scientific
  Office_Products

Base model:
  Qwen2.5-0.5B

Training:
  sample = 30000
  epoch = 1
  task = sid-only
  no early stopping
  no RL

Decoding:
  constrained decoding
  beam = 20

Rerank:
  exact-only
  exact + prefix@3
  source_weight = 4.0
```

## 6. 最终结果

### 6.1 Main Table

| Category | Baseline | Text HR@20 | Text NDCG@20 | Best CS Setting | CS HR@20 | CS NDCG@20 |
|---|---|---:|---:|---|---:|---:|
| Industrial_and_Scientific | text_mbk_k512 exact | 0.1286 | 0.0724 | cs_alpha0.2_k512 + prefix@3 rerank | 0.1366 | 0.0784 |
| Office_Products | text_mbk_k512 exact | 0.1445 | 0.0853 | cs_alpha0.7_k512 + prefix@3 rerank | 0.1492 | 0.0884 |

### 6.2 Required Numbers

Industrial:

```text
Text exact HR@20/NDCG@20 = 0.1286 / 0.0724
CS alpha0.2 + prefix@3 rerank HR@20/NDCG@20 = 0.1366 / 0.0784
```

Office:

```text
Text exact HR@20/NDCG@20 = 0.1445 / 0.0853
CS alpha0.7 + prefix@3 rerank HR@20/NDCG@20 = 0.1492 / 0.0884
```

### 6.3 Industrial Detail

| Setting | HR@20 | NDCG@20 | Candidate Recall |
|---|---:|---:|---:|
| Text exact | 0.1286 | 0.0724 | 0.1286 |
| CS alpha0.2 exact | 0.1348 | 0.0779 | 0.1348 |
| CS alpha0.2 prefix@3 + rerank | 0.1366 | 0.0784 | 0.1516 |

### 6.4 Office Detail

| Setting | HR@20 | NDCG@20 | Candidate Recall |
|---|---:|---:|---:|
| Text exact | 0.1445 | 0.0853 | 0.1445 |
| CS alpha0.7 exact | 0.1478 | 0.0882 | 0.1478 |
| CS alpha0.7 prefix@3 + rerank | 0.1492 | 0.0884 | 0.1617 |

## 7. 机制解释

### 7.1 Industrial 为什么 alpha=0.2 最好？

Industrial 中用户行为协同信号更强。较低 alpha 意味着 CF 权重更高，能让行为上相关的 item 更容易被聚到相近 SID prefix 中。

结果：

```text
Industrial best = cs_alpha0.2_k512_dedup
```

### 7.2 Office 为什么 alpha=0.7 最好？

Office 商品的文本语义更稳定。过强 CF 会破坏语义 exact 结构，导致 alpha=0.2 HR 不如 Text。alpha=0.7 保留更多 text semantic，同时加入少量 CF，因此最终 HR/NDCG 同时超过 Text。

结果：

```text
Office best = cs_alpha0.7_k512_dedup
```

### 7.3 CS exact-only 已经超过 Text exact-only

这说明 CS 的收益不是单纯来自 candidate pool 更大。

Industrial:

```text
Text exact HR@20 = 0.1286
CS exact HR@20   = 0.1348
```

Office:

```text
Text exact HR@20 = 0.1445
CS exact HR@20   = 0.1478
```

### 7.4 prefix@3 对 CS 更有效

prefix@3 与完整 SID 只差 dedup suffix，更接近 item neighborhood。

CS 的 prefix@3 candidate recall 提升明显，并能通过 rerank 转化为 HR/NDCG 增益。

### 7.5 prefix@2 太宽

prefix@2 会引入大量候选，candidate recall 虽然更高，但 target 的平均原始位置太靠后，当前轻量 reranker 排不回 Top20。

因此最终方案选择：

```text
exact + prefix@3
```

而不是 prefix@2。

## 8. 局限性

1. 30k sid-only 不是 full SFT。
2. Reranker 是轻量手工特征，不是学习型排序模型。
3. alpha sweep 还可以更细。
4. RL 已复现和探索，但不是当前主贡献。
5. Tail item 并非所有 category 都提升。
6. 当前实验使用单 test split，不能把 rerank 权重过度调到 test。

## 9. 面试表述

### 9.1 两分钟版本

我在 MiniOneRec 复现基础上发现，生成式推荐的关键瓶颈不只是 SFT/RL，而是 SID 本身的结构质量。原项目的 Text-SID 主要依赖文本 embedding，可能存在 collision、prefix 组织不匹配用户行为、SID-level 指标无法反映 item-level 推荐质量等问题。

我先建立了 SID 契约和 manifest，保证 index/info/csv/evaluate/calc 使用同一个 SID version；然后做 SID 诊断，统计 collision、prefix、head/mid/tail、3/4 层 SID 表现。接着我只用 train 数据构建 item co-occurrence，通过 PPMI+SVD 得到 CF embedding，再和 text embedding 融合成 CS embedding。之后用 RQ-KMeans 生成 CS-SID，并通过 append dedup suffix 消除 collision，再重写 train/valid/test CSV 训练 sid-only SFT。

最后我实现 candidate expansion 和 lightweight rerank，把 SID-level 指标推进到 item-level HR/NDCG。结果在 Industrial 和 Office 两个 category 上，CS-SID 都超过 Text-SID baseline；Industrial 最优 alpha=0.2，Office 最优 alpha=0.7，说明不同类目需要不同的协同/语义融合比例。

### 9.2 五分钟版本

这个项目的背景是 MiniOneRec 这类生成式推荐系统输出的不是 item score，而是 Semantic ID。SID 的结构会影响模型学习、constrained decoding、beam candidate、prefix expansion 和最终 item-level 推荐质量。

我做的第一件事是工程基础设施：把原来散落在 index/info/csv/evaluate/calc 里的 SID 统一成 item2sid、sid2items、valid_sid_set、item_mapping 和 manifest。这样每个 SID version 都能完整复现实验，不会出现训练一个 SID、评估另一个 SID 的问题。

第二步是诊断。我实现了 analyze_sid 和 calc_plus，不只看 HR/NDCG，还看 collision、prefix entropy、3/4 层 SID、head/mid/tail、history length 分桶和 per-sample transition。这样可以解释为什么某个 SID version 变好或变差。

第三步是方法设计。我只用 train_csv 构建 item-item co-occurrence，做 PPMI 和 TruncatedSVD 得到 CF embedding，再和 text embedding 融合成 CS embedding。公式是 `normalize(alpha * text + (1-alpha) * CF)`。alpha 控制文本语义和协同行为的权重。

第四步是重新生成 SID。我用 RQ-KMeans / MiniBatchKMeans 从 CS embedding 生成 SID，并设计 append dedup suffix。如果 3 层 SID 有 collision，就追加 `<d_i>` 作为第 4 层；如果没有 collision，就保持 3 层。这样 post-dedup collision 为 0，同时前三层 prefix 仍有语义。

第五步是训练和评估。我重写 CSV 中的 `history_item_sid` 和 `item_sid`，做 sid-only SFT，避免 title2sid 辅助任务干扰。最后用 evaluate_candidates 和 rerank 从 SID beams 展开 item candidates，用 train-only popularity 和 history embedding cosine 做轻量 rerank。

最终结果是：Industrial 上 Text exact HR@20/NDCG@20 是 0.1286/0.0724，CS alpha0.2 + prefix@3 rerank 达到 0.1366/0.0784；Office 上 Text exact 是 0.1445/0.0853，CS alpha0.7 + prefix@3 rerank 达到 0.1492/0.0884。两个 category 都超过 baseline，但最佳 alpha 不同，说明 CS-SID 的协同/语义权重需要按类目调整。

### 9.3 高压面试问答

#### Q1：为什么不直接做 RL？

RL 变量太多，包括 reward、KL、beam rollout、group size、SFT checkpoint 质量。当前更核心的问题是 SID 结构本身是否好。先把 SID 契约、诊断、CS-SID、item-level candidate/rerank 做清楚，能更稳定地定位收益来源。RL 可以作为后续补充，不适合作为主贡献。

#### Q2：CS-SID 是否只是 candidate pool 更大？

不是。exact-only 下候选数量都是 20，CS 仍然超过 Text。

Industrial：

```text
Text exact HR@20 = 0.1286
CS exact HR@20 = 0.1348
```

Office：

```text
Text exact HR@20 = 0.1445
CS exact HR@20 = 0.1478
```

说明 CS-SID 本身的 item-to-SID 组织更有效，不只是扩大候选池。

#### Q3：为什么 Industrial alpha=0.2，Office alpha=0.7？

alpha 控制 text 与 CF 的权重。Industrial 行为协同更强，所以更低 alpha、更高 CF 权重效果更好。Office 的文本语义更稳定，过强 CF 会破坏 exact semantic structure，因此 alpha=0.7 更好。

#### Q4：为什么不用 SASRec embedding 构建 CF？

第一版选择 PPMI+SVD 是为了轻量、可控、避免引入新的模型训练变量。SASRec embedding 需要额外训练和调参，难以区分收益来自序列模型还是 SID 结构。后续可以扩展，但第一版用 PPMI+SVD 更适合做可解释消融。

#### Q5：怎么避免数据泄漏？

CF embedding 只用 train_csv；popularity 只用 train_csv；rerank 只用 history_item_id 和 candidate embedding；target item 只用于 metrics。valid/test 不参与 embedding 构建、popularity 统计或 rerank 特征。

#### Q6：为什么要 dedup suffix？

SID collision 会导致 SID-level hit 不等于 item-level hit。append `<d_i>` 后，post-dedup collision 为 0，同时保留前三层 semantic prefix。

#### Q7：为什么 prefix@3 有效、prefix@2 无效？

prefix@3 与完整 SID 只差 dedup suffix，候选邻域较精确。prefix@2 太宽，会召回大量噪声候选。实验中 prefix@2 candidate recall 更高，但 target 平均排名太靠后，轻量 reranker 排不回 Top20。

#### Q8：SID-level 和 item-level 有什么区别？

SID-level 只看生成的 SID 是否等于 target SID；item-level 要看最终推荐 item 是否命中。存在 collision、dedup、bucket expansion 时，两者不等价。因此本项目加入 candidate expansion 和 item-level rerank。

#### Q9：rerank 是否使用 target？

不使用。target 只在最后计算 HR/NDCG。rerank 特征包括 SID rank、source type、train-only popularity、bucket penalty、history cosine、recent history cosine。

#### Q10：这个项目真正创新点是什么？

不是简单调参，而是把生成式推荐中的 SID 当作核心可优化对象：

1. SID 契约与版本管理；
2. SID 静态诊断与分层评测；
3. train-only Collaborative-Semantic embedding；
4. append dedup SID；
5. versioned CSV 重写；
6. SID-level 到 item-level candidate/rerank 的闭环；
7. 在 Industrial 和 Office 上验证不同 alpha 的 category-specific 最优解。

## 10. 最小复现命令

在 AutoDL 结果目录完整时，生成最终汇总：

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python summarize_final_results.py --output-dir results/final_summary
cat results/final_summary/final_results_summary.md
cat results/final_summary/final_consistency_report.md
```

如果要重跑 Stage 6：

```bash
CATEGORY=Industrial_and_Scientific \
TEXT_VERSION=text_mbk_k512_dedup \
CS_VERSION=cs_alpha0.2_k512_dedup \
bash scripts/run_stage6_candidate_rerank.sh

CATEGORY=Office_Products \
TEXT_VERSION=text_mbk_k512_dedup \
CS_VERSION=cs_alpha0.7_k512_dedup \
bash scripts/run_stage6_candidate_rerank.sh
```
