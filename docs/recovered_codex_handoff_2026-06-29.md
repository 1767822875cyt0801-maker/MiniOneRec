# MiniOneRec 交接总结

基于当前本地只读检查整理。

## 1. 当前仓库改动文件

### 已跟踪文件修改

| 文件 | 作用 |
|---|---|
| `calc_plus.py` | 增强评估脚本。新增动态 `prefix-levels`、SID 长度分布、dedup SID 说明，并让 per-sample CSV 支持 3/4 层 SID。 |
| `check_sid_stage0.py` | SID 契约检查。新增 manifest 路径检查，支持 versioned `train/valid/test` CSV、`--category`、`--manifest`。 |
| `data.py` | 数据集读取兼容 4 层及以上 SID。新增 `combine_sid_tokens()`，避免只拼前三层；sample 超过数据量时自动取 `min(sample, len(data))`。 |
| `evaluate.py` | 推理评估兼容可变 SID 长度。默认自动推断 `max_new_tokens`，输出 SID 长度分布和 prefix ambiguity。 |
| `run_rqkmeans_with_emb.py` | 从 embedding 生成 SID。改为 canonical `data/Amazon/sid_versions/<version>/<category>/` 布局，新增 append dedup、pre/post collision 报告、overwrite guard。 |
| `sft.py` | SFT 训练增强。支持显式 `sid_index_path`、DDP 参数、`sid_only` task mode、关闭 wandb、控制保存策略、主进程保存 `final_checkpoint`。 |

### 新增未跟踪文件

| 文件 | 作用 |
|---|---|
| `evaluate_candidates.py` | 将 SID beam 展开成 item candidates，支持 exact 和 prefix expansion，并计算 item-level candidate recall。 |
| `rerank.py` | 非泄漏轻量 rerank，使用 SID rank、source、train-only popularity、history/recent cosine、bucket penalty。 |
| `rewrite_sid_csv.py` | 用新 `item2sid.json` 重写 CSV 的 `history_item_sid` 和 `item_sid`。 |
| `update_sid_manifest.py` | 安全更新 `experiment_manifest.json` 中的 SID version 条目。 |
| `summarize_final_results.py` | 汇总 AutoDL 上最终 CS-SID 结果，生成 summary、artifact manifest 和 consistency report。 |
| `fuse_dual_sid_candidates.py` | Text SID 与 behavior/CF SID 两路 candidate 的 item-level fusion 工具。 |
| `scripts/generate_sid_versions_and_rewrite_csv.sh` | 一键生成 SID version、重写 CSV、更新 manifest。 |
| `scripts/run_sft_smoke_sid_version.sh` | 按 manifest 跑某个 SID version 的 sid-only SFT smoke。 |
| `scripts/eval_smoke_sid_version.sh` | 按 manifest 对 smoke checkpoint 做 evaluate + calc_plus。 |
| `scripts/run_sid_30k_validation.sh` | 30k sid-only validation 主脚本。 |
| `scripts/run_stage6_candidate_rerank.sh` | Stage 6 candidate expansion + rerank 主脚本。 |
| `scripts/run_sid_k_ablation_10k.sh` | Industrial 10k SID k/alpha 消融脚本。 |
| `scripts/tmp_run_sidonly_10k_compare.sh` | 临时 10k sid-only 对比脚本。 |
| `scripts/tmp_run_sidonly_10k_compare_autostop.sh` | 10k 对比后自动关机 wrapper。 |
| `scripts/diagnose_sid_smoke_by_groups.py` | 从 `per_sample_eval.csv` 做分组诊断。 |
| `scripts/summarize_sidonly_smoke_grid.py` | 汇总多 SID version smoke/validation grid。 |
| `scripts/run_dual_sid_cf_fusion.sh` | Text + CF dual SID candidate fusion 流程脚本。 |
| `scripts/ddp_sanity.py` | DDP all-reduce sanity check。 |
| `scripts/run_sft_ddp.sh` | 通用单机多卡 SFT 启动脚本。 |
| `scripts/run_sft_ddp_qwen25_15b_4gpu.sh` | Qwen2.5-1.5B 4GPU SFT preset。 |
| `scripts/run_sft_ddp_qwen25_3b_4gpu.sh` | Qwen2.5-3B 4GPU SFT preset。 |
| `scripts/run_sft_ddp_smoke_4gpu.sh` | 4GPU SFT smoke preset。 |
| `docs/MULTIGPU_SFT.md` | 多卡 SFT 使用说明。 |
| `completion_audit.md` | Stage 8 完成度审计，记录本地与 AutoDL 产物差异。 |
| `CS_SID_阶段性技术总结.md` | CS-SID 阶段性技术总结。 |
| `MiniOneRec_CS_SID_final_report.md` | CS-SID 最终报告草稿，含最终表述和关键数字。 |
| `MiniOneRec_复现实验与面试材料.md` | 早期 SFT/RL 复现实验和面试材料。 |

## 2. 之前做到哪个阶段

已经做到 `Stage 8 Completion Audit`：代码链路和总结材料基本收口。

主线阶段：

1. Stage 0：SID 契约与 manifest。
2. Stage 1：SID 静态诊断与 `calc_plus` 增强评估。
3. Stage 2：train-only CF embedding 与 CS embedding。
4. Stage 3/4：RQ-KMeans / MiniBatchKMeans 生成新 SID，append dedup，重写 CSV。
5. Stage 5：sid-only SFT 消融。
6. Stage 6：candidate expansion + lightweight rerank。
7. Stage 8：最终结果汇总和 consistency audit 工具。

当前本地 `results/final_summary/final_consistency_report.md` 的 `Overall OK` 是 `False`，原因是本地缺少多数最终 AutoDL 结果文件。

## 3. 已有实验结论和关键数字

### Text-SID baseline 静态检查

| Category | num_items | num_unique_sid | collision | parse/coverage |
|---|---:|---:|---:|---|
| Office_Products | 3459 | 3444 | 约 0.43% | 1.0 |
| Industrial_and_Scientific | 3686 | 3670 | 约 0.43% | 1.0 |

结论：官方 Text-SID 可作为干净 baseline，index/info/csv 冲突为 0。

### Stage 5 sid-only 30k

| Category | Text HR@20 | Text NDCG@20 | Best CS | CS HR@20 | CS NDCG@20 |
|---|---:|---:|---|---:|---:|
| Industrial_and_Scientific | 0.1286 | 0.0709 | `cs_alpha0.2_k512_dedup` | 0.1348 | 0.0764 |
| Office_Products | 0.1445 | 0.0844 | `cs_alpha0.7_k512_dedup` | 0.1478 | 0.0870 |

### Stage 6 item-level final

| Category | Text exact HR/NDCG@20 | Best CS setting | CS HR/NDCG@20 | Candidate recall |
|---|---|---|---|---:|
| Industrial_and_Scientific | 0.1286 / 0.0724 | `cs_alpha0.2_k512_dedup + prefix@3 rerank` | 0.1366 / 0.0784 | 0.1516 |
| Office_Products | 0.1445 / 0.0853 | `cs_alpha0.7_k512_dedup + prefix@3 rerank` | 0.1492 / 0.0884 | 0.1617 |

核心结论：CS-SID 在两个 category 上均超过 Text-SID baseline。Industrial 最优 alpha 更低，说明协同信号更重要；Office 最优 alpha 更高，说明文本语义更重要。CS exact-only 也超过 Text exact-only，所以收益不只是扩大 candidate pool。

## 4. 当前本地缺少哪些 AutoDL 产物

本地已存在但被 `.gitignore` 忽略：

- `data/Amazon/cs_embeddings/<category>/...` 两个 category 的 CF/CS embedding。
- `data/Amazon/sid_maps/...` Text baseline mapping。
- `data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/...` 一个 CF SID 版本。
- 少量 `results/candidates_Industrial_and_Scientific_text_k512_30k_*` 和空的 `results/final_summary`。

仍缺少或不完整的最终主结果产物：

- `data/Amazon/sid_versions/text_mbk_k512_dedup/<category>/...`
- `data/Amazon/sid_versions/cs_alpha0.2_k512_dedup/Industrial_and_Scientific/...`
- `data/Amazon/sid_versions/cs_alpha0.7_k512_dedup/Office_Products/...`
- 对应的 `train.csv`、`valid.csv`、`test.csv`、`reports/generation_report.json`、`reports/rewrite_*_report.json`、`reports/stage0_check_report.json`
- `results/calc_plus_sidonly_*_sample30000_ep1_noearly_beam20/eval_report_*.json`
- `results/eval_sidonly_*_sample30000_ep1_noearly_beam20/predictions.json`
- `results/candidates_*_30k_exact*/report.json`
- `results/candidates_*_30k_p3*/report.json`
- `results/rerank_*_30k_exact*/rerank_report.json`
- `results/rerank_*_30k_p3*/rerank_report.json`

注意：本地有 `results/candidates_Industrial_and_Scientific_text_k512_30k_exact_c500_v2/...`，但 `summarize_final_results.py` 当前期望的是 `results/candidates_text_k512_30k_exact_v2/...`。AutoDL 上需要确认命名是否一致，否则 summary 会继续报 missing。

## 5. 下一步 AutoDL 核验命令

先生成最终汇总：

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python summarize_final_results.py --output-dir results/final_summary
cat results/final_summary/final_results_summary.md
cat results/final_summary/final_consistency_report.md
```

严格检查：

```bash
python summarize_final_results.py --output-dir results/final_summary --strict
```

核验 SID version 文件：

```bash
python check_sid_stage0.py --manifest data/Amazon/sid_maps/experiment_manifest.json --sid-version text_mbk_k512_dedup --category Industrial_and_Scientific --report-path /tmp/check_ind_text.json
python check_sid_stage0.py --manifest data/Amazon/sid_maps/experiment_manifest.json --sid-version cs_alpha0.2_k512_dedup --category Industrial_and_Scientific --report-path /tmp/check_ind_cs.json
python check_sid_stage0.py --manifest data/Amazon/sid_maps/experiment_manifest.json --sid-version text_mbk_k512_dedup --category Office_Products --report-path /tmp/check_office_text.json
python check_sid_stage0.py --manifest data/Amazon/sid_maps/experiment_manifest.json --sid-version cs_alpha0.7_k512_dedup --category Office_Products --report-path /tmp/check_office_cs.json
```

检查 Office `cs_alpha0.7` 是否误写到 Industrial 路径：

```bash
find results -path "*Industrial*cs_alpha0.7*k512*30000*" -print
find results -path "*Office_Products*cs_alpha0.7*k512*30000*" -print
```

如 Stage 6 结果缺失，重跑：

```bash
CATEGORY=Industrial_and_Scientific TEXT_VERSION=text_mbk_k512_dedup CS_VERSION=cs_alpha0.2_k512_dedup bash scripts/run_stage6_candidate_rerank.sh
CATEGORY=Office_Products TEXT_VERSION=text_mbk_k512_dedup CS_VERSION=cs_alpha0.7_k512_dedup bash scripts/run_stage6_candidate_rerank.sh
```

如 SID version/CSV 缺失，先生成：

```bash
CATEGORY=Industrial_and_Scientific SID_VERSION_LIST="text_mbk_k512_dedup cs_alpha0.2_k512_dedup" CODEBOOK_SIZE=512 bash scripts/generate_sid_versions_and_rewrite_csv.sh
CATEGORY=Office_Products SID_VERSION_LIST="text_mbk_k512_dedup cs_alpha0.7_k512_dedup" CODEBOOK_SIZE=512 bash scripts/generate_sid_versions_and_rewrite_csv.sh
```

## 6. Git 提交建议

建议提交的代码与文档：

- 6 个 modified tracked 文件：`calc_plus.py`、`check_sid_stage0.py`、`data.py`、`evaluate.py`、`run_rqkmeans_with_emb.py`、`sft.py`
- 新增核心脚本：`evaluate_candidates.py`、`rerank.py`、`rewrite_sid_csv.py`、`update_sid_manifest.py`、`summarize_final_results.py`
- 可选提交：`fuse_dual_sid_candidates.py`、`scripts/run_dual_sid_cf_fusion.sh`
- 运行脚本：`scripts/*.sh`、`scripts/*.py`
- 多卡说明：`docs/MULTIGPU_SFT.md`

建议等 AutoDL 确认后再提交或更新的文件：

- `MiniOneRec_CS_SID_final_report.md`
- `CS_SID_阶段性技术总结.md`
- `completion_audit.md`
- `MiniOneRec_复现实验与面试材料.md`

不要提交的结果产物：

- `data/Amazon/cs_embeddings/`
- `data/Amazon/sid_maps/`
- `data/Amazon/sid_versions/`
- `results/`
- `outputs/`
- 所有 `*:Zone.Identifier`

这些路径当前被 `.gitignore` 的 `data/` 和 `results/` 规则忽略。最终应先让 AutoDL 的 `final_consistency_report.md` 变成 `Overall OK: True`，再固定最终报告里的数字并提交代码与文档。
