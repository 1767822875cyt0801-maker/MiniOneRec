# Phase 3 Repository Cleanup Audit

审计日期：2026-07-22  
仓库：`/home/dell/projects/MiniOneRec`  
审计基线：`be65b86`（`yuting-reproduce`）

## 1. 审计范围与约束

本报告记录 Phase 3 清理开始前的只读盘点结果。审计期间未运行训练、推理或 GPU 实验，未改动算法逻辑、配置或实验结果，也未删除文件。

执行的基线命令包括：

```bash
git status
git branch -vv
git log --oneline -10
git ls-files --modified
git ls-files --others --exclude-standard
find . -path './.git' -prune -o -type f -size +100M -print
```

## 2. Git 基线状态

- 当前分支：`yuting-reproduce`
- 上游分支：`origin/yuting-reproduce`
- 审计时分支关系：up to date
- HEAD：`be65b86 Close S5 frozen auxiliary fusion final test`
- tracked 文件总数：140
- modified tracked 文件：3
- untracked 文件：96（按 `git ls-files --others --exclude-standard` 统计）
- 已被现有规则忽略的文件：631
- 暂存区：空

最近 10 个提交：

```text
be65b86 Close S5 frozen auxiliary fusion final test
2dd3bbf fix sasrec padding masks and finite ranking evaluation
d6cf860 feat: add validation fusion sweep and train-only SASRec exporter
1b52b50 feat1: add CS-SID and dual-SID fusion evaluation pipeline
4598e5e feat: add single-GPU reproduction diagnostics and evaluation tools
49d0344 sync initial MiniOneRec test code and local docs
0c64b95 Merge pull request #2 from yuting-02/yuting-reproduce
f5c61ac add QR-SID code and fixed trainer state
e379980 Add serial script for sequential SFT experiments
308ec65 Add serial SFT experiments for Industrial and Office
```

## 3. Modified tracked 文件

| 文件 | 现有变更性质 | 分类 | 提交 | 删除 | 忽略 |
|---|---|---|---:|---:|---:|
| `LogitProcessor.py` | 约束生成遇到 EOS 后允许安全结束，另补末尾换行 | A / Phase 3 source | 是 | 否 | 否 |
| `scripts/run_stage6_candidate_rerank.sh` | 增加 valid/test split、manifest 路径解析、结果目录和候选模式契约 | A / Phase 3 reproducibility script | 是 | 否 | 否 |
| `tests/test_minimax_provider.py` | 为无重依赖测试环境补充 `openai`、`aiohttp` mock | A / test portability | 是 | 否 | 否 |

现有 diff 共 82 行新增、31 行删除；`git diff --check` 无空白错误。本任务不再修改这些逻辑，只将既有 Phase 3 状态纳入发布。

## 4. Untracked 文件分类

### A. Must Commit

审计基线中共有 84 个 untracked 文件属于必须提交类别；加上 3 个 modified tracked 文件，共 87 个既有发布内容文件。

#### 根目录源码和设计文档（3）

- `7.3MiniOneRec_Strong_Behavior_SID_最终实现方案.md`
- `export_sasrec_direct_candidates.py`
- `merge_direct_sasrec_candidates.py`

#### 配置（3）

- `configs/s6_cost_aware_aux/dev_config.json`
- `configs/s6_cost_aware_aux/direct_promotion_protocol.json`
- `configs/s6_cost_aware_aux/lightweight_ranker_protocol.json`

#### Phase 3/Phase 4 交接文档（26）

- `docs/s4_0_cf_valid_restore_and_parity_closeout.md`
- `docs/s4_1_smoke_runner_env_contract_hardening.md`
- `docs/s4_2_dual_stream_candidate_routing_closeout.md`
- `docs/s4_3_smoke_full_valid_closeout.md`
- `docs/s4_4_formal_transfer_diagnosis.md`
- `docs/s4_sasrec_sid_valid_experiment.md`
- `docs/s5_0_frozen_auxiliary_fusion_conversion.md`
- `docs/s6_2_smoke_audit_and_formal_plan.md`
- `docs/s6_3_formal_direct_sasrec_validation_audit.md`
- `docs/s6_4_cf_direct_sasrec_union_validation.md`
- `docs/s6_5_frozen_ranker_validation.md`
- `docs/s6_6_lightweight_cf_preserving_ranker.md`
- `docs/s6_6r_cf_anchored_direct_promotion_gate.md`
- `docs/s6_cost_evidence_closeout.md`
- `docs/s6_direct_sasrec_audit.md`
- `docs/s6_direct_sasrec_protocol.md`
- `docs/s6_stage_closeout.md`
- `docs/s6_final/S6_Direct_SASRec_阶段技术总结.md`
- `docs/s6_final/S6_实验结果与指标汇总.md`
- `docs/s6_final/S6_工程实现与代码主线.md`
- `docs/s6_final/S6_证据归档与复现说明.md`
- `docs/s6_final/S6_面试与论文表述材料.md`
- `docs/optimization_phase4_audit_report.md`
- `docs/优化四：MiniOneRec + ChronoSID + UniSGR/7.22 MiniOneRec_Chrono_UniRank_project_plan.md`
- `docs/优化四：MiniOneRec + ChronoSID + UniSGR/BeyondItemOrderTemporalGapTokenizationforGenerative.pdf`
- `docs/优化四：MiniOneRec + ChronoSID + UniSGR/UniSGR Unified Framework for Semantic ID Generation and.pdf`

两份论文 PDF 分别约 1.08 MiB 和 1.56 MiB，低于 100 MiB 门槛；它们是 Phase 4 架构审计的输入文档，不是模型或生成结果。将其作为 Phase 3 到 Phase 4 的可追溯交接资料提交。

#### 归档/回传校验清单（8）

- `archive/autodl_legacy_outputs/ARCHIVED_PATHS.txt`
- `archive/autodl_legacy_outputs/REMOTE_SHA256SUMS.txt`
- `incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036.tar.gz.sha256`
- `incoming/s4_smoke_full_valid/s4_smoke_full_valid_closeout_20260714_144743.tar.gz.sha256`
- `incoming/s5_final_test_closeout/s5_final_test_closeout_bundle.tar.gz.sha256`
- `incoming/s6_direct_sasrec_smoke/s6_direct_sasrec_valid_fit_k20_smoke_bundle.tar.gz.sha256`
- `incoming/s6_formal_direct_sasrec/s6_formal_direct_sasrec_formal_v1_bundle.tar.gz.sha256`
- `incoming/s6_qwen_cost_profile/cost_profile_qwen_sasrec_sid_v2_bundle.tar.gz.sha256`

这些文件体积小，记录归档对象或 SHA-256，可用于证据完整性校验；提交清单本身，不提交其指向的大型输出与压缩包。

#### Phase 3 工程/复现脚本（30）

- `scripts/audit_s6_direct_sasrec_smoke.py`
- `scripts/audit_s6_formal_direct_sasrec.py`
- `scripts/audit_stage7_p3_behavior_representation.py`
- `scripts/build_s6_validation_split.py`
- `scripts/build_sasrec_v3_behavior_sid.py`
- `scripts/build_stage7_p2_valid_split.py`
- `scripts/check_split_alignment.py`
- `scripts/diagnose_stage7_rerank_headroom.py`
- `scripts/freeze_stage7_p2_history_ranker.py`
- `scripts/run_s4_sasrec_sid_valid.sh`
- `scripts/run_sasrec_v3_behavior_sid.sh`
- `scripts/run_stage7_industrial_valid_exact_smoke.sh`
- `scripts/run_stage7_p2_frozen_ranker_closeout.sh`
- `scripts/run_stage7_p2_headroom_diagnosis.sh`
- `scripts/run_stage7_p2_history_ranker.sh`
- `scripts/run_stage7_p2_minimal_ranker.sh`
- `scripts/run_stage7_p2_valid_split.sh`
- `scripts/run_stage7_p3_behavior_audit.sh`
- `scripts/s4_sasrec_sid_valid_pipeline.py`
- `scripts/s5_auxiliary_fusion_conversion.py`
- `scripts/s6_cf_direct_union_validation.py`
- `scripts/s6_cost_evidence_closeout.py`
- `scripts/s6_direct_promotion_gate_validation.py`
- `scripts/s6_final_artifact_inventory.py`
- `scripts/s6_frozen_ranker_validation.py`
- `scripts/s6_lightweight_ranker_validation.py`
- `scripts/s6_sasrec_checkpoint_audit.py`
- `scripts/s6_stage_closeout_audit.py`
- `scripts/train_stage7_p2_history_ranker.py`
- `scripts/train_stage7_p2_minimal_ranker.py`

脚本名中包含 `train` 的文件仅作为可复现工具提交；本次清理不执行它们。

#### 测试（14）

- `tests/test_s4_sasrec_sid_valid_pipeline.py`
- `tests/test_s5_auxiliary_fusion_conversion.py`
- `tests/test_s6_cf_direct_union_validation.py`
- `tests/test_s6_checkpoint_audit.py`
- `tests/test_s6_cost_evidence_closeout.py`
- `tests/test_s6_direct_promotion_gate_validation.py`
- `tests/test_s6_direct_sasrec_candidates.py`
- `tests/test_s6_final_artifact_inventory.py`
- `tests/test_s6_frozen_ranker_validation.py`
- `tests/test_s6_lightweight_ranker_validation.py`
- `tests/test_s6_smoke_audit.py`
- `tests/test_s6_stage_closeout_audit.py`
- `tests/test_s6_validation_split.py`
- `tests/test_sasrec_v3_behavior_sid.py`

### B. Must Ignore（保留本地，不提交、不删除）

审计基线中有 12 个当前可见的 untracked 文件应新增忽略规则：

| 文件/模式 | 原因 | 提交 | 删除 | 忽略 |
|---|---|---:|---:|---:|
| `incoming/**/*.tar.gz`（6 个） | AutoDL 回传实验包，合计约 120 MiB；内容属于输出/结果 | 否 | 否 | 是 |
| `incoming/s4_formal_full_valid/LATEST_EXTRACTED_S4_FORMAL_DIR.txt` | 含机器本地绝对路径的临时指针 | 否 | 否 | 是 |
| `build_sasrec_embeddings_numbered.txt` | 已有源码的带行号临时快照，不是可执行主线 | 否 | 否 | 是 |
| `sasrec_key_lines.txt` | 从源码提取的临时关键行快照 | 否 | 否 | 是 |
| `*:Zone.Identifier`（当前可见 3 个，忽略目录内另有若干） | Windows 下载来源元数据，不属于项目内容 | 否 | 否 | 是 |

同时必须继续忽略并原地保留：

- `archive/` 中模型输出、final checkpoints、训练参数和校验日志；
- `results/` 中 predictions、candidate JSONL、rerank 输出和指标证据；
- `data/` 中本地数据、SID 产物、embedding、checkpoint 和 manifest；
- `__pycache__/`、`*.pyc`、日志及其他缓存；
- `outputs/`、`checkpoints/`、`saved_models/` 等模型输出目录。

`.gitignore` 应新增：archive/incoming 的精确规则（仅放行文本清单与 `.sha256`）、压缩包、candidate JSONL、下载元数据及两份根目录临时快照。已有 `data/`、`results/`、模型权重、日志和 Python cache 规则继续保留。

### C. Safe Delete

本次结论：**0 个文件**。

虽然 `Zone.Identifier` 和两个源码快照通常可视为临时文件，但无法同时证明“已被新脚本替代、无任何引用、不影响已有实验复现”三项删除条件。按保守策略只忽略、不删除。所有 Phase 3 results evidence、validation reports、manifest、实验总结及 AutoDL 回传包均保留。

## 5. 大文件与实验产物审计

### 5.1 超过 100 MiB

共发现 27 个文件超过 100 MiB：

- `archive/`：22 个模型 checkpoint，单文件约 0.92–1.84 GiB；最大文件为约 1.84 GiB 的 RL `model.safetensors`。
- `results/`：5 个 candidate/rerank JSONL，约 103–256 MiB。

这些文件均未被 Git 跟踪，也不会暂存。

`results/` 中 5 个超过 100 MiB 的文件为：

- `results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/valid_fit/k100/candidates.jsonl`
- `results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/valid_fit/k50/candidates.jsonl`
- `results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/valid_fit/k20/candidates.jsonl`
- `results/s5_auxiliary_fusion/Industrial_and_Scientific/s5_0_candidate_provenance.jsonl`
- `results/stage7_validation_protocol/valid/Industrial_and_Scientific/p2_history_ranker/history_reranked_candidates.jsonl`

### 5.2 目录规模

| 目录 | 本地规模 | 文件数 | 处理 |
|---|---:|---:|---|
| `archive/` | 约 24 GiB | 297 | 保留；仅提交路径/哈希清单，输出全部忽略 |
| `incoming/` | 约 492 MiB | 43 | 保留；仅提交 `.sha256`，bundle/extracted outputs 忽略 |
| `results/` | 约 1.4 GiB | 135 | 保留证据；全部忽略 |
| `data/` | 约 454 MiB | 未纳入 untracked 提交统计 | 保留本地数据、SID/embedding/checkpoint；全部忽略 |

全仓库还发现 37 个 checkpoint/weight-like 扩展名文件（含 `training_args.bin`）、24 个 JSONL、2 个日志和 6 个压缩包。Git 当前未跟踪任何 outputs/results/checkpoints/logs/archive/incoming 路径，也未跟踪 `.pt`、`.safetensors`、`.jsonl`、`.tar.gz` 或 `.log` 实验产物。

## 6. 清理决策摘要

| 分类 | 数量/范围 | 是否提交 | 是否删除 | 是否加入/保持忽略 |
|---|---:|---:|---:|---:|
| Modified source/script/test | 3 | 是 | 否 | 否 |
| Untracked code/config/docs/tests/manifests | 84 | 是 | 否 | 否 |
| 当前可见本地产物/临时文件 | 12 | 否 | 否 | 是 |
| 已忽略数据、结果、checkpoint、log、cache | 631 个文件 | 否 | 否 | 是 |
| Safe Delete | 0 | 否 | 否 | 不适用 |

提交前必须再次验证：

1. 暂存区没有 checkpoint、model weight、candidate/prediction、log、压缩包或 results/output 文件；
2. 暂存区没有超过 100 MiB 的文件；
3. `.gitignore` 对本地 evidence artifacts 生效，且 SHA-256/归档清单仍可提交；
4. 工作区除被忽略的本地证据外无可见未提交项。
