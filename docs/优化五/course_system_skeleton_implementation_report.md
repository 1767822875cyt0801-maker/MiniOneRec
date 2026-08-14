# COURSE-SYSTEM-SKELETON-V1 实施与正式冻结报告

## 1. Verdict

```text
VERDICT=COURSE_SYSTEM_IMPLEMENTED_AND_CARDINALITY_GRID_FROZEN
FORMAL_EXPERIMENTS_RUN=NO
WAITING_FOR_USER_MANUAL_RUN=YES
```

课程系统已从骨架阶段收口到可提交状态：正式预算顺序固定为 `20/50/75/90/all`，新增 frozen valid-select K-matrix 配置，并在真实执行入口加入输入身份、参数、样本、split、Git 与环境保护。本轮没有运行 cardinality v2 或正式 K matrix。

## 2. Repository closeout

| 字段 | 状态 |
|---|---|
| repository root | `/home/dell/projects/MiniOneRec` |
| starting branch | `phase4-chrono-unirank` |
| starting HEAD | `8c4aa08ed76dd7b93836502e474d7093073e42ff` |
| course branch | `course-system-bounded-rerank-v1` |
| frozen implementation commit | `ca98c381f39b1fc5a38e3093bc611eb64d4d1f92` |
| handoff-fix commit subject | `Fix course AutoDL preflight and staged handoff` |
| push | 未执行 |
| unrelated Phase 4 untracked files | 保留，不删除、不移动、不暂存 |

提交哈希不写入提交自身；以最终交付信息和该分支的 `git log -1` 为准。提交前只显式暂存课程系统授权路径，并检查 `git diff --cached --name-only`。

## 3. Frozen architecture

```text
CF frozen prediction ───────┐
                            ├─ exact candidate expansion
SASRec frozen prediction ───┘
             ↓
source-aware RRF (lambda=0.75, source_bonus=0.01)
             ↓
stable item dedup
             ↓
cardinality gate
             ↓
generic prefix budget: 20 / 50 / 75 / 90 / all
             ↓
P2 frozen history-aware ranker (source_independent_projection)
             ↓
Top-20 downstream output
```

预算位置保持 `post_fusion_pre_rerank`。K90 使用与其他有限预算相同的通用 `min(post_dedup_count, K)`/稳定前缀逻辑，没有 K90 专用算法分支，也没有 ground-truth 特殊保留。

## 4. Files and responsibilities

| 路径 | 用途 |
|---|---|
| `minionerec_system/config.py` | 冻结配置合同、repo-root 路径解析、不可变输入 SHA-256/大小、Python 3.11 与 Git preflight |
| `minionerec_system/pipeline.py` | 正式执行前 guard、1360 样本/valid-select 检查、每次运行重新执行 cardinality gate |
| `minionerec_system/artifacts.py` | 将 formal guard 证据写入 manifest |
| `minionerec_system/cardinality.py` | 通用预算 effective count、逐样本与汇总 gate |
| `minionerec_system/checker.py` | K90 列、五档覆盖、正式样本数/预算顺序/guard 检查 |
| `configs/course_system/smoke_cf_sasrec_exact.yaml` | 95 候选 synthetic K90 smoke |
| `configs/course_system/valid_cf_sasrec_exact_template.yaml` | clean cardinality v2 正式模板 |
| `configs/course_system/valid_select_cf_sasrec_exact_kmatrix_v1.yaml` | 冻结 valid-select K matrix 配置 |
| `scripts/run_course_system.py` | 统一执行入口 |
| `scripts/check_course_system_artifacts.py` | 只读 artifact checker 入口 |
| `tests/course_system/` | K90、formal guard、路径可移植性、v1 分布与 checker 定向测试 |
| `docs/优化五/course_system_cardinality_closeout_v1.md` | v1 只读复算、证据定性与 v2/K-matrix gate |
| `docs/优化五/course_system_autodl_manual_runbook_v1.md` | WSL 传输、AutoDL 获取/preflight/v2 与未来 K matrix 的分阶段手册 |

## 5. Frozen formal contract

正式配置固定：

- dataset=`Industrial_and_Scientific`，split=`valid`，quality split=`valid_select`；
- streams=`[cf_sid, sasrec_sid]`，candidate mode=`exact`；
- fusion=`source_aware_rrf`，`lambda_sasrec=0.75`，`source_bonus=0.01`；
- reranker=`p2_frozen_history_aware`，compatibility=`source_independent_projection`；
- Top-N=20，budgets=`[20,50,75,90,all]`；
- profiling scope=`frozen_prediction_downstream`，warmup=20，repeats=3；
- Python major/minor=`3.11`；
- formal execution 要求调用方通过 `--expected-course-commit` 明确给出完整 commit，并验证当前 HEAD 精确相等；
- formal execution 只允许课程分支、clean tracked worktree、clean index、课程路径全部由 HEAD 覆盖且课程路径内没有 untracked 文件；
- `data/`、`incoming/`、`results/`、`outputs/`、`logs/` 和课程范围外 Phase 4 untracked 不导致误拒绝，只记录 count/path warning。

CF/SASRec prediction、两份 SID mapping、valid CSV、frozen S5 config 与 P2 model 的仓库相对路径、大小和 SHA-256 均冻结。真实 formal run 在创建输出目录之前重新计算这些文件的 SHA-256；plan-only 不扫描它们。

## 6. Formal execution gates

任何一项不满足即拒绝正式运行：

1. 当前 HEAD 与调用方明确给出的课程 commit 不一致，或课程代码未被 HEAD 覆盖；
2. tracked worktree 或 index 不干净；
3. Python 不是 3.11；
4. 不可变输入 path/size/SHA-256 漂移；
5. split、selected split、1360 样本、预算顺序或 Top-20 漂移；
6. candidate/fusion/ranker 参数漂移；
7. 出现 test 输入或 unresolved placeholder；
8. 运行时 cardinality gate 非 PASS。

该保护在 `CoursePipeline.run()` 内执行；formal execution 缺少 `--expected-course-commit` 会直接拒绝。`--formal-preflight` 只检查 HEAD、Git、Python、输入、split、ranker、plan 和输出 run-id，不创建正式实验。plan-only 和 synthetic dry-run 同样不创建正式实验。

## 7. Cardinality decision

`course-valid-cardinality-v1` 的只读逐样本复算支持以下冻结网格：

```text
20, 50, 75, 90, all
```

K20 是极端预算，K50 是中等预算，K75 约减少 21.72% candidate scoring，K90 约减少 6.07% 并用于寻找 near-lossless operating point，all 是完整候选基线。v1 因 dirty provenance 和 Python 环境不一致，仅标记为 `PRELIMINARY_CARDINALITY_EVIDENCE`。完整数字见 closeout 文档。

## 8. Targeted validation

实际执行且通过：

```text
python -m py_compile minionerec_system/*.py scripts/run_course_system.py scripts/check_course_system_artifacts.py
PASS

pytest -q tests/course_system/
39 passed

smoke config plan-only
PASS

valid template plan-only
PASS

frozen K-matrix config plan-only
PASS

4-sample synthetic dry-run
PASS; post-dedup=95; K20/K50/K75/K90/all vectors distinct; gate=PASS

4-sample synthetic smoke + read-only checker
COMPLETED; checker=PASS
```

定向测试另覆盖无 untracked PASS、`results/`/`incoming/`/Phase 4 untracked PASS、课程目录 untracked FAIL、runner tracked diff FAIL、index FAIL、HEAD mismatch FAIL、checker guard provenance，以及 guard 不删除/移动/忽略文件。

## 9. Explicitly not run

本轮未运行 cardinality v2、正式 K matrix、full-valid、任何 test、模型 inference、ranker 训练、GPU profiling、正式 warm-up/repeat、prefix expansion、全量测试或大规模哈希扫描，也未修改历史 S4/S5/S6/Stage 7 结果与 Phase 4 文件。

## 10. Next gate

WSL 传输准备环境为 `minionerec-dev`；AutoDL 执行环境为 `minionerec`。用户按照 `course_system_autodl_manual_runbook_v1.md` 的 Stage A/B/C/D 手动运行 `course-valid-cardinality-v2`。Stage D 完成后必须停止并回传 v2，不能自动进入 K matrix。未来 Stage E 只有在 Codex 审计 v2 且用户再次明确确认后才能开展；任一不一致均为：

```text
DO_NOT_RUN_K_MATRIX
```

## 11. AutoDL handoff repair

上一版交付命令在交互 shell 中组合了全局 untracked 断言与 `set -e`，会被 59 个已知 Phase 4 untracked 触发并退出 shell；同时误用了 AutoDL 环境名 `minionerec-dev`，并把 v2 PASS 与 K matrix 串联。修复后：

- WSL Stage A 与 AutoDL Stage B/C/D 完全分离；
- AutoDL 环境固定为 `minionerec`，不存在即明确停止，不创建环境、不安装依赖；
- Git guard 只把课程路径 untracked 视为错误，范围外 untracked 只记录 provenance；
- expected course commit 由 Stage A 人工传递，并在 formal preflight/正式运行/checker 三处核对；
- fail-fast 检查只在显式子 shell 中执行，每项输出 PASS/FAIL、expected、actual；
- Stage D 只生成 cardinality v2 return bundle，然后停止等待人工审计；Stage E 本轮没有执行命令。
