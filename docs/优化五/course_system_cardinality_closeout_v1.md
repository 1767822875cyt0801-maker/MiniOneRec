# COURSE-CARDINALITY-CLOSEOUT-V1

## 1. v1 artifact inventory

只读复核目录：

```text
results/course_system/valid/Industrial_and_Scientific/course-valid-cardinality-v1
```

目录中的 9 个合同文件均存在且非空：

| 文件 | 只读用途 |
|---|---|
| `cardinality_per_sample.csv` | 独立逐样本复算 |
| `cardinality_summary.json` | 与独立聚合结果交叉核对 |
| `command.txt` | 运行入口与 run id |
| `command_plan.json` | split、stream、exact 模式和阶段顺序 |
| `config.resolved.json` | 当次 resolved 参数 |
| `environment.json` | Python/torch 环境证据 |
| `manifest.json` | Git provenance、sample/split 与完成状态 |
| `run.log` | 阶段与 gate 状态 |
| `status.json` | lifecycle 完成状态 |

复核前记录了全部 9 个文件的 SHA-256；closeout 后再次比对，v1 目录未被修改。

## 2. Sample/split consistency

从 CSV 独立计数，而不是采用给定摘要：

```text
quality_split=valid_select
sample_count=1360
unique_sample_id=1360
CF candidate count unique values=[50]
SASRec candidate count unique values=[50]
pre-dedup count unique values=[100]
```

manifest、plan、resolved config 与逐样本计数均指向 `valid_select`；观察到的 split evidence 为 `valid_fit=3172`、`valid_select=1360`。

## 3. Exact candidate-pool statistics

对 `post_dedup_count` 排序并用线性分位插值独立计算：

| 统计量 | 值 |
|---|---:|
| min | 89 |
| mean | 95.81470588235294 |
| median | 96 |
| p75 | 97 |
| p95 | 99 |
| max | 100 |
| total before budget | 130308 |

输入是 CF-SID + SASRec-SID frozen prediction 的 exact expansion；每条样本融合前为 100 个候选。

## 4. Dedup statistics

逐样本计算 `pre_dedup_count - post_dedup_count`：

```text
total_removed=5692
mean_removed=4.185294117647059
fraction_removed=0.041852941176470586
```

即 stable dedup 总体移除约 4.1853%。

## 5. Observed K20/K50/K75/all reductions

对每条样本使用通用公式 `effective_count=min(post_dedup_count,K)`；all 直接保留 post-dedup 候选：

| Budget | truncated samples | total after | mean removed | candidate reduction |
|---|---:|---:|---:|---:|
| K20 | 1360 | 27200 | 75.81470588235294 | 0.7912637750560211 |
| K50 | 1360 | 68000 | 45.81470588235294 | 0.4781594376400528 |
| K75 | 1360 | 102000 | 20.814705882352943 | 0.21723915646007919 |
| all | 0 | 130308 | 0 | 0 |

这些是 v1 逐样本候选规模的真实削减结果，不是质量或性能结果。

## 6. Expected K90 from per-sample data

K90 不依赖新增算法分支。对 1360 条 `post_dedup_count` 逐一计算 `min(count,90)` 得到：

```text
truncated_samples=1351
truncated_fraction=0.9933823529411765
total_after_budget=122399
mean_removed=5.815441176470588
candidate_reduction=0.0606946618780121
```

K90 effective-count vector 与 K75、all 均不同，因而在该分布上实际生效。

## 7. Budget-grid freeze

正式 candidate-budget grid 冻结为且仅为：

```text
K=[20,50,75,90,all]
```

- K20：极端预算；
- K50：中等预算；
- K75：约减少 21.72% candidate scoring；
- K90：保守预算，约减少 6.07%，用于寻找 near-lossless operating point；
- all：完整候选基线。

不删除原有有限预算，也不增加其他 K。

## 8. Acceptance

```text
CARDINALITY_GRID_ACCEPTED
```

该接受仅冻结候选规模网格，不把 v1 升级为最终 provenance 证据。

## 9. Dirty-state caveat

v1 manifest 记录：

```text
dirty_state=true
head=8c4aa08ed76dd7b93836502e474d7093073e42ff
```

当次课程系统代码尚未被该 HEAD 覆盖，因此数字可用于预算设计，但 Git provenance 不满足最终冻结条件。

## 10. Python environment mismatch

v1 environment 记录 `Python 3.13.13`、`torch=null`。WSL handoff 环境为 `minionerec-dev`；预定 AutoDL 正式环境为 `minionerec`、Python 3.11。这里只报告环境与 provenance 不一致，不推测运行者。

## 11. Evidence classification

v1 的唯一分类是：

```text
PRELIMINARY_CARDINALITY_EVIDENCE
```

不得称为最终 frozen cardinality evidence。

## 12. Clean cardinality v2 reproduction contract

`course-valid-cardinality-v2` 必须在以下条件下由用户手动运行：课程独立 branch、调用方提供的精确课程 commit 已 checkout、tracked worktree clean、index clean、课程路径内无 untracked、AutoDL `minionerec` Python 3.11 环境、不可变输入身份检查 PASS。课程范围外 untracked 数据和 Phase 4 文件只记录 provenance，不作为失败。预期精确值为：

```text
sample_count=1360
unique_sample_id=1360
min=89
mean=95.81470588235294
median=96
p75=97
p95=99
max=100
total_before_budget=130308
K20_total=27200
K50_total=68000
K75_total=102000
K90_total=122399
all_total=130308
K90_truncated_samples=1351
gate=PASS
checker=PASS
```

任一不一致都必须输出并遵守：

```text
DO_NOT_RUN_K_MATRIX
```

## 13. Formal K-matrix prerequisites

正式执行入口会在创建输出前检查 expected commit/HEAD、Python、branch、tracked/index clean、课程路径由 HEAD 覆盖且无 untracked、固定输入 path/size/SHA-256、exact 模式、冻结 fusion/ranker、Top-20、valid-select、1360 样本和五档预算。

本轮 Stage D 只允许 cardinality v2、checker、exact-value verification 和 v2 return bundle。完成后必须输出 `WAITING_FOR_USER_CARDINALITY_V2_REVIEW` 并停止。未来 K matrix 必须等待用户回传 v2、Codex 审计和用户明确确认；本轮 runbook 不提供从 Stage D 自动跳转的代码。

无关 Phase 4 untracked 文件不会被删除或移动；guard 能区分它们并给出 warning。最终证据仍规定使用 AutoDL clean checkout。

## 14. Test-read statement

本轮没有打开或读取任何 test split、test CSV、test prediction 或 test artifact，没有创建 test config，也没有用推荐质量调整预算。正式课程配置明确拒绝 test 输入。

WSL/AutoDL 分阶段命令见 `docs/优化五/course_system_autodl_manual_runbook_v1.md`。
