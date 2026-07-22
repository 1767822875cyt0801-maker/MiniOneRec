# Phase 3 Repository Cleanup Final Report

- 完成日期：2026-07-22
- 发布名称：Phase3 MiniOneRec Strong Behavior Optimization Release
- 发布分支：`yuting-reproduce`

## 1. Executive Summary

Phase 3 仓库清理、发布提交和 GitHub 同步已完成。发布内容以 commit `3778ddb46dfde5fd150311cc27b2b7c7bf0313a7` 为代码/实验工程基线；大型 checkpoints、模型权重、generated predictions、candidate JSONL、logs、cache、压缩回传包及本地数据均未进入提交。

本次未运行训练、推理或 GPU 实验，未新增或修改算法逻辑。清理动作只新增仓库卫生规则和报告，并将工作区中已经存在的 Phase 3 代码、脚本、测试、配置、文档及可复现性清单纳入发布。

## 2. Deleted Files

删除文件数：**0**。

没有文件同时满足以下三个删除条件：

1. 已被新脚本替代；
2. 无任何引用；
3. 不影响已有实验复现。

因此，本次没有删除 `results/` evidence、validation reports、manifest、实验总结、AutoDL 回传包、模型 checkpoint 或其他用途不确定文件。`Zone.Identifier`、带行号源码快照和关键行快照也采用“保留但忽略”的保守策略。

## 3. Ignored Files

### 3.1 原有规则继续覆盖

- Python cache：`__pycache__/`、`*.pyc`、`*.pyo`、`*.pyd`
- 环境：`.env`、`.venv/`、`venv/`、`conda-meta/`
- 日志：`*.log`、`wandb/`、`runs/`、`logs/`
- 数据/输出：`data/`、`dataset/`、`datasets/`、`outputs/`、`output/`、`results/`、`checkpoints/`、`checkpoint/`、`saved_models/`
- 模型权重：`*.pt`、`*.pth`、`*.bin`、`*.safetensors`、`*.ckpt`
- IDE/system：`.vscode/`、`.idea/`、`.DS_Store`

### 3.2 本次新增规则

- 测试/工具 cache：`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、`.cache/`
- `archive/` 大型输出；仅放行并提交 `ARCHIVED_PATHS.txt` 与 `REMOTE_SHA256SUMS.txt`
- `incoming/` 回传内容；仅放行并提交 `*.sha256`
- generated candidates/predictions：`*.jsonl`、`predictions*.json`
- transfer bundles：`*.tar`、`*.tar.gz`、`*.tgz`、`*.zip`
- 本地临时快照：`build_sasrec_embeddings_numbered.txt`、`sasrec_key_lines.txt`
- Windows 下载元数据：`*:Zone.Identifier`
- 其他临时文件：`*.tmp`、`*.temp`、`*.swp`、`*~`

清理后约 24 GiB 的 `archive/`、492 MiB 的 `incoming/`、1.4 GiB 的 `results/` 和 454 MiB 的 `data/` 均保留在本地且被忽略。没有 evidence artifact 被删除。

## 4. Committed Files

Phase 3 release commit 共包含 **90 个文件变更**：26,737 行新增、31 行删除；其中 86 个新文件、4 个既有文件修改。提交总工作树文件体积约 3.9 MB，最大单文件约 1.63 MB，没有超过 100 MiB 的文件。

按目录分类：

| 类别 | 数量 | 主要内容 |
|---|---:|---|
| 根目录/仓库规则 | 6 | `.gitattributes`、`.gitignore`、Phase 3 SID 方案、约束生成修正、SASRec candidate export/merge 工具 |
| `archive/` | 2 | 归档路径清单、远端 SHA-256 清单 |
| `configs/` | 3 | S6 cost-aware auxiliary 配置和 promotion/ranker protocol |
| `docs/` | 27 | S4/S5/S6 审计与 closeout、S6 final 总结、Phase 4 交接审计/计划/参考论文、本次 cleanup audit |
| `incoming/` | 6 | S4/S5/S6 AutoDL 回传 bundle 的 SHA-256 sidecar |
| `scripts/` | 31 | S4/S5/S6/Stage7 构建、审计、验证、rerank、复现与运行脚本 |
| `tests/` | 15 | S4/S5/S6/SASRec 流程和 artifact audit 单元测试，以及 provider 测试环境兼容修正 |

两份 Phase 4 参考论文 PDF 已通过 `.gitattributes` 明确标记为 binary；它们分别约 1.08 MiB 和 1.56 MiB，不是模型、checkpoint 或生成结果。

发布提交 message：

```text
Finalize Phase3 strong behavior generative recommendation optimization
```

## 5. Commit Hash

Phase 3 release commit：

```text
3778ddb46dfde5fd150311cc27b2b7c7bf0313a7
```

该 commit 是 Phase 4 开始前的代码与实验工程基线。本报告在 release 已产生并同步后生成，因此以独立 docs-only 跟随提交归档；该跟随提交不改变上述代码基线。

## 6. GitHub Sync Status

- Remote：`origin`
- URL：`git@github.com:1767822875cyt0801-maker/MiniOneRec.git`
- Branch：`yuting-reproduce`
- Upstream：`origin/yuting-reproduce`
- push 类型：普通 fast-forward push，无 force
- push 范围：`be65b86..3778ddb`
- release push 状态：**成功**
- release push 后分歧：`local-only=0`、`remote-only=0`
- release push 后本地与远端 hash：均为 `3778ddb46dfde5fd150311cc27b2b7c7bf0313a7`

push 前已执行 `git fetch origin yuting-reproduce`。当时分歧为 `local-only=1`、`remote-only=0`，确认无 branch mismatch、remote divergence 或 authentication problem 后才执行 push。

## 7. Validation

- 配置：3 个新增 JSON 均通过 `python -m json.tool`。
- Shell：10 个 changed/untracked shell 脚本均通过 `bash -n`。
- 单元测试：`PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests`，结果为 **166 passed, 5 skipped**。
- 全仓库默认 pytest 收集：在历史根目录 `data_test.py` 导入阶段因本地缺少 `pandas` 而停止；该次没有执行测试、训练或实验。正式 `tests/` 目录随后完整通过。
- 暂存安全检查：checkpoint/model weight/output/results/candidate JSONL/log/archive bundle 命中数为 0。
- 大文件检查：暂存文件超过 100 MiB 的数量为 0。
- 源码/脚本/配置空白检查：通过。
- 密钥扫描：未发现硬编码 API key、token 或 private key；匹配项仅为从运行环境/配置读取 key 的既有代码。

## 8. Repository State Before Phase 4

Phase 3 release push 完成时：

- `yuting-reproduce` 与 `origin/yuting-reproduce` 同步；
- tracked 工作树 clean；
- 可见 untracked 文件为 0；
- 本地大型 outputs/results/checkpoints/data/incoming bundles 仍在原位并被忽略；
- Phase 3 evidence、validation reports、manifest、实验总结和 SHA-256 清单均被保留；
- 没有 checkpoint、generated output 或大文件进入 Git 历史；
- Phase 4 架构审计、项目计划和论文输入已作为交接文档纳入仓库；
- Phase 4 应以 release commit `3778ddb46dfde5fd150311cc27b2b7c7bf0313a7` 为代码基线。

完成本报告的 docs-only 归档并同步后，仓库即可进入 Phase 4；本阶段不再开展实现、训练或实验。
