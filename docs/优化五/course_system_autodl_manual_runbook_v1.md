# COURSE-SYSTEM AutoDL Manual Runbook V1

本手册只用于课程 cardinality v2 的人工 handoff。五个阶段必须分开执行，Stage D 后强制停止。

```text
WSL environment=minionerec-dev
AutoDL environment=minionerec
budget grid=20,50,75,90,all
cardinality v2 run id=course-valid-cardinality-v2
```

不要在交互 shell 中直接设置 `set -euo pipefail`。本手册需要 fail-fast 的检查均放在显式子 shell 内；失败只结束子 shell，不会关闭当前终端。不要把 WSL 和 AutoDL 命令混在同一命令块中。

## Stage A：WSL 提交检查与传输准备

环境/仓库：WSL，`/home/dell/projects/MiniOneRec`。本阶段只检查 Git 和准备传输，不运行课程 Python；如需本地 Python，环境只能使用 `minionerec-dev`。

### A0. 检查 handoff commit

```bash
cd /home/dell/projects/MiniOneRec

bash -s -- <<'COURSE_STAGE_A'
EXPECTED_BRANCH="course-system-bounded-rerank-v1"
EXPECTED_PARENT="ca98c381f39b1fc5a38e3093bc611eb64d4d1f92"
EXPECTED_SUBJECT="Fix course AutoDL preflight and staged handoff"
FAILURES=0

check_equal() {
  LABEL="$1"
  EXPECTED="$2"
  ACTUAL="$3"
  if [ "$ACTUAL" = "$EXPECTED" ]; then
    printf 'PASS | %s | expected=%s | actual=%s\n' "$LABEL" "$EXPECTED" "$ACTUAL"
  else
    printf 'FAIL | %s | expected=%s | actual=%s\n' "$LABEL" "$EXPECTED" "$ACTUAL"
    FAILURES=$((FAILURES + 1))
  fi
}

BRANCH="$(git branch --show-current 2>/dev/null)"
HEAD_COMMIT="$(git rev-parse HEAD 2>/dev/null)"
PARENT_COMMIT="$(git rev-parse HEAD^ 2>/dev/null)"
SUBJECT="$(git log -1 --format=%s 2>/dev/null)"
TRACKED_DIFF="$(git diff --name-only)"
INDEX_DIFF="$(git diff --cached --name-only)"

check_equal "branch" "$EXPECTED_BRANCH" "$BRANCH"
check_equal "parent commit" "$EXPECTED_PARENT" "$PARENT_COMMIT"
check_equal "commit subject" "$EXPECTED_SUBJECT" "$SUBJECT"
check_equal "tracked worktree" "clean" "$([ -z "$TRACKED_DIFF" ] && printf clean || printf '%s' "$TRACKED_DIFF")"
check_equal "index" "clean" "$([ -z "$INDEX_DIFF" ] && printf clean || printf '%s' "$INDEX_DIFF")"

printf 'INFO | handoff commit | actual=%s\n' "$HEAD_COMMIT"
printf 'INFO | commit files begin\n'
git show --format='' --name-only HEAD
printf 'INFO | commit files end\n'
printf 'INFO | remotes begin\n'
git remote -v
printf 'INFO | remotes end\n'
printf 'INFO | unrelated untracked count | actual=%s | not a failure\n' "$(git ls-files --others --exclude-standard | wc -l)"

if [ "$FAILURES" -ne 0 ]; then
  printf 'STAGE_A=FAIL | expected=all checks PASS | actual=%s failures\n' "$FAILURES"
  exit 1
fi
printf 'STAGE_A=PASS\n'
printf 'COURSE_HANDOFF_COMMIT=%s\n' "$HEAD_COMMIT"
COURSE_STAGE_A
```

记录 `COURSE_HANDOFF_COMMIT`，Stage B 必须逐字使用这个 40 位 commit。59 个 Phase 4 untracked 只作信息显示，不删除、不移动、不暂存。

### A1. 二选一：通过 Git remote

由用户确认 remote 后手动执行；本任务未执行 push：

```bash
cd /home/dell/projects/MiniOneRec
git remote -v
git push -u origin course-system-bounded-rerank-v1
```

### A2. 二选一：通过 git bundle

不使用 remote 时手动创建 bundle：

```bash
cd /home/dell/projects/MiniOneRec
COURSE_BRANCH="course-system-bounded-rerank-v1"
COURSE_COMMIT="$(git rev-parse HEAD)"
COURSE_BUNDLE="${HOME}/course-system-${COURSE_COMMIT}.bundle"

if git bundle create "${COURSE_BUNDLE}" "${COURSE_BRANCH}"; then
  git bundle verify "${COURSE_BUNDLE}"
  printf 'PASS | git bundle | expected=tracked Git history only | actual=%s\n' "${COURSE_BUNDLE}"
else
  printf 'FAIL | git bundle | expected=created | actual=git bundle create failed\n'
fi
```

`git bundle` 只包含 Git objects/refs，不会打入 Phase 4 untracked、数据、prediction 或日志。将生成的 `.bundle` 文件人工传到 AutoDL。

## Stage B：AutoDL 获取精确 commit

环境/仓库：AutoDL，默认仓库 `/root/autodl-tmp/projects/MiniOneRec`。不要假设 commit object 已存在。

### B0. 设置人工记录的精确 commit

```bash
AUTODL_REPO="/root/autodl-tmp/projects/MiniOneRec"
COURSE_BRANCH="course-system-bounded-rerank-v1"
read -r -p "Paste COURSE_HANDOFF_COMMIT from Stage A: " EXPECTED_COURSE_COMMIT
export AUTODL_REPO COURSE_BRANCH EXPECTED_COURSE_COMMIT

case "${EXPECTED_COURSE_COMMIT}" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f])
    printf 'PASS | expected commit format | expected=40 lowercase hex | actual=%s\n' "${EXPECTED_COURSE_COMMIT}" ;;
  *)
    printf 'FAIL | expected commit format | expected=40 lowercase hex | actual=%s\n' "${EXPECTED_COURSE_COMMIT}"
    printf 'STOP: correct EXPECTED_COURSE_COMMIT before continuing.\n' ;;
esac
```

### B1. 二选一：从 remote fetch

```bash
cd "${AUTODL_REPO}"
if git fetch origin "refs/heads/${COURSE_BRANCH}:refs/remotes/origin/${COURSE_BRANCH}"; then
  export COURSE_SOURCE_REF="refs/remotes/origin/${COURSE_BRANCH}"
  printf 'PASS | fetch remote | expected=%s | actual=%s\n' "${COURSE_BRANCH}" "${COURSE_SOURCE_REF}"
else
  printf 'FAIL | fetch remote | expected=commit fetched | actual=git fetch failed\n'
  printf 'STOP: do not continue to checkout or preflight.\n'
fi
```

### B2. 二选一：从 bundle fetch

把 bundle 路径改为实际传输位置：

```bash
cd "${AUTODL_REPO}"
COURSE_BUNDLE="/root/autodl-tmp/course-system.bundle"

if git bundle verify "${COURSE_BUNDLE}" && \
   git fetch "${COURSE_BUNDLE}" "refs/heads/${COURSE_BRANCH}:refs/remotes/course-bundle/${COURSE_BRANCH}"; then
  export COURSE_SOURCE_REF="refs/remotes/course-bundle/${COURSE_BRANCH}"
  printf 'PASS | fetch bundle | expected=%s | actual=%s\n' "${COURSE_BRANCH}" "${COURSE_SOURCE_REF}"
else
  printf 'FAIL | fetch bundle | expected=commit fetched | actual=bundle verify/fetch failed\n'
  printf 'STOP: do not continue to checkout or preflight.\n'
fi
```

### B3. Checkout/switch 并验证精确 commit

只在 B1 或 B2 成功并设置 `COURSE_SOURCE_REF` 后运行：

```bash
bash -s -- "${AUTODL_REPO}" "${COURSE_BRANCH}" "${COURSE_SOURCE_REF}" "${EXPECTED_COURSE_COMMIT}" <<'COURSE_STAGE_B'
REPO="$1"
BRANCH="$2"
SOURCE_REF="$3"
EXPECTED="$4"
cd "$REPO" || { printf 'FAIL | repository | expected=%s | actual=cd failed\n' "$REPO"; exit 1; }

SOURCE_HEAD="$(git rev-parse "$SOURCE_REF" 2>/dev/null)"
if [ "$SOURCE_HEAD" != "$EXPECTED" ]; then
  printf 'FAIL | fetched ref HEAD | expected=%s | actual=%s\n' "$EXPECTED" "$SOURCE_HEAD"
  exit 1
fi
printf 'PASS | fetched ref HEAD | expected=%s | actual=%s\n' "$EXPECTED" "$SOURCE_HEAD"

if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  git switch "$BRANCH" || { printf 'FAIL | switch branch | expected=%s | actual=git switch failed\n' "$BRANCH"; exit 1; }
  git merge --ff-only "$SOURCE_REF" || { printf 'FAIL | fast-forward branch | expected=%s | actual=non-fast-forward or dirty branch\n' "$SOURCE_REF"; exit 1; }
else
  git switch -c "$BRANCH" "$SOURCE_REF" || { printf 'FAIL | create branch | expected=%s | actual=git switch failed\n' "$BRANCH"; exit 1; }
fi

ACTUAL="$(git rev-parse HEAD 2>/dev/null)"
if [ "$ACTUAL" != "$EXPECTED" ]; then
  printf 'FAIL | exact checkout | expected=%s | actual=%s\n' "$EXPECTED" "$ACTUAL"
  printf 'STOP: do not reset automatically; inspect the AutoDL branch state.\n'
  exit 1
fi
printf 'PASS | exact checkout | expected=%s | actual=%s\n' "$EXPECTED" "$ACTUAL"
COURSE_STAGE_B
```

### B4. 激活 AutoDL 环境

AutoDL 只能使用 `minionerec`，不能使用 `minionerec-dev`：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -Fxq minionerec; then
  conda activate minionerec
  printf 'PASS | AutoDL conda environment | expected=minionerec | actual=%s\n' "${CONDA_DEFAULT_ENV:-unset}"
else
  printf 'FAIL | AutoDL conda environment | expected=minionerec | actual=missing\n'
  printf 'STOP: do not create an environment or install dependencies automatically.\n'
fi
```

## Stage C：AutoDL preflight

确认 Stage B 成功、当前环境显示 `minionerec` 后，运行以下受控子 shell。它不全局禁止 untracked；只拒绝 protected course paths 内的 untracked。所有失败均输出 `expected` 和 `actual`。

```bash
bash -s -- "${AUTODL_REPO}" "${EXPECTED_COURSE_COMMIT}" <<'COURSE_STAGE_C'
REPO="$1"
EXPECTED="$2"
RUN_ID="course-valid-cardinality-v2"
CONFIG="configs/course_system/valid_cf_sasrec_exact_template.yaml"
FAILURES=0

record_equal() {
  LABEL="$1"
  EXPECTED_VALUE="$2"
  ACTUAL_VALUE="$3"
  if [ "$ACTUAL_VALUE" = "$EXPECTED_VALUE" ]; then
    printf 'PASS | %s | expected=%s | actual=%s\n' "$LABEL" "$EXPECTED_VALUE" "$ACTUAL_VALUE"
  else
    printf 'FAIL | %s | expected=%s | actual=%s\n' "$LABEL" "$EXPECTED_VALUE" "$ACTUAL_VALUE"
    FAILURES=$((FAILURES + 1))
  fi
}

cd "$REPO" || { printf 'FAIL | repository | expected=%s | actual=cd failed\n' "$REPO"; exit 1; }

record_equal "exact HEAD" "$EXPECTED" "$(git rev-parse HEAD 2>/dev/null)"
record_equal "branch" "course-system-bounded-rerank-v1" "$(git branch --show-current 2>/dev/null)"

TRACKED_DIFF="$(git diff --name-only)"
INDEX_DIFF="$(git diff --cached --name-only)"
COURSE_UNTRACKED="$(git ls-files --others --exclude-standard -- \
  minionerec_system configs/course_system \
  scripts/run_course_system.py scripts/check_course_system_artifacts.py)"

record_equal "tracked worktree" "clean" "$([ -z "$TRACKED_DIFF" ] && printf clean || printf '%s' "$TRACKED_DIFF")"
record_equal "index" "clean" "$([ -z "$INDEX_DIFF" ] && printf clean || printf '%s' "$INDEX_DIFF")"
record_equal "protected course untracked" "none" "$([ -z "$COURSE_UNTRACKED" ] && printf none || printf '%s' "$COURSE_UNTRACKED")"

MISSING_HEAD_PATHS=""
for PATH_IN_HEAD in minionerec_system configs/course_system scripts/run_course_system.py scripts/check_course_system_artifacts.py; do
  if ! git cat-file -e "HEAD:${PATH_IN_HEAD}" 2>/dev/null; then
    MISSING_HEAD_PATHS="${MISSING_HEAD_PATHS} ${PATH_IN_HEAD}"
  fi
done
record_equal "protected paths tracked by HEAD" "all tracked" "$([ -z "$MISSING_HEAD_PATHS" ] && printf 'all tracked' || printf '%s' "$MISSING_HEAD_PATHS")"

record_equal "AutoDL conda environment" "minionerec" "${CONDA_DEFAULT_ENV:-unset}"
record_equal "Python major.minor" "3.11" "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)"

UNRELATED_COUNT="$(git ls-files --others --exclude-standard | wc -l)"
printf 'INFO | unrelated untracked | expected=allowed outside protected paths | actual_count=%s\n' "$UNRELATED_COUNT"

PLAN_LOG="$(mktemp)"
python scripts/run_course_system.py --config "$CONFIG" --plan-only >"$PLAN_LOG" 2>&1
PLAN_RC=$?
cat "$PLAN_LOG"
rm -f "$PLAN_LOG"
record_equal "config plan-only exit" "0" "$PLAN_RC"

PREFLIGHT_LOG="$(mktemp)"
python scripts/run_course_system.py \
  --config "$CONFIG" \
  --formal-preflight \
  --expected-course-commit "$EXPECTED" \
  --run-id "$RUN_ID" >"$PREFLIGHT_LOG" 2>&1
PREFLIGHT_RC=$?
cat "$PREFLIGHT_LOG"
rm -f "$PREFLIGHT_LOG"
record_equal "formal input/split/ranker/output preflight exit" "0" "$PREFLIGHT_RC"

if [ "$FAILURES" -ne 0 ]; then
  printf 'STAGE_C=FAIL | expected=all checks PASS | actual=%s failures\n' "$FAILURES"
  exit 1
fi
printf 'STAGE_C=PASS | expected=no experiment run | actual=preflight only\n'
COURSE_STAGE_C
```

`results/`、`incoming/`、`data/`、`outputs/`、`logs/` 或课程路径外 Phase 4 untracked 可以存在；它们只会出现在 provenance/warning 中。

## Stage D：只运行 clean cardinality v2

只有 Stage C 为 PASS 时运行。此子 shell 只执行 cardinality audit、artifact checker、exact-value verification 和 v2 return bundle；不包含 K matrix 命令。

```bash
bash -s -- "${AUTODL_REPO}" "${EXPECTED_COURSE_COMMIT}" <<'COURSE_STAGE_D'
REPO="$1"
EXPECTED="$2"
RUN_ID="course-valid-cardinality-v2"
CONFIG="configs/course_system/valid_cf_sasrec_exact_template.yaml"
V2_REL="results/course_system/valid/Industrial_and_Scientific/${RUN_ID}"
V2_DIR="${REPO}/${V2_REL}"

stop_stage() {
  LABEL="$1"
  EXPECTED_VALUE="$2"
  ACTUAL_VALUE="$3"
  printf 'FAIL | %s | expected=%s | actual=%s\n' "$LABEL" "$EXPECTED_VALUE" "$ACTUAL_VALUE"
  printf 'DO_NOT_RUN_K_MATRIX\n'
  exit 1
}

cd "$REPO" || stop_stage "repository" "$REPO" "cd failed"

python scripts/run_course_system.py \
  --config "$CONFIG" \
  --formal-preflight \
  --expected-course-commit "$EXPECTED" \
  --run-id "$RUN_ID"
RC=$?
[ "$RC" -eq 0 ] || stop_stage "formal preflight exit" "0" "$RC"
printf 'PASS | formal preflight exit | expected=0 | actual=%s\n' "$RC"

python scripts/run_course_system.py \
  --config "$CONFIG" \
  --cardinality-audit \
  --expected-course-commit "$EXPECTED" \
  --run-id "$RUN_ID"
RC=$?
[ "$RC" -eq 0 ] || stop_stage "cardinality v2 exit" "0" "$RC"
printf 'PASS | cardinality v2 exit | expected=0 | actual=%s\n' "$RC"

python scripts/check_course_system_artifacts.py --run-dir "$V2_DIR"
RC=$?
[ "$RC" -eq 0 ] || stop_stage "artifact checker exit" "0" "$RC"
printf 'PASS | artifact checker exit | expected=0 | actual=%s\n' "$RC"

python - "$V2_DIR" "$EXPECTED" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_commit = sys.argv[2]
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
status = json.loads((root / "status.json").read_text(encoding="utf-8"))
summary = json.loads((root / "cardinality_summary.json").read_text(encoding="utf-8"))
with (root / "cardinality_per_sample.csv").open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

assert status["status"] == "completed"
assert manifest["head"] == expected_commit
assert manifest["formal_execution_guard"]["expected_course_commit"] == expected_commit
assert manifest["formal_execution_guard"]["status"] == "PASS"
assert manifest["sample_count"] == 1360
assert manifest["quality_split"] == "valid_select"
assert manifest["budgets"] == [20, 50, 75, 90, "all"]
assert summary["gate"]["status"] == "PASS"
assert len(rows) == 1360
assert len({row["sample_id"] for row in rows}) == 1360
assert summary["min"] == 89
assert math.isclose(summary["mean"], 95.81470588235294, rel_tol=0, abs_tol=1e-12)
assert summary["median"] == 96
assert summary["p75"] == 97
assert summary["p95"] == 99
assert summary["max"] == 100
assert summary["total_candidates_before_budget"] == 130308

expected_totals = {"k20": 27200, "k50": 68000, "k75": 102000, "k90": 122399, "all": 130308}
for label, expected in expected_totals.items():
    actual = summary["budgets"][label]["total_candidates_after_budget"]
    assert actual == expected, (label, actual, expected)

k90_truncated = sum(row["truncated_at_k90"].strip().lower() in {"true", "1"} for row in rows)
assert k90_truncated == 1351, k90_truncated
print("PASS | v2 exact values | expected=frozen contract | actual=all exact values matched")
PY
RC=$?
[ "$RC" -eq 0 ] || stop_stage "v2 exact-value verification exit" "0" "$RC"

RETURN_BUNDLE="/root/autodl-tmp/course-valid-cardinality-v2-${EXPECTED}.tar.gz"
tar -czf "$RETURN_BUNDLE" -C "$REPO" "$V2_REL"
RC=$?
[ "$RC" -eq 0 ] || stop_stage "v2 return bundle" "created" "tar exit ${RC}"
sha256sum "$RETURN_BUNDLE"
printf 'PASS | v2 return bundle | expected=created | actual=%s\n' "$RETURN_BUNDLE"
printf 'WAITING_FOR_USER_CARDINALITY_V2_REVIEW\n'
COURSE_STAGE_D
```

Stage D 到此结束。请把 return bundle 和其 SHA-256 回传，不要继续运行其他正式阶段。

## Stage E：未来 K matrix

本轮不提供 Stage E 执行命令。进入未来 K matrix 必须同时满足：

1. 用户回传完整 cardinality v2 bundle 与 SHA-256；
2. Codex 对 v2 artifact、checker 和 exact values 完成只读审计；
3. 用户在审计后再次明确确认运行 K matrix。

在这三项完成之前，状态固定为：

```text
WAITING_FOR_USER_CARDINALITY_V2_REVIEW
DO_NOT_RUN_K_MATRIX
```
