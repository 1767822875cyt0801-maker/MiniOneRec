# MiniOneRec Multi-GPU SFT Guide

This guide adds a single-node DDP path for SFT while keeping the original
single-GPU `sft.py` and `sft.sh` workflows available.

## What Changed

- `sft.py` still accepts the legacy `batch_size` and `micro_batch_size`
  arguments.
- New explicit DDP arguments are supported:
  - `per_device_train_batch_size`
  - `per_device_eval_batch_size`
  - `gradient_accumulation_steps`
  - `bf16`
  - `gradient_checkpointing`
  - `ddp_find_unused_parameters`
  - `save_on_each_node`
  - `save_total_limit`
  - `load_best_model_at_end`
  - `early_stopping_patience`
- `final_checkpoint` is now written only by the main process in DDP.
- Evaluation still uses the original single-process `evaluate.py`.

Effective batch size:

```text
effective_batch_size =
  per_device_train_batch_size * num_gpus * gradient_accumulation_steps
```

## Single-GPU Training Still Works

The old command style is still valid:

```bash
CUDA_VISIBLE_DEVICES=0 python sft.py \
  --base_model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --batch_size 128 \
  --micro_batch_size 4 \
  --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output_dir outputs/sft_single_gpu_test \
  --category Industrial_and_Scientific \
  --train_from_scratch False \
  --seed 42 \
  --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
  --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
  --freeze_LLM False
```

The existing `sft.sh` is not deleted or replaced.

## DDP Sanity Check

Run this first on the 4-GPU AutoDL instance:

```bash
cd /root/autodl-tmp/projects/MiniOneRec
torchrun --standalone --nproc_per_node=4 scripts/ddp_sanity.py
```

Expected result:

- Four lines are printed, one for each rank.
- `WORLD_SIZE` is `4`.
- Each process reports a valid `LOCAL_RANK`.
- `all_reduce_rank_sum` is `6.0`, because `0 + 1 + 2 + 3 = 6`.

## 4-GPU Smoke Run

Use this to confirm the SFT path, tokenizer resize, DDP training, and checkpoint
saving all work before launching full training:

```bash
cd /root/autodl-tmp/projects/MiniOneRec

BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
bash scripts/run_sft_ddp_smoke_4gpu.sh
```

Defaults:

- category: `Industrial_and_Scientific`
- sample: `1024`
- epochs: `1`
- GPUs: `4`
- per-device train batch: `4`
- gradient accumulation: `2`
- effective batch: `4 * 4 * 2 = 32`

The script refuses to run if the output directory already exists. Set
`ALLOW_EXISTING_OUTPUT_DIR=1` only when intentionally resuming.

## Qwen2.5-1.5B-Instruct Full SFT

This is the preferred first real multi-GPU SFT experiment:

```bash
cd /root/autodl-tmp/projects/MiniOneRec

BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
OUTPUT_DIR=outputs/sft_ddp_Industrial_and_Scientific_Qwen25_15B_Instruct_4a100 \
bash scripts/run_sft_ddp_qwen25_15b_4gpu.sh
```

Defaults:

- category: `Industrial_and_Scientific`
- epochs: `10`
- learning rate: `2e-5`
- per-device train batch: `8`
- gradient accumulation: `8`
- effective batch: `8 * 4 * 8 = 256`
- gradient checkpointing: disabled by default

To switch to Office later:

```bash
CATEGORY=Office_Products \
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
OUTPUT_DIR=outputs/sft_ddp_Office_Products_Qwen25_15B_Instruct_4a100 \
bash scripts/run_sft_ddp_qwen25_15b_4gpu.sh
```

## Qwen2.5-3B-Instruct Optional Full SFT

Run this only after the 1.5B experiment is healthy:

```bash
cd /root/autodl-tmp/projects/MiniOneRec

BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
OUTPUT_DIR=outputs/sft_ddp_Industrial_and_Scientific_Qwen25_3B_Instruct_4a100 \
bash scripts/run_sft_ddp_qwen25_3b_4gpu.sh
```

Defaults:

- learning rate: `1e-5`
- per-device train batch: `4`
- gradient accumulation: `8`
- effective batch: `4 * 4 * 8 = 128`
- gradient checkpointing: enabled by default

## Running In The Background

For long AutoDL runs:

```bash
mkdir -p logs

nohup bash scripts/run_sft_ddp_qwen25_15b_4gpu.sh \
  > logs/sft_ddp_Industrial_Qwen25_15B_4gpu.log 2>&1 &

echo $! > logs/sft_ddp_Industrial_Qwen25_15B_4gpu.pid
```

Watch logs:

```bash
tail -f logs/sft_ddp_Industrial_Qwen25_15B_4gpu.log
```

Watch GPU usage:

```bash
watch -n 2 nvidia-smi
```

Check the process:

```bash
ps -fp "$(cat logs/sft_ddp_Industrial_Qwen25_15B_4gpu.pid)"
```

## Checkpoints

The training output directory contains normal Trainer checkpoints and:

```text
<output_dir>/final_checkpoint/
```

In DDP, `final_checkpoint` is saved only by rank 0 to avoid multiple ranks
writing the same files.

A healthy `final_checkpoint` should contain model and tokenizer files, such as:

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

Quick load check:

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

p = "outputs/sft_ddp_Industrial_and_Scientific_Qwen25_15B_Instruct_4a100/final_checkpoint"
tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(p, local_files_only=True)
print("LOAD_OK")
print("tokenizer_len:", len(tok))
print("vocab_size:", model.get_input_embeddings().weight.shape[0])
PY
```

## Evaluation

Evaluate the multi-GPU SFT checkpoint with the original single-process
`evaluate.py`:

```bash
cd /root/autodl-tmp/projects/MiniOneRec

category=Industrial_and_Scientific
test_file=$(ls ./data/Amazon/test/${category}*.csv | head -1)
train_file=$(ls ./data/Amazon/train/${category}*.csv | head -1)
info_file=$(ls ./data/Amazon/info/${category}*.txt | head -1)

mkdir -p results/eval_sft_ddp_Industrial_15B_beam50

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --base_model outputs/sft_ddp_Industrial_and_Scientific_Qwen25_15B_Instruct_4a100/final_checkpoint \
  --train_file "$train_file" \
  --info_file "$info_file" \
  --category "$category" \
  --test_data_path "$test_file" \
  --result_json_data results/eval_sft_ddp_Industrial_15B_beam50/predictions.json \
  --batch_size 4 \
  --num_beams 50 \
  --max_new_tokens 32
```

Then calculate enhanced metrics:

```bash
python calc_plus.py \
  --prediction-file results/eval_sft_ddp_Industrial_15B_beam50/predictions.json \
  --test-csv "$test_file" \
  --train-csv "$train_file" \
  --item2sid data/Amazon/sid_maps/${category}/item2sid_text.json \
  --sid2items data/Amazon/sid_maps/${category}/sid2items_text.json \
  --valid-sid-set data/Amazon/sid_maps/${category}/valid_sid_set_text.json \
  --output-dir results/calc_plus_sft_ddp_Industrial_15B_beam50 \
  --sid-version text
```

## Practical Notes

- Do not use `device_map="auto"` for DDP SFT.
- Do not start multiple DDP jobs in the same output directory.
- Keep `NCCL_IB_DISABLE=1` on AutoDL unless the instance has working IB/RoCE.
- For 4xA100 80GB, start with Qwen2.5-1.5B-Instruct before trying 3B.
- If 3B OOMs, lower `PER_DEVICE_TRAIN_BATCH_SIZE` before changing learning rate.
- If the run is interrupted and you intentionally resume, set:

```bash
ALLOW_EXISTING_OUTPUT_DIR=1 \
bash scripts/run_sft_ddp.sh ... \
  --resume-from-checkpoint outputs/.../checkpoint-xxxx
```
