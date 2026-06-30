import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset


class TokenExtender:
    def __init__(self, data_path=None, dataset=None, index_file=".index.json", index_path=None):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.index_path = index_path
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        if self.index_path:
            path = self.index_path
        else:
            path = os.path.join(self.data_path, self.dataset + self.index_file)
        with open(path, 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens

    def get_token_stats(self):
        stats = {}
        for token in self.get_new_tokens():
            if token.startswith("<") and "_" in token:
                level = token[1:token.index("_")]
            else:
                level = "other"
            stats[level] = stats.get(level, 0) + 1
        return dict(sorted(stats.items()))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    per_device_train_batch_size: Optional[int] = None,
    per_device_eval_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    bf16: bool = True,
    gradient_checkpointing: bool = False,
    ddp_find_unused_parameters: bool = False,
    save_on_each_node: bool = False,
    save_total_limit: int = 1,
    save_during_training: bool = True,
    load_best_model_at_end: bool = True,
    early_stopping_patience: int = 3,
    sft_task_mode: str = "all",
):
    set_seed(seed)
    if wandb_project:
        os.environ['WANDB_PROJECT'] = wandb_project
    else:
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_DISABLED", "true")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    is_main_process = rank == 0

    train_micro_batch_size = (
        per_device_train_batch_size
        if per_device_train_batch_size is not None
        else micro_batch_size
    )
    eval_micro_batch_size = (
        per_device_eval_batch_size
        if per_device_eval_batch_size is not None
        else train_micro_batch_size
    )
    if train_micro_batch_size <= 0:
        raise ValueError("per-device train batch size must be > 0")
    if eval_micro_batch_size <= 0:
        raise ValueError("per-device eval batch size must be > 0")

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = max(1, batch_size // train_micro_batch_size)
        if ddp:
            gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)
    else:
        gradient_accumulation_steps = int(gradient_accumulation_steps)
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")

    if is_main_process:
        effective_batch_size = train_micro_batch_size * world_size * gradient_accumulation_steps
        print(
            "SFT launch config: "
            f"ddp={ddp}, world_size={world_size}, local_rank={local_rank}, "
            f"per_device_train_batch_size={train_micro_batch_size}, "
            f"per_device_eval_batch_size={eval_micro_batch_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"effective_batch_size={effective_batch_size}, bf16={bf16}, "
            f"gradient_checkpointing={gradient_checkpointing}"
        )

    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    model_kwargs = {}
    if bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    original_vocab_size = len(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    new_tokens = []
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(index_path=sid_index_path)
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            print(f"SID token stats by level: {token_extender.get_token_stats()}")
            tokenizer.add_tokens(new_tokens)
            try:
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")

    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if not save_during_training and load_best_model_at_end:
        raise ValueError("load_best_model_at_end=True requires save_during_training=True")
        
    train_datasets = []
    # train_data1 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data1)
    sft_task_mode = sft_task_mode.lower().replace("-", "_")
    if sft_task_mode in {"all", "mixed"}:
        train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
        train_datasets.append(train_data2)
        train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
        train_datasets.append(train_data3)
    elif sft_task_mode in {"sid_only", "rec_only"}:
        if is_main_process:
            print("SFT task mode: sid_only; using SidSFTDataset only.")
    else:
        raise ValueError("sft_task_mode must be one of: all, mixed, sid_only, rec_only")
    # train_data4 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data4)
    # train_data5 = TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data5)
    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=train_micro_batch_size,
            per_device_eval_batch_size=eval_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps" if save_during_training else "no",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            ddp_find_unused_parameters=ddp_find_unused_parameters if ddp else None,
            save_on_each_node=save_on_each_node,
            log_on_each_node=False,
            group_by_length=group_by_length,
            report_to="none",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = (
            [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            if load_best_model_at_end and early_stopping_patience and early_stopping_patience > 0
            else []
        ),
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if save_during_training:
        trainer.save_model(output_dir)
    trainer.accelerator.wait_for_everyone()

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    if trainer.is_world_process_zero():
        os.makedirs(final_output_dir, exist_ok=True)
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped_model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        print(f"Saved final checkpoint to {final_output_dir}")
    trainer.accelerator.wait_for_everyone()



if __name__ == "__main__":
    fire.Fire(train)
