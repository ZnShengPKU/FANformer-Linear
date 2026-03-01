from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DatasetConfig, RunConfig, load_config
from .data import JsonlZstStreamingDataset, MultiFormatStreamingDataset, ParquetStreamingDataset
from .model import build_model, load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: DatasetConfig, tokenizer_path: str):
    dataset_name = cfg.dataset
    if dataset_name == "jsonl_zst_streaming":
        return JsonlZstStreamingDataset(
            cfg.paths,
            input_ids_field=cfg.input_ids_field,
            text_field=cfg.text_field,
            tokenizer_identifier=tokenizer_path,
            eos_token_id=cfg.eos_token_id,
            max_sequence_length=cfg.max_sequence_length,
            exclude_paths=cfg.exclude_paths,
        )
    if dataset_name == "parquet_streaming":
        return ParquetStreamingDataset(
            cfg.paths,
            input_ids_field=cfg.input_ids_field,
            text_field=cfg.text_field,
            tokenizer_identifier=tokenizer_path,
            eos_token_id=cfg.eos_token_id,
            max_sequence_length=cfg.max_sequence_length,
            exclude_paths=cfg.exclude_paths,
        )
    if dataset_name == "multi_format_streaming":
        return MultiFormatStreamingDataset(
            cfg.paths,
            input_ids_field=cfg.input_ids_field,
            text_field=cfg.text_field,
            tokenizer_identifier=tokenizer_path,
            eos_token_id=cfg.eos_token_id,
            max_sequence_length=cfg.max_sequence_length,
            exclude_paths=cfg.exclude_paths,
        )
    raise ValueError(f"Unknown dataset type: {dataset_name}")


def build_collate_fn(pad_token_id: int):
    def collate(batch):
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        max_len = max(t.size(0) for t in input_ids)
        batch_size = len(input_ids)
        padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, ids in enumerate(input_ids):
            length = ids.size(0)
            padded[i, :length] = ids
            attention_mask[i, :length] = 1
        labels = padded.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": padded, "attention_mask": attention_mask, "labels": labels}

    return collate


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


try:
    import swanlab as wandb_lib
except Exception:
    wandb_lib = None


def init_wandb(cfg: RunConfig):
    if cfg.wandb is None or wandb_lib is None:
        return None
    return wandb_lib.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.run_name,
        config={
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "model": cfg.model.__dict__,
            "training": cfg.training.__dict__,
        },
    )


def compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    return float(total_norm)


def run_eval(
    model: torch.nn.Module,
    eval_loaders: list[tuple[str, DataLoader]],
    device: torch.device,
    autocast_ctx,
    max_batches: int,
):
    model.eval()
    results = {}
    with torch.no_grad():
        for name, loader in eval_loaders:
            losses = []
            for idx, batch in enumerate(loader):
                if idx >= max_batches:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast_ctx:
                    outputs = model(**batch)
                    losses.append(outputs.loss.item())
            if losses:
                results[name] = float(sum(losses) / len(losses))
    model.train()
    return results


def train(cfg: RunConfig) -> None:
    set_seed(cfg.seed)
    tokenizer_path = cfg.tokenizer.path
    tokenizer = load_tokenizer(tokenizer_path)

    if cfg.model.pad_token_id is None:
        cfg.model.pad_token_id = tokenizer.pad_token_id
    if cfg.model.eos_token_id is None:
        cfg.model.eos_token_id = tokenizer.eos_token_id
    if cfg.data.train.eos_token_id is None:
        cfg.data.train.eos_token_id = tokenizer.eos_token_id

    model = build_model(cfg.model)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataset = build_dataset(cfg.data.train, tokenizer_path)
    collate_fn = build_collate_fn(cfg.model.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.micro_batch_size,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    eval_loaders = []
    if cfg.data.eval:
        for idx, eval_cfg in enumerate(cfg.data.eval):
            eval_dataset = build_dataset(eval_cfg, tokenizer_path)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=cfg.training.micro_batch_size,
                num_workers=4,
                prefetch_factor=2,
                collate_fn=collate_fn,
            )
            eval_loaders.append((f"eval_{idx}", eval_loader))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.betas),
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg.training.warmup_steps, cfg.training.max_steps)

    grad_accum_steps = max(1, cfg.training.global_batch_size // cfg.training.micro_batch_size)
    precision = cfg.training.precision.lower()
    use_amp = precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp and device.type != "cpu" else nullcontext()
    )

    wandb_run = init_wandb(cfg)

    global_step = 0
    step = 0
    last_grad_norm = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    # Check for FLA / Flash Attention
    try:
        import fla
        print("FLA (Flash Linear Attention) is installed.")
    except ImportError:
        print("WARNING: FLA (Flash Linear Attention) is NOT installed. Linear attention layers will be slow and memory intensive!")
        
    try:
        import flash_attn
        print("Flash Attention is installed.")
    except ImportError:
        print("WARNING: Flash Attention is NOT installed. Full attention layers might be slower.")

    for batch in loader:
        step += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx:
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0:
            global_step += 1
            last_grad_norm = compute_grad_norm(model)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.training.log_interval == 0:
            print(f"step={step} global_step={global_step} loss={loss.item() * grad_accum_steps:.6f}")

        if wandb_run is not None and step % cfg.wandb.log_interval == 0:
            total_loss = loss.item() * grad_accum_steps
            wandb_lib.log(
                {
                    "train/ce_loss": total_loss,
                    "train/total_loss": total_loss,
                    "train/grad_norm": last_grad_norm,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                    "micro_step": step,
                },
                step=global_step,
            )

        if eval_loaders and step % cfg.training.eval_interval == 0:
            eval_results = run_eval(
                model,
                eval_loaders,
                device,
                autocast_ctx,
                cfg.training.eval_num_batches,
            )
            if eval_results:
                mean_eval_loss = sum(eval_results.values()) / len(eval_results)
                if wandb_run is not None:
                    payload = {"eval/loss": mean_eval_loss, "step": global_step}
                    for name, value in eval_results.items():
                        payload[f"eval/{name}_loss"] = value
                    wandb_lib.log(payload, step=global_step)
                print(f"step={step} global_step={global_step} eval_loss={mean_eval_loss:.6f}")

        if global_step >= cfg.training.max_steps:
            print(f"Reached max_steps ({cfg.training.max_steps}). Stopping training.")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name")
    parser.add_argument("--device")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--micro_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--wandb_log_interval", type=int)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--eval_num_batches", type=int)
    parser.add_argument("--fan", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.fan:
        cfg.model.use_fan = True
    if args.run_name:
        cfg.run_name = args.run_name
    if args.device:
        cfg.training.device = args.device
    if args.batch_size:
        cfg.training.global_batch_size = args.batch_size
    if args.micro_batch_size:
        cfg.training.micro_batch_size = args.micro_batch_size
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
    if args.max_steps:
        cfg.training.max_steps = args.max_steps
    if args.eval_interval:
        cfg.training.eval_interval = args.eval_interval
    if args.eval_num_batches:
        cfg.training.eval_num_batches = args.eval_num_batches
    if args.wandb_log_interval:
        if cfg.wandb is None:
            raise ValueError("wandb config is required when setting wandb_log_interval")
        cfg.wandb.log_interval = args.wandb_log_interval
    train(cfg)


if __name__ == "__main__":
    main()
