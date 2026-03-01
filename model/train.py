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

    dataset = build_dataset(cfg.data.train, tokenizer_path)
    collate_fn = build_collate_fn(cfg.model.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.micro_batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

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

    os.makedirs(cfg.training.save_dir, exist_ok=True)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        step += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx:
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.training.log_interval == 0:
            print(f"step={step} loss={loss.item() * grad_accum_steps:.6f}")

        if step % cfg.training.save_interval == 0:
            save_path = os.path.join(cfg.training.save_dir, f"step-{step}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if step >= cfg.training.max_steps:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
