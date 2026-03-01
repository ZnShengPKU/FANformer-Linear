from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: int = 10000
    layer_types: Union[str, List[str]] = "full_attention"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_cache: bool = False
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class TokenizerConfig:
    path: str


@dataclass
class DatasetConfig:
    dataset: str
    paths: List[str]
    exclude_paths: Optional[List[str]] = None
    text_field: Optional[str] = None
    input_ids_field: str = "input_ids"
    max_sequence_length: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class DataConfig:
    train: DatasetConfig
    eval: Optional[List[DatasetConfig]] = None


@dataclass
class TrainingConfig:
    max_steps: int
    global_batch_size: int
    micro_batch_size: int
    learning_rate: float
    weight_decay: float
    betas: List[float]
    warmup_steps: int
    log_interval: int
    eval_interval: int
    eval_num_batches: int
    gradient_checkpointing: bool = True
    device: str = "cuda"
    precision: str = "bf16"


@dataclass
class WandbConfig:
    project: str
    entity: Optional[str] = None
    log_interval: int = 10


@dataclass
class RunConfig:
    run_name: str
    seed: int
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    training: TrainingConfig
    wandb: Optional[WandbConfig] = None


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    model = ModelConfig(**raw["model"])
    tokenizer = TokenizerConfig(**raw["tokenizer"])
    train_data = DatasetConfig(**raw["data"]["train"])
    eval_data_raw = raw["data"].get("eval")
    eval_data = [DatasetConfig(**item) for item in eval_data_raw] if eval_data_raw else None
    data = DataConfig(train=train_data, eval=eval_data)
    training = TrainingConfig(**raw["training"])
    wandb_raw = raw.get("wandb")
    wandb_cfg = WandbConfig(**wandb_raw) if wandb_raw else None
    return RunConfig(
        run_name=raw["run_name"],
        seed=raw["seed"],
        model=model,
        tokenizer=tokenizer,
        data=data,
        training=training,
        wandb=wandb_cfg,
    )
