from __future__ import annotations

from typing import List

from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from .config import ModelConfig


def load_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(config: ModelConfig) -> Qwen3_5ForCausalLM:
    layer_types: List[str]
    if isinstance(config.layer_types, str):
        layer_types = [config.layer_types for _ in range(config.num_hidden_layers)]
    else:
        layer_types = config.layer_types

    rope_parameters = {
        "rope_type": "default",
        "rope_theta": config.rope_theta,
        "mrope_section": [11, 11, 10],
    }

    text_config = Qwen3_5TextConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_word_embeddings,
        rope_parameters=rope_parameters,
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
        use_cache=config.use_cache,
        layer_types=layer_types,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        use_fan=config.use_fan,
    )
    return Qwen3_5ForCausalLM(text_config)
