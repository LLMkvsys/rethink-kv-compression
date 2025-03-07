# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Type

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Output model local path, do not set manually"}
    )
    quant_method: Optional[str] = field(
        default=None,
        metadata={"help": "The quantization approach. (currently support kivi, gear)"}
    )
    eviction_method: Optional[str] = field(
        default=None,
        metadata={"help": "The eviction approach. (currently support streamingllm, h2o)"}
    )

    """
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    load_quant: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a quantized model"},
    )
    w_bit: Optional[int] = field(
        default=4,
        metadata={"help": "The model weight bit width."},
    )
    lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA"},
    )
    lora_mode: Optional[str] = field(
        default="q",
        metadata={"help": "LoRA mode"},
    )
    lora_r: Optional[int] = field(
        default=1,
        metadata={"help": "LoRA r"},
    )
    lora_alpha: Optional[float] = field(
        default=1.,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: Optional[float] = field(
        default=0.,
        metadata={"help": "LoRA dropout"},
    )
    """
    

# quant arguments of KIVI algorithm
@dataclass
class KIVIArguments:
    k_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV_cache quantization bits."},
    )
    v_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV_cache quantization bits."},
    )
    k_quant_dim: Optional[str] = field(
        default='token',
        metadata={"help": "KV_cache quantization bits."},
    )
    v_quant_dim: Optional[str] = field(
        default='token',
        metadata={"help": "KV_cache quantization bits."},
    )
    group_size: Optional[int] = field(
        default=128,
        metadata={"help": "KV_cache quantization group size."},
    )
    residual_length: Optional[int] = field(
        default=128,
        metadata={"help": "KV_cache residual length."},
    )


# quant arguments of GEAR algorithm
@dataclass
class GEARArguments:
    compress_mode: Optional[str] = field(
        default='gear_batch',
        metadata={"help": "batchwise-GEAR"}
    )
    quantize_bit: Optional[int] = field(
        default=4,
        metadata={"help": "outlier quantization bit"}
    )
    left: Optional[float] = field(
        default=0.02,
        metadata={"help": "outlier extraction rate"}
    )
    rank: Optional[float] = field(
        default=0.02,
        metadata={"help": "setting rank for Key and value cache quantization error"}
    )
    rankv: Optional[float] = field(
        default=0.0,
        metadata={"help": "setting rank for Key and value cache quantization error"}
    )
    loop: Optional[int] = field(
        default=3,
        metadata={"help": "constant for power iteration(an efficient SVD solver)"}
    )
    stream: Optional[bool] = field(
        default=True,
        metadata={"help": "streaming-gear set to true to perform better efficiency"}
    )
    streaming_gap: Optional[int] = field(
        default=20,
        metadata={"help": "re-compress every 20 iteration"}
    )


# arguments for sparsity-based kv compression methods, pyramidkv, snapkv, h2o, streamingllm
@dataclass
class SparseKVArguments:
    max_capacity_prompts: Optional[int] = field(
        default=512,
        metadata={"help": "max prompt kv cache budget size"}
    )
    window_sizes: Optional[int] = field(
        default=8,
        metadata={"help": "recent kv cache window size"}
    )
    kernel_sizes: Optional[int] = field(
        default=7,
        metadata={"help": "pooling kernel size for merging kv cache"}
    )
    pooling: Optional[str] = field(
        default="maxpool",
        metadata={"help": "pooling method"}
    )
    attn_impl: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "attention implementations, support flash_attention_2, sdpa, eager"}
    )


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default='c4',
        metadata={"help": "The dataset used for fine-tuning the model."},
    )
    eval_tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The batch size."},
    )
    num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "The number of fewshot examples."},
    )
    output_path: Optional[str] = field(
        default='./outputs',
        metadata={"help": "The output path."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )
    use_our_imp: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )
    add_prefix: Optional[str] = field(
        default='',
        metadata={"help": "Added prefix."},
    )
    add_prefix_ident: Optional[str] = field(
        default='',
        metadata={"help": "Added prefix."},
    )


@dataclass
class Coarse2fineDataArguments:
    data_path: Optional[str] = field(
        default="data/coarse2fine",
        metadata={"help": "If specified, we will load the data to generate the predictions."},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
    remove_row_delimiter: Optional[bool] = field(
        default=False,
        metadata={"help": "If given, we will remove row delimiter repeated four times."},
    )
    ignore_eos: Optional[bool] = field(
        default=False,
        metadata={"help": "If given, we will continue generating tokens after the EOS token is generated."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="./outputs")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    num_train_epochs: Optional[int] = field(default=1)
    n_train_samples: Optional[int] = field(default=None)
    n_eval_samples: Optional[int] = field(default=None)
    qat: Optional[bool] = field(default=False)
    exp_name: Optional[str] = field(default="test")


def process_args() -> Dict[str, Any]:
    arg_classes: Dict[str, Type[Any]] = {
        "model_args": ModelArguments,
        "data_args": DataArguments,
        "coarse2fine_data_args": Coarse2fineDataArguments,
        "training_args": TrainingArguments,
        "kivi_args": KIVIArguments,
        "gear_args": GEARArguments,
        "sparsekv_args": SparseKVArguments,
    }

    parser = transformers.HfArgumentParser(list(arg_classes.values()))
    parsed_args = parser.parse_args_into_dataclasses()

    # extract parsed arguments into a dictionary
    parsed_args_dict = {name: arg for name, arg in zip(arg_classes.keys(), parsed_args)}

    # ensure the output directory exists
    os.makedirs(parsed_args_dict["training_args"].output_dir, exist_ok=True)

    # create the output model local path
    parsed_args_dict["model_args"].output_model_local_path = os.path.join(
        parsed_args_dict["training_args"].output_dir, "models", str(parsed_args_dict["model_args"].model_name_or_path)
    )

    return parsed_args_dict