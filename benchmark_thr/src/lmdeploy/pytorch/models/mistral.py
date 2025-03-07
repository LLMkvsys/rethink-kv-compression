# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, AttentionQuantFull, RMSNorm, RopeType,
                                 SiluAndMul, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_merged_colwise_linear,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from lmdeploy.utils import calculate_time, calculate_time_adaptive_input
from .mistral_sparse import init_StreamingLLM, init_H2O
import numpy as np

class MistralAttention(nn.Module):
    """Rewrite module of MistralAttention."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 quant_policy: str = "None",
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config 
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        self.num_key_value_groups = num_heads // num_key_value_heads 

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        if quant_policy in ["KIVI"]: 
            # attention
            self.attn_fwd = AttentionQuantFull(
                num_heads,
                head_dim,
                num_kv_heads=num_key_value_heads,
                v_head_size=head_dim,
            )
        else:
            # attention
            self.attn_fwd = Attention(
                num_heads,
                head_dim,
                num_kv_heads=num_key_value_heads,
                v_head_size=head_dim,
                sliding_window=config.sliding_window,
            )

        # o_proj
        self.o_proj = build_rowwise_linear(num_heads * head_dim,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)
        self.compress_method =quant_policy

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     attn_metadata: Any = None,
    # ):
    #     """Rewrite of LlamaAttention.forward."""
    #     # qkv proj
    #     qkv_states = self.qkv_proj(hidden_states)
    #     # (-1, heads, head_dim)
    #     qkv_states = qkv_states.flatten(0, -2)
    #     query_states, key_states, value_states = self.qkv_proj.split_qkv(
    #         qkv_states)

    #     # apply rotary embedding
    #     cos, sin = rotary_pos_emb
    #     query_states, key_states = self.apply_rotary_pos_emb(
    #         query_states,
    #         key_states,
    #         cos,
    #         sin,
    #         inplace=True,
    #     )

    #     # attention
    #     attn_output = self.attn_fwd(
    #         query_states,
    #         key_states,
    #         value_states,
    #         past_key_value[0],
    #         past_key_value[1],
    #         attn_metadata,
    #         k_scales_zeros=None
    #         if len(past_key_value) == 2 else past_key_value[2],
    #         v_scales_zeros=None
    #         if len(past_key_value) == 2 else past_key_value[3],
    #         quant_bits=0 if len(past_key_value) == 2 else past_key_value[4],
    #         inplace=True,
    #     )
    #     attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

    #     # o proj
    #     attn_output = self.o_proj(attn_output)
    #     return attn_output

    # FIXME: XYZ, enable this to measure the att layer time
    # @calculate_time_adaptive_input(show=True, min_cost_ms=0.1, signature='mistral-att-forward')
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        if self.compress_method in ["None", "KIVI", "HEAD", "GEAR"]: 
            pass 
        elif self.compress_method == 'StreamingLLM':
            init_StreamingLLM(self)
        elif self.compress_method == "H2O":
            init_H2O(self)
        else: 
            raise NotImplementedError
        # import pdb; pdb.set_trace() 
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(
            qkv_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )
        
        # import pdb; pdb.set_trace() 
        # attention
        if self.compress_method in ["None", "StreamingLLM", "H2O", "HEAD", "GEAR"]:
            attn_output = self.attn_fwd(
                query_states,
                key_states,
                value_states,
                past_key_value[0],
                past_key_value[1],
                attn_metadata,
                k_scales_zeros=None
                if len(past_key_value) == 2 else past_key_value[2],
                v_scales_zeros=None
                if len(past_key_value) == 2 else past_key_value[3],
                quant_bits=0 if len(past_key_value) == 2 else past_key_value[-1],
                inplace= False if self.compress_method in ["H2O"] else True,
            )
        elif self.compress_method in ["KIVI"]:
            assert past_key_value[4].dtype == torch.bfloat16 and past_key_value[5].dtype == torch.bfloat16, "The data type should be bfloat16"
            attn_output = self.attn_fwd(
                query_states,
                key_states,
                value_states,
                past_key_value[0],
                past_key_value[1],
                past_key_value[4],
                past_key_value[5],
                attn_metadata,
                k_scales_zeros=None
                if len(past_key_value) == 2 else past_key_value[2],
                v_scales_zeros=None
                if len(past_key_value) == 2 else past_key_value[3],
                quant_bits=0 if len(past_key_value) == 2 else past_key_value[-1],
                inplace= False if self.compress_method in ["H2O"] else True,
            )
        else: 
            raise NotImplementedError
        
        if self.layer_idx == 0: 
            pass
            
        if self.compress_method in ["StreamingLLM", "H2O"] and attn_metadata.is_decoding == False:
            # seq, block, head, hidden_dim
            num_key_value_heads = past_key_value[0].size(2)
            if self.compress_method in ["H2O"]: 
                attn_cache = self.attn_fwd.impl.forward_att_scores(
                    query=query_states, 
                    k_cache=past_key_value[0],
                    attn_metadata=attn_metadata,
                    recent_window_size=torch.LongTensor([self.config.max_capacity_prompt-self.config.window_size for _ in range(len(attn_metadata.q_seqlens))]).to(query_states.device),
                )
                attn_cache = attn_cache.view(-1, self.num_key_value_groups, num_key_value_heads).sum(-2)
            elif self.compress_method in ["StreamingLLM"]: 
                attn_cache = None
            else: 
                raise NotImplementedError 
            
            _, _, attn_metadata_copy = self.kv_cluster.update_kv(past_key_value[0], query_states, past_key_value[1], 
                                                                attn_metadata, self.num_key_value_groups, self.layer_idx, attn_cache=attn_cache)
            # import pdb; pdb.set_trace() 
            # print("max_capacity_prompt is ", self.kv_cluster.max_capacity_prompt)
        # print(attn_metadata.q_seqlens)
        elif self.compress_method in ["StreamingLLM", "H2O"] and attn_metadata.is_decoding == False:
            num_key_value_heads = past_key_value[0].size(2)
            attn_metadata_copy = None 
        else: 
            attn_metadata_copy = None 
        
        if self.layer_idx == 0: 
            # print(f"q_seqlens is {attn_metadata.q_seqlens}, kv_lens {attn_metadata.kv_seqlens}")
            self.attn_metadata_copy = attn_metadata_copy 
            # if attn_metadata_copy is not None: 
            #     print("attn_metadata_copy.q_seqlens is ", attn_metadata_copy.q_seqlens)
            # else: 
            #     print("attn_metadata.q_seqlens is ", attn_metadata.q_seqlens)
            # print("attn_metadata_copy is ", attn_metadata_copy, attn_metadata)
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class MistralMLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(config.intermediate_size,
                                              config.hidden_size,
                                              bias=False,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class MistralDecoderLayer(nn.Module):
    """decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 extended_config: Any = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        # self.self_attn = MistralAttention(config, dtype=dtype, device=device)
        self.self_attn = MistralAttention(config, layer_idx=layer_idx, dtype=dtype, device=device, quant_policy=extended_config["quant_policy"])

        # builf MLP
        self.mlp = MistralMLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class MistralModel(nn.Module):
    """model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 extended_config: Any = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        quantization_config = getattr(config, 'quantization_config', None)

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            MistralDecoderLayer(config, layer_idx, dtype=dtype, device=device, extended_config=extended_config)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=quantization_config,
                            dtype=dtype,
                            device=device)

        # build rotary embedding
        emb_type = RopeType.LinearScaling
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            emb_type=emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embed_tokens


class MistralForCausalLM(nn.Module):
    """ModelForCausalLM."""

    support_cuda_graph = True

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 extended_config: Any = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build model
        self.model = MistralModel(config, dtype=dtype, device=device,  extended_config=extended_config)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    @calculate_time_adaptive_input(show=True, min_cost_ms=0.1, signature='mistral-forward')
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.model.layers[0].self_attn.attn_metadata_copy is not None: 
            block_size = past_key_values[0][0].size(1)
            attn_metadata_copy = self.model.layers[0].self_attn.attn_metadata_copy
            for idx, msg in enumerate(attn_metadata.messages): 
                old_num_blocks =  (msg._num_history_ids + msg._num_token_ids - 1) // block_size + 1
                if attn_metadata_copy.is_decoding: 
                    msg._num_ignore_token_ids += (msg._num_history_ids - attn_metadata_copy.kv_seqlens[idx].item())
                    msg._num_history_ids = attn_metadata_copy.kv_seqlens[idx].item() 
                else:
                    msg._num_ignore_token_ids += (msg._num_token_ids - attn_metadata_copy.kv_seqlens[idx].item())
                    msg._num_token_ids = attn_metadata_copy.kv_seqlens[idx].item() 
                
                if attn_metadata_copy.is_decoding:
                    pass 
                else: 
                    msg.history_cache.trim(0, msg._num_token_ids)
                    # print(len(msg.history_cache))
                    # import pdb; pdb.set_trace() 
                
                num_blocks = (msg._num_history_ids + msg._num_token_ids - 1) // block_size + 1
                
                assert old_num_blocks >= num_blocks
                if old_num_blocks > num_blocks: 
                    msg.evict_blocks = np.array(msg.logical_blocks._blocks[num_blocks:old_num_blocks].tolist())
                msg.logical_blocks._blocks[num_blocks:] = 0 
                msg.logical_blocks._num_real = num_blocks
        return logits

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)


class LlavaMistralForCausalLM(MistralForCausalLM):
    """llava forcausallm."""

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""

        new_weights = dict()
        for key, val in weights:
            if key.startswith('model.vision_tower'):
                continue
            if key.startswith('model.mm_projector'):
                continue
            if key.startswith('model.image_newline'):
                continue
            new_weights[key] = val

        super().load_weights(new_weights.items())
