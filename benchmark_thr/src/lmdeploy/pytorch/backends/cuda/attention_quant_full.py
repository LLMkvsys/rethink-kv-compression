# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
import torch.distributed as dist

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata


def get_world_rank():
    """get current world size and rank."""
    world_size = 1
    rank = 0

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    return world_size, rank


class TritonAttentionQuantFullMetadata(AttentionMetadata):
    """triton attention metadata."""
    pass


class TritonAttentionQuantFullImpl(AttentionImpl[TritonAttentionQuantFullMetadata]):
    """triton attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = None,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            **kwargs,
        )

        from lmdeploy.pytorch.kernels.cuda import (fill_kv_cache_quant_full,
                                                   paged_attention_quant_full_fwd)
        self.fill_kv_cache_quant_full = fill_kv_cache_quant_full
        self.paged_attention_quant_full_fwd = paged_attention_quant_full_fwd
        # self.alibi_paged_attention_fwd = alibi_paged_attention_fwd

        # for alibi attention
        world_size, rank = get_world_rank()
        self.alibi_head_offset = self.num_heads * rank
        self.alibi_num_heads = self.num_heads * world_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_full_cache: torch.Tensor,
        v_full_cache: torch.Tensor,
        attn_metadata: TritonAttentionQuantFullMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        quant_bits: Literal[0, 2, 4, 8] = 0,
        inplace: bool = True,
    ) -> torch.Tensor:
        """forward."""

        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        # import pdb; pdb.set_trace() 
        # print("run quant policy")
        # fill kv cache
        if not attn_metadata.is_decoding: 
            self.fill_kv_cache_quant_full(
                key,
                value,
                k_cache,
                v_cache,
                k_full_cache, 
                v_full_cache,
                q_start_loc,
                q_seqlens,
                kv_seq_length=kv_seqlens,
                max_q_seq_length=max_q_seqlen,
                block_offsets=block_offsets,
                full_block_offsets=attn_metadata.block_full_offsets,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_bits=quant_bits,
            )
        # import pdb; pdb.set_trace() 
        if inplace:
            attn_output = query[..., :self.v_head_size]
        else:
            q_shape = query.shape
            o_shape = q_shape[:-1] + (self.v_head_size, )
            attn_output = query.new_empty(o_shape)

        if not self.alibi:
            self.paged_attention_quant_full_fwd(
                query,
                k_cache,
                v_cache,
                attn_output,
                k_full_cache, 
                v_full_cache, 
                attn_metadata.block_full_offsets,
                block_offsets,
                q_start_loc=q_start_loc,
                q_seqlens=q_seqlens,
                kv_seqlens=kv_seqlens,
                max_seqlen=max_q_seqlen,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_bits=quant_bits,
                window_size=self.sliding_window,
                sm_scale=self.scale,
                logit_softcapping=self.logit_softcapping,
            )
        else:
            raise NotImplementedError

        return attn_output

class TritonAttentionQuantFullBuilder(AttentionBuilder[TritonAttentionQuantFullMetadata]):
    """triton attention builder."""

    @staticmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logical_softcapping: float = None,
        **kwargs,
    ) -> TritonAttentionQuantFullImpl:
        """build."""
        return TritonAttentionQuantFullImpl(num_heads,
                                   head_size,
                                   scale=scale,
                                   num_kv_heads=num_kv_heads,
                                   v_head_size=v_head_size,
                                   alibi=alibi,
                                   sliding_window=sliding_window,
                                   logical_softcapping=logical_softcapping,
                                   **kwargs)
