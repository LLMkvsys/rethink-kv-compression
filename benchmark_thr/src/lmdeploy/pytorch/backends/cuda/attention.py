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


class TritonAttentionMetadata(AttentionMetadata):
    """triton attention metadata."""
    pass


class TritonAttentionImpl(AttentionImpl[TritonAttentionMetadata]):
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

        from lmdeploy.pytorch.kernels.cuda import (alibi_paged_attention_fwd,
                                                   fill_kv_cache,
                                                   paged_attention_fwd, 
                                                   paged_attention_fwd_attn)
        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = paged_attention_fwd
        self.paged_attention_fwd_attn = paged_attention_fwd_attn
        self.alibi_paged_attention_fwd = alibi_paged_attention_fwd

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
        attn_metadata: TritonAttentionMetadata,
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

        # fill kv cache
        self.fill_kv_cache(
            key,
            value,
            k_cache,
            v_cache,
            q_start_loc,
            q_seqlens,
            kv_seq_length=kv_seqlens,
            max_q_seq_length=max_q_seqlen,
            block_offsets=block_offsets,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            quant_bits=quant_bits,
        )

        if inplace:
            attn_output = query[..., :self.v_head_size]
        else:
            q_shape = query.shape
            o_shape = q_shape[:-1] + (self.v_head_size, )
            attn_output = query.new_empty(o_shape)

        # import pdb; pdb.set_trace()
        if not self.alibi:
            self.paged_attention_fwd(
                query,
                k_cache,
                v_cache,
                attn_output,
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
            
            # target_len = 515
            # if kv_seqlens[0] == target_len: 
            #     print('-- internal attention operation')
            #     print(query[0:1].sum() / query[0:1].numel())
            #     print(attn_output.size(), attn_output[0, 0, :10])
            #     # [-0.0005,  0.0032,  0.0220,  0.0126, -0.0008, -0.0010,  0.0085,  0.0033, 0.0014, -0.0004] # size 2 
            #     # import pdb; pdb.set_trace() 
            # if kv_seqlens[0] > target_len and q_seqlens[0] < 10: 
            #     exit(0)
                
            # if query.size(0) == 2: 
            #     new_attn_output = query.new_empty((1, )+o_shape[1:])
                
            #     startp, endp = 1, 2
            #     gt_attn_output = attn_output[startp:endp]
            #     sub_query = query[startp:endp]
            #     self.paged_attention_fwd(
            #         sub_query,
            #         k_cache,
            #         v_cache,
            #         new_attn_output,
            #         block_offsets[startp:endp],
            #         q_start_loc=q_start_loc[startp:endp],
            #         q_seqlens=q_seqlens[startp:endp],
            #         kv_seqlens=kv_seqlens[startp:endp],
            #         max_seqlen=1,
            #         window_size=self.sliding_window,
            #         sm_scale=self.scale,
            #         logit_softcapping=self.logit_softcapping,
            #     )
                # torch.allclose(new_attn_output, gt_attn_output)
                # import pdb; pdb.set_trace() 
        else:
            self.alibi_paged_attention_fwd(
                query,
                k_cache,
                v_cache,
                attn_output,
                block_offsets,
                b_start_loc=q_start_loc,
                b_seq_len=q_seqlens,
                b_kv_seq_len=kv_seqlens,
                max_input_len=max_q_seqlen,
                head_offset=self.alibi_head_offset,
                num_heads=self.alibi_num_heads,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_bits=quant_bits,
            )

        return attn_output

    def forward_att_scores(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        recent_window_size: torch.Tensor,
    ): 
        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        
        attn_cache = torch.zeros((query.size(0), recent_window_size[0].item(), query.size(1)), dtype=torch.float32).to(query.device)
        if not self.alibi:
            self.paged_attention_fwd_attn(
                q=query,
                k=k_cache,
                o=attn_cache,
                query_recent_window_size=recent_window_size,
                block_offsets=block_offsets,
                q_start_loc=q_start_loc,
                q_seqlens=q_seqlens,
                kv_seqlens=kv_seqlens,
                max_seqlen=max_q_seqlen,
                window_size=self.sliding_window,
                sm_scale=self.scale,
                logit_softcapping=self.logit_softcapping,
                shared_kv=False,
            )
        else:
            raise NotImplementedError

        return attn_cache.sum(1)

class TritonAttentionBuilder(AttentionBuilder[TritonAttentionMetadata]):
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
    ) -> TritonAttentionImpl:
        """build."""
        return TritonAttentionImpl(num_heads,
                                   head_size,
                                   scale=scale,
                                   num_kv_heads=num_kv_heads,
                                   v_head_size=v_head_size,
                                   alibi=alibi,
                                   sliding_window=sliding_window,
                                   logical_softcapping=logical_softcapping,
                                   **kwargs)
