# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
from typing import Literal

import torch
import triton
import triton.language as tl
from packaging import version
from torch import Tensor

from lmdeploy.utils import get_logger

from .triton_utils import get_kernel_meta, wrap_jit_func

logger = get_logger('lmdeploy')

TRITON_VERSION = version.parse(triton.__version__)

assert TRITON_VERSION >= version.parse('2.1.0')

if TRITON_VERSION >= version.parse('3.0.0'):

    @triton.jit
    def tanh(x):
        """tanh."""
        return 2 * tl.sigmoid(2 * x) - 1

    fast_expf = tl.math.exp
    fast_dividef = tl.math.fdiv
else:
    tanh = tl.math.tanh
    fast_expf = tl.math.fast_expf
    fast_dividef = tl.math.fast_dividef


@triton.autotune(configs=[
    triton.Config({}, num_stages=2, num_warps=16),
    triton.Config({}, num_stages=2, num_warps=8),
    triton.Config({}, num_stages=2, num_warps=4),
],
                 key=['BLOCK_H', 'BLOCK_N', 'BLOCK_DMODEL', 'BLOCK_DV'])
@wrap_jit_func(type_hint=dict(
    Q=torch.Tensor,
    K=torch.Tensor,
    V=torch.Tensor,
    sm_scale=float,
    KV_seqlens=torch.Tensor,
    Block_offsets=torch.Tensor,
    Acc_out=torch.Tensor,
    stride_qbs=int,
    stride_qh=int,
    stride_qd=int,
    stride_kbs=int,
    stride_kh=int,
    stride_kd=int,
    stride_vbs=int,
    stride_vh=int,
    stride_vd=int,
    stride_ok=int,
    stride_obs=int,
    stride_oh=int,
    stride_od=int,
    stride_boffb=int,
    kv_group_num=torch.int32,
    block_per_cta=torch.int32,
    window_size=torch.int32,
    head_size=torch.int32,
    head_size_v=torch.int32,
    shared_kv=bool,
    BLOCK_DMODEL=torch.int32,
    BLOCK_DV=torch.int32,
    BLOCK_N=torch.int32,
    BLOCK_H=torch.int32,
    BLOCK_DMODEL1=torch.int32,
))
@triton.jit
def _fwd_grouped_split_kernel(
    Q,
    K,
    V,
    sm_scale,
    KV_seqlens,
    Block_offsets,
    Acc_out,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ok: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_boffb,
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    num_heads_q: tl.constexpr,
    shared_kv: tl.constexpr,
    logit_softcapping: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
):
    """first step kernel of split k attention."""
    cur_batch = tl.program_id(2)
    cur_kv_head = tl.program_id(0)
    split_k_id = tl.program_id(1)

    if BLOCK_H < kv_group_num:
        HEAD_PER_CTA: tl.constexpr = BLOCK_H
    else:
        HEAD_PER_CTA: tl.constexpr = kv_group_num
    cur_head = cur_kv_head * HEAD_PER_CTA + tl.arange(0, BLOCK_H)
    mask_h = cur_head < cur_kv_head * HEAD_PER_CTA + HEAD_PER_CTA
    mask_h = mask_h & (cur_head < num_heads_q)

    q_seqlen = 1
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    if kv_seqlen <= 0:
        return
    history_len = kv_seqlen - q_seqlen

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_size
    offs_d = offs_d % head_size
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < head_size_v
    offs_dv = offs_dv % head_size_v
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
             offs_n[None, :] * stride_kbs)
    off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd +
             offs_n[:, None] * stride_vbs)

    off_q = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
             offs_d[None, :] * stride_qd)
    q = tl.load(Q + off_q, mask=mask_h[:, None] & mask_d[None, :], other=0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    if BLOCK_DMODEL1 != 0:
        offs_d1 = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL1)
        mask_d1 = offs_d1 < head_size
        offs_d1 = offs_d1 % head_size
        off_q1 = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
                  offs_d1[None, :] * stride_qd)
        q1 = tl.load(Q + off_q1,
                     mask=mask_h[:, None] & mask_d1[None, :],
                     other=0)
        off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                  offs_n[None, :] * stride_kbs)
        k1_ptrs = K + off_k1

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    num_total_blocks = tl.cdiv(kv_seqlen, BLOCK_N)
    BLOCK_PER_CTA = tl.cdiv(num_total_blocks, SPLIT_K)
    kv_len_per_prog = BLOCK_PER_CTA * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, kv_seqlen)

    # load block offset
    # dirty
    start_block_id = loop_start // BLOCK_N
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size,
                                    loop_start) // BLOCK_N
        kv_min_loc = tl.maximum(history_len - window_size, 0)

    loop_start = start_block_id * BLOCK_N
    block_offset_ptrs += start_block_id
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_offset = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(k_ptrs + b_offset * stride_kp)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(k1_ptrs + b_offset * stride_kp)

        if shared_kv:
            v = tl.trans(k)
        else:
            v = tl.load(v_ptrs + b_offset * stride_vp)

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL1 != 0:
            qk += tl.dot(q1, k1)
        qk *= sm_scale
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping
        # NOTE: inf - inf = nan, and nan will leads to error
        if start_n + BLOCK_N > history_len or window_size > 0:
            qk_mask = history_len >= (start_n + offs_n)
            if window_size > 0:
                qk_mask = qk_mask and ((start_n + offs_n) >= kv_min_loc)
            qk = tl.where(
                qk_mask[None, :],
                qk,
                -float('inf'),
            )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = fast_expf(qk - m_i_new[:, None])
        alpha = fast_expf(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)

        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    off_acc = (cur_batch * stride_obs + split_k_id * stride_ok +
               cur_head[:, None] * stride_oh + offs_dv[None, :] * stride_od)
    tl.store(Acc_out + off_acc, acc, mask=mask_h[:, None] & mask_dv[None, :])

    off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                cur_head * stride_oh + head_size_v)
    tl.store(Acc_out + off_meta, m_i, mask=mask_h)
    tl.store(Acc_out + off_meta + 1, l_i, mask=mask_h)


@triton.jit
def _unpack_kv_int4(k):
    """k with shape [head_num, head_dim]"""
    # return tl.zeros((2, k.shape[0], k.shape[1]), dtype=tl.uint8)
    return tl.full((k.shape[0], 2, k.shape[1]), 1, dtype=tl.uint8)
    # k1 = k - (k >> 4) * 16
    # k2 = k >> 4
    k1 = k
    k2 = k 
    k = tl.zeros((k.shape[0], 2, k.shape[1]), dtype=tl.uint8)
    k1 = tl.expand_dims(k1, 1)
    k2 = tl.expand_dims(k2, 1)
    k1 = tl.broadcast_to(k1, k.shape)
    k2 = tl.broadcast_to(k2, k.shape)
    k = tl.where(tl.arange(0, 2)[None, :, None] == 1, k, k1)
    k = tl.where(tl.arange(0, 2)[None, :, None] == 0, k, k2)
    return k

# @triton.jit
# def _unpack_kv_int4_transposed(k):
#     """k with shape [head_dim, head_num]"""
#     k1 = tl.zeros((2, k.shape[0], k.shape[1]), dtype=tl.uint8)
#     k1[0, ...] = k 
#     k1[1, ...] = k
#     return k1


# FIXME: it is not correct
@triton.jit
def _unpack_kv_int2_transposed(k):
    """k with shape [head_dim, head_num]"""
    k1 = k - (k >> 2) * 4
    k2 = (k>>2) - (k>>4)*4
    k3 = (k>>4) - (k>>6)*4
    k4 = k>>6
    
    k = tl.zeros((4, k.shape[0], k.shape[1]), dtype=tl.uint8)
    k1 = tl.expand_dims(k1, 0)
    k2 = tl.expand_dims(k2, 0)
    k3 = tl.expand_dims(k3, 0)
    k4 = tl.expand_dims(k4, 0)
    
    k1 = tl.broadcast_to(k1, k.shape)
    k2 = tl.broadcast_to(k2, k.shape)
    k3 = tl.broadcast_to(k3, k.shape)
    k4 = tl.broadcast_to(k4, k.shape)
    
    k = tl.where(tl.arange(0, 4)[:, None, None] == 3, k, k1)
    k = tl.where(tl.arange(0, 4)[:, None, None] == 2, k, k2)
    k = tl.where(tl.arange(0, 4)[:, None, None] == 1, k, k3)
    k = tl.where(tl.arange(0, 4)[:, None, None] == 0, k, k4)
    return k


@triton.jit
def _unpack_kv_int2(k):
    """k with shape [head_num, head_dim]"""
    k1 = k - (k >> 2) * 4
    k2 = (k>>2) - (k>>4)*4
    k3 = (k>>4) - (k>>6)*4
    k4 = k>>6
    
    k = tl.zeros((k.shape[0], 4, k.shape[1]), dtype=tl.uint8)
    k1 = tl.expand_dims(k1, 1)
    k2 = tl.expand_dims(k2, 1)
    k3 = tl.expand_dims(k3, 1)
    k4 = tl.expand_dims(k4, 1)
    
    k1 = tl.broadcast_to(k1, k.shape)
    k2 = tl.broadcast_to(k2, k.shape)
    k3 = tl.broadcast_to(k3, k.shape)
    k4 = tl.broadcast_to(k4, k.shape)
    
    k = tl.where(tl.arange(0, 4)[None, :, None] == 3, k, k1)
    k = tl.where(tl.arange(0, 4)[None, :, None] == 2, k, k2)
    k = tl.where(tl.arange(0, 4)[None, :, None] == 1, k, k3)
    k = tl.where(tl.arange(0, 4)[None, :, None] == 0, k, k4)
    return k




@triton.jit
def _unpack_kv_int4_transposed(k):
    """k with shape [head_dim, head_num]"""
    # return tl.cat(tl.expand_dims(k, 0), tl.expand_dims(k, 0))
    # k = tl.join(k, k)
    # k = tl.permute(k, (2, 0, 1))
    # k = tl.expand_dims(k, 0)
    # print(k.shape)
    # k = tl.broadcast_to(k, (2, k.shape[0], k.shape[1]))
    return tl.full((2, k.shape[0], k.shape[1]), 3, dtype=tl.uint8)
    # return tl.zeros((2, k.shape[0], k.shape[1]), dtype=tl.uint8)
    # return tl.random((2, k.shape[0], k.shape[1]), dtype=tl.uint8) 
    k1 = k # k - (k >> 4) * 16
    k2 = k # k >> 4
    k = tl.zeros((2, k.shape[0], k.shape[1]), dtype=tl.uint8)
    k1 = tl.expand_dims(k1, 0)
    k2 = tl.expand_dims(k2, 0)
    k1 = tl.broadcast_to(k1, k.shape)
    k2 = tl.broadcast_to(k2, k.shape)
    k = tl.where(tl.arange(0, 2)[:, None, None] == 1, k, k1)
    k = tl.where(tl.arange(0, 2)[:, None, None] == 0, k, k2)
    return k


@triton.autotune(configs=[
    triton.Config({}, num_stages=2, num_warps=16),
    triton.Config({}, num_stages=2, num_warps=8),
    triton.Config({}, num_stages=2, num_warps=4),
],
                 key=['BLOCK_H', 'BLOCK_N', 'BLOCK_DMODEL', 'BLOCK_DV'])
@wrap_jit_func(type_hint=dict(
    Q=torch.Tensor,
    K=torch.Tensor,
    V=torch.Tensor,
    KScalesZeros=torch.Tensor,
    VScalesZeros=torch.Tensor,
    sm_scale=float,
    KV_seqlens=torch.Tensor,
    Block_offsets=torch.Tensor,
    Acc_out=torch.Tensor,
    stride_qbs=int,
    stride_qh=int,
    stride_qd=int,
    stride_kbs=int,
    stride_kh=int,
    stride_kd=int,
    stride_vbs=int,
    stride_vh=int,
    stride_vd=int,
    stride_kszp=int,
    stride_kszbs=int,
    stride_kszh=int,
    stride_kszd=int,
    stride_vszp=int,
    stride_vszbs=int,
    stride_vszh=int,
    stride_vszd=int,
    quant_bits=int,
    stride_ok=int,
    stride_obs=int,
    stride_oh=int,
    stride_od=int,
    stride_boffb=int,
    kv_group_num=torch.int32,
    block_per_cta=torch.int32,
    window_size=torch.int32,
    head_size=torch.int32,
    head_size_v=torch.int32,
    shared_kv=bool,
    BLOCK_DMODEL=torch.int32,
    BLOCK_DV=torch.int32,
    BLOCK_N=torch.int32,
    BLOCK_H=torch.int32,
    BLOCK_DMODEL1=torch.int32,
))
@triton.jit
def _fwd_grouped_split_quant_kernel(
    Q,
    K,
    V,
    KScalesZeros,
    VScalesZeros,
    sm_scale,
    KV_seqlens,
    Block_offsets,
    Acc_out,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_kszp: tl.constexpr,
    stride_kszbs: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,
    stride_vszp: tl.constexpr,
    stride_vszbs: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    quant_bits: tl.constexpr,
    stride_ok: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_boffb,
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    num_heads_q: tl.constexpr,
    shared_kv: tl.constexpr,
    logit_softcapping: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
):
    """first step kernel of split k attention."""
    cur_batch = tl.program_id(2)
    cur_kv_head = tl.program_id(0)
    split_k_id = tl.program_id(1)

    if BLOCK_H < kv_group_num:
        HEAD_PER_CTA: tl.constexpr = BLOCK_H
    else:
        HEAD_PER_CTA: tl.constexpr = kv_group_num
    cur_head = cur_kv_head * HEAD_PER_CTA + tl.arange(0, BLOCK_H)
    mask_h = cur_head < cur_kv_head * HEAD_PER_CTA + HEAD_PER_CTA
    mask_h = mask_h & (cur_head < num_heads_q)

    q_seqlen = 1
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    if kv_seqlen <= 0:
        return
    history_len = kv_seqlen - q_seqlen

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dsz = tl.arange(0, 1)
    mask_d = offs_d < head_size
    offs_d = offs_d % head_size
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < head_size_v
    offs_dv = offs_dv % head_size_v
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
             offs_n[None, :] * stride_kbs)
    off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd +
             offs_n[:, None] * stride_vbs)
    off_ksz = (cur_kv_head * stride_kszh + offs_dsz[:, None] * stride_kszd +
               offs_n[None, :] * stride_kszbs)
    off_vsz = (cur_kv_head * stride_vszh + offs_dsz[None, :] * stride_vszd +
               offs_n[:, None] * stride_vszbs)

    off_q = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
             offs_d[None, :] * stride_qd)
    q = tl.load(Q + off_q, mask=mask_h[:, None] & mask_d[None, :], other=0)

    v_ptrs = V + off_v
    ksz_ptrs = KScalesZeros + off_ksz
    vsz_ptrs = VScalesZeros + off_vsz

    if BLOCK_DMODEL1 != 0:
        offs_d1 = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL1)
        mask_d1 = offs_d1 < head_size
        offs_d1 = offs_d1 % head_size
        off_q1 = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
                  offs_d1[None, :] * stride_qd)
        q1 = tl.load(Q + off_q1,
                     mask=mask_h[:, None] & mask_d1[None, :],
                     other=0)
        off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                  offs_n[None, :] * stride_kbs)

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    if quant_bits == 4:
        if BLOCK_DMODEL1 != 0:
            offs_d1 = BLOCK_DMODEL // 2 + tl.arange(0, BLOCK_DMODEL1 // 2)
            offs_d1 = offs_d1 % head_size // 2
            off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                      offs_n[None, :] * stride_kbs)
        offs_d = tl.arange(0, BLOCK_DMODEL // 2) % (head_size // 2)
        off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
                 offs_n[None, :] * stride_kbs)
        acc = tl.zeros([BLOCK_H, BLOCK_DV * 2],
                       dtype=tl.float32)  # v head_dim packed
        mask_dv = tl.arange(0, BLOCK_DV * 2) < (head_size_v * 2)
        offs_dv = tl.arange(0, BLOCK_DV * 2) % (head_size_v * 2)
    elif quant_bits == 2:
        if BLOCK_DMODEL1 != 0:
            offs_d1 = BLOCK_DMODEL // 4 + tl.arange(0, BLOCK_DMODEL1 // 4)
            offs_d1 = offs_d1 % head_size // 4
            off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                      offs_n[None, :] * stride_kbs)
        offs_d = tl.arange(0, BLOCK_DMODEL // 4) % (head_size // 4)
        off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
                 offs_n[None, :] * stride_kbs)
        acc = tl.zeros([BLOCK_H, BLOCK_DV * 4],
                       dtype=tl.float32)  # v head_dim packed
        mask_dv = tl.arange(0, BLOCK_DV * 4) < (head_size_v * 4)
        offs_dv = tl.arange(0, BLOCK_DV * 4) % (head_size_v * 4)
    else:
        acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    num_total_blocks = tl.cdiv(kv_seqlen, BLOCK_N)
    BLOCK_PER_CTA = tl.cdiv(num_total_blocks, SPLIT_K)
    kv_len_per_prog = BLOCK_PER_CTA * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, kv_seqlen)

    # load block offset
    # dirty
    start_block_id = loop_start // BLOCK_N
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size,
                                    loop_start) // BLOCK_N
        kv_min_loc = tl.maximum(history_len - window_size, 0)

    loop_start = start_block_id * BLOCK_N
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_offset = tl.load(block_offset_ptrs + start_n // BLOCK_N)

        # -- compute qk ----
        # k = tl.load(k_ptrs + b_offset * stride_kp)
        k = tl.load(K + off_k + b_offset * stride_kp)
        if quant_bits == 4:
            k = _unpack_kv_int4_transposed(k)
            k = tl.view(k, (k.shape[0] * k.shape[1], k.shape[2]))
        if quant_bits == 2:
            k = _unpack_kv_int2_transposed(k)
            k = tl.view(k, (k.shape[0] * k.shape[1], k.shape[2]))
            
        ks = tl.load(ksz_ptrs + b_offset * stride_kszp)
        kz = tl.load(ksz_ptrs + b_offset * stride_kszp + 1)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(K + off_k1 + b_offset * stride_kp)
            if quant_bits == 4:
                k1 = _unpack_kv_int4_transposed(k1)
                k1 = tl.view(k1, (k1.shape[0] * k1.shape[1], k1.shape[2]))
            if quant_bits == 2:
                k1 = _unpack_kv_int2_transposed(k1)
                k1 = tl.view(k1, (k1.shape[0] * k1.shape[1], k1.shape[2]))
            k1 = ((k1 - kz) * ks).to(q.dtype)

        if shared_kv:
            v = tl.trans(k)
            vs = tl.trans(ks)
            vz = tl.trans(kz)
        else:
            # v = tl.load(v_ptrs + b_offset * stride_vp)
            if quant_bits == 4:
                v = tl.load(v_ptrs + b_offset * stride_vp)
                v = _unpack_kv_int4(v)
                v = tl.view(v, (v.shape[0], v.shape[1] * v.shape[2]))
            elif quant_bits == 2:
                v = tl.load(v_ptrs + b_offset * stride_vp)
                v = _unpack_kv_int2(v)
                v = tl.view(v, (v.shape[0], v.shape[1] * v.shape[2]))
            else:
                v = tl.load(v_ptrs + b_offset * stride_vp)
            vs = tl.load(vsz_ptrs + b_offset * stride_vszp)
            vz = tl.load(vsz_ptrs + b_offset * stride_vszp + 1)

        k = ((k - kz) * ks).to(q.dtype)
        v = ((v - vz) * vs).to(q.dtype)
        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL1 != 0:
            qk += tl.dot(q1, k1)
        qk *= sm_scale
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping
        # NOTE: inf - inf = nan, and nan will leads to error
        if start_n + BLOCK_N > history_len or window_size > 0:
            qk_mask = history_len >= (start_n + offs_n)
            if window_size > 0:
                qk_mask = qk_mask and ((start_n + offs_n) >= kv_min_loc)
            qk = tl.where(
                qk_mask[None, :],
                qk,
                -float('inf'),
            )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = fast_expf(qk - m_i_new[:, None])
        alpha = fast_expf(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)

        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    off_acc = (cur_batch * stride_obs + split_k_id * stride_ok +
               cur_head[:, None] * stride_oh + offs_dv[None, :] * stride_od)
    tl.store(Acc_out + off_acc, acc, mask=mask_h[:, None] & mask_dv[None, :])

    if quant_bits == 4:
        off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                    cur_head * stride_oh + head_size_v * 2)
    elif quant_bits == 2: 
        off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                    cur_head * stride_oh + head_size_v * 4)
    else:
        off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                    cur_head * stride_oh + head_size_v)
    tl.store(Acc_out + off_meta, m_i, mask=mask_h)
    tl.store(Acc_out + off_meta + 1, l_i, mask=mask_h)


@wrap_jit_func(type_hint=dict(
    Acc=torch.Tensor,
    Out=torch.Tensor,
    stride_ak=int,
    stride_abs=int,
    stride_ah=int,
    stride_ad=int,
    stride_obs=int,
    stride_oh=int,
    stride_od=int,
    head_size_v=torch.int32,
    SPLIT_K=torch.int32,
    BLOCK_DV=torch.int32,
))
@triton.jit
def _reduce_split_kernel(
    Acc,
    Out,
    stride_ak,
    stride_abs,
    stride_ah,
    stride_ad,
    stride_obs,
    stride_oh,
    stride_od,
    head_size_v: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """second step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # initialize offsets
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_k = tl.arange(0, SPLIT_K)
    mask_dv = offs_dv < head_size_v

    offs_acc = (cur_batch * stride_abs + cur_head * stride_ah +
                offs_k[:, None] * stride_ak + offs_dv[None, :] * stride_ad)
    offs_mi = (cur_batch * stride_abs + cur_head * stride_ah +
               stride_ak * offs_k + head_size_v)

    acc_k = tl.load(Acc + offs_acc, mask=mask_dv[None, :], other=0.0)
    m_k = tl.load(Acc + offs_mi)
    l_k = tl.load(Acc + offs_mi + 1)

    m_max = tl.max(m_k, 0)
    alpha = fast_expf(m_k - m_max)
    acc_k = acc_k * alpha[:, None]
    l_k = l_k * alpha

    acc = tl.sum(acc_k, 0)
    l_sum = tl.sum(l_k, 0)
    acc = acc / l_sum

    out_offs = (cur_batch * stride_obs + cur_head * stride_oh +
                offs_dv * stride_od)
    tl.store(Out + out_offs, acc, mask=mask_dv)


def _get_convert_pv(nv_capability):
    """lazy load convert_pv."""
    if nv_capability[0] >= 8:

        @triton.jit
        def convert_pv(p, v):
            """convert pv."""
            p = p.to(v.dtype)
            return p, v
    else:

        @triton.jit
        def convert_pv(p, v):
            """convert pv."""
            v = v.to(p.dtype)
            return p, v

    return convert_pv


_convert_pv = None


# TODO: how to support inplace autotune?
# @triton.autotune(configs=[
#     triton.Config({}, num_stages=1, num_warps=16),
#     triton.Config({}, num_stages=1, num_warps=8),
#     triton.Config({}, num_stages=1, num_warps=4),
# ],
#                  key=['BLOCK_M', 'BLOCK_N', 'BLOCK_DMODEL', 'BLOCK_DV'])
@wrap_jit_func
@triton.jit
def _fwd_kernel_quant(
    Q,
    K,
    V,
    KScalesZeros,
    VScalesZeros,
    sm_scale,
    Q_start_loc,
    Q_seqlens,
    KV_seqlens,
    Block_offsets,
    Kfull, 
    Vfull, 
    Full_block_offsets,
    Out,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    full_stride_kp: tl.constexpr,
    full_stride_kbs: tl.constexpr,
    full_stride_kh: tl.constexpr,
    full_stride_kd: tl.constexpr,
    full_stride_vp: tl.constexpr,
    full_stride_vbs: tl.constexpr,
    full_stride_vh: tl.constexpr,
    full_stride_vd: tl.constexpr,
    stride_kszp: tl.constexpr,
    stride_kszbs: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,
    stride_vszp: tl.constexpr,
    stride_vszbs: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    quant_bits: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_boffb,
    stride_full_boffb,
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    full_head_size_v: tl.constexpr, 
    shared_kv: tl.constexpr,
    logit_softcapping: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    full_BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
):
    """paged attention kernel."""
    cur_batch = tl.program_id(2)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(0)

    cur_kv_head = cur_head // kv_group_num

    q_seqlen = tl.load(Q_seqlens + cur_batch)
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    q_start_loc = tl.load(Q_start_loc + cur_batch)
    history_len = kv_seqlen - q_seqlen

    block_start_loc = BLOCK_M * start_m
    if block_start_loc >= q_seqlen:
        return

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dsz = tl.arange(0, 1)
    mask_d = offs_d < head_size
    offs_d = offs_d % head_size
    mask_dv = offs_dv < head_size_v
    offs_dv = offs_dv % head_size_v
    

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((q_start_loc + offs_m[:, None]) * stride_qbs +
             cur_head * stride_qh + offs_d[None, :] * stride_qd)
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
             offs_n[None, :] * stride_kbs)
    off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd +
             offs_n[:, None] * stride_vbs)
    off_ksz = (cur_kv_head * stride_kszh + offs_dsz[:, None] * stride_kszd +
               offs_n[None, :] * stride_kszbs)
    off_vsz = (cur_kv_head * stride_vszh + offs_dsz[None, :] * stride_vszd +
               offs_n[:, None] * stride_vszbs)

    # full precision 
    full_offs_dv = tl.arange(0, full_BLOCK_DV)
    full_offs_dv = full_offs_dv % full_head_size_v
    # full_mask_dv = full_offs_dv < full_head_size_v
    full_off_k = (cur_kv_head * full_stride_kh + offs_d[:, None] * full_stride_kd +
             offs_n[None, :] * full_stride_kbs)
    full_off_v = (cur_kv_head * full_stride_vh + full_offs_dv[None, :] * full_stride_vd +
             offs_n[:, None] * full_stride_vbs)
    
    q = tl.load(Q + off_q,
                mask=(offs_m[:, None] < q_seqlen) & mask_d[None, :],
                other=0.0)

    v_ptrs = V + off_v
    ksz_ptrs = KScalesZeros + off_ksz
    vsz_ptrs = VScalesZeros + off_vsz

    
    full_v_ptrs = Vfull + full_off_v
    
    if BLOCK_DMODEL1 != 0:
        offs_d1 = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL1)
        mask_d1 = offs_d1 < head_size
        offs_d1 = offs_d1 % head_size
        off_q1 = ((q_start_loc + offs_m[:, None]) * stride_qbs +
                  cur_head * stride_qh + offs_d1[None, :] * stride_qd)
        q1 = tl.load(Q + off_q1, mask=(offs_m[:, None] < q_seqlen) & mask_d1)
        off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                  offs_n[None, :] * stride_kbs)

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    if quant_bits == 4:
        offs_d = tl.arange(0, BLOCK_DMODEL // 2) % (head_size // 2)
        off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
                 offs_n[None, :] * stride_kbs)
        if BLOCK_DMODEL1 != 0:
            offs_d1 = BLOCK_DMODEL // 2 + tl.arange(0, BLOCK_DMODEL1 // 2)
            offs_d1 = offs_d1 % head_size // 2
            off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                      offs_n[None, :] * stride_kbs)
        acc = tl.zeros([BLOCK_M, BLOCK_DV * 2],
                       dtype=tl.float32)  # v head_dim packed
        mask_dv = tl.arange(0, BLOCK_DV * 2) < (head_size_v * 2)
        offs_dv = tl.arange(0, BLOCK_DV * 2) % (head_size_v * 2)
    elif quant_bits == 2:
        offs_d = tl.arange(0, BLOCK_DMODEL // 4) % (head_size // 4)
        off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
                 offs_n[None, :] * stride_kbs)
        if BLOCK_DMODEL1 != 0:
            offs_d1 = BLOCK_DMODEL // 4 + tl.arange(0, BLOCK_DMODEL1 // 4)
            offs_d1 = offs_d1 % head_size // 4
            off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                      offs_n[None, :] * stride_kbs)
        acc = tl.zeros([BLOCK_M, BLOCK_DV * 4],
                       dtype=tl.float32)  # v head_dim packed
        mask_dv = tl.arange(0, BLOCK_DV * 4) < (head_size_v * 4)
        offs_dv = tl.arange(0, BLOCK_DV * 4) % (head_size_v * 4)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    # this is dirty
    start_block_id = kv_seqlen - kv_seqlen
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size, 0) // BLOCK_N
        kv_min_loc = tl.maximum(history_len + offs_m - window_size, 0)
    kv_start_loc = start_block_id * BLOCK_N
    block_offset_ptrs += start_block_id
    # full_block_offset_ptrs = Full_block_offsets + cur_batch * stride_full_boffb
    
    # seperate loop to process kv cache 
    round_kv_seq_len = tl.cdiv(kv_seqlen, BLOCK_N) * BLOCK_N
    
    for start_n in range(kv_start_loc, round_kv_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        b_offset = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(K + off_k + b_offset * stride_kp)
        if quant_bits == 4:
            k = _unpack_kv_int4_transposed(k)
            k = tl.view(k, (k.shape[0] * k.shape[1], k.shape[2]))
        if quant_bits == 2:
            k = _unpack_kv_int2_transposed(k)
            k = tl.view(k, (k.shape[0] * k.shape[1], k.shape[2]))
        ks = tl.load(ksz_ptrs + b_offset * stride_kszp)
        kz = tl.load(ksz_ptrs + b_offset * stride_kszp + 1)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(K + off_k1 + b_offset * stride_kp)
            if quant_bits == 4:
                k1 = _unpack_kv_int4_transposed(k1)
                k1 = tl.view(k1, (k1.shape[0] * k1.shape[1], k1.shape[2]))
            if quant_bits == 2:
                k1 = _unpack_kv_int2_transposed(k1)
                k1 = tl.view(k1, (k1.shape[0] * k1.shape[1], k1.shape[2]))
            k1 = ((k1 - kz) * ks).to(q.dtype)

        if shared_kv:
            v = tl.trans(k)
            vs = tl.trans(ks)
            vz = tl.trans(kz)
        else:
            if quant_bits == 4:
                v = tl.load(v_ptrs + b_offset * stride_vp)
                v = _unpack_kv_int4(v)
                v = tl.view(v, (v.shape[0], v.shape[1] * v.shape[2]))
            elif quant_bits == 2:
                v = tl.load(v_ptrs + b_offset * stride_vp)
                v = _unpack_kv_int2(v)
                v = tl.view(v, (v.shape[0], v.shape[1] * v.shape[2]))
            else:
                v = tl.load(v_ptrs + b_offset * stride_vp)
            vs = tl.load(vsz_ptrs + b_offset * stride_vszp)
            vz = tl.load(vsz_ptrs + b_offset * stride_vszp + 1)

        # k = tl.view(k, (ks.shape[0], ks.shape[1]))
        v = ((v - vz) * vs).to(q.dtype)
        k = ((k - kz) * ks).to(q.dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL1 != 0:
            qk += tl.dot(q1, k1)
        qk *= sm_scale
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping
        # NOTE: inf - inf = nan, and nan will leads to error
        if start_n + BLOCK_N > history_len or window_size > 0:
            qk_mask = (history_len + offs_m[:, None]) >= (start_n +
                                                          offs_n[None, :])
            if window_size > 0:
                qk_mask = qk_mask and (
                    (start_n + offs_n[None, :]) >= kv_min_loc[:, None])
            qk = tl.where(
                qk_mask,
                qk,
                float(-1e30),
            )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = fast_expf(qk - m_i_new[:, None])
        alpha = fast_expf(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)
        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]
        
        # print(p, v)
        # update acc
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    """
    # print(BLOCK_M, BLOCK_N)
    for start_n in range(round_kv_seq_len, kv_seqlen, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        full_b_offset = tl.load(full_block_offset_ptrs)

        # -- compute qk ----
        k = tl.load(Kfull + full_off_k + full_b_offset * full_stride_kp)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(Kfull + off_k1 + full_b_offset * full_stride_kp)
            k1 = ((k1 - kz) * ks).to(q.dtype)

        if shared_kv:
            v = tl.trans(k)
        else:
            v = tl.load(full_v_ptrs + full_b_offset * full_stride_vp)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL1 != 0:
            qk += tl.dot(q1, k1)
        qk *= sm_scale
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping
        # NOTE: inf - inf = nan, and nan will leads to error
        if start_n + BLOCK_N > history_len or window_size > 0:
            qk_mask = (history_len + offs_m[:, None]) >= (start_n +
                                                          offs_n[None, :])
            if window_size > 0:
                qk_mask = qk_mask and (
                    (start_n + offs_n[None, :]) >= kv_min_loc[:, None])
            qk = tl.where(
                qk_mask,
                qk,
                float(-1e30),
            )

        # -- compute p, m_i and l_i for full precision 
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = fast_expf(qk - m_i_new[:, None])
        alpha = fast_expf(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)
        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    """
    
    acc = fast_dividef(acc, l_i[:, None])
    # initialize pointers to output
    off_o = ((q_start_loc + offs_m[:, None]) * stride_obs +
             cur_head * stride_oh + offs_dv[None, :] * stride_od)
    out_ptrs = Out + off_o
    tl.store(out_ptrs,
             acc,
             mask=(offs_m[:, None] < q_seqlen) & mask_dv[None, :])


from lmdeploy.utils import calculate_time
# @calculate_time(show=True, min_cost_ms=1, signature='paged_attention_quant_full_fwd')
def paged_attention_quant_full_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    k_full: Tensor, 
    v_full: Tensor, 
    full_block_offsets: Tensor, 
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    k_scales_zeros: Tensor = None,
    v_scales_zeros: Tensor = None,
    quant_bits: Literal[0, 2, 4, 8] = 0,
    window_size: int = None,
    sm_scale: float = None,
    logit_softcapping: float = None,
    shared_kv: bool = False,
):
    """Paged Attention forward.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state caches.
        v (Tensor): Value state caches.
        o (Tensor): Output state.
        block_offsets (Tensor): The block offset of key and value.
        q_start_loc (Tensor): Start token location of each data in batch.
        q_seqlens (Tensor): Query length for each data in batch.
        kv_seqlens (Tensor): Key/Value length for each data in batch.
        max_seqlen (int): The max input length.
        BLOCK (int): The kernel block size.
    """
    global _convert_pv
    if _convert_pv is None:
        nv_cap = torch.cuda.get_device_capability()
        _convert_pv = _get_convert_pv(nv_cap)

    if window_size is None:
        window_size = -1

    if logit_softcapping is None:
        logit_softcapping = -1.0

    def _get_block_d(Lk):
        """get block d."""
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DMODEL1 = 0
        if BLOCK_DMODEL != Lk and not shared_kv:
            BLOCK_DMODEL = BLOCK_DMODEL // 2
            BLOCK_DMODEL1 = max(16, triton.next_power_of_2(Lk - BLOCK_DMODEL))
        if shared_kv:
            BLOCK_DV = BLOCK_DMODEL
        else:
            BLOCK_DV = triton.next_power_of_2(Lv)
        return BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    if quant_bits == 4:
        assert Lq == Lk * 2 and Lv * 2 == o.shape[-1]
    elif quant_bits == 2: 
        assert Lq == Lk * 4 and Lv * 4 == o.shape[-1]
    else:
        assert Lq == Lk and Lv == o.shape[-1]

    fullLk, fullLv = k_full.shape[-1], v_full.shape[-1]
    
    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = q_seqlens.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k.shape[-2]

    BLOCK = k.size(1)
    assert BLOCK >= 16
    if Lq > 512 and BLOCK > 32:
        logger.warning(f'`head_dim={Lq}` and `block_size={BLOCK}` '
                       'might leads to bad performance. '
                       'Please reduce `block_size`.')

    kernel_meta = get_kernel_meta(q)
    is_decoding = q.shape[-3] == q_seqlens.size(0)
    
    # import pdb; pdb.set_trace() 
    if not is_decoding:
        BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV = _get_block_d(Lq)
        full_BLOCK_DV = triton.next_power_of_2(fullLv)
        BLOCK_M = max(16, min(BLOCK, 16384 // BLOCK_DMODEL))
        # print(BLOCK_M, BLOCK)
        num_warps = 4
        num_stages = 2
        grid = (triton.cdiv(max_seqlen, BLOCK_M), head, batch)
        # assert BLOCK_M == BLOCK, "BLOCK_M should be equal to BLOCK"
        # print(k.shape, v.shape, q.shape, q_seqlens)
        # print("k_scales_zeros ", k_scales_zeros.shape)
        # import pdb; pdb.set_trace() 
        if quant_bits > 0:
            _fwd_kernel_quant[grid](q,
                                    k,
                                    v,
                                    k_scales_zeros,
                                    v_scales_zeros,
                                    sm_scale,
                                    q_start_loc,
                                    q_seqlens,
                                    kv_seqlens,
                                    block_offsets,
                                    k_full, 
                                    v_full, 
                                    full_block_offsets,
                                    o,
                                    stride_qbs=q.stride(-3),
                                    stride_qh=q.stride(-2),
                                    stride_qd=q.stride(-1),
                                    stride_kp=k.stride(-4),
                                    stride_kbs=k.stride(-3),
                                    stride_kh=k.stride(-2),
                                    stride_kd=k.stride(-1),
                                    stride_vp=v.stride(-4),
                                    stride_vbs=v.stride(-3),
                                    stride_vh=v.stride(-2),
                                    stride_vd=v.stride(-1),
                                    
                                    full_stride_kp=k_full.stride(-4),
                                    full_stride_kbs=k_full.stride(-3),
                                    full_stride_kh=k_full.stride(-2),
                                    full_stride_kd=k_full.stride(-1),
                                    full_stride_vp=v_full.stride(-4),
                                    full_stride_vbs=v_full.stride(-3),
                                    full_stride_vh=v_full.stride(-2),
                                    full_stride_vd=v_full.stride(-1),
                                    
                                    stride_kszp=k_scales_zeros.stride(-4),
                                    stride_kszbs=k_scales_zeros.stride(-3),
                                    stride_kszh=k_scales_zeros.stride(-2),
                                    stride_kszd=k_scales_zeros.stride(-1),
                                    stride_vszp=v_scales_zeros.stride(-4),
                                    stride_vszbs=v_scales_zeros.stride(-3),
                                    stride_vszh=v_scales_zeros.stride(-2),
                                    stride_vszd=v_scales_zeros.stride(-1),
                                    quant_bits=quant_bits,
                                    stride_obs=o.stride(-3),
                                    stride_oh=o.stride(-2),
                                    stride_od=o.stride(-1),
                                    stride_boffb=block_offsets.stride(0),
                                    stride_full_boffb=full_block_offsets.stride(0),
                                    kv_group_num=kv_group_num,
                                    window_size=window_size,
                                    head_size=Lq,
                                    head_size_v=Lv,
                                    full_head_size_v=fullLv,
                                    shared_kv=shared_kv,
                                    logit_softcapping=logit_softcapping,
                                    BLOCK_M=BLOCK_M,
                                    BLOCK_DMODEL=BLOCK_DMODEL,
                                    BLOCK_DV=BLOCK_DV,
                                    BLOCK_N=BLOCK,
                                    BLOCK_DMODEL1=BLOCK_DMODEL1,
                                    full_BLOCK_DV=full_BLOCK_DV,
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                    **kernel_meta)
        
    else:
        SPLIT_K = 4
        if quant_bits not in [2, 4]:
            acc = q.new_empty(batch,
                              head,
                              SPLIT_K,
                              Lv + 2,
                              dtype=torch.float32)
        else:
            acc = q.new_empty(batch,
                              head,
                              SPLIT_K,
                              o.shape[-1] + 2,
                              dtype=torch.float32)
        
        # print("q.shape ", q.shape)
        # print("k.shape ", k.shape)
        # print("v.shape ", v.shape)
        # print("k_full.shape ", k_full.shape)
        # print("v_full.shape ", v_full.shape)
        
        BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV = _get_block_d(Lq)
        p2_kv_group_num = triton.next_power_of_2(kv_group_num)
        BLOCK_H = max(16, min(BLOCK, p2_kv_group_num))
        grid_1 = triton.cdiv(head, min(BLOCK_H, kv_group_num))
        grid = (
            grid_1,
            SPLIT_K,
            batch,
        )
        if quant_bits > 0:
            _fwd_grouped_split_quant_kernel[grid](
                q,
                k,
                v,
                k_scales_zeros,
                v_scales_zeros,
                sm_scale,
                kv_seqlens // BLOCK * BLOCK,
                block_offsets,
                acc,
                stride_qbs=q.stride(-3),
                stride_qh=q.stride(-2),
                stride_qd=q.stride(-1),
                stride_kp=k.stride(-4),
                stride_kbs=k.stride(-3),
                stride_kh=k.stride(-2),
                stride_kd=k.stride(-1),
                stride_vp=v.stride(-4),
                stride_vbs=v.stride(-3),
                stride_vh=v.stride(-2),
                stride_vd=v.stride(-1),
                stride_kszp=k_scales_zeros.stride(-4),
                stride_kszbs=k_scales_zeros.stride(-3),
                stride_kszh=k_scales_zeros.stride(-2),
                stride_kszd=k_scales_zeros.stride(-1),
                stride_vszp=v_scales_zeros.stride(-4),
                stride_vszbs=v_scales_zeros.stride(-3),
                stride_vszh=v_scales_zeros.stride(-2),
                stride_vszd=v_scales_zeros.stride(-1),
                quant_bits=quant_bits,
                stride_ok=acc.stride(-2),
                stride_obs=acc.stride(-4),
                stride_oh=acc.stride(-3),
                stride_od=acc.stride(-1),
                stride_boffb=block_offsets.stride(0),
                kv_group_num=kv_group_num,
                window_size=window_size,
                head_size=Lq,
                head_size_v=Lv,
                num_heads_q=head,
                shared_kv=shared_kv,
                logit_softcapping=logit_softcapping,
                SPLIT_K=SPLIT_K,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DV=BLOCK_DV,
                BLOCK_N=BLOCK,
                BLOCK_H=BLOCK_H,
                BLOCK_DMODEL1=BLOCK_DMODEL1,
                **kernel_meta)

        

        num_warps = 4
        grid = (batch, head)
        if quant_bits == 4:
            Lv *= 2
            BLOCK_DV *= 2
        elif quant_bits == 2: 
            Lv *= 4
            BLOCK_DV *= 4
        _reduce_split_kernel[grid](acc,
                                   o,
                                   stride_ak=acc.stride(-2),
                                   stride_abs=acc.stride(-4),
                                   stride_ah=acc.stride(-3),
                                   stride_ad=acc.stride(-1),
                                   stride_obs=o.stride(-3),
                                   stride_oh=o.stride(-2),
                                   stride_od=o.stride(-1),
                                   SPLIT_K=SPLIT_K,
                                   head_size_v=Lv,
                                   BLOCK_DV=BLOCK_DV,
                                   num_warps=num_warps,
                                   num_stages=1,
                                   **kernel_meta)
