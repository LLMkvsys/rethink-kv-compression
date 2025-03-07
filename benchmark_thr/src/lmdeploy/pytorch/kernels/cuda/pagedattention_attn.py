# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
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
def _fwd_kernel_attn(
    Q,
    K,
    sm_scale,
    Q_start_loc,
    Q_seqlens,
    K_start_loc,
    K_seqlens,
    KV_seqlens,
    Block_offsets,
    Out,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_obs: tl.constexpr, 
    stride_ow: tl.constexpr, 
    stride_oh: tl.constexpr,
    stride_boffb,
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    shared_kv: tl.constexpr,
    logit_softcapping: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
):
    """paged attention kernel."""
    cur_batch = tl.program_id(2) # batch dim
    cur_head = tl.program_id(1) # head dim
    start_m = tl.program_id(0) # recent_window dim

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
    mask_d = offs_d < head_size
    offs_d = offs_d % head_size
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((q_start_loc + offs_m[:, None]) * stride_qbs +
             cur_head * stride_qh + offs_d[None, :] * stride_qd)
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
             offs_n[None, :] * stride_kbs)
    # print("q_start_loc", q_start_loc)
    # print("off_q", off_q)
    # off_o = (cur_kv_head * stride_oh + offs_dv[None, :] * stride_od +
    #          offs_n[:, None] * stride_obs)
    q = tl.load(Q + off_q,
                mask=(offs_m[:, None] < q_seqlen) & mask_d[None, :],
                other=0.0)

    k_ptrs = K + off_k

    if BLOCK_DMODEL1 != 0:
        offs_d1 = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL1)
        mask_d1 = offs_d1 < head_size
        offs_d1 = offs_d1 % head_size
        off_q1 = ((q_start_loc + offs_m[:, None]) * stride_qbs +
                  cur_head * stride_qh + offs_d1[None, :] * stride_qd)
        q1 = tl.load(Q + off_q1, mask=(offs_m[:, None] < q_seqlen) & mask_d1)
        off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                  offs_n[None, :] * stride_kbs)
        k1_ptrs = K + off_k1

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # this is dirty
    start_block_id = kv_seqlen - kv_seqlen
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size, 0) // BLOCK_N
        kv_min_loc = tl.maximum(history_len + offs_m - window_size, 0)
    
    
    kv_start_loc = start_block_id * BLOCK_N
    block_offset_ptrs += start_block_id
    for start_n in range(kv_start_loc, kv_seqlen, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        b_offset = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(k_ptrs + b_offset * stride_kp)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(k1_ptrs + b_offset * stride_kp)

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
        if start_n + BLOCK_N > kv_seqlen:
            qk = tl.where(
                (offs_n[None, :] + start_n) < kv_seqlen,
                qk,
                float(-1e30),
            )
        
        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = fast_expf(qk - m_i_new[:, None])
        alpha = fast_expf(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    
    # reinit 
    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb
    kv_start_loc = start_block_id * BLOCK_N
    block_offset_ptrs += start_block_id
    # # save back att out 
    for start_n in range(kv_start_loc, kv_seqlen, BLOCK_N):
    # for start_n in range(kv_start_loc, kv_start_loc+BLOCK_N+1, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_offset = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(k_ptrs + b_offset * stride_kp)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(k1_ptrs + b_offset * stride_kp)

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
        
        if start_n + BLOCK_N > kv_seqlen:
            qk = tl.where(
                (offs_n[None, :] + start_n) < kv_seqlen,
                qk,
                float(-1e30),
            )
        # initialize pointers to output
        # torch.Size([80, 16, 32])
        off_o = ((offs_n[:, None] + start_n) * stride_obs) + (offs_m[None, :] * stride_ow) + cur_head
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                tl.trans(fast_expf(qk - m_i[:, None]) / l_i[:, None]),
                mask=(offs_n[:, None] + start_n) < kv_seqlen)
        
        


def paged_attention_fwd_attn(
    q: Tensor,
    k: Tensor,
    o: Tensor,
    query_recent_window_size: Tensor, 
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
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
            BLOCK_DV = triton.next_power_of_2(Lo)
        return BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV

    # shape constraints
    # import pdb; pdb.set_trace()
    # print(q.shape, k.shape, o.shape)
    # torch.Size([2664, 32, 128]) torch.Size([1362, 64, 8, 128]) torch.Size([2664, 32])
    Lq, Lk, Lo = q.shape[-1], k.shape[-1], o.shape[-1]
    assert Lq == Lk, Lo == o.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = q_seqlens.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k.shape[-2]
    
    BLOCK = k.size(1)
    assert BLOCK >= 16
    if Lk > 512 and BLOCK > 32:
        logger.warning(f'`head_dim={Lk}` and `block_size={BLOCK}` '
                       'might leads to bad performance. '
                       'Please reduce `block_size`.')

    kernel_meta = get_kernel_meta(q)
    is_decoding = q.shape[-3] == q_seqlens.size(0)
    assert q.shape[-3] != q_seqlens.size(0)
    if not is_decoding:
        BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV = _get_block_d(Lk)
        BLOCK_M = min(max(16, min(BLOCK, 16384 // BLOCK_DMODEL)), query_recent_window_size[0].item())
        num_warps = 4
        num_stages = 2
        grid = (triton.cdiv(query_recent_window_size[0].item(), BLOCK_M), head, batch)
        def run_soft_max(key_states, query_states, window_size=16): 
            mask = torch.full((window_size, window_size), torch.finfo(query_states.dtype).min, device=query_states.device)
            mask_cond = torch.arange(mask.size(-1), device=query_states.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            attention_mask = mask 
            
            # attn weights update 
            attn_weights = torch.matmul(query_states.transpose(0, 1)[:, -window_size:, :], key_states.transpose(0, 1).transpose(1, 2)) * sm_scale
            attn_weights[:, -window_size:, -window_size:] += attention_mask

            # compute the softmax 
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            return attn_weights.transpose(1, 2)
    
        # true_k = torch.zeros((80, 8, 128), dtype=torch.bfloat16).to(k.device)
        # true_k[:64, ...] = k[0, ...]
        # true_k[64:, ] = k[1, :16, :]
        # true_k = true_k.unsqueeze(-2).repeat(1, 1, 4, 1).view(-1, 32, 128)
        # true_q = q
        # gt_attn_out = run_soft_max(true_k, true_q)
        
        # kv_seqlen - q_seqlen
        # import pdb; pdb.set_trace()
        print(f"grid => len: {len(grid)}, ele0: {grid[0]}, ele1: {grid[1]}, ele2: {grid[2]}") # 3, 1, 8 ,1
        _fwd_kernel_attn[grid](Q=q,
                          K=k,
                          sm_scale=sm_scale,
                          Q_start_loc=q_start_loc+q_seqlens-query_recent_window_size, 
                          Q_seqlens=query_recent_window_size,
                          K_start_loc=q_start_loc,
                          K_seqlens=q_seqlens,
                          KV_seqlens=kv_seqlens,
                          Block_offsets=block_offsets,
                          Out=o,
                          stride_qbs=q.stride(-3),
                          stride_qh=q.stride(-2),
                          stride_qd=q.stride(-1),
                          stride_kp=k.stride(-4),
                          stride_kbs=k.stride(-3),
                          stride_kh=k.stride(-2),
                          stride_kd=k.stride(-1),
                          # o.shape is (seq, window, head)
                          stride_obs=o.stride(-3), 
                          stride_ow=o.stride(-2), 
                          stride_oh=o.stride(-1),
                          stride_boffb=block_offsets.stride(0),
                          kv_group_num=kv_group_num,
                          window_size=window_size,
                          head_size=Lk,
                          shared_kv=shared_kv,
                          logit_softcapping=logit_softcapping,
                          BLOCK_M=BLOCK_M,
                          BLOCK_DMODEL=BLOCK_DMODEL,
                          BLOCK_DV=BLOCK_DV,
                          BLOCK_N=BLOCK,
                          BLOCK_DMODEL1=BLOCK_DMODEL1,
                          num_warps=num_warps,
                          num_stages=num_stages,
                          **kernel_meta)

