
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import copy 
from lmdeploy.utils import calculate_time
# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PyramidKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None):
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.steps = -1
        self.beta = beta

        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups): 
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num

        steps = (max_num - min_num) // self.num_hidden_layers
        max_capacity_prompt = max_num - self.layer_idx * steps

        # print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}")
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling



    # @calculate_time(show=True, min_cost_ms=1, signature='H2O KV cluster')
    def update_kv(self, key_states, query_states, value_states, attn_metadata, num_key_value_groups, layer_idx, **kwargs):
        """
        key_states: [seq, head, head_dim]
        attn_metadata: 
                is_decoding: bool
                block_offsets: torch.Tensor
                q_start_loc: torch.Tensor = None
                q_seqlens: torch.Tensor = None
                kv_seqlens: torch.Tensor = None
        """
        
        # return key_states, value_states, None
        # if (attn_metadata.kv_seqlens[0] <= self.max_capacity_prompt + 32 and attn_metadata.is_decoding == True): 
        if attn_metadata.is_decoding == True: 
            return key_states, value_states, None
        attn_cache = kwargs.get("attn_cache", None)
        block_size = key_states.size(1)
        num_kv_heads = key_states.size(2)
        num_q_heads =query_states.size(1)
        head_dim = key_states.size(3)
        bsz = attn_metadata.q_seqlens.size(0)
        attn_metadata_copy = None 
        if layer_idx == 0: 
            attn_metadata_copy = copy.deepcopy(attn_metadata)
        
        for i in range(bsz): 
            # index_start = time.time()
            start, length = attn_metadata.q_start_loc[i], attn_metadata.kv_seqlens[i]
            BLOCK_OFFSETS = attn_metadata.block_offsets[i]
            # start_loc = attn_metadata.block_offsets[i][0].item() * block_size
            if (attn_metadata.kv_seqlens[i] < self.max_capacity_prompt and attn_metadata.is_decoding == False) :
                # indices = torch.tensor(range(attn_metadata.kv_seqlens[i]), dtype=torch.int64).to(key_states.device) + start_loc
                indices = list() 
                for block_id in range(0, attn_metadata.kv_seqlens[i] // block_size):
                    indices.extend([BLOCK_OFFSETS[block_id] * block_size + j for j in range(block_size)])
                if attn_metadata.kv_seqlens[i] % block_size != 0: 
                    indices.extend([BLOCK_OFFSETS[attn_metadata.kv_seqlens[i] // block_size] * block_size + j for j in range(attn_metadata.kv_seqlens[i] % block_size)])
                indices = torch.tensor(indices, dtype=torch.int64).to(key_states.device)
                indices = indices.unsqueeze(-1).unsqueeze(-1).repeat(1, num_kv_heads, head_dim)
            else: 
                sub_attn_cache = attn_cache[start:start+length-self.window_size]
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(sub_attn_cache, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(sub_attn_cache, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')

                indices_past = sub_attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=0).indices 
                for window_id in range(indices_past.size(0)): 
                    for head_id in range(num_kv_heads): 
                        # import pdb; pdb.set_trace() 
                        block_id = indices_past[window_id, head_id] // block_size
                        indices_past[window_id, head_id] = BLOCK_OFFSETS[block_id] * block_size + indices_past[window_id, head_id] % block_size
                indices_cur = torch.tensor(range(self.window_size), dtype=torch.int64).to(key_states.device)
                indices_cur = list() 
                for j in range(attn_metadata.kv_seqlens[i] - self.window_size, attn_metadata.kv_seqlens[i]): 
                    indices_cur.append(BLOCK_OFFSETS[j // block_size] * block_size + j % block_size )
                
                # indices_cur = torch.tensor(range(self.window_size), dtype=torch.int64).to(key_states.device) + attn_metadata.kv_seqlens[i] - self.window_size
                indices_cur = torch.tensor(indices_cur, dtype=torch.int64).to(key_states.device)
                
                indices_cur = indices_cur.unsqueeze(-1).repeat(1, indices_past.size(1)) # align size 
                indices = torch.cat([indices_past, indices_cur], dim = 0)
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.kv_seqlens[i] = indices.size(0)
                indices = indices.unsqueeze(-1).repeat(1, 1, head_dim)
            
            # print("time cost to cal index is ", time.time() - index_start)
            compress_key_states = key_states.view(-1, num_kv_heads, head_dim).gather(dim=0, index=indices)
            compress_value_states = value_states.view(-1, num_kv_heads, head_dim).gather(dim=0, index=indices)
            # import pdb; pdb.set_trace() 
            num_blocks = (compress_key_states.size(0) - 1) // block_size + 1
            block_offsets = attn_metadata.block_offsets[i]
            for j in range(num_blocks): 
                length = min(block_size, compress_key_states.size(0) - j * block_size)
                key_states[block_offsets[j], :length, :, :] = compress_key_states[j * block_size:j * block_size+length, :, :]
                value_states[block_offsets[j], :length, :, :] = compress_value_states[j * block_size:j * block_size+length, :, :]
            if attn_metadata_copy is not None: 
                attn_metadata_copy.block_offsets[i][num_blocks:] = 0
        # print(f"keep {num_blocks} blocks")
        return key_states, value_states, attn_metadata_copy
    
    # def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
    #     # check if prefix phase
    #     assert key_states.shape[-2] == query_states.shape[-2]
    #     bsz, num_heads, q_len, head_dim = query_states.shape

    #     # print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")

    #     if q_len < self.max_capacity_prompt:
    #         return key_states, value_states
    #     else:
    #         attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
    #         mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    #         mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    #         mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    #         mask = mask.to(attn_weights.device)
    #         attention_mask = mask[None, None, :, :]

    #         attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

    #         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #         attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
    #         if self.pooling == 'avgpool':
    #             attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
    #         elif self.pooling == 'maxpool':
    #             attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
    #         else:
    #             raise ValueError('Pooling method not supported')
    #         indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
    #         indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    #         k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
    #         v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
    #         k_cur = key_states[:, :, -self.window_size:, :]
    #         v_cur = value_states[:, :, -self.window_size:, :]
    #         key_states = torch.cat([k_past_compress, k_cur], dim = 2)
    #         value_states = torch.cat([v_past_compress, v_cur], dim = 2)
    #         return key_states, value_states


class H2OKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.recent_window_size = self.max_capacity_prompt - self.window_size

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.recent_window_size = self.max_capacity_prompt - self.window_size

    
    # @calculate_time(show=True, min_cost_ms=1, signature='H2O KV cluster')
    def update_kv(self, key_states, query_states, value_states, attn_metadata, num_key_value_groups, layer_idx, **kwargs):
        """
        key_states: [seq, head, head_dim]
        attn_metadata: 
                is_decoding: bool
                block_offsets: torch.Tensor
                q_start_loc: torch.Tensor = None
                q_seqlens: torch.Tensor = None
                kv_seqlens: torch.Tensor = None
        """
        
        # return key_states, value_states, None
        # if (attn_metadata.kv_seqlens[0] <= self.max_capacity_prompt + 32 and attn_metadata.is_decoding == True): 
        if attn_metadata.is_decoding == True: 
            return key_states, value_states, None
        attn_cache = kwargs.get("attn_cache", None)
        block_size = key_states.size(1)
        num_kv_heads = key_states.size(2)
        num_q_heads =query_states.size(1)
        head_dim = key_states.size(3)
        bsz = attn_metadata.q_seqlens.size(0)
        attn_metadata_copy = None 
        if layer_idx == 0: 
            attn_metadata_copy = copy.deepcopy(attn_metadata)
        
        for i in range(bsz):
            start, length = attn_metadata.q_start_loc[i], attn_metadata.kv_seqlens[i]
            window_size = self.window_size if attn_metadata.kv_seqlens[i] % self.window_size == 0 else attn_metadata.kv_seqlens[i] % self.window_size
            max_capacity_prompt = self.max_capacity_prompt - self.window_size + window_size
            if False: 
                num_blocks = (max_capacity_prompt - 1) // block_size + 1
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.block_offsets[i][num_blocks:] = 0
                
                if (attn_metadata.kv_seqlens[i] < max_capacity_prompt and attn_metadata.is_decoding == False):
                    pass 
                else: 
                    if attn_metadata_copy is not None: 
                        attn_metadata_copy.kv_seqlens[i] = max_capacity_prompt
                continue 
            
            # start_loc = attn_metadata.block_offsets[i][0].item() * block_size
            if (attn_metadata.kv_seqlens[i] < max_capacity_prompt and attn_metadata.is_decoding == False):
                block_indices = None 
            else: 
                sub_attn_cache = attn_cache[start:start+length-window_size]
                seq_num, head_num = sub_attn_cache.shape 
                # sub_attn_cache = sub_attn_cache.view(seq_num//block_size, block_size * head_num).sum(-1)
                # block_indices = sub_attn_cache.topk((max_capacity_prompt - window_size) // block_size, dim=0).indices.to("cpu").tolist()
                block_indices = [i for i in range((max_capacity_prompt - window_size) // block_size)]
                num_blocks = (max_capacity_prompt - 1) // block_size + 1
                recent_blocks = (window_size - 1) // block_size + 1
                start = max(num_blocks - recent_blocks, max(block_indices))
                end = num_blocks
                for j in range(start, end): block_indices.append(j)
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.kv_seqlens[i] = max_capacity_prompt
                
            
            if block_indices is None: 
                num_blocks = (attn_metadata.kv_seqlens[i] - 1) // block_size + 1
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.block_offsets[i][num_blocks:] = 0
                continue 
            
            if True:
                # the logical operation is wrong, just for speed test
                num_blocks = (max_capacity_prompt - 1) // block_size + 1
                block_offsets = attn_metadata.block_offsets[i]
                for j in range(len(block_indices)): 
                    if block_offsets[j] == block_indices[j]: continue 
                    length = min(block_size, max_capacity_prompt - j * block_size)
                    key_states[block_offsets[j], :length, :, :] = key_states[block_indices[j], :length, :]
                    value_states[block_offsets[j], :length, :, :] = value_states[block_indices[j], :length, :]
                
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.block_offsets[i][num_blocks:] = 0
        
        return key_states, value_states, attn_metadata_copy
    

class StreamingLLMKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    # @calculate_time(show=True, min_cost_ms=0.1, signature='StreammingLLM KV cluster')
    def update_kv(self, key_states, query_states, value_states, attn_metadata, num_key_value_groups, layer_idx, **kwargs):
        """
        key_states: [seq, head, head_dim]
        attn_metadata: 
                is_decoding: bool
                block_offsets: torch.Tensor
                q_start_loc: torch.Tensor = None
                q_seqlens: torch.Tensor = None
                kv_seqlens: torch.Tensor = None
        """
        # return key_states, value_states, None
        # if (attn_metadata.kv_seqlens[0] <= self.max_capacity_prompt + 32 and attn_metadata.is_decoding == True): 
            
        if attn_metadata.is_decoding == True: 
            return key_states, value_states, None

        block_size = key_states.size(1)
        num_q_heads =query_states.size(1) # seq, head, hidden_dim 
        num_kv_heads = key_states.size(2) # seq, blk, head, hidden_dim
        head_dim = key_states.size(3)
        bsz = attn_metadata.q_seqlens.size(0)
        attn_metadata_copy = None 
        
        

        if layer_idx == 0: 
            attn_metadata_copy = copy.deepcopy(attn_metadata)


        for i in range(bsz): 
            # FIXME: update q_start_loc
            BLOCK_OFFSETS = attn_metadata.block_offsets[i]
            if (attn_metadata.kv_seqlens[i] < self.max_capacity_prompt and attn_metadata.is_decoding == False):
                block_indices = None 
            else: 
                block_indices = [0] # FIXME: assume self.window_size as block_size 
                # length_past = self.max_capacity_prompt - self.window_size
                block_indices.append(0)
                start_idx = (attn_metadata.kv_seqlens[i] - self.window_size) // block_size
                end_idx = attn_metadata.kv_seqlens[i] // block_size
                for j in range(max(1, start_idx), end_idx): block_indices.append(j)
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.kv_seqlens[i] = self.max_capacity_prompt
            
            if block_indices is None: 
                num_blocks = (attn_metadata.kv_seqlens[i] - 1) // block_size + 1
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.block_offsets[i][num_blocks:] = 0
                continue 
            
            if True:
                # the logical operation is wrong, just for speed test 
                num_blocks = (self.max_capacity_prompt - 1) // block_size + 1
                block_offsets = attn_metadata.block_offsets[i]
                for j in range(len(block_indices)): 
                    if block_offsets[j] == block_indices[j]: continue 
                    length = min(block_size, self.max_capacity_prompt - j * block_size)
                    key_states[block_offsets[j], :length, :, :] = key_states[block_indices[j], :length, :]
                    value_states[block_offsets[j], :length, :, :] = value_states[block_indices[j], :length, :]
                
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.block_offsets[i][num_blocks:] = 0
            else:  
                # print("time cost to cal index is ", time.time() - index_start)
                indices = indices.unsqueeze(-1).unsqueeze(-1).repeat(1, num_kv_heads, head_dim)
                compress_key_states = key_states.view(-1, num_kv_heads, head_dim).gather(dim=0, index=indices)
                compress_value_states = value_states.view(-1, num_kv_heads, head_dim).gather(dim=0, index=indices)
                
                num_blocks = (compress_key_states.size(0) - 1) // block_size + 1
                block_offsets = attn_metadata.block_offsets[i]
                for j in range(num_blocks): 
                    length = min(block_size, compress_key_states.size(0) - j * block_size)
                    if length % block_size != 0:
                        key_states[block_offsets[j], length:, :, :].fill_(0.)
                        value_states[block_offsets[j], length:, :, :].fill_(0.)
                        # import pdb; pdb.set_trace() 
                        
                    key_states[block_offsets[j], :length, :, :] = compress_key_states[j * block_size:j * block_size+length, :, :]
                    value_states[block_offsets[j], :length, :, :] = compress_value_states[j * block_size:j * block_size+length, :, :]

                # for j in range(num_blocks, len(block_offsets)):
                #     key_states[block_offsets[j], :, :, :].fill_(0.)
                #     value_states[block_offsets[j], :, :, :].fill_(0.)
                
                if attn_metadata_copy is not None: 
                    attn_metadata_copy.block_offsets[i][num_blocks:] = 0
        
        return key_states, value_states, attn_metadata_copy


def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    self.kv_cluster = PyramidKVCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    self.config.window_size = 512-16-16
    self.config.max_capacity_prompt = 512-16 # 2664 - 5 # 512
    self.config.pooling = 'avgpool'
    self.config.kernel_size = 5
    
    self.kv_cluster = SnapKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )

def init_H2O(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
            
    self.config.window_size = 512-64
    self.config.max_capacity_prompt = 512 # 2664 - 5 # 512
    self.config.pooling = 'maxpool'
    self.config.kernel_size = 7
    
    self.kv_cluster = H2OKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )

def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    self.config.window_size = 512 - 64
    self.config.max_capacity_prompt = 512 # 2664 - 5 # 512
    self.config.pooling = 'maxpool'
    self.config.kernel_size = 7
    self.kv_cluster = StreamingLLMKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )


# import pdb; pdb.set_trace() 
# # FIXME: update q_start_loc
# # att-based importance score
# attn_weights = torch.matmul(query_states_per_sample[..., -self.window_size:, :], key_states_per_sample.transpose(1, 2)) / math.sqrt(head_dim)
# indices_past = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device) + attn_metadata.q_start_loc[i]
# mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
# mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
# mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
# mask = mask.to(attn_weights.device)
# # attn_weights[:, :, -self.window_size:, -self.window_size:] += 下斜角
# attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)