# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.messages import SchedulerSequence
from .adapter.adapter import SchedulerAdapter


@dataclass
class AdapterInfo:
    adapter_ids: torch.LongTensor
    rank_offsets: torch.LongTensor

    @classmethod
    def from_adapters(cls, adapters: List[SchedulerAdapter]):
        """from adapters."""
        if len(adapters) == 0:
            return None
        adapter_ids = [ada.adapter_id for ada in adapters]
        adapter_ids = torch.tensor(adapter_ids)
        rank_offsets = [torch.from_numpy(ada.rank_offset) for ada in adapters]
        rank_offsets = torch.stack(rank_offsets)

        return cls(
            adapter_ids=adapter_ids,
            rank_offsets=rank_offsets,
        )

    def update_offsets(self, rank_offsets: torch.LongTensor):
        """update rank offsets."""
        rank_offsets[self.adapter_ids] = self.rank_offsets

    def to_device(self, device: str):
        """to device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            out_dict[k] = v

        return AdapterInfo(**out_dict)


@dataclass
class VisionModelInputs:
    """Vision model inputs."""
    history_lengths: torch.LongTensor = None
    history_image_nums: torch.LongTensor = None
    history_image_token_lengths: torch.LongTensor = None
    input_embeddings: List[List[torch.Tensor]] = None
    input_embedding_ranges: List[torch.LongTensor] = None
    input_embedding_indexing: torch.BoolTensor = None

    def to_device(self, device: str):
        """to device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            elif k == 'input_embedding_ranges' and v is not None:
                v = [e.to(device) for e in v]
            elif k == 'input_embeddings' and v is not None:
                v = [[e.to(device) for e in li] for li in v]
            out_dict[k] = v

        return VisionModelInputs(**out_dict)

    def get_inputs(self, history_lengths: torch.Tensor,
                   seq_lengths: torch.Tensor):
        """get vision embedding inputs."""
        input_embeddings = None
        input_embedding_indexing = None
        if self.input_embeddings is not None and len(
                self.input_embeddings) > 0:
            input_embedding_li = []
            for (his_len, seq_len, embeddings,
                 emb_ranges) in zip(history_lengths, seq_lengths,
                                    self.input_embeddings,
                                    self.input_embedding_ranges):
                for emb, (emb_start, emb_end) in zip(embeddings, emb_ranges):
                    start = max(emb_start, his_len) - emb_start
                    end = min(emb_end, his_len + seq_len) - emb_start
                    if 0 <= start < end:
                        input_embedding_li.append(emb[start:end])
            # has embeddings
            if len(input_embedding_li) > 0:
                input_embeddings = torch.cat(input_embedding_li, dim=0)
                device = input_embeddings.device
                starts = history_lengths - self.history_lengths
                ends = starts + seq_lengths
                input_embedding_indexing = torch.cat([
                    indexing[s:e] for indexing, s, e in zip(
                        self.input_embedding_indexing, starts, ends)
                ],
                                                     dim=0)
                index_ranges = torch.arange(input_embedding_indexing.numel(),
                                            device=device)
                input_embedding_indexing = index_ranges[
                    input_embedding_indexing]
        return input_embeddings, input_embedding_indexing


@dataclass
class ModelInputs:
    """Input of the model."""
    input_ids: torch.LongTensor
    seq_length: torch.LongTensor
    history_lengths: torch.LongTensor
    history_ignore_lengths: torch.LongTensor
    block_offsets: torch.LongTensor
    block_full_offsets: torch.LongTensor
    is_decoding: bool
    num_ignored_history: torch.LongTensor
    local_adapter_ids: torch.LongTensor = None
    adapter_info: AdapterInfo = None
    vision_inputs: VisionModelInputs = None
    messages: List[SchedulerSequence] = None

    def update(self, input_ids: torch.LongTensor):
        """update input ids."""
        assert self.is_decoding
        self.history_lengths = self.history_lengths + 1
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        self.input_ids = input_ids
        return self

    def split(self, split_size: int, block_size: int):
        """split inputs."""
        assert len(
            self.seq_length) == 1, ('Can not perform split on batched input.')
        assert split_size % block_size == 0, (
            'split_size should be multi of block_size.')

        input_ids = self.input_ids
        if input_ids.numel() < split_size:
            return self

        num_blocks = split_size // block_size
        overlap = (self.history_lengths[0] % block_size != 0)
        max_seq_len = self.seq_length[0].item()
        ret = []
        block_start = 0
        for i in range(0, max_seq_len, split_size):
            start = i
            end = min(max_seq_len, i + split_size)
            block_end = block_start + num_blocks
            if overlap:
                block_end += 1
            import pdb; pdb.set_trace() 
            block_offsets = self.block_offsets
            inp = ModelInputs(
                input_ids=self.input_ids[:, start:end],
                seq_length=input_ids.new_tensor([end - start]),
                block_offsets=block_offsets,
                history_lengths=self.history_lengths + start,
                is_decoding=self.is_decoding,
                num_ignored_history=self.num_ignored_history,
                local_adapter_ids=self.local_adapter_ids,
                adapter_info=self.adapter_info,
                vision_inputs=self.vision_inputs,
            )
            ret.append(inp)
            block_start += num_blocks

        return ret

    def to_device(self, device: str):
        """to device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            elif isinstance(v, VisionModelInputs):
                v = v.to_device(device)
            elif isinstance(v, AdapterInfo):
                v = v.to_device(device)
            out_dict[k] = v

        return ModelInputs(**out_dict)


@dataclass
class StepContext:
    """context of Model.

    patched model might need extra information to perform inference. This
    dataclass provide these infos and tools.
    """
    input_ids: torch.LongTensor
    block_offsets: torch.LongTensor
    block_full_offsets: torch.LongTensor
    position_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    q_seqlens: torch.LongTensor
    kv_seqlens: torch.LongTensor
    q_start_loc: torch.LongTensor
    kv_caches: List
    is_decoding: bool
    world_size: int = 1
    local_adapter_ids: torch.LongTensor = None
    adapter_params: Dict[str, AdapterInfo] = None
    input_embeddings: torch.Tensor = None
    input_embedding_indexing: torch.Tensor = None
    vision_inputs: VisionModelInputs = None
    attn_metadata: Any = None

    _outputs: Dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        inputs: ModelInputs,
        world_size: int = 1,
        kv_caches: List = None,
    ):
        """build step context.

        Args:
            inputs (ModelInputs): packaged model inputs.
            world_size (int): The distribution world size.
            device (str): The device of the tensors.
        Examples: 
            (Pdb) q_seqlens
            tensor([708, 748, 688, 693], device='cuda:0')
            (Pdb) history_seqlens
            tensor([0, 0, 0, 0], device='cuda:0')
        """
        q_seqlens = inputs.seq_length
        history_seqlens = inputs.history_lengths
        history_ignore_lengths = inputs.history_ignore_lengths
        device = q_seqlens.device

        # for vlm
        input_embeddings, input_embedding_indexing = None, None
        if (inputs.vision_inputs is not None
                and inputs.vision_inputs.input_embeddings is not None):
            input_embeddings, input_embedding_indexing = \
                inputs.vision_inputs.get_inputs(history_seqlens, q_seqlens)

        # kv_seqlens
        if inputs.is_decoding:
            attention_mask = torch.ones_like(q_seqlens)[:, None]
            position_ids = (history_seqlens + history_ignore_lengths).unsqueeze(-1) 
        else:
            max_q_seqlen = q_seqlens.max().item()
            mask_range = torch.arange(max_q_seqlen, device=device)[None, :]
            attention_mask = (mask_range < q_seqlens[:, None]).long()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids += history_seqlens.unsqueeze(-1)
        q_start_loc = q_seqlens.cumsum(0) - q_seqlens

        # position ids 1d
        position_ids = cls.get_position_ids_1d(position_ids, q_seqlens)[None]
        # seq_len + history_length
        kv_seqlens = q_seqlens + history_seqlens
        kv_seqlens -= inputs.num_ignored_history

        ret = StepContext(
            input_ids=inputs.input_ids,
            block_offsets=inputs.block_offsets,
            block_full_offsets=inputs.block_full_offsets,
            position_ids=position_ids,
            input_embeddings=input_embeddings,
            input_embedding_indexing=input_embedding_indexing,
            attention_mask=attention_mask,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            q_start_loc=q_start_loc,
            kv_caches=kv_caches,
            is_decoding=inputs.is_decoding,
            world_size=world_size,
            local_adapter_ids=inputs.local_adapter_ids,
            vision_inputs=inputs.vision_inputs,
        )

        ret = get_backend().update_step_context(ret)
        cls.update_messages(ret, inputs)
        return ret

    @classmethod
    def update_messages(cls, ret, inputs): 
        ret.attn_metadata.messages = inputs.messages
        # import pdb; pdb.set_trace() 
        
    @classmethod
    def get_position_ids_1d(cls, position_ids: torch.LongTensor,
                            seq_length: torch.LongTensor):
        """get 1d position_ids."""
        if position_ids.size(0) == 1 or position_ids.size(1) == 1:
            position_ids_1d = position_ids.flatten()
        else:
            device = position_ids.device
            position_ids_1d = [
                ids[:l] for ids, l in zip(position_ids.cpu(), seq_length.cpu())
            ]
            position_ids_1d = torch.cat(position_ids_1d).to(device)
        return position_ids_1d


class StepContextManager:

    def __init__(self):
        self._current_ctx = None

    @staticmethod
    def build_context(
        inputs: ModelInputs,
        world_size: int = 1,
        kv_caches: List = None,
    ):
        """build context."""
        return StepContext.new(
            inputs,
            world_size,
            kv_caches,
        )

    @contextmanager
    def context(self, ctx: StepContext):
        """context context."""
        self._current_ctx = ctx
        yield ctx
        self._current_ctx = None

    def current_context(self):
        """get current_context."""
        return self._current_ctx
