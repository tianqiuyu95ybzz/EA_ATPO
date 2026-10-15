"""Torch fallbacks for the subset of flash_attn.bert_padding used by ATPO."""

import torch
import torch.nn.functional as F
from einops import rearrange


def index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor[indices]


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    mask = attention_mask.reshape(-1).to(dtype=torch.bool)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    flat = rearrange(hidden_states, "b s ... -> (b s) ...")
    unpadded = index_first_axis(flat, indices)
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cumulative = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    return unpadded, indices, cumulative, max_seqlen


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch, s=seqlen)
