"""Flash Attention compatibility layer for ARM64 where flash_attn is unavailable.

Provides fallback implementations for the flash_attn functions that verl uses
in its training pipeline. These are functionally correct but slower than the
real flash_attn CUDA kernels.
"""
import torch
import torch.nn.functional as F


def pad_input(hidden_states, indices, batch_size, seqlen):
    """Pad hidden_states to (batch_size, seqlen, ...) based on indices."""
    output = torch.zeros(
        batch_size * seqlen,
        *hidden_states.shape[1:],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output[indices] = hidden_states
    return output.view(batch_size, seqlen, *hidden_states.shape[1:])


def unpad_input(hidden_states, attention_mask):
    """Remove padding from hidden_states based on attention_mask."""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    hidden_states_unpad = hidden_states.reshape(-1, *hidden_states.shape[2:])[indices]
    return hidden_states_unpad, indices, cu_seqlens, max_seqlen


def index_first_axis(x, indices):
    """Index the first axis of x with indices."""
    return x[indices]


def rearrange(x, pattern, **kwargs):
    """Minimal rearrange for 'b s ... -> (b s) ...' pattern."""
    if pattern == "b s ... -> (b s) ...":
        return x.reshape(-1, *x.shape[2:])
    raise NotImplementedError(f"rearrange pattern not supported: {pattern}")
