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


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """Fallback flash_attn_func using PyTorch SDPA.

    q, k, v: (batch, seqlen, nheads, headdim)
    Returns: (batch, seqlen, nheads, headdim)
    """
    q = q.transpose(1, 2)  # (B, H, S, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)  # (B, S, H, D)


def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
):
    """Fallback flash_attn_varlen_func using PyTorch SDPA with manual padding.

    q, k, v: (total_tokens, nheads, headdim) — variable-length packed tensors
    cu_seqlens_q/k: cumulative sequence lengths (batch_size+1,)
    Returns: (total_tokens, nheads, headdim)
    """
    batch_size = len(cu_seqlens_q) - 1
    nheads = q.shape[1]
    headdim = q.shape[2]

    outputs = []
    for i in range(batch_size):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]

        qi = q[start_q:end_q].unsqueeze(0).transpose(1, 2)  # (1, H, Sq, D)
        ki = k[start_k:end_k].unsqueeze(0).transpose(1, 2)
        vi = v[start_k:end_k].unsqueeze(0).transpose(1, 2)

        oi = F.scaled_dot_product_attention(
            qi, ki, vi,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        outputs.append(oi.transpose(1, 2).squeeze(0))  # (Sq, H, D)

    return torch.cat(outputs, dim=0)
