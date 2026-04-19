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


def cross_entropy_loss(logits, labels, smoothing=0.0, **kwargs):
    """Fallback cross-entropy loss using PyTorch (replaces flash_attn Triton kernel).

    Args:
        logits: (batch * seqlen, vocab_size) or (batch, seqlen, vocab_size)
        labels: (batch * seqlen,) or (batch, seqlen)
    Returns:
        (losses, z_losses) tuple where losses is per-token CE and z_losses is aux loss
    """
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
    if labels.dim() > 1:
        labels = labels.view(-1)

    losses = F.cross_entropy(logits, labels, reduction="none", label_smoothing=smoothing)
    z_losses = torch.zeros_like(losses)
    return losses, z_losses


def rearrange(x, pattern, **kwargs):
    """Minimal rearrange for 'b s ... -> (b s) ...' pattern."""
    if pattern == "b s ... -> (b s) ...":
        return x.reshape(-1, *x.shape[2:])
    raise NotImplementedError(f"rearrange pattern not supported: {pattern}")


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """Fallback flash_attn_func using PyTorch SDPA. Handles GQA.

    q: (batch, seqlen, nheads_q, headdim)
    k, v: (batch, seqlen, nheads_kv, headdim)
    Returns: (batch, seqlen, nheads_q, headdim)
    """
    nheads_q = q.shape[2]
    nheads_kv = k.shape[2]
    n_rep = nheads_q // nheads_kv

    q = q.transpose(1, 2)  # (B, Hq, S, D)
    k = k.transpose(1, 2)  # (B, Hkv, S, D)
    v = v.transpose(1, 2)

    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    out = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)  # (B, S, Hq, D)


def _repeat_kv(x, n_rep):
    """Repeat KV heads to match Q heads for GQA. x: (1, H_kv, S, D) → (1, H_q, S, D)."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
):
    """Fallback flash_attn_varlen_func using PyTorch SDPA.

    Handles GQA (grouped-query attention) where Q has more heads than K/V.
    q: (total_tokens, nheads_q, headdim)
    k, v: (total_tokens, nheads_kv, headdim)
    """
    batch_size = len(cu_seqlens_q) - 1
    nheads_q = q.shape[1]
    nheads_kv = k.shape[1]
    n_rep = nheads_q // nheads_kv

    outputs = []
    for i in range(batch_size):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]

        qi = q[start_q:end_q].unsqueeze(0).transpose(1, 2)  # (1, Hq, Sq, D)
        ki = k[start_k:end_k].unsqueeze(0).transpose(1, 2)  # (1, Hkv, Sk, D)
        vi = v[start_k:end_k].unsqueeze(0).transpose(1, 2)

        ki = _repeat_kv(ki, n_rep)  # (1, Hq, Sk, D)
        vi = _repeat_kv(vi, n_rep)

        oi = F.scaled_dot_product_attention(
            qi, ki, vi,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        outputs.append(oi.transpose(1, 2).squeeze(0))

    return torch.cat(outputs, dim=0)
