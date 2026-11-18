"""独立稠密参考实现（不受 O(N + C^2) 工作张量限制）。

与分块算子完全独立：手工构造完整 N x N 掩码与注意力矩阵，
不使用任何现成注意力算子，仅使用基础张量运算。
用于核对分块实现的前向输出与反向梯度。
"""

from __future__ import annotations

import torch

from .attention import HEAD_DIM, NEG_INF  # 仅共享常量，不共享算法代码


def dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """稠密文档内因果注意力，返回带 autograd 历史的输出。"""
    n = q.shape[0]
    pos = torch.arange(n, dtype=torch.int64)
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    causal = pos[None, :] <= pos[:, None]
    valid = same_doc & causal

    scores = scale * (q @ k.T)
    scores = torch.where(valid, scores, NEG_INF)

    # 手工 softmax：行最大值平移后指数归一化（不调用 torch.softmax，
    # 保持“独立参考”的完整可审计性）。每行至少有自身一个有效位置。
    row_max = torch.amax(scores, dim=1, keepdim=True)
    exp_scores = torch.exp(scores - row_max)
    exp_scores = torch.where(valid, exp_scores, 0.0)
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
    return probs @ v


def dense_reference_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    scale: float,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """稠密参考的前向输出与 Q/K/V 梯度（借助 autograd 反传）。"""
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = dense_reference(q_ref, k_ref, v_ref, doc_ids, scale)
    gq, gk, gv = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), grad_outputs=grad_out
    )
    return out_ref.detach(), gq, gk, gv


def compressed_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    scale: float,
    keep: torch.Tensor,
) -> torch.Tensor:
    """KV 缓存压缩后的文档内因果注意力，返回不带梯度的输出。

    ``keep[j] = True`` 表示历史 key j 保留在缓存中。每个 query 始终读取自身；
    其余只能读取同一篇文档中、不晚于自己、且被保留的 key。删除 key 后 softmax
    在剩余 key 上重新归一化。使用完整 N x N 掩码与手写 softmax，作为独立回算。
    """
    n = q.shape[0]
    if keep.shape != (n,):
        raise ValueError("keep 必须是一维布尔张量且长度等于 token 总数")
    keep = keep.to(dtype=torch.bool)

    pos = torch.arange(n, dtype=torch.int64)
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    causal = pos[None, :] <= pos[:, None]
    self_key = pos[None, :] == pos[:, None]
    valid = same_doc & causal & (self_key | keep[None, :])

    scores = scale * (q @ k.T)
    scores = torch.where(valid, scores, NEG_INF)
    row_max = torch.amax(scores, dim=1, keepdim=True)
    exp_scores = torch.where(valid, torch.exp(scores - row_max), 0.0)
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
    return probs @ v


__all__ = [
    "HEAD_DIM",
    "compressed_reference",
    "dense_reference",
    "dense_reference_grads",
]
