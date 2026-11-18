"""KV 缓存预算分配与独立回算。

问题
----
把两篇文档（各 2..12 token）拼入同一批次。每个 query 始终读取自身；除此以外
只能读取**本篇中位于自己之前、且被保留在缓存中**的 key；删除 key 后 softmax
在剩余 key 上重新归一化。每篇文档的最后一个 token 不可能是任何 query 的历史
key，因此可保留的历史 key 恰好是总长减 2 个位置。给定预算 ``B``，求实际占用
不超过 ``B`` 的最优保留位置。

目标按字典序：
1. 全部 query × 全部输出维度上相对完整注意力的最大绝对误差最小；
2. 误差并列时，实际占用 key 数更少者优先；
3. 再并列时，保留位置（升序全局下标）字典序更小者优先。

算法
----
每篇文档可选历史位置至多 11 个，故对单篇穷举全部 ``2^(L-1)`` 个保留子集，
批量计算该文档所有 query 的压缩输出与误差（O(M·L²·d)，M ≤ 2048）。组合两篇
文档时，对文档 2 按（误差, 占用, 字典序）排序并做前缀最优 + 二分，避免
``2^22`` 级别的双重枚举：对每个文档 1 子集，只需一次候选查询即可得到其最优
文档 2 子集，结果与全量枚举严格等价（已用随机化交叉验证）。

回算
----
搜索阶段使用“按子集批量掩码”的独立向量化路径。选定方案后，另用
``reference.compressed_reference`` 从原始 Q/K/V 重新构造完整 N×N 掩码并计算
压缩输出，重新求得误差；若与搜索报告值之差、或两路输出之差超过 1e-8，或出现
任何非有限值，则明确失败且不返回可用方案。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import torch

from .attention import NEG_INF
from .reference import compressed_reference, dense_reference
from .schemas import CacheBudgetRequest, CacheBudgetResponse, VerifyRequest
from .verification import TOLERANCE, generate_inputs


@dataclass
class DocSubsetStats:
    """单篇文档上全部保留子集的统计量（子集按 bitmask 编号）。"""

    err: list[float]  # 该文档所有 query 的最大绝对误差
    size: list[int]  # 子集占用数
    rank: list[int]  # 升序位置元组的字典序排名
    positions: list[tuple[int, ...]]  # 文档内局部位置（升序）
    outputs: torch.Tensor  # (M, L, d) 各子集下该文档全部 query 的压缩输出


def _enumerate_doc_subsets(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    full_out: torch.Tensor,
    scale: float,
) -> DocSubsetStats:
    """穷举单篇文档的保留子集，批量计算压缩输出与误差。

    bit j（0 ≤ j < L-1）表示文档内局部位置 j 被保留。位置 L-1 是文档最后一个
    token，不会成为历史 key，故不进入枚举。每个 query 的自身位置始终可见。
    """
    length = q.shape[0]
    hist = length - 1  # 可保留的历史位置 0..L-2
    num_masks = 1 << hist
    device = q.device

    masks = torch.arange(num_masks, dtype=torch.int64, device=device)
    bit_ids = torch.arange(hist, dtype=torch.int64, device=device)
    bits = ((masks[:, None] >> bit_ids[None, :]) & 1).bool()  # (M, hist)

    idx = torch.arange(length, device=device)
    strictly_earlier = idx[None, :] < idx[:, None]  # [t, j] -> j < t
    history_allowed = strictly_earlier[:, :hist][None, :, :] & bits[:, None, :]
    allowed = torch.zeros(
        (num_masks, length, length), dtype=torch.bool, device=device
    )
    allowed[:, :, :hist] = history_allowed
    allowed |= torch.eye(length, dtype=torch.bool, device=device)[None, :, :]

    scores = scale * (q @ k.T)  # (L, L)，与子集无关
    masked = torch.where(allowed, scores[None, :, :], NEG_INF)
    # 自身始终可见，故每行至少有一个有限分数，手写 softmax 不会出现空行。
    row_max = torch.amax(masked, dim=2, keepdim=True)
    weights = torch.where(allowed, torch.exp(masked - row_max), 0.0)
    probs = weights / torch.sum(weights, dim=2, keepdim=True)
    outputs = probs @ v  # (M, L, d)

    err = torch.amax(torch.abs(outputs - full_out[None, :, :]), dim=(1, 2))

    positions = [
        tuple(j for j in range(hist) if (mask >> j) & 1) for mask in range(num_masks)
    ]
    rank = [0] * num_masks
    for r, mask in enumerate(sorted(range(num_masks), key=lambda m: positions[m])):
        rank[mask] = r
    return DocSubsetStats(
        err=[float(x) for x in err.tolist()],
        size=[int(x) for x in bits.sum(dim=1).tolist()],
        rank=rank,
        positions=positions,
        outputs=outputs,
    )


def select_plan(
    stats: list[DocSubsetStats],
    doc_offsets: list[int],
    budget: int,
) -> tuple[tuple[int, int], tuple[float, int, tuple[int, ...]]]:
    """在预算约束下按（误差, 占用, 保留位置字典序）选最优子集对。

    误差为两篇文档所有 query 的最大绝对误差。并列时占用少者优先；仍并列时
    比较合并后的升序全局位置元组（标准字典序，短元组为前缀时更小）。

    返回 ((mask1, mask2), (误差, 总占用, 合并位置元组))。
    纯函数，便于独立测试并列规则。
    """
    err1, size1 = stats[0].err, stats[0].size
    err2, size2, rank2 = stats[1].err, stats[1].size, stats[1].rank
    pos1, pos2 = stats[0].positions, stats[1].positions
    offset2 = doc_offsets[1]  # 文档 2 的全局位置偏移

    # 对文档 2 的每个占用上限 cap，预计算占用 ≤ cap 的子集：
    #   errs_sorted —— 按误差升序的误差序列，用于二分
    #   prefix_best —— 前 i+1 个元素中 (占用, 字典序) 最小者
    max_size2 = max(size2)
    cap_tables: dict[int, tuple[list[int], list[float], list[int]]] = {}
    for cap in range(min(budget, max_size2) + 1):
        order = sorted(
            (m for m in range(len(err2)) if size2[m] <= cap),
            key=lambda m: (err2[m], size2[m], rank2[m]),
        )
        errs_sorted = [err2[m] for m in order]
        prefix_best: list[int] = []
        best: int | None = None
        for m in order:
            if best is None or (size2[m], rank2[m]) < (size2[best], rank2[best]):
                best = m
            prefix_best.append(best)
        cap_tables[cap] = (order, errs_sorted, prefix_best)

    best_key: tuple[float, int, tuple[int, ...]] | None = None
    best_plan: tuple[int, int] | None = None

    for m1 in range(len(err1)):
        remaining = budget - size1[m1]
        if remaining < 0:
            continue
        order, errs_sorted, prefix_best = cap_tables[min(remaining, max_size2)]
        e1 = err1[m1]

        # 固定 m1 时 max(e1, err2) 关于 err2 单调，故只需文档 2 误差最小的一档：
        #   * 存在 err2 ≤ e1：总误差恒为 e1，在整档中取 (占用, 字典序) 最小者；
        #   * 否则取误差最小的 order[0]（同误差内 (占用, 字典序) 已排序）。
        cut = bisect.bisect_right(errs_sorted, e1) - 1
        m2 = prefix_best[cut] if cut >= 0 else order[0]

        merged = tuple(pos1[m1]) + tuple(offset2 + p for p in pos2[m2])
        key = (max(e1, err2[m2]), size1[m1] + size2[m2], merged)
        if best_key is None or key < best_key:
            best_key = key
            best_plan = (m1, m2)

    if best_plan is None or best_key is None:
        raise ValueError("预算内不存在可行方案")
    return best_plan, best_key


def _failure(
    reasons: list[str],
    req: CacheBudgetRequest,
    scale: float,
    finite_ok: bool,
    search_err: float | None = None,
    error_delta: float | None = None,
    output_delta: float | None = None,
) -> CacheBudgetResponse:
    """构造“明确失败”的响应：不带保留位置、占用与压缩输出。"""
    return CacheBudgetResponse(
        passed=False,
        usable_plan=False,
        tolerance=TOLERANCE,
        seed=req.seed,
        doc_lengths=list(req.doc_lengths),
        chunk_size=req.chunk_size,
        s=req.s,
        scale_used=scale,
        total_tokens=sum(req.doc_lengths),
        budget=req.budget,
        retained_positions=None,
        retained_by_doc=None,
        actual_occupancy=None,
        max_abs_err=None,
        search_max_abs_err=search_err,
        error_delta=error_delta,
        output_delta=output_delta,
        finite_ok=finite_ok,
        failure_reasons=reasons,
        compressed_output=None,
    )


def run_cache_budget(req: CacheBudgetRequest) -> CacheBudgetResponse:
    """搜索最优保留位置，并从原始输入独立回算核对。"""
    base = VerifyRequest(
        seed=req.seed,
        doc_lengths=list(req.doc_lengths),
        chunk_size=req.chunk_size,
        s=req.s,
    )
    data = generate_inputs(base)
    scale = float(req.s) / 4.0
    n = sum(req.doc_lengths)

    with torch.no_grad():
        full_out = dense_reference(data.q, data.k, data.v, data.doc_ids, scale)

        stats: list[DocSubsetStats] = []
        offset = 0
        for length in req.doc_lengths:
            sl = slice(offset, offset + length)
            stats.append(
                _enumerate_doc_subsets(
                    data.q[sl], data.k[sl], data.v[sl], full_out[sl], scale
                )
            )
            offset += length

    plan, key = select_plan(stats, [0, req.doc_lengths[0]], req.budget)
    search_err, occupancy = float(key[0]), int(key[1])

    offsets = [0, req.doc_lengths[0]]
    retained_by_doc = [
        sorted(offsets[d] + p for p in stats[d].positions[mask])
        for d, mask in enumerate(plan)
    ]
    retained = sorted(p for group in retained_by_doc for p in group)

    # ---- 从原始输入独立回算：完整 N×N 掩码 + 手写 softmax ----
    keep = torch.zeros(n, dtype=torch.bool)
    if retained:
        keep[torch.tensor(retained, dtype=torch.int64)] = True
    with torch.no_grad():
        recomputed = compressed_reference(
            data.q, data.k, data.v, data.doc_ids, scale, keep
        )
        recomputed_err = float(torch.max(torch.abs(recomputed - full_out)).item())
        optimizer_out = torch.cat(
            [stats[d].outputs[mask] for d, mask in enumerate(plan)], dim=0
        )
        output_delta = float(
            torch.max(torch.abs(recomputed - optimizer_out)).item()
        )
    error_delta = abs(recomputed_err - search_err)

    finite_ok = bool(
        torch.isfinite(full_out).all().item()
        and torch.isfinite(recomputed).all().item()
        and torch.isfinite(optimizer_out).all().item()
        and math.isfinite(search_err)
        and math.isfinite(recomputed_err)
        and math.isfinite(output_delta)
        and math.isfinite(error_delta)
    )

    failure_reasons: list[str] = []
    if not finite_ok:
        failure_reasons.append("完整注意力或压缩注意力中出现 NaN/Inf")
    if occupancy > req.budget:
        failure_reasons.append(
            f"实际占用 {occupancy} 超过预算 {req.budget}"
        )
    if occupancy != len(retained):
        failure_reasons.append("占用计数与保留位置数量不一致")
    if any(not (0 <= p < n) for p in retained):
        failure_reasons.append("保留位置越界")
    # 每篇文档最后一个 token 不是历史 key，不应被保留。
    last_positions = {
        sum(req.doc_lengths[:d]) + req.doc_lengths[d] - 1 for d in range(len(req.doc_lengths))
    }
    if any(p in last_positions for p in retained):
        failure_reasons.append("保留位置包含文档最后一个 token")
    if error_delta > TOLERANCE:
        failure_reasons.append(
            f"回算误差 {recomputed_err:.3e} 与搜索误差 {search_err:.3e} "
            f"相差 {error_delta:.3e}，超过容差 {TOLERANCE:.0e}"
        )
    if output_delta > TOLERANCE:
        failure_reasons.append(
            f"独立回算输出与搜索输出最大绝对差 {output_delta:.3e} "
            f"超过容差 {TOLERANCE:.0e}"
        )

    if failure_reasons:
        return _failure(
            failure_reasons,
            req,
            scale,
            finite_ok=finite_ok,
            search_err=search_err if math.isfinite(search_err) else None,
            error_delta=error_delta if math.isfinite(error_delta) else None,
            output_delta=output_delta if math.isfinite(output_delta) else None,
        )

    return CacheBudgetResponse(
        passed=True,
        usable_plan=True,
        tolerance=TOLERANCE,
        seed=req.seed,
        doc_lengths=list(req.doc_lengths),
        chunk_size=req.chunk_size,
        s=req.s,
        scale_used=scale,
        total_tokens=n,
        budget=req.budget,
        retained_positions=retained,
        retained_by_doc=retained_by_doc,
        actual_occupancy=occupancy,
        max_abs_err=recomputed_err,
        search_max_abs_err=search_err,
        error_delta=error_delta,
        output_delta=output_delta,
        finite_ok=True,
        failure_reasons=[],
        compressed_output=recomputed.tolist(),
    )
