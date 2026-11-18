"""KV 缓存预算分配（/cache-budget）测试：最优性、并列规则、独立回算与 API。"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from fastapi.testclient import TestClient

import app.cache_budget as cache_budget_module
from app.cache_budget import select_plan
from app.main import app
from app.reference import compressed_reference, dense_reference
from app.schemas import CacheBudgetRequest, VerifyRequest
from app.verification import generate_inputs

client = TestClient(app)


def _brute_force(seed: int, doc_lengths: list[int], chunk_size: int, s: float, budget: int):
    """独立穷举：对每篇文档的全部保留子集做笛卡尔积，逐个用稠密参考回算。"""
    data = generate_inputs(
        VerifyRequest(seed=seed, doc_lengths=doc_lengths, chunk_size=chunk_size, s=s)
    )
    scale = s / 4.0
    n = sum(doc_lengths)
    full = dense_reference(data.q, data.k, data.v, data.doc_ids, scale)

    offsets = [0]
    for length in doc_lengths[:-1]:
        offsets.append(offsets[-1] + length)
    candidates = [
        [offsets[d] + j for j in range(doc_lengths[d] - 1)] for d in range(len(doc_lengths))
    ]

    # 注意：先物化每个文档的子集列表，避免惰性生成器捕获外层循环变量。
    per_doc = [
        [
            subset
            for k in range(len(candidates[d]) + 1)
            for subset in itertools.combinations(candidates[d], k)
        ]
        for d in range(len(doc_lengths))
    ]

    best = None
    for choice in itertools.product(*per_doc):
        retained = tuple(sorted(p for group in choice for p in group))
        if len(retained) > budget:
            continue
        keep = torch.zeros(n, dtype=torch.bool)
        if retained:
            keep[torch.tensor(retained, dtype=torch.int64)] = True
        out = compressed_reference(data.q, data.k, data.v, data.doc_ids, scale, keep)
        err = float(torch.max(torch.abs(out - full)).item())
        key = (err, len(retained), retained)
        if best is None or key < best[0]:
            best = (key, out)
    return best


@pytest.mark.parametrize(
    "doc_lengths,budget,s",
    [
        ([3, 2], 0, 1.0),
        ([3, 3], 2, 1.0),
        ([4, 3], 3, 2.0),
        ([5, 4], 5, 0.5),
        ([4, 4], 6, 10.0),
    ],
)
def test_plan_matches_brute_force(doc_lengths, budget, s):
    """搜索方案必须与独立穷举的最优解（含占用、字典序并列规则）完全一致。"""
    req = CacheBudgetRequest(
        seed=13, doc_lengths=doc_lengths, chunk_size=4, s=s, budget=budget
    )
    result = cache_budget_module.run_cache_budget(req)
    assert result.passed and result.usable_plan, result.failure_reasons

    (expected_key, expected_out) = _brute_force(13, doc_lengths, 4, s, budget)
    expected_err, expected_size, expected_positions = expected_key
    assert result.retained_positions == list(expected_positions)
    assert result.actual_occupancy == expected_size
    assert result.max_abs_err == pytest.approx(expected_err, abs=1e-12)
    assert torch.allclose(
        torch.tensor(result.compressed_output, dtype=torch.float64), expected_out, atol=1e-12, rtol=0
    )


def test_large_docs_return_complete_attention_error_free():
    """预算足以保留全部历史 key 时误差必须为 0，且只用必要的 2×(L-1) 个位置。"""
    req = CacheBudgetRequest(
        seed=5, doc_lengths=[12, 12], chunk_size=32, s=3.0, budget=22
    )
    result = cache_budget_module.run_cache_budget(req)
    assert result.passed and result.usable_plan
    assert result.max_abs_err == 0.0
    # 每篇文档 L-1 = 11 个历史 key，共 22 个；预算恰为上限。
    assert result.actual_occupancy == 22
    assert result.retained_positions == list(range(11)) + list(range(12, 23))
    # 每篇最后一个 token 永远不会作为历史 key 出现
    assert 11 not in result.retained_positions
    assert 23 not in result.retained_positions


def test_zero_budget_reads_only_self():
    """B=0 时每个 query 只读取自身，输出应等于各 token 的 V。"""
    req = CacheBudgetRequest(
        seed=9, doc_lengths=[5, 7], chunk_size=6, s=1.0, budget=0
    )
    result = cache_budget_module.run_cache_budget(req)
    assert result.passed and result.usable_plan
    assert result.retained_positions == []
    assert result.retained_by_doc == [[], []]
    assert result.actual_occupancy == 0
    data = generate_inputs(VerifyRequest(seed=9, doc_lengths=[5, 7], chunk_size=6, s=1.0))
    assert torch.allclose(torch.tensor(result.compressed_output, dtype=torch.float64), data.v, atol=1e-15)


def test_unused_budget_is_not_spent():
    """预算充足时不得多用位置：方案应与“无预算上限”的最优解一致。"""
    doc_lengths = [4, 3]
    max_budget = sum(doc_lengths) - 2
    (expected_key, _), = [_brute_force(11, doc_lengths, 8, 1.0, max_budget)]
    req = CacheBudgetRequest(
        seed=11, doc_lengths=doc_lengths, chunk_size=8, s=1.0, budget=max_budget
    )
    result = cache_budget_module.run_cache_budget(req)
    assert result.passed
    assert result.retained_positions == list(expected_key[2])
    assert result.actual_occupancy == expected_key[1]
    # 最优方案未耗尽预算（存在误差相同但占用更少的方案时）。


def test_select_plan_tie_break_rules_are_lexicographic():
    """构造并列情形，直接核对（误差, 占用, 字典序）的优先级。"""
    from app.cache_budget import DocSubsetStats

    # 文档 1：mask0 = {} 误差 0.5；mask1 = {0} 误差 0.5；
    # 文档 2：mask0 = {} 误差 0.5；mask1 = {0} 误差 0.5。
    def stats(positions, errs, sizes):
        ranks = [0] * len(positions)
        for r, m in enumerate(sorted(range(len(positions)), key=lambda i: positions[i])):
            ranks[m] = r
        return DocSubsetStats(
            err=list(errs),
            size=list(sizes),
            rank=ranks,
            positions=list(positions),
            outputs=torch.zeros(1, 1, 1),
        )

    s1 = stats([(), (0,)], [0.5, 0.5], [0, 1])
    s2 = stats([(), (0,)], [0.5, 0.5], [0, 1])
    plan, key = select_plan([s1, s2], [0, 4], budget=2)
    assert plan == (0, 0)  # 误差并列 -> 占用最少
    assert key[:2] == (0.5, 0)

    # 预算强制占用 1：两个空位方案不可达（此处用误差构造区分字典序）。
    s1 = stats([(0,), (1,)], [0.5, 0.5], [1, 1])
    s2 = stats([(0,), (1,)], [0.5, 0.5], [1, 1])
    plan, _ = select_plan([s1, s2], [0, 4], budget=2)
    assert plan == (0, 0)
    assert plan[0] != 1 and plan[1] != 1  # 全局字典序最小

    # 文档 2 的 (0,) 与 (1,) 误差相同占用相同时，字典序 (0,) 胜出。
    s1 = stats([()], [0.0], [0])
    s2 = stats([(1,), (0,)], [0.2, 0.2], [1, 1])
    plan, key = select_plan([s1, s2], [0, 4], budget=1)
    # s2 mask0 = (1,)，mask1 = (0,)；(0,) 字典序更小 -> mask1
    assert plan == (0, 1)
    assert key[2] == (4,)


def test_failure_returns_no_usable_plan(monkeypatch):
    """回算出现偏差时必须明确失败，且不返回位置/占用/输出。"""
    original = cache_budget_module.compressed_reference

    def broken(*args, **kwargs):
        out = original(*args, **kwargs)
        out = out.clone()
        out[0, 0] += 1.0
        return out

    monkeypatch.setattr(cache_budget_module, "compressed_reference", broken)
    req = CacheBudgetRequest(
        seed=3, doc_lengths=[4, 5], chunk_size=4, s=1.0, budget=2
    )
    result = cache_budget_module.run_cache_budget(req)
    assert result.passed is False
    assert result.usable_plan is False
    assert result.retained_positions is None
    assert result.retained_by_doc is None
    assert result.actual_occupancy is None
    assert result.compressed_output is None
    assert result.failure_reasons


def test_nonfinite_failure(monkeypatch):
    """回算出现 NaN 时必须明确失败。"""
    original = cache_budget_module.compressed_reference

    def nan_ref(*args, **kwargs):
        out = original(*args, **kwargs).clone()
        out[1, 1] = float("nan")
        return out

    monkeypatch.setattr(cache_budget_module, "compressed_reference", nan_ref)
    req = CacheBudgetRequest(
        seed=4, doc_lengths=[3, 4], chunk_size=2, s=1.0, budget=1
    )
    result = cache_budget_module.run_cache_budget(req)
    assert result.passed is False and result.usable_plan is False
    assert result.finite_ok is False
    assert result.retained_positions is None
    assert result.compressed_output is None


def test_cache_budget_api():
    resp = client.post(
        "/cache-budget",
        json={
            "seed": 42,
            "doc_lengths": [6, 5],
            "chunk_size": 4,
            "s": 1.0,
            "budget": 4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert body["usable_plan"] is True
    assert body["tolerance"] == 1e-8
    assert body["total_tokens"] == 11
    assert body["actual_occupancy"] == len(body["retained_positions"]) <= 4
    assert body["error_delta"] <= 1e-8
    assert body["output_delta"] <= 1e-8
    assert len(body["compressed_output"]) == 11
    assert len(body["compressed_output"][0]) == 16
    assert [p for group in body["retained_by_doc"] for p in group] == sorted(
        body["retained_positions"]
    )

    # 确定性
    again = client.post(
        "/cache-budget",
        json={
            "seed": 42,
            "doc_lengths": [6, 5],
            "chunk_size": 4,
            "s": 1.0,
            "budget": 4,
        },
    ).json()
    assert again["retained_positions"] == body["retained_positions"]
    assert again["compressed_output"] == body["compressed_output"]


def test_cache_budget_docs_and_verify_still_available():
    openapi = client.get("/openapi.json").json()
    assert "/cache-budget" in openapi["paths"]
    assert "/verify" in openapi["paths"]
    assert client.get("/docs").status_code == 200
    verify = client.post(
        "/verify?include_tensors=false",
        json={"seed": 1, "doc_lengths": [7, 11, 5], "chunk_size": 8, "s": 1.0},
    )
    assert verify.status_code == 200 and verify.json()["passed"] is True


@pytest.mark.parametrize(
    "payload",
    [
        # 只允许两篇
        {"seed": 0, "doc_lengths": [3], "chunk_size": 4, "s": 1.0, "budget": 0},
        {
            "seed": 0,
            "doc_lengths": [3, 3, 3],
            "chunk_size": 4,
            "s": 1.0,
            "budget": 0,
        },
        # 单篇长度越界
        {"seed": 0, "doc_lengths": [1, 4], "chunk_size": 4, "s": 1.0, "budget": 0},
        {"seed": 0, "doc_lengths": [13, 4], "chunk_size": 4, "s": 1.0, "budget": 0},
        # 预算越界（总长减 2 为上限）
        {"seed": 0, "doc_lengths": [4, 4], "chunk_size": 4, "s": 1.0, "budget": 7},
        {"seed": 0, "doc_lengths": [4, 4], "chunk_size": 4, "s": 1.0, "budget": -1},
        # 块长 / s / seed 越界
        {"seed": 0, "doc_lengths": [4, 4], "chunk_size": 33, "s": 1.0, "budget": 0},
        {"seed": 0, "doc_lengths": [4, 4], "chunk_size": 4, "s": 0.0, "budget": 0},
        {"seed": -1, "doc_lengths": [4, 4], "chunk_size": 4, "s": 1.0, "budget": 0},
    ],
)
def test_cache_budget_validation_errors(payload):
    resp = client.post("/cache-budget", json=payload)
    assert resp.status_code == 422


def test_error_is_max_over_all_queries_and_dims():
    """报告误差必须覆盖所有 query 与全部 16 个输出维度。"""
    req = CacheBudgetRequest(
        seed=21, doc_lengths=[7, 9], chunk_size=3, s=2.0, budget=3
    )
    result = cache_budget_module.run_cache_budget(req)
    assert result.passed
    data = generate_inputs(VerifyRequest(seed=21, doc_lengths=[7, 9], chunk_size=3, s=2.0))
    full = dense_reference(data.q, data.k, data.v, data.doc_ids, 0.5)
    keep = torch.zeros(16, dtype=torch.bool)
    keep[torch.tensor(result.retained_positions, dtype=torch.int64)] = True
    out = compressed_reference(data.q, data.k, data.v, data.doc_ids, 0.5, keep)
    err = float(torch.max(torch.abs(out - full)).item())
    assert math.isclose(result.max_abs_err, err, rel_tol=0, abs_tol=1e-15)
