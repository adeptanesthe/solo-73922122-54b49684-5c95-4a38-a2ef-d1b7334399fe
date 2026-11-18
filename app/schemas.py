"""请求/响应模型与输入校验。"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

MAX_TOTAL_TOKENS = 512
MAX_CHUNK_SIZE = 32
MIN_CHUNK_SIZE = 1
MAX_S = 1000.0
HEAD_DIM = 16
CACHE_DOC_COUNT = 2
CACHE_MIN_DOC_LENGTH = 2
CACHE_MAX_DOC_LENGTH = 12


class VerifyRequest(BaseModel):
    """/verify 请求体。"""

    seed: int = Field(..., ge=0, description="随机种子（非负整数）")
    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 512"
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        if any(not isinstance(x, int) or x <= 0 for x in value):
            raise ValueError("doc_lengths 必须为正整数列表")
        total = sum(value)
        if total == 0 or total > MAX_TOTAL_TOKENS:
            raise ValueError(
                f"文档总长度必须在 1..{MAX_TOTAL_TOKENS} 之间，当前为 {total}"
            )
        return value

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("s 必须为有限正数")
        return value


class ErrorMetrics(BaseModel):
    output_max_abs_err: float
    grad_q_max_abs_err: float
    grad_k_max_abs_err: float
    grad_v_max_abs_err: float


class VerifyResponse(BaseModel):
    passed: bool
    tolerance: float
    errors: ErrorMetrics
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
    num_docs: int
    chunk_size: int
    s: float
    seed: int
    finite_ok: bool = Field(..., description="所有输出与梯度是否均为有限值")
    failure_reasons: list[str]
    output: list[list[float]] | None = None
    grad_q: list[list[float]] | None = None
    grad_k: list[list[float]] | None = None
    grad_v: list[list[float]] | None = None


class ScenarioSummary(BaseModel):
    name: str
    description: str
    request: VerifyRequest


class CacheBudgetRequest(BaseModel):
    """/cache-budget 请求体：两篇拼接文档的 KV 缓存预算分配。"""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "seed": 42,
                    "doc_lengths": [7, 9],
                    "chunk_size": 8,
                    "s": 1.0,
                    "budget": 5,
                }
            ]
        }
    )

    seed: int = Field(..., ge=0, description="随机种子（非负整数）")
    doc_lengths: list[int] = Field(
        ...,
        min_length=CACHE_DOC_COUNT,
        max_length=CACHE_DOC_COUNT,
        description="两篇文档的长度，各为 2..12",
    )
    chunk_size: int = Field(
        ..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE, description="块长（与 /verify 一致，1..32）"
    )
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")
    budget: int = Field(
        ...,
        ge=0,
        description="可保留的历史 key 总数 B；上限为总长减 2（每篇最后一个 token 不是历史 key）",
    )

    @field_validator("doc_lengths")
    @classmethod
    def _check_cache_doc_lengths(cls, value: list[int]) -> list[int]:
        if any(not isinstance(x, int) or x <= 0 for x in value):
            raise ValueError("doc_lengths 必须为正整数列表")
        for length in value:
            if not (CACHE_MIN_DOC_LENGTH <= length <= CACHE_MAX_DOC_LENGTH):
                raise ValueError(
                    f"每篇文档长度必须在 {CACHE_MIN_DOC_LENGTH}..{CACHE_MAX_DOC_LENGTH} "
                    f"之间，当前为 {length}"
                )
        return value

    @model_validator(mode="after")
    def _check_budget(self) -> "CacheBudgetRequest":
        max_budget = sum(self.doc_lengths) - CACHE_DOC_COUNT
        if self.budget > max_budget:
            raise ValueError(
                f"budget 必须在 0..{max_budget} 之间（总长减 2），当前为 {self.budget}"
            )
        return self

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("s 必须为有限正数")
        return value


class CacheBudgetResponse(BaseModel):
    passed: bool = Field(..., description="独立回算核对是否通过")
    usable_plan: bool = Field(..., description="失败时不返回可用方案")
    tolerance: float
    seed: int
    doc_lengths: list[int]
    chunk_size: int
    s: float
    scale_used: float
    total_tokens: int
    budget: int
    retained_positions: list[int] | None = Field(
        default=None, description="保留的全局 key 位置（升序）"
    )
    retained_by_doc: list[list[int]] | None = Field(
        default=None, description="按文档拆分的保留位置"
    )
    actual_occupancy: int | None = Field(default=None, description="实际保留 key 数")
    max_abs_err: float | None = Field(
        default=None, description="回算得到的最大绝对误差（所有 query × 输出维度）"
    )
    search_max_abs_err: float | None = Field(
        default=None, description="搜索阶段报告的误差，用于与回算值交叉核对"
    )
    error_delta: float | None = Field(default=None, description="回算误差与搜索误差之差")
    output_delta: float | None = Field(
        default=None, description="独立回算输出与搜索输出最大绝对差"
    )
    finite_ok: bool
    failure_reasons: list[str]
    compressed_output: list[list[float]] | None = Field(
        default=None, description="压缩注意力输出；失败时为 null"
    )
