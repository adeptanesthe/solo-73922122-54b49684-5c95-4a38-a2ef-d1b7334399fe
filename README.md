# 分块文档内因果注意力核对 API

把多篇文档拼入同一批次（总长 N ≤ 512），在 **CPU + float64** 下核对一个
自定义分块注意力 autograd 算子与独立稠密参考实现的前向输出及 Q/K/V 梯度。

- 每个 token 只能读取**同一篇文档中位置不晚于自己**的 token
- 输出：`O = softmax(s · Q Kᵀ / 4) · V`
- Q/K/V 与上游梯度均为 16 维，float64
- 自定义算子前向、后向**均分块**计算，块长 `chunk_size ∈ [1, 32]`
- 不调用任何框架内置注意力算子，也不依赖内置 softmax 原语（online softmax 手写）
- 工作张量与跨前后向保存量规模 **O(N + C²)**，C 为块长；独立稠密参考不受此限制
- 容差：输出与 Q/K/V 梯度的最大绝对误差均 **≤ 1e-8**，否则 `passed=false` 并给出原因
- 另有 `/cache-budget`：在保留 key 数不超过 B 的前提下，为两篇拼接文档选择
  使压缩输出误差最小的保留位置，并从原始输入独立回算核对

## 目录结构

```
app/
  attention.py     # 自定义分块 autograd 算子（online softmax 前向 + 分块反向）
  reference.py     # 独立稠密参考（完整 N×N 掩码与注意力矩阵）+ 压缩（删 key）参考
  verification.py  # 数据生成、双路计算与误差核对
  cache_budget.py  # KV 缓存预算分配：子集穷举搜索 + 独立回算
  schemas.py       # 请求/响应模型与输入校验
  scenarios.py     # 内置边界情形样例
  main.py          # FastAPI 入口
tests/            # pytest：正确性、gradcheck、边界情形、内存与源码审计、API
Dockerfile
docker-compose.yml
```

## 启动（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000
# Swagger 文档：http://localhost:8000/docs
# OpenAPI：     http://localhost:8000/openapi.json
```

## 本地（无 Docker）运行

```bash
pip install -r requirements.txt   # torch 使用 CPU wheel 源
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q                         # 运行自测
```

## 接口

### `POST /verify`

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0，决定 Q/K/V 与上游梯度的确定性生成 |
| `doc_lengths` | int[] | 正整数列表，总长 1..512 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 1000 |

查询参数 `include_tensors`（默认 `true`）：是否回传完整输出/梯度张量。

curl 示例：

```bash
curl -X POST "http://localhost:8000/verify?include_tensors=false" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,11,5],"chunk_size":8,"s":1.0}'
```

响应关键字段：`passed`、`tolerance=1e-8`、`errors`（四项最大绝对误差）、
`finite_ok`、`failure_reasons`。核对失败是业务结果而非 HTTP 错误，仍返回 200，
通过 `passed=false` 与 `failure_reasons` 明确表达；参数非法返回 422。

### `GET /scenarios` / `POST /scenarios/run`

列出或运行内置边界情形；`POST /scenarios/run?name=<名称>` 只跑单个样例。

### `POST /cache-budget`

两篇拼接文档的 KV 缓存预算分配。沿用 `/verify` 的 `seed`、`chunk_size`、`s`
生成 float64 Q/K/V（16 维），每篇文档长度 2..12。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0 |
| `doc_lengths` | int[] | 恰好两篇，每篇 2..12 |
| `chunk_size` | int | 1..32（与 `/verify` 相同，不影响压缩语义） |
| `s` | float | 0 < s ≤ 1000 |
| `budget` | int | 0..总长减 2，可保留的历史 key 总数 B |

规则：每个 query 始终读取自己；其余只能读取**本篇中位于自己之前、且被保留**
的 key；删除 key 后 softmax 在剩余 key 上重新归一化。每篇文档的最后一个 token
不可能是任何 query 的历史 key，故可保留的历史位置共 `总长 - 2` 个。

目标按字典序：

1. 全部 query × 全部 16 个输出维度上，相对完整注意力的最大绝对误差最小；
2. 误差并列时，实际占用 key 数更少者优先；
3. 再并列时，升序保留位置元组字典序更小者优先。

“并列”按 float64 误差**逐位相等**判定（例如保留全部历史 key 时误差恰为 0.0）；
仅数值相近但不相等的方案仍以误差更小者为准。

```bash
curl -X POST "http://localhost:8000/cache-budget" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,9],"chunk_size":8,"s":1.0,"budget":5}'
```

响应关键字段：`retained_positions`（全局升序）、`retained_by_doc`、
`actual_occupancy`、`max_abs_err`、`compressed_output`（N×16）。

服务会从原始 Q/K/V 用独立的完整 N×N 掩码路径**重新回算**所选方案：
`max_abs_err` 即回算误差，`search_max_abs_err` 是搜索阶段报告值，
`error_delta` / `output_delta` 是两路差值。任一路出现非有限值，或两项差值
超过 `tolerance=1e-8` 时，返回 200 但 `passed=false`、`usable_plan=false`，
`retained_positions` / `actual_occupancy` / `compressed_output` 均为 `null`，
并在 `failure_reasons` 中说明原因。参数非法（长度、篇数、预算越界等）返回 422。

## 内置边界情形（题目要求）

| 名称 | 配置 | 覆盖点 |
| --- | --- | --- |
| `doc_cross_chunk` | 文档 7/11/5，块长 8，s=1 | 文档边界落在块内部，切块不改变归一化 |
| `fully_masked_block` | 单文档 16，块长 4，s=2.5 | 整块被因果屏蔽时安全跳过、无 NaN |
| `large_score` | 文档 33/31，块长 32，s=1000 | 大分数下 online softmax 前后向无溢出 |
| `combined_stress` | 文档 100×4+112（N=512），块长 7，s=999 | 三类情形叠加 + 块长不整除 |
| `minimal_single_token` | 两篇各 1 token，块长 32 | 最小批次与退化分块循环 |

```bash
curl -X POST "http://localhost:8000/scenarios/run"
```

## 实现要点

1. **掩码**：`valid(i,j) = (doc_i == doc_j) ∧ (pos_j ≤ pos_i)`，按块即时构造 C×C 布尔掩码。
2. **前向 online softmax**：逐 key 块维护每行的运行最大值 `m`、归一化分母 `d`
   与归一化输出 `O`。遇到新块时以 `exp(m_old − m_new)` 修正旧累积量，
   全程只做最大值平移，不保存 N×N 矩阵。
3. **整块屏蔽**：块内无有效位置时直接跳过；行级 `-inf` 通过 `isfinite` 判定，
   以 0 参与指数运算，杜绝 `−inf − −inf`、`0/0` 产生的 NaN。
4. **后向**：用保存的行最大值/分母在每个块上**重算**块内概率 P（仅 C×C），
   利用 `dP = P ⊙ (dO·Vᵀ − Σ_j P_ij dO_iᵀV_j)` 与 `dS = scale·dP` 分块累加
   `dQ/dK/dV`；保存量仅为 Q/K/V、doc_ids、行最大值、行分母、输出，均为 O(N)。
5. **数值稳定**：float64 全程；s=1000 时原始分数很大，最大值平移保证
   `exp` 输入 ≤ 0，前后向均为有限值。
6. **确定性**：同一 `seed` 生成完全相同的输入与结果。

### `/cache-budget` 的搜索与回算

1. **单篇穷举**：每篇可选历史位置至多 11 个，直接枚举全部 `2^(L-1)` 个子集
   （≤ 2048），按子集批量构造掩码并计算该篇所有 query 的压缩输出与误差。
2. **跨篇组合**：对文档 2 按占用上限预排序（误差, 占用, 字典序）并做前缀最优
   + 二分；对每个文档 1 子集一次查询即得其最优文档 2 子集。结果与
   `2^22` 级全量组合严格等价。
3. **独立回算**：选定方案后，用 `reference.compressed_reference` 从原始 Q/K/V
   重新构造完整 N×N 掩码、手写 softmax 计算压缩输出，重新求误差并与搜索值
   交叉核对；任何偏差 > 1e-8 或非有限值都会使请求明确失败。

## 自测结果

`pytest -q` 共 50 项全部通过，包含：

- 原 `/verify` 的 27 项：11 组参数化前向+梯度核对（误差均 ≤ 1e-8，实测在
  1e-12 ~ 1e-16 量级）、`torch.autograd.gradcheck`、块长不变性、确定性、
  掩码语义、大分数有限性、保存张量线性规模审计、禁用算子源码审计、API 与校验。
- `/cache-budget` 的 23 项：与独立穷举逐位对齐的 5 组最优解、零预算只读自身、
  预算不浪费、并列规则（误差 → 占用 → 字典序）、L=12 全子集枚举正确性、
  回算偏差/NaN 时明确失败的注入测试、API 响应结构与 9 组参数越界校验。

另外用随机化脚本在 60 组（长度 2..6、预算 0..总长-2、s ∈ {0.5,1,2,7.5,40}、
块长 1..32）上把搜索方案与全量笛卡尔积穷举逐项比较，位置、占用、误差与输出
全部一致；L=12 时单篇 2048 子集与 `2048×2048` 组合也做了全量交叉验证。
