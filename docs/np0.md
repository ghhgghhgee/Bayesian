# NP-0 implementation

`algorithms/np0.py` implements the noise-free NP-0 iteration in Zhou, Liu,
and Fang, *Bayesian Compressive Sensing Using Normal Product Priors*,
IEEE Signal Processing Letters, 2015, equations (10) and (12).

The model writes each signal coefficient as `x_i = a_i b_i`. NP-0 initializes
the factor means to ones and fixes the scale vectors `kappa` and `gamma` to
ones. It alternates the two factor updates and reports `x_hat = a * b`:

```text
a <- pinv(A @ diag(b)) @ y
b <- pinv(A @ diag(a)) @ y
x_hat <- a * b
```

The iteration stops when
`norm(x_new - x_old) / norm(x_old) <= eps`, or when `max_iter` is reached.
The defaults are `eps=1e-3` and `max_iter=300`, matching the paper. When the
previous estimate has zero norm, the implementation treats an unchanged zero
estimate as converged instead of evaluating `0 / 0`.

## Interface

```python
from algorithms.np0 import np0

x_hat = np0(A, y)
x_hat, info = np0(A, y, return_info=True)
```

`A` must be a finite real `M x N` matrix, and `y` a finite real vector of
length `M`. The function returns the length-`N` estimate. With
`return_info=True`, it also returns the iteration count, convergence flag, and
last relative change. NumPy is the only runtime dependency.

This module implements the noise-free limit used in the paper's numerical
experiments. It does not estimate observation-noise variance or implement the
NP-1 precision updates.

## 接入统一实验框架

`np0()` 保留上述独立接口；`solve()` 与其共用同一个迭代核心，提供框架要求的接口：

```python
from algorithms.np0 import solve

result = solve(A, y, seed=0, max_iter=300, tol=1e-3, rcond=1e-15)
```

- `tol` 对应独立接口的 `eps`。算法确定性初始化为全 1，接收 `seed` 但不使用随机数，也不修改全局随机状态。
- 返回 `bcs.algorithms.Result`，包含 `x`、`history`、`times`、`converged` 和 `status`。
- `history` 保存每轮完整 a/b 更新之后的 `x.copy()`，不包含初始化值；形状为 `(iterations, N)`，最后一行等于 `x`。
- `times` 为各轮结束时的累计秒数，使用 `perf_counter()`；耗时不参与确定性数值复跑比较。
- `status` 为 `converged` 或 `max_iter`；不因达到最大迭代数而宣称收敛。
- `Result` 仅在调用 `solve()` 时导入。独立调用 `np0()` 只需要 NumPy；运行统一框架需安装仓库 `requirements.txt` 中的锁定依赖。

注册名为 `NP0_MEMBER`，入口为 `algorithms.np0:solve`。组长 `NP0` 注册项保留；成员配置 `configs/np0_smoke.json` 使用原 smoke 的数据设置，同时运行这两个实现，不读取真实信号、不改动传入的 A/y。

### 数值参数

独立接口和框架入口均显式使用 `rcond=1e-15`，保留最初成员代码使用的 NumPy 默认伪逆截断值。框架入口允许调整 `rcond`，要求有限且在 `[0, 1)` 内；注册文件显式记录此值。组长参考实现使用 `1e-12`，接近秩亏的加权矩阵可能因此产生不同估计或迭代轨迹，不能据此要求两个实现逐位一致。论文没有给出该数值截断参数。

### 交付与验收状态

2026-10-20 已通过全部 21 项测试（其中新增 13 项 NP-0 测试），并通过组长 `main.py` 执行小规模与 3 个工况各 300 次的成员实验，两次独立进程复跑均通过。逐次指标、汇总、估计、轨迹、时间与数据哈希另经独立脚本核查，0 不一致。

本次成员证据使用 `results/np0_20261020_*`；原有 `results/validation_*` 仍仅属于组长参考实现。实际数值生成提交为 `0cf3287`，后续提交补充报告与审计工具。另一位成员的公式复核、独立机器复跑以及论文全扫描仍待完成。详细统计、命令与边界见 [测试与接入验证记录](../paper-workspace/submission/artifact-repro-checklist.md)。
