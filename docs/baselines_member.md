# Python 基线算法成员模块与接口说明

本模块对应分工“基线算法：SBL、FOCUSS、IRLS、BP”，将此前独立 MATLAB 实验中的四种方法移植到仓库的 Python 接口。实现是按参考文献编写的成员版本，不是论文作者提供的原版软件。

## 1. 接入方式：只新增文件

已有 `main.py`、`bcs/`、`algorithms.json`、NP0/NP1 实现、根目录说明与历史结果不修改。入口为：

| 新名字 | Python 入口 | 原有参考入口 |
|---|---|---|
| SBL_MEMBER | `algorithms.baselines_member:solve_sbl` | SBL |
| FOCUSS_MEMBER | `algorithms.baselines_member:solve_focuss` | FOCUSS |
| IRLS_MEMBER | `algorithms.baselines_member:solve_irls` | IRLS |
| BP_MEMBER | `algorithms.baselines_member:solve_bp` | BP |

新增 `algorithms_baselines_member.json` 是独立注册文件，保留原注册文件的全部条目，另外加入上面四个名字。运行旧的主脚本时显式传 `--registry algorithms_baselines_member.json` 即可。**不传这个参数时，原有实验行为保持原样。** 新配置放在 `configs/baselines_member/`，不会改变旧配置或原测试对顶层配置的约定。

使用仓库原来的 `requirements.txt`：Python 3.11/3.12、NumPy 2.2.4、SciPy 1.15.2、Matplotlib 3.10.0；无新增第三方依赖。

## 2. 直接显示，不自动保存

在仓库根目录执行：

```powershell
python -m scripts.baselines_member_display
```

默认跑小规模检查（3 个工况，各 5 次，成员基线与原有基线共 8 个算法），打印成功率并显示两个图窗；不会保存数据缓存、CSV、MAT、NPZ、日志或图片。绘图不是完整 Fig.3 的声明。窗口需要有 GUI 的本地 Python 环境。

完整基线扫描也可以只在内存计算并显示：

```powershell
python -m scripts.baselines_member_display --config configs/baselines_member/paper.json
```

完整扫描包含 59 个不同工况 × 300 次 × 4 个成员算法，即 70,800 次重建；交叉设置 (M=30,K=3) 只生成一次。运行可能耗时较长；关闭程序后内存结果不持久化。无界面检查可加 `--no-show`，仍不导出文件。

## 3. 原统一框架调用（明确需要落盘时）

下面使用仓库本来就有的 `main.py run`，**会保存结果**，与上面的显示入口区分：

```powershell
python main.py run --config configs/baselines_member/smoke.json --registry algorithms_baselines_member.json --out results/baselines_member_smoke_a
python main.py run --config configs/baselines_member/smoke.json --registry algorithms_baselines_member.json --out results/baselines_member_smoke_b
python main.py compare results/baselines_member_smoke_a results/baselines_member_smoke_b --report results/baselines_member_smoke_reproducibility.json
python main.py plot results/baselines_member_smoke_a
```

输出目录必须是新目录。`validation.json` 是 3 个代表工况、各 300 次、8 种算法的对比验收；`paper.json` 只包含 4 个成员基线的完整网格。把 `--config` 替换为对应路径即可。`main.py` 会记录估计、真实逐次轨迹、时间、数据哈希、源码哈希、配置和环境版本；compare 排除时间，检查独立进程的数值一致性。

## 4. 函数签名与返回值

```python
from algorithms.baselines_member import solve_sbl, solve_focuss, solve_irls, solve_bp
from bcs.data import generate
from bcs.metrics import metrics

A, y, truth = generate(100, 30, 3, seed=20261019, trial=0)
result = solve_sbl(A, y, seed=0, max_iter=300, tol=1e-8)
print(metrics(result.x, truth))
```

统一入口：

```python
solve(A, y, *, seed=0, max_iter=300, tol=1e-8, kind="SBL",
      convergence_tol=None, rcond=1e-12, variance_floor=1e-30, p=None,
      epsilon_start=1.0, epsilon_min=1e-8, epsilon_factor=0.1,
      lp_tolerance=1e-9, fixed_iterations=False)
```

四个 `solve_*` 适配器锁定对应 kind，其余数值选项通过关键字传入。拼错的关键字报 `TypeError`，不静默吞掉。

输入 A 为实数 M×N；y 可为 `(M,)`、`(M,1)` 或 `(1,M)`，转成 float64 的一维向量。不接受复数、非有限数、空矩阵和维数不匹配。算法不修改输入，支持只读数组，不读取真实 x 或 K。

返回仓库原有的 `bcs.algorithms.Result`：

| 字段 | 定义 |
|---|---|
| x | `(N,)` 的最终估计 |
| history | 每次完整更新后的独立数组副本，转数组后为 `(iterations,N)` |
| times | 与 history 等长的实际累计秒数，非负且单调不减 |
| converged | 是否满足迭代变化条件；与“真实信号恢复成功”不是一回事 |
| status | `converged`、`max_iter`，或 BP 的 `lp_optimal` |

`history[-1]` 与 x 数值完全相同。BP 只返回一个真实最终估计，history 长度 1，不能拿它当 LP 内部迭代次数。不可行 LP 或失败状态抛出异常，交由原框架记录错误并计入成功率分母。

seed 为框架兼容参数；这四种方法使用确定性初始化，不消耗或改变全局随机状态。随机输入仍由原来的 `bcs.data.generate` 提供，因此和 NP0/NP1 可以共享完全相同的 A、y。

## 5. 算法与参数

### SBL-EM

令 Gamma=diag(gamma)，C=A Gamma Aᵀ。后验均值 mu=Gamma Aᵀ C†y，后验协方差 Sigma=Gamma-Gamma Aᵀ C†A Gamma。每轮更新 gamma_i=mu_i²+Sigma_ii。gamma 初始化为 1，默认不学习噪声、不剪枝、不估计真实支撑。

### 基础 FOCUSS

最小二范数解初始化。Q=diag(|x_i|^(2-p))，x_new=Q Aᵀ(A Q Aᵀ)†y；默认 p=0。基础非凸更新可能收敛到错误支撑，不能只凭测量残差很小判定恢复成功。

### 平滑 IRLS

默认 p=0.5；Q 对角元是 `(x_i²+epsilon)^(1-p/2)`。epsilon 从 1 开始；只有当相邻估计的相对变化 `<sqrt(epsilon)/100` 时才乘 0.1，最小为 1e-8。在达到最后平滑层之前不允许按 tol 提前结束。epsilon 是直接加在 x² 上，**没有平方 epsilon**。

这和旧参考 IRLS 中 `epsilon=0.5**step`、权重使用 `x²+epsilon²` 的局部版本不同；旧实现不改，两个版本用不同名字独立比较。不能要求二者逐位一致，也不能宣称任一版本就是原文未提供的作者代码。

### BP

用 x=u-v、u/v>=0 把 `min ||x||₁, Ax=y` 变成等式约束 LP，调用 SciPy 官方 `linprog(method="highs")`。可行性／对偶可行性容差为 1e-9，失败抛异常。未改成 LASSO 或 BPDN。

### 数值方案与停止

所有迭代方法对 B=A sqrt(Q) 做 SVD，不显式形成 Gram 逆；截断为 `s > rcond*s_max`，默认 rcond=1e-12，方差底限 1e-30。全部 N 列保留。最后检查相对测量残差 <=1e-7，违反无噪声约束时报错。这个检查不使用真实 x。

最大迭代次数默认 300。新增注册文件对三种迭代基线明确设置 `convergence_tol=1e-8`，覆盖主配置通用 `tol=1e-3`；这是为了和先前基线设置一致，并避免迭代变化小被误当成真实误差小。BP 的 LP 内部迭代数不等于 max_iter，其 history 始终只包含最终点。`fixed_iterations=True` 可用于固定预算诊断，正常验收不开启；零观测直接返回零向量。

原文只明确给出 NP-0/NP-1 的停止阈值，没有披露基线的全部调参；这里的 p、epsilon、停止阈值和数值截断均是明确报告的复现选择。

## 6. 数据、指标和 MATLAB 差别

完整采用仓库的 N=100、标准正态 A、无列归一化、无噪声、均匀随机支撑和标准正态非零幅度；成功为相对二范数误差严格 `<1e-3`。普通 MSE、NMSE、论文 dB 由原 `bcs.metrics.metrics` 计算，论文 dB 为 10log10(NMSE)。

Python 与先前 MATLAB 的随机数生成器、样本种子组织和伪逆实现不同，不能把相同数字的种子视为相同 A、x。MATLAB 曾使用 60 个扫描设置；这里遵循仓库去重后的 59 个工况。Python 结果需实际重新运行，不能沿用 MATLAB 的成功率。

显示入口的收敛图遵循仓库已有约定：M=30,K=3,trial=0 的单样本，使用真实累计耗时和 history；低于 -160 dB 只在图中截断，原指标不改。BP 是单点。绝对耗时不用于对照 2015 年旧硬件或不同语言平台。

## 7. 测试与验收

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p test_baselines_member.py -v
python -m scripts.baselines_member_display --no-show
```

新增测试核对独立条件高斯公式、连续三轮 EM、FOCUSS 和 IRLS 的首步权重、BP 已知最优解与异常传播、秩亏／不可行系统、参数拒绝、只读输入与随机状态、实际历史和停止状态，以及原 main.py 的双进程 run/compare/plot。验证记录见 `docs/baselines_member_validation.md`；未运行的完整扫描不能仅凭配置存在就宣称完成。

## 8. 来源

- Zhou, Liu & Fang (2015), *Bayesian Compressive Sensing Using Normal Product Priors*, [DOI](https://doi.org/10.1109/LSP.2014.2364255)。
- Wipf & Rao (2004), *Sparse Bayesian Learning for Basis Selection*, [作者论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/01315936.pdf)。SBL-EM 与无噪声后验。
- Gorodnitsky & Rao (1997), *Sparse Signal Reconstruction from Limited Data Using FOCUSS*, [DOI](https://doi.org/10.1109/78.558475)。
- Chartrand & Yin (2008), *Iteratively Reweighted Algorithms for Compressive Sensing*, [Rice 原文](https://repository.rice.edu/server/api/core/bitstreams/4256317a-eaaf-4b14-94c0-43f0f6153559/content)。IRLS 平滑项和收敛触发递减。
- [SciPy 1.15.2 linprog / HiGHS 官方文档](https://docs.scipy.org/doc/scipy-1.15.2/reference/optimize.linprog-highs.html)。
