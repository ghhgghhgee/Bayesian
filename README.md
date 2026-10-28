# Normal Product Priors：组长统一实验框架

对应《Bayesian Compressive Sensing Using Normal Product Priors》，Zhou Zhou、Kaihui Liu、Jun Fang，IEEE SPL 22(5), 2015，DOI 10.1109/LSP.2014.2364255。

此目录包含组长的数据生成、指标计算、算法调度、独立进程复跑比对、结果汇总和打包工具。NP0/NP1/SBL/IRLS/FOCUSS 为组长参考实现，BP 是 SciPy HiGHS 的等式约束线性规划封装。本分支另接入 NP0_MEMBER 成员实现，已通过本机测试及代表工况复跑，跨成员独立环境复核仍待完成。详见 `docs/交付与验收.md`。

## 运行

建议 Python 3.11 或 3.12。系统自带的旧 Anaconda Python 3.8 不适用锁定依赖。

数据由 `bcs/data.py` 按配置和固定种子生成，无需下载外部数据（synthetic dataset; no external data download）。`python main.py generate --config configs/np0_smoke.json` 会在忽略目录 `data/` 保存共享数据缓存及生成参数。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\reproduce.ps1 -Python .\.venv\Scripts\python.exe -Config configs/smoke.json -Tag my_smoke
```

若 PowerShell 执行策略禁止脚本，可逐条运行下列命令，不需要更改系统策略：

```powershell
python -m unittest discover -s tests -v
python main.py generate --config configs/smoke.json
python main.py run --config configs/smoke.json --out results/my_a
python main.py run --config configs/smoke.json --out results/my_b
python main.py compare results/my_a results/my_b --report results/my_reproducibility.json
python main.py plot results/my_a
python main.py package --out dist/leader_submission.zip
```

输出目录必须为空，避免覆盖历史结果。执行目录为仓库根目录。所有运行中 BLAS 线程固定为 1；相同环境中比较数值与迭代轨迹，耗时不会要求逐位相同。

| 配置 | 用途 | 试验量 |
|---|---|---|
| `configs/smoke.json` | 接口与快速复跑 | 3 个工况 × 5 次 × 6 算法 |
| `configs/validation.json` | 代表工况验收 | 3 个工况 × 300 次 × 6 算法 |
| `configs/paper.json` | Fig.3 完整扫描计划 | 59 个不同工况 × 300 次 × 6 算法 |

完整扫描需另行运行 `python main.py run --config configs/paper.json --out results/paper_a`，再用新的目录复跑、compare、plot。完整配置的存在不等于已经完成该实验。查看交付记录确定实际运行范围。

## 输出内容

- `data/*.npz`：共享的 A、y、真实 x；每个工况/试验的随机流独立于算法次序。
- `trials.csv`：逐次指标、成功与否、算法异常、停止状态、迭代数、耗时和数据 SHA256。
- `summary.csv`：成功率、平均 MSE/NMSE、平均迭代数、中位耗时。异常计入成功率分母，平均误差只统计正常返回的结果，同时显式给出 errors。
- `estimates.npz`、`histories.npz`：最终估计、各步估计和累计时间，支持复核指标及收敛轨迹。
- `manifest.json`：配置、种子、算法所有者/入口/源码哈希、环境版本、Git 状态、数据及输出哈希。
- `*_reproducibility.json`：两次独立进程运行的数值/轨迹/输入/源码/环境比对结果。任何算法异常或不一致都会退出失败。
- `success_rate.png`：当前配置包含的采样点；3 个工况的检查图不代表完整 Fig.3。
- `convergence.png`：M=30、K=3、trial=0 的单个样本；误差轴为论文定义，绘图下限 -160 dB，仅用于显示。

## 接入组员代码

详见 `docs/算法接口.md`。各模块通过 `algorithms.json` 注册，主脚本不需要修改。原始真值仅交给评估器，不传入算法；传入算法的 A/y 均为副本，防止某算法原地修改后影响他人。

## 论文口径

详见 `docs/论文实验口径.md`。特别注意：普通 MSE、NMSE 和论文的 MSE(dB) 是不同字段；BP 必须解等式约束 L1 问题，不能把带正则参数的 Lasso 当作无噪声 BP。

## NP-0 成员实现

`algorithms/np0.py` 提供无噪声 NP-0 的独立实现，接口为
`np0(A, y, eps=1e-3, max_iter=300)`，默认返回信号估计；
`return_info=True` 同时返回迭代诊断。公式、输入约定与适用范围见
[`docs/np0.md`](docs/np0.md)。

统一框架入口为 `algorithms.np0:solve`，已在 `algorithms.json` 注册为
`NP0_MEMBER`。`configs/np0_smoke.json` 同时列出组长 `NP0` 和成员
`NP0_MEMBER`，共享相同的 A/y；运行命令为：

```powershell
python main.py run --config configs/np0_smoke.json --out results/np0_member_a
```

已运行小规模配置和 `configs/np0_validation.json`（3 个工况各 300 次），
两次独立进程复跑均通过；全部 21 项测试通过。成员实现默认 `rcond=1e-15`，保留原有计算方式；
组长参考值为 `1e-12`。统一框架使用仓库锁定依赖，独立 `np0()` 模块仅依赖 NumPy。

完整命令、统计与证据见 [NP-0 测试与接入验证记录](paper-workspace/submission/artifact-repro-checklist.md)。
代表工况验证不等于完成论文 Fig.3 全扫描。

## 最新验收与材料归档

2026-10-23：理论推导稿已交付；NP1 数值修复经组长重新验收。当前状态、证据及剩余任务见 [交付与验收](docs/交付与验收.md)。

`python main.py package` 会包含 Word、PDF、PPT 交付文件及本地完整实验输出，可能较大；原论文所在 Projects、缓存和临时目录仍排除。阶段代码归档可使用 `git archive --format=zip --output=dist/team_checkpoint.zip HEAD`，只打包已提交文件，不含忽略的大型实验数据。最终全组包应在报告、PPT、正式实验齐全后生成。

## 后四项同学从这里开始

2026-10-24：成员基线已接入，见 [基线验收](docs/基线接入验收_20261024.md)。实验、报告、PPT、答辩同学请读 [后四项仓库使用指南](docs/后四项仓库使用指南.md)。正式六成员配置是 `configs/team_members/paper.json`，运行必须加 `--registry algorithms_baselines_member.json`；旧 paper_members 配置仍保留旧参考基线供历史追溯。
