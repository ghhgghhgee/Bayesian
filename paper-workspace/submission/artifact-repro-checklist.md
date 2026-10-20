# NP-0 测试与框架接入验证

日期：2026-10-20。范围：确认成员 NP-0 能通过组长的注册、调度、指标、轨迹、复跑比较和绘图流程。数值结果来自提交 `0cf3287b58afcf24573083bdec69acef7a3e9943`；本报告和独立审计工具在后续提交补充。源代码在两次复跑之间保持一致，manifest 中的未跟踪项仅为先前生成的结果目录及复跑比较 JSON，无源码改动。

## 结论与检查清单

| 项目 | 结果 | 证据 |
|---|---|---|
| 测试 | 21/21 通过，其中 13 项为 NP-0 新测试 | `results/np0_20261020_tests.txt` |
| 实际注册调用 | `algorithms.np0:solve` 经 `main.py run` 调用，无异常 | 四个新运行目录中的 manifest/trials |
| 小规模复跑 | 3 工况 × 5 次 × 2 算法，独立运行两次，9 项比较均通过 | `results/np0_20261020_smoke_reproducibility.json` |
| 代表工况复跑 | 3 工况 × 300 次 × 2 算法，独立运行两次，9 项比较均通过 | `results/np0_20261020_validation_reproducibility.json` |
| 独立重算 | 审计 3660 条运行记录，0 不一致 | `results/np0_20261020_numerical_audit.json` |
| 绘图 | 成功率图及 NP0/NP0_MEMBER 的真实收敛轨迹已生成并查看 | `results/np0_20261020_validation_a/*.png` |

测试覆盖原接口、全 1 初始化的直接矩阵公式参照、输入校验、零观测、输入及随机状态不变、迭代上限、Result/真实 history/times、实际 CLI 注册调用与绘图。发现并修复了旧 CLI 测试以系统默认 GBK 读取 UTF-8 JSON 的问题；绘图入口从固定算法名改为遍历配置，避免漏画成员算法。核心 NP-0 更新公式未改动。

## 环境与数据

Python 3.11.15、NumPy 2.2.4、SciPy 1.15.2、Matplotlib 3.10.0；Windows x64。虚拟环境为仓库 `.venv`，没有替换系统 Python。直接依赖沿用 `requirements.txt`；完整实际依赖版本保存在 `results/np0_20261020_requirements.txt`，各 manifest 同时记录解释器、平台和源码哈希。三个 BLAS/OMP 线程数均由主脚本固定为 1。

`bcs/data.py` 生成无噪声合成数据，不需要下载外部数据。N=100，种子 20261019，A 为标准高斯矩阵，非零系数为高斯随机数；每次试验的 A/y/真实 x 共享给两个算法。真实 x 仅用于评估。最大迭代 300，停止容差 1e-3，成功标准为相对 L2 恢复误差严格小于 1e-3。

## 300 次代表工况结果

下表为成员实现首轮结果；第二轮的全部数值指标和轨迹一致。每行是 300 个独立样本，第二轮复跑相同样本，不增加独立样本数。

四目录合计 3660 条算法调用记录，其中成员与参考各 1830 次；按数据哈希去重共 900 个独立输入。smoke 的 15 个输入完全包含在 validation 的 900 个输入中，因此 smoke 和第二轮复跑均不额外增加独立样本。

| M | K | 成功数/300 | 成功率 | 平均 MSE | 平均 NMSE | 平均迭代数 | 异常数 |
|---|---|---|---|---|---|---|---|
| 10 | 3 | 18 | 6% | 1.4183846222e-2 | 4.4997587915e-1 | 33.72 | 0 |
| 30 | 3 | 300 | 100% | 9.0415448292e-11 | 2.7275094144e-9 | 4.7766666667 | 0 |
| 30 | 15 | 0 | 0% | 5.0529315189e-2 | 3.4053062809e-1 | 32.9566666667 | 0 |

本次所有样本中成员估计与组长 NP0 参考估计完全一致，虽然注册的伪逆截断值分别为 1e-15 和 1e-12。该观察不意味着其他矩阵条件下也逐位一致。900 个成员样本均达到迭代停止条件，但停止收敛不等于恢复真实信号成功；两个困难工况的低成功率与参考实现一致。最大测量相对残差约 1.19e-14。

独立审计脚本不调用 `bcs.metrics`，从保存的估计和重建真值直接计算 MSE、NMSE、相对误差和成功率，核对逐次表及汇总；另核对共享输入哈希、history/times、停止状态和输出 SHA256。独立审计仍使用仓库数据生成器重建数据，不属于另一团队或独立机器的复现。

## 实际执行命令与重跑

在仓库根目录执行。以下为本次实际目录；重跑必须改用新的空输出目录，避免覆盖证据。完整 NPZ 及数据缓存按原有 Git 忽略规则仅保留本地；克隆后先重新生成再做 compare/审计。

```powershell
$np0Python = ".\.venv\Scripts\python.exe"
& $np0Python -m unittest discover -s tests -v
& $np0Python main.py generate --config configs/np0_smoke.json
& $np0Python main.py generate --config configs/np0_validation.json
& $np0Python main.py run --config configs/np0_smoke.json --out results/np0_20261020_smoke_a
& $np0Python main.py run --config configs/np0_smoke.json --out results/np0_20261020_smoke_b
& $np0Python main.py compare results/np0_20261020_smoke_a results/np0_20261020_smoke_b --report results/np0_20261020_smoke_reproducibility.json
& $np0Python main.py run --config configs/np0_validation.json --out results/np0_20261020_validation_a
& $np0Python main.py run --config configs/np0_validation.json --out results/np0_20261020_validation_b
& $np0Python main.py compare results/np0_20261020_validation_a results/np0_20261020_validation_b --report results/np0_20261020_validation_reproducibility.json
& $np0Python main.py plot results/np0_20261020_smoke_a
& $np0Python main.py plot results/np0_20261020_validation_a
& $np0Python scripts/audit_np0_outputs.py results/np0_20261020_smoke_a results/np0_20261020_smoke_b results/np0_20261020_validation_a results/np0_20261020_validation_b --out results/np0_20261020_numerical_audit.json
```

绘图会更新 manifest 的 PNG 哈希；本次在绘图后再次执行 compare，仍全部通过。累计耗时被保存但不作为确定性比较项目，也不据此作算法加速结论。

## 可运行性审计

已执行 `test-research-code` 的只读 `repro_check.py`。第一次发现 DATA 说明缺失，已补充 README 的合成数据来源和生成命令；复查六项均 OK、无 MISSING/WARN。原始审计结果保存在 `results/np0_20261020_repro_audit.json`。

| 审计项 | 状态 | 处理/证据 |
|---|---|---|
| README | OK | 根目录运行说明及成员验证链接 |
| ENV | OK | 直接依赖固定版本，另保存本次完整依赖快照 |
| SEED | OK | 配置固定种子，成员算法无随机初始化 |
| ENTRYPOINT | OK | `main.py` 与 `algorithms.json` 注册入口 |
| DATA | OK | 本地合成 dataset，无外部下载需求 |
| RESULTS_CMD | OK | 上述实际执行命令及结果目录 |

仍待完成：另一位成员的公式与代码复核、独立机器复跑，以及论文 Fig.3 全扫描。这些记录确认本机接口与代表工况运行正常，不宣称已完成全组验收或整篇论文复现。

后续若需要正式打包可使用 prepare-artifacts；若需要逐图比较论文结果可使用 verify-results。本次仅完成 NP-0 接入验证。
