# Python 基线成员模块验收记录

日期：2026-10-23（Asia/Shanghai）。基于主分支提交 `6b9588fbaba1889c0ca70ab84c7fc04da478c26a`。

## 环境与变更范围

Windows，Python 3.12，NumPy 2.2.4、SciPy 1.15.2、Matplotlib 3.10.0，与原 `requirements.txt` 一致。没有增加第三方依赖。

只新增成员基线模块、单独注册文件、子目录配置、直接显示入口、测试与说明。原有 121 个受版本控制文件的本地字节 SHA256 在运行前后保持一致；原 `main.py`、`algorithms.json`、参考算法和历史结果未修改。新注册文件完整保留基准提交原有 8 项注册，再增加四个 `_MEMBER` 条目。

原仓库在 Windows 检出时，`tests/test_np1_formulas.py` 的最后一个空白行会触发 Git 的 CRLF 规范化差异；它在首次检出就被 Git 标为 modified。该文件未编辑、未暂存，不包含在本次新增提交中，字节哈希与运行前一致。

核心模块 `algorithms/baselines_member.py` 的 SHA256：

```text
737e4c983511ebd95bdfa896b570f6eac3ba2b9d4f840b0f554a8e7c39732bcf
```

## 单元与集成测试

- 新增前原仓库：51 项通过。
- 新增模块测试：16 项通过。
- 加入后的完整测试集：67 项通过。
- 原 `main.py run` + 新注册文件：可调用四个成员入口。
- 新增 CLI 集成测试通过两个独立进程执行 run，再用原 compare 和 plot 验证。
- 默认直接显示入口的 `--no-show` 检查通过；只在内存评估和绘图，不调用保存接口。

命令：

```powershell
python -m unittest discover -s tests -v
python -m scripts.baselines_member_display --no-show
```

本机旧版 Matplotlib 与较新 pyparsing 的组合会打印第三方弃用提示；测试仍通过。未为消除提示改动原来的依赖文件。

## 每个代表工况 300 次的对照验收

使用 `configs/baselines_member/validation.json`，原参考基线与四个成员基线共 8 项。每次运行包含 3 个工况 × 300 次 × 8 算法 = 7,200 次重建；两次独立进程总计 14,400 次，无算法异常。

每个 (M,K,trial) 的 8 个算法共享完全相同的输入哈希；共检查 900 组配对输入。

成员版本成功次数：

| M | K | SBL_MEMBER | FOCUSS_MEMBER | IRLS_MEMBER | BP_MEMBER |
|---:|---:|---:|---:|---:|---:|
| 10 | 3 | 233/300 | 33/300 | 91/300 | 47/300 |
| 30 | 3 | 300/300 | 264/300 | 300/300 | 300/300 |
| 30 | 15 | 121/300 | 0/300 | 77/300 | 1/300 |

原参考版本成功次数（同一批 A/y）：

| M | K | SBL | FOCUSS | IRLS | BP |
|---:|---:|---:|---:|---:|---:|
| 10 | 3 | 233/300 | 32/300 | 55/300 | 47/300 |
| 30 | 3 | 300/300 | 264/300 | 295/300 | 300/300 |
| 30 | 15 | 121/300 | 0/300 | 4/300 | 1/300 |

版本差异包括成员迭代停止阈值更严格、FOCUSS 使用明确的方差底限和 SVD、IRLS 使用收敛触发的 epsilon 递减以及 x²+epsilon 的平滑。差异已报告，不要求和参考版本逐位一致。上表 IRLS 的成功率变化同时包含平滑规则与停止条件变化，不能据此把全部提升归因于某个单独改动；更长的运行时间与更高迭代预算也须一起考虑。

## 独立进程复跑

实际用原 `main.py compare` 检查两次代表验收，9 项检查全部通过：

```text
completed_without_errors: true
same_config: true
same_source: true
same_algorithms: true
same_environment: true
output_integrity: true
same_trials_metrics_and_data: true
estimates.npz: true
histories.npz: true
```

与原框架一致，数值与历史比较使用 rtol=1e-10、atol=1e-12，耗时不要求逐位相同。比较是同机同环境中的独立 Python 进程，并不是另一位成员在独立机器上的交叉验收。

本次验收数据写到仓库忽略的 `tmp/baselines_member_validation_a`、`tmp/baselines_member_validation_b` 和 `tmp/baselines_member_data`；compare 证据为 `tmp/baselines_member_reproducibility.json`。这是明确运行原框架进行验收所生成的临时证据，默认显示入口仍不自动保存文件。没有覆盖或发布仓库已有结果。

## 完整扫描与后续交叉复核

`configs/baselines_member/paper.json` 是四个成员基线的 59 工况、每点 300 次配置。交付打包时，完整扫描尚未取得完成的 manifest，因此不将它列为已经完成的 Fig.3 复现。此文件包的已完成验收依据是 67 项测试和两次各 7,200 次的代表工况运行；不能把配置文件的存在当成完整扫描结果。完整扫描的运行命令见接口说明。

本包仅交付新增源代码、配置、测试与接口说明，不包含临时实验数据、Python 环境或登录凭据。尚未通过 GitHub 写入接口上传这些新增文件，用户可保留目录结构手动上传到仓库根目录。

组长合并前仍应由另一名成员在自己的环境执行文档中的 run/compare 命令，并核对论文与接口。相同成功率、确定性复跑通过或测量残差很小，都不能替代数学公式核对；基础 FOCUSS 的错误支撑解仍然可以满足 Ax=y。
