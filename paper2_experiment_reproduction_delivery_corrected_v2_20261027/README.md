# member5：论文 2 实验与绘图交付

**2026-10-28 审查修订：正式入口已切换六成员版本。** 旧参考实现不能代替正式扫描。完整命令见 [member5_PR4修订说明](../docs/member5_PR4修订说明_20261028.md)，运行与验证结果见 [RESULTS_REPORT.md](RESULTS_REPORT.md)。

## 正式交付

- `results/team_members_20261028/`：统一 `configs/team_members/paper.json` + `algorithms_baselines_member.json` 的正式结果、CSV、manifest、Fig.1/3/4 和论文对照图。
- `../scripts/member5_plot_paper.py`：正式绘图入口，读取统一扫描结果。请从仓库根目录运行，不使用本目录旧脚本作为正式扫描入口。
- `../tests/test_member5_figures.py`：针对本次审查的回归测试。
- Fig.4 使用本机实测串行时间，不做终点对齐；原始 dB 与显示截断值分列保存。
- IRLS 如实标注 `IRLS (p=0.5, member)`，与论文 IRLS0 仍有方法差异，不为拟合图片改算法或挑样本。

## 探索性补充（2026-10-27 历史结果）

本目录 `bcs/`、`scripts/`、`results/data/`、`results/raw_cases/`、`results/calibration/`、`results/figures/`、`results/comparison/` 以及旧日期说明均为参考实现探索记录，**不是成员正式结果**。旧 IRLS 使用真实 K 的额外先验、NP0 tol=1e-4，不能与正式算法等条件比较。

旧 `paper_scaled` 图片使用任意公共因子 `0.095 / SBL终点`，只是人为形状对照，**不是硬件换算，也不是计时正确性的证据**。正式报告/PPT 使用 `results/team_members_20261028/figures/fig4_measured_trial0.pdf`。

原有 51 项测试只描述旧工作区，不证明新逻辑正确；新验证以 RESULTS_REPORT 为准。完整 NPZ 按团队规则不进 Git，位置和 SHA256 见正式 manifest 与报告。

探索脚本入口检查（仅查看帮助）：

```powershell
python paper2_experiment_reproduction_delivery_corrected_v2_20261027/scripts/calibrate_paper_figures.py --help
```
