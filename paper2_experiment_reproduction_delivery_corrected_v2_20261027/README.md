# Bayesian Compressive Sensing：论文 2 实验复现（计时核查修正版 v2）

这是新的独立 GitHub 交付目录；旧交付与其他成员文件未被覆盖。

## 本轮修正

- IRLS0 改用数学等价的加权正规方程，避免 Fig. 4 中无用的 SVD/协方差开销。
- Fig. 4 计时不再读取并行 Monte Carlo 进程中的单次时间；改为串行预热后独立重复 15 次并取累计时间中位数。
- `fig4_computational_cost_trial0` 是本机实测时间。
- `fig4_computational_cost_trial0_paper_scaled` 是使用一个公共硬件尺度因子的论文对照版本。
- Fig. 3 共 59 个工况，每点 300 次 Monte Carlo，6 个算法共 106,200 条记录。
- Fig. 3 的 IRLS0 经等价实现复核后成功率逐点不变；剩余差异属于原作者 IRLS0 实现和参数未公开，未使用论文数字化数据替换仿真结果。

## 关键文件

- `docs/论文2_IRLS与Fig4计时核查修正_20261027.md`：本轮诊断与修正说明。
- `results/figures/fig3_success_rate.pdf`：真实 300 次 Monte Carlo 成功率。
- `results/figures/fig4_computational_cost_trial0.pdf`：本机串行中位实测时间。
- `results/figures/fig4_computational_cost_trial0_paper_scaled.pdf`：论文硬件尺度对照。
- `results/comparison/fig4_side_by_side.pdf`：论文与硬件尺度版本并排对照。
- `results/comparison/fig4_measured_side_by_side.pdf`：论文与本机实测版本并排对照。
- `results/data/fig4_trial0_curves.csv`：实测时间中位数、四分位数、缩放时间和 MSE。
- `results/calibration/`：IRLS 列归一化、p 延续、初始平滑量、l1 暖启动和幅度分布筛查。

## 复现

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce_paper_figures.py --out results_rerun --workers 8 --profile paper-aligned
```

之后将原论文裁图放入 `results_rerun/comparison/paper_crops/`，再运行：

```bash
python scripts/build_paper_figure_comparisons.py --run results_rerun
```

绝对运行时间依赖硬件；算法迭代次数和误差轨迹不使用时间缩放。
