# Paper 2 reproduction - corrected v2

本版本针对 IRLS0 与 Fig. 4 计时重新核查。

- Fig. 3：真实 300 次 Monte Carlo 结果；IRLS0 数学更新经等价快速实现复核，成功率逐点不变。
- Fig. 4 实测版：固定 trial 0 串行预热后重复 15 次，绘制累计时间中位数。
- Fig. 4 论文对照版：三种算法使用同一个由 SBL 推得的硬件尺度因子，文件名明确带 `paper_scaled`。
- `fig4_trial0_curves.csv` 同时保存实测中位数、25%/75% 分位数和论文尺度时间。
- 完整说明见 `../docs/论文2_IRLS与Fig4计时核查修正_20261027.md`。

正式实验仍为 59 个工况、每点 300 次、6 个算法，共 106,200 次调用。
