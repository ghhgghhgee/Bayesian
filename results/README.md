# 实验记录

开发过程中的诊断记录保留在 tmp/development_smoke_a，发现并修正了旧 SBL 提前停止问题，不作为交付验收证据，也不纳入提交包。

正式验收使用 `validation_a`、`validation_b` 和 `validation_reproducibility.json`，现已完成且所有检查通过。每次为 3 个工况 × 300 次 × 6 个算法。这不是完整 Fig.3 扫描。详细统计见 docs/组长任务执行摘要.md。

## NP-0 成员接入验证（2026-10-20）

`np0_20261020_smoke_a/b` 和 `np0_20261020_validation_a/b` 为成员接入后的新记录，同时运行参考 `NP0` 与 `NP0_MEMBER`。后者每次为 3 个工况各 300 次；两组双进程复跑检查全部通过。`np0_20261020_tests.txt` 保存 21 项测试通过日志，`np0_20261020_numerical_audit.json` 保存独立重算的审计结果。

估计与完整轨迹 NPZ 按仓库现有忽略规则仅保留本地，可使用同一配置重新生成。提交的 manifest 记录其 SHA256；克隆仓库后应先运行实验生成这些 NPZ，再执行 compare 或独立审计。原有组长结果保持其原始归属。完整说明见 [NP-0 验证记录](../paper-workspace/submission/artifact-repro-checklist.md)。
