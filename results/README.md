# 实验记录

`smoke_a` 是开发过程中的诊断记录，发现了旧 SBL 提前停止问题，不作为交付验收证据。其 manifest 保留了当时源码哈希；后续正式运行不会覆盖它。

正式验收使用 `validation_a`、`validation_b` 和 `validation_reproducibility.json`（生成后才视为完成）。每次为 3 个工况 × 300 次 × 6 个算法。这不是完整 Fig.3 扫描。
