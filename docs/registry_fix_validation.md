# 注册表 JSON 修复与主框架验证

日期：2026-10-21。基于远程 main 的 `77fc5ff` 修复 `algorithms.json` 中 BP 项末尾缺失的逗号；此前读取注册表会在第 9 行触发 JSON 解析错误。算法入口、所有者和参数均保持原值。

## 验证结果

- 全部 34 项 unittest 通过，包括 NP-0、NP-1 成员测试和 3 项新注册表回归测试。日志：`results/registry_fix_20261021_tests.txt`。
- 新回归测试拒绝非法 JSON、重复键、不可调用入口及无效配置引用，并通过真实 CLI 调用全部 8 个注册入口，检查来源 SHA256、共享输入、估计和 history/times 合同。
- `configs/all_algorithms_smoke.json` 使用 N=100、种子 20261019，3 个工况 `[10,3]`、`[30,3]`、`[30,15]`，每工况 5 次，共 8 个算法。每轮 120 次调用，两轮共 240 次，均无算法异常；共 15 个独立输入，第二轮复用相同输入。
- `generate`、`run`、`compare`、`plot` 流程均成功。复跑比较的 9 项检查全部为 true，比较估计、轨迹、逐次指标、源码/环境/输入与输出完整性，排除耗时差异；绘图后再次 compare 仍通过。
- 环境：Python 3.11.15、NumPy 2.2.4、SciPy 1.15.2、Matplotlib 3.10.0；沿用仓库 `.venv` 和锁定依赖。

数值生成提交为 `e752c22e1ad443c75c103e309abe3332e08a6c89`；后续提交仅补充本报告及结果证据。结果目录为 `results/registry_fix_20261021_a`、`results/registry_fix_20261021_b`，复跑报告为 `results/registry_fix_20261021_reproducibility.json`。完整 NPZ 和数据缓存按原有规则仅保留本地，克隆后可重新运行生成；manifest 保存相应哈希。

## 实际执行命令

在仓库根目录执行；重新运行时请换用新的空输出目录。

```powershell
$registryPython = ".\.venv\Scripts\python.exe"
& $registryPython -m unittest discover -s tests -v
& $registryPython main.py generate --config configs/all_algorithms_smoke.json
& $registryPython main.py run --config configs/all_algorithms_smoke.json --out results/registry_fix_20261021_a
& $registryPython main.py run --config configs/all_algorithms_smoke.json --out results/registry_fix_20261021_b
& $registryPython main.py compare results/registry_fix_20261021_a results/registry_fix_20261021_b --report results/registry_fix_20261021_reproducibility.json
& $registryPython main.py plot results/registry_fix_20261021_a
```

这是全部注册入口的小规模集成验证，不代替论文全扫描、算法数学正确性审查或其他机器的独立复跑。
