# 实验版本管理规范

本项目用git管理实验代码、论文稿件、结果表和轻量元数据。原始真实数据、模拟大CSV、逐点预测表和模型文件不直接提交。

## 每次实验的最小记录

1. 先确认代码状态：`git status --short`。
2. 运行实验脚本，脚本会为主要输出生成同名 `.meta.json` 文件。
3. 检查结果表、图和元数据是否匹配本次实验目的。
4. 提交代码、结果表、图和 `.meta.json`，提交信息使用 `Experiment: ...` 或 `Update experiment: ...`。

## 元数据内容

`experiment_versioning.py` 会记录：

- git commit、branch、dirty状态和短状态列表。
- 运行脚本、命令行参数和Python环境。
- 输入、输出文件路径、大小和SHA256。
- 本次实验说明。

如果元数据中的 `git.is_dirty=true`，说明实验是在有未提交改动的工作区运行的。正式写入论文表格前，应重新提交代码后再复现实验一次。

## 数据提交边界

- 提交：`experiments/*.py`、`results/*.csv`、`results/*.meta.json`、`paper/figures/*`、论文文档。
- 不提交：`data/raw/`、`data/real/`、`data/real_like/*.csv`、`data/simulated/*.csv`、`results/sim_predictions.csv`、`models/`。

真实数据到位后，先在 `paper/real_data_replacement_guide.md` 中登记字段映射，再按本规范生成新的结果版本。
