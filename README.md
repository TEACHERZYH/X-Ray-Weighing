# X-Ray Weighing Experiments: 2026-07-13 Snapshot

本仓库保存 X 射线煤炭传送带称重研究在 2026-07-13 的历史实验快照，源自原工作区提交 `37f14c0ffcb0e9813c1962c2695c189dd8e7ef49`。该快照仅用于复现实验代码、轻量结果表和数据接口约定，不包含论文稿件或论文图件。

## 包含内容

- `experiments/`：仿真、预训练、迁移微调、少样本校准、鲁棒性与现场 CSV 接口实验脚本；
- `tests/`：时间隔离、在线空皮带基准和校准候选选择的单元测试；
- `results/`：历史实验的汇总 CSV、安装级误差表和同名运行元数据；
- `data/`：真实数据字段映射与统一数据模式模板，不含任何原始采集数据；
- `requirements-pretrain.txt`：预训练与实验运行的核心 Python 依赖；
- `.gitignore`：排除原始数据、生成的大型 CSV、模型文件和逐点预测结果。

## 明确排除内容

- `paper/` 下的论文 Markdown、DOCX、图件、投稿材料和质量检查记录；
- 原始现场数据、模拟大 CSV、类真实 CSV、逐点预测结果、检查点和模型文件；
- 任何账号凭据、设备私有配置或远程主机信息。

## 环境与复现入口

建议使用 Python 3.10 或以上版本，并在仓库根目录执行：

```powershell
pip install -r requirements-pretrain.txt
python -m unittest tests.test_evaluation_protocol
python experiments/run_sim_baselines.py
python experiments/run_sim_ml.py
python experiments/run_sim_tcn.py
python experiments/run_pretrain_tcn.py
python experiments/run_transfer_finetune.py
```

各实验脚本支持 `--help` 查询参数。运行会在 `results/` 生成或更新对应的轻量汇总表与 `.meta.json` 运行记录；大规模生成数据、原始数据和模型文件按 `.gitignore` 保留在本地，不应提交。

## 证据边界

本仓库中的结果文件是 2026-07-13 的历史实验输出。仿真和类真实结果不能替代多设备、长期、可溯源的工业现场计量验收；`real_xray_*.csv` 仅保留汇总结果和接口验证记录，不包含源数据，也不代表完整现场闭环验证。后续使用真实数据时，应先依据 `data/` 中的字段模板完成映射和数据质量检查，再重新运行相关实验。
