# 实验日志

## 2026-06-23 模拟数据论文初稿实验

- 代码版本：`67a5b5a`之前已完成模拟实验、图表和论文初稿；本轮开始增加机器可读实验元数据。
- 数据状态：使用模拟数据，真实现场数据待替换。
- 主要结果表：`sim_baselines.csv`、`sim_ml_results.csv`、`sim_tcn_results.csv`、`sim_stratified_errors.csv`、`sim_install_errors.csv`、`sim_calibration_curve.csv`。
- 论文状态：已形成第三版模拟数据稿，13篇参考文献，DOCX结构检查通过但未完成LibreOffice视觉渲染QA。

## 2026-06-23 实验版本元数据重跑

- 代码版本：`058a88a`。
- 运行环境：本地Python解释器；元数据中已记录Python、平台、命令行参数和输入输出SHA256。
- 主要变化：所有主要结果表新增同名 `.meta.json`，`git.is_dirty=false`，生成物状态单独记录在 `generated_status_short`。
- 关键结果：PG-TSC-TCN未校准累计误差0.155%，20个校准样本后累计误差0.076%；PG-TSC-lite在180个校准样本后累计误差0.027%。

## 2026-06-23 补充消融实验

- 代码版本：`3dc0927`。
- 结果文件：`results/sim_ablation_summary.csv`，元数据：`results/sim_ablation_summary.csv.meta.json`。
- 关键结果：在线`I0`累计误差0.173%，固定初始`I0`累计误差2.020%；灰分缺失标记当前模拟中无独立收益；少样本校准对采样策略敏感，20个开头连续样本可退化到100.000%，20个时间均匀样本为0.401%，20个载荷分层样本为0.456%。

## 2026-06-23 模拟压力测试

- 数据状态：继续使用模拟数据，真实现场数据待替换。
- 结果文件：`results/sim_stress_tests.csv`，元数据：`results/sim_stress_tests.csv.meta.json`。
- 图表文件：`paper/figures/fig7_stress_tests.png`、`paper/figures/fig7_stress_tests.svg`。
- 关键结果：有载误判为空载比例从0%升至3%时累计误差由0.173%升至23.201%；按煤种和目标现场共同偏置的非随机灰分缺失机制，目标缺失率60%时，使用缺失标记和不使用缺失标记的累计误差分别为0.683%和0.837%，仅作为边界性风险提示，不写成真实现场性能贡献；高隐藏水分/粒度扰动下，物理+时序模型累计误差为8.921%。

## 2026-06-26 部署可行性补强实验

- 代码版本：`f262e85`。
- 数据状态：继续使用模拟数据，真实现场数据待替换。
- 结果文件：`results/sim_deployment_readiness.csv`，元数据：`results/sim_deployment_readiness.csv.meta.json`。
- 图表文件：`paper/figures/fig8_deployment_readiness.png`、`paper/figures/fig8_deployment_readiness.svg`。
- 关键结果：平衡自动空载门控精确率99.690%、召回率64.809%、有载误写入率0.023%，累计误差0.021%；宽松门控有载误写入率升至14.104%时累计误差升至4.742%；3%接收端射线值尖峰下无防护累计误差0.395%，轻量防护后为0.018%；5%尖峰下无防护和轻量防护分别为1.135%和0.186%。

后续每次重跑实验时，在此追加日期、git commit、数据版本、脚本命令和论文中受影响的表图。

## 2026-06-28 跨安装bootstrap不确定性补强

- 代码版本：`1ea663f`。
- 数据状态：复用`results/sim_predictions.csv`中的逐点预测，不重新训练模型。
- 执行脚本：`experiments/analyze_cross_install_uncertainty.py`。
- 输出文件：`results/sim_cross_install_uncertainty.csv`及同名元数据。
- 分析口径：以S04测试现场4个安装段为bootstrap单元，重复抽样10000次，计算MAE、RMSE和累计误差的点估计、安装段范围和95%区间。
- 关键结果：Physics-Ridge累计误差点估计0.446%，bootstrap 95%区间0.004%-2.024%，安装段范围0.623%-2.887%；Physics-Temporal-Ridge累计误差点估计0.173%，bootstrap 95%区间0.013%-1.688%，安装段范围0.372%-2.160%。
- 结论边界：该结果说明物理约束方法在安装段重采样下仍保持较低总量偏差，同时提示全局累计误差可能受不同安装段偏差抵消影响；不能写成真实现场置信区间。

## 2026-06-26 大规模仿真预训练

- 代码版本：`184f78e`。
- 远程项目：`[remote-project-root]/xray-sim-pretrain-184f78e`。
- 远程数据：`[remote-dataset-root]/xray-sim-pretrain-184f78e/xray_belt_pretrain_large.csv`。
- 数据规模：24个现场、每现场3台设备、每设备3次安装、每安装7200 s，共1555200行；空载比例0.119421，灰分缺失比例0.397510。
- 训练设置：S01-S20训练，S21-S24未见现场验证；TCN序列长度60，4层，48通道，16个epoch，CUDA训练，最终训练损失0.103159。
- 结果文件：`results/sim_pretrain_dataset_summary.csv`、`results/sim_pretrain_tcn_results.csv`及同名元数据。
- 模型文件：`models/pgt_sc_tcn_pretrained.pt`、`models/pgt_sc_base_model.pkl`，按规则不提交git。
- 关键结果：PG-TSC-TCN-pretrain在未见现场上MAE 5.498 kg/s、RMSE 7.310 kg/s、累计误差0.007%；20、60、180、600样本校准后的累计误差分别为0.239%、0.334%、0.035%、0.042%。

## 2026-06-26 仿真目标域迁移与累计质量一致性微调

- 代码版本：`e91b0f2`。
- 远程项目：`[remote-project-root]/xray-mass-consistency-e91b0f2`。
- 远程数据：`[remote-dataset-root]/xray-mass-consistency-e91b0f2/xray_belt_transfer_target.csv`。
- 数据规模：8个目标现场、每现场2台设备、每设备2次安装、每安装3600 s，共115200行；空载比例0.125425，灰分缺失比例0.400243。
- 迁移协议：每安装前20%时间用于适配，后80%时间用于评估；校准层使用有载样本；残差微调和从零训练均使用每安装最多180个序列、8个epoch；累计质量一致性微调在普通残差损失中加入批内累计质量误差项。
- 结果文件：`results/sim_transfer_target_summary.csv`、`results/sim_transfer_finetune_results.csv`及同名元数据。
- 模型文件：`models/pgt_sc_tcn_transfer_finetuned.pt`，按规则不提交git。
- 关键结果：预训练直接迁移累计误差1.822%；60样本/安装校准后累计误差0.854%；一参数累计比例校准在180样本/安装时MAE 9.245 kg/s、累计误差0.925%；累计质量一致性微调MAE 8.072 kg/s、RMSE 10.552 kg/s、累计误差1.371%，优于从零训练的12.945 kg/s、16.281 kg/s和4.990%。

## 2026-06-26 主动校准选样模拟

- 代码版本：`581cd37`。
- 远程项目：`[remote-project-root]/xray-active-calibration-581cd37`。
- 远程数据：`[remote-dataset-root]/xray-mass-consistency-e91b0f2/xray_belt_transfer_target.csv`。
- 执行脚本：`experiments/analyze_active_calibration.py`，远程包装脚本为`scripts/run_remote_active_calibration.sh`。
- 输出文件：`results/sim_active_calibration_results.csv`及其元数据。
- 关键结果：20样本/安装预算下，预测分层样本MAE/RMSE为10.549/14.324 kg/s、累计误差0.623%，低于开头连续样本的12.214/16.782 kg/s和1.945%、随机样本均值的11.170/15.011 kg/s和0.991%。特征多样性样本在20样本预算下MAE/RMSE为9.866/12.816 kg/s，但累计误差为1.370%。
- 结论边界：主动选样不能写成所有预算和所有指标均最优；当前结果支持“标定样本覆盖范围影响少样本校准效果”，真实数据阶段需要复核选样策略和累计质量目标的一致性。

## 2026-06-26 预测可靠性边界与再校准触发模拟

- 代码版本：`b08beb2`，分支`codex/innovation-sim-experiments`。
- 远程项目：`[remote-project-root]/xray-prediction-reliability-b08beb2`。
- 复用目标域数据：`[remote-dataset-root]/xray-mass-consistency-e91b0f2/xray_belt_transfer_target.csv`。
- 复用预训练模型：`[remote-project-root]/xray-sim-pretrain-184f78e/models/pgt_sc_tcn_pretrained.pt`和`pgt_sc_base_model.pkl`。
- 执行脚本：`experiments/analyze_prediction_reliability.py`，远程包装脚本为`scripts/run_remote_prediction_reliability.sh`。
- 输出文件：`results/sim_prediction_reliability_results.csv`及其元数据。
- 核心结果：预测分层选样、90%名义覆盖率下，全局残差区间实际覆盖率约54%，安装级残差区间约51%-53%，流量分层残差区间约42%-49%；风险触发用约41%-42%触发率捕获61.183%-66.225%的Top10%高误差样本，触发/非触发MAE为1.222-1.299。
- 结论边界：少样本残差区间不能写成真实现场置信保证；当前结果只支持“风险触发可用于复核和追加标定提示”，真实数据阶段需要验证覆盖率和触发闭环。

## 2026-06-29 类真实数据替换流程演练

- 代码版本：`cdc8cf1`生成报告图，`5844e61`完成证据链对齐，分支`codex/innovation-sim-experiments`。
- 数据文件：`data/real_like/pseudo_real_xray_belt.csv`，按规则不作为真实现场数据提交；数据画像为`results/pseudo_real_data_profile.csv`。
- 执行命令：

```text
python experiments/generate_pseudo_real_xray_data.py --output data/real_like/pseudo_real_xray_belt.csv --summary results/pseudo_real_data_profile.csv --sites 5 --devices-per-site 2 --installs-per-device 3 --seconds 1200 --seed 20260629
python scripts/validate_real_data_schema.py data/real_like/pseudo_real_xray_belt.csv
python scripts/check_real_data_readiness.py data/real_like/pseudo_real_xray_belt.csv --out paper/checkpoints/55_pseudo_real_data_readiness_check.md
python scripts/check_real_data_coverage_acceptance.py
python experiments/run_pseudo_real_experiments.py --input data/real_like/pseudo_real_xray_belt.csv --summary results/pseudo_real_experiment_summary.csv --calibration results/pseudo_real_calibration_strategy.csv --stratified results/pseudo_real_stratified_errors.csv --install-errors results/pseudo_real_install_errors.csv
python experiments/plot_pseudo_real_experiments.py --out-base paper/figures/pseudo_real_experiment_summary
```

- 输出文件：`results/pseudo_real_experiment_summary.csv`、`results/pseudo_real_calibration_strategy.csv`、`results/pseudo_real_stratified_errors.csv`、`results/pseudo_real_install_errors.csv`及同名元数据；报告为`paper/pseudo_real_data_experiment_report.md`，图件为`paper/figures/pseudo_real_experiment_summary.png/.svg/.pdf`。
- 数据覆盖：36000行，5个现场、10台设备、30个安装段；空带比例0.1596，灰分缺失比例0.3755，环境温度跨度16.7 ℃，湿度跨度37.9%，接收端射线值与载荷相关系数-0.9030。
- 验收结果：字段检查通过；真实数据替换就绪检查PASS=14、WARN=0、BLOCK=0；覆盖验收PASS=29、WARN=0、BLOCK=0。
- 关键结果：未见S04-S05现场上，Physics-Temporal-Ridge为MAE 7.351 kg/s、累计误差1.046%；PG-TSC-lite 20样本时间均匀校准为MAE 5.164 kg/s、累计误差0.258%；180样本时间均匀校准为MAE 5.172 kg/s、累计误差0.520%。安装级统计显示180样本校准累计误差中位数0.777%，最大5.134%。
- 结论边界：该实验只证明真实数据替换流程、分层评估、安装级尾部风险检查和图表生成链路可执行；不得写成真实工业现场计量性能。

## 2026-06-29 类真实人工干预成本-计量误差折中实验

- 代码版本：`43dbb07`基础上新增脚本，分支`codex/innovation-sim-experiments`。
- 数据文件：`data/real_like/pseudo_real_xray_belt.csv`，继续作为类真实模拟数据，不作为真实现场数据。
- 执行命令：

```text
python experiments/analyze_manual_intervention_reduction.py --input data/real_like/pseudo_real_xray_belt.csv --output results/pseudo_real_manual_intervention_reduction.csv
```

- 输出文件：`results/pseudo_real_manual_intervention_reduction.csv`及同名元数据；计划与报告为`paper/manual_intervention_reduction_experiment_plan.md`和`paper/manual_intervention_reduction_experiment_report.md`。
- 实验口径：训练现场S01-S03，未见测试现场S04-S05；测试安装段数量12；每安装段前30%用于安装期校准，中间35%用于运行期维护候选，后35%用于最终评价；初始安装期标签预算为20个/安装。
- 关键结果：`risk_covered_anchor`在20个追加标签/安装、40个总标签/安装时，全局累计误差0.018%，相对`scheduled_uniform`的180个追加标签/安装、200个总标签/安装减少80.000%参考标签；但安装级最大累计误差仍为11.800%，安装级低于1%比例为41.667%。
- 结论边界：该实验支持“从固定周期人工校准转向风险触发式少样本维护”的包装，不支持“后期完全无需人工校准”。

## 2026-07-03 真实数据清洗与训练实验

- 数据文件：`data/real_xray_belt.csv`，真实数据按规则不直接提交；清洗输出位于`data/real/real_xray_belt_clean.csv`和`data/real/real_xray_belt_clean_qc.csv`。
- 远程项目：`[remote-project-root]/X-RAY`。
- 远程数据：`[remote-dataset-root]/real_xray_belt.csv`。
- 执行脚本：`experiments/clean_real_xray_data.py`、`experiments/run_real_xray_experiments.py`。
- 输出文件：`results/real_data_cleaning_report.csv`、`results/real_data_profile.csv`、`results/real_xray_experiment_summary.csv`、`results/real_xray_calibration_strategy.csv`、`results/real_xray_stratified_errors.csv`、`results/real_xray_install_errors.csv`、`results/real_xray_qc_holdout_errors.csv`及同名元数据。
- 清洗结果：36000行真实数据，覆盖5个现场、10台设备、30个安装段和4类煤种；空带5747行，有载30253行；点异常360行，连续异常285行，训练排除642行；灰分补全108行；分安装时间和累计质量单调性均无下降。`validate_real_data_schema.py`通过，`check_real_data_readiness.py`为PASS=14、WARN=0、BLOCK=0。
- 训练协议：S01-S03训练，S04-S05目标现场评价；目标安装段前30%为安装期标定候选，后70%为评价段；评价样本9873行。
- 关键结果：Physics-Temporal-Ridge为MAE 8.851 kg/s、RMSE 11.529 kg/s、累计误差0.960%；HGB为MAE 8.431 kg/s、RMSE 11.645 kg/s、累计误差0.129%；PG-TSC-real-mass-scale在5样本/安装、预测分层策略下为MAE 7.296 kg/s、RMSE 9.665 kg/s、累计误差0.296%；60样本/安装、预测分层策略下为MAE 6.579 kg/s、RMSE 9.017 kg/s、累计误差0.416%。
- 结论边界：真实数据支持“物理时序模型 + 少样本累计质量比例锚定”的安装自校准主张；无约束线性斜率+截距校准存在过校准风险，异常留出样本必须作为运行期风险触发和复核边界，不能写成无条件自动更新。
# 2026-06-29 统一核心科学问题实验

- Plan: `paper/unified_core_problem_experiment_plan.md`
- Script: `experiments/run_unified_core_problem_experiment.py`
- Results: `results/sim_unified_core_problem.csv`, `results/sim_unified_core_problem_summary.csv`
- Report: `paper/unified_core_problem_experiment_report.md`
- Purpose: 将不完全观测、环境/设备漂移、安装差异、少样本安装校准和累计质量误差放入同一模拟矩阵，支撑PG-TSC作为仪器级自校准框架的创新主线。
- Key finding: nominal场景60样本/安装累计误差中位数为0.352%，install_shift_high场景60样本/安装为0.427%；combined_high场景60样本/安装为1.428%，说明统一高扰动下只能支持有边界改善，不能写成完全自动或真实现场已达标。
