# pipline

当前版本是固定编排的时序预测流水线，数据处理主链已经收敛为 DS 直连协议，不再存在 `data_formatter` 阶段。

## 流程

数据处理链：

```text
data_reading
-> data_analysis
-> feature_engineering
-> split_strategy
-> datanorm
-> preprocess
-> summary
```

建模链：

```text
data_reading
-> data_analysis
-> feature_engineering
-> split_strategy
-> datanorm
-> preprocess
-> model_selection
-> model_training
-> model_integration
-> evaluator
-> summary
```

各阶段职责：

- `data_reading`：读取 `station=<unit>` 分区目录，执行 `ODS -> DS`，保存排序后的原始 DS，并构建 `dataset_profile`
- `data_analysis`：基于 DS 做缺失值检查、统计分析、数据维度检查和序列长度检查
- `feature_engineering`：基于 DS 和 `data_analysis` 结果生成确定性的统计特征
- `split_strategy`：直接对 DS 做 train/val 切分
- `datanorm`：决定是否标准化以及使用哪种策略
- `preprocess`：将 DS 或 engineered DS 展开成模型输入表，并按 train 集统计量做补齐/缩放
- `summary`：输出 iteration 级和任务级总结

流程开关：

- `enable_feature_engineering=false`：跳过 `feature_engineering`，`preprocess` 直接读取 DS
- `enable_split=false`：跳过 `split_strategy`，`preprocess` 直接输出 train 数据
- `enable_normalization=false`：保留 `preprocess`，但不做缩放

## DS 输出约定

`data_reading.py` 当前对外输出的是排序后的原始 DS parquet：

- 路径：`output/<task-id>/data/data_reading_ds_dataset.parquet`
- 标识列：`station`、`timestamp_win`
- 序列列保持 `convert_ods_to_ds` 的原始长度，不在 `data_reading` 阶段补齐或填补
- `dataset_profile` 会同步写入 state，包含：
  - `target_column` / `target_columns`
  - `feature_columns`
  - `scalar_columns`
  - `dimensions`
  - `sequence_summary`
  - `missing_summary`

下游阶段统一读取这份 DS，不再读取 `formatted_dataset_*.parquet`。

## 中间启动

`dataset_path` 约束：

- `start_stage=data_reading`：必须传原始站点目录，格式为 `station=<unit>/...`
- `start_stage=data_analysis` / `feature_engineering` / `split_strategy` / `datanorm`：应传 `data_reading_ds_dataset.parquet` 或其所在任务目录
- `start_stage=preprocess`：
  - `enable_feature_engineering=true`：应传 `feature_engineering_engineered_dataset_<station>.parquet` 所在 iteration 目录，或任务根目录
  - `enable_feature_engineering=false`：应传 `data_reading_ds_dataset.parquet` 或其所在任务目录
- `start_stage=model_selection` / `model_training` / `model_integration` / `evaluator` / `summary`：应传包含 `preprocess_train_preprocessed_<station>.parquet` 的 iteration 目录或任务目录

## 常见产物

- `output/<task-id>/data/data_reading_ds_dataset.parquet`
- `output/<task-id>/data/ds/station_<station>.parquet`
- `output/<task-id>/data/split/train_row_ids.parquet`
- `output/<task-id>/data/split/val_row_ids.parquet`
- `output/<task-id>/iteration_001/feature_engineering_engineered_dataset_<station>.parquet`
- `output/<task-id>/iteration_001/preprocess_train_preprocessed_<station>.parquet`
- `output/<task-id>/iteration_001/preprocess_val_preprocessed_<station>.parquet`
- `output/<task-id>/iteration_001/summary.md`
- `output/<task-id>/final_summary.md`
- `output/<task-id>/task_plan.json`

## 主要文件

- [main.py](/Users/monychen/Documents/demo-zjl/pipline/main.py)
- [webapp.py](/Users/monychen/Documents/demo-zjl/pipline/webapp.py)
- [orchestrator.py](/Users/monychen/Documents/demo-zjl/pipline/orchestrator.py)
- [config_all.json](/Users/monychen/Documents/demo-zjl/pipline/config_all.json)
- [subagents/data_reading.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/data_reading.py)
- [subagents/data_analysis.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/data_analysis.py)
- [subagents/feature_engineering.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/feature_engineering.py)
- [subagents/split_strategy.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/split_strategy.py)
- [subagents/preprocess.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/preprocess.py)

## 运行参数

默认参数来自 [config_all.json](/Users/monychen/Documents/demo-zjl/pipline/config_all.json)，CLI 和 Web 共用这份默认值。

关键字段：

- `unit`
- `start_stage`
- `end_stage`
- `enable_split`
- `enable_feature_engineering`
- `col_ls`
- `pred_col_ls`
- `targ_col_ls`
- `target_col`
- `input_feature_cols`
- `split_method`
- `train_ratio`
- `val_ratio`
- `input_length`
- `output_length`
- `points_per_day`
- `enable_normalization`
- `normalization_policy`
- `max_iterations`

## 说明

- `data_reading` 直接要求显式提供 `col_ls`、`pred_col_ls`、`targ_col_ls`
- `target_col` 与 `input_feature_cols` 只作为分析、特征工程和训练阶段的兼容字段
- `points_per_day` 当前仍保留在 config / CLI / Web 中，但这版流程不再在 `data_reading` 阶段使用它做序列补齐或 NaN 填补
- 仓库约定需要运行 Python 脚本时由用户执行，因此本轮改动默认只做静态校验
