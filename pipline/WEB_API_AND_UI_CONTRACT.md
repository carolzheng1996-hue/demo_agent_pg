# pipline Web API And UI Contract

## API

### `GET /api/config`

- 返回默认表单参数和最近任务列表
- 默认表单参数来自 [config_all.json](/Users/monychen/Documents/demo-zjl/pipline/config_all.json)

### `POST /api/jobs`

提交一个确定性 pipeline 任务。

请求字段：

- `config_all`
- `query`
- `dataset_path`
- `dataset_name`
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
- `split_cutoff_date`
- `split_test_units`
- `train_ratio`
- `val_ratio`
- `input_length`
- `output_length`
- `points_per_day`
- `enable_normalization`
- `normalization_policy`
- `use_system_random`
- `max_iterations`

`dataset_path` 约束：

- `start_stage=data_reading`：必须是按 `station=<unit>` 分区的数据目录
- `start_stage=data_analysis` / `feature_engineering` / `split_strategy` / `datanorm`：可以是 `data_reading_ds_dataset.parquet`，或包含该文件的任务目录
- `start_stage=preprocess`：
  - `enable_feature_engineering=true`：应指向包含 `feature_engineering_engineered_dataset_<station>.parquet` 的 iteration 目录或任务目录
  - `enable_feature_engineering=false`：应指向 `data_reading_ds_dataset.parquet` 或其所在任务目录
- `start_stage=model_selection` / `model_training` / `model_integration` / `evaluator` / `summary`：应指向包含 `preprocess_train_preprocessed_<station>.parquet` 的 iteration 目录或任务目录

阶段控制：

- 已移除 `data_formatter`
- `start_stage` / `end_stage` 可选值现在为：
  - `data_reading`
  - `data_analysis`
  - `feature_engineering`
  - `split_strategy`
  - `datanorm`
  - `preprocess`
  - `model_selection`
  - `model_training`
  - `model_integration`
  - `evaluator`
  - `summary`

流程开关：

- `enable_feature_engineering=false`：跳过 `feature_engineering`，`preprocess` 直接读取 DS
- `enable_split=false`：跳过 `split_strategy`，并同时跳过 `model_integration` 与 `evaluator`
- `enable_normalization=false`：保留 `preprocess`，但不执行缩放标准化

DS 约束：

- `data_reading` 直接输出排序后的原始 DS 到 `data/data_reading_ds_dataset.parquet`
- 当前不会在 `data_reading` 阶段做序列补齐或数组内 NaN 填补
- `target_col` / `input_feature_cols` 只保留下游兼容语义，不再驱动 `convert_ods_to_ds`
- `points_per_day` 当前仍保留在请求字段中，但这版流程不会在 `data_reading` 阶段使用它

`split_method` 支持：

- `global_last_k`
- `station_last_k`
- `station_month_last_k`
- `fixed_date`
- `leave_stations_out`

### `GET /api/jobs/<job_id>`

- 返回任务执行状态、plan、task_id、summary_text 和 task_detail

### `GET /api/tasks`

- 返回历史任务列表

### `GET /api/tasks/<task_id>`

- 返回任务详情：
  - `task_id`
  - `status`
  - `task_type`
  - `requires_modeling`
  - `plan`
  - `dataset_profile`
  - `iteration_count`
  - `iterations`
  - `final_summary`

## UI

- 左侧展示任务列表和搜索
- 右侧表单提交任务
- 表单阶段选项已去掉 `data_formatter`
- `Points Per Day` 只在阶段范围包含 `data_reading` 时显示
- 详情区优先读取任务级 `task_plan.json`、`final_summary.md`、iteration `summary.md`

## 当前模型配置

- 当前默认候选模型为 `arima`、`xgboost`、`linear`

## 约束

- 当前 `pipline` 不提供在线改代码能力
- 当前 `pipline` 不依赖 LLM，所有计划和步骤均由预定义流程驱动
