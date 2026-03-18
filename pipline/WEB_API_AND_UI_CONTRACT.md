# pipline Web API And UI Contract

## API

### `GET /api/config`

- 返回默认表单参数和最近任务列表
- 默认表单参数来自 [config_all.json](/Users/monychen/Documents/demo-zjl/pipline/config_all.json)

### `POST /api/jobs`

- 提交一个新的确定性 pipeline 任务
- 请求字段：
  - `config_all`
  - `query`
  - `dataset_path`
  - `dataset_name`
  - `unit`
  - `formatter_unit`
  - `start_stage`
  - `end_stage`
  - `enable_split`
  - `enable_feature_engineering`
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

`normalization_policy` 支持：
- `auto`
- `off`
- `zscore`
- `minmax`

`use_system_random` 支持：
- `true`
- `false`

`dataset_path` 约束：
- 必须是按 `station=<unit>` 分区的数据目录
- `unit` 为空时默认读取目录下全部站点；多个站点用逗号分隔
- 当 `start_stage` 不是 `data_reading` 时，`dataset_path` 可以改为对应中间产物路径：
  - `data_formatter`：`ds_dataset.parquet` 或其所在目录
  - `data_analysis` / `feature_engineering` / `split_strategy` / `datanorm`：`formatted_dataset_*.parquet` 所在目录
  - `preprocess`：
    - `enable_feature_engineering=true`：`feature_engineering_engineered_dataset_*.parquet` 所在 iteration 目录，或包含这些工件的任务根目录
    - `enable_feature_engineering=false`：`formatted_dataset_*.parquet` 所在目录，或包含这些工件的任务根目录
  - `model_selection` / `model_training`：`train_preprocessed.parquet` 所在目录

`split_method` 支持：
- `global_last_k`
- `station_last_k`
- `station_month_last_k`
- `fixed_date`
- `leave_stations_out`

附加切分参数：
- `split_cutoff_date` 仅在 `split_method=fixed_date` 时生效
- `split_test_units` 仅在 `split_method=leave_stations_out` 时生效，多个站点用逗号分隔
- `enable_split=false` 时会跳过 `split_strategy`，并同时跳过 `model_integration` 与 `evaluator`
- `enable_feature_engineering=false` 时会跳过 `feature_engineering`，`preprocess` 将直接读取 formatted 数据
- `enable_normalization=false` 时会保留 `preprocess`，但不执行缩放标准化

阶段控制参数：
- `start_stage`：允许从指定阶段启动流程
- `end_stage`：允许在指定阶段结束流程，仅做数据处理
- `formatter_unit`：允许在 `data_formatter` 阶段重新选择要处理的站点
- 当 query 被识别为“只做数据处理”时，`end_stage=summary` 只表示生成总结，不会再自动进入模型训练

DS 清洗参数：
- `points_per_day` 表示一天内的采样点数
- `data_formatter` 会在 DS 进入切分和模型前，按 `input_length/output_length` 做序列补齐，并按 `points_per_day` 用“前一天同一时间点”优先填补数组内 NaN

切分比例说明：
- `train_ratio`、`val_ratio` 允许输入非归一化数值
- 后端会自动归一化，例如 `8 / 2` 会转换为 `0.8 / 0.2`

UI 交互约束：
- `Split Cutoff` 仅在 `split_method=fixed_date` 时显示
- `Split Test Units` 仅在 `split_method=leave_stations_out` 时显示
- 如果阶段范围不包含 `split_strategy`，切分相关字段会自动折叠
- 如果当前任务不进入模型阶段，`max_iterations` 会自动折叠
- 如果当前阶段范围不包含 `data_reading/data_formatter`，`Points Per Day` 会自动折叠
- `enable_feature_engineering`、`enable_split`、`enable_normalization` 是当前版本最主要的三个流程控制开关

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
  - `iteration_history`
  - `cross_iteration_ensemble`
  - `iterations[*].summary_markdown`
  - `iterations[*].artifacts`

## UI

- 左侧展示任务列表和搜索
- 右侧表单提交预测任务或分析任务
- 表单支持配置标准化方式以及是否启用 `SystemRandom`
- 详情区展示：
  - 优先读取任务级 `task_plan.json`、`final_summary.md`、各 iteration 的 `summary.md`
  - 不再依赖旧的每阶段 `result.json` 目录结构
  - 当缺少逐阶段结构化结果时，会回退展示任务概览、任务级数据工件和 markdown 总结
  - 任务级最终报告
  - 输入数据基础统计文本框
- 大量新增特征的滚动文本框

## 当前模型配置

- 当前默认候选模型为 `arima`、`xgboost`、`linear`
- 为了快速测试，`lstm` 已临时从默认候选链路中移除
- `model_selection`、`model_training` 和前端模型摘要中，当前应看到 `linear` 而不是 `lstm`

## 约束

- 当前 `pipline` 不提供代码审批或在线修改代码
- 当前 `pipline` 不依赖 LLM，所有计划和步骤均由预定义流程驱动
`config_all` 说明：
- 默认读取仓库根目录的 `config_all.json`
- CLI 与 Web 都会先读这份 JSON，再用显式传入的字段覆盖
- 读取后的配置会写入 state 中的 `runtime_config` 和 `runtime_config_path`
