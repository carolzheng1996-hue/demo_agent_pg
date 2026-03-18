# pipline Web API And UI Contract

## API

### `GET /api/config`

- 返回默认表单参数和最近任务列表

### `POST /api/jobs`

- 提交一个新的确定性 pipeline 任务
- 请求字段：
  - `query`
  - `dataset_path`
  - `dataset_name`
  - `unit`
  - `formatter_unit`
  - `start_stage`
  - `end_stage`
  - `skip_split`
  - `target_col`
  - `input_feature_cols`
  - `split_method`
  - `split_cutoff_date`
  - `split_test_units`
  - `train_ratio`
  - `val_ratio`
  - `test_ratio`
  - `input_length`
  - `output_length`
  - `time_increment`
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
  - `data_analysis` / `feature_engineering` / `split_strategy` / `datanorm` / `preprocess`：`formatted_dataset_*.parquet` 所在目录
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
- `skip_split=true` 时会跳过 `split_strategy`，并同时跳过 `model_integration` 与 `evaluator`

阶段控制参数：
- `start_stage`：允许从指定阶段启动流程
- `end_stage`：允许在指定阶段结束流程，仅做数据处理
- `formatter_unit`：允许在 `data_formatter` 阶段重新选择要处理的站点

切分比例说明：
- `train_ratio`、`val_ratio`、`test_ratio` 允许输入非归一化数值
- 后端会自动归一化，例如 `8 / 2 / 1` 会转换为 `8/11`、`2/11`、`1/11`

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
  - `dataset_profile`
  - `iteration_count`
  - `iterations`
  - `final_summary`
  - `iteration_history`
  - `cross_iteration_ensemble`

## UI

- 左侧展示任务列表和搜索
- 右侧表单提交预测任务或分析任务
- 表单支持配置标准化方式以及是否启用 `SystemRandom`
- 详情区展示：
  - 数据读取、数据规范化、统计分析、特征工程
  - 切分策略、标准化决策、预处理
  - 模型选择、训练、集成、评估
  - 跨 iteration 最终集成结果
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
