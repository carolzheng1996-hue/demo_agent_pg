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
  - `target_col`
  - `input_feature_cols`
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
