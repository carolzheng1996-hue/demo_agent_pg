# loongflow_DGagent Web API / UI Contract

本文件用于约束 `webapp.py` 返回的数据结构与 `web/app.js` 的消费方式，避免新增 subagent 或字段后出现前后端不匹配。

## 1. API Endpoints

### `GET /api/config`

返回：

- `defaults`
  - `query`
  - `dataset_path`
  - `dataset_name`
  - `approve_generated_code`
  - `train_ratio`
  - `val_ratio`
  - `test_ratio`
  - `max_iterations`
- `recent_tasks`
  - 当前后端会返回，但前端未直接使用

前端消费：

- `web/app.js` 的 `loadConfig()`

### `GET /api/tasks`

返回数组 `tasks`，每项包含：

- `task_id`
- `status`
- `iteration_count`
- `dataset_path`
- `target_column`
- `updated_at`

前端消费：

- 左侧历史任务栏
- 搜索 / 日期分组 / 当前选中状态

### `GET /api/tasks/<task_id>`

返回任务详情对象：

- `task_id`
- `task_dir`
- `status`
- `task_type`
- `requires_modeling`
- `dataset_profile`
- `iteration_count`
- `iterations`
- `pending_approval`
- `final_summary`
- `iteration_history`

前端消费：

- 任务总览卡
- 统计分析报告卡
- 模型迭代报告卡（仅当 `requires_modeling === true`）

### `GET /api/jobs/<job_id>`

返回任务运行态：

- `job_id`
- `status`
- `payload`
- `created_at`
- `updated_at`
- `result`（运行中/完成后）
  - `plan`
  - `task_id`
  - `awaiting_user_confirmation`
  - `summary_text`
  - `task_detail`
  - `tasks`（运行中监控用，前端当前未直接展示）

前端消费：

- 顶部状态卡
- 运行中实时刷新 `task_detail`

### `POST /api/jobs`

请求体：

- `query`
- `dataset_path`
- `dataset_name`
- `target_col`
- `approve_generated_code`
- `train_ratio`
- `val_ratio`
- `test_ratio`
- `max_iterations`

### `POST /api/tasks/<task_id>/approve`

用途：

- 确认当前任务里最新一个待审批的 LLM 生成代码
- 后端会沿用同一 `task_id` 和同一份代码继续执行，而不是重新生成一版代码

返回：

- `job_id`
- `status`

### `POST /api/tasks/<task_id>/modify`

用途：

- 对当前待审批代码提交用户反馈
- 后端基于原 proposal context 和反馈重生成代码，并直接覆盖当前待审批 proposal

请求体：

- `feedback`

## 2. Task Detail Contract

### 顶层字段

- `task_type`
  - 例如：`analysis_only` / `forecast_modeling`
- `requires_modeling`
  - 前端是否渲染模型迭代区的唯一显式开关
- `pending_approval`
  - 当前待用户确认的代码块；若为空则说明当前没有审批节点

### `dataset_profile`

由 `data_formatter` 主导写入，常见字段：

- `dataset_path`
- `selected_file`
- `shape`
- `standardized_shape`
- `columns`
- `date_column`
- `target_column`
- `target_columns`
- `feature_columns`
- `need_split`
- `split_ratios`

前端依赖：

- 总览卡
- 统计分析区顶部指标卡

### `iterations[*]`

每轮包含：

- `iteration_id`
- `plan`
  - 来自 `plan.json`
- `summary_markdown`
  - 前端不展开正文
- `steps`

### `steps[*]`

每步包含：

- `step_name`
- `result`
- `markdown`
- `files`

前端只消费：

- `step_name`
- `result`
- `files`（少量展示）
- `markdown` 仅作为“已生成报告”提示，不展开全文

## 3. Step Result Contract

### `data_reading`

当前 `result.json` 结构：

- `dataset_path`
- `is_directory`
- `selected_file`
- `supported_files`
- `description`
- `preview_rows`
- `shape`
- `columns`
- `file_type`

前端使用：

- `summarizeDataReading()`

注意：

- 前端**不能**再按旧的 `dataset_profile` 结构读取 `data_reading.result`

### `data_formatter`

当前 `result.json` 结构：

- `approved`
- `dataset_profile`
- `standardized_preview_rows`

未审批时：

- `approved = false`
- `proposal`
- `proposal.context`
- `proposal.feedback_history`
- `dataset_profile`

前端使用：

- `summarizeDataFormatter()`
- 审批提示来源于 `awaiting_user_confirmation.subagent === "data_formatter"`
- 详情区代码确认卡片直接展示 `proposal.generated_code`

### `data_analysis`

当前 `result.json` 结构：

- `base_analysis`
- `llm_plan`
- `llm_generated_code`
- `llm_findings`

前端使用：

- `statistics`
- `stationarity`

### `feature_engineering`

当前 `result.json` 结构：

- `engineered_columns`
- `shape`

### `split_strategy`

当前 `result.json` 结构：

- `approved`
- `proposal`（未审批时）
- `proposal.context`
- `proposal.feedback_history`
- `strategy`
- `rows`
- `ratios`
- `window_config`
  - `input_length`
  - `output_length`
  - `time_increment`
- `indices`

前端使用：

- `summarizeSplitStrategy()`
- 若 `approved = false`，详情区代码确认卡片展示 `proposal.generated_code`

### `datanorm`

当前 `result.json` 结构：

- `should_normalize`
- `recommended_mode`
- `reason`

前端使用：

- `summarizeDataNorm()`

### `preprocess`

当前 `result.json` 结构：

- `numeric_columns`
- `scaling`
- `normalization_applied`
- `normalization_mode`

前端使用：

- `summarizePreprocess()`

### `model_selection`

- `models`
- `selection_source`
- `iteration_index`

### `model_training`

- `results`
- `ranking`
- `target_column`

### `model_integration`

- `strategy`
- `member_models`
- `member_count`
- `predictions`

前端约束：

- 只能显示 `predictions.length`
- 不能展开完整预测数组

### `evaluator`

- `selected_strategy`
- `best_single_score`
- `ensemble_score`
- `iteration_index`
- `best_score`
- `best_previous_score`
- `improved`
- `stagnant_rounds`
- `should_continue`
- `stop_reason`

## 4. Frontend Rendering Rules

### 统计分析区

必须展示这些阶段（若存在）：

- `data_reading`
- `data_formatter`
- `data_analysis`
- `feature_engineering`
- `split_strategy`
- `datanorm`
- `preprocess`

### 模型迭代区

必须满足两个条件才展示：

- `detail.requires_modeling === true`
- 当前 iteration 中存在以下至少一个 step：
  - `model_selection`
  - `model_training`
  - `model_integration`
  - `evaluator`

### 审批提示

当前统一由：

- `result.awaiting_user_confirmation.subagent`

驱动展示

当前应为：

- `data_formatter`

### 长文本和长数组

前端必须遵守：

- 不展开 `summary.md` / `final_summary.md` 全文
- 不展开 `predictions` 全数组
- 只显示摘要字段

## 5. 后续新增字段/子智能体时的更新顺序

新增或修改 step/result 字段时，至少同步检查：

1. `subagents/<step>.py`
2. `webapp.py` 中 `load_task_detail()`
3. `web/app.js` 中统计区 / 模型区摘要函数
4. `README.md`
5. 本文件

如果 step 是新阶段，还要同步检查：

1. `subagents/__init__.py`
2. `orchestrator.py`
3. `teams.py`
4. 前端阶段分组函数 `getStageGroup()`
