# 5_DGagent

`5_DGagent` 是一个基于 `learn-claude-code` 风格代码结构改造的数据分析智能体，保留了 `agent loop + subagent + todo + teams + global state` 的组织方式，并把主智能体升级为基于 LLM 理解任务的编排器。

当前版本已切换为直接使用 `5_DGagent/LLM/` 中提供的 LLM 调用代码，统一采用：

```python
from LLM import get_llm
model = get_llm()
resp = model.invoke("hello")
```

系统会自动扫描 `5_DGagent/LLM/.env`、`5_DGagent/.env` 或仓库根目录 `.env`，不需要手动通过 CLI 传入 API Key、Base URL 或代理参数。

如果 `.env` 中提供的是 `OUT_OPENAI_API_KEY/OUT_OPENAI_API_BASE`，当前适配层会自动按 `is_outside=True` 调用 `get_llm(...)`。

## 目录结构

- `main.py`: CLI 入口
- `orchestrator.py`: 主智能体，负责基于用户任务语义生成 plan 和 team 路由
- `agent_loop.py`: 顺序执行 subagent 的核心循环
- `state.py`: 全局共享状态总线
- `task_manager.py`: 任务清单持久化
- `teams.py`: team 到 subagent 的映射
- `subagents/`: 6 个业务 subagent
- `tools/`: 分析、评估、沙箱执行工具
- `output/`: 汇总报告输出目录
- `.state/`: state 与 task 持久化目录

## 智能体组成

1. 主智能体 `DGOrchestrator`
- 输入用户任务描述和数据路径。
- 使用 LLM 直接理解任务目标，生成 `plan_meta + subagent plan`。
- 不按关键词硬编码触发 subagent。
- 输出选中的 `teams`、是否需要建模、是否需要数据集划分。

2. `data_reading` subagent
- 读取用户输入 CSV。
- 让 LLM 生成数据读取与标准化代码。
- 代码必须产出标准格式 `standardized_array` 和 `target_array`。
- 若主智能体判定任务需要训练，则必须提供 `train/val/test` 比例，并在生成代码中完成切分。
- 默认只生成代码并写入 state，等待用户确认。
- 用户加上 `--approve-generated-code` 后，才会在本地沙箱上下文中执行该代码。

3. `data_analysis` subagent
- 调用已有统计工具计算均值、方差、分布、趋势、平稳性、自相关、季节性等特征。
- 再让 LLM 基于任务生成额外分析计划和示例代码。

4. `model_selection` subagent
- 只在需要建模时触发。
- 结合任务和数据分析结果，从 `arima/xgboost/lstm` 中选择 3 个候选模型并生成参数。

5. `model_training` subagent
- 按选定参数训练 3 个模型。
- 输出每个模型的 `MSE/MAE/RMSE/MAPE`。

6. `model_integration` subagent
- 将三个模型预测值做简单平均。
- 输出最终集成预测和评估指标。

7. `summary` subagent
- 汇总任务计划、数据分析、模型选择、训练指标和集成结果。
- 输出 markdown 报告到 `5_DGagent/output/`。

## Teams 调用机制

- `discovery_team`: `data_reading -> data_analysis`
- `modeling_team`: `model_selection -> model_training -> model_integration`
- `reporting_team`: `summary`

主智能体会先让 LLM 输出建议的 `teams` 和 `subagents`，然后本地做依赖归一化：

- 所有任务最后都会执行 `summary`
- 只要涉及建模，就强制补齐 `data_reading -> data_analysis -> model_selection -> model_training -> model_integration -> summary`
- 纯分析任务会归一化为 `data_reading -> data_analysis -> summary`

## 全局 State

所有 subagent 共用 `DGGlobalState`，关键字段包括：

- `user_query`: 用户任务描述
- `dataset_path`: 数据路径
- `plan`: 当前执行计划
- `plan_meta`: 主智能体对任务的结构化理解
- `selected_teams`: 本次启用的 team
- `dataset_profile`: 数据集基础信息、目标列、特征列、是否切分、切分比例
- `data_reader_proposal`: 数据读取 subagent 生成的代码草案与审批状态
- `data_analysis_result`: 统计分析结果和 LLM 生成的扩展计划
- `model_selection_result`: 3 个模型及参数
- `model_training_result`: 训练结果和指标
- `model_integration_result`: 集成结果
- `summary_text` / `report_path`: 最终总结与报告路径

## 使用方式

注意：
- 主智能体计划生成依赖 `5_DGagent/LLM/` 中的 `get_llm().invoke()`。
- 系统启动时会自动扫描 `.env`，不需要手动传 `--api-key`、`--api-base`。
- 如果 `.env` 未配置可用模型或接口，系统会回退为安全的分析型默认计划。

### 1. 仅生成数据读取代码并等待确认

```bash
cd 5_DGagent
python main.py \
  --query "针对 ETTh1 数据做统计特性分析" \
  --dataset-path ../data/ETTh1.csv
```

行为：
- 主智能体生成计划
- `data_reading` 读取数据并生成标准化代码
- 程序输出待确认代码，不执行
- `state` 中写入 `awaiting_user_confirmation`

### 2. 用户确认后执行标准化和分析

```bash
cd 5_DGagent
python main.py \
  --query "针对 ETTh1 数据做统计特性分析" \
  --dataset-path ../data/ETTh1.csv \
  --approve-generated-code
```

行为：
- 在本地受限上下文中执行生成的数据读取代码
- 继续执行统计分析与总结
- 产出 markdown 报告

### 3. 建模任务

```bash
cd 5_DGagent
python main.py \
  --query "针对 ETTh1 数据构建一个时序预测模型" \
  --dataset-path ../data/ETTh1.csv \
  --approve-generated-code \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --test-ratio 0.2
```

行为：
- 主智能体识别为需要建模和切分数据
- `data_reading` 在生成代码里完成标准化和切分
- `model_selection` 给出 3 个模型及参数
- `model_training` 训练 3 个模型并计算指标
- `model_integration` 对 3 个模型结果取均值
- `summary` 输出报告

## 调用链

```text
User Query
  -> DGOrchestrator
  -> plan_meta + teams + normalized subagent plan
  -> agent_loop
  -> data_reading
  -> data_analysis
  -> model_selection (optional)
  -> model_training (optional)
  -> model_integration (optional)
  -> summary
```

## 用户确认机制

`data_reading` 是唯一被设计成默认需要人工确认的 subagent。

原因：
- 它会让 LLM 生成待执行代码。
- 为避免主智能体直接执行未经审查代码，框架默认停在“待审批”状态。
- 只有显式传入 `--approve-generated-code` 才会执行。

## 约束与当前实现说明

- 当前训练模型固定在 `arima/xgboost/lstm` 三类，便于和现有仓库工具复用。
- 集成方式按你的要求实现为“三模型输出简单平均”。
- 代码执行沙箱是轻量级受限 `exec` 上下文，不是容器级隔离；因此仍建议保留人工审查流程。
- `5_DGagent` 已不再使用仓库根目录的 `llm_client.py`。
- `5_DGagent` 已内置本地依赖工具模块，可直接在目录内执行 `python main.py`，不依赖上一级源码模块。
- 本次未在仓库内直接运行 Python 验证，遵循项目约定，由你本地执行命令确认。
