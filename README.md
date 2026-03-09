# Time Series Multi-Agent System

## Directory

- `main.py`: CLI 入口
- `config.py`: API 配置、模型参数、目录路径
- `global_state.py`: GlobalState 共享状态总线
- `agent_loop.py`: `agent_loop` + `run_subagent`
- `task_manager.py`: 任务追踪（JSON 持久化）
- `orchestrator.py`: 主编排器（LLM 主智能体生成 plan + 调度）
- `subagents/`: 6 个专业子 Agent
- `tools/`: 19 个工具函数
- `data/`: 数据目录
- `output/`: 生成报告
- `.tasks/`: `global_state.json` + `tasks.json`

## Subagents

1. `data_reading`
2. `data_analysis`
3. `model_selection`
4. `model_training`
5. `result_integration`
6. `summary`

## Tools

- `tools/file_tools.py` (5): `read_csv`, `list_directory`, `detect_date`, `set_target`, `set_features`
- `tools/analysis_tools.py` (6): `statistics`, `stationarity`, `ACF`, `seasonality`, `trend`, `distribution`
- `tools/model_tools.py` (3): `train_arima`, `train_xgboost`, `train_lstm`
- `tools/eval_tools.py` (3): `compute_metrics`, `compare_models`, `create_ensemble`
- `tools/state_tools.py` (2): `read_state`, `write_state`

## Usage

推荐：使用 `5_DGagent/LLM` 提供的统一 LLM 接入方式。系统会自动扫描 `5_DGagent/LLM/.env`、`5_DGagent/.env` 和项目根目录 `.env`。

```bash
API_KEY=你的平台key
API_BASE=你的平台base_url
AGENT_MODEL=你的平台模型名
```

已内置自动加载，无需再通过 CLI 传 `--api-key/--api-base/--api-model`。

如果 `.env` 中使用的是 `OUT_OPENAI_API_KEY/OUT_OPENAI_API_BASE`，当前项目会自动切到 `get_llm(..., is_outside=True)`。

统计分析任务：

```bash
python main.py --query "针对etth数据集进行统计特性分析" --dataset-path data/ETTh1.csv
```

全流程建模任务：

```bash
python main.py --query "针对etth数据集构建时序预测模型" --dataset-path data/ETTh1.csv
```

打印完整状态：

```bash
python main.py --query "针对etth数据集构建时序预测模型" --dataset-path data/ETTh1.csv --print-state
```

闲聊或非时序请求（例如 `--query "你好"`）：

- 主编排器会识别为 `general_chat`，仅执行 `summary`。
- 不会触发 `model_selection/model_training/result_integration`。

说明：

- LLM 调用已统一复用 `5_DGagent/LLM/llm.py`，通过 `get_llm().invoke(...)` 执行。
- 根目录主流程和 `5_DGagent` 现已共用同一套 `.env` 扫描和代理配置逻辑。
- 未配置 Key 时：自动回退到本地规则，不影响主流程运行。

API 调试：

```bash
python llm_test.py
```

- `llm_test.py` 会直接通过 `get_llm().invoke(...)` 测试当前项目实际使用的 LLM 启动方式。
