# Time Series Multi-Agent System

## Directory

- `main.py`: CLI 入口
- `config.py`: API 配置、模型参数、目录路径
- `global_state.py`: GlobalState 共享状态总线
- `agent_loop.py`: `agent_loop` + `run_subagent`
- `task_manager.py`: 任务追踪（JSON 持久化）
- `orchestrator.py`: 主编排器（意图识别 + 调度）
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

推荐：在项目根目录使用 `.env` 配置 OpenAI 兼容接口

```bash
API_KEY=你的平台key
API_BASE=你的平台base_url
AGENT_MODEL=你的平台模型名
```

已内置自动加载：程序启动时会读取项目根目录 `.env`。

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

不设环境变量，直接命令行传 key + base：

```bash
python main.py \
  --query "针对etth数据集构建时序预测模型" \
  --dataset-path data/ETTh1.csv \
  --api-key "你的平台key" \
  --api-base "你的平台base_url" \
  --api-model "你的平台模型名"
```

说明：

- 配置 `api_key` 后：`orchestrator` 意图识别、`model_selection`、`summary` 会优先走大模型（OpenAI 兼容接口）。
- `.env` 与 CLI 参数同时存在时：CLI 参数优先。
- 未配置 Key 时：自动回退到本地规则，不影响主流程运行。
