# pipline

`pipline` 是基于当前 `loongflow_DGagent` 结构复制出的一个新项目，目录组织、前后端入口和工件输出方式保持一致，但整个执行链已经改成纯代码驱动的时序预测 pipeline，不再依赖 LLM 生成计划、生成代码或生成总结。

## 目标

- 保留 `loongflow_DGagent` 的前后端骨架、subagent 分层和输出目录结构
- 去掉 LLM 参与的计划生成、代码审批和报告生成逻辑
- 使用固定的时序预测流程完成数据读取、规范化、特征工程、切分、建模、评估和总结

## 目录结构

- `main.py`: CLI 入口
- `webapp.py`: 轻量 Web 服务入口
- `orchestrator.py`: 固定 pipeline 编排器
- `agent_loop.py`: subagent 执行循环
- `state.py`: 全局状态
- `task_manager.py`: 任务清单
- `teams.py`: team 到 subagent 的映射
- `tools/`: 分析、评估、沙箱、工件输出工具
- `subagents/`: 各阶段子模块
- `web/`: 前端静态页面

## 固定执行链

### 数据处理任务

```text
User Query
  -> DGOrchestrator
  -> data_reading
  -> data_formatter
  -> data_analysis
  -> feature_engineering
  -> split_strategy
  -> datanorm
  -> preprocess
  -> summary
```

### 时序预测任务

```text
User Query
  -> DGOrchestrator
  -> data_reading
  -> data_formatter
  -> data_analysis
  -> iteration loop (max 10)
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

## 模块功能

### 编排与入口

- `main.py`：CLI 入口。读取 `config_all.json` 和命令行参数，初始化 state 并执行 orchestrator。
- `webapp.py`：Web 服务入口。对外提供配置、提交任务、查询任务详情等接口。
- `orchestrator.py`：根据 `query + start_stage/end_stage + skip_split` 计算最终 plan，并决定哪些步骤只执行一次、哪些步骤按 iteration 执行。
- `agent_loop.py`：串行执行 subagent，并在终端输出 step start/done/failed 日志。
- `state.py`：统一保存运行参数、plan、各步骤结果摘要和任务级路径。
- `task_manager.py`：维护当前任务的步骤清单与完成状态。

### 数据处理模块

- `subagents/data_reading.py`：按 `station=<unit>` 目录读取原始数据，拼出 ODS，并调用 `convert_ods_to_ds` 生成 DS。
- `subagents/data_formatter.py`：先对 DS 中的序列列做长度补齐和 NaN 填补，再展开序列列，按站点生成 `formatted_dataset_<station>.parquet`，并确定目标列、特征列、时间列等数据画像。
- `subagents/data_analysis.py`：对 formatted 数据做基础统计、平稳性分析和文本化统计摘要。
- `subagents/feature_engineering.py`：按站点逐个读取 formatted 数据并做特征工程，避免一次拼全量大表；当前默认只保留较轻量的 lag/diff 等特征。
- `subagents/split_strategy.py`：基于 formatted 数据做 train/val/test 切分，支持 `global_last_k`、`station_last_k`、`station_month_last_k`、`fixed_date`、`leave_stations_out`。
- `subagents/datanorm.py`：根据配置或规则决定是否标准化，以及使用 `off/zscore/minmax/auto` 中哪一种。
- `subagents/preprocess.py`：读取 engineered 数据和切分结果，按 train 集统计量完成标准化，输出模型可直接读取的 preprocessed parquet。

### 模型模块

- `subagents/model_selection.py`：根据当前 iteration 的特征策略生成候选模型和参数配置。
- `subagents/model_training.py`：读取 preprocessed 数据，训练候选模型并输出指标排序。
- `subagents/model_integration.py`：对候选模型结果做集成，生成 ensemble 结果。
- `subagents/evaluator.py`：比较单模型与集成效果，判断是否继续下一轮 iteration。
- `subagents/summary.py`：生成 iteration 级 `summary.md` 和任务级 `final_summary.md`。

### 工件与工具

- `tools/artifacts.py`：负责任务级和 iteration 级 parquet / markdown / code 工件落盘。
- `config.py`：定义路径、运行配置模板，并负责从 `config_all.json` 构建统一 runtime state。
- `config_all.json`：统一运行参数入口，适合手工调试和 agent 批量改参。

## 输出结构

```text
pipline/output/
  <task-id>/
    data/
      data_reading_ods_dataset.parquet
      data_reading_ds_dataset.parquet
      data_formatter_ds_cleaned.parquet
      formatted/
        formatted_dataset_<station>.parquet
      split/
        train_dataset.parquet
        val_dataset.parquet
        test_dataset.parquet
    final_summary.md
    <iteration-id>/
      feature_engineering_engineered_dataset_<station>.parquet
      preprocess_train_preprocessed.parquet
      preprocess_val_preprocessed.parquet
      preprocess_test_preprocessed.parquet
      model_training_executed_code.py
      summary.md
```

说明：

- 不再为每个 subagent 单独创建文件夹和 `result.json`
- 一次性产物统一保存在任务级 `data/` 目录
- 迭代型产物仍保存在对应 `iteration_*` 目录
- 无论任务是否进入模型训练，执行结束后都会生成 `final_summary.md`
- 任务级会额外保存 `task_plan.json`，供前端和后续复盘读取真实 plan/plan_meta

## 阶段执行说明

当前支持通过 `--start-stage` 和 `--end-stage` 只执行部分流程，也支持从中间产物继续启动。

### 合法组合

| start_stage | end_stage | 是否可用 | 说明 |
|---|---|---:|---|
| `data_reading` | `data_reading` | 是 | 只做原始目录读取和 ODS/DS 转换 |
| `data_reading` | `data_formatter` | 是 | 生成每站 formatted parquet |
| `data_reading` | `data_analysis` | 是 | 在 formatted 基础上做分析 |
| `data_reading` | `split_strategy` | 是 | 做到数据切分 |
| `data_reading` | `preprocess` | 是 | 完整数据处理到预处理结束 |
| `data_reading` | `model_training` / `summary` | 是 | 完整链路 |
| `data_formatter` | `data_formatter` | 是 | 输入必须是 DS 数据或任务目录 |
| `data_formatter` | `preprocess` | 是 | 从 DS 继续跑后续数据处理 |
| `data_analysis` | `data_analysis` | 是 | 输入必须是 formatted 数据或任务目录 |
| `data_analysis` | `preprocess` | 是 | 从 formatted 开始做到预处理 |
| `feature_engineering` | `feature_engineering` | 是 | 只做特征工程 |
| `feature_engineering` | `preprocess` | 是 | 特征工程后接预处理 |
| `split_strategy` | `split_strategy` | 是 | 只做数据切分 |
| `model_selection` | `summary` | 是 | 输入必须是 preprocessed 数据目录 |
| `model_training` | `summary` | 是 | 从已有预处理结果直接训练 |

### 当前不支持的组合

| start_stage | end_stage | 原因 |
|---|---|---|
| `split_strategy` | `preprocess` | `preprocess` 依赖 `feature_engineering` 输出的 engineered dataset，不能跳过 |
| `datanorm` | `preprocess` | 同样缺少 engineered dataset |
| 任意阶段 | 比 `start_stage` 更早的 `end_stage` | orchestrator 会直接报错 |

### `dataset_path` 输入要求

| start_stage | dataset_path 应指向 |
|---|---|
| `data_reading` | 原始站点目录，形如 `station=<unit>/...` |
| `data_formatter` | `ds_dataset.parquet` 文件，或包含 `data/data_reading_ds_dataset.parquet` 的任务目录 |
| `data_analysis` / `feature_engineering` / `split_strategy` / `datanorm` | 包含 `formatted_dataset_*.parquet` 的目录，或任务根目录 |
| `preprocess` | 包含 `feature_engineering_engineered_dataset_*.parquet` 的 iteration 目录，或包含这些工件的任务根目录 |
| `model_selection` / `model_training` / `model_integration` / `evaluator` / `summary` | 包含 `train_preprocessed.parquet` 的目录 |

## 使用方式

默认运行参数现在统一从 [config_all.json](/Users/monychen/Documents/demo-zjl/pipline/config_all.json) 读取，底层默认模板定义在 [config.py](/Users/monychen/Documents/demo-zjl/pipline/config.py) 的 `PIPELINE_RUNTIME_DEFAULTS` 中，例如：

- `train_ratio` / `val_ratio` / `test_ratio`
- `split_method`
- `input_length` / `output_length`
- `points_per_day`
- `normalization_policy`
- `max_iterations`

CLI 和 Web 都会先读取 `config_all.json`，再叠加用户显式传入的参数。要修改默认行为，优先改这一份 JSON，不需要再去 `main.py`、`webapp.py` 或 subagent 中分别找默认值。

### CLI

```bash
cd pipline
python main.py \
  --config-all ./config_all.json \
  --query "针对当前数据集构建一个时序预测模型" \
  --dataset-path ../data/ETTh1.csv \
  --target-col OT \
  --input-feature-cols TEMP,WS,WD,PREC,PWAT,SDWE,GHI \
  --train-ratio 8 \
  --val-ratio 2 \
  --test-ratio 1 \
  --normalization-policy minmax \
  --use-system-random \
  --max-iterations 3
```

常用可选参数：

- `--config-all`：统一运行配置 JSON 路径，默认读取仓库根目录 `config_all.json`
- `--use-system-random`：特征工程在不同 iteration 中使用 `SystemRandom` 做随机策略选择
- `--disable-system-random`：关闭 `SystemRandom`，改为确定性种子策略，便于复现
- `--train-ratio/--val-ratio/--test-ratio`：支持非归一化输入，程序会自动归一化，例如 `8/2/1`
- `--start-stage/--end-stage`：控制流程起止阶段
- `--formatter-unit`：在 `data_formatter` 阶段重新选择要处理的站点
- `--points-per-day`：每天采样点数，用于 DS 清洗阶段按“前一天同一时刻”填补 NaN
- `--skip-split`：跳过 `split_strategy`，用于只做数据处理或直接训练
- `--enable-split`：显式启用切分；当 `config_all.json` 里把 `skip_split` 设为 `true` 时，可用它覆盖回来

### 常见运行场景

#### 1. 完整数据处理到总结

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
python main.py \
  --config-all ./config_all.json \
  --query "只做数据处理" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/sample_station_data
```

说明：

- 适合 `start_stage=data_reading` 且 `end_stage=preprocess` 或 `summary`
- 如果 query 被识别为数据处理任务，`end_stage=summary` 只表示生成总结，不会进入模型训练

#### 2. 完整预测链路

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
python main.py \
  --config-all ./config_all.json \
  --query "构建一个时序预测模型" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/sample_station_data
```

说明：

- 这会从 `data_reading` 一直跑到 `summary`
- 是否切分、迭代轮数、窗口长度等参数默认都从 `config_all.json` 读取

#### 3. 只做 ODS/DS

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
python main.py \
  --config-all ./config_all.json \
  --query "只做ods ds处理" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/sample_station_data \
  --end-stage data_reading
```

#### 4. 从 DS 继续做到 formatted

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
python main.py \
  --config-all ./config_all.json \
  --query "只做格式化" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/output/<task-id> \
  --start-stage data_formatter \
  --end-stage data_formatter
```

#### 5. 从 formatted 继续做到 preprocess

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
python main.py \
  --config-all ./config_all.json \
  --query "继续做预处理" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/output/<task-id> \
  --start-stage feature_engineering \
  --end-stage preprocess
```

#### 6. 从已有 preprocessed 数据直接训练

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
python main.py \
  --config-all ./config_all.json \
  --query "直接训练模型" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/output/<task-id> \
  --start-stage model_training \
  --end-stage summary
```

### Web UI

```bash
cd pipline
python webapp.py --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

如果要切换运行配置，可以在表单中的 `Config All` 输入框填写另一个 JSON 路径。

前端支持：

- 提交预测或分析任务
- 配置数据集路径、目标列、输入列、窗口参数和切分比例
- 配置模型输入列，多个列用逗号分隔
- 配置标准化模式：`auto`、`off`、`zscore`、`minmax`
- 配置是否启用 `SystemRandom`
- 支持目录输入并拼接多个文件
- 浏览任务列表、统计分析结果、模型迭代结果、跨 iteration 最终集成结果和最终报告
- 表单会按 `query`、`start_stage/end_stage`、`split_method`、`skip_split` 自动折叠当前不生效的字段，例如只有在 `fixed_date` 时才显示 `Split Cutoff`

## 测试命令

下面的命令只用于验证数据处理链路，不要求进入模型训练。

先准备公共变量：

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

DATASET="/Users/monychen/Documents/demo-zjl/pipline/sample_station_data"
FEATURES="GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS,GHI_SOLARGIS_predict,TEMP_SOLARGIS_predict,WS_SOLARGIS_predict,WD_SOLARGIS_predict,ssrd_pos_1,ssrd_pos_2,ssrd_pos_3,t2m_pos_1,t2m_pos_2,t2m_pos_3,ssrd_pos_1_predict,ssrd_pos_2_predict,ssrd_pos_3_predict,t2m_pos_1_predict,t2m_pos_2_predict,t2m_pos_3_predict"
```

### 生成示例站点数据

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
python tools/generate_sample_station_parquet.py \
  --output-dir /Users/monychen/Documents/demo-zjl/pipline/sample_station_data \
  --stations 1,2,3 \
  --periods 384
```

### 只测试 ODS/DS

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --query "只做ods ds处理" \
  --dataset-path "$DATASET" \
  --unit "1,2" \
  --start-stage "data_reading" \
  --end-stage "data_reading" \
  --target-col "observe_power" \
  --input-feature-cols "$FEATURES"
```

### 测试完整数据处理到 preprocess

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --query "只做数据处理" \
  --dataset-path "$DATASET" \
  --unit "1,2" \
  --start-stage "data_reading" \
  --end-stage "preprocess" \
  --target-col "observe_power" \
  --input-feature-cols "$FEATURES" \
  --max-iterations 1
```

### 从已有任务目录继续测试 feature_engineering -> preprocess

先取倒数第二新的任务目录：

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
TASK_DIR="$(ls -dt output/* | sed -n '2p')"
echo "$TASK_DIR"
```

再执行：

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --query "只做feature split preprocess" \
  --dataset-path "$TASK_DIR" \
  --start-stage "feature_engineering" \
  --end-stage "preprocess" \
  --target-col "observe_power" \
  --max-iterations 1
```

### 只测试 split_strategy

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --query "只做split" \
  --dataset-path "$TASK_DIR" \
  --start-stage "split_strategy" \
  --end-stage "split_strategy" \
  --target-col "observe_power"
```

跑完后可重点检查这些产物：

- `output/<task-id>/data/data_reading_ods_dataset.parquet`
- `output/<task-id>/data/data_reading_ds_dataset.parquet`
- `output/<task-id>/data/data_formatter_ds_cleaned.parquet`
- `output/<task-id>/data/formatted/formatted_dataset_<station>.parquet`
- `output/<task-id>/data/split/train_dataset.parquet`
- `output/<task-id>/iteration_001/feature_engineering_engineered_dataset_<station>.parquet`
- `output/<task-id>/iteration_001/preprocess_train_preprocessed.parquet`
- `output/<task-id>/final_summary.md`

## 当前测试配置说明

- 当前默认候选模型为 `arima`、`xgboost`、`linear`
- 为了加快调试速度，原本的 `lstm` 已临时替换为 `linear`
- `linear` 支持快速多输入特征拟合，适合验证特征工程、切分、集成和前后端展示链路
- 如果后续需要恢复深度学习训练，再将 `linear` 切回 `lstm`

## 验证说明

- 本次只做代码改造，没有运行 Python 脚本或前端服务
- 按仓库约定，涉及 Python 运行验证需要你来执行
