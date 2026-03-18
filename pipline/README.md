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
  -> split_strategy
  -> datanorm
  -> feature_engineering
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
  -> split_strategy
  -> datanorm
  -> iteration loop (max 10)
       -> feature_engineering
       -> preprocess
       -> model_selection
       -> model_training
       -> model_integration
       -> evaluator
       -> summary
```

## 确定性替换说明

- `orchestrator.py`：改为规则判断任务类型并生成固定 plan
- `data_reading.py`：支持按 `station=<unit>` 分区目录读取原始数据，并完成 ODS -> DS 转换
- `data_formatter.py`：将 DS 序列列展开后按站点分别保存 `formatted_dataset_<station>.parquet`
- `data_analysis.py`：基于本地分析工具生成统计信息和规则摘要
- `split_strategy.py`：按时间顺序完成训练/验证/测试切分
- `datanorm.py`：支持 `off/zscore/minmax/auto` 四种标准化策略选择
- `feature_engineering.py`：支持用户选择是否使用 `SystemRandom`，关闭后回退到可复现的确定性种子策略
- `model_selection.py`：按轮次和特征策略做固定候选模型扰动；当前为了快速测试，第三个候选模型临时使用 `linear`
- `summary.py`：直接输出结构化 markdown 报告

## 输出结构

```text
pipline/output/
  <task-id>/
    data/
      data_reading_ods_dataset.parquet
      data_reading_ds_dataset.parquet
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
| `data_analysis` / `feature_engineering` / `split_strategy` / `datanorm` / `preprocess` | 包含 `formatted_dataset_*.parquet` 的目录，或任务根目录 |
| `model_selection` / `model_training` / `model_integration` / `evaluator` / `summary` | 包含 `train_preprocessed.parquet` 的目录 |

## 使用方式

### CLI

```bash
cd pipline
python main.py \
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

- `--use-system-random`：特征工程在不同 iteration 中使用 `SystemRandom` 做随机策略选择
- `--disable-system-random`：关闭 `SystemRandom`，改为确定性种子策略，便于复现
- `--train-ratio/--val-ratio/--test-ratio`：支持非归一化输入，程序会自动归一化，例如 `8/2/1`
- `--start-stage/--end-stage`：控制流程起止阶段
- `--formatter-unit`：在 `data_formatter` 阶段重新选择要处理的站点
- `--skip-split`：跳过 `split_strategy`，用于只做数据处理或直接训练

### Web UI

```bash
cd pipline
python webapp.py --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

前端支持：

- 提交预测或分析任务
- 配置数据集路径、目标列、输入列、窗口参数和切分比例
- 配置模型输入列，多个列用逗号分隔
- 配置标准化模式：`auto`、`off`、`zscore`、`minmax`
- 配置是否启用 `SystemRandom`
- 支持目录输入并拼接多个文件
- 浏览任务列表、统计分析结果、模型迭代结果、跨 iteration 最终集成结果和最终报告

## 当前测试配置说明

- 当前默认候选模型为 `arima`、`xgboost`、`linear`
- 为了加快调试速度，原本的 `lstm` 已临时替换为 `linear`
- `linear` 支持快速多输入特征拟合，适合验证特征工程、切分、集成和前后端展示链路
- 如果后续需要恢复深度学习训练，再将 `linear` 切回 `lstm`

## 验证说明

- 本次只做代码改造，没有运行 Python 脚本或前端服务
- 按仓库约定，涉及 Python 运行验证需要你来执行
