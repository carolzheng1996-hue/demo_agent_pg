# pipline

当前版本的 `pipline` 是一个固定流程的时序预测工程，不依赖 LLM 生成计划或代码。  
运行参数统一来自 `config_all.json`，也可以通过 CLI 显式覆盖。

## 当前流程

数据处理链：

```text
data_reading
-> data_formatter
-> data_analysis
-> feature_engineering
-> split_strategy
-> datanorm
-> preprocess
-> summary
```

建模链：

```text
data_reading
-> data_formatter
-> data_analysis
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

说明：

- `data_reading`：读取按 `station=<unit>` 分区的原始目录，并做 `ODS -> DS`
- `data_formatter`：直接复用 `data_loading_pg/ds_to_train.py` 中的长度补齐和 NaN 填补函数清洗 DS，再展开并按站点输出 formatted parquet
- `data_analysis`：生成基础统计与分析摘要
- `feature_engineering`：按站点做特征工程
- `split_strategy`：生成 train/val 切分
- `datanorm`：决定标准化策略
- `preprocess`：按 train 集统计量完成预处理，输出模型输入数据
- `model_selection` / `model_training` / `model_integration` / `evaluator`：完成候选模型训练、集成和评估
- `summary`：生成 iteration 级和任务级总结 markdown

阶段开关说明：

- `enable_feature_engineering=false`：跳过 `feature_engineering`，`preprocess` 直接读取 formatted 数据
- `enable_split=false`：跳过 `split_strategy`，`preprocess` 不再按 `row_ids` 切分
- `enable_normalization=false`：保留 `preprocess`，但只做清洗，不做缩放标准化

## 主要文件

- [main.py](/Users/monychen/Documents/demo-zjl/pipline/main.py)：CLI 入口
- [webapp.py](/Users/monychen/Documents/demo-zjl/pipline/webapp.py)：Web 服务入口
- [orchestrator.py](/Users/monychen/Documents/demo-zjl/pipline/orchestrator.py)：流程编排
- [config_all.json](/Users/monychen/Documents/demo-zjl/pipline/config_all.json)：统一运行配置
- [subagents/data_reading.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/data_reading.py)
- [subagents/data_formatter.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/data_formatter.py)
- [subagents/feature_engineering.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/feature_engineering.py)
- [subagents/split_strategy.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/split_strategy.py)
- [subagents/preprocess.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/preprocess.py)
- [subagents/model_training.py](/Users/monychen/Documents/demo-zjl/pipline/subagents/model_training.py)

## 输出目录

任务产物保存在：

```text
output/<task-id>/
```

其中常见文件有：

- `data/data_reading_ds_dataset.parquet`
- `data/formatted/formatted_dataset_<station>.parquet`
- `data/split/train_dataset.parquet`
- `data/split/val_dataset.parquet`
- `iteration_001/feature_engineering_engineered_dataset_<station>.parquet`
- `iteration_001/preprocess_train_preprocessed.parquet`
- `iteration_001/preprocess_val_preprocessed.parquet`
- `iteration_001/summary.md`
- `final_summary.md`
- `task_plan.json`

## 数据处理阶段说明

### 1. `data_reading`

功能：

- 读取按 `station=<unit>` 分区的原始目录
- 合并选中站点的数据
- 调用 `convert_ods_to_ds` 完成 `ODS -> DS`

输出：

- 转换后的 `DS`

存储路径：

- `output/<task-id>/data/data_reading_ds_dataset.parquet`

### 2. `data_formatter`

功能：

- 对 DS 序列列做长度检查
- 历史列按 `input_length` 头部补齐
- `*_predict` / `*_future` 列按 `output_length` 尾部补齐
- 按 `points_per_day` 用“前一天同一时刻优先”规则填补数组内 NaN
- 上述清洗函数直接复用 `data_loading_pg/ds_to_train.py`
- 将清洗后的 DS 展开成普通数值列
- 按站点分别输出 formatted parquet

输出：

- 每个站点一份 formatted parquet

存储路径：

- `output/<task-id>/data/formatted/formatted_dataset_<station>.parquet`

### 3. `data_analysis`

功能：

- 基于 formatted 数据做基础统计
- 生成缺失率、均值、方差、平稳性等分析摘要

输出：

- 不单独生成新的数据集文件
- 分析结果进入 state，并汇总到 summary markdown

存储路径：

- `output/<task-id>/iteration_001/summary.md`
- `output/<task-id>/final_summary.md`

### 4. `feature_engineering`

功能：

- 按站点读取 formatted 数据
- 生成 lag / rolling / diff / ewm 等特征
- 输出 engineered parquet

输出：

- 每个站点一份 engineered parquet

存储路径：

- `output/<task-id>/iteration_001/feature_engineering_engineered_dataset_<station>.parquet`

### 5. `split_strategy`

功能：

- 基于 formatted 数据切分 train / val
- 保存切分结果和对应 `__row_id__`

输出：

- train / val 两份切分数据

存储路径：

- `output/<task-id>/data/split/train_dataset.parquet`
- `output/<task-id>/data/split/val_dataset.parquet`

### 6. `datanorm`

功能：

- 根据配置或规则决定是否标准化
- 选择 `off` / `zscore` / `minmax` / `auto`

输出：

- 不单独生成新的数据集文件
- 决策结果写入 state，供 `preprocess` 使用

存储路径：

- `output/<task-id>/iteration_001/summary.md`
- `output/<task-id>/final_summary.md`

### 7. `preprocess`

功能：

- `enable_feature_engineering=true` 时读取 engineered 数据；否则直接读取 formatted 数据
- `enable_split=true` 时根据 `split_strategy` 保存的 `row_ids` 还原 train / val；否则只输出 train
- 做数值化、缺失值补齐、标准化
- 输出模型可直接读取的 preprocessed parquet

输出：

- 开启切分时输出 train / val
- 关闭切分时只输出 train

存储路径：

- `output/<task-id>/iteration_001/preprocess_train_preprocessed.parquet`
- `output/<task-id>/iteration_001/preprocess_val_preprocessed.parquet`

## 运行配置

默认运行参数来自 [config_all.json](/Users/monychen/Documents/demo-zjl/pipline/config_all.json)。

当前关键配置项：

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
- `train_ratio` / `val_ratio`
- `input_length`
- `output_length`
- `points_per_day`
- `enable_normalization`
- `normalization_policy`
- `max_iterations`

最常改的三个布尔开关：

- `enable_feature_engineering`
- `enable_split`
- `enable_normalization`

参数覆盖规则：

1. 先读 `config_all.json`
2. 再用 CLI 显式参数覆盖

## dataset_path 约束

- `start_stage=data_reading` 时，`dataset_path` 必须是原始站点目录，目录格式应为 `station=<unit>/...`
- `start_stage=data_formatter` 时，`dataset_path` 应指向 `ds_dataset.parquet` 文件或对应任务目录
- `start_stage=data_analysis` / `feature_engineering` / `split_strategy` / `datanorm` 时，`dataset_path` 应指向包含 `formatted_dataset_*.parquet` 的目录，或任务根目录
- `start_stage=preprocess` 时：
  - `enable_feature_engineering=true`：`dataset_path` 应指向包含 `feature_engineering_engineered_dataset_*.parquet` 的 iteration 目录，或对应任务根目录
  - `enable_feature_engineering=false`：`dataset_path` 应指向包含 `formatted_dataset_*.parquet` 的目录，或对应任务根目录
- `start_stage=model_training` / `summary` 时，`dataset_path` 应指向包含 `train_preprocessed.parquet` 的目录，或对应任务根目录

## 当前版本推荐命令

下面这些命令是适合当前版本的最常用示例。`convert_ods_to_ds` 相关列参数使用显式的 `col_ls`、`pred_col_ls`、`targ_col_ls`；`target_col` 和 `input_feature_cols` 继续作为下游格式化、分析和训练阶段的字段说明。

### 1. 只做 ODS/DS

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --config-all ./config_all.json \
  --query "只做ods ds处理" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/sample_station_data \
  --unit "1,2" \
  --start-stage data_reading \
  --end-stage data_reading \
  --col-ls GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS,ssrd_pos_1,ssrd_pos_2,ssrd_pos_3,t2m_pos_1,t2m_pos_2,t2m_pos_3 \
  --pred-col-ls GHI_SOLARGIS_predict,TEMP_SOLARGIS_predict,WS_SOLARGIS_predict,WD_SOLARGIS_predict,ssrd_pos_1_predict,ssrd_pos_2_predict,ssrd_pos_3_predict,t2m_pos_1_predict,t2m_pos_2_predict,t2m_pos_3_predict \
  --targ-col-ls observe_power \
  --target-col observe_power \
  --input-feature-cols GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS,GHI_SOLARGIS_predict,TEMP_SOLARGIS_predict,WS_SOLARGIS_predict,WD_SOLARGIS_predict,ssrd_pos_1,ssrd_pos_2,ssrd_pos_3,t2m_pos_1,t2m_pos_2,t2m_pos_3,ssrd_pos_1_predict,ssrd_pos_2_predict,ssrd_pos_3_predict,t2m_pos_1_predict,t2m_pos_2_predict,t2m_pos_3_predict
```

### 2. 只做数据处理到 preprocess

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --config-all ./config_all.json \
  --query "只做数据处理" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/sample_station_data \
  --unit "1,2" \
  --start-stage data_reading \
  --end-stage preprocess \
  --col-ls GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS,ssrd_pos_1,ssrd_pos_2,ssrd_pos_3,t2m_pos_1,t2m_pos_2,t2m_pos_3 \
  --pred-col-ls GHI_SOLARGIS_predict,TEMP_SOLARGIS_predict,WS_SOLARGIS_predict,WD_SOLARGIS_predict,ssrd_pos_1_predict,ssrd_pos_2_predict,ssrd_pos_3_predict,t2m_pos_1_predict,t2m_pos_2_predict,t2m_pos_3_predict \
  --targ-col-ls observe_power \
  --target-col observe_power \
  --input-feature-cols GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS,GHI_SOLARGIS_predict,TEMP_SOLARGIS_predict,WS_SOLARGIS_predict,WD_SOLARGIS_predict,ssrd_pos_1,ssrd_pos_2,ssrd_pos_3,t2m_pos_1,t2m_pos_2,t2m_pos_3,ssrd_pos_1_predict,ssrd_pos_2_predict,ssrd_pos_3_predict,t2m_pos_1_predict,t2m_pos_2_predict,t2m_pos_3_predict \
  --points-per-day 96 \
  --max-iterations 1
```

### 2a. 关闭特征工程、切分、标准化的最小命令

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --config-all ./config_all.json \
  --query "只做数据处理" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/sample_station_data \
  --col-ls GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS \
  --pred-col-ls "" \
  --targ-col-ls observe_power \
  --target-col observe_power \
  --input-feature-cols GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS \
  --disable-feature-engineering \
  --disable-split \
  --disable-normalization \
  --end-stage preprocess
```

### 3. 完整训练到 summary

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --config-all ./config_all.json \
  --query "构建一个时序预测模型" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/sample_station_data \
  --unit "1,2" \
  --start-stage data_reading \
  --end-stage summary \
  --col-ls GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS,ssrd_pos_1,ssrd_pos_2,ssrd_pos_3,t2m_pos_1,t2m_pos_2,t2m_pos_3 \
  --pred-col-ls GHI_SOLARGIS_predict,TEMP_SOLARGIS_predict,WS_SOLARGIS_predict,WD_SOLARGIS_predict,ssrd_pos_1_predict,ssrd_pos_2_predict,ssrd_pos_3_predict,t2m_pos_1_predict,t2m_pos_2_predict,t2m_pos_3_predict \
  --targ-col-ls observe_power \
  --target-col observe_power \
  --input-feature-cols GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS,GHI_SOLARGIS_predict,TEMP_SOLARGIS_predict,WS_SOLARGIS_predict,WD_SOLARGIS_predict,ssrd_pos_1,ssrd_pos_2,ssrd_pos_3,t2m_pos_1,t2m_pos_2,t2m_pos_3,ssrd_pos_1_predict,ssrd_pos_2_predict,ssrd_pos_3_predict,t2m_pos_1_predict,t2m_pos_2_predict,t2m_pos_3_predict \
  --points-per-day 96 \
  --max-iterations 1
```

### 4. 从 DS 继续做到 formatted

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --config-all ./config_all.json \
  --query "只做格式化" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/output/<task-id> \
  --start-stage data_formatter \
  --end-stage data_formatter \
  --target-col observe_power \
  --input-feature-cols GHI_real,GHI_SOLARGIS,TEMP_SOLARGIS,WS_SOLARGIS,WD_SOLARGIS,GHI_SOLARGIS_predict,TEMP_SOLARGIS_predict,WS_SOLARGIS_predict,WD_SOLARGIS_predict,ssrd_pos_1,ssrd_pos_2,ssrd_pos_3,t2m_pos_1,t2m_pos_2,t2m_pos_3,ssrd_pos_1_predict,ssrd_pos_2_predict,ssrd_pos_3_predict,t2m_pos_1_predict,t2m_pos_2_predict,t2m_pos_3_predict
```

### 5. 从 engineered 数据继续做到 preprocess

```bash
cd /Users/monychen/Documents/demo-zjl/pipline

python main.py \
  --config-all ./config_all.json \
  --query "继续做预处理" \
  --dataset-path /Users/monychen/Documents/demo-zjl/pipline/output/<task-id> \
  --start-stage preprocess \
  --end-stage preprocess \
  --target-col observe_power
```

如果 `enable_feature_engineering=false`，那么从中间阶段启动 `preprocess` 时，`dataset_path` 应该指向 formatted 数据目录，而不是 engineered 目录。

## Web

启动方式：

```bash
cd /Users/monychen/Documents/demo-zjl/pipline
python webapp.py --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

Web 表单与 CLI 共用同一套配置来源：

- 默认值来自 `config_all.json`
- 前端会自动折叠当前阶段不需要的字段
- `是否启用特征工程`、`是否切分`、`是否标准化` 是三个最主要的流程开关
- `切分日期` 只在 `split_method=fixed_date` 时显示
- `切分站点` 只在 `split_method=leave_stations_out` 时显示
- `每天采样点数` 只在 `data_reading/data_formatter` 阶段相关场景显示

## 说明

- `data_reading` 不再根据 `target_col` / `input_feature_cols` 自动推断 `convert_ods_to_ds` 的三组列；必须显式传 `col_ls`、`pred_col_ls`、`targ_col_ls`
- 如果未显式传 `target_col` 或 `input_feature_cols`，系统会在 `data_reading` 成功后分别用 `targ_col_ls` 和 `col_ls + pred_col_ls` 补给下游阶段
- 多行命令必须在每一行末尾加 `\`，否则 zsh 会把下一行当成新命令
- Python 运行验证需要你自己执行
