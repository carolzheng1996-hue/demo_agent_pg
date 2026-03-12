# loongflow_DGagent

`loongflow_DGagent` 是在现有 `5_DGagent` 之外新增的一套 LoongFlow 风格数据分析智能体实现，不会覆盖原有 `5_DGagent` 代码。

## 目标

这一版重点参考 LoongFlow `ml_agent` 的设计，做了三类增强：

- 细化数据科学阶段：新增 `feature_engineering`、`split_strategy`、`preprocess`、`evaluator`
- 引入 evaluator 驱动的 iteration 闭环：建模任务不只训练一次，自动做多轮参数/模型尝试，最多 10 轮
- 增加明确的执行工件结构：每轮都会输出独立目录和 `plan.json` / `summary.md`

## 目录结构

- `main.py`: CLI 入口
- `webapp.py`: 轻量 Web 服务入口
- `orchestrator.py`: 主智能体与迭代控制器
- `agent_loop.py`: subagent 执行循环
- `state.py`: 全局状态
- `task_manager.py`: 任务清单
- `teams.py`: team 到 subagent 的映射
- `LLM/`: 复用现有项目 LLM 接入代码，使用 `get_llm().invoke(...)`
- `tools/`: 分析、评估、沙箱、工件输出工具
- `subagents/`: 各阶段子智能体
- `web/`: 前端静态页面（HTML/CSS/JS）
- `WEB_API_AND_UI_CONTRACT.md`: 前后端接口字段对照清单

## 子智能体阶段

### 非建模前置阶段

- `data_reading`
- `data_formatter`
- `data_analysis`
- `feature_engineering`
- `split_strategy`
- `datanorm`
- `preprocess`

### 建模迭代阶段

- `model_selection`
- `model_training`
- `model_integration`
- `evaluator`
- `summary`

## 调用链

### 统计分析任务

```text
User Query
  -> DGOrchestrator
  -> data_reading
  -> data_formatter
  -> data_analysis
  -> feature_engineering
  -> summary
```

### 建模任务

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
  -> iteration loop (max 10)
       -> model_selection
       -> model_training
       -> model_integration
       -> evaluator
       -> summary
```

## 人工审批与自动优化并存

这一版保留了前端可见的代码审批机制：

- `data_reading` 先完成多类型数据定位与加载
- `data_formatter` 生成数据规范化/处理代码
- `split_strategy` 在建模任务中额外生成切分代码
- 前端会直接展示待执行代码，用户点击确认后才继续执行
- 只有审批通过，后续自动优化流程才会继续

这意味着：

- 自动化发生在审批后的稳定建模阶段
- 不会绕过人工直接执行未经审查的 LLM 代码

## Iteration 机制

建模任务下：

- `evaluator` 会比较当前轮的最佳单模型和集成结果
- 如果仍有改进空间，则进入下一轮
- 默认最多 3 轮，可通过 `--max-iterations` 提高
- 最大不超过 10 轮

每轮都会记录：

- 候选模型与参数
- 训练指标
- 集成结果
- evaluator 判断
- 该轮 summary

## 工件目录结构

输出目录参考 LoongFlow 风格：

```text
loongflow_DGagent/output/
  <task-id>/
    final_summary.md
    iteration_history.json
    <iteration-id>/
      plan.json
      data_reading/
      data_formatter/
      data_analysis/
      feature_engineering/
      split_strategy/
      datanorm/
      preprocess/
      model_selection/
      model_training/
      model_integration/
      evaluator/
      summary/
      summary.md
```

## 使用方式

### 1. 仅生成读取代码并等待审批

```bash
cd loongflow_DGagent
python main.py \
  --query "针对 ETTh1 数据做统计特性分析" \
  --dataset-path ../data/ETTh1.csv \
  --target-col LULL
```

默认行为说明：

- CLI 默认从干净 state 启动，不会复用上一次审批结果
- 可通过 `--target-col` 显式指定目标列；多目标时用逗号分隔，例如 `--target-col LULL,OT`
- 若未传 `--approve-generated-code`，遇到 `data_formatter` 或 `split_strategy` 的待执行代码时，会直接在终端展示代码并询问是否继续
- 若代码生成有误，可以在 CLI 中选择 `m` 并输入反馈，让系统基于反馈重新生成当前 proposal
- 你输入 `y` 后，程序会在同一命令里继续执行到下一个审批点或最终结果

### 2. 审批后执行统计分析

```bash
cd loongflow_DGagent
python main.py \
  --query "针对 ETTh1 数据做统计特性分析" \
  --dataset-path ../data/ETTh1.csv \
  --approve-generated-code
```

如果你明确希望跳过终端确认并直接执行所有生成代码，再使用 `--approve-generated-code`。

### 3. 执行多轮建模优化

```bash
cd loongflow_DGagent
python main.py \
  --query "针对 ETTh1 数据构建一个时序预测模型" \
  --dataset-path ../data/ETTh1.csv \
  --approve-generated-code \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --test-ratio 0.2 \
  --max-iterations 5
```

### 4. 启动前端界面

```bash
cd loongflow_DGagent
python webapp.py --host 127.0.0.1 --port 8000
```

然后在浏览器打开 `http://127.0.0.1:8000`。

前端界面当前提供三类能力：

- 参数输入：填写 query、dataset path、dataset name、split ratio、max iterations
- 参数输入：支持显式填写目标列 `target_col`，多目标用逗号分隔
- 高级参数：填写 input/output length、time increment、normalization policy，以及自定义数据处理说明
- 审批提示：未勾选批准时，任务会停在 `data_formatter`，页面会展示待审批代码和该阶段结果
- 在线确认：若任务停在 `data_formatter` 或 `split_strategy`，页面会展示 LLM 生成代码和“确认并继续执行”按钮
- 在线修改：若生成代码有误，可在代码下方填写反馈并点击“按反馈重新生成”
- 结果展示：按 `task -> iteration -> subagent` 展示 `result.json`、`summary.md` 和任务级 `final_summary.md`

实现说明：

- 后端没有引入额外 Web 框架，直接使用 Python 标准库 `http.server`
- 页面数据直接读取 `loongflow_DGagent/output/` 下的工件目录，避免再维护一套独立数据库
- 如果你后续要扩展下载工件、在线批准、富文本渲染，可以继续在 `web/app.js` 和 `webapp.py` 上增量演进
- 如果你后续新增 subagent 或调整字段，建议先同步更新 `WEB_API_AND_UI_CONTRACT.md`

## 与 `5_DGagent` 的关系

- `5_DGagent` 保持原样不动
- `loongflow_DGagent` 是一套新的试验性增强版本
- 如果后续验证效果稳定，再考虑是否把其中一部分能力回迁到原架构
- `loongflow_DGagent` 也已内置本地依赖工具模块，可直接在目录内执行 `python main.py`，不依赖上一级源码模块
