# 仓库功能说明与 LoongFlow 对比

## 1. 本仓库的主要功能

本仓库当前包含两套相关但定位不同的智能体实现。

### 1.1 根目录主流程

根目录是一套轻量级时序多智能体系统，主要面向以下两类任务：

- 对时间序列数据集做统计特性分析
- 对时间序列数据集做预测建模、训练与结果集成

核心模块包括：

- `orchestrator.py`
  - 主编排器
  - 基于 LLM 理解用户任务并生成执行计划
  - 决定执行哪些 subagent
- `agent_loop.py`
  - 顺序执行 subagent
- `global_state.py`
  - 共享状态总线
  - 在各 subagent 间传递数据、分析结果、模型结果、报告路径
- `task_manager.py`
  - 跟踪 plan 中各步骤的执行状态
- `subagents/`
  - `data_reading`: 数据读取与标准字段识别
  - `data_analysis`: 统计分析
  - `model_selection`: 候选模型选择
  - `model_training`: 模型训练
  - `result_integration`: 结果集成
  - `summary`: 汇总报告输出
- `tools/`
  - 数据读取、统计分析、模型训练、评估工具

### 1.2 `5_DGagent`

`5_DGagent` 是在根目录框架基础上进一步重构的数据分析智能体，偏向“可控、多子智能体、支持人工审批”的流程化设计。

它的目标包括：

- 主智能体基于 LLM 语义理解用户任务，不按关键词硬编码触发子模块
- 数据读取 subagent 支持“先生成代码、后人工确认、再执行”
- 支持数据分析、模型选择、模型训练、模型集成、总结报告的完整链路
- 维护独立的全局 state 和 task 清单
- 使用 `teams` 机制组织子智能体

`5_DGagent` 的主要模块：

- `5_DGagent/orchestrator.py`
  - 主智能体 `DGOrchestrator`
  - 负责理解任务、生成 `teams + subagents + plan_meta`
- `5_DGagent/agent_loop.py`
  - 顺序执行 subagent
  - 支持在审批节点 `pause_execution`
- `5_DGagent/state.py`
  - 独立共享状态
- `5_DGagent/task_manager.py`
  - 独立任务追踪
- `5_DGagent/teams.py`
  - team 到 subagent 的映射
- `5_DGagent/subagents/`
  - `data_reading`
  - `data_analysis`
  - `model_selection`
  - `model_training`
  - `model_integration`
  - `summary`
- `5_DGagent/LLM/`
  - 项目统一的大模型接入代码
  - 外层通过 `get_llm()` + `model.invoke(...)` 调用

## 2. 本仓库当前的主调用链

### 2.1 根目录主流程调用链

```text
User Query
  -> main.py
  -> Orchestrator
  -> plan generation
  -> agent_loop
  -> data_reading
  -> data_analysis
  -> model_selection (optional)
  -> model_training (optional)
  -> result_integration (optional)
  -> summary
```

### 2.2 `5_DGagent` 调用链

```text
User Query
  -> 5_DGagent/main.py
  -> 5_DGagent.main
  -> DGOrchestrator
  -> teams + normalized subagent plan
  -> agent_loop
  -> data_reading
     -> 生成数据处理代码
     -> 等待用户审批
     -> 审批后执行
  -> data_analysis
  -> model_selection (optional)
  -> model_training (optional)
  -> model_integration (optional)
  -> summary
```

## 3. LoongFlow `ml_agent` 的代码架构

参考代码：

- GitHub 目录：<https://github.com/baidu-baige/LoongFlow/tree/main/agents/ml_agent>
- README：<https://github.com/baidu-baige/LoongFlow/tree/main/agents/ml_agent>
- 主入口：<https://github.com/baidu-baige/LoongFlow/blob/main/agents/ml_agent/ml_evolve_agent.py>

从公开目录与 README 可以确定，`ml_agent` 不是一个简单的“planner 调 executor”的流程，而是一套面向机器学习任务的演化式执行框架。

### 3.1 目录分层

`ml_agent` 下主要包含：

- `planner/`
- `evocoder/`
- `executor/`
- `summary/`
- `evaluator/`
- `prompt/`
- `utils/`
- `examples/`
- `ml_evolve_agent.py`

这说明它的核心流程被拆成了：

- 规划
- 代码生成/修复
- 执行
- 评估
- 总结
- 配置样例

### 3.2 运行形态

根据 README，`ml_agent` 是一个“LLM 驱动代码生成 + 迭代优化”的 ML evolution framework，主要面向 MLE-Bench 和 Kaggle 风格任务。

它的输出结构是按 iteration 保存的：

```text
output/
  <task-uuid>/
    <iteration-id>/
      planner/
      evocoder/
      executor/
      summary/
  logs/
  database/
  evaluate/
```

这说明它不是单次执行，而是持续迭代、逐轮保存候选方案和结果。

### 3.3 演化式执行结构

从 README 中的配置可看出 `ml_agent` 至少包含以下稳定角色：

- `ml_planner`
- `ml_executor`
- `ml_summary`
- `evaluator`
- `database`

并且由 `evolve.max_iterations`、`target_score`、`population_size`、`num_islands` 等配置控制演化过程。

这类架构的核心不是“一次规划然后执行完”，而是：

```text
planner -> 生成候选方案
  -> coder/evocoder 生成代码
  -> executor 执行
  -> evaluator 评估
  -> 根据结果继续下一轮迭代
```

## 4. LoongFlow 是否有主智能体

有，但它的“主智能体”形态和 `learn-to-claude` 风格的 orchestrator 不完全一样。

### 4.1 结论

可以认为 LoongFlow `ml_agent` 有主智能体，主角色就是：

- `MLAgent` / `ml_evolve_agent.py` 对应的顶层 runner

它承担的是：

- 读取任务配置
- 初始化演化流程
- 注册 planner / executor / summary / evaluator
- 控制 iteration 轮次
- 管理输出目录、数据库、日志和 checkpoint

### 4.2 为什么说它“有主智能体”

因为它具备典型主控职责：

- 统一入口
- 统一调度各模块
- 统一控制执行轮次
- 统一管理全局上下文和输出

### 4.3 为什么又和 `5_DGagent` 的主智能体不同

`5_DGagent` 中的主智能体更像“语义规划器 + subagent 调度器”：

- 直接依据用户任务生成 plan
- 决定本轮调用哪些 subagent
- 更强调流程编排

LoongFlow 中的主控更像“演化运行器”：

- 更关注 iteration、evaluator、database、candidate solution
- 更像一个 evolution runtime，而不是一个单轮对话式 orchestrator

所以准确说法是：

- LoongFlow `ml_agent` 有主智能体/主控层
- 但它更偏 runner / evolution controller
- `5_DGagent` 更偏 orchestrator / planner

## 5. `5_DGagent` 可以借鉴 LoongFlow 的哪些点

`5_DGagent` 目前更强在流程可控、人工确认、状态清晰，但如果想进一步提高“自动优化建模效果”的能力，可以借鉴 LoongFlow 的以下设计。

### 5.1 引入 evaluator 驱动的闭环

当前 `5_DGagent` 的执行链路基本是一次性：

```text
data_reading -> data_analysis -> model_selection -> model_training -> model_integration -> summary
```

可以借鉴 LoongFlow，把它改成：

```text
model_selection
  -> model_training
  -> evaluator
  -> 若效果不佳则重新生成候选参数/模型
```

好处：

- 不只训练一次
- 可以自动尝试更优参数或不同模型组合
- 更接近真实 AutoML 流程

### 5.2 引入 iteration 概念

LoongFlow 的一个核心优点是每轮 iteration 都有独立产物。

`5_DGagent` 现在更像单轮流水线，后续可以增加：

- `iteration_id`
- 每轮独立输出目录
- 每轮模型候选与结果记录
- 每轮总结

好处：

- 能复盘每一轮效果
- 更容易做失败恢复
- 更容易做最佳方案回放

### 5.3 把数据科学流程拆得更细

LoongFlow 的 ML 流程更细粒度。

`5_DGagent` 现在的 `data_analysis` 和 `model_training` 之间还可以继续细分出：

- `feature_engineering`
- `split_strategy`
- `preprocess`
- `train_and_predict`
- `evaluator`
- `ensemble_strategy`

好处：

- 子智能体职责更单一
- 更容易对某一阶段单独优化
- 更方便引入不同任务模板

### 5.4 引入候选方案池

LoongFlow 不是只保留一个解，而是保留多个候选方案并演化。

`5_DGagent` 可以借鉴：

- 保存多个 `model_selection_result`
- 保存多组参数
- 保存每轮 top-k 结果
- 对 top-k 再做 ensemble

好处：

- 减少“一次 LLM 决策失误就全流程偏掉”的风险
- 更利于探索更优模型

### 5.5 引入更明确的执行工件结构

LoongFlow 每轮都会输出 `planner/evocoder/executor/summary`。

`5_DGagent` 可以借鉴为：

```text
5_DGagent/output/
  <task-id>/
    <iteration-id>/
      plan.json
      data_reading/
      data_analysis/
      model_selection/
      model_training/
      model_integration/
      summary.md
```

好处：

- 报障更容易
- 审计更清楚
- 更适合团队协作

### 5.6 在保留人工审批的前提下做自动优化

这是 `5_DGagent` 最值得借鉴 LoongFlow、但又不能完全照搬的地方。

LoongFlow 偏自动搜索，`5_DGagent` 偏人工可控。

比较合理的融合方式是：

- 继续保留 `data_reading` 的人工审批
- 对审批后的分析/建模阶段引入 evaluator + iteration
- 让自动优化只发生在“审批后、结构稳定”的区域

这样既保留安全性，又提高自动优化能力。

## 6. 两者的本质差异

### 6.1 LoongFlow `ml_agent`

更像：

- 面向 ML 竞赛/AutoML 的演化求解系统
- 核心是“不断试、不断评估、不断保留更优方案”

### 6.2 `5_DGagent`

更像：

- 面向业务型数据分析的多智能体编排系统
- 核心是“可控流程、人工确认、明确状态、可解释结果”

### 6.3 如何理解两者关系

可以把它们理解为：

- LoongFlow 更强在“优化”
- `5_DGagent` 更强在“编排与可控性”

如果后续希望把 `5_DGagent` 做强，最值得吸收的不是 LoongFlow 的全部框架，而是：

- evaluator 闭环
- iteration 产物管理
- 候选方案池
- 更细粒度的 ML stage 拆分

## 7. 建议的后续演进方向

对本仓库而言，一个比较稳妥的演进路径是：

1. 保持当前 `5_DGagent` 的主智能体 + teams + state 结构不变
2. 在 `model_selection/model_training/model_integration` 之间新增 `evaluator`
3. 增加 iteration 和 task artifact 目录
4. 引入 top-k 候选模型与多轮参数搜索
5. 最后再考虑是否需要演化式 database / checkpoint

这样可以在不破坏当前可控架构的前提下，逐步吸收 LoongFlow 的优势。
