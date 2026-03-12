# Todo

- [x] 按指定结构创建根模块与目录
- [x] 实现 GlobalState 共享状态总线（含 JSON 持久化）
- [x] 实现 TaskManager（任务追踪 + JSON 持久化）
- [x] 实现 19 个 tools 模块函数
- [x] 实现 6 个 subagent 并注册到 SUBAGENT_REGISTRY
- [x] 实现 orchestrator 意图识别与计划编排
- [x] 实现 agent_loop + run_subagent 核心循环
- [x] 实现 main.py CLI 入口
- [x] 更新 requirements 与 README
- [x] 更新 todo review

## 2026-03-05 OpenAI 接入改造

- [x] 新增统一 LLM 客户端，支持 OpenAI API Key / Base URL / Model
- [x] 将 orchestrator 意图识别接入 LLM（无 key 时回退规则）
- [x] 将模型选择与总结报告接入 LLM（无 key 时回退原逻辑）
- [x] 更新 CLI 参数与 README，说明如何以 OpenAI 模式运行
- [x] 更新 todo review，记录本次验证结论

## 2026-03-05 第三方平台 API 兼容修正

- [x] 支持 `API_KEY/API_BASE` 与 `OPENAI_API_BASE` 等常见环境变量别名
- [x] CLI 增加 `--api-base` 参数别名，兼容用户口径
- [x] LLM 启用条件改为“有 key 即可”，不强依赖 provider 值
- [x] 更新 README 为 OpenAI 兼容接口说明
- [x] 更新 lessons，记录本次纠正规则

## 2026-03-05 .env 配置改造

- [x] 引入 `.env` 自动加载机制（项目根目录）
- [x] 增加 `.env.example` 模板并忽略真实 `.env`
- [x] 更新 README，改为 `.env` 配置示例
- [x] 更新 lessons，记录本次纠正规则
- [x] 更新 todo review，记录改造结论

## 2026-03-05 第三方 API 适配与调试增强

- [x] 定位当前 `.env` 接入失败根因并给出结论
- [x] 在 LLM 客户端增加代理与参数兼容配置（适配 OpenAI 兼容平台）
- [x] 增强 `llm_test.py` 的错误诊断与模型探测输出
- [x] 更新 `.env.example` 与 README 调试说明
- [x] 更新 todo review，记录本次调试结论

## 2026-03-05 报告结果细节增强

- [x] 在评估层补充全模型与 ensemble 选择细节字段
- [x] 在 summary 报告中展示全部模型结果（逐模型指标）
- [x] 在 summary 报告中展示 ensemble 两个成员及各自指标/权重
- [x] 在 summary 报告中展示 ensemble 指标与相对最佳单模型差值
- [x] 运行命令验证报告内容并更新 review

## 2026-03-05 Orchestrator 主智能体计划生成

- [x] 将 orchestrator 从“关键词映射 plan”升级为“LLM 直接生成 plan”
- [x] 增加 plan 归一化与依赖校验，保证 plan 可执行
- [x] 增加计划来源与原因字段（`plan_source`、`plan_reason`）
- [x] 保留无 key 场景的回退规则
- [x] 更新 README 对应说明

## 2026-03-06 闲聊 query 误触发训练修复

- [ ] 定位 `query="你好"` 触发建模训练的根因
- [ ] 新增非时序意图并修正 fallback/LLM 计划生成策略
- [ ] 调整 plan 归一化逻辑，支持非训练的最小计划
- [ ] 更新 README 使用说明
- [ ] 更新 lessons 并补充 review 结论

## 2026-03-08 5_DGagent 数据分析智能体

- [x] 审查现有 agent 框架并确定 `5_DGagent` 的复用边界
- [x] 创建 `5_DGagent` 目录、模块分层与 teams/subagent 注册结构
- [x] 实现主智能体：基于 LLM 理解生成任务计划，不依赖关键词硬编码触发
- [x] 实现全局 state，维护数据集基础信息供各 subagent 共享
- [x] 实现数据读取 subagent：代码生成、标准数组转换、按任务决定是否划分数据集
- [x] 实现数据分析 subagent：基础统计 tools + LLM 计划/代码生成接口
- [x] 实现模型选择 subagent：给出三个候选时序模型及参数
- [x] 实现模型训练 subagent：训练三个模型并产出 MSE/MAE 等指标
- [x] 实现模型集成 subagent：对三个模型输出做平均集成
- [x] 实现 summary subagent：汇总结果并输出 md 文档
- [x] 编写 `5_DGagent/README.md`，说明使用方式、调用机制、确认节点与状态流转
- [x] 更新 todo review，记录本次实现范围与未执行验证项

## 2026-03-09 5_DGagent LLM 接入重构

- [x] 审查 `5_DGagent/LLM` 目录实现与当前调用点
- [x] 修正 `5_DGagent/LLM/llm.py` 为本仓库可直接加载 `.env` 与本地代理配置
- [x] 将 `5_DGagent` 中主智能体与各 subagent 替换为 `get_llm().invoke()` 调用方式
- [x] 移除 `5_DGagent` 中手动 API CLI 参数依赖
- [x] 更新 `5_DGagent/README.md` 使用说明
- [x] 更新 `tasks/lessons.md` 记录本次纠正规则
- [x] 更新 todo review，记录本次重构范围与未执行验证项

## 2026-03-09 根目录主流程 LLM 接入统一

- [x] 审查根目录 `orchestrator/subagents/main/llm_test` 的旧 LLM 依赖
- [x] 新增根目录 `llm_utils.py`，复用 `5_DGagent/LLM` 的 `get_llm().invoke()` 路径
- [x] 替换根目录 `orchestrator.py`、`subagents/model_selection.py`、`subagents/summary.py` 的 LLM 调用
- [x] 移除根目录 `main.py` 与 `llm_test.py` 的手动 API 参数依赖
- [x] 更新根目录 `README.md`
- [x] 更新 `tasks/lessons.md` 与 todo review

# Review

- 目录结构已按要求创建。
- 主流程采用：`orchestrator -> agent_loop -> subagents`。
- `GlobalState` 与 `TaskManager` 均支持 JSON 持久化到 `.tasks/`。
- 19 个 tools 已按分组实现，6 个 subagent 已接入注册表。
- 未执行 Python 脚本验证（按仓库约定由你执行）。
- 新增 `llm_client.py`，统一封装 OpenAI 文本/JSON 调用与异常回退。
- `orchestrator`、`model_selection`、`summary` 已接入 LLM，缺省自动回退本地规则。
- `main.py` 新增 `--api-provider/--api-key/--api-base-url/--api-model` 参数。
- `config.py` 支持 `OPENAI_API_KEY/OPENAI_BASE_URL` 以及 CLI 覆盖。
- `README.md` 和 `requirements.txt` 已更新。
- 未运行 Python 级验证（遵循仓库约定，需你本地执行验证命令）。
- 兼容第三方平台口径：支持 `API_KEY/API_BASE`、`--api-base`，并放宽 provider 约束。
- 已支持启动时自动读取项目根目录 `.env`（`config.py`）。
- 新增 `.env.example` 模板与 `.gitignore`（忽略真实 `.env`）。
- 新增 `.env` 占位文件，直接替换 key/base/model 即可运行。
- 已定位 `llm_test.py` 失败根因：环境代理 `http_proxy/https_proxy` 指向 `127.0.0.1:17890`，导致 SDK 连接失败。
- `LLMClient` 增加 `AGENT_TRUST_ENV_PROXY` 与 `AGENT_TIMEOUT_SECONDS` 配置，并加入 chat 参数兼容回退。
- `llm_test.py` 增加 `/models` 探测、错误分层输出（含 status code/response body 片段）和参数兼容重试。
- 报告增强：已输出全模型逐项指标、ensemble 两成员选择与权重、以及 ensemble 相对最佳单模型的指标差值。
- 验证通过：建模任务报告已包含上述明细；统计任务报告不再混入历史模型结果。
- Orchestrator 已支持主智能体按用户输入直接生成计划，并通过本地依赖校验修正为可执行 plan。
- 未执行 Python 级验证（遵循仓库约定，需你本地执行验证命令）。
- 已新增 `5_DGagent/`，复用现有 `llm_client`、分析工具和模型训练工具，重组为主智能体 + 6 个 subagent 的数据分析框架。
- `5_DGagent/orchestrator.py` 通过 LLM 直接生成 `teams + subagents + plan_meta`，不使用关键词硬编码路由。
- `5_DGagent/subagents/data_reading.py` 实现了“先生成代码、再人工确认、后沙箱执行”的双阶段流程；未审批时 agent loop 会暂停，不继续下游 subagent。
- `5_DGagent/subagents/model_selection.py`、`model_training.py`、`model_integration.py` 分别完成三模型选择、训练评估和平均集成。
- `5_DGagent/subagents/summary.py` 会输出 markdown 报告到 `5_DGagent/output/`。
- 已新增 `run_5_dgagent.py` 作为启动器，避免目录名 `5_DGagent` 在不同 Python 调用方式下的兼容性问题。
- 未执行 Python 级验证（遵循仓库约定，需你本地执行验证命令）。
- `5_DGagent` 的 LLM 接入已重构为直接复用 `5_DGagent/LLM/`，不再依赖自写 `llm_client.py`。
- `5_DGagent/LLM/llm.py` 已支持自动加载 `5_DGagent/LLM/.env`、`5_DGagent/.env` 和仓库根目录 `.env`。
- `5_DGagent/orchestrator.py` 与各 LLM subagent 已统一改为 `get_llm().invoke(...)` 路径，通过 `5_DGagent/llm_utils.py` 做文本/JSON 解析。
- `5_DGagent/main.py` 已移除 `--api-key/--api-base/--api-model` 等手动参数入口。
- 未执行 Python 级验证（遵循仓库约定，需你本地执行验证命令）。
- 根目录主流程也已统一复用 `5_DGagent/LLM`，新增 `llm_utils.py` 作为共享 invoke 封装。
- `main.py`、`orchestrator.py`、`subagents/model_selection.py`、`subagents/summary.py`、`llm_test.py` 已不再依赖手写 `LLMClient` 参数链路。
- 根目录 `README.md` 已改为 `.env` 自动扫描和 `get_llm().invoke(...)` 的统一说明。
- `llm_client.py` 目前仍保留在仓库中，但已不再被主流程或 `5_DGagent` 主流程引用。
- 未执行 Python 级验证（遵循仓库约定，需你本地执行验证命令）。

## 2026-03-09 仓库功能与 LoongFlow 对比文档

- [x] 审查当前仓库结构与现有 README
- [x] 整理根目录主流程与 `5_DGagent` 的功能说明
- [x] 整理 LoongFlow `ml_agent` 架构、主智能体判断与调用链
- [x] 输出独立 Markdown 文档
- [x] 更新 todo review

- 已新增 `REPO_FUNCTION_AND_LOONGFLOW_COMPARISON.md`，说明本仓库功能、根目录与 `5_DGagent` 的调用链、LoongFlow `ml_agent` 是否存在主智能体，以及 `5_DGagent` 可借鉴的演进方向。
- 已新增 `REPO_FUNCTION_AND_LOONGFLOW_COMPARISON_BRIEF.md`，作为适合答辩/PPT 口径的精简版说明文档。

## 2026-03-09 5_DGagent LoongFlow 风格优化

- [ ] 细化 `5_DGagent` 建模前后阶段，新增 `feature_engineering`、`split_strategy`、`preprocess`、`evaluator`
- [ ] 重构 `DGOrchestrator`，支持建模任务的 iteration 闭环与最多 10 轮自动优化
- [ ] 在保留 `data_reading` 人工审批前提下，将自动优化限制在审批后的建模阶段
- [ ] 增加明确的工件输出结构：`output/<task-id>/<iteration-id>/<step>/...`
- [ ] 更新 `summary` 逻辑，支持每轮总结与最终总结
- [ ] 更新 `README.md` 与 review，说明 LoongFlow 风格优化后的调用机制

## 2026-03-09 loongflow_DGagent 新目录实现

- [x] 撤回本轮误写入 `5_DGagent` 的 LoongFlow 风格改动
- [x] 创建新的 `loongflow_DGagent` 目录并复制基础骨架
- [x] 新增 `feature_engineering`、`split_strategy`、`preprocess`、`evaluator` 子智能体
- [x] 引入 evaluator 驱动的 iteration 闭环，支持最多 10 轮自动优化
- [x] 保留 `data_reading` 人工审批，并将自动优化限制在审批后阶段
- [x] 实现 `output/<task-id>/<iteration-id>/<step>/...` 工件输出结构
- [x] 新增 `run_loongflow_dgagent.py` 与 `loongflow_DGagent/README.md`
- [x] 更新 lessons，记录“新目录实现”纠正规则
- [x] 更新 todo review，记录本次新增目录与未执行验证项

- 已新增 `loongflow_DGagent/`，在不覆盖 `5_DGagent` 的前提下引入 LoongFlow 风格的 stage 拆分、iteration 闭环和工件目录结构。
- `loongflow_DGagent/orchestrator.py` 已支持审批后的多轮建模优化，默认 3 轮，最大不超过 10 轮。
- `loongflow_DGagent/output/<task-id>/<iteration-id>/` 下会输出 `plan.json`、各阶段结果和 `summary.md`，任务根目录会输出 `final_summary.md` 与 `iteration_history.json`。
- 未执行 Python 级验证（遵循仓库约定，需你本地执行验证命令）。
- `5_DGagent` 与 `loongflow_DGagent` 已补齐本地 `tools/file_tools.py`、`tools/model_tools.py`、`tools/analysis_tools.py`，并增加脚本直跑导入兜底。
- 两个目录现已支持在各自目录下直接执行 `python main.py`，不再要求以 `run_*.py` 作为主入口。

## 2026-03-09 5_DGagent 目录内建模链路修复

- [x] 定位 `python main.py` 建模任务未进入稳定训练链路的根因
- [x] 修复 `llm_utils` 参数兼容，支持当前调用点透传的 `max_tokens/temperature`
- [x] 修复 orchestrator fallback，在建模语义下生成包含训练步骤的计划
- [x] 修复 `data_reading` 回退代码构造，避免缩进错误导致标准化执行失败
- [x] 请用户运行建模命令验证 `model_selection/model_training/model_integration` 是否实际触发
- [x] 验证通过后提交并推送到 `origin/dev`

- 用户已确认 `5_DGagent` 目录内运行“看起来已经没问题”。
- 本轮修复聚焦三个点：LLM 调用参数兼容、建模语义 fallback 计划、数据读取代码清洗与回退执行。
- 运行产物 `.state/` 和 `__pycache__/` 已按既有仓库习惯保留在工作区，但本次提交仅纳入源码与任务记录。

## 2026-03-09 loongflow_DGagent data_reading 执行修复

- [x] 定位 `Generated data reading code did not produce required outputs` 的根因
- [x] 修复 `loongflow_DGagent/subagents/data_reading.py` 的代码清洗、回退执行和输出提取逻辑
- [x] 补强 `loongflow_DGagent/orchestrator.py` 的建模 fallback 计划，避免无 LLM 时误走分析链路
- [x] 请用户重新运行 `loongflow_DGagent` 命令验证
- [x] 验证通过后提交并推送到 `origin/dev`

- 用户已确认 `loongflow_DGagent` 重新运行后通过。
- 本轮修复与 `5_DGagent` 对齐，重点是稳定的数据读取代码执行与建模 fallback 计划。

## 2026-03-11 loongflow_DGagent 前端界面

- [x] 审查 `loongflow_DGagent` 的 CLI 参数、状态持久化与工件输出结构
- [x] 为 `loongflow_DGagent` 实现轻量 Web 服务，提供任务提交与结果查询接口
- [x] 实现前端页面，支持参数输入、审批提示与各 subagent 结果展示
- [x] 为新增前后端代码补充清晰注释，降低后续维护成本
- [x] 更新 `loongflow_DGagent/README.md` 与 todo review，说明启动与验证方式

- 已新增 `loongflow_DGagent/webapp.py`，基于 Python 标准库 `http.server` 提供 `/api/jobs`、`/api/tasks`、`/api/config` 接口，并在后台线程里调用现有 `DGOrchestrator`。
- 已新增 `loongflow_DGagent/web/` 静态前端，支持参数填写、任务提交、运行状态轮询、审批提示，以及按 `task -> iteration -> subagent` 展示工件结果。
- 页面展示数据直接从 `loongflow_DGagent/output/` 目录回放，不引入额外数据库，后续扩展下载工件或在线审批时改动面较小。
- 已新增 `run_loongflow_dgagent_web.py` 作为仓库根目录启动入口。
- 已执行 `git diff --check` 静态检查，未发现格式错误。
- 未执行 Python 级运行验证；遵循仓库约定，需要你本地启动 `python loongflow_DGagent/webapp.py --host 127.0.0.1 --port 8000` 或 `python run_loongflow_dgagent_web.py` 进行实际联调。
## 2026-03-11 loongflow_DGagent 前端优化
- [x] 定位前端标题与 prediction 渲染位置
- [x] 修改页面标题为 agent for time series analysis
- [x] 调整 prediction 展示，避免前端完整展开
- [x] 复查变更并补充 review

- 已将 `loongflow_DGagent/web/index.html` 的页面标题与主标题统一修改为 `agent for time series analysis`。
- 已将 `loongflow_DGagent/web/app.js` 的 step 结果渲染改为对 `prediction/predictions` 字段做摘要展示，仅保留前 10 个点和总量信息，避免前端完整展开全部预测值。
- 已复查前端差异；本轮仅改动标题文案与结果展示层，不影响后端接口和工件读取逻辑。

## 2026-03-11 loongflow_DGagent 前端布局重构
- [x] 重构页面为左侧历史任务栏 + 右侧主工作区
- [x] 移除顶部说明文案与 framework badge
- [x] 将任务详情按分析/处理/建模阶段分栏展示
- [x] 前端彻底隐藏长 prediction 数组，仅展示摘要
- [x] 复查差异并补充 review

- 已将页面重构为左侧历史任务栏、右侧工作区，移除了底部历史任务区块，并保留刷新入口。

## 2026-03-12 loongflow_DGagent 前端布局收紧
- [x] 定位右侧顶部“任务概览”栏与留白来源
- [x] 移除顶部概览栏，避免重复占位
- [x] 调整右侧模块为紧凑纵向排布，消除大块空白
- [x] 复查前端差异并补充 review

- 已移除 `loongflow_DGagent/web/index.html` 右侧顶部 `任务概览` 栏，改为在 `任务详情` 面板头部直接显示当前任务元信息。
- 已将 `loongflow_DGagent/web/styles.css` 的右侧 `top-grid` 从双列改为单列堆叠，避免“任务参数”与“当前任务”高度不一致时在中间留下大面积空白。
- 已同步精简 `loongflow_DGagent/web/app.js` 的详情渲染，去掉单独的“任务总览”卡片，让审批、统计和模型结果直接紧凑排列。
- 已移除顶部“页面直接读取...”说明文案与 `Zero extra frontend framework` badge，仅保留简洁标题栏。
- 已将每轮结果按“数据分析 / 数据处理 / 模型结果”三栏展示，减少单列超长滚动。
- 已将 `prediction/predictions` 改为仅展示点数、首尾值和范围摘要，不再渲染完整预测数组。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；未执行 Python 服务联调。

## 2026-03-11 loongflow_DGagent 侧栏增强
- [x] 为左侧历史任务栏增加搜索框
- [x] 将历史任务按日期分组展示
- [x] 调整任务项样式为聊天侧栏风格
- [x] 复查差异并执行静态检查

- 已为左侧任务栏增加搜索框，可按任务 ID、数据集路径、目标列和状态过滤。
- 已将任务列表按 `Today / Yesterday / YYYY-MM-DD` 分组，呈现更接近聊天产品的历史会话侧栏。
- 已调整任务项为更紧凑的圆角聊天列表卡片，并突出当前选中任务。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；未执行 Python 服务联调。

## 2026-03-11 loongflow_DGagent 结果展示收敛
- [x] 停止直接展示长 markdown 正文
- [x] 将统计特性与模型迭代拆成独立结果框
- [x] 支持运行中优先展示已完成的统计分析
- [x] 复查差异并执行静态检查

- 已停止在前端直接展开 `summary.md`、`final_summary.md` 等 markdown 正文，避免长预测值拉长页面。
- 已将详情区重构为“任务总览 / 统计特性 / 模型迭代结果”三个独立结果框，其中统计特性和模型迭代分开展示。
- 已在后端增加运行中进度监控，前端轮询时一旦工件目录出现统计分析结果，就会优先展示，无需等到全部建模完成。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；未执行 Python 服务联调。

## 2026-03-11 loongflow_DGagent 分析任务展示修正
- [x] 修复纯分析任务误显示模型迭代区块
- [x] 收紧详情卡片布局，避免长文本出框
- [x] 复查差异并执行静态检查

- 根因是前端把 `summary` 也当成了模型步骤；现已改为只有 `model_selection / model_training / model_integration / evaluator` 才会显示在模型迭代区。
- 已为详情卡片、计划文本、路径和值字段补充 `min-width: 0` 与自动换行规则，减少内容出框问题。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；未执行 Python 服务联调。

## 2026-03-11 loongflow_DGagent Web 状态隔离修复
- [x] 修复 Web 新任务复用旧 `task_id` 和旧结果的问题
- [x] 确认分析任务不会再混入历史建模结果
- [x] 复查差异并执行静态检查

- 根因是 `DGGlobalState` 在 Web 提交新任务时默认加载了 `.state/global_state.json`，把上一次任务的 `task_id`、iteration 和建模结果带进了新任务。
- 已为 `DGGlobalState` 增加 `load_existing` 参数，并在 `webapp.py` 中将每次 Web 新任务改为 `load_existing=False`，确保从干净 state 启动。
- 已执行 `git diff --check` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-11 loongflow_DGagent 建模任务显式标记
- [x] 后端在任务详情中显式返回是否需要建模
- [x] 前端对非建模任务完全隐藏模型迭代区块
- [x] 复查差异并执行静态检查

- 已在 `load_task_detail` 中增加 `task_type` 与 `requires_modeling` 字段，基于 `plan.json` 显式区分分析任务和建模任务。
- 前端现在只有在 `detail.requires_modeling === true` 时才会渲染“模型迭代结果”区块，分析任务将完全不显示该区域。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-11 loongflow_DGagent 统计报告卡片化
- [x] 将统计分析区改为报告卡片布局
- [x] 提炼关键指标卡与结构化摘要块
- [x] 优化样式层级与留白
- [x] 复查差异并执行静态检查

- 已将统计分析区改为“关键指标卡 + 结构化摘要卡”布局，不再沿用通用 step 卡片。
- 已提炼 `mean/std/missing ratio/stationary/target column/engineered features` 等关键指标，放在顶部 dashboard 卡片中。
- 已将数据概览、统计特性、特征工程、切分与预处理拆成独立报告卡，增强层级、留白和可读性。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-11 loongflow_DGagent 模型迭代卡片化
- [x] 将模型迭代区改为报告卡片布局
- [x] 提炼轮次核心指标与模型摘要
- [x] 优化建模区样式层级与可读性
- [x] 复查差异并执行静态检查

- 已将模型迭代区改为“每轮一张报告卡”的布局，顶部展示核心指标卡，下面展示模型选择、训练结果、集成结果、评估结论四类摘要。
- 已提炼 `best score / best single / ensemble score / strategy / models trained / continue search` 等轮次核心指标。
- 已为建模区新增独立的卡片背景、阴影和留白，整体风格与统计分析卡片保持一致。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-11 loongflow_DGagent old_subagents 功能整合
- [x] 审查 `old_subagents` 与当前子智能体链路
- [x] 将多类型数据加载能力整合进 `data_reading`
- [x] 新增 `data_formatter` 子智能体承接数据规范化与审批
- [x] 接通 orchestrator、registry、team、前端与 README
- [x] 复查差异并执行静态检查

- 已将 `old_subagents/data_loader.py` 的核心能力以当前架构方式合入 `subagents/data_reading.py`，支持从文件或目录中发现并加载 `.csv/.pkl/.npy` 数据。
- 已新增 `subagents/data_formatter.py`，承接原 `data_reading` 中的标准化/切分准备与 LLM 生成代码审批逻辑，成为新的前置子智能体。
- 已更新 `orchestrator.py`、`subagents/__init__.py`、`teams.py`、`summary.py`、`webapp.py`、`web/app.js`、`web/index.html` 与 `README.md`，让新子智能体进入计划生成、任务展示和文档说明。
- 已执行 `git diff --check` 与前端 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-11 loongflow_DGagent datanorm 与 datasplit 整合
- [x] 新增 `datanorm` 子智能体，按任务决定是否标准化
- [x] 将 `old_subagents/data_split.py` 的窗口与切分配置能力合入 `split_strategy`
- [x] 更新 preprocess 以遵循 datanorm 决策
- [x] 接通 orchestrator、前端和 README
- [x] 复查差异并执行静态检查

- 已新增 `subagents/datanorm.py`，根据任务描述、计划元信息和 foundation model 关键词判断是否需要标准化。
- 已将 `old_subagents/data_split.py` 中体现的窗口/步长/切分配置需求合入 `subagents/split_strategy.py`，新增 `window_config` 输出。
- 已更新 `subagents/preprocess.py`，只有在 `datanorm_result.should_normalize=true` 时才执行 z-score 标准化；否则保留原始数值仅做清洗补齐。
- 已更新 `orchestrator.py`、`subagents/__init__.py`、`teams.py`、`summary.py`、`web/app.js` 与 `README.md`，让 `datanorm` 进入执行链、报告与前端展示。
- 已执行 `git diff --check` 与 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-11 loongflow_DGagent 方案B表单参数化
- [x] 新增前端表单字段承接 old_subagents 交互选项
- [x] 新增 CLI/Web 参数并写入全局状态
- [x] 让 data_formatter/split_strategy/datanorm 读取用户显式配置
- [x] 更新文档与前端展示
- [x] 复查差异并执行静态检查

- 已新增表单/CLI 参数：`use_custom_processing`、`custom_processing_steps`、`input_length`、`output_length`、`time_increment`、`normalization_policy`。
- 已更新 `main.py` 与 `webapp.py`，将这些参数写入 `DGGlobalState`，供后续 subagent 显式读取。
- 已更新 `data_formatter.py`、`split_strategy.py`、`datanorm.py`，优先遵循用户显式配置，而不是完全依赖自动推断。
- 已更新前端表单与统计报告摘要，让新参数能输入、提交并在结果中查看。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-11 loongflow_DGagent 前端文案统一
- [x] 统一统计区与模型区摘要标签为中文
- [x] 统一布尔值与文件类型显示文案
- [x] 复查差异并执行静态检查

- 已将统计分析区和模型迭代区中的关键标签统一调整为中文，例如“已选文件 / 文件类型 / 切分策略 / 标准化模式 / 最佳得分”等。
- 已将布尔值统一显示为“是/否”，并对文件类型与标准化模式增加了更自然的中文/可读映射。
- 已执行 `node --check loongflow_DGagent/web/app.js` 与 `git diff --check` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-12 loongflow_DGagent 前端代码确认
- [x] 审查 `old_subagents` 中数据处理/切分代码确认交互并对齐当前 Web 流程
- [x] 为后端增加待确认代码读取与确认后继续执行接口
- [x] 在前端展示待确认代码并提供确认按钮
- [x] 复查差异并执行静态检查

- 已将 `old_subagents/data_formatter.py` 与 `old_subagents/data_split.py` 中“生成代码 -> 用户确认 -> 再执行”的交互迁移到当前 Web 流程。
- 已在 `webapp.py` 新增 `POST /api/tasks/<task_id>/approve`，并通过 `job_payload.json + approval_state.json` 复用同一 `task_id` 与同一份已确认代码继续执行。
- 已让 `data_formatter` 与 `split_strategy` 都支持把待执行代码写入工件，前端详情页会展示代码并提供“确认并继续执行”按钮。
- 已更新 `README.md` 与 `WEB_API_AND_UI_CONTRACT.md` 说明新的在线确认流程。
- 已执行 `git diff --check` 与 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 服务或脚本。

## 2026-03-12 loongflow_DGagent CLI 代码确认
- [x] 修正 CLI 默认复用旧 state 导致跳过确认的问题
- [x] 为 CLI 增加终端内代码展示与确认继续执行循环
- [x] 复查差异并执行静态检查

- 已将 `main.py` 改为 `load_existing=False`，避免 CLI 继承旧的 `.state/global_state.json` 而直接跳过代码确认。
- 已为 CLI 增加确认循环：遇到 `data_formatter` 或 `split_strategy` 的待执行代码时，会在终端打印代码并询问是否继续；输入 `y` 后在同一命令里继续执行。
- 已更新 `README.md` 说明新的 CLI 默认行为与 `--approve-generated-code` 的含义。
- 已执行 `git diff --check` 静态检查；遵循仓库约定，未替你运行 Python 命令验证。

## 2026-03-12 loongflow_DGagent 代码修改反馈
- [x] 检查目标列推断为何在用户指定 `lull` 时仍落到 `OT`
- [x] 为后端增加“用户反馈 -> 重新生成代码 proposal”的统一机制
- [x] 在 CLI 与 Web 中增加代码生成有误后的修改入口
- [x] 更新文档并执行静态检查

- 根因是 `tools/file_tools.py` 的 `set_target()` 默认优先 `OT`，而 `data_formatter` 在生成 proposal 前没有从用户 query 中解析显式目标列。
- 已新增 `tools/codegen.py`，提供 `infer_target_column_from_query()` 与 `regenerate_code_with_feedback()`，让目标列识别与代码重生成逻辑在 CLI/Web 共用。
- 已新增显式入参 `target_col`，并更新 `data_formatter.py`：优先使用用户指定目标列；未指定时，再从 query 中识别 `lull/LULL` 这类列名，不再直接落回 `OT`。多目标输入时使用逗号分隔，并兼容生成代码中的 `target_columns`。
- 已为 CLI 增加 `confirm / modify / cancel` 三段式交互；为 Web 增加 `POST /api/tasks/<task_id>/modify` 和前端反馈输入框，支持按用户反馈重生成当前待审批代码。
- 已更新 `README.md` 与 `WEB_API_AND_UI_CONTRACT.md` 说明修改入口。
- 已执行 `git diff --check` 与 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 命令或服务。

## 2026-03-12 loongflow_DGagent 反馈重生成兜底
- [x] 检查 Web “按反馈重新生成”失败原因
- [x] 为反馈重生成增加本地 fallback 与更清晰错误信息
- [x] 复查差异并执行静态检查

- 根因是 `tools/codegen.py` 的 `regenerate_code_with_feedback()` 完全依赖 LLM 二次重写，只要 `invoke_text()` 返回空，就会直接把 `None` 传回 Web，最终显示 `Failed to regenerate code from feedback.`。
- 已为 `regenerate_code_with_feedback()` 增加本地 fallback：`data_formatter` 会基于 proposal context 重新生成确定性处理代码，`split_strategy` 会基于 rows/ratios/window_config 重新生成确定性切分代码。
- 已同步收紧 Web 端错误信息，若未来仍失败，会明确指出是哪个 subagent 的反馈重生成失败。
- 已执行 `git diff --check` 与 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 命令或服务。

## 2026-03-12 loongflow_DGagent 侧栏任务显示修正
- [x] 检查左侧任务栏文件名竖排显示与状态文案问题
- [x] 优化任务项标题/路径换行策略
- [x] 将内部状态值映射为更友好的展示文案
- [x] 复查差异并执行静态检查

- 根因是侧栏通用文本规则对 `.task-item h3` 和路径文本启用了 `overflow-wrap:anywhere`，导致 `ETTh1.csv` 这类英文文件名按字符强制换行。
- 已将任务标题和路径改为单行省略显示，并保留 `title` 提示，避免出现“一字一行”的竖排效果。
- 已为 `awaiting_confirmation` 增加前端状态映射和单独样式，侧栏与详情区现显示为“待确认”，不再暴露内部枚举名。
- 已执行 `git diff --check` 与 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 服务。

## 2026-03-12 loongflow_DGagent 顶部栏信息收敛
- [x] 检查右侧顶部栏冗余标题和默认提示文案
- [x] 压缩顶部栏高度并改为更实用的概览文案
- [x] 复查差异并执行静态检查

- 已将右侧顶部栏从重复的大标题改为紧凑的“当前任务概览”，减少无效占位。
- 已把默认提示改成“选中左侧任务后，这里会显示任务 ID、状态和迭代轮次”，并将选中后元信息统一为中文短句。
- 已执行 `git diff --check` 与 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 服务。

## 2026-03-12 loongflow_DGagent 顶部栏空状态压缩
- [x] 检查顶部栏空状态占位过大的来源
- [x] 将说明文字并入右侧短提示并进一步压缩高度
- [x] 复查差异并执行静态检查

- 已移除顶部栏左侧的额外说明段落，仅保留“任务概览”标题，避免初始状态占用过高垂直空间。
- 已将空状态提示收敛到右侧单行文案，并增加省略策略，保证未选中任务时界面更干净。
- 已执行 `git diff --check` 与 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 服务。

## 2026-03-12 loongflow_DGagent 右上区块等高修正
- [x] 检查右上区域为何在初始状态被拉伸到接近半页
- [x] 修正 `top-grid` 等高拉伸并收紧空状态卡片高度
- [x] 复查差异并执行静态检查

- 根因不是顶部细栏，而是 `top-grid` 的默认 `align-items: stretch` 让左侧高表单与右侧“当前任务”面板被强制拉成同高。
- 已为 `top-grid` 改为顶部对齐，并让 `panel-status` 与空状态 `status-card` 保持紧凑高度，避免初始界面右上区域被无意义撑高。
- 已执行 `git diff --check` 与 `node --check loongflow_DGagent/web/app.js` 静态检查；遵循仓库约定，未替你运行 Python 服务。
