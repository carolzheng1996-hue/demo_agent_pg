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
