// Frontend state is intentionally tiny:
// - `defaults` caches `/api/config` so the form can be reset locally.
// - `activeJobId` and `pollTimer` drive the live polling loop for a submitted task.
// - `activeTaskId` keeps the selected task highlighted in the sidebar.
// - the two signature fields avoid expensive DOM rerenders when polled payloads are unchanged.
const state = {
  defaults: null,
  activeJobId: null,
  activeTaskId: null,
  pollTimer: null,
  tasks: [],
  taskSearchQuery: "",
  lastJobStatusSignature: "",
  lastTaskDetailSignature: "",
};

const STAGE_ORDER = [
  "data_reading",
  "data_formatter",
  "data_analysis",
  "feature_engineering",
  "split_strategy",
  "datanorm",
  "preprocess",
  "model_selection",
  "model_training",
  "model_integration",
  "evaluator",
  "summary",
];

const MODEL_STAGES = new Set(["model_selection", "model_training", "model_integration", "evaluator"]);

// Centralizing DOM lookups here keeps the rest of the file focused on state transitions
// and rendering logic instead of repeated `getElementById` calls.
const elements = {
  jobForm: document.getElementById("jobForm"),
  submitButton: document.getElementById("submitButton"),
  loadDefaultsButton: document.getElementById("loadDefaultsButton"),
  refreshTasksButton: document.getElementById("refreshTasksButton"),
  taskSearchInput: document.getElementById("taskSearchInput"),
  taskList: document.getElementById("taskList"),
  taskDetail: document.getElementById("taskDetail"),
  taskDetailMeta: document.getElementById("taskDetailMeta"),
  jobStatusCard: document.getElementById("jobStatusCard"),
  jobStatusPill: document.getElementById("jobStatusPill"),
  stepTemplate: document.getElementById("stepTemplate"),
};

// Initial page load does three things:
// 1. bind event listeners
// 2. load default form values from backend config
// 3. fetch existing task history for the sidebar
document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await loadConfig();
  await refreshTasks();
});

// All interactive controls are wired once on boot.
function bindEvents() {
  elements.jobForm.addEventListener("submit", onSubmitJob);
  elements.loadDefaultsButton.addEventListener("click", () => fillForm(state.defaults));
  elements.refreshTasksButton.addEventListener("click", refreshTasks);
  elements.taskSearchInput.addEventListener("input", onTaskSearch);
  ["query", "start_stage", "end_stage", "split_method", "skip_split"].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", syncFormVisibility);
    document.getElementById(id)?.addEventListener("change", syncFormVisibility);
  });
}

// `/api/config` returns backend-provided defaults so CLI and Web UI share one source of truth.
async function loadConfig() {
  const payload = await fetchJson("/api/config");
  state.defaults = payload.defaults;
  fillForm(payload.defaults);
}

// Fill the form from backend defaults or the "reset to defaults" action.
// Checkbox fields need explicit boolean handling; all others can be assigned directly.
function fillForm(defaults) {
  if (!defaults) return;
  for (const [key, value] of Object.entries(defaults)) {
    const field = document.getElementById(key);
    if (!field) continue;
    if (field.type === "checkbox") {
      field.checked = Boolean(value);
    } else {
      field.value = value ?? "";
    }
  }
  syncFormVisibility();
}

// Form submission only enqueues a backend job.
// Detailed progress is handled separately by `pollJob`.
async function onSubmitJob(event) {
  event.preventDefault();
  const payload = formToPayload();
  setJobStatus("queued", "任务已提交，等待调度。");
  elements.submitButton.disabled = true;

  try {
    const result = await fetchJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.activeJobId = result.job_id;
    pollJob(result.job_id);
  } catch (error) {
    elements.submitButton.disabled = false;
    setJobStatus("failed", `提交失败: ${error.message}`);
  }
}

// Convert native form data into the backend payload shape.
// Checkboxes must be read from the DOM because `FormData` omits unchecked values.
function formToPayload() {
  const formData = new FormData(elements.jobForm);
  return Object.fromEntries(formData.entries());
}

function inferIntent(query) {
  const text = String(query || "").toLowerCase();
  const forecastKw = ["预测", "建模", "训练", "forecast", "model", "train"];
  const processingKw = ["处理", "预处理", "清洗", "格式化", "切分", "特征工程", "ods", "ds", "split", "preprocess", "feature"];
  const analysisKw = ["统计", "分析", "概览", "distribution", "analysis", "summary"];
  if (forecastKw.some((item) => text.includes(item))) return "forecast_modeling";
  if (processingKw.some((item) => text.includes(item))) return "data_processing";
  if (analysisKw.some((item) => text.includes(item))) return "analysis_only";
  return "forecast_modeling";
}

function stageRange(startStage, endStage) {
  const start = STAGE_ORDER.indexOf(startStage);
  const end = STAGE_ORDER.indexOf(endStage);
  if (start === -1 || end === -1 || start > end) return [];
  return STAGE_ORDER.slice(start, end + 1);
}

function toggleField(fieldId, visible) {
  const field = document.getElementById(fieldId);
  const label = field?.closest("label");
  if (!field || !label) return;
  label.style.display = visible ? "" : "none";
  field.disabled = !visible;
}

function syncFormVisibility() {
  const query = document.getElementById("query")?.value || "";
  const startStage = document.getElementById("start_stage")?.value || "data_reading";
  const endStage = document.getElementById("end_stage")?.value || "summary";
  const splitMethod = document.getElementById("split_method")?.value || "global_last_k";
  const skipSplit = (document.getElementById("skip_split")?.value || "false") === "true";
  const intent = inferIntent(query);
  const activeStages = stageRange(startStage, endStage);
  const includesSplit = activeStages.includes("split_strategy") && !skipSplit;
  const includesFeature = activeStages.includes("feature_engineering");
  const includesDsCleanup = activeStages.includes("data_reading") || activeStages.includes("data_formatter");
  const includesModel = activeStages.some((stage) => MODEL_STAGES.has(stage)) || (intent === "forecast_modeling" && endStage === "summary");

  toggleField("formatter_unit", startStage === "data_formatter");
  toggleField("train_ratio", includesSplit);
  toggleField("val_ratio", includesSplit);
  toggleField("test_ratio", includesSplit);
  toggleField("split_method", includesSplit);
  toggleField("split_cutoff_date", includesSplit && splitMethod === "fixed_date");
  toggleField("split_test_units", includesSplit && splitMethod === "leave_stations_out");
  toggleField("skip_split", activeStages.includes("split_strategy") || includesModel);
  toggleField("max_iterations", includesModel);
  toggleField("points_per_day", includesDsCleanup);
  toggleField("use_system_random", includesFeature);
}

// Poll the job endpoint until the task either completes or fails.
// The job response is the live bridge between "a submitted request" and "a persisted task".
function pollJob(jobId) {
  clearPollTimer();
  const tick = async () => {
    try {
      const job = await fetchJson(`/api/jobs/${jobId}`);
      renderJob(job);

      if (job.status === "completed") {
        elements.submitButton.disabled = false;
        clearPollTimer();
        await refreshTasks();
        if (job.result?.task_id) {
          await loadTaskDetail(job.result.task_id);
        }
        return;
      }

      if (job.status === "failed") {
        elements.submitButton.disabled = false;
        clearPollTimer();
        return;
      }

      state.pollTimer = window.setTimeout(tick, 1500);
    } catch (error) {
      elements.submitButton.disabled = false;
      clearPollTimer();
      setJobStatus("failed", `轮询失败: ${error.message}`);
    }
  };
  tick();
}

function clearPollTimer() {
  if (state.pollTimer) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

// Render the top-right job status panel and, when possible, update the detail view
// from the same polling payload so the user does not need to re-click the task manually.
function renderJob(job) {
  const result = job.result || {};
  const taskDetail = result.task_detail || null;
  setJobStatus(job.status, buildJobStatusMarkup(job, result), true);

  if (taskDetail?.task_id) {
    const shouldRefreshTaskList = state.activeTaskId !== taskDetail.task_id;
    state.activeTaskId = taskDetail.task_id;
    if (shouldRefreshTaskList) {
      renderTaskList(state.tasks);
    }
    const detailSignature = buildTaskDetailSignature(taskDetail);
    if (detailSignature !== state.lastTaskDetailSignature) {
      state.lastTaskDetailSignature = detailSignature;
      renderTaskDetail(taskDetail);
    }
  }
}

// The status card intentionally stays compact:
// it shows operational state, the selected task id, and the concrete pipeline plan.
// Detailed per-step data is rendered in the task detail panel below.
function buildJobStatusMarkup(job, result) {
  if (job.status === "failed") {
    return `
      <div class="error-block fade-in">
        <strong>${escapeHtml(job.error?.message || "任务失败")}</strong>
      </div>
      <pre>${escapeHtml(job.error?.traceback || "")}</pre>
    `;
  }

  const plan = Array.isArray(result.plan) ? result.plan.join(" -> ") : "尚未生成";
  const taskId = result.task_id || "待分配";
  const summary = result.summary_text
    ? `<div class="inline-note fade-in">已生成阶段总结；完整 markdown 保留在工件目录，前端不展开长文本。</div>`
    : "";

  return `
    <div class="detail-summary-top">
      <strong>Job ${escapeHtml(job.job_id)}</strong>
      <span class="muted-text">${escapeHtml(formatStatusLabel(job.status))}</span>
    </div>
    <div class="kv-grid">
      ${kvRow("任务状态", job.status)}
      ${kvRow("Task ID", taskId)}
      ${kvRow("执行计划", plan)}
    </div>
    ${summary}
  `;
}

// Skip DOM writes when status text is unchanged.
// This avoids visible flicker during high-frequency polling.
function setJobStatus(status, content, isHtml = false) {
  const signature = `${status}::${isHtml ? content : String(content)}`;
  if (signature === state.lastJobStatusSignature) {
    return;
  }
  state.lastJobStatusSignature = signature;
  elements.jobStatusPill.textContent = formatStatusLabel(status);
  elements.jobStatusPill.className = `status-pill ${statusClassName(status)}`;
  elements.jobStatusCard.classList.add("updating");
  if (isHtml) {
    elements.jobStatusCard.innerHTML = content;
  } else {
    elements.jobStatusCard.textContent = content;
  }
  window.setTimeout(() => {
    elements.jobStatusCard.classList.remove("updating");
  }, 180);
}

// Sidebar history is always rebuilt from backend data because task counts are small
// and the grouped rendering is simpler than trying to patch individual rows.
async function refreshTasks() {
  const payload = await fetchJson("/api/tasks");
  state.tasks = payload.tasks || [];
  renderTaskList(state.tasks);
}

// Tasks are grouped by day and rendered as clickable cards.
// The list is lightweight enough that full rerendering is acceptable here.
function renderTaskList(tasks) {
  const filteredTasks = filterTasks(tasks, state.taskSearchQuery);

  if (!filteredTasks.length) {
    elements.taskList.textContent = "暂无任务记录。";
    return;
  }

  const taskGroups = groupTasksByDate(filteredTasks);
  const fragment = document.createDocumentFragment();

  taskGroups.forEach(([label, groupTasks]) => {
    const group = document.createElement("section");
    group.className = "task-group fade-in";

    const title = document.createElement("div");
    title.className = "task-group-title";
    title.textContent = label;
    group.appendChild(title);

    groupTasks.forEach((task) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `task-item fade-in ${task.task_id === state.activeTaskId ? "active" : ""}`;
      const title = buildTaskTitle(task);
      button.innerHTML = `
        <div class="task-item-top">
          <h3 title="${escapeHtml(title)}">${escapeHtml(title)}</h3>
          <span class="status-pill ${statusClassName(task.status)}">${escapeHtml(formatStatusLabel(task.status))}</span>
        </div>
        <p class="task-item-subtitle">${escapeHtml(task.task_id)}</p>
        <p class="task-path" title="${escapeHtml(task.dataset_path || "未记录数据集路径")}">${escapeHtml(task.dataset_path || "未记录数据集路径")}</p>
        <p>${escapeHtml(task.is_directory ? "目录输入" : "单文件输入")} · 文件数: ${escapeHtml(String(task.available_file_count || 1))} · 目标列: ${escapeHtml((Array.isArray(task.target_columns) && task.target_columns.length ? task.target_columns.join(", ") : task.target_column) || "unknown")}</p>
        <p>更新时间: ${escapeHtml(formatTaskTime(task.updated_at || ""))}</p>
      `;
      button.addEventListener("click", () => loadTaskDetail(task.task_id));
      group.appendChild(button);
    });

    fragment.appendChild(group);
  });

  elements.taskList.innerHTML = "";
  elements.taskList.appendChild(fragment);
}

async function loadTaskDetail(taskId) {
  const detail = await fetchJson(`/api/tasks/${taskId}`);
  state.activeTaskId = taskId;
  renderTaskList(state.tasks);
  state.lastTaskDetailSignature = buildTaskDetailSignature(detail);
  renderTaskDetail(detail);
}

function onTaskSearch(event) {
  state.taskSearchQuery = event.target.value.trim().toLowerCase();
  renderTaskList(state.tasks);
}

// The detail area is split into:
// - a statistics panel for shared preprocessing / analysis steps
// - an iteration panel for modeling-only tasks
function renderTaskDetail(detail) {
  elements.taskDetailMeta.textContent = `${detail.task_id} | ${formatStatusLabel(detail.status)} | ${detail.iteration_count} 轮迭代`;
  elements.taskDetail.innerHTML = "";
  elements.taskDetail.appendChild(renderStatsPanel(detail));
  if (detail.requires_modeling) {
    if (detail.cross_iteration_ensemble?.available) {
      elements.taskDetail.appendChild(renderCrossIterationEnsemblePanel(detail.cross_iteration_ensemble));
    }
    elements.taskDetail.appendChild(renderModelIterationsPanel(detail));
  }
}

// Statistics panel merges the non-modeling steps across iterations.
// For deterministic forecasting tasks these steps define the data contract used by training.
function renderStatsPanel(detail) {
  const panel = document.createElement("section");
  panel.className = "detail-section fade-in";
  panel.innerHTML = `
    <div class="panel-head">
      <h2>统计特性</h2>
      <span class="muted-text">先展示数据统计分析，再进入模型阶段结果</span>
    </div>
  `;

  const statsGrid = document.createElement("div");
  statsGrid.className = "stats-report";
  const allSteps = detail.iterations.flatMap((iteration) => iteration.steps || []);
  const statsSteps = allSteps.filter((step) =>
    ["data_reading", "data_formatter", "data_analysis", "feature_engineering", "split_strategy", "datanorm", "preprocess"].includes(step.step_name),
  );

  if (!statsSteps.length) {
    panel.appendChild(renderSummaryOnlyStats(detail));
    return panel;
  }

  const stepMap = Object.fromEntries(uniqueByStepName(statsSteps).map((step) => [step.step_name, step]));
  statsGrid.appendChild(renderStatsHeroCards(detail, stepMap));
  statsGrid.appendChild(renderStatsSummaryCards(stepMap));
  panel.appendChild(statsGrid);
  return panel;
}

function renderSummaryOnlyStats(detail) {
  const wrapper = document.createElement("div");
  wrapper.className = "report-card-grid";

  const profile = detail.dataset_profile || {};
  const overview = buildSummarySection("任务概览", [
    { label: "数据集路径", value: profile.dataset_path || "—" },
    { label: "目标列", value: formatTargetColumns(profile) || "—" },
    { label: "站点", value: (profile.selected_units || []).join(", ") || "—" },
    { label: "起始阶段", value: profile.start_stage || detail.plan?.plan_meta?.start_stage || "—" },
    { label: "结束阶段", value: profile.end_stage || detail.plan?.plan_meta?.end_stage || "—" },
    { label: "是否跳过切分", value: formatBooleanDisplay(profile.skip_split) },
  ]);
  wrapper.appendChild(renderSummarySection(overview));

  if ((profile.data_artifacts || []).length) {
    wrapper.appendChild(
      renderSummarySection(
        buildSummarySection(
          "任务级数据产物",
          profile.data_artifacts.map((item) => ({
            label: item.name,
            value: item.relative_path || item.name,
          })),
        ),
      ),
    );
  }

  const summaryText =
    detail.iterations.find((iteration) => iteration.summary_markdown)?.summary_markdown ||
    detail.final_summary ||
    "";
  if (summaryText) {
    wrapper.appendChild(renderTextSummarySection("任务总结", summaryText));
  } else {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "当前任务尚未生成可展示的阶段摘要。";
    wrapper.appendChild(empty);
  }
  return wrapper;
}

// Hero cards surface the fastest-to-read dataset signals so users do not need to inspect raw JSON.
function renderStatsHeroCards(detail, stepMap) {
  const wrapper = document.createElement("div");
  wrapper.className = "hero-metrics";

  const analysis = stepMap.data_analysis?.result?.base_analysis || {};
  const statistics = analysis.statistics || {};
  const stationarity = analysis.stationarity || {};
  const featureEngineering = stepMap.feature_engineering?.result || {};
  const datasetProfile = detail.dataset_profile || stepMap.data_formatter?.result?.dataset_profile || {};

  const metrics = [
    { label: "均值", value: statistics.mean, accent: "warm" },
    { label: "标准差", value: statistics.std, accent: "cool" },
    { label: "缺失率", value: statistics.missing_ratio, accent: "neutral" },
    { label: "平稳性", value: formatBooleanDisplay(stationarity.is_stationary), accent: "success" },
    { label: "目标列", value: formatTargetColumns(datasetProfile) || "—", accent: "neutral" },
    { label: "新增特征数", value: featureEngineering.engineered_columns?.length ?? 0, accent: "warm" },
  ];

  metrics.forEach((metric) => wrapper.appendChild(renderMetricCard(metric)));
  return wrapper;
}

function renderMetricCard(metric) {
  const card = document.createElement("article");
  card.className = `metric-card ${metric.accent || "neutral"}`;
  card.innerHTML = `
    <span class="metric-label">${escapeHtml(metric.label)}</span>
    <strong class="metric-value">${escapeHtml(formatMetricValue(metric.value))}</strong>
  `;
  return card;
}

// Summary cards convert verbose backend JSON into curated human-readable sections.
// Each summarizer below selects only the most decision-relevant fields for its stage.
function renderStatsSummaryCards(stepMap) {
  const grid = document.createElement("div");
  grid.className = "report-card-grid";

  const inputStatisticsText = stepMap.data_analysis?.result?.input_statistics_text || "";
  if (inputStatisticsText) {
    grid.appendChild(renderTextSummarySection("输入数据基础统计", inputStatisticsText));
  }

  const sections = [
    buildSummarySection("数据概览", summarizeDataReading(stepMap.data_reading?.result || {})),
    buildSummarySection("数据规范化", summarizeDataFormatter(stepMap.data_formatter?.result || {})),
    buildSummarySection("统计特性", summarizeDataAnalysis(stepMap.data_analysis?.result || {})),
    buildSummarySection("特征工程", summarizeFeatureEngineering(stepMap.feature_engineering?.result || {})),
    buildSummarySection("切分策略", summarizeSplitStrategy(stepMap.split_strategy?.result || {})),
    buildSummarySection("标准化决策", summarizeDataNorm(stepMap.datanorm?.result || {})),
    buildSummarySection("预处理", summarizePreprocess(stepMap.preprocess?.result || {})),
  ].filter((section) => section.items.length);

  sections.forEach((section) => grid.appendChild(renderSummarySection(section)));
  return grid;
}

function buildSummarySection(title, items) {
  return { title, items };
}

function renderSummarySection(section) {
  const card = document.createElement("article");
  card.className = "report-card";

  const title = document.createElement("h3");
  title.textContent = section.title;
  card.appendChild(title);

  const list = document.createElement("div");
  list.className = "report-list";
  section.items.forEach(({ label, value }) => {
    const row = document.createElement("div");
    row.className = "report-row";
    const labelNode = document.createElement("span");
    labelNode.textContent = label;
    row.appendChild(labelNode);

    if (shouldUseScrollableValue(label, value)) {
      const textArea = document.createElement("textarea");
      textArea.className = "summary-textarea summary-textarea-compact";
      textArea.readOnly = true;
      textArea.value = String(value ?? "");
      row.appendChild(textArea);
    } else {
      const valueNode = document.createElement("strong");
      valueNode.textContent = formatMetricValue(value);
      row.appendChild(valueNode);
    }
    list.appendChild(row);
  });
  card.appendChild(list);

  return card;
}

function shouldUseScrollableValue(label, value) {
  const normalizedLabel = String(label || "").toLowerCase();
  if (normalizedLabel.includes("新增特征列")) return true;
  if (typeof value !== "string") return false;
  return value.length > 120 || value.includes("\n");
}

function renderTextSummarySection(titleText, valueText) {
  const card = document.createElement("article");
  card.className = "report-card report-card-wide";

  const title = document.createElement("h3");
  title.textContent = titleText;
  card.appendChild(title);

  const textArea = document.createElement("textarea");
  textArea.className = "summary-textarea";
  textArea.readOnly = true;
  textArea.value = valueText;
  card.appendChild(textArea);

  return card;
}

// The following `summarize*` helpers intentionally flatten backend result payloads into
// short label/value rows. They are presentation adapters, not sources of truth.
function summarizeDataReading(result) {
  const shape = result.shape;
  const previewRows = result.preview_rows || [];
  const supportedFiles = Array.isArray(result.supported_files) ? result.supported_files : [];
  return [
    { label: "输入类型", value: result.is_directory ? "目录" : "单文件" },
    { label: "发现文件数", value: supportedFiles.length || 1 },
    { label: "已加载文件", value: (result.loaded_files || []).join(", ") || "—" },
    { label: "文件类型", value: formatFileType(result.file_type) },
    { label: "数据形状", value: shape ? JSON.stringify(shape) : "—" },
    { label: "字段数", value: result.columns?.length ?? 0 },
    { label: "预览行数", value: previewRows.length },
  ];
}

function summarizeDataAnalysis(result) {
  const analysis = result.base_analysis || {};
  const stats = analysis.statistics || {};
  const stationarity = analysis.stationarity || {};
  return [
    { label: "中位数", value: stats.median },
    { label: "最小值", value: stats.min },
    { label: "最大值", value: stats.max },
    { label: "ADF p 值", value: stationarity.p_value },
    { label: "ADF 统计量", value: stationarity.adf_stat },
  ].filter((item) => item.value !== undefined);
}

function summarizeDataFormatter(result) {
  const profile = result.dataset_profile || {};
  return [
    { label: "输入类型", value: profile.is_directory ? "目录" : "单文件" },
    { label: "发现文件数", value: profile.available_file_count ?? 0 },
    { label: "目标列", value: formatTargetColumns(profile) || "—" },
    { label: "用户输入列", value: (profile.input_feature_columns || []).join(", ") || "—" },
    { label: "时间列", value: profile.date_column || "—" },
    { label: "时间派生列", value: (profile.derived_time_columns || []).join(", ") || "—" },
    { label: "特征列", value: (profile.feature_columns || []).join(", ") || "—" },
    { label: "规范化形状", value: profile.standardized_shape ? JSON.stringify(profile.standardized_shape) : "—" },
    { label: "按时间排序", value: formatBooleanDisplay(profile.sorted_by_time) },
    { label: "是否切分", value: formatBooleanDisplay(profile.need_split) },
    { label: "处理来源", value: result.pipeline_source || "deterministic_formatter" },
  ];
}

function summarizeFeatureEngineering(result) {
  return [
    { label: "特征策略", value: (result.selected_methods || []).join(", ") || "—" },
    { label: "随机模式", value: result.random_mode || "—" },
    { label: "输入源列", value: (result.source_columns || []).join(", ") || "—" },
    { label: "新增特征列", value: (result.engineered_columns || []).join("\n") || "—" },
    { label: "新增特征数", value: result.engineered_feature_count },
    { label: "输出形状", value: result.shape ? JSON.stringify(result.shape) : "—" },
  ];
}

function summarizeSplitStrategy(splitResult) {
  const ratios = splitResult.ratios || {};
  return [
    { label: "切分策略", value: splitResult.strategy || "—" },
    { label: "训练/验证/测试比例", value: Object.keys(ratios).length ? `${ratios.train_ratio} / ${ratios.val_ratio} / ${ratios.test_ratio}` : "—" },
    { label: "输入长度", value: splitResult.window_config?.input_length },
    { label: "输出长度", value: splitResult.window_config?.output_length },
    { label: "滑窗步长", value: splitResult.window_config?.time_increment },
    { label: "处理来源", value: splitResult.pipeline_source || "deterministic_splitter" },
  ].filter((item) => item.value !== undefined);
}

function summarizeDataNorm(result) {
  return [
    { label: "是否标准化", value: formatBooleanDisplay(result.should_normalize) },
    { label: "标准化模式", value: formatNormalizationMode(result.recommended_mode) },
    { label: "决策原因", value: result.reason || "—" },
  ];
}

function summarizePreprocess(preprocessResult) {
  const scaling = preprocessResult.scaling || {};
  return [
    { label: "已执行标准化", value: formatBooleanDisplay(preprocessResult.normalization_applied) },
    { label: "标准化模式", value: formatNormalizationMode(preprocessResult.normalization_mode) },
    { label: "已缩放列数", value: Object.keys(scaling).length },
    { label: "数值列数", value: preprocessResult.numeric_columns?.length ?? 0 },
    { label: "增量拟合", value: formatBooleanDisplay(preprocessResult.used_partial_fit) },
  ].filter((item) => item.value !== undefined);
}

// Model iteration cards are rendered one iteration at a time so users can compare
// how feature selection and model search evolve across the optimization loop.
function renderModelIterationsPanel(detail) {
  const panel = document.createElement("section");
  panel.className = "detail-section fade-in";
  panel.innerHTML = `
    <div class="panel-head">
      <h2>模型迭代结果</h2>
      <span class="muted-text">不同轮次的建模摘要与评估结果</span>
    </div>
  `;

  const iterationList = document.createElement("div");
  iterationList.className = "iteration-report-list";

  detail.iterations.forEach((iteration) => {
    const modelSteps = (iteration.steps || []).filter((step) => isModelStep(step.step_name));

    if (!modelSteps.length) {
      return;
    }

    const stepMap = Object.fromEntries(modelSteps.map((step) => [step.step_name, step]));

    const card = document.createElement("section");
    card.className = "iteration-report-card fade-in";
    card.innerHTML = `
      <div class="detail-summary-top">
        <h3>${escapeHtml(iteration.iteration_id)}</h3>
        <span class="muted-text">${escapeHtml((iteration.plan?.plan || []).join(" -> ") || "未记录计划")}</span>
      </div>
    `;

    card.appendChild(renderIterationHeroCards(stepMap));
    card.appendChild(renderIterationSummaryCards(stepMap));
    iterationList.appendChild(card);
  });

  if (!iterationList.children.length) {
    const summaryText =
      detail.iterations.find((iteration) => iteration.summary_markdown)?.summary_markdown ||
      detail.final_summary ||
      "";
    if (summaryText) {
      panel.appendChild(renderTextSummarySection("建模总结", summaryText));
    } else {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "模型训练等 subagent 尚未开始执行。";
      panel.appendChild(empty);
    }
    return panel;
  }

  panel.appendChild(iterationList);
  return panel;
}

// Hero metrics in the modeling section emphasize decision variables:
// current best score, best single model, ensemble score, chosen strategy, and stop/continue signal.
function renderIterationHeroCards(stepMap) {
  const wrapper = document.createElement("div");
  wrapper.className = "hero-metrics";

  const evaluator = stepMap.evaluator?.result || {};
  const integration = stepMap.model_integration?.result || {};
  const training = stepMap.model_training?.result || {};

  const metrics = [
    { label: "最佳得分", value: evaluator.best_score, accent: "cool" },
    { label: "最佳单模型", value: evaluator.best_single_score, accent: "neutral" },
    { label: "集成得分", value: evaluator.ensemble_score, accent: "warm" },
    { label: "策略", value: evaluator.selected_strategy || integration.strategy || "—", accent: "success" },
    { label: "已训练模型数", value: training.results?.length ?? 0, accent: "neutral" },
    { label: "继续搜索", value: formatBooleanDisplay(evaluator.should_continue), accent: "warm" },
  ];

  metrics.forEach((metric) => wrapper.appendChild(renderMetricCard(metric)));
  return wrapper;
}

function renderIterationSummaryCards(stepMap) {
  const grid = document.createElement("div");
  grid.className = "report-card-grid";

  const sections = [
    buildSummarySection("模型选择", summarizeModelSelection(stepMap.model_selection?.result || {})),
    buildSummarySection("训练结果", summarizeModelTraining(stepMap.model_training?.result || {})),
    buildSummarySection("集成结果", summarizeModelIntegration(stepMap.model_integration?.result || {})),
    buildSummarySection("评估结论", summarizeEvaluator(stepMap.evaluator?.result || {})),
  ].filter((section) => section.items.length);

  sections.forEach((section) => grid.appendChild(renderSummarySection(section)));
  return grid;
}

function summarizeModelSelection(result) {
  const models = result.models || [];
  return [
    { label: "候选来源", value: result.selection_source || "—" },
    { label: "特征策略", value: (result.feature_methods || []).join(", ") || "—" },
    { label: "候选模型", value: models.map((item) => item.name).join(", ") || "—" },
    { label: "参数摘要", value: models.map((item) => `${item.name}: ${JSON.stringify(item.params || {})}`).join("\n") || "—" },
    { label: "轮次编号", value: result.iteration_index },
  ].filter((item) => item.value !== undefined);
}

function summarizeModelTraining(result) {
  const ranking = result.ranking || {};
  const bestModel = ranking.best_model || result.results?.[0] || {};
  const bestMetrics = bestModel.metrics || {};
  return [
    { label: "最佳模型", value: bestModel.name || "—" },
    { label: "最佳 MAE", value: bestMetrics.mae },
    { label: "最佳 RMSE", value: bestMetrics.rmse },
    { label: "训练后端", value: bestModel.backend || "—" },
    { label: "训练模式", value: bestModel.training_mode || "—" },
    { label: "最佳模型参数", value: bestModel.params ? JSON.stringify(bestModel.params, null, 2) : "—" },
    { label: "目标列", value: result.target_column || "—" },
    { label: "建模特征数", value: result.modeling_feature_count },
  ].filter((item) => item.value !== undefined);
}

function summarizeModelIntegration(result) {
  return [
    { label: "集成策略", value: result.strategy || "—" },
    { label: "成员模型", value: (result.member_models || []).join(", ") || "—" },
    { label: "成员数量", value: result.member_count },
    { label: "预测点数", value: result.predictions?.length },
  ].filter((item) => item.value !== undefined);
}

function summarizeEvaluator(result) {
  return [
    { label: "最佳得分", value: result.best_score },
    { label: "轮次最优结果", value: result.iteration_best_result?.name || "—" },
    { label: "轮次最优策略", value: result.iteration_best_result?.strategy || "—" },
    { label: "是否提升", value: result.improved == null ? undefined : formatBooleanDisplay(result.improved) },
    { label: "停滞轮数", value: result.stagnant_rounds },
    { label: "停止原因", value: result.stop_reason || "—" },
  ].filter((item) => item.value !== undefined);
}

function renderCrossIterationEnsemblePanel(result) {
  const panel = document.createElement("section");
  panel.className = "detail-section fade-in";
  panel.innerHTML = `
    <div class="panel-head">
      <h2>跨迭代最终集成</h2>
      <span class="muted-text">对每一轮最优模型或最优集成再次做平均集成</span>
    </div>
  `;

  const grid = document.createElement("div");
  grid.className = "report-card-grid";
  grid.appendChild(
    renderSummarySection(
      buildSummarySection("最终集成结果", [
        { label: "集成策略", value: result.strategy || "—" },
        { label: "参与轮数", value: result.iteration_count },
        { label: "成员轮次", value: (result.member_iterations || []).join(", ") || "—" },
        { label: "成员策略", value: (result.member_strategies || []).join(", ") || "—" },
        { label: "成员名称", value: (result.member_names || []).join("\n") || "—" },
        { label: "最终 MAE", value: result.metrics?.mae },
        { label: "最终 RMSE", value: result.metrics?.rmse },
        { label: "最终 MAPE", value: result.metrics?.mape },
      ]),
    ),
  );
  panel.appendChild(grid);
  return panel;
}

function renderStageColumns(steps) {
  const groups = [
    { key: "analysis", title: "数据分析", description: "均值、方差、平稳性和基础统计", steps: [] },
    { key: "processing", title: "数据处理", description: "读取、特征工程、切分与预处理", steps: [] },
    { key: "modeling", title: "模型结果", description: "选模、训练、集成与评估", steps: [] },
  ];

  steps.forEach((step) => {
    groups.find((group) => group.key === getStageGroup(step.step_name))?.steps.push(step);
  });

  const columns = document.createElement("div");
  columns.className = "stage-columns";
  groups.forEach((group) => {
    const column = document.createElement("section");
    column.className = "stage-column";
    column.innerHTML = `
      <div class="stage-column-head">
        <div>
          <h4>${escapeHtml(group.title)}</h4>
          <p>${escapeHtml(group.description)}</p>
        </div>
      </div>
    `;

    if (group.steps.length) {
      group.steps.forEach((step) => {
        column.appendChild(renderStepCard(step));
      });
    } else {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "该阶段暂无结果。";
      column.appendChild(empty);
    }

    columns.appendChild(column);
  });

  return columns;
}

// Generic step cards are still kept in the file because they are useful for future expansion
// and for rendering raw step payloads without designing a custom card for every new subagent.
function renderStepCard(step, options = {}) {
  const { showFiles = true, hideMarkdown = false } = options;
  const node = elements.stepTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".step-title").textContent = step.step_name;
  node.querySelector(".step-file-count").textContent = showFiles ? `${step.files?.length || 0} files` : "summary";

  const body = node.querySelector(".step-body");

  if (step.result) {
    body.appendChild(renderResultSummary(step.result));
  }

  if (step.markdown && !hideMarkdown) {
    const note = document.createElement("div");
    note.className = "inline-note";
    note.textContent = "Markdown 报告已生成，前端不直接展开全文。";
    body.appendChild(note);
  }

  if (showFiles && step.files?.length) {
    const fileList = document.createElement("div");
    fileList.className = "file-list";
    step.files.forEach((file) => {
      const row = document.createElement("div");
      row.className = "muted-text";
      row.textContent = `${file.name} (${file.relative_path})`;
      fileList.appendChild(row);
    });
    body.appendChild(fileList);
  }

  if (!body.children.length) {
    body.innerHTML = '<p class="muted-text">该步骤暂无可展示结果。</p>';
  }

  return node;
}

// Result payloads can be large and deeply nested.
// The frontend sanitizes them before display so the detail panel remains readable.
function renderResultSummary(result) {
  const container = document.createElement("div");
  container.className = "result-list";
  const rows = flattenResultEntries(buildDisplayResult(result)).filter(([label]) => !isHiddenResultLabel(label));

  if (!rows.length) {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(buildDisplayResult(result), null, 2);
    container.appendChild(pre);
    return container;
  }

  rows.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "result-item";

    const title = document.createElement("strong");
    title.textContent = label;
    item.appendChild(title);

    const content = document.createElement("div");
    if (typeof value === "string" && value.includes("\n")) {
      const pre = document.createElement("pre");
      pre.textContent = value;
      content.appendChild(pre);
    } else {
      content.textContent = formatDisplayValue(value);
    }

    item.appendChild(content);
    container.appendChild(item);
  });

  return container;
}

function buildDisplayResult(result) {
  return sanitizeDisplayValue(result);
}

// Large numeric arrays, especially predictions, are collapsed into compact summaries.
// This prevents the browser from rendering thousands of values into the page.
function sanitizeDisplayValue(value, key = "") {
  if (Array.isArray(value)) {
    if (isPredictionKey(key)) {
      return summarizePredictionArray(value);
    }
    if (value.length > 8 && value.every((item) => typeof item !== "object")) {
      return {
        count: value.length,
        preview: value.slice(0, 5),
        note: "列表已折叠展示",
      };
    }
    return value.map((item) => sanitizeDisplayValue(item));
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([childKey, childValue]) => {
        return [childKey, sanitizeDisplayValue(childValue, childKey)];
      }),
    );
  }

  return value;
}

function isPredictionKey(key) {
  return ["prediction", "predictions"].includes(String(key).toLowerCase());
}

function summarizePredictionArray(values) {
  return {
    total_points: values.length,
    min: Math.min(...values),
    max: Math.max(...values),
    first_value: values[0],
    last_value: values[values.length - 1],
    note: "Prediction values are hidden in frontend to keep the layout compact.",
  };
}

// Flatten nested result objects into "Section / Field" rows for the generic renderer.
// Compact leaf objects are preserved; only deeper structures are recursively expanded.
function flattenResultEntries(value, prefix = "") {
  if (value == null) return [];

  if (typeof value !== "object" || Array.isArray(value)) {
    return prefix ? [[prefix, value]] : [];
  }

  return Object.entries(value).flatMap(([key, childValue]) => {
    const label = prefix ? `${prefix} / ${humanizeKey(key)}` : humanizeKey(key);
    if (childValue && typeof childValue === "object" && !Array.isArray(childValue) && !isCompactObject(childValue)) {
      return flattenResultEntries(childValue, label);
    }
    return [[label, childValue]];
  });
}

function isHiddenResultLabel(label) {
  const normalized = String(label).toLowerCase();
  return normalized.includes("markdown") || normalized.includes("report path");
}

function isCompactObject(value) {
  return Object.values(value).every((item) => item == null || typeof item !== "object");
}

// The helpers below keep all display formatting decisions in one place:
// numeric precision, status labels, target-column text, and humanized field names.
function formatDisplayValue(value) {
  if (value == null) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(6);
  if (Array.isArray(value)) return JSON.stringify(value);
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function formatMetricValue(value) {
  if (value == null || value === "") return "—";
  if (typeof value === "number") {
    return Math.abs(value) >= 1000 || Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function formatBooleanDisplay(value) {
  if (value == null) return "—";
  return value ? "是" : "否";
}

function formatFileType(value) {
  const mapping = {
    csv: "CSV",
    pkl: "Pickle",
    npy: "NumPy",
    parquet: "Parquet",
    directory_mixed: "Mixed Directory",
  };
  return mapping[String(value || "").toLowerCase()] || value || "—";
}

function formatNormalizationMode(value) {
  const mapping = {
    auto: "自动",
    skip: "跳过",
    off: "关闭",
    zscore: "Z-Score",
    minmax: "Min-Max",
  };
  return mapping[String(value || "").toLowerCase()] || value || "—";
}

function statusClassName(status) {
  const normalized = String(status || "").toLowerCase();
  return normalized || "idle";
}

function formatStatusLabel(status) {
  const mapping = {
    idle: "空闲",
    queued: "排队中",
    running: "运行中",
    completed: "已完成",
    failed: "失败",
  };
  return mapping[String(status || "").toLowerCase()] || String(status || "未知");
}

function formatTargetColumns(profile) {
  if (!profile || typeof profile !== "object") return "";
  const targets = Array.isArray(profile.target_columns) ? profile.target_columns.filter(Boolean) : [];
  if (targets.length) return targets.join(", ");
  return profile.target_column || "";
}

function humanizeKey(key) {
  return String(key)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function getStageGroup(stepName) {
  const name = String(stepName || "").toLowerCase();
  if (isModelStep(name)) {
    return "modeling";
  }
  if (["data_reading", "feature_engineering", "split_strategy", "datanorm", "preprocess"].includes(name)) {
    return "processing";
  }
  if (["data_formatter"].includes(name)) {
    return "processing";
  }
  return "analysis";
}

function isModelStep(stepName) {
  return ["model_selection", "model_training", "model_integration", "evaluator"].includes(
    String(stepName || "").toLowerCase(),
  );
}

function uniqueByStepName(steps) {
  const seen = new Set();
  return steps.filter((step) => {
    if (seen.has(step.step_name)) return false;
    seen.add(step.step_name);
    return true;
  });
}

function filterTasks(tasks, query) {
  if (!query) return tasks;
  return tasks.filter((task) => {
    const haystack = [
      task.task_id,
      task.dataset_path,
      task.target_column,
      ...(Array.isArray(task.target_columns) ? task.target_columns : []),
      task.is_directory ? "directory" : "file",
      task.status,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
}

function groupTasksByDate(tasks) {
  const groups = new Map();
  tasks.forEach((task) => {
    const label = formatTaskDateGroup(task.updated_at || task.task_id || "");
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(task);
  });
  return [...groups.entries()];
}

function formatTaskDateGroup(value) {
  const date = parseTaskDate(value);
  if (!date) return "Earlier";
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const target = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const diffDays = Math.round((today - target) / 86400000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function formatTaskTime(value) {
  const date = parseTaskDate(value);
  if (!date) return value || "未记录";
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function parseTaskDate(value) {
  if (!value) return null;
  const normalized = String(value).replace(" ", "T");
  const parsed = new Date(normalized);
  if (!Number.isNaN(parsed.getTime())) return parsed;

  const match = String(value).match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  if (!match) return null;
  return new Date(
    Number(match[1]),
    Number(match[2]) - 1,
    Number(match[3]),
    Number(match[4]),
    Number(match[5]),
    Number(match[6]),
  );
}

function buildTaskTitle(task) {
  const dataset = task.dataset_path ? task.dataset_path.split("/").pop() : "";
  return dataset || task.target_column || "Untitled task";
}

function buildTaskDetailSignature(detail) {
  return JSON.stringify({
    task_id: detail.task_id,
    status: detail.status,
    iteration_count: detail.iteration_count,
    final_summary: detail.final_summary,
    iteration_history: detail.iteration_history,
    cross_iteration_ensemble: detail.cross_iteration_ensemble,
    iterations: (detail.iterations || []).map((iteration) => ({
      iteration_id: iteration.iteration_id,
      summary_markdown: iteration.summary_markdown,
      steps: (iteration.steps || []).map((step) => ({
        step_name: step.step_name,
        result: step.result,
        markdown: step.markdown,
        files: step.files,
      })),
    })),
  });
}

function kvRow(label, value) {
  return `
    <div class="kv-row">
      <strong>${escapeHtml(label)}</strong>
      <span>${escapeHtml(String(value ?? ""))}</span>
    </div>
  `;
}

// All API requests flow through this helper so error parsing is consistent.
// If the backend returns JSON with `message`, the UI shows that message directly.
async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = {};

  if (text) {
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new Error(`接口返回了不可解析内容: ${text.slice(0, 120)}`);
    }
  }

  if (!response.ok) {
    throw new Error(data.message || response.statusText);
  }

  return data;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
