# SOP Video Analysis UI Implementation

## 1. 目标

结果页需要让用户快速回答三个问题：

1. 整个操作是否合规？
2. 哪些步骤通过、失败或无法确认？
3. 模型依据视频中的哪一段作出判断？

页面采用“结果总览 + 视频证据 + 步骤时间轴 + 步骤明细”结构。用户点击任一步骤后，视频跳转到对应时间，并显示该步骤的模型证据。

本 UI 面向单机 Demo。优先保证核心分析流程清楚、可操作和可演示，不实现多用户、权限管理、跨设备同步或生产级后台管理。

V1 仅提供英文界面，模型证据也要求返回英文。V1.1 增加简体中文；语言集合固定为 `en` 和 `zh-CN`，不支持繁体中文或其他语言。V1 不引入国际化框架或语言切换控件。

## 2. 页面结构

```text
┌─────────────────────────────────────────────────────────────┐
│ WHO Hand-rub Analysis                  △ Needs review       │
│ handrub_demo_03.mp4 · 30.2 seconds                          │
├─────────────────────────────────────────────────────────────┤
│ Steps passed        Procedure duration       Review needed  │
│ 5 / 7               24.7 s ✓                 2 steps        │
├──────────────────────────────────┬──────────────────────────┤
│                                  │ Step results             │
│          Video player            │ 1 Apply handrub    Pass  │
│                                  │ 2 Palm to palm     Pass  │
│                                  │ 3 Palm over dorsum Pass  │
│                                  │ 4 Fingers interlaced      │
│                                  │ 5 Backs of fingers Pass  │
├──────────────────────────────────┤ 6 Thumb rubbing  Review │
│ Colored step timeline            │ 7 Fingertips       Fail  │
├──────────────────────────────────┤                          │
│ Selected-step evidence           │                          │
└──────────────────────────────────┴──────────────────────────┘
```

移动端改为单栏：总览、视频、时间轴、证据、步骤列表依次排列。

## 3. 数据来源

上传视频时创建异步分析任务：

```http
POST /api/analyses
Content-Type: multipart/form-data
```

成功接收后返回：

```json
{
  "analysis_id": "01J...",
  "status": "queued",
  "expires_at": null
}
```

如果已有任务处于 `queued` 或 `analyzing`，创建接口返回 HTTP `409 Conflict`：

```json
{
  "error": "analysis_in_progress",
  "message": "Another analysis is already in progress."
}
```

页面随后查询任务：

```http
GET /api/analyses/{analysis_id}
```

任务状态为 `queued`、`analyzing`、`succeeded` 或 `failed`。任务进入 `succeeded` 或 `failed` 后，响应包含 30 分钟后的 `expires_at`；`succeeded` 时还包含以下 `result`：

```json
{
  "analysis_id": "01J...",
  "status": "succeeded",
  "expires_at": "2026-08-15T12:30:00Z",
  "result": {
  "sop_id": "who_handrub",
  "standard_version": "WHO-2009",
  "definition_version": 1,
  "overall_status": "uncertain",
  "source_video_duration_sec": 30.2,
  "procedure_start_sec": 2.1,
  "procedure_end_sec": 26.8,
  "procedure_duration_sec": 24.7,
  "duration_compliant": true,
  "summary": "Most required steps are visible.",
  "steps": [
    {
      "step_id": "thumbs",
      "step_order": 6,
      "step_label": "Rotational thumb rubbing",
      "status": "uncertain",
      "confidence": 0.48,
      "start_sec": 19.0,
      "end_sec": 22.0,
      "observation": "The right thumb is visible, but the left thumb is partially occluded."
    }
  ],
  "warnings": ["Dryness cannot be verified visually."]
  }
}
```

### 字段映射

| API 字段 | UI 位置 | 显示方式 |
|---|---|---|
| `sop_id`、`standard_version`、`definition_version` | 页面标题附近 | 标识本次审核使用的 SOP Identity |
| `overall_status` | 页面右上角 | Overall status badge |
| `steps` | 总览和右侧步骤列表 | UI 计算通过数 / 必需步骤总数，并显示每步状态、时间和名称 |
| `procedure_duration_sec` | 总览区域 | 秒数及合规标记 |
| `duration_compliant` | 总览区域 | `✓` 或警告提示 |
| `step.step_order` | 步骤列表和证据区域 | SOP 标准顺序和步骤编号 |
| `step.step_label` | 步骤列表和证据区域 | 本次 SOP 定义中的显示名称 |
| `step.start_sec/end_sec` | 时间轴和视频 | 时间段位置、视频跳转点 |
| `step.observation` | 证据区域 | 模型可见证据说明 |
| `step.confidence` | 证据区域 | 百分比，作为辅助信息 |
| `warnings` | 总览下方 | 可操作警告信息 |

步骤名称和顺序直接使用分析响应中的 `step_label` 与 `step_order`。这些值由后端从本次审核使用的版本化 SOP 定义复制，UI 不维护名称映射，也不直接显示 `step_id`。

证据视频不由任务 API 返回。用户选择文件后，浏览器通过 `URL.createObjectURL(file)` 创建仅限当前页面会话的播放地址；选择其他文件或离开页面时调用 `URL.revokeObjectURL()`。后端推理副本在任务成功或失败后删除。

## 4. 状态展示

状态不能只依赖颜色，必须同时使用文字和符号。

| 状态 | 用户文案 | 符号 | 含义 |
|---|---|---|---|
| `passed` | Passed | `✓` | 有明确证据完成 |
| `failed` | Failed | `×` | 有明确证据失败或不完整 |
| `uncertain` | Needs review | `△` | 遮挡或证据不足，需要人工复核 |

建议配色：

- Passed：绿色
- Failed：红色
- Needs review：琥珀色
- 未选择、边框和辅助文字：中性灰色

模型置信度不用于改变最终状态，也不要单独显示成巨大的评分。它只在步骤证据中作为辅助信息。

## 5. 结果总览

顶部显示三个核心指标：

```text
Steps passed          Procedure duration       Review needed
5 / 7                 24.7 s ✓                 2 steps
```

整体状态计算结果显示在页面右上角：

```text
✓ Passed
× Failed
△ Needs review
```

当存在明确失败步骤时，整体显示 `Failed`；只有不确定步骤而没有明确失败时，显示 `Needs review`。

对于 SOP 要求双侧完成的步骤，只完成一侧属于明确失败；另一侧被遮挡或无法确认时属于 `Needs review`。证据说明必须指出具体是哪一侧及原因。

WHO hand-rub V1 只显示 7 个 Auditable Steps。干燥不作为步骤或评分项；页面通过警告显示 `Dryness cannot be verified visually.`。

## 6. 视频证据区域

使用浏览器原生 `<video controls>`：

```html
<video id="analysis-video" controls preload="metadata">
  <source id="analysis-video-source" type="video/mp4">
</video>
```

用户点击某一步骤时：

1. 设置视频 `currentTime = step.start_sec`。
2. 更新选中步骤样式。
3. 更新时间轴播放头位置。
4. 更新证据说明。
5. 可选：自动播放到 `end_sec` 后暂停。

```javascript
function selectStep(step) {
  selectedStepId = step.step_id;

  if (step.start_sec != null) {
    video.currentTime = step.start_sec;
  }

  renderSelectedStep(step);
  renderTimelinePlayhead(step.start_sec);
}
```

不建议点击步骤后强制自动播放，避免突然产生声音。默认只跳转，用户自行点击播放。

V1 不支持刷新后恢复视频和结果。页面刷新或关闭后，浏览器本地播放地址失效。

## 7. 步骤时间轴

时间轴宽度代表整个源视频长度，每个步骤显示为对应时间段：

```javascript
function segmentPosition(step, videoDuration) {
  if (step.start_sec == null || step.end_sec == null || videoDuration <= 0) {
    return null;
  }

  return {
    leftPercent: (step.start_sec / videoDuration) * 100,
    widthPercent: Math.max(
      ((step.end_sec - step.start_sec) / videoDuration) * 100,
      1
    )
  };
}
```

每个时间段颜色对应步骤状态。用户点击时间段，也应执行 `selectStep(step)`。

时间轴至少显示：

- 视频开始和结束时间
- 检测到的流程开始和结束时间
- 每个步骤的时间段
- 当前视频播放位置
- Passed、Failed、Needs review 图例

如果步骤没有时间戳，不要伪造位置；只在步骤列表中显示 `Not located`。

## 8. 步骤列表

每个步骤行为一个可点击按钮：

```html
<button class="step-row" type="button" aria-pressed="false">
  <span class="step-number">6</span>
  <span class="step-name">Rotational thumb rubbing</span>
  <span class="step-time">19.0–22.0s</span>
  <span class="step-status">△ Needs review</span>
</button>
```

排序始终使用 `step_order`，而不是模型返回顺序或 UI 硬编码顺序。

优先把失败和不确定步骤标得清楚，但不要改变 SOP 顺序，否则用户难以检查动作流程。

## 9. 证据说明

选中步骤后显示：

```text
Step 6 · Rotational thumb rubbing

Right thumb is clearly rubbed. The left thumb is partially
occluded, so completion cannot be confirmed. Confidence 48%.
```

证据区域必须包含：

- 步骤编号与名称
- 状态
- 时间范围
- 模型观察结果
- 置信度
- 失败或不确定的具体原因

避免只显示“Step failed”，因为用户无法验证模型判断。

## 10. 页面状态

### 上传前

- 显示文件选择器。
- 显示支持的视频格式和时长限制。
- 明确提示抽取的视频帧将发送到所配置的模型服务，原始视频不会作为整体发送。
- 不显示空结果面板。

### 分析中

- 禁用重复提交。
- 根据任务状态显示明确文案：`Waiting to analyze…` 或 `Analyzing video…`。
- 不伪造百分比进度，因为本地推理没有可靠的细粒度进度值。
- UI 定时查询当前 `analysis_id`，收到 `succeeded` 或 `failed` 后停止查询。
- 收到 `409 analysis_in_progress` 时保留已选视频，并提示等待当前任务结束；V1 不提供取消、排队或自动重试操作。

### 分析成功

- 默认选择第一个 `failed` 步骤。
- 如果没有失败，选择第一个 `uncertain` 步骤。
- 如果全部通过，选择第一个步骤。

```javascript
function getDefaultStep(steps) {
  return steps.find(step => step.status === "failed")
    ?? steps.find(step => step.status === "uncertain")
    ?? steps[0]
    ?? null;
}
```

### 分析失败

- 保留已选择的视频，方便重试。
- 显示后端返回的安全错误文案。
- 不在 UI 中显示 Python traceback、模型路径或系统环境变量。
- 使用简短英文错误信息并提供可执行的下一步建议，不暴露服务器地址、API Key 或模型内部信息。
- 读取任务返回的安全错误文案，并停止状态查询。
- 查询返回 HTTP `410 Gone`，或服务器重启导致内存任务不存在时，显示 `Analysis expired; please run it again.`。

## 11. 响应式布局

### 桌面端（大于 720px）

```css
.result-main {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.85fr);
  gap: 18px;
}
```

### 移动端（720px 及以下）

```css
@media (max-width: 720px) {
  .result-main,
  .result-summary {
    grid-template-columns: 1fr;
  }
}
```

移动端顺序：

1. 整体状态
2. 三个总览指标
3. 视频
4. 时间轴
5. 选中步骤证据
6. 完整步骤列表

## 12. 无障碍要求

- 步骤使用原生 `<button>`，支持键盘操作。
- 选中状态使用 `aria-pressed`。
- 视频使用原生 controls。
- 状态同时显示文字、符号和颜色。
- 证据区域使用 `aria-live="polite"`，步骤切换时辅助技术可以读出更新。
- 文本与背景保持足够对比度。
- 不移除浏览器默认焦点样式。

## 13. 推荐组件划分

即使使用原生 HTML/JavaScript，也建议按以下职责拆分渲染函数：

```text
AnalysisPage
├── UploadForm
├── AnalysisSummary
├── VideoEvidence
├── StepTimeline
├── SelectedStepEvidence
└── StepResultList
```

对应函数：

```javascript
renderUploadForm();
renderSummary(result);
renderVideoEvidence(videoUrl, selectedStep);
renderTimeline(result.steps, result.source_video_duration_sec);
renderSelectedStep(selectedStep);
renderStepList(result.steps, selectedStep);
```

页面只保留两个主要状态：

```javascript
let analysisResult = null;
let selectedStepId = null;
```

视频播放时间由 `<video>` 元素自身维护，不要复制到多个状态变量中。

## 14. 第一阶段范围

第一阶段实现：

- 单视频上传
- 分析中和错误状态
- 整体状态及三个总览指标
- 视频播放
- 步骤列表
- 点击步骤跳转视频
- 步骤时间轴
- 模型证据说明
- 桌面与移动端布局

第一阶段暂不实现：

- 人工修改模型结果
- 多视频比较
- 历史记录
- PDF 报告
- 人工标注工具
- 实时摄像头检测
- 帧级手部边界框

### V1.1 本地持久化边界

V1.1 同时加入简体中文界面与简体中文模型证据，英文仍为默认语言；语言选择只提供 English (`en`) 和简体中文 (`zh-CN`)。

- 始终保存分析 JSON、SOP Identity、模型与提示词版本、原文件名、文件哈希和创建时间。
- 上传时提供 `Keep source video in local history` 选项，默认关闭。
- 未保存原始视频的历史结果可以查看结论，但证据区域明确显示视频不可用。
- 不单独保存抽帧；需要证据帧时从已保存的原始视频重新生成。
- 历史界面提供删除单条 Audit Record 和清空本地历史操作。
- Audit Record 和用户选择保存的原始视频不自动过期，保留至用户主动删除。
- 删除单条记录时同时删除其关联视频；清空本地历史前显示确认对话框。
- 历史界面显示 Audit Record、视频及总计占用的本地存储空间。
- Audit Record 和任务状态保存在 SQLite；用户选择保留的视频作为普通文件保存在系统应用数据目录，不写入项目目录或 SQLite BLOB。
- 每条 Audit Record 最多对应一个以 `analysis_id` 命名的视频文件，不实现视频去重或引用计数。
- UI 删除包含视频的 Audit Record 时，提示对应视频文件也将被删除。

## 15. 验收标准

- API 返回结果后，页面能展示全部 SOP 步骤及其 `step_label`。
- 创建任务后，UI 能展示 `queued` 和 `analyzing` 状态且不显示虚假百分比。
- 当前任务结束前重复提交会被 UI 阻止；直接调用 API 时返回 `409 analysis_in_progress`。
- 任务成功或失败后停止状态查询，后端删除推理临时文件。
- 成功或失败结果在内存中保留 30 分钟；过期响应不会暴露之前的结果。
- 步骤始终按照响应中的 `step_order` 排列。
- 点击带时间戳的步骤，视频跳转误差不超过 0.2 秒。
- Failed、Passed、Needs review 不仅通过颜色区分。
- 没有时间戳的步骤显示 `Not located`，不会出现在时间轴上。
- 默认优先选中失败步骤，其次是不确定步骤。
- 20–30 秒范围内显示时长合规；范围外显示警告。
- 320px 宽度下没有横向溢出或文字重叠。
- 键盘可以遍历并选择所有步骤。
- 模型或网络错误不会清除用户已经选择的视频。
- 页面更换视频或卸载时会释放旧的浏览器 `Object URL`。
