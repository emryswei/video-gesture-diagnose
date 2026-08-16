# 视频分析优化方法

## 1. 目标

当前 Demo 使用 Qwen3-VL 对少量均匀抽取的视频帧进行一次性判断。该方法实现简单，但容易漏掉短动作，也可能把多个相似的后续动作集中判断为同一个步骤。

优化目标不是让 VLM 单独承担全部判断，而是把视频切分、动作检测、规则评分和解释分开处理，同时保持项目精简、可运行、适合演示。

## 2. 市场方案对比

| 方案 | 擅长 | 对 WHO 动作合规的适用性 |
| --- | --- | --- |
| Google Video Intelligence、AWS Rekognition | 场景、物体、活动、镜头和时间戳 | 只能识别通用内容，不能直接判断精细手部步骤 |
| Qwen3-VL、商业 Video VLM | 视频问答、摘要、粗粒度动作定位 | 适合 Demo 和解释，不适合作为唯一裁判 |
| 手部关键点 + 动作分类模型 | 姿态、运动方向、持续时间、重复次数 | 适合精细动作，但需要标注数据 |
| 混合方案 | 时序模型负责检测，规则负责评分，VLM 负责复核和解释 | 最适合本项目 |

通用视频分析服务适合发现人物、活动、镜头和大致时间段。WHO hand-rub 合规检测属于细粒度 SOP 审核，需要区分手掌、手背、交叉手指、拇指旋转、指尖旋转以及左右侧完成情况，因此必须加入领域规则和连续动作分析。

## 3. 成熟的 SOP 视频分析流程

```text
输入视频
  -> 定位有效操作区间并去除片头、片尾
  -> 检测双手并裁剪手部区域
  -> 连续提取手部关键点、图像和运动特征
  -> 使用重叠时间窗口切分候选动作
  -> 判断每个片段属于哪个 SOP 步骤
  -> 合并连续预测并生成步骤时间段
  -> 统计持续时间、双侧完成情况和重复次数
  -> 由 SOP 规则引擎计算 pass / failed / uncertain
  -> 由 VLM 复核低置信度片段并生成解释
  -> 输出带时间戳、截图和置信度的证据报告
```

### 3.1 SOP 定义层

模型不自行决定步骤数量。系统从版本化 SOP 配置读取允许的步骤、显示顺序和评分规则。

WHO hand-rub Demo 保留 7 个可审核步骤：涂抹产品以及 6 种揉搓动作，不把 `hands_dry` 作为评分步骤。

每个步骤可以逐步增加以下字段：

```json
{
  "step_id": "thumb_rotation",
  "order": 6,
  "requires_both_sides": true,
  "min_duration_sec": 3,
  "min_repetitions": 5,
  "observable_cues": {
    "motion_pattern": "rotation",
    "target_region": "thumb"
  },
  "completion_criteria": {
    "min_confidence": 0.8,
    "min_hold_sec": 1.5
  }
}
```

WHO 官方材料规定整个 hand-rub 流程约 20–30 秒，但没有在官方海报中规定每一步必须 3 秒或 5 次。此类阈值如果启用，必须标记为项目内部评分策略，而不能显示为 WHO 官方硬性要求。

### 3.2 视频和手部特征层

- 使用较密集的视频帧进行运动分析，不再只从整个视频均匀抽取 8 帧。
- 在 CPU 上使用 MediaPipe Hand Landmarker 跟踪每只手的 21 个关键点。
- 同时保留手部 RGB 裁剪和运动信息，处理双手重叠、遮挡和关键点交换。
- 无法可靠看到双手时输出 `uncertain`，不把“没有证据”直接判成“没有执行”。

### 3.3 时序动作层

- 使用约 1.5–2 秒的重叠窗口分析连续动作。
- 每个窗口只能输出 7 个 SOP 步骤之一，或 `other`、`uncertain`。
- 使用时序平滑或轻量状态机合并连续结果，避免一个动作被切成多个短片段。
- `order_strict: false` 时按覆盖度评分；只有明确配置为严格顺序时才执行序列比对。

### 3.4 规则评分层

动作检测与合规评分分开：

- 动作检测回答“当前是什么动作”；
- 规则引擎回答“动作是否充分完成”；
- VLM 回答“证据画面是否支持该判断，以及如何解释”。

建议每一步输出：

```json
{
  "step_id": "thumb_rotation",
  "detected": true,
  "start_sec": 31.2,
  "end_sec": 35.8,
  "left_side": true,
  "right_side": false,
  "estimated_repetitions": 3,
  "confidence": 0.84,
  "status": "failed",
  "reason": "The right thumb rotation was not confirmed."
}
```

### 3.5 证据与人工复核层

每一步保存开始和结束时间、代表截图、置信度和判断原因。低置信度、遮挡、缺少一侧证据或动作边界不明确时标记为 `uncertain`，交由人工复核。

视频只能审核动作过程，不能证明酒精实际覆盖效果或微生物消毒结果。若需要验证真实覆盖质量，应结合荧光手消毒液或其他专门检测手段。

## 4. 最适合当前 Demo 的精简 V2

当前项目不需要先重写成大型系统，也不需要立即训练新的深度学习模型。V2 可以直接在现有 FastAPI、SOP 配置、任务接口和 UI 之上渐进实现。

### 阶段 A：先修复时序覆盖

1. 自动定位有效 hand-rub 区间，排除片头和片尾。
2. 使用重叠时间窗口代替全视频 8 帧一次判断。
3. 每个候选片段单独分类，不让模型一次决定全部步骤。
4. 聚合连续窗口，为每个步骤生成开始和结束时间。

这一阶段是进入 V2 的最小必要改动，优先解决漏动作和多个步骤被合并的问题。

### 阶段 B：加入 CPU 手部跟踪

1. 使用 MediaPipe 提取双手关键点和手部区域。
2. 计算手部是否可见、左右侧、运动方向和周期运动。
3. 使用这些特征寻找候选片段，减少 Qwen3-VL 调用数量。

### 阶段 C：Qwen3-VL 定向复核

1. Qwen3-VL 只接收候选片段中的少量连续关键帧。
2. 每次只判断一个片段或一个目标步骤。
3. 对低置信度结果进行复核并生成英文解释。

### 阶段 D：增加质量评分

在动作定位稳定后，再启用单步时长、重复次数、双侧完成和覆盖度评分。顺序默认不严格，除非 SOP 配置明确要求严格顺序。

## 5. V1 到 V2 的实施边界

不需要先完成一个独立的“完整 V2”才能继续。下一次代码修改本身就是 V2 的开始，可以在现有代码库中按阶段 A 到 D 逐步替换分析管线。

以下部分可以继续复用：

- FastAPI 路由和上传流程；
- 当前异步任务管理；
- SOP ID、步骤 ID 和版本化配置；
- `pass / failed / uncertain` 结果语义；
- 当前结果页面的步骤列表和证据区域。

必须修改的是后端视频分析管线：从“全视频少量抽帧后一次调用模型”改为“候选片段检测、逐片段识别和时序聚合”。仅调整提示词不能解决当前漏动作和步骤合并问题。

## 6. 建议验收指标

- 每个配置步骤都返回一条结果，不允许静默缺失。
- 每一步包含独立的开始和结束时间或明确的 `uncertain` 原因。
- 后续相似动作不会无证据地集中到同一个步骤。
- 双侧步骤能区分完成、只完成一侧和因遮挡无法判断。
- 结果页面能定位到支持判断的代表截图或短片段。
- 使用人工标注视频计算每一步的 precision、recall 和 F1，而不只统计整体通过率。

## 7. 参考资料

- [WHO How to Handrub](https://www.who.int/publications/m/item/how-to-handrub)
- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
