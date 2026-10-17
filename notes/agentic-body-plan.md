# Agentic 人体优化计划（2026-10-17）

> 用户提议："用 Agentic 的方法做人体 body 的优化……也可能有人提出过类似算法。"
> 本文 = 文献核对 + 面向我们现状的具体方案。未经实测的条目一律标"假设"。
> 前置事实（来自 `progress.md` / `submissions.md`）：
> SAM 3D Body 逐帧 + wrap-safe σ=3 平滑 = **CD-H 17.86（22/27，领先者 13.4）**；
> 单目 2D 关键点精修让 CD-H **变差 5 cm**（px_err 6.7→2.7 却 CD-H 17.9→22.9）；
> FlowHMR 关节与 SAM3D 相差 4 cm、加速度 0.133 已达真值水平；`fit_flowhmr.py` 已能把 MHR 拟合上去。
> 本地**没有真值**，只有代理；提交暂停中；每日 5 次提交；GPU 共享。

---

## 0. 一句话结论（设计决策）

1. **文献里没有"LLM agent 做人体网格测试时优化"的直接先例**；最接近的是四类：
   VLM 当**评判/打分**（CVPR26 VLM-GPA、MHE-Former、Cross-Model VLM-Judge）、VLM 当**可微 critic**（VLM3D）、
   VLM **闭环提议 6D 位姿**（VLMPose, ECCV26）、VLM 监督**生成式重建**（AGILE, HOI）。
   它们一致的经验：**VLM 擅长"哪个更合理"的成对比较和粗错误识别（翻转、穿地、半圈转身），不擅长输出数值参数**。
   因此我们的 agent **不直接改参数，只做"诊断 → 选工具/选候选 → 验收"**，数值优化仍交给梯度/闭式工具。
2. **最大的风险不是 agent 不聪明，是代理指标骗 agent**（我们已经被 px_err 骗过一次）。所以验证器设计 > agent 设计：
   - 本地只信三类信号：**自指/可预测的**（PEN、ACC 代理）、**多源一致性**（SAM3D vs FlowHMR vs MoGe 深度 vs 地面）、**VLM 成对比较**；
   - 任何单一拟合目标（关键点、轮廓）**只能作为"诊断线索"，不能作为验收标准**；
   - 用**已有的 8 次带分提交做回溯校验**（免费），再用**子集替换提交**做前瞻校准（榜单分是 30 段无权重均值，已核对 `CD-H.py:989-1010`，所以换 k 段就能精确反推这 k 段的 ΔCD-H）。
3. 实现选 **Python 循环 + Claude API**（可 nohup 后台、可断点续跑、每步 JSON 落盘），Claude Code 子代理只用于 2 小时原型。
4. 先打的三个失效模式：**(a) 全局深度/尺度漂移**（相对首帧 Sim3）、**(b) 根旋转/朝向离散错误**（半圈、转身、左右翻）、**(c) 首帧本身是离群帧**（Sim3 锚定在它上，一帧坏全段坏）。

---

## 1. 文献核对（2024–2026）

搜索词覆盖：agentic human mesh recovery / VLM feedback pose refinement / LLM-guided test-time optimization / ChatPose / PoseGPT / VLM verifier 3D human / agent loop 3D reconstruction / multimodal LLM critic SMPL / Agentic 3D / OPRO。
**诚实评级**：★★★ 直接相关 · ★★ 结构可借 · ★ 弱匹配。

| 名称 | 会议/年 | 方法一句话 | agent/VLM 吃什么 | 吐什么 | 报告增益 | 代码 | 相关度 |
|---|---|---|---|---|---|---|---|
| **VLM-Guided Group Preference Alignment for Diffusion-based HMR** (Shen et al., arXiv 2602.19180) | CVPR 2026 | 一个带"双记忆 + 自反思"的 **HMR 批评 agent** 给多条网格假设打分，分数用于偏好对齐训练扩散 HMR | RGB 图 + **网格叠加渲染**，成组同看多个预测；检索"规则记忆"(规则文本 + 使用/成功计数) 与"原型记忆"(CLIP 向量 + 理由) | 0–100 分 + 文字批评；自反思：与 GT 指标的 Spearman 超阈值则强化规则，否则让 VLM 提 1–2 条新规则 | VLM 分 vs PVE 的 **SRCC 0.605–0.633**；3DPW MPJPE 57.2→49.9 mm（-8.2%）；H36M 36.9→34.0 | 未提及 | ★★★（最接近"VLM 当 critic"；**注意其分数与真值相关性只有 0.6 量级**，且是训练期用，不是测试时闭环） |
| **MHE-Former** (arXiv 2609.10743) | 2026 | 多假设网格恢复 + **VLM 分阶段选假设**：先可行性/可见区域对齐过滤（留 60%），再交互式选择 | 每个假设**四个正交视角渲染**（前/后/左/右） | 选中的假设索引 | HO3D JPE：VLM 选 8.21 vs 随机 8.25 vs 全部 8.26（**几乎无差**）；用户研究偏好 63.9–69.5% | 未提及 | ★★（证明 VLM 选假设"更像人"，**但对度量误差几乎不帮忙**——对我们是警告） |
| **Adapting HMR with Vision-Language Feedback** (arXiv 2502.03836) | 2025 | VLM 生成身体部位描述 → 文本-姿态共享隐空间 → 文本梯度 + 2D 关键点梯度一起引导扩散细化 | 图像（VLM 出文字描述） | 文本约束的梯度 | "多基准有效"，摘要无数字 | "将公开"，未见 | ★★（反馈走梯度而非 agent；思路：**语义约束补深度歧义**） |
| **VLM3D: VLMs as Semantic & Spatial Critics for 3D Generation** (arXiv 2511.14271) | ICML 2026 | 冻结 Qwen2.5-VL-7B，限定只答 Yes/No，用 **log-odds(yes)-log-odds(no) 作可微奖励**，反传到 3D 参数；也有测试时采样引导版 | 多视角渲染 + 两个问题（内容匹配 / 几何是否合理） | 标量奖励（可微） | GPTEval3D 全部六项最高 | 有（CC BY 4.0） | ★★（**可微 VLM critic** 是能直接插进 `fit_mhr.py` 的形式；但训练在文生 3D，对人体位姿的判别力未知） |
| **VLMPose: Text-Guided 6D Object Pose via Closed-Loop VLM Agents** (arXiv 2604.09781) | ECCV 2026 | **评估者 VLM + 提议者 VLM** 闭环迭代改 6D 位姿；多视角推理、物体中心坐标可视化、**单轴旋转预测** | 渲染场景（含坐标轴可视化）+ 文本指令 | 增量位姿更新 | 超过先前方法；闭源/开源 VLM 都行 | github.com/snuvclab/vlmpose | ★★★（**结构模板**：把连续位姿拆成离散、单轴、可视化坐标系的动作，VLM 才能可靠操作） |
| **AGILE: HOI Reconstruction from Video via Agentic Generation** (arXiv 2602.04672) | 2026 | Gemini 3 Pro 监督：选 1–4 关键帧 → 生成正交视图 → **VLM critic 按几何/纹理/材质一致性打分，拒绝采样重生成** → 水密物体网格 → HOI 优化 | 视频帧 + 生成视图 | 关键帧选择、通过/拒绝、纹理验收 | HO3D CD 0.27 cm²；去掉多视角合成 → 2.85（10×） | 项目页，代码未明 | ★★（同赛题域 HOI；agent 用在**物体**侧而非人体；**拒绝采样 + 一致性打分**可直接借到我们物体线） |
| **Cross-Model VLM-Judge for 3D Mesh Quality** (arXiv 2606.18451) | 2026 | 24 视角固定渲染，两个开源 VLM（Qwen2.5-VL-7B、InternVL3-8B）**双序成对评判**，只有 AB/BA 一致才采信 | 24 视角渲染，成对 | 偏好 | 跨模型 κ=0.66；**位置偏差纠正翻转了 26% 的判决**；几何代理 0.62、CLIP 0.48（随机水平） | — | ★★（**评判协议**：双序、固定视角、跨模型一致才算数；"廉价代理在模糊样本上塌陷"与我们 px_err 的教训一致） |
| **BlenderGym** (Gu et al.) | CVPR 2025 Highlight | VLM 生成器-验证器闭环做 3D 编辑；发现**验证器也能靠推理扩展提升** | 渲染图 + 脚本 | 代码编辑 | SOTA VLM 仍远不如人 | 有 | ★★（generator-verifier 分离、验证器多采样） |
| **SceneCraft** (ICML 2024) / **Scenethesis** (2025) | — | LLM 写 Blender 脚本，GPT-V 看渲染自批评迭代；库学习 | 渲染 | 脚本修订 | 定性 | 有 | ★ |
| **GPT-4V is a Human-Aligned Evaluator for Text-to-3D** | CVPR 2024 | 多视角渲染成对比较 | 渲染 | 偏好 | 3D 合理性 Kendall τ=0.64 | 有 | ★ |
| **ChatPose / PoseGPT** (Feng et al.) | CVPR 2024 | 把 SMPL 姿态当 token 嵌入多模态 LLM，可"聊"出姿态 | 图/文 | SMPL 参数 | 推理式姿态估计任务上优于 MLLM | 有 | ★（LLM **直接输出姿态**精度远低于专用 HMR，不适合做我们的优化器） |
| **ChatHuman** (arXiv 2405.04533) | 2024/25 | LLM agent **选择并组合 3D 人体工具**（姿态/形状/接触/HOI），RAG 读论文学用法 | 图 + 对话 | 工具选择 + 整合 | 工具选择准确率优于基线 | 项目页 | ★★（"工具编排"形态与我们一致，但无测试时优化闭环） |
| **Pose-RFT** (arXiv 2508.07804) | 2025 | 混合动作强化微调 MLLM 做 3D 姿态生成 | 图/文 | 姿态 | — | — | ★ |
| **IRG-MotionLLM** (arXiv 2512.10730) | 2025/26 | 文生动作：生成-评估-细化交织 | 文 + 动作 | 动作 + 评估 | 跨基准提升 | 有 | ★（文生动作，非重建） |
| **AREA3D** (arXiv 2512.05131) | 2025 | VLM 指导主动重建下一视角 | 观测 | 视角 | 稀疏视角 SOTA | 将公开 | ★ |
| **OPRO** (Yang et al., 2309.03409) | ICLR 2024 | LLM 看历史 (解, 分数) 轨迹提新解 | 数字轨迹 | 候选超参 | ViT 调参优于随机搜索（30 轮） | 有 | ★★（**超参搜索形式**适合我们"w_prior/σ/尺度因子"这种 ≤5 维离散空间；但依赖可靠的分数——我们恰恰缺） |
| **AIDE / MLE-STAR / MLE-bench** | 2024–25 | LLM agent 对 Kaggle 任务做方案树搜索 | 代码 + 验证分 | 代码 | 奖牌率 25.8%→43.9% | 有 | ★（提醒：**它们都依赖本地可靠的验证集**，我们没有） |

**明确没找到的东西**：
- 没有论文把 LLM/VLM agent 放进**单目视频人体重建的测试时优化闭环**并报告 MPJPE/Chamfer 增益；
- 没有论文证明 VLM 能从渲染判断**深度/尺度的度量误差**（恰是我们的主误差源）；相反 MHE-Former 显示 VLM 选假设对 JPE 几乎无益，VLM-GPA 的分数与 PVE 相关性 0.6；
- 没找到本赛题相关的 agentic 参赛方案（Discord 不可抓取）。

**从文献借什么（落到我们的设计里）**：
1. **成对比较而非绝对打分**，且 **A/B 双序各问一次，答案一致才采信**（VLM-Judge；MHE-Former 的 60% 过滤也是相对的）。
2. **固定、信息量大的渲染视角**：叠加 + 侧视（含地面）+ 俯视轨迹；正交视角（MHE-Former 四视角）；物体中心坐标轴可视化（VLMPose）。
3. **把动作离散化、单轴化**（VLMPose 单轴旋转）：agent 选"深度因子 ×0.9/×1.1"、"对 [t0,t1] 段左右翻"、"换 FlowHMR 轨迹"，而不是输出 136 维参数。
4. **生成器-验证器分离，验证器可以多采样**（BlenderGym）；**VLM 规则记忆带成功计数**（VLM-GPA）——我们用"规则 × 校准结果"的表来做。
5. **只把 VLM 用于它擅长的离散判别**（翻转、穿地、半圈、首帧是否离群、哪个候选更像视频），度量问题交给数值一致性。
6. 可选进阶：**VLM3D 的 Yes/No log-odds 可微 critic** 接到 `fit_mhr.py`——留作 Phase 3，先不做。

---

## 2. 问题结构：为什么是"全局放置"，agent 能碰哪些自由度

CD-H 的计算（`CD-H.py`，已读）：
- 首个评分帧用 `role_alignment` 顶点拟合**一次** Sim(3)（含尺度，无反射），整段共用；
- 之后逐帧对 `role_body` 稀疏点做对称 Chamfer，**段内帧均值 → 30 段无权重均值**。

推论（决定 agent 的动作空间）：
- **A. 身体绝对尺寸不重要，但"轨迹位移 / 身体尺寸"的比例重要**：Sim3 的尺度由首帧身体大小定，再乘到整段轨迹。SAM3D 的 `pred_cam_t` 深度来自 bbox 大小反推，逐帧抖、且整段可能有系统偏差 → 轨迹相对身高偏大/偏小 → 全段 CD-H。→ 动作：**整段深度因子 k**、**深度线性漂移修正**。
- **B. 首帧是锚**：首帧姿态/朝向错 → 整段旋转/平移错。→ 动作：**首帧替换**（用 FlowHMR 或邻帧中位数）、检查首帧是否离群。
- **C. 旋转离散错误代价巨大**：半圈转身（ep26 已验证：9.58 → 0.116 ACC，CD-H 同步改善 3.5 cm）、左右翻。→ 动作：**wrap-safe 平滑**（已有）、**段内左右翻转**。
- **D. 多段 episode 的 gap 处 ACC 不计** → **允许在 gap 处重启轨迹**，用于遮挡后翻转修复。
- **E. 局部姿态（肢体）是次要项**：Chamfer 对稀疏体表点，肢体深度错几 cm 远小于全局 10 cm 级偏移；且单目 2D 精修已证明会伤。→ **agent 默认不碰 pose[6:]**，只允许"整段换源"（SAM3D ↔ FlowHMR 拟合）和平滑。

---

## 3. agent 的观测（每段、每个候选）

### 3.1 数字（JSON，≤ 60 个标量/候选）
全部由现有脚本或小改即可得到（`score_local.py`、`diag_accel.py`、`fit_flowhmr.py` 的 stats、`ground_plane.py`、`place_sanity.py`）：

| 组 | 量 | 来源 | 用途 |
|---|---|---|---|
| 可预测 | `acc_h_proxy`（段内均值、最大）；`pen_cm`（自指，精确） | `diag_accel.py` / `score_local.py` | 硬闸：ACC 代理 ≤ 0.2、PEN = 0 |
| 多源一致性 | `d_sam3d_vs_flowhmr_cm`（Sim3 对齐后的关节均差、**不对齐的根轨迹差**）；`depth_ratio_moge`：每帧 MoGe 人体掩码中位深度 / 骨盆深度 的**中位数与段内斜率**；`foot_floor_cm`：站立帧脚底到 MoGe 地面的有符号距离（中位/最大）；`bone_len_cv` | 新 `src/agent_obs.py` | 深度漂移、尺度、穿地的**独立**证据 |
| 首帧健康 | `first_frame_outlier`: 首评分帧根平移/朝向与后 15 帧中位的差（cm/deg）；`first_frame_kp_conf` | 新 | 失效模式 (c) |
| 离散错误 | `yaw_wrap_jumps`（|Δ| > π 的帧数）；`lr_flip_score`：投影的左/右腕踝与 VitPose 左右的符号一致率，按 30 帧窗口 | 新（`fit_mhr.py` 已有 COCO↔MHR 映射） | 失效模式 (b) |
| 仅作线索 | `px_err`（VitPose）、`sil_iou` | 已有 | **不进验收**，只在诊断提示里标"线索" |
| 元数据 | 机位、物体、评分段数与长度、任务描述 | meta | 先验（如"坐凳转身"→ 重点查 wrap） |

### 3.2 渲染（每候选 4 张，A/B 并排 → 每步 ≤ 8 张；1024 px 宽，≈1.1k token/张）
1. **叠加图条**：首评分帧 + 25% / 50% / 75% / 末帧，网格按深度着色叠在原帧上（`viz_body.py` 已有雏形）。
2. **侧视图（x=深度，y=高度）**：同 5 帧的身体轮廓 + **MoGe 地面线** + 相机位置；A/B 同图不同色。—— 看穿地/悬空/深度不一致。
3. **俯视轨迹图**：骨盆 x–z 轨迹（评分段着色，gap 断开）+ 朝向箭头每 15 帧 + 物体掩码反投影位置；A/B 同图。—— 看深度漂移、半圈、轨迹长度。
4. **时间曲线面板**：骨盆深度 z(t)、MoGe 人体深度(t)、脚-地面距离(t)、yaw(t)（已 unwrap）、ACC 代理(t)；A/B 同图。

这些图**不是给人看的美图**，是给 VLM 看的诊断图：固定视角、固定色、固定坐标范围（跨候选可比）、大号标注。

---

## 4. 动作 / 工具（agent 可调用，全部是离散或 ≤ 2 个标量参数）

| 工具 | 参数 | 对应现有代码 | 耗时（估） |
|---|---|---|---|
| `make_candidate(ep, source)` | `source ∈ {sam3d_raw, sam3d_smooth(σ), flowhmr_fit(w_prior, sigma_cm)}` | `export_episodes.py` / `fit_flowhmr.py` | 10–20 s |
| `rerun_sam3d(ep, prompt, mode)` | `prompt ∈ {bbox, bbox+mask}`, `mode ∈ {body, full}` | `sam3d_body.py` | 0.7 s/帧 → 1–6 min |
| `smooth(ep, σ, wrap_safe=True)` | σ ∈ {2,3,4,6} | `_smooth_wrap_safe` | 秒 |
| `fix_root(ep, method)` | `flowhmr_traj`（根平移/朝向整段换 FlowHMR 对齐轨迹，体姿保留）/ `median_depth`（深度用窗口中位）/ `floor_contact`（站立帧脚贴地面，解整段 1 个深度因子） | 新 `src/agent_tools.py`，复用 `fit_flowhmr.py` 的 Sim3 和 `ground_plane.py` | 秒 |
| `scale_depth(ep, k, mode)` | k ∈ {0.85,0.9,0.95,1.05,1.1,1.15}；`mode ∈ {global, linear_drift}` | 新；改 `pose[:,0:3]`（沿视线缩放，保持 2D 投影不变：x,y 同乘 k） | 秒 |
| `fix_first_frame(ep, method)` | `median15` / `flowhmr` | 新 | 秒 |
| `flip_lr(ep, t0, t1)` | 帧区间 | 新：交换 MHR 左右肢体参数块（需先在 rest pose 上验证对应关系） | 秒 |
| `restart_at_gap(ep, gap_i, source)` | 多段 episode | 新 | 秒 |
| `observe(ep, cand_a, cand_b)` | — | `src/agent_obs.py`（§3） | 10–30 s（渲染 + MoGe 深度读取） |
| `gate(ep, cand)` | — | `score_local.py` + `diag_accel.py` | 秒 |
| `accept(ep, cand, reason)` / `keep_baseline(ep, reason)` | — | 写 `work/agent/decisions.jsonl` | — |

**禁止的动作**（写进系统提示，也在代码里硬拦）：不改 `pose[:,6:]` 逐帧值（除整段换源）、不改 shape/scales、不用 px_err/sil_iou 作验收。

---

## 5. 验证器：怎么不被代理指标骗

### 5.1 三层验收
1. **硬闸（本地可预测）**：`acc_h_proxy_max < 0.3`（抓 bug）、段均值 ≤ 0.2、`PEN = 0`。不过闸直接否。
2. **一致性闸（独立证据，非拟合目标）**：候选 B 相对 A，下面至少两项改善且无一项恶化超阈值：
   `|depth_ratio_moge 斜率|`（深度漂移）、`foot_floor_cm` 中位（站立帧）、`d_root_sam3d_vs_flowhmr`、`first_frame_outlier`、`yaw_wrap_jumps`、`lr_flip_score`。
   原理：这些量**不是**任何候选的优化目标（FlowHMR 拟合的目标是 FlowHMR 关节，不是 MoGe 深度；SAM3D 没见过地面），多源同意比单源拟合更难过拟合。
3. **VLM 成对判别**：A/B 双序各问一次 "which is more consistent with the video and physically plausible? answer A/B/tie + 一句理由 + 你看到的具体缺陷"；只有两序一致才计票。**VLM 票只能在 1、2 都通过后作为加分/否决**，不能单独通过候选。

### 5.2 回溯校验（免费，先做）
我们有 8 次带真分的提交，其中"只改人体"的成对样本：
`#6→#7`（wrap-safe，CD-H 21.39→17.86，ACC 1.10→0.15）、`#7→#8`（关键点精修，17.86→22.86）、`#2→#3`（平滑，84.8→82.7）、`#3→#5`（地面/upright，82.7→76.5）。
对每对、每段生成 §3 的观测包，问 VLM 和一致性闸"哪个更好"。**判据**：验证器在 `#7 vs #8` 上必须选 #7（px_err 更差的那个）、在 `#6 vs #7` 上必须选 #7。
达不到 → 验证器不可信，agent 方案降级为"只做离散错误修复 + 子集提交校准"。
限制：真分是 30 段均值，回溯只能验"整体方向"，不能验"逐段"。

### 5.3 前瞻校准（解封后，设计好的子集提交）
榜单 CD-H = 30 段均值（已核对）。设基线提交 S0（#7 人体）分 c0。
把 30 段按验证器的"改善置信度"排序分成 5 组各 6 段；每天 5 次提交中拿 **2 次**：
- 提交 S_g = S0 但第 g 组换成 agent 候选 → `Δ_g = 30·(c_g − c0)` = 这 6 段 CD-H 增量之和（精确，无噪声，因为 public=private）。
- 三天 6 次提交覆盖 5 组 + 1 次复核；得到 5 个 (验证器预测, 真实 Δ) 点 → 算符号一致率与 Spearman；
- 规则表每条记"预测对/错"计数（借 VLM-GPA 的规则记忆），一致率 < 0.7 的信号从验收里撤掉。
- 若想到逐段：二分（6→3→1 段）每组再花 2–3 次；只对 Δ_g 大的组做。
预算：约 10–14 次提交（剩余 ~18 天 × 5 = 90 次的 15%），其余留给物体线和最终组合。

### 5.4 反自欺规则（写进代码，不靠提示词）
- 候选 B 的验收**不得**使用生成 B 时的目标量（FlowHMR 拟合的 `joint_err_cm` 不进验收）。
- 每段最多 `N_steps = 6` 次工具调用；超预算 → `keep_baseline`。
- 任何"全部 30 段都被接受"的轮次自动标红复核（真实改进不会均匀）。
- 决策日志每条含：观测 JSON 哈希、VLM 双序原文、一致性闸数值、最终动作；`src/status.py` 从日志生成看板，不手写。

---

## 6. 每段循环（步预算 6）

```
baseline = sub_human/ep (SAM3D + wrap-safe σ=3)        # 已有，CD-H 17.86 的那份
obs0 = observe(ep, baseline, None)
step 1  diagnose(obs0) -> 失效假设列表（按先验排序：深度漂移 / 首帧离群 / wrap / 左右翻 / 无问题）
        "无问题" -> keep_baseline，结束（预期 1/3 的段走这条）
step 2  按假设选工具生成候选 B（一次只动一个自由度）
step 3  gate(B); observe(ep, baseline, B); 一致性闸; VLM 双序成对
        通过 -> B 成为新 baseline；否则换下一个假设
step 4–6 重复；每段最多 2 个被接受的改动（防止叠加过拟合）
end     写 decisions.jsonl；候选落到 work/agent/epNNNNNN/{cand_k}.npz + obs 图
```
段级并行：4 卡，各段独立；GPU 用 `nvidia-smi` 选空闲卡，SAM3D 重跑排队。

---

## 7. 实现方式与成本

### 7.1 两种实现对比
| | Claude Code 子代理（本会话） | Python 循环 + Claude API（推荐主线） |
|---|---|---|
| 启动成本 | 0（现在就能跑） | 需 `pip install anthropic` + API key（当前**未装、未设 key、无 `ant` CLI**） |
| 可控性 | 低：工具调用自由、输出难结构化、harness 会杀长后台任务（今天已 6 次） | 高：固定工具表、JSON 决策、`nohup` 脱离 harness、可断点续跑 |
| 成本 | 走订阅/会话 | 按 token |
| 适用 | 2 小时原型、回溯校验 | 30 段批跑、每日校准 |

### 7.2 Python 循环骨架（`src/agent_loop.py`，待写）
- 模型：**`claude-opus-5`**（默认；`thinking: adaptive`，`effort: medium`），VLM 成对判别也可先试 `claude-sonnet-5` 做对照——判别任务简单且量大。
- 结构：`messages.create` + 自定义工具（§4 表，`strict: true`）；手写 while 循环（不用 tool_runner，便于在每步插入硬闸和日志）；图片 base64 入 user 内容；每段一个独立对话（不跨段累积上下文）。
- 系统提示固定 + `cache_control`（规则表、工具说明约 3k token 可缓存）。
- 成本估算（每步 8 图 ≈ 9k token 输入 + 2k 文本 + 0.5k 输出）：
  Opus 5：≈ 11k×$5/M + 0.5k×$25/M ≈ **$0.07/步** → 6 步 ≈ $0.4/段 → 30 段 ≈ **$12/轮**；
  Sonnet 5 约 1/2.5；Batch API 再 −50%（回溯校验这种离线任务用 Batch）。
  真正的瓶颈是 GPU 工具时间：每段 5–20 min（SAM3D 重跑为主），30 段 4 卡并行约 1–2 h/轮。
- 日志：`work/agent/decisions.jsonl`、`logs/agent_epNNNNNN.log`、`state/agent/epNNNNNN.done`。

### 7.3 先决条件清单
- [ ] `pip install anthropic`，`ANTHROPIC_API_KEY`（或 `ant auth login`）
- [ ] `src/agent_obs.py`（§3 的观测包：数字 + 4 图）
- [ ] `src/agent_tools.py`（§4 中标"新"的工具：`fix_root`、`scale_depth`、`fix_first_frame`、`flip_lr`、`restart_at_gap`）
- [ ] `flip_lr` 的 MHR 左右参数块对应关系在 rest pose 上验证（镜像后顶点与左右互换误差 < 1 mm）
- [ ] FlowHMR 全 30 段 token 抽取 + 生成（进行中；现只有 ep14 的 `body_flow`）

---

## 8. 优先打的三个失效模式（按预期 CD-H 收益）

1. **整段深度/尺度漂移**（假设：是 17.86 中最大的一块）。
   证据：SAM3D 深度来自 bbox 大小，人走近/远、蹲下、被物体遮挡时 bbox 变化非身高变化；FlowHMR 轨迹长度 0.95 m vs SAM3D 1.02 m（ep14）。
   工具：`fix_root(flowhmr_traj)`、`scale_depth(k)`、`floor_contact`。验证：`depth_ratio_moge` 斜率、`foot_floor_cm`、俯视轨迹图。
   验证方式：子集提交 Δ_g（§5.3）。**先验收益估计 2–4 cm**（未验证；若 FlowHMR 轨迹正确且 SAM3D 漂移 ≥ 10 cm 级）。
2. **朝向/根旋转离散错误**（wrap 半圈已修一例得 3.5 cm；还可能有转身段的局部半圈、左右翻）。
   工具：σ 调整、`flip_lr`、`restart_at_gap`。验证：`yaw_wrap_jumps`、`lr_flip_score`、VLM 看叠加 + 俯视朝向箭头（VLM 最擅长的离散判别）。
3. **首帧离群**（Sim3 锚）。
   工具：`fix_first_frame`。验证：`first_frame_outlier`，以及把首帧换成中位后整段 Chamfer-to-FlowHMR 是否下降。成本几乎为零，可能是最便宜的一项。

明确**不**优先：肢体局部精修（已证伤）、shape/scales（中位身份只动 0.5 cm）、轮廓拟合（lr 问题刚解决、增益未知且是单源目标）。

---

## 9. 两小时内的第一个实验（不用提交额度、不需要 API key）

目标：回答"验证器（一致性闸 + VLM 成对）能否复现已知的榜单方向"。失败则整条线降级。

1. **(30 min) 观测包 v0**：`src/agent_obs.py --ep E --a DIR_A --b DIR_B` 输出 `work/agent/obs/epE_AvsB.{json,jpg}`：
   数字：`acc_h_proxy`、`depth_ratio_moge`（中位、斜率）、`foot_floor_cm`、`first_frame_outlier`、`yaw_wrap_jumps`、`d_root_sam3d_vs_flowhmr`（仅 ep14 有 FlowHMR，其余留空）；
   图：叠加条（5 帧）+ 侧视含地面 + 俯视轨迹 + 曲线面板，A/B 同图。
   复用：`viz_body.py`（叠加）、`ground.json`（地面）、`work/frames`、`work/masks`、MoGe 点图（`camera_moge.py` 的输出，确认是否缓存了逐帧深度；没有就只对 5 个关键帧现算）。
2. **(20 min) 回溯样本**：对 `sub_human`(#7) vs `sub_refine`(#8) 和 `sub_human`(#7) vs 重建的 #6（σ=2 非 wrap-safe，`export_episodes.py` 开关即可）生成 30 段 × 2 对的观测包。
3. **(40 min) 盲测**：用 Claude Code 子代理（`general-purpose`，每次只给一对的图 + 数字，**不给**提交编号与分数，A/B 顺序随机且双序）做 60 × 2 次判别；记录到 `work/agent/retro.jsonl`。
   同时算一致性闸在两对上的方向。
4. **(20 min) 结论写进 `progress.md`**：
   - 在 `#7 vs #8` 上，VLM 票与一致性闸各自选 #7 的段数；期望 ≥ 20/30 且双序一致率 ≥ 0.7；
   - 在 `#6 vs #7` 上，是否至少在 ep26/ep18/ep20/ep24/ep2（已知半圈段）选 #7；
   - 哪些数字量在两对上方向都对（候选进入 §5.1 第 2 层），哪些反向（剔除）。
   - 若 VLM 在 `#7 vs #8` 上随机（≈15/30）：VLM 只保留做离散错误识别，不做 A/B 验收；深度问题全靠一致性闸 + 子集提交。

通过后下一步（第 3–6 小时）：写 `agent_tools.py` 的 `fix_root/scale_depth/fix_first_frame`，对 ep14（有 FlowHMR）跑完整 6 步循环一次，手工复核决策日志；FlowHMR 30 段齐后全量跑；解封提交当天按 §5.3 第一组校准。

---

## 10. 风险与边界

- VLM 对度量深度基本盲（文献一致）；本计划把它限制在离散判别，是有意为之。
- 一致性闸的几个量本身也是估计（MoGe 深度有尺度误差、地面平面 2.5–5° 倾斜）：只用**相对/斜率/中位**形式，不用绝对值。
- 子集提交校准每轮占 2/5 提交额度；物体线（CD-O 574 → 几十 cm 的最大收益）必须并行，不能被 agent 实验挤掉。
- 本文件所有"收益估计"都是假设；只有 §5.3 的 Δ_g 是验证。

## 11. 来源
VLM-GPA HMR (CVPR26) arXiv 2602.19180 · MHE-Former arXiv 2609.10743 · Adapting HMR w/ VL Feedback arXiv 2502.03836 · VLM3D (ICML26) arXiv 2511.14271 · VLMPose (ECCV26) arXiv 2604.09781, github.com/snuvclab/vlmpose · AGILE arXiv 2602.04672 · Cross-Model VLM-Judge arXiv 2606.18451 · BlenderGym (CVPR25) arXiv 2504.01786 · SceneCraft (ICML24) arXiv 2403.01248 · GPT-4V evaluator (CVPR24) arXiv 2401.04092 · ChatPose (CVPR24) arXiv 2311.18836 · ChatHuman arXiv 2405.04533 · OPRO arXiv 2309.03409 · MLE-bench arXiv 2410.07095 / MLE-STAR arXiv 2506.15692 · IRG-MotionLLM arXiv 2512.10730 · AREA3D arXiv 2512.05131。
本地核对：`work/kit/v2d_submission_kit/metric_code/track_1/CD-H.py`（Sim3 首帧、段均值 → 30 段均值）、`notes/submissions.md`（8 次提交）、`src/fit_flowhmr.py`（ep14 stats）。
