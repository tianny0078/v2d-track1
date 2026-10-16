# Track 1 计划 v2（实测修订版）

> v1 见 `track1-plan.md`（Google Doc 快照，2026-10-15）。本文件是 **在本地跑过数据和评测代码之后** 的修订版。
> 修订时间：2026-10-15。所有"已验证"的条目都附带验证方式。
> 原则：v1 的结论凡被实测推翻的，这里明确写"**v1 修正**"。

---

## 0. 本机环境（v1 未知，现已确定）

| 项 | 实测值 | 对计划的影响 |
|---|---|---|
| GPU | **4× RTX PRO 6000 Blackwell Max-Q, 96 GB each（共 384 GB）** | 官方 `MESH_TO_MHR_PARAMS.md` 的性能数字正是在 RTX PRO 6000 (Blackwell) 上测的 → 官方转换器属于已验证路径 |
| GPU 共享 | **可能有其他进程占用，必须监控** | 所有长任务按单卡可被抢占设计：按 episode 切分、可中断、可续跑；调度前查 `nvidia-smi` 空闲显存 |
| CPU / RAM | 128 核 / 995 GB | Chamfer、SDF、网格处理可大量走 CPU 并行（host 的 PEN 就是 8 线程 numpy） |
| 磁盘 | `/` 3.6T，**仅剩 293 G（92% 满）** | 帧缓存必须压缩/按需；16,563 帧 1536×1152 PNG ≈ 80 GB，不能无脑全解码 |
| Python | 系统仅 **3.10**；kit 要求 **3.11+**（`kaggle>=2.0` 装不上） | ⚠️ 提交前必须建 3.11+ 环境，否则 `submit/v2d_submit.py` 跑不了。已列为阻塞项 B-1 |
| torch | 2.9.1+cu128，4 卡可见 | sm_120 可用 |

**v1 修正**：v1 说"不假设未知 GPU 可用时长"是对的，但低估了可用算力。4×96GB 意味着**可以三条路线真正并行**，而不是必须串行。瓶颈从算力变成了"人（我）的验证带宽"和"提交次数"。

---

## 1. 数据：已下载并核对（v1 为纸面推测）

```
/home/tianyuan/projects/v2d_track1/data/hf/track_1/   490 MB，67 文件，13 秒下完
  videos/chunk-000/observation.images.exo_camera/episode_0000NN.mp4   30 个，1536×1152 H.264 30fps
  data/chunk-000/episode_0000NN.parquet                               仅 LeRobot 索引列
  meta/{info,episodes,episodes_metadata,episodes_stats,tasks}.jsonl
```

已核对：30 episode / 16,563 帧 / 22 个 task 描述 / 10 个物体 × **各 3 段** / 4 个物理机位
（left 11、front 9、back 5、right 5）。`data/` 确实**不含任何真值**，v1 的警告成立。

**10 个物体**：`hula_hoop, big_red_bowl, iron, white_desk, black_pan, foam_grass_block,
paint_roller, pink_foam_roll, short_wood_stool, white_laptop_cart`

**v1 未注意到的物体特性**（直接决定难度分层）：
- 大件家具：`white_desk`、`white_laptop_cart`、`short_wood_stool` — 尺寸远超常见 HOI 数据集（BEHAVE 量级），单目深度/尺度先验会系统性偏小
- 细长/薄环：`hula_hoop`、`paint_roller`、`pink_foam_roll` — 分割掩码细、CD-O 对朝向极敏感、PEN 容易误判（手环绕但不穿透）
- 旋转对称：`black_pan`（柄破对称）、`big_red_bowl`（绕轴对称）、`pink_foam_roll`、`paint_roller` — 朝向翻转风险最高
- `foam_grass_block` 近似立方体 → 24 重离散对称

**每个物体 3 段 → 必须共享同一张网格**。这不是取巧：同一物理物体的几何在三段里就是同一个，复用能消除 CD-O 的几何方差、统一尺度。
（v1 说"不得默认所有视频的物体坐标和尺度已对齐"仍然正确 —— 共享的是 **mesh 几何**，每段的 `R,t,s` 独立估计。）

---

## 2. 提交格式：已逐位解码（v1 只有定性描述）

`data/track_1_sample_submission.parquet`：**740,780 行**，列 `row_id,x,y,z,code_commit_url`。
`row_id = t1_{episode:06d}_{frame:06d}_{array_index:1d}_{point:06d}`，`frame=999999` 表示 episode 级。

| array_index | 名称 | 粒度 | 行数/单位 | 总行数 | 说明 |
|---|---|---|---|---|---|
| 0 | `mhr_pose` | 帧 | 46 (=⌈136/3⌉) | 454,342 | `[0:3]`根平移(MHR原生单位) `[3:6]`根欧拉 `[6:136]`体姿 |
| 1 | `mhr_scales` | episode | 23 (=⌈68/3⌉) | 690 | `mhr_model_params[136:204]`，整段恒定 |
| 2 | `mhr_shape` | episode | 15 (=⌈45/3⌉) | 450 | 身份 shape，整段恒定 |
| 3 | `object_rotation` | 帧 | 3（R 的三行） | 29,631 | |
| 4 | `object_translation` | 帧 | 1 | 9,877 | |
| 5 | `object_scale` | episode | 1，写在 x，**y=z=0** | 30 | 正标量 |
| 6 | `object_mesh_vertices` | episode | **4096** | 122,880 | 固定预算 |
| 7 | `object_mesh_faces` | episode | **4096** | 122,880 | 顶点索引写成 float |

- 末行未用的尾部必须填 0。
- 网格预算**固定 4096/4096**；不足由 packer 补齐（顶点补任意有限值，面补 `0,0,0` → 零面积，永不被采样）。所以**实际可用网格可以远小于 4096 面**，不必强行做满。
- `point_scene = s · R · point_mesh + t`

### 2.1 评分帧 = 手物接触区间（**v1 完全没有这一条，是本次最重要的发现之一**）

kit 源码原文：*"Every valid frame inside the hand-object contact span is scored, grouped into
contiguous stretches separated by invalid frames."*

实测：**9,877 / 16,563 帧被评分（59.6%）**，每段 83–803 帧。绝大多数 episode 的首个评分帧是
**frame 90**（= 3.0 秒，人走到物体前），episode 11 从 0 开始、全段 803 帧都评分。

| | |
|---|---|
| 单一连续段 | 21 个 episode |
| 多段（2–7 段） | 9 个：ep 0(6), 1(6), 2(3), 3(7), 4(2), 5(6), 12(5), 13(4), 14(4), 20(2) |
| 最短 | ep 14：83 帧 | 
| 最长 | ep 11：803 帧 |

**推论（改变方法优先级）：**
1. 评分窗口内**几乎每一帧都有接触** → 接触约束不是"只在估计接触阶段启用"的可选项，而是**全程主约束**。v1 把接触当成 B 路线的增量项，低估了它。
2. ACC 的二阶差**只在连续段内计算**，跨 gap 不计 → 多段 episode 的段边界不会被惩罚，可以在 gap 处允许轨迹重启/重定位，这是处理遮挡后翻转的**免费修复窗口**。
3. 非评分帧（前 90 帧等）仍要跑，用于初始化和时序传播，但其误差不计分 → 可以把算力倾斜到评分窗口。
4. 评分帧清单**已经公开**（就在 sample_submission 里），v1 列为"待确认问题"的"评分帧清单"**已解决**。

### 2.2 对齐与度量（已读完 `metric_code/track_1/*.py` 全部 5 个）

- 五个指标的 host 代码**完整随 kit 发布**，连 MHR 前向都有一份 NumPy 重实现（`_MHRBody`）。
  缺的只有 asset npz（含 `role_alignment / role_body / role_left_hand / role_right_hand` 顶点索引）和参考真值。
- **Sim(3) 只在首个评分帧用 `mhr_vertices[0]`（= `role_alignment` 顶点）拟合一次**，之后整段人与物共用。无反射，scale 必须 > 0。
- `human_surface_points = vertices[:, role_body]`；`object_surface_points` 从**我们提交的网格**上采样
  **同样数量**的点（`len(body.roles["body"])`，seed 按 episode 固定）→ CD-H 与 CD-O 的点数相同，
  组合 Chamfer 对人和物等权。**CD-O 的分辨率被 `|role_body|` 锁死**（文档提到 CD-H 用 64 个体表顶点）。
- ACC：`a[t] = p[t] - 2p[t+1] + p[t+2]`，**不乘 fps²**，首尾各复制一次补齐长度，单位 cm/frame²。
- CD = 对称 Chamfer（两个方向的最近邻均值**相加**，不是取平均）。

### 2.3 ⭐ PEN 是自指的 —— 可以本地精确优化到 ~0（**v1 未发现，战略级**）

`_episode_penetration` 原文：*"the participant's hand against the participant's **OWN** object mesh."*

```python
align = fit_similarity_transform(pred_mhr_vertices[0], target_mhr_vertices[0])[0]   # 唯一的真值入口，一个标量
local = (hands - t) @ R / s                      # 手点进我们自己的 mesh 坐标系
depth = penetration_depth(local, 我们自己的 mesh_vertices, 我们自己的 mesh_faces)
PEN   = mean(relu(depth) * s * align) * 100      # cm，外部点计 0 但分母仍是全部 512 个手点
```

- 真值**只通过 `align` 这一个标量**进入 PEN。其余全是我们自己的输出。
- → **PEN 可以在本地逐帧精确计算并直接优化**，不需要任何榜单反馈。
- → 把 512 个手点推到自己网格外侧（哪怕 0.5mm），PEN 就是 **0.0**，这是五个榜单之一的近乎免费的第一名。
- 风险与边界：PEN→0 必须**不以牺牲 CD-O 为代价**。合法做法是在联合优化里加硬约束
  `signed_distance(hand_point, own_mesh) ≥ 0`，而不是把物体推远（推远会抬高 CD-O 和 ACC-O）。
  真实抓握中手指本就该贴着表面而不是嵌入，所以这个约束与物理合理性**同向**，不是 hack。
- 实施：手点集用 MHR 蒙皮权重归属到腕/指关节的顶点近似（官方 `role_left_hand/right_hand` 未随 kit 发布）。
  近似集合比真集合大是安全的——在更大的点集上无穿透 ⟹ 在真子集上也无穿透。

### 2.4 榜单结构（v1 的"最终排名聚合"问题，部分解决）

- `tracks.json` 注释：**`private_fraction is 0` —— 每一个评分行都出现在 Kaggle Public 榜上。**
  → **没有 public/private 洗牌，榜单分数就是最终分数。** 这彻底改变了提交策略（见 §5）。
- 官网：两轴等权。轴 1 准确性 = CD-H + CD-O；轴 2 物理合理性 = ACC-H + ACC-O + PEN。
  FAQ 另述各指标名次求和 —— 仍需向组织者确认，**但两种聚合下 PEN 都是独立一票**，§2.3 的结论不变。
- 奖项：Track 冠军 DGX Spark（$6K），亚军 $3K GPU credits，总奖池 $54K。
- 冻结：**2026-11-04 (三) 17:00 EST**；获奖公布 11-12。距今 **20 天**。

---

## 3. MHR 接口：已跑通（v1 的头号风险，现已降级）

```
models/mhr_model.pt   sha256 352e271a...377bc  ✅ 与官方文档一致（Apache-2.0 的 Meta MHR 发布版，
                                                  **不是** SAM 3D Body 里那份不同许可的副本）
forward(identity_coeffs[B,45], model_parameters[B,204], face_expr_coeffs[B,72], apply_correctives=True)
        -> (vertices[B,18439,3], joints[B,127,8])
```

实测（cuda:0, B=64）：**0.51 ms/帧，峰值 1.07 GB**。`face_expr_coeffs` 维度是 **72**（文档未写，零填充即可，与 GT 一致）。
单位核对：原生输出 cm，`diag(1,-1,-1)·v/100` 后人体高 **1.727 m** ✅ 与文档的坐标约定一致。

→ **Gate A（MHR 接口）通过。** v1 中"MHR 是首先要解决的接口"的风险已拆除，整段 16,563 帧前向 < 10 秒。
→ `pose[0:3]` 根平移保持 **MHR 原生单位（cm）**，不要整体除 100 —— v1 这条警告正确，务必保留。

---

## 4. 路线修订

v1 的 A 保底 / B 主攻 / C 难例三分法**保留**，但重心按新发现调整：

### A｜MHR 原生人体 + 模板化物体跟踪（保底，必须最先打通）
链路：抽帧 → SAM 3 分割（人 + 目标物，`object_prompt` 直接可用作文本提示）→ SAM 3D Body 出 MHR 参数
→ 每段身份（scales/shape）固定 → **每个物体一张共享网格**（SAM 3D Objects 在该物体 3 段里可见性最好的帧上重建，
人工挑帧 + 对称性消歧）→ 单目深度初始化尺度 → 物体 6D 跟踪 → 官方 packer 打包。

**v1 修正 — 物体跟踪不要默认 FoundationPose**：它需要深度 + 带纹理的参考资产，且 Blackwell 上的 CUDA 扩展编译是已知风险。
先用**可微渲染 + 掩码/轮廓对齐的直接优化**（纯 PyTorch，无自定义 CUDA kernel，Blackwell 零风险），
FoundationPose 作为可选加速器而非前提。这把 v1 里"FoundationPose 的深度条件需上游提供"的隐患直接消除。

### B｜联合优化（主攻）—— 重心从"加接触"变成"接触是主约束"
在 A 初始化上联合优化 {人体姿态, 物体 R/t, 共享尺度 s, 接触}。损失：
掩码/轮廓重投影 + 接触吸附 + **穿透硬约束（§2.3，直打 PEN）** + 速度/加速度先验（温和，不过平滑）。
保留多位姿候选对抗对称翻转。先固定网格优化轨迹，再对困难物体改几何。

**新增**：在**多段 episode 的 gap 处允许轨迹重启**（§2.1 推论 2），这是 v1 没有的免费修复手段。

### C｜多关键帧几何 + 分段恢复（难例补强）
只用于 CD-O 高风险物体（细长/对称/大件）。从遮挡少、朝向多样的帧提多个几何/朝向候选，
用多帧轮廓一致性选择。人体复用 A/B。GVHMR 仅作时序/重力一致性参考，**不替换人体模块**（其输出模型非 MHR）。

### 相关工作（v1 之后新检索到，均 CVPR 2026 前后）
- **CARI4D**（CVPR 2026, NVIDIA+Tübingen+MPI）—— 本赛题评测代码就是"vendored from Bowen's CARI4D evaluator"，
  **出题人的方法就是 CARI4D**。类别无关、度量尺度、位姿假设选择 + render-and-compare + 接触细化。比 prior art 好 35%/36%。
- **ArtHOI**（CVPR 2026）—— Adaptive Sampling Refinement 优化物体度量尺度与位姿；MLLM 引导的手物对齐，
  把接触推理当作约束。**它的 ASR 尺度策略直接可借鉴到我们的 §4-B 尺度自由度**。
- **CrossHOI**（CVPR 2026）—— 跨视角表征用于单目 HOI；评测同样是 Procrustes 对齐后的 CD-H/CD-O。
- **MetricHMSR**（CVPR 2026）—— 单图度量尺度人体 + 场景恢复，可作尺度先验的替代源。
- **RHINO** / **Mesh4D**（CVPR 2026）/ *Efficient and Scalable Monocular HOI Motion Reconstruction* —— 候选对照。
- **SAM 3D Body**（arXiv 2602.15989, CVPR 2026 poster）—— 首个输出 MHR 的模型，`facebook/sam-3d-body-dinov3` 已在 HF。
- **SAM 3D Objects** —— 注意其输出是**归一化相机坐标下的位姿与任意全局尺度**，度量尺度需要另行恢复（与 ArtHOI/CARI4D 的做法一致）。

社群渠道：NVIDIA Omniverse Discord `#v2d-challenge` 是主要讨论区（不在公开网页上）。
社群已流传的一条建议：**用已知物体的模板做跟踪，胜过从零重建物体** —— 与本计划 §1 的"每物体共享网格"一致。

---

## 5. 提交策略（**因 `private_fraction=0` 而整体重写**）

v1 说"只在完整有效候选上消耗正式提交次数"。**这条在 public=private 的前提下是错的**：

- 榜单分数 = 最终分数，没有过拟合 public 榜的风险（因为没有 private 榜）。
- 五个指标是**五个独立 Kaggle 竞赛**，同一个 parquet 投五次 → 一次提交拿到五个真实指标读数。
- → **榜单就是我们唯一的验证集**（真值全隐藏，本地任何代理指标都不能证明得分）。

**新策略：尽早、频繁、带消融地提交。**
1. **D+1 目标：一次 "全零但格式合法" 的提交**，拿到五个指标的"地板分"基线，同时端到端验证 packer/uploader。
2. 之后每个有意义的改动都提交，用榜单做 A/B。先查清每个竞赛的**每日提交上限**（阻塞项 B-2）。
3. 保留每次提交的 `(commit sha, 配置, 五指标分数)` 台账 → `notes/submissions.md`。

---

## 6. 排期（剩 20 天，截止 11-04 17:00 EST）

| 窗口 | 目标 | 门槛（不达标就砍） |
|---|---|---|
| 10/15–10/16 | 环境、MHR 回环、packer 跑通、**零基线提交** | 能产出被 Kaggle 接受的 parquet |
| 10/17–10/19 | A 路线单段打通 → 全 30 段跑完 | 30 段都有合法 NPZ + 网格 |
| 10/20–10/24 | B 联合优化主线；PEN→0 约束落地；每物体共享网格定版 | 榜单上 CD-O 或 PEN 有可见改善 |
| 10/25–10/29 | C 难例（细长/对称/大件）；多段 gap 重启；消融 | 每项改动都有榜单读数 |
| 10/30–11/02 | 误差分层、复核、技术报告、代码 commit 公开 | 候选冻结 |
| 11/03–11/04 | 最终提交 + 故障缓冲 | — |

**消融清单**：A0 分模块 → A1 固定身份 → **A2 共享网格** → B1 接触吸附 → **B2 穿透硬约束(PEN)** →
B3 遮挡鲁棒 → B4 多候选 → **B5 gap 重启** → C1 多关键帧几何。（粗体为 v1 没有的新项。）

---

## 7. 阻塞项与待确认

| ID | 项 | 状态 |
|---|---|---|
| B-1 | kit 需 Python 3.11+（`kaggle>=2.0`），本机仅 3.10 | ✅ **已解决**：`work/venv-submit`（uv, Python 3.12.15, kaggle 2.2.4） |
| B-2 | 五个 Kaggle 竞赛的每日提交上限 + 规则接受 | ✅ **已解决**：规则已接受，**每竞赛 5 次/天**（20 天共 ~100 次） |
| B-3 | `gh` 未登录；提交需要公开 commit URL | **待办**：需 GitHub 仓库 + `gh auth login` |
| B-4 | `role_alignment/body/left_hand/right_hand` 顶点索引未随 kit 发布 | 以 MHR 蒙皮权重近似；可向组织者索取 |
| B-5 | 五指标最终聚合方式（官网两轴等权 vs FAQ 名次求和） | 向组织者确认；两种下策略不变 |
| B-6 | GPU 可能被其他进程占用 | 已纳入调度与监控 |

**已从 v1 的待确认列表中解决**：评分帧清单（§2.1）、MHR 版本与导出单位（§3）、
物体几何复用边界（§1，同物体 3 段共享网格）。

---

## 8. 与 v1 的差异摘要

| # | v1 | v2（实测后） |
|---|---|---|
| 1 | 算力未知，三路线建议串行 | 4×96GB Blackwell，可真并行；瓶颈是验证带宽与提交次数 |
| 2 | 评分帧"以 sample_submission 为准"（未解析） | 已解析：9,877 帧 = 手物接触区间，21 段连续 / 9 段多段 |
| 3 | 接触约束是 B 路线的增量项 | **接触是全程主约束**（评分窗口几乎全程接触） |
| 4 | 未发现 PEN 的自指性 | **PEN 自指，可本地精确优化到 ~0**，五榜之一近乎免费 |
| 5 | "不擅自消耗提交次数" | `private_fraction=0` → **榜单即验证集，尽早频繁提交** |
| 6 | FoundationPose 作为物体跟踪主力 | 降级为可选；主力改为纯 PyTorch 可微渲染对齐（Blackwell 零编译风险） |
| 7 | 网格预算"4096 由 packer 处理" | 固定 4096/4096 且**允许补零面**，实际网格可更小更干净 |
| 8 | 未提多段 gap | **gap 处 ACC 不跨段计算 → 免费的轨迹重启窗口** |
| 9 | MHR 接口为头号风险 | 已跑通并计时，风险拆除 |
| 10 | 相关工作止于 CARI4D/SAM3D | 补 ArtHOI / CrossHOI / MetricHMSR / RHINO / Mesh4D；确认**出题人评测器即 CARI4D 作者的代码** |

---

## 9. 来源

本地已验证：`work/kit/v2d_submission_kit/`（README、`tracks/track_1/README.md`、
`tools/pack_reconstruction.py`、`tools/track1/MESH_TO_MHR_PARAMS.md`、`config/tracks.json`、
`v2dlb/{mhr_metrics,mhr_submission,vertices,track3_pose}.py`、`metric_code/track_1/{CD-H,CD-O,ACC-H,ACC-O,PEN}.py`）、
`data/hf/track_1/`、`models/mhr_model.pt`。

外部：[赛事官网](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) ·
[CARI4D](https://nvlabs.github.io/CARI4D/) ([arXiv](https://arxiv.org/abs/2512.11988)) ·
[ArtHOI (CVPR26)](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_ArtHOI_Taming_Foundation_Models_for_Monocular_4D_Reconstruction_of_Hand-Articulated-Object_CVPR_2026_paper.pdf) ·
[CrossHOI (CVPR26)](https://openaccess.thecvf.com/content/CVPR2026/papers/Geng_CrossHOI_Learning_Cross-View_Representations_for_Monocular_3D_Human-Object_Interaction_Reconstruction_CVPR_2026_paper.pdf) ·
[MetricHMSR (CVPR26)](https://openaccess.thecvf.com/content/CVPR2026/papers/Song_MetricHMSR_Metric_Human_Mesh_and_Scene_Recovery_from_Monocular_Images_CVPR_2026_paper.pdf) ·
[RHINO](https://arxiv.org/pdf/2605.17014) · [Mesh4D](https://github.com/jzr99/Mesh4D) ·
[SAM 3D Body](https://arxiv.org/abs/2602.15989) · [MHR](https://facebookresearch.github.io/MHR/) ·
[Track 1 数据](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/track_1/README.md) ·
[NVIDIA 开发者论坛公告](https://forums.developer.nvidia.com/t/nvidia-video-to-data-v2d-challenge-turn-videos-into-robot-skills/384050)
