# BEHAVE 本地验证集（Track 1，带真值）

建立：2026-10-18 22:30–23:05 PDT。用途：方法改动先在有真值的数据上用和官方同构的指标打分，再决定是否花榜单提交。
**本地分数是消融信号，不是榜单预测**（原因见 §6 第 1 条）。

## 1. 许可

- 页面：https://virtualhumans.mpi-inf.mpg.de/behave/license.html 。**不需要注册或填表**，直接 HTTP 下载，下载或使用即视为同意条款。
- 条款要点：仅限**非商业的科研用途**；只能装在自己或单位控制的机器上；未经 Max-Planck 书面许可**不得再分发或修改后分享**；公开发表图像要**给人脸打码**；报告结果要引用 BEHAVE（CVPR'22）。
- 对我们的含义：本地验证可以。**不要把 `work/behave*` 的帧或网格发到看板或公开 artifact**（看板只放数字）。比赛 FAQ 允许外部数据。

## 2. 下载了什么（总计 2.10 GB 网络传输，`data/behave/` 占盘 3.10 GB，含解压）

| 文件 | 大小 | 用途 |
|---|---:|---|
| `behave-30fps-params-v1.tar` → `data/behave/params/<seq>/{smpl_fit_all,object_fit_all}.npz, info.json` | 159.0 MB | 30 fps SMPL-H（`fit03`）和物体（`fit01-smooth`）参数，全部 321 段 |
| `calibs.zip` → `data/behave/calibs/calibs/{DateXX/config/k/config.json, intrinsics/k/calibration.json}` | 395.9 MB | 外参（local→world）和彩色内参/畸变 |
| `objects.zip` → `data/behave/objects/objects/<obj>/<obj>_fNNNN.ply` | 196.5 MB | 物体模板（注册时用的简化网格） |
| `video/date01_time.tar`、`date02_time.tar` → `data/behave/time/*.time.json` | 5.9 + 5.6 MB | 每台相机每帧的时间戳 |
| `split.json`、`behave-30fps-error_frames.json` | <0.1 MB | 划分、坏帧列表 |
| 8 段 × 4 相机的 `*.color.mp4` → `data/behave/video/` | 1.34 GB（实际使用的 k1：0.38 GB） | 彩色视频，2048×1536，30 fps |
| HF `nvidia/CARI4D/mhr2smplh_mapping.npz` → `work/behave_gt/_assets/` | 1.2 MB | SMPL-H 顶点 → MHR 面上重心坐标的对应 |

彩色 mp4 不是整包下的（`date01_color.tar` 5.9 GB、`date02_color.tar` 5.8 GB）。做法是用 HTTP Range 遍历 tar 头，建索引
（`data/behave/dl/color_tar_index.json`），再只取需要的 mp4。另外 3 台相机的 24 个 mp4（0.96 GB）只是选相机时用到的，可以删。
为了核对约定，另从 `Date01.zip`（15 GB）里按 zip 中央目录 Range 取了 1 帧的 `person_fit.{ply,pkl}`、`chair_fit.{ply,pkl}`，约 7 MB，放在 scratchpad，不入库。

## 3. 选的片段

全部用 **Kinect 1**，每段取 **450 帧（15 s，30 fps）连续**。窗口的取法：在连续的 30 fps 网格上、避开 error_frames，取物体移动路径最长的那 15 s。

| seq | 物体 | GT 帧区间 [s,e) | 视频帧（k1） | 人深度 | 物体网格面数 | 拟合残差 corr / 面距离 (cm) |
|---|---|---|---|---:|---:|---|
| Date01_Sub01_chairwood_hand | chairwood | 585–1035 | 667–1116 | 2.50 m | 2499 | 0.899 / 0.528 |
| Date01_Sub01_stool_move | stool | 960–1410 | 1038–1487 | 2.32 m | 1000 | 0.962 / 0.536 |
| Date01_Sub01_trashbin | trashbin | 270–720 | 351–800 | 2.39 m | 999 | 0.818 / 0.503 |
| Date01_Sub01_boxmedium_hand | boxmedium | 375–825 | 457–906 | 2.25 m | 1000 | 0.908 / 0.537 |
| Date01_Sub01_plasticcontainer | plasticcontainer | 90–540 | 170–619 | 2.27 m | 1000 | 0.764 / 0.483 |
| Date02_Sub02_boxsmall_hand | boxsmall | 105–555 | 177–626 | 2.08 m | 1000 | 0.853 / 0.509 |
| Date02_Sub02_tablesmall_move | tablesmall | 585–1035 | 664–1113 | 2.08 m | 999 | 0.829 / 0.502 |
| Date02_Sub02_toolbox | toolbox | 240–690 | 322–770 | 2.00 m | 1000 | 0.907 / 0.526 |

**选相机的依据**（实测，不是假设）：
- Date01/02 上 k0 和 k3 的彩色流**丢帧 20–40%**：约 50 s 只有 1100–1250 个时间戳，15 s 窗口内会有 71–184 帧重复。只有 k1 和 k2 是干净的 30 fps。
- k1 和 k2 之间：前 2 段 k1 正对人的帧占 90%（k2 只有 19–27%），其余各段两者都在 50% 左右，人体都完整在画面内（99.4–100% 的顶点）。所以全部统一用 k1。
- k1 恰好就是 BEHAVE 的世界系（`config/1` 为单位阵），外参只对 k1 以外的相机起作用。
- toolbox 段 k1 有 1 帧重复（时间戳同步按官方 `video2images.py` 的规则取最近帧），其余 7 段 0 重复。

## 4. 产物

- `work/behave/<seq>/frames/000000…000449.jpg`：k1 彩色帧，**已用 OpenCV 去畸变**，K 保持不变，原生 2048×1536，JPEG q95。
- `work/behave/<seq>/meta.json`：fps、fx/fy/cx/cy（针孔，去畸变后的）、物体名、prompt、GT 帧区间和时间戳、对应视频帧号、来源。
- `work/behave_gt/<seq>/`：
  - `episode_000000.npz`：我们的导出格式。pose[450,136]、scales[68]、shape[45]，以及 object_rotation[450,3,3]、object_translation[450,3]、object_scale=1，额外附 fit_residual_cm。
  - `episode_000000_object.obj`：物体模板居中后的网格。面数最多 2499，本来就在 4096 的预算内，`budget_mesh` 是 no-op，没有调用。
  - `scored_frames.json`：0..449 作为一整段连续区间。
  - `gt_smplh.npz`：相机系下 SMPL-H 的顶点和关节，外加原始参数。
  - `object_pose.npz`、`fit_report.json`。
  - `overlay_check.jpg`（SMPL-H + 物体投影）、chairwood 额外有 `overlay_mhr.jpg`（拟合出的 MHR 投影）。
- 代码：`src/behave_smplh.py`（torch SMPL-H）、`src/behave_prep.py`、`src/behave_fit_mhr.py`、`src/score_behave.py`、`src/behave_sanity_sam3d.py`。

## 5. 转换方法和验证

**坐标系 / 约定**，每一项都写了怎么验的：

| 项 | 结论 | 验证 |
|---|---|---|
| SMPL-H 模型文件 | 用 MANO v1.2 的 `SMPLH_MALE.pkl`（chumpy 对象手工解包，存为 `work/behave_gt/_assets/smplh_{male,female}_v12.npz`） | 用 BEHAVE 自带的 `person_fit.pkl` 参数重建，和 `person_fit.ply` 比：**0.0002 mm**。如果用 `work/models/smplh/camd/.../SMPLH_MALE.npz`，误差是 9.2 mm 均值、18.9 mm 最大（shapedirs 不同） |
| SMPL-H 前向 | 52 关节全轴角、无手部均值、trans 在蒙皮后加 | `src/behave_smplh.py` 和 smplx（flat_hand_mean）对比 0.0005 mm；见上一行 |
| 物体 | `R=rotvec(angles)`，`v = R·(tmpl − mean(tmpl)) + trans`，模板用 `_fNNNN.ply` 简化网格 | 和 `chair_fit.ply` 比 **0.0002 mm** |
| 相机 | `config/k` 是 local→world，世界系 = k1 彩色相机，OpenCV 系（x 右、y 下、z 前，米） | 投影叠图（3 帧/段 × 8 段）人和物体都对得上 |
| 时间 | `frame_times` 是相机时钟秒数；按 `video2images.py` 的规则取 4 台相机中最近的时间戳，再取 k1 中最近的帧 | 叠图里首/中/末帧对齐 |
| MHR 映射 | `mhr_faces` 和我们 `mhr_model.pt` 的 faces 完全相同；gender=male | 逐元素比较 |

**SMPL-H → MHR**：做法同 `src/fit_mhr.py`，直接对 MHR 参数做梯度优化。损失 = 6890 个 SMPL-H 顶点到映射后 MHR 面点（重心插值）的 L2。
分三步：
1. 24 个八面体根朝向多起点，只拟合刚体（每 15 帧取一个关键帧）；
2. 关键帧上拟合全部姿态 + 共享 identity；
3. 全部 450 帧从最近的关键帧初始化，联合精修（Adam，余弦退火 300 轮）。

shape/scales 每段一份。每段 1–2.5 min（单卡）。

**残差，也就是 CD-H 能测到的下限**：

- 8 段平均：对应点距离 **0.87 cm**，SMPL-H 顶点到 MHR 表面 **0.52 cm**，MHR 顶点到 SMPL-H 表面 **0.52 cm**。
- 单帧最差 3.2 cm（stool_move），顶点 p95 1.8–2.2 cm。
- 映射文件自带的 `distances`（官方在模板上做的 SMPL-H→MHR 面距离）均值就是 0.54 cm，我们已经到了这个底。
- 所以本地 CD-H 低于约 0.5 cm 的差异没有意义。实际的方法差异在 cm 量级，不受影响。

## 6. 打分

```bash
python src/score_behave.py --pred <dir>  [--seqs ...] [--frames a-b] [--out res.json] [--device cuda:0]
# <dir>/<seq>/episode_000000.npz + episode_000000_object.obj，导出格式和比赛一样（pose 按 clip 帧 0..449 索引）
```

和 kit 同构的地方：
- 第一个计分帧做一次 Sim(3)（alignment 顶点），整段共用；
- CD-H 是稀疏体表点的对称 Chamfer；
- CD-O 是物体表面采样点经同一个 Sim(3) 后的对称 Chamfer；
- ACC-H 是 22 个关节的二阶差分误差，ACC-O 只看平移，都按连续段算；
- PEN 是我方手部点在我方物体里的深度，乘 align scale。

直接复用 `src/score_local.py` 的 `load_episode / compare / penetration`。

**自检**：GT 对 GT，CD-H 0.0001、CD-O / ACC 全 0、align_scale 1.000。

**端到端 sanity（只跑了这一次，不是流水线结果）**：
- 设置：SAM 3D Body，body 模式，chairwood 段第 200–209 帧，bbox/mask 用 GT 投影（只测约定和打分链路），转换方式同 `export_episodes._sam3d_as_body`。
- 结果：**CD-H 6.86 cm**、ACC-H 1.04、align_scale 0.992。
- pred_cam_t 和 GT 骨盆深度相差约 5 cm（2.84 vs 2.79 m），说明帧约定正确。
- CD-O 6.0 是 GT 物体经人体 Sim(3) 泄漏出来的误差，不代表物体估计。

## 7. 已知局限

1. **点集是代理**：host 的 role_alignment / role_body / hand 索引不在 kit 里，用的是 `mhr_local` 的 FPS 代理（64 体表点、256 对齐点），和 `score_local.py` 一样。绝对数和榜单不可比，只看 A/B 的方向和幅度。
2. **GT 本身有噪声**：30 fps 标注（fit03 / fit01-smooth）和逐帧注册（fit02 / fit01）在同一帧上差约 3 cm（人：均值 2.99 cm、最大 8.3 cm；物体：3.0 cm）。BEHAVE 是 4 台 Kinect 的注册结果，不是动捕。
3. **ACC 标尺不同**：BEHAVE GT 自身关节加速度均值 0.48 cm/f²（0.33–0.75），比赛 GT 是 0.145；物体 0.36。所以在这里，完全静止的预测 ACC-H 约 0.48。本地 ACC 只用来比较平滑策略的相对好坏。
4. **域差异**：室内单人，Kinect 2–2.5 m 距离，视场小（fx≈980 @2048），比比赛相机（fx≈820 @1536）远且窄。去畸变后图像四边有黑边。物体是 BEHAVE 的 8 类，没有呼啦圈、推车这类。
5. **性别和受试者**：8 段都是 Sub01/Sub02（男）。映射文件是男性模板。女性受试者（Sub06–08）没测过。
6. 每段只有 1 个相机视角。想要更多数据可以换 k2，或者从 Date01/02 再加段（视频 ~45 MB/段/相机）。k0/k3 在这两天丢帧，别用。
7. 物体网格是 BEHAVE 注册用的简化模板（1000–2500 面）。我们自己的流水线会自建网格，CD-O 的比较对象是这个模板。
