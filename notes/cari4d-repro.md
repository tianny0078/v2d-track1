# 官方 CARI4D baseline（`nvidia-isaac/video_to_data` → `reconstruction/modules/v2d_cari4d`）复现记录

> 2026-10-18 开始。目标：把出题方的 Track 1 baseline 流水线（榜上 "CARI4D (commercial) Baseline"：CD-H 15.58 / CD-O 23.16）
> 在我们 30 集上原样跑通，导出成我们的提交格式（`work/sub_cari4d/`），过门控。
> 仓库：`work/v2d_official/`（浅克隆，commit `193382a`，2026-10-08，与 `work/v2d_latest` 同一 commit）。
> 本文件只记事实；进度流水在 `progress.md`。

## 0. 结论先行（随进展更新）

- **（2026-10-19 更新：已解锁并完成，见 §6）** 原阻塞：CoCoNet 权重 `nvidia/cari4d_commercial`（HF，gated=auto，需登录账号接受条款）本机没有任何 HF token**
  （`~/.cache/huggingface/token`、`HF_TOKEN`、`.bashrc`、`.netrc` 全无；匿名 `resolve` 返回 401）。
  → 第 06 步（CoCoNet 时空网络）**跑不了**。其余 7 步全部可跑。
  没有 CoCoNet 时，第 07 步（接触引导精修）从**初始化**（SAM 3D Body + FoundationPose）起跑，
  用 `src/cari4d_init_bundle.py` 伪造第 06 步的 bundle（接触 logits 用几何门限代替网络预测）。这是 "CARI4D 去掉 CoCoNet" 的版本，
  **不是**榜上 baseline 的完整复现；拿到 token 后只需补跑 06/07 两步（`src/cari4d_run.py --stages 06,07`）。
- 官方 Docker 镜像基于 `pytorch 2.5.1 + cu124`、`TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"`，本机是 Blackwell sm_120 → 不用 Docker，
  直接用系统 python3（torch 2.9.1+cu128）跑它的 stage 脚本，Blackwell 变通见 §3。

## 1. 流水线 8 步：输入 / 输出 / 我们能替换什么

编排器 `lib/run_inference.py`（Docker 内 `python -m v2d.cari4d.lib.run_inference`）依次 subprocess 调用 `lib/cari4d/` 下的脚本，
每步有 `.stages/<name>.json` 标记（输入签名 + 输出 identity）。我们的宿主机版编排器：`src/cari4d_run.py`（同一批脚本、同一批参数）。

| 步 | 脚本 | 输入 | 输出 | 我们的替代品 |
|---|---|---|---|---|
| 01 depth | `prep/mhr_wild_depth.py` | `<seq>.0.color.mp4`（全部帧） | `depth/moge2/raw.npy` uint16 mm [T,H,W]；`intrinsics.pkl`（**MoGe-2 第一帧估的 K**，policy `first_frame_export_per_frame_depth`）；`depth_report.json` | MoGe-2 模型同一版本（`Ruicheng/moge-2-vitl-normal` pinned rev b135031，和本机 cache 的 blob sha256 相同 → 建符号链接的 snapshot，不重下载）。**K 用我们的** `work/camera/intrinsics.json`（在 01 之后改写 pkl 的 fx/fy/cx/cy，MoGe 估计值保留在 `moge_estimate` 字段），`--moge-k` 可回到官方行为 |
| 02 export | `prep/prepare_mhr_wild_export.py` | 视频 + masks H5（键 `<seq>/<f>-k0.person_mask.png` / `obj_rend_mask.png`，**每一帧都要有**）+ 物体 mesh + 01 的 intrinsics.pkl | `export/<seq>/images/*.h5`（JPEG）、`human_masks/`、`object_masks/`、`edex`（K，外参=单位阵 → 世界系 = 相机系）、`object_mesh/output_aligned.glb`（**mesh 做了 oriented-bounds 对齐 + 居中**，变换存在 `wild_export.json: source_object_mesh_to_aligned_transform`） | 视频：从我们抽好的 `work/frames` JPG 重编码（ffmpeg crf 10），**裁到有 mask 的帧段 [first_scored−30, len−1]**（我们的 mask 不覆盖前 30~90 帧），帧偏移记在 `inputs/<seq>/meta.json`；masks：`work/masks/*.npz` 解包写 H5（通道 0 物体、1 人）；mesh：我们的 SAM 3D Objects 每物体 mesh `work/meshes_v2/<obj>.glb` × 三集共识尺寸（README 说的 "mesh autoscaling" 仓库里**没有**这个工具），再 quadric 抽稀到 3 万面（顶点色最近邻带过去） |
| 03 human init | `prep/run_sam3d_mhr_export.py --human-prompt-mode sam2_bbox_mask --no-align-to-gt-depth` | export 的 RGB + 人 mask + K；SAM 3D Body ckpt + `mhr_model.pt` | `mhr_init/<seq>.pkl`：`mhr_global_rot6d[T,6]`, `mhr_trans[T,3]`, `mhr_body_pose_cont[T,260]`, `mhr_hand[T,108]`, `mhr_shape[T,45]`, `mhr_scale[T,28]`(PCA), `mhr_face[T,72]`, `mhr_joints`, `mhr_keypoints` + 直出缓存 `<seq>.sam3d.h5` | 权重：`work/models/sam3dbody/model.ckpt`（Blendcap 非门禁镜像，和官方 `facebook/sam-3d-body-dinov3` 同文件）+ `work/models/mhr_model.pt`（Apache 版）符号链接进 `weights/sam3d_body/checkpoints/sam-3d-body-dinov3/`；DINOv3/DINOv2 hub 代码用 `~/.cache/torch/hub` 已有的。**注意**：官方 wild 推理用 `--no-align-to-gt-depth`，即人体深度 = SAM3D `pred_cam_t`，没有深度 ICP（深度 ICP 只在 BEHAVE 训练数据导出时用） |
| 04 depth align | `prep/align_depth_to_mhr_wild.py` | 01 的 raw 深度 + 03 的人体 | `depth/moge2/aligned.h5`：每帧一个标量 scale（`compute_scale_robust_log`：人 mask 内 MoGe 深度 vs nvdiffrast 渲染的 MHR 深度的 Huber log-ratio），`aligned = raw × scale` | 无替换（我们没有这一步的等价物；它是"用人体给 MoGe 定标"，和我们 `objdepth` 的"物/人深度比"同源） |
| 05 object pose | `prep/run_foundationpose_mhr_export.py --register-first-then-track --no-first-usable-frame-gt-rotation-oracle` | export + 04 的对齐深度 + 物体 mesh；FoundationPose 权重（NGC/gdown：`2024-01-11-20-02-45` scorer、`2023-10-28-18-33-37` refiner） | `foundationpose/<seq>.pkl`：`obj_pose_world[T,4,4]`（aligned mesh 系 → 相机系）、可靠性/可见性诊断 | 权重用 NAS `/mnt/nas3/simfoundry-deps/FoundationPose/weights/`（同名 run）。可替换项：无（我们的 `objtrack` 轮廓跟踪是另一套） |
| 06 CoCoNet | `tools/run_mhr_wild_inference.py --stride 96 --offline-supervision-contract` | 03/04/05 + ckpt `cari4d/2026-08-25-09-35-57/step200000.pth` + `resolved_config.yaml` | `inference/coconet.pth` bundle：`pr`（精修后的 7 组 MHR 参数 + `pose_abs[T,4,4]` + `contact_logits[T,2]`）、`pr_initial`、`in`（初值）、`observations`（256² 的 mask crop + ROI K） | **缺权重 → 跳过**；`src/cari4d_init_bundle.py` 用 03+05 拼 bundle |
| 07 refine | `python -m learning.training.mhr_opt_refineout --mode smplh_parity --num-steps 300 --batch-size 0 --w-temporal 100 --w-human-pose-prior 200 --contact-activation-distance-m 0.05` | 06 的 bundle + `output_aligned.glb` | `inference/refined.pth`（同 bundle 结构，`pr.pose_abs` 与 `pr.mhr_body_pose_cont` 被优化；其余 MHR 块、物体旋转**固定**） | 无替换。损失：接触 200（手表面顶点↔物体三角面距离²，只在"网络说接触 且 初始距离<5 cm"的手）、轮廓 0.002、穿插 2.0（物体表面点在人体 4000 顶点代理网格内的深度，**从第 60% 步起**）、人体姿态先验 200（smooth-L1，β 0.05）、时序 100（关节 + 物体表面二阶差分）、物体平移先验 100；Adam lr 1e-3 + cosine |
| 08 render | `tools/render_mhr_wild_inference.py` | 06/07 bundle | 两个 mp4 | 只在 ep14 上跑作目检（`--render`） |

MHR 参数约定（`lib_mhr/mhr_layer.py`）：`vertices = MHRHead.mhr_forward(trans=0, rot=euler_ZYX(rot6d), body=cont→133, hand, scale(PCA 28), shape, face) * diag(1,−1,−1) + mhr_trans`，
MHRHead 内部 `full_pose = [trans*10, euler, body130]`，输出 /100 → **米**；`mhr_trans` 是以根关节 1 为轴的"decode-consistent"平移（`camera_conventions.mhr_translation_between_frames`）。
→ 导出到我们 136 维 pose 时要做：rot6d→euler ZYX（3）；cont260→133 的 body（取前 130）；手 108 放回 pose 的手部槽位（`replace_hands_in_pose`）；
平移要从"根关节轴"约定反解成 `global_trans*10`；scales = `scale_mean + scale_params @ scale_comps`（68）。**用回环验证**（他们的 layer 顶点 vs 我们 `src/mhr_local.py` 的顶点）。

## 2. 环境（系统 python3，无 venv）

已有：torch 2.9.1+cu128、pytorch3d 0.7.8（sm_120）、moge 3.0.0、open3d、trimesh、h5py、roma、timm、transformers 4.57.3。
新装（`pip install --user`，约束 `work/v2d_official/pip-constraints.txt` 钉死 torch/numpy/opencv）：
av 17.1、pyrender、pyglet、transformations、ruamel.yaml、warp-lang 1.18、hdf5plugin、pytransform3d、imgaug、kornia 0.8.2、videoio、gdown、shapely、scikit-image；
nvdiffrast 0.4.0（源码装，`CUDA_HOME=/usr/local/cuda-12.8`，CUDA 算子在首次使用时 JIT 编译，`TORCH_CUDA_ARCH_LIST=12.0`）。

不需要装的：detectron2（只在 SAM3D 可视化）、tensorrt（只在 FoundationPose 的 TRT 后端；我们走 `nvlabs_pytorch`）、
FoundationPose `mycpp`（需 Boost；但 CARI4D 自带的 `lib/cari4d/estimater.py` 用纯 python `cluster_poses`，PYTHONPATH 里它排在 FoundationPose 前面）、
`bundlesdf/mycuda`（只给八叉树光线追踪，register/track 不用）。

## 3. Blackwell（sm_120）变通

1. **kaolin**：0.18 只有 torch≤2.8 的 wheel，源码编译没必要——CARI4D 只用两个函数（`lib_mhr/sdf.py` 的穿插项）：
   `kaolin.metrics.trianglemesh.point_to_mesh_distance` → 用 pytorch3d `point_face_distance` 实现（对点可微，损失只需要这个）；
   `kaolin.ops.mesh.check_sign` → 广义绕数（solid angle 求和，|w|>0.5 在内）。shim 在 `work/v2d_official/shims/kaolin/`，PYTHONPATH 首位。
2. **nvdiffrast**：pip 从 git 装，构建时必须 `CUDA_HOME=/usr/local/cuda-12.8`（系统 nvcc 11.5 会报 CUDA 版本不匹配）。
3. **TensorRT 10.7 不支持 Blackwell** → FoundationPose 用 PyTorch 后端（权重 `FOUNDATIONPOSE_WEIGHTS_DIR`），这正是 CARI4D 自己的默认（`predict_score.py` 读该环境变量）。
4. **warp-lang**（深度腐蚀/双边滤波的 kernel）1.18 直接支持 sm_120，无需处理。

## 4. 运行方式

```
python3 src/cari4d_prep.py --eps 14                   # inputs/ep000014/{ep000014.0.color.mp4, ep000014_masks_k0.h5, object_metric.glb, meta.json}
python3 src/cari4d_run.py --eps 14 --gpu 2 --stages 01,02,03,04,05,06,07 [--render]
python3 src/cari4d_export.py --eps 14                 # → work/sub_cari4d/episode_000014.npz + _object.obj
```
状态标记 `state/cari4d/<stage>/epNNNNNN.done`，日志 `logs/cari4d/epNNNNNN_<stage>.log`，产物 `work/v2d_official/runs/epNNNNNN/`。

## 5. 结果（随进展更新）

### 5.1 ep14（382 帧，GPU2）——全链路首跑

| 步 | 耗时 | 备注 |
|---|---|---|
| 01 MoGe-2 | 39 s | bs 8, fp16；MoGe 估的 fx 812.6 vs 我们 821.5（差 1%）|
| 02 export | 31 s | |
| 03 SAM3D | 49 s | 直出缓存 + 无深度对齐；f90 的 cam_t 与我们 `work/sam3d` 的差 4 mm |
| 04 depth align | 268 s | 含 nvdiffrast JIT；MoGe→人体 scale 中位 1.08（1.05–1.11）|
| 05 FoundationPose | 49 s | register 1 + track 381，IoU 中位 0.67，无 fallback |
| 06 init bundle（替代 CoCoNet）| 17 s | 几何接触门限只激活 10 帧（左手）|
| 07 refinement | 390 s | 300 步全片；关节位移均值 4 mm、最大 6.5 cm；脚 2 mm |
| 08 render | ~60 s | `runs/ep000014/inference/ep000014_before_after.mp4`，人体贴合（frame300.jpg）|

导出（`src/cari4d_export.py`）：回环 1e-6 m；根关节投影 (843,665) vs 人 mask 质心 (842,669)。
代理（只此一集）：ACC-H **0.412**（sub_track2 0.072；SAM3D 逐帧抖动，07 的时序项 w=100 压不住）、重投影 4.9 px（5.4）；
物体：z 中位 3.90 m（sub_v6 2.55），到手腕中位 **67 cm**（32），|a| 0.64（0.07，门槛 0.3）。
**物体深度 = 对齐后的 MoGe 深度**（FoundationPose 用 depth 初始化平移并逐帧 track），而 MoGe 在锅上的深度相对手腕漂 −0.3…+1.2 m。
官方链路里纠正这个的是 CoCoNet（它预测物体平移增量）；没有它，这条链的物体不会比我们的 MoGe 比值方案好。

### 5.2 30 集（2026-10-18 13:40 → 18:05，GPU2/3）

**跑了什么**：01–05、07 全部 30/30 原样跑通（官方脚本 + 官方参数）；06 CoCoNet 缺权重 → 30/30 用初始化 bundle 代替（§0）。
**替换**：相机 K（我们的）、视频（裁到有 mask 的段、从 JPG 重编码）、masks（我们的 SAM2）、SAM3D/MHR 权重（镜像）、物体 mesh（我们的 SAM 3D Objects × 共识尺寸，抽稀 3 万面）、CoCoNet（几何接触门限 bundle）。
**本地补丁**：`prep/run_foundationpose_mhr_export.py` 一行（track 模式物体全遮挡帧 `fp_centroid_distance=inf` 被输出校验拒绝 → nan；9 集因此失败后重跑通过）。

| 步 | 30 集中位 | min–max | 合计 GPU 时 |
|---|---|---|---|
| 01 MoGe-2 | 40 s | 27–65 | 0.35 h |
| 02 export | 27 s | 17–47 | 0.23 h |
| 03 SAM3D | 38 s | 30–105 | 0.38 h |
| 04 深度对齐 | 336 s | 199–584 | 2.89 h |
| 05 FoundationPose | 60 s | 36–267 | 0.73 h |
| 06 init bundle | 17 s | 12–26 | 0.14 h |
| 07 精修（300 步全片）| 819 s | 363–1954 | 7.30 h |

导出：`work/sub_cari4d/`（原样）与 `work/sub_cari4d_s3/`（+ 我们的 wrap-safe σ=3 姿态平滑、σ=6 物体平移平滑，**非官方步骤**）。
30 集回环最大 1e-6 m；用每集中位 identity 代替逐帧 shape/scale 的顶点误差中位 6 mm；根关节投影到人 mask 质心中位 13 px（最大 55 px，ep1）。
接触门限（<5 cm）在 30 集共激活 左 1812 / 右 2622 帧-手（9877 计分帧）。

**门控**（`src/acc_gate.py` / `body_proxy.py` vs sub_track2 / `obj_hand_dist.py` vs sub_v6 / `pen_gate.py`）：

| | sub_cari4d（原样）| sub_cari4d_s3（+平滑）| 参照 |
|---|---|---|---|
| ACC-H 代理 mean / max | **1.207 / 2.736** | 0.089 / 0.189 | sub_track2 0.067 / 0.177 |
| 重投影 px mean / max | 6.9 / 12.5 | 6.9 / 11.6 | 7.3 / 10.3 |
| ACC-O 代理 mean / max（门槛 <0.3）| **1.906 / 10.25 FAIL** | 0.093 / 0.273 PASS | — |
| 物体-手腕距离中位（30 集中位）| 59 cm | 59 cm | sub_v6 39 cm（cari4d 只在 11/30 集更近）|
| 物体 z 中位 | 3.16 m | 3.16 m | sub_v6 2.69 m |
| PEN（手在物体内）| **worst 1.34 cm，2023/9877 帧 FAIL** | 1.42 cm，1977/9877 FAIL | 我们的导出 0 |

解读：
- 人体：逐帧 SAM3D 初始化 + 07 精修的姿态**本身抖动很大**（ACC-H 1.2，官方靠 CoCoNet 的时序注意力平滑）；加上我们的平滑后 ACC-H 0.089、重投影 6.9 px（比 sub_track2 的 7.3 略好）——人体这条线"官方初始化 ≈ 我们的 raw SAM3D（#7，CD-H 17.86）"。
- 物体：官方链路的物体深度 = 用人体定标后的 MoGe 深度（FoundationPose 由 depth 初始化并逐帧 track），离手腕中位 59 cm，比 sub_v6（39 cm）更远；没有 CoCoNet 的物体平移修正，CD-O 预计不会好于 sub_v6（41.7）。
- PEN：官方 07 的穿插项是"物体表面点在人体 4000 顶点代理网格内"（权重 2、只在后 40% 步），并不约束"手顶点在物体内"（赛题 PEN 的定义），所以 20% 计分帧有 ≤1.4 cm 的手-物穿插。这条链不能用我们的推离重导出（流程不同）；若要提交需另写一个针对 refined.pth 的推离后处理。

**结论**：没有 `nvidia/cari4d_commercial` 权重，这条复现只到"CARI4D − CoCoNet"，不构成榜上 baseline（15.58 / 23.16）的复现；拿到 HF token 后 `src/cari4d_run.py --stages 06,07 --force` 即可补齐（06 需要 03/04/05 的产物，已保留；04 的 aligned.h5 已被剪，需 `--stages 04,06,07`，04 约 6 min/集）。



## 6. With CoCoNet（真正的官方复现，2026-10-18 22:30 → 2026-10-19 03:50）

HF token（Tony332）解锁 `nvidia/cari4d_commercial`：step200000.pth（sha256 与官方 pin 一致）+ resolved_config + manifest；DINOv2 vitb14/vits14 按官方 pin sha 下载。
**第二遍流程**（每集）：01 重跑（raw 深度已剪）→ `src/cari4d_regen_export.py` 重建 export RGB/mask h5（不动 output_aligned.glb / edex，FoundationPose 位姿系不变）→ 04 → **官方 06 CoCoNet**（stride 96，offline supervision contract）→ 07（同 §1 参数）。03/05 沿用第一遍。
调度：先 3 卡固定列表，后改 `src/run_cari4d_cc2.sh` 锁队列（每卡 2 worker）；GPU0 未用，GPU3 在 Track-3 占用后停用。06 中位 87 s，07 中位 891 s。
**第二个本地补丁**（`tools/run_mhr_wild_inference.py`，原件 `.orig`）：ep2 呼啦圈被 FoundationPose 跟丢到画面外 27 帧（原始帧 402–476，含 5 个计分帧），官方几何裁剪抛 "mesh geometry projects entirely outside the image"；补丁让这些帧走裁剪器自带的 `object_pose_valid=False` 分支。

CoCoNet 的作用（ep14）：物体平移相对 FoundationPose 改动中位 0.86 m（z 3.90→3.03），接触 logits 左 222 / 右 125 帧；人体平移只改 1 cm。
30 集物体-手腕距离（每集中位的中位）：**31.5 cm**（无 CoCoNet 59 cm，sub_v6 39 cm），22/30 集比 sub_v6 更近；物体 z 中位 2.98 m（sub_v6 2.69）。

| 变体 | ACC-H 代理 mean/max | px mean/max | ACC-O 代理 mean/max（<0.3）| PEN（帧/9877，最深）| 物体-手腕 |
|---|---|---|---|---|---|
| `sub_cari4d_cc`（官方原样）| 1.183 / 2.709 | 8.4 / 11.8 | 6.28 / 111.9 FAIL | 4387，1.63 cm FAIL | 31.5 cm |
| `sub_cari4d_cc_s3`（+σ3 平滑）| 0.090 / 0.205 | 8.4 / 11.8 | 0.086 / 0.371 FAIL（ep2）| 4281，1.64 cm FAIL | 31.5 cm |
| `sub_cari4d_cc_pen`（原样+推离）| 1.183 / 2.709 | 8.4 / 11.8 | 6.31 / 112.1 FAIL | **0 PASS** | 37 cm |
| `sub_cari4d_cc_s3_pen`（平滑+推离）| 0.090 / 0.205 | 8.4 / 11.8 | 0.140 / 0.647 FAIL（ep2 0.65、ep15 0.52）| **0 PASS** | 35 cm |
| 参照 sub_track2 / sub_v6 | 0.067 / 0.177 | 7.3 / 10.3 | — | 0 | 39 cm |

- ep2 是 ACC-O 的最大来源：计分段 [401,418] 里官方输出把物体放到相机后方（z 最低 −1.96 m，即 CoCoNet 对跟丢帧的输出），原样 ACC-O 112；σ3 平滑后仍 0.37。这是官方流水线在细环物体上的失败，不是导出问题。
- 推离（`src/cari4d_pushout.py`，手→质心两遍包络，只改最终平移）：每集最大平滑推离中位 17–20 cm、最大 0.74–0.83 m，残余逐帧推离 ≤6 帧；PEN 全清零，代价是 ACC-O 代理 0.086→0.140 且物体离手远 4–6 cm。
- 人体：CoCoNet 对人体姿态几乎不平滑（ACC-H 1.18，与无 CoCoNet 的 1.21 相同），重投影反而变差（6.9→8.4 px）。
- 没有一个变体同时过 ACC-O 和 PEN；如要提交，`_cc_s3_pen` 最接近（只差 ep2/ep15 的 ACC-O），可单独处理 ep2 的跟丢段后再门控。未提交。
