# Track 1 进度日志

追加式，最新在最下。每条：`时间 · 阶段 · 做了什么 · 结果/数字 · 下一步`。
机器可读状态在 `web/status.json`（由 `src/status.py` 从真实文件系统生成，不手写）。

## 2026-10-15

- `23:39` **环境** · 盘点硬件 · 4× RTX PRO 6000 Blackwell 96GB（共 384GB），128 核 / 995GB RAM，
  磁盘 `/` 剩 293GB（92% 满）。Python 3.10（kit 要 3.11+，记为阻塞项 B-1）。torch 2.9.1+cu128 四卡可见。
- `23:40` **数据** · `hf download nvidia/video_to_data_challenge --include "track_1/*"` ·
  **490MB / 67 文件 / 13 秒**。30 episode、16,563 帧、1536×1152 H.264 30fps、10 物体×3 段、4 机位。
  `data/` 确认只有 LeRobot 索引列，无任何真值。
- `23:40` **数据** · 下载并解包官方 `v2d_submission_kit.zip`（698KB）· 含五个指标的**完整 host 评分代码**
  （`metric_code/track_1/{CD-H,CD-O,ACC-H,ACC-O,PEN}.py`），以及 MHR 前向的 NumPy 重实现。
  缺 asset npz（`role_*` 顶点索引）与参考真值。
- `23:44` **格式** · 解码 `track_1_sample_submission.parquet`（740,780 行）· 8 个 array、块号 0–7 全部对上；
  网格预算**固定 4096 顶点 + 4096 面**，允许补零面。
- `23:45` **格式** · ⭐ 解析评分帧 · **9,877 / 16,563 帧被评分（59.6%）= 手物接触区间**。
  21 个 episode 单一连续段，9 个多段（最多 7 段）。首评分帧多为 frame 90。
  → 接触是全程主约束；ACC 不跨段计算 → gap 是免费的轨迹重启窗口。
- `23:47` **格式** · ⭐ 读完 `_episode_penetration` · **PEN 是自指的**：用我们自己的手点打我们自己的网格，
  真值仅通过首帧 Sim(3) 的 scale 标量进入。→ **PEN 可本地精确优化到 ~0**，五榜之一近乎免费。
- `23:47` **规则** · `tracks.json` 注释：`private_fraction is 0` · **榜单全公开，没有 private 洗牌** →
  提交策略由"省着用"改为"尽早频繁提交，榜单即验证集"。
- `23:48` **Gate A** · MHR TorchScript 跑通（Blackwell）· sha256 与官方一致；
  `forward(identity[B,45], model_params[B,204], face_expr[B,72])` → `verts[B,18439,3] / joints[B,127,8]`；
  **0.51 ms/帧，峰值 1.07GB**；`diag(1,-1,-1)·v/100` 后身高 **1.727m** ✅ 单位与坐标约定核对通过。
  → **MHR 接口风险拆除**（v1 的头号风险）。
- `23:50` **计划** · 写出 `notes/00-plan-v2.md`，相对 v1 共 10 处实质修订（见该文件 §8）。
- `23:50` **基建** · `src/status.py` + `web/status.json` + 30 张接触区间中点缩略图。

**下一步**：抽帧缓存（只缓存评分窗口+前导）→ 分割 → SAM 3D Body → 零基线提交打通 packer。

## 2026-10-16（凌晨）

- `00:05` **Gate B 通过 · 输出契约** · 造 30 个基线 NPZ（MHR rest pose + 单位盒）跑官方
  `eval_reconstruction.py` → **740,780 行，3.5MB，4.3 秒**，`row_id` 与 sample **逐行完全一致**，
  全部有限值。→ packer / 输出格式风险拆除。
  副产物：packer 会对网格质量告警（"very low detail / box-like primitive"），
  说明**组织者会审查网格**，最终提交不能用几何占位符。
- `00:12` **工具 · MHR 角色顶点** · 从 TorchScript 里直接取出骨架与蒙皮
  （`joint_parents[127]`、`skin_{indices,weights}_flattened[51337]`、`faces[36874,3]`、`rest_vertices[18439,3]`）。
  按"主导关节属于腕关节子树"推出手部顶点：**每只手 2318 个，bbox 14×12×15 cm** ✅ 形状正确。
  这是 host 那 512 个手点的**超集** —— 超集上无穿透 ⟹ 子集上无穿透，正是 PEN 优化需要的性质。
  `role_body`（64 点）与 `role_alignment` 用 FPS 代理，**明确标注为消融信号而非榜单预测**。
- `00:20` **Gate C 通过 · 本地精确 PEN** · `src/score_local.py` 复用 host 自己的
  `penetration_depth / symmetric_chamfer_distance / acceleration_error_per_frame / stretches`。
  敏感性验证（盒子逐步移离右手）：
  `0.00m → 5.30cm · 0.05m → 4.55cm · 0.10m → 2.67cm · 0.20m → 0.012cm · 0.40m → 0.000cm`
  单调且量纲正确。→ **PEN 现在是一个可以本地直接优化的目标函数，不需要任何榜单反馈。**
- `00:22` **数据** · 启动抽帧（只从 `first_scored - 30` 开始，JPEG q2，文件名即真实视频帧号）。
- `00:45` **凭证阻塞** · `facebook/sam-3d-body-dinov3` 与 `facebook/sam-3d-objects` 都是 **gated repo**，
  本机无 HF token / Kaggle token / gh 登录 / git identity。→ 阻塞项 B-1~B-3 升级，需要人工介入。
  未被阻塞的替代链路已就位：**OWLv2-large（文本→框）+ SAM 2.1-hiera-large（框→掩码+视频传播）**，两者均未受限。
- `00:50` **分割链路跑通** · ep0：seed=首个评分帧 90，前后双向传播 **730/730 帧无丢失**；
  物体检测分 0.638、人 0.159（偏低，已目视核验）。掩码 bit-packed 存储，**1.4MB/episode**。
  目视核验（`web/assets/seg_ep000000.jpg`）：人与呼啦圈都正确，且**正确避开了桌边的旁观者**。
  新观察：① 相机视角很宽，人在画面里只占约 1/6 高度 → SAM 3D Body 必须**先裁剪人体再推理**；
  ② 场景里有第二个人（操作员）坐在桌边，是持续的干扰源；③ 地面平整 → **地面接触可作为度量尺度的强先验**。
- `00:52` **全量分割启动**（30 段，后台）。
