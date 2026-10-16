# V2D Challenge 环境与凭证配置结论

> 面向在同一台机器上做 **Track 2 / Track 3** 的其他 agent。
> 所有条目都是在 `/home/tianyuan/projects/v2d_track1` 实测通过的，不是推测。
> 更新时间：2026-10-16。

---

## 1. GitHub（提交必需）

每一行提交都带 `code_commit_url`，组织者要能打开那个 commit 看代码。**URL 必须以 `/commit/<sha>` 结尾，且同一次提交内保持一致。**

**不要用 `gh auth login`。** 这台是无头机：`DISPLAY` 未设，`gh`（2.4.0，2022 年的 Ubuntu 包）会走
`xdg-open → sensible-browser → snap firefox`，snap 首次运行解包失败，报的就是
`gzip: stdin: not in gzip format`，而且永远不会弹出网页。

**用 SSH key：**
```bash
ssh-keygen -t ed25519 -C "<你的邮箱>" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub     # 贴到 https://github.com/settings/ssh/new
ssh -T git@github.com         # 看到 "Hi <用户名>!" 即成功（后面那句 shell access 提示是正常的）
```
**密钥已经生成好了**（`~/.ssh/id_ed25519`，无口令，已加到 GitHub）→ **其他 track 直接复用，不要重新生成，
重新生成会覆盖掉现有的那把、把已经配好的连接弄断。**

当前身份：`git config user.email tianny007@gmail.com` / `user.name tianny0078`
Track 1 的仓库：`git@github.com:tianny0078/v2d-track1.git`
私有仓库要把 [`Nvidia-V2D-Challenge`](https://github.com/Nvidia-V2D-Challenge) 加为协作者，否则提交作废。

---

## 2. Kaggle（提交必需）

Token 已配在 `~/.kaggle/access_token`（**不是** `kaggle.json`，kaggle 2.x 用这个路径），
**全机器共用，其他 track 不需要再配。**

### ⚠️ 必须用 Python 3.11+
系统只有 **Python 3.10**，而 kit 的 `requirements.txt` 要 `kaggle>=2.0`，它需要 3.11+，在 3.10 上
`pip install` 直接报 `No matching distribution found`。

已经建好一个可直接复用的环境：
```
/home/tianyuan/projects/v2d_track1/work/venv-submit/bin/python   # Python 3.12.15 + kaggle 2.2.4
```
（用 `uv` 建的；`uv` 已装在 `~/.local/bin/uv`。）

### 提交上限：**每个竞赛每天 5 次**
`competition_get_submission_limits` 实测返回 `numAllowedNow: 5`。
Track 1 是 5 个竞赛 → 每天 25 次提交机会。Track 3 有 6 个竞赛。
**每个竞赛的规则都要在网页上单独点接受**，API 代替不了；漏一个，那个指标就不评分。

### 验证连通性
```bash
work/venv-submit/bin/python -c "
from kaggle.api.kaggle_api_extended import KaggleApi
a=KaggleApi(); a.authenticate()
print(a.competition_get_submission_limits('<competition-slug>'))"
```

---

## 3. Hugging Face

**`facebook/sam-3d-body-dinov3` 和 `facebook/sam-3d-objects` 都是 gated repo**，本机未登录，权限申请中。

**未受限、已下载好、可直接复用**（在 `work/models/`）：
- `facebook/sam2.1-hiera-large` — SAM 2.1，分割 + 视频传播
- `google/owlv2-large-patch14-ensemble` — 开放词表检测（文本→框）
- MHR 模型：`https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip`
  里的 `assets/mhr_model.pt`（Apache-2.0，**不要**用 SAM 3D Body 里那份，许可不同）
  sha256 `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc`

---

## 4. ⭐ /mnt/nas3 上已有的资源（别重复下载）

- **`/mnt/nas3/v2d`** — 官方 `nvidia-isaac/video_to_data` 完整克隆，36 个重建模块：
  `v2d_sam3d_body`（`lib/estimate_mhr_params.py` 接受 RGB + mask 提示直接出 `mhr_params.pt`）、
  `v2d_sam3d`、`v2d_foundation_pose`、`v2d_hamer`、`v2d_wilor`、`v2d_moge`、`v2d_unidepth`、
  `v2d_depth_anything`、`v2d_grounding_dino`、`v2d_sam2`、`v2d_hoi_object_reconstruction`、
  `v2d_mesh`、`v2d_anycalib`、`v2d_geocalib`、`v2d_droid_slam`、`v2d_bundlesdf` …
  ⚠️ **SAM 3D Body 的权重不在 NAS 上**（全盘搜过），只有代码。
- **`/mnt/nas3/simfoundry-deps/sam-3d-objects`** — **13GB 完整 SAM 3D Objects 权重**（slat/ss 全套）+ 代码。

---

## 5. ⚠️ Blackwell (sm_120) 的坑

4× RTX PRO 6000 Blackwell 96GB。**GPU 是共享的，跑之前查 `nvidia-smi`，别假设独占。**

- **`/usr/bin/nvcc` 是 CUDA 11.5，和 torch 的 cu128 不匹配**，任何编译 CUDA 扩展的包都会报
  `The detected CUDA version (11.5) mismatches the version that was used to compile PyTorch (12.8)`。
- **解法：`/usr/local/cuda-12.8` 是存在的。**
  ```bash
  CUDA_HOME=/usr/local/cuda-12.8 PATH="/usr/local/cuda-12.8/bin:$PATH" \
  TORCH_CUDA_ARCH_LIST="12.0" pip3 install --no-build-isolation <pkg>
  ```
  （`--no-build-isolation` 也必需，否则构建环境里看不到 torch。）
- **SAM 3D Objects 官方环境是 CUDA 12.1 + kaolin 0.17 + gsplat，在 sm_120 上装不了。**
  但查过实际调用图：**mesh 路径对 kaolin 的全部依赖只有
  `from kaolin.utils.testing import check_tensor`（纯 Python 断言），gsplat 只在高斯渲染用。**
  → 只要 mesh 就能 shim 掉 kaolin、完全跳过 gsplat，不碰那套工具链。
- MHR TorchScript 在 Blackwell 上原生可用：0.51 ms/帧，峰值 1.07 GB。
  注意 `face_expr_coeffs` 维度是 **72**（官方文档没写），零填充即可。

---

## 6. 对 Track 3 可能同样成立的两条赛制事实

这两条是我在 Track 1 的 kit 里挖出来的，Track 3 请**自行用自己的 config 核对**，别直接照搬：

1. **`config/tracks.json` 里 Track 1 的 `private_fraction` 是 0** —— 榜单全公开，没有 private 洗牌，
   **榜单分数就是最终分数**。如果 Track 3 也是 0，那么策略应该是"尽早频繁提交，榜单即验证集"，
   而不是"省着用提交次数"。
2. **五个指标的 host 评分代码完整随 kit 发布**（`metric_code/track_N/*.py`），连 MHR 前向都有一份
   NumPy 重实现。**读评分代码本身**比读评分说明有用得多 —— Track 1 的 PEN 就是这样被发现是
   "自指"的（拿参赛者自己的手点打参赛者自己提交的网格，真值只通过一个标量进入），因而可以本地
   精确优化到 0。实测基线提交 PEN = 0.00000，验证了这个判断。

---

## 7. Track 1 首次提交实测分数（供对照量级）

commit `9300a24`，内容是 MHR rest pose + 原点单位盒的平凡基线：

| CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|
| 70.64959 cm | 145.40682 cm | 0.14528 | 0.07429 | 0.00000 |

因为预测的加速度恒为 0，ACC 读数直接等于**真值自身的加速度幅值**：
关节 ≈ 0.145 cm/frame²，物体平移 ≈ 0.074 cm/frame²。真值运动很平缓 →
**在 ACC 类指标上保守平滑是安全的，激进追求运动细节反而危险。**
