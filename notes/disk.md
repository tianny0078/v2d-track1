# 磁盘盘点与清理（2026-10-17）

机器：`/` 3.6 TB（nvme0n1p2），盘点时 **已用 3.3 TB / 可用 163 GB（96%）**。无 sudo，所以 `du` 只能看到本用户可读的部分：
`/home` 1.7 TB + `/tmp` 51 GB + `/usr` 31 GB + `/var` 12 GB ≈ 1.8 TB；剩下 ~1.5 TB 在不可读目录里，
其中 **docker（/var/lib/docker）自报 257 GB 镜像 + 334 GB build cache ≈ 590 GB**，是单个最大消费者。

口径：`du -xsh`（同文件系统、实际占用）。verdict：keep / delete-now / delete-later / not-ours。

## 1. 全盘视角（按 owner）

| 路径 | 大小 | owner | verdict | 说明 |
|---|---|---|---|---|
| docker images（28 个，3 个在用） | 257 GB（221 GB 可回收） | 系统/多人（v2d_* 系列镜像 ≈ Track-3 / 其他人） | **delete-later（owner 决定）** | `docker image prune -a` 会删掉 v2d_sam3d、v2d_wilor、ego_vipe 等 25 个未被容器引用的镜像；需确认是否还有人要用 |
| docker build cache（358 条） | 334 GB（127 GB 可回收） | 系统 | **delete-later（owner 决定）** | `docker builder prune` 安全（只删缓存层），能立刻回收 ~127 GB；无 sudo 但 docker 组可执行 |
| /home/yunze | 1.2 TB | yunze | not-ours | |
| /home/peiran | 252 GB | peiran | not-ours | |
| /home/tianyuan | 129 GB | 我们 | 见下表 | |
| /home/zhuorui | 47 GB | zhuorui | not-ours | |
| /home/enmin | 39 GB | enmin | not-ours | |
| /tmp 合计 | 51 GB | 混合 | 见下 | 7644 个条目 |
| /tmp/tmp*.obj（tianyuan） | 7.8 GB（743 个） | 我们 | **delete-now** | 7.8–57 MB/个，10-16/10-17 —— `src/export_episodes.py` / `object_track.py` 的 `NamedTemporaryFile(delete=False)` 泄漏，无进程打开。已修源码（见 §4） |
| /tmp/track3-hourly-*.tar.gz 等 track3-*（tianyuan 用户，Track-3 agent 写的） | 21.3 GB（991 个，~30 MB/个） | **Track-3 agent** | not-ours → owner 决定 | 每小时一份归档，全部 10-17 当天；同一 Unix 用户但不是我们的项目，脚本不碰。建议 Track-3 agent 自己加保留策略（例如只留最近 24 份） |
| /tmp/claude-1017 | 919 MB | 我们 | 部分 delete-now | 会话 `27036691…`（09-15，129 MB）过期；其余 bash-edit-diff 均为今日，保留 |
| /tmp/wp_pch_5xwgol0l | 274 MB | 我们 | delete-later | NVIDIA warp 预编译头缓存（10-16），可能被 sam3d-objects/kaolin 复用，不值得动 |
| /tmp/egoomni_media | 8.9 GB | peiran | not-ours | |
| /tmp/g1bench | 4.1 GB | dylanlin | not-ours | |
| /tmp/p10proxy、vh_*.json、ptcheck、tmpg88xi0ae、mai-model-review-pnpm | ~2.4 GB | dylanlin | not-ours | |
| /tmp/mimo-browser-binaries | 658 MB | peiran | not-ours | |
| /tmp/intact-r6000-wheelhouse-* | 244 MB | dongxu | not-ours | |
| /var/tmp | 8 MB | 系统 | keep | |
| /swapfile | 2.1 GB | 系统 | keep | |
| /usr | 31 GB | 系统 | keep | |

## 2. /home/tianyuan（129 GB）

| 路径 | 大小 | verdict | 说明 |
|---|---|---|---|
| projects/v2d | 83 GB | **not-ours**（Track-3 agent） | 有 watchdog / continuation / hourly agent 三个 python 进程在跑 |
| projects/v2d_track1 | 26 GB | 见 §3 | |
| .cache/pip | 4.4 GB（1690 个 http 文件） | **delete-now** | `pip cache purge` |
| .cache/huggingface/hub | 3.3 GB | keep（大部分） | moge-2-vitl-normal 1.3 GB、sam2.1-hiera-large 857 MB 为 pipeline 所用；其它 repo 只有 refs（权重在 work/models） |
| .cache/huggingface/hub/models--Ruicheng--moge-vitl | 1.2 GB | **delete-later** | MoGe **v1**，`src/` 里只引用 `moge-2-vitl-normal`；但 HF cache 与 Track-3 agent 共用，由 owner 确认后 `rm -rf` |
| .cache/huggingface/**/*.incomplete | 0 | — | 没有失败下载残留 |
| .cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth | 1.2 GB | keep | SAM 3D Body / DINOv2 backbone |
| .cache/torch_extensions | 不存在 | — | |
| .cache/uv | 288 MB | delete-later | `uv cache clean`，Track-3 的 `.deps/v2d-submit-venv` 可能用 uv，先不动 |
| .cache/warp | 97 MB | keep | warp kernel cache |
| .vscode-server | 4.0 GB | delete-later | 3.9 GB 是 `cli/` 下多版本 server；owner 自己清 |
| .local/lib | 3.0 GB | keep | 用户级 site-packages |
| .local/share | 1.3 GB | keep | |
| .codex | 1.0 GB | not-ours（codex CLI 缓存） | |
| .nv/ComputeCache | 321 MB | keep | CUDA JIT 缓存 |
| hot3d_aria_benchmark | 2.4 GB（全部是 `.venv`） | delete-later | 09-15 的旧项目 venv；owner 决定 |
| conda | 不存在 | — | |

## 3. v2d_track1/work（25 GB）

| 路径 | 大小 | verdict | 说明 |
|---|---|---|---|
| models/ | 15 GB | keep | 权重 |
| v2d_latest/ | 3.7 GB | keep | 官方 repo clone，`submit_*.sh`/`object_mesh.py` 引用 |
| frames/ | 3.0 GB | keep | pipeline 输入 |
| sub_place4/5/6/8 | 4 × 485 MB | **delete-now** | 过时实验（含 mesh），不在 KEEP 也不在 submissions.md |
| sub_place7 | 485 MB | keep | KEEP 列表 |
| venv-submit/ | 300 MB | keep | 提交用 venv |
| objtrack/ 224 MB、moge_pts/ 212 MB、flowhmr_tokens/ 163 MB、flowhmr/ 109 MB、meshes_v2/ 79 MB、meshes/ 73 MB、sam3dbody_code/ 47 MB、flowhmr_out/ 42 MB、masks/ 33 MB、kit/ 18 MB、sam3d、kp2d、camera、objdepth、body_* | ~1.1 GB | keep | 中间产物，pipeline 需要 |
| sub_track1 / sub_track2 / sub_v4b / sub_moge2 | 4 × 16 MB | keep | KEEP 列表（submission 候选/已提交） |
| sub_v3、sub_v3b、sub_v4、sub_flow1、sub_flow2、sub_moge1、sub_moge_p20、sub_rot_test、sub_track3、sub_place、sub_place2、sub_place3 | ~100 MB | **delete-now** | 过时实验 |
| sub_ground、sub_sf、sub_smooth、sub_human、sub_sam3d、sub_sam3d_s3、sub_refine | 7 × 9.6 MB | **delete-now（仅目录）** | 已提交版本：`work/<name>.parquet` + `.kaggle-receipt.json` 保留（台账可追溯），导出目录可从 parquet 重建 |
| *.parquet + *.kaggle-receipt.json | ~95 MB | keep | 提交台账证据，脚本永不触碰 |
| stale_ep1/ | 6.3 MB | **delete-now** | |
| __pycache__（184 个） | 21 MB | **delete-now** | |
| logs/ | 4.3 MB | keep | 没有 > 50 MB 的日志 |
| baseline_episodes/、inspect/、shims/、review_*.jpg | ~15 MB | keep | 小 |

## 4. 根因修复

`src/export_episodes.py`（2 处）和 `src/object_track.py`（1 处）用 `NamedTemporaryFile(suffix=".obj", delete=False)`
给 `budget_mesh` 喂文件后从不删除 —— 每次导出泄漏 8–57 MB 到 `/tmp`，两天累积 743 个 / 7.8 GB。已改为 `try/finally: os.unlink`。
`src/disk_cleanup.sh` 第 5 步兜底清理 > 60 min 且未被进程打开的 `/tmp/tmp*.obj`。

## 5. 需要 owner 决定的（按收益排序）

1. `docker builder prune`：**~127 GB**，只删 build cache，零风险。
2. `docker image prune -a`：**~221 GB**，删 25 个无容器引用的镜像（v2d_* 系列、ego_*、ray-inference、pytorch devel 等）——要先问清楚是不是 Track-3 或其他人的工具链。
3. `~/.vscode-server/cli` 旧版本 3.9 GB、`~/hot3d_aria_benchmark/.venv` 2.4 GB、HF `moge-vitl`(v1) 1.2 GB、`~/.cache/uv` 288 MB。
4. 其他用户的 `/tmp`（peiran 9.5 GB、dylanlin ~7 GB）—— 只能他们自己删。

## 6. 清理结果

`src/disk_cleanup.sh` 2026-10-17 23:07 实跑（先 `--dry-run` 审过一遍）：

| 动作 | 释放 |
|---|---|
| `pip cache purge` | 4.32 GB |
| `/tmp/tmp*.obj` 泄漏文件 743 个 | 7.27 GB |
| work/sub_* 过时导出目录 23 个 + stale_ep1 | 2.0 GB（sub_place4/5/6/8 各 0.47 GB） |
| /tmp/claude-1017 过期会话 1 个 | 0.13 GB |
| __pycache__ 31 个 | ~0 |
| **合计（df 实测）** | **13.76 GB：可用 162 GB → 176 GB（96% → 95%）** |

保留核对：work/ 下只剩 sub_moge2 sub_place7 sub_track1 sub_track2 sub_v4b；18 个 `*.parquet` / `*.kaggle-receipt.json` 全部在。
第二次 `--dry-run` 报 0.01 GB → 幂等。周期运行：`src/disk_cleanup.sh`（可加进 cron / refresh_dashboard.sh）。
- 2026-10-18 21:48 定时清理：释放 0.01 GB（sub_size_test、__pycache__）；根分区可用 181 GB (95%)；/tmp 43 GB，其中 Track-3 的 track3-hourly-*.tar.gz 20 GB（到阈值，未删）；sub_v5 已加入 KEEP。
- 2026-10-18 21:53 定时清理：释放 0.01 GB（pip 缓存）；根分区可用 181G；/tmp 43G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）。
- 2026-10-19 03:48 定时清理：释放 0.01 GB（pip 缓存、__pycache__）；根分区可用 170G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）；sub_v6/sub_v7 加入 KEEP。
- 2026-10-19 03:53 定时清理：释放 0.01 GB（pip 缓存）；根分区可用 170G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）。
- 2026-10-18 22:00 PDT 定时清理：释放 0.28 GB（pip 缓存、27 个 __pycache__）；根分区可用 153G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）；sub_cari4d/_s3 加入 KEEP。
- 2026-10-18 22:08 PDT 用户授权后 `docker builder prune -f`：释放 **126.9 GB**，根分区可用 153 → 265 GB。未引用镜像 221 GB 仍待确认归属。
- 2026-10-19 00:48 PDT 定时清理：释放 0.01 GB（pip 缓存、53 个 __pycache__）；根分区可用 231G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）；KEEP 加入 sub_cari4d_cc/_cc_s3/_cc_pen。
- 2026-10-19 00:53 PDT 定时清理：释放 0.01 GB（pip 缓存）；根分区可用 231G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）。
- 2026-10-19 06:48 PDT 定时清理：释放 0.01 GB（pip 缓存、31 个 __pycache__）；根分区可用 220G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）；我们最大的目录：12G	work/behave_root 39G	work/v2d_official 
- 2026-10-19 06:53 PDT 定时清理：释放 0.01 GB（pip 缓存、18 个 __pycache__）；根分区可用 212G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）。
- 2026-10-19 12:48 PDT 定时清理：释放 0.01 GB（pip 缓存、__pycache__）；根分区可用 224G（一小时前 241G，减少的部分不是我们的进程写的）；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 20G（未删）；我们最大的目录：39G work/v2d_official、15G work/models、7.0G work/behave_root。
- 2026-10-19 12:53 PDT 定时清理：释放 0.01 GB（pip 缓存）；根分区可用 224G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 21.2 GB（未删）；我们最大的目录：39G work/v2d_official。
- 2026-10-19 18:48 PDT 定时清理：释放 0.01 GB（pip 缓存）；根分区可用 896G（别人释放了约 650 GB，不是我们）；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 21.2 GB（未删）；Docker 镜像可回收 221 GB（未删）；我们最大的目录：39G work/v2d_official。
- 2026-10-19 18:53 PDT 定时清理：释放 0.01 GB（pip 缓存）；根分区可用 896G；/tmp 44G，其中 Track-3 的 track3-hourly-*.tar.gz 21.2 GB（未删）；Docker 镜像可回收 221 GB（未删）；我们最大的目录：39G work/v2d_official。
