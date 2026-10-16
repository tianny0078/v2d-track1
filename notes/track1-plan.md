> 来源：[V2D Challenge Track 1｜4D重建参赛分析与三路线实验计划](https://docs.google.com/document/d/1g9wWpB0f-8hpaNcB5_RCDBloIHQZuI0gumlQUWzkVQQ)  
> 保存时间：2026-10-15 23:24:06（America/Los_Angeles）  
> 原文最后修改时间：2026-10-15T21:30:06.913Z  
> 说明：这是原文内容快照，保留原规划日期及历史状态；供并行开展 Track 1 时使用，不代表已完成实验。

# Track 1｜单目 4D 人物交互重建

参赛评估、三条路线与实验计划 · 2026-10-14

## 1. 建议与可行性

建议参赛，以 MHR 原生人体重建为接口底座，以 CARI4D 思路的人物联合优化为主攻。相比另外两轨，本轨不需要机器人策略训练，主要成本在视觉推理、网格生成与序列优化；适合先争取完整有效提交。最大难点是遮挡、单目尺度、接触和输出表示统一，不是单帧模型排行榜。以下路线是工程判断，尚无本赛题实测成绩。

三条路线按 A 保底、B 主攻、C 难例补强推进，不建议分别维护三套完整系统。三轨共同截止为 2026-11-04 17:00 EST（洛杉矶 14:00 PST）。在三周窗口内，优先保证全部样本可输出，再做局部创新。

## 2. 数据与输入输出

官方 Track 1 README：30 个 episode、16,563 帧、30 fps、10 个目标物体、22 个动作描述；每段只有一个静态第三视角 RGB 视频，分辨率 1536×1152。不同 episode 来自四个物理机位，不等于提供同步多视角。页面显示约 513 MB。[数据来源见文末]

公开 Parquet 只含时间、帧、episode、任务等索引，未提供人体/物体位姿、可见性或网格。metadata 提供目标物体、prompt、camera、sequence_id 等。不要把 data/ 目录误认为 GT。每个物体有三段视频，可用于规划物体分组，但不得默认所有视频的物体坐标和尺度已对齐。

输出契约（官方 pack_reconstruction.py 已读取）：每 episode 一个 NPZ 与一个物体三角网格。pose [T,136] 为 MHR 参数；scales [68]、shape [45] 在整段共享；object_rotation [T,3,3]、object_translation [T,3]、object_scale 为正标量。物体变换为 p_scene = s R p_mesh + t。按视频帧索引覆盖所有 scored frame。

官方打包器生成一个 Parquet（row_id,x,y,z,code_commit_url），同一文件投五个指标榜。不要手工拼 row_id；以 sample_submission 的实际评分帧为准。物体网格预算在配置中为 4096 顶点/面，由打包器处理。

### MHR 是首先要解决的接口

普通 SMPL/SMPL-X/SMPL-H 参数不能直接作为 MHR 参数提交。工具包提供的是 MHR 拓扑网格→MHR 参数转换，要求 [T,18439,3] 的对应顶点；它不是任意人体网格或 SMPL 的通用转换器。优先选择能直接输出 MHR 的人体模块。若引入其他人体模型，需要另做表面对应和 MHR 拟合，并计入额外误差。

转换说明的输入网格约定为 diag(1,-1,-1) × MHR_native / 100（米）；MHR 根平移等参数保留模型原生单位，不能把所有字段一律除以 100。先做参数→顶点→转换→顶点的回环检查，确认坐标轴、单位、手部误差及整段固定身份。工具包的近无损结果只适用于其可表示的 MHR 网格，不保证任意预测网格都无损。

## 3. 指标如何决定方法

五指标都越低越好：CD-H 人体表面距离、CD-O 世界系物体表面距离、ACC-H 22 个人体关节的二阶帧差误差、ACC-O 物体平移二阶帧差误差、PEN 手点进入提交物体网格的穿透深度。工具包只对连续评分帧计算二阶差，单位为 cm/frame²，不能未经核对乘 fps²。

评分只用首个评分帧的人体顶点拟合一个 Sim(3)，并用于整段人物与物体；不会每帧重新对齐。因此第一帧人体质量、物体相对人体位置、全程漂移都重要。物体初始误差不会自动消失。

ACC 是相对参考加速度的误差，不是越平滑越好；过度滤波会抹掉真实抓取、碰撞与换手。PEN 降低也不能靠把物体移远：必须联合检查 CD-O、接触位置与视频重投影。接触约束应只在估计接触阶段启用。

官网描述准确性/物理合理性两轴等权，FAQ 又描述各指标名次求和。最终跨轴聚合方式应向组织者确认；本计划保留五指标独立结果，不擅自用自定义加权和宣称排名。

## 4. 三条优先路线

### A｜MHR 原生人体 + 分模块物体跟踪：保底

组合：SAM 3D Body/MHR → 每段身份参数固定 → 目标分割与可见片段筛选 → SAM 3D Objects 等自建网格 → 单目深度/尺度初始化 → FoundationPose 物体跟踪 → 温和时序正则 → 官方打包。FoundationPose 的深度/参考资产条件需要由上游提供，不能把它当作无条件 RGB 到 6D 模型。

优势：模块容易缓存、定位错误，MHR 接口成本较低。弱点：各模块独立估计会出现尺度不一致、手物错位与遮挡后方向翻转。首测选清晰单手片段，输出人体/物体叠加、单位检查和完整 NPZ；以能否全链路通过验证作为门槛，不以可视化好看作为完成标准。

### B｜CARI4D 式人物联合优化：主攻

在 A 的初始化上联合优化人体姿态、物体位姿、共享尺度与接触。使用可见轮廓/重投影、接触距离、穿透、速度/加速度先验；保留多个位姿候选，减少对称物体翻转。先固定网格优化轨迹，后对少量困难物体改几何，避免一次释放所有自由度。

CARI4D 是最贴近本轨的参考：类别无关、单目人物交互、候选选择与接触细化。原论文仓库含 SMPL-H 依赖；V2D 集成版也必须核实实际导出表示。不能假设直接运行即得到比赛所需 MHR。优先把接触/候选优化接到 A 的 MHR 底座，或明确验证适配层。

实验：A → 加接触 → 加遮挡鲁棒项 → 加多候选，逐项消融。继续条件：独立验证数据上几何误差改善，且接触与动态误差无明显退化。两天仍主要卡在表示转换则退回 A 的原生 MHR，不继续重写整套系统。

### C｜多关键帧几何 + 分段时序恢复：难例补强

从不同物体朝向和遮挡较少的帧提出多个几何/方向候选，用多帧轮廓和重投影一致性选择，再对换手、翻转、重遮挡做分段恢复。优先在少量 CD-O 高风险物体上使用，人体仍复用 A/B。

GVHMR 可作为人体时序/重力一致性参考，但输出模型与比赛 MHR 不同，需拟合验证；不列为立即替换全部人体模块的承诺。该路线目标是解决遮挡后的连贯性与物体形状，而不是用更强平滑压低运动幅度。

停止条件：多视角候选仅改善单帧渲染而无整段几何证据，或转换误差抵消人体改进，则只保留关键帧选择和异常帧重估。评测 GT 隐藏，视觉代理指标不能单独证明本轨得分提升。

## 5. 实验与验证计划

启动前：固定数据和代码版本；解析 30 段元数据及 sample 评分帧；解码视频并核对帧索引；选清晰单手、双手、强遮挡、物体翻转、长段五类代表。当前仅查阅官方数据说明及本地提交代码，尚未下载/解码 Track 1 视频或运行模型。

正确性检查：有限数值、正物体尺度、旋转矩阵正交/行列式、整段 shape/scales 固定、MHR 回环拟合、网格变换一致、遮挡帧连续、帧数与 row_id 完整。把 2D 重投影、轮廓 IoU、穿透和速度曲线作为诊断，绝不称为隐藏 GT 分数。

外部验证可选 CARI4D 使用的 BEHAVE 等公开 HOI 数据，但先确认访问许可及人体表示适配；按物体或序列留出验证，避免把同一视频相邻帧当独立测试。每个实验固定网格预算与评估协议，记录各指标、失败率、显存峰值和每段耗时。

最小消融：A0 分模块；A1 固定人体身份；B1 接触约束；B2 接触+遮挡；C1 多关键帧。先用代表样本筛选，再全 30 段；只在完整有效候选上消耗正式提交次数。

## 6. 三轨并行时的排期与资源

10/15–17：MHR 与打包回环，完成 A 单段。10/18–22：A 全覆盖，B 小样消融。10/23–27：B 主线与 C 难例修复，形成完整候选。10/28–31：提交反馈、错误分层与复核。11/01–03：冻结产物、哈希、技术报告；11/04 留故障缓冲。

不假设未知 GPU 可用时长。先测推理、网格生成、优化与转换的独立耗时。MHR 转换说明提供约 14 GB 峰值的特定硬件示例，可减 batch；这不是整套重建系统最低显存保证。核实 GPU 架构与各容器支持，尤其官方对部分 Blackwell 重建依赖的兼容限制。

与 Track 3 共享：分割、物体生成、FoundationPose、坐标工具、可视化、接触优化。与 Track 2 Tier 3 共享：视频重建代码及经验证的结果映射。必须保持数据隔离：FAQ 明确 Track 1 不得使用 Track 2 资产补网格或恢复相机参数；共用代码不等于允许共用真值或资产。

## 7. 待确认问题与阅读顺序

优先确认：MHR 模型版本与导出单位；评分帧清单；集成版 CARI4D 的真实人体输出；五指标最终排名聚合；不同 episode 的几何复用边界。技术报告必须如实记录自动/人工步骤与外部模型。

阅读顺序：提交工具包 Track 1 README/pack_reconstruction/MESH_TO_MHR_PARAMS → MHR 与 SAM 3D Body → CARI4D → FoundationPose → SAM 3D Objects → GVHMR。前三项先解决输出可提交性，后几项服务于精度改进。

## 8. 来源与关联规划

[官网与赛道规则](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)

[官方 FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)

[提交工具包（本地已检查 README、配置及代码）](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip)

[V2D 发布记录与兼容性](https://github.com/nvidia-isaac/video_to_data/releases)

[Track 1 数据说明](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/track_1/README.md)

[CARI4D 论文](https://arxiv.org/abs/2512.11988)

[CARI4D 官方代码](https://github.com/NVlabs/CARI4D)

[SAM 3D Body 论文](https://arxiv.org/abs/2602.15989)

[SAM 3D Body 代码](https://github.com/facebookresearch/sam-3d-body)

[MHR 模型](https://github.com/facebookresearch/MHR)

[FoundationPose](https://github.com/NVlabs/FoundationPose)

[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)

[GVHMR](https://github.com/zju3dv/GVHMR)

[Track 2 规划](https://docs.google.com/document/d/1OJ-Q7KDToEjTca36JSLt7viWCRIVdqIBCopjVz2BE70)

[Track 3 规划](https://docs.google.com/document/d/1N2ynwr2BCK-mGDOQK9iroy4VXMmQCVpTXzrCv-cUO28)

