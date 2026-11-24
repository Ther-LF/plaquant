# Motivation §3.1：精度分配策略与分配范围实验任务书

日期：2026-11-24  
论文：AdaMix（仓库和历史报告仍可能使用 ProMix）  
状态：待实现、待运行的实验设计；本文不包含新测量结果。  
用途：交给实验 agent 执行，返回可核验数据和论文图。仅覆盖 §3.1，不启动 §3.2 旋转消融或 §3.3 性能实验。

## 1. 叙事目标与边界

本节先问：在相同精度比例下，层内选择哪些通道是否影响模型质量？随后问：保持选择依据不变，允许跨层分配是否产生进一步变化？

实验分为两部分：

- **M31-A：层内策略比较。** 所有策略采用相同逐层配额，比较 Random、Squared-sum、Mean-abs、Activation-variance 四种策略。
- **M31-B：分配范围比较。** 对三个有分数的策略，分别比较 within-layer 和 across-layer；固定同一策略的表示与分数定义，只改变配额约束。

两部分都报告端到端 WikiText-2/C4 PPL，以及逐线性层输出量化误差的 mean ± std。PPL 不敏感不等于数值误差没有变化；局部误差下降也不自动证明模型质量改善。

不预设全局分配最优，不要求 PCA 策略胜出，不删除不利结果。不加入 measured quantization-error reduction 选择基线，不用 ring/group 作为本实验的精度选择单位，不做组内随机正交旋转。

**表示差异必须公开：**前三种策略在共同原始表示上选择；Activation-variance 按 ResQ 思路使用 PCA 构造表示。A 比较的是不同选择策略（其中包含这一表示差异），不是严格的“只换排序指标”消融。B 中同一策略的 within/across 对照才仅改变分配范围。

## 2. 与现有实验文档和代码的关系

基准文件：`docs/paper_experiments.md`（已读，含 2026-11-24 更新）。

沿用其候选格式 NVFP4/MXFP8、PPL seqlen=2048、主比例网格 0.0–1.0 共 11 点，以及可复用的评估器。现有块 1 的最终方法曲线为 ring512 + measured 分配，**不能直接充当本任务的通道级策略曲线**。

本地已检查的实现入口：

| 文件 | 可复用部分 | 不能直接沿用之处 |
|---|---|---|
| `promix/scripts/rd_mmlu.py` | 模型配置、Wiki/C4 PPL、结果输出的入口 | 默认 ring512/measured；不是四种通道策略 runner |
| `promix/scripts/rd_validity.py` | 校准 hook、重排/GPTQ 配套路径的参考 | RING=64，旧 variance proxy 与本任务不同；含预设 gate，不作为新实验成败标准 |
| `promix/quantize/quant_utils.py` | 真实格式 fake-quant、scale 规则 | 不能改成每通道独立 scale；按最终分区重新计算 block scale |
| `promix/quantize/basis.py` | 已有基构造与保存流程 | 当前累积 X^T X；不能未经核验称为中心化 covariance 或直接把其 eigenvalue 当方差 |
| `promix/quantize/rotation.py`、`promix/models/loader.py` | 合法变换、列排列和 hook 路径 | 需检查原始表示路径是否暗中仍加载 PCA/随机旋转 |

`CLAUDE.md` 中部分 INT4/INT8、早期 GPU 状态已与正式实验文档不同；当前格式从实际配置和量化函数确认，不据旧说明替换为整数格式。

现有文档的“bias 实验”指**组件逐项收益分解**，不应改称统计偏差分析。本任务另存各层配额和分数分布，用于检查跨层尺度影响；不改写原 bias 实验的含义。

开跑前先返回 `inventory.md`：代码 commit/dirty 状态、可用 GPU 与运行环境、模型及 tokenizer revision、可复用数据、需要新增的 runner、变换路径和 eligible 层清单。不要覆盖其他 agent 正在使用的配置或历史结果。

## 3. 模型、数据与固定条件

### 3.1 执行规模

- 主矩阵：Llama-3.2-3B、Llama-3-8B。两个模型都完成，不能只选趋势明显者。
- 1B 可用于低成本实现检查，不替代主模型。
- Qwen3-8B 作为后续跨架构扩展，不阻塞本节首轮结果；70B/32B、MMLU、0-shot 不属于本任务必需范围。
- GPU 使用实际可用硬件并记录；本任务是 fake-quant 算法级质量评估，不声称验证生产 kernel 的性能或任意 mask 的原生可执行性。
- 固定 KV16，不在本节同时扫 KV4。其余未研究模块（embedding/lm_head/norm 等）采用统一配置。

### 3.2 校准与诊断数据

- 选择/PCA/PTQ fit 数据沿用已验证的 WikiText-2 train 512 个窗口、长度 2048、seed0 配方；记录窗口 ID、tokenizer、BOS 与样本 hash。若某子步骤只用子集，明确样本数和同一子集映射。
- 每个被测投影从相同 fit token 位置收集分数，默认上限 4096 个非 padding token；不能不同层因 hook 顺序取得不同样本数。
- PCA 基及统计在一个 seed 内冻结，跨比例和 within/across 共用，不能为每个结果重搜表示。
- 另取 32 个不与 fit token 区间重叠的 train 窗口（长度 2048）作为 held-out 局部误差诊断集；固定该集合，不用于调分配、调 scale 或挑模型。完整处理其有效 tokens，显存不足用流式累计。
- PPL：WikiText-2 test 与 C4 validation，seqlen2048；C4 沿用现有 256 窗口协议，固定窗口 ID。不能用测试 PPL 搜索比例、分数归一化或配置。
- 先以 seed0 做全矩阵。Random 另用 assignment_seed=1,2（校准、基和量化配方固定），三次仅改变随机通道优先级。
- 校准 seed 的稳健性可在 r=0.1/0.5/0.9 补做，但不与 Random 的分配随机性或层间 std 混为同一种误差棒。

### 3.3 被测投影与模型等价性

默认候选为各 Transformer block 的 q/k/v/gate/up/down 线性投影；以当前模型配置和合法变换路径核对。o_proj 若必须使用独立 per-head 路径，则所有配置保持同一设置，排除出分配集合，并记录原因。不要依结果调整 eligible 集合。

原始表示基线采用共同浮点预处理后的表示，列明 norm fusion、不可缺少的在线变换；不能默默加载全套 AdaMix 变换后称“原始通道”。PCA 组使用已验证的等价实现，关闭用于 §3.2 的额外组内随机旋转。

量化关闭时，各策略的排列/变换模型须与共同浮点模型在输出和 PPL 上一致到明确数值容差。若某个局部 PCA 变换不能合法贯穿完整 Transformer，不能强行吸收穿过非线性或残差连接：修复正确性，或明确仅有局部结果、PPL 未完成。若采用显式在线正交变换做算法实验，应披露其额外操作，不宣称无成本部署。

共享 q/k/v 输入不等于必须共享独立线性层的精度 mask；fake-quant 可分开执行，但必须正确配对激活和权重列。若实现强制共享 mask，就将共享单元列入协议，并对所有策略采用相同约束；不能各策略暗中使用不同的可行集合。

## 4. 四种策略的具体定义

给定投影 l 的 fit 激活 X_l ∈ R^(N×d)，使用相同非 padding token 数 N。

| ID | 图例 | 表示 | 通道分数 |
|---|---|---|---|
| RND | Random | 原始 | 每通道一个固定随机优先级 |
| SS | Squared-sum | 原始 | s_lj = sum_t X_l,tj² / N |
| MA | Mean-abs | 原始 | s_lj = sum_t abs(X_l,tj) / N |
| AV | Activation-variance | ResQ 式 PCA 基构造的表示 Z_l=X_l U_l | s_lj = mean_t Z_l,tj² − mean_t Z_l,tj 的平方 |

SS 除以 N 不改变同层平方和排序；跨层这样避免 token 数不同造成虚假偏置，正文说明采用 mean squared activation。AV 的方差用稳定累计法计算，仅对舍入造成的微小负值夹至零。主分数使用实际 Z 的中心化方差，不直接把 X^T X 的特征值叫作方差。

PCA 实现应记录使用中心化 covariance 还是未中心化 second moment、damping、共享基/逐投影基/分块基。沿用现有 ResQ 式未中心化基时如实写明，并在其输出上测量上述分数；不要静默换成另一种 PCA 定义。基只作完整正交换基，不做低秩截断。

原方法来源：Atom 的 outlier 平方和、MicroMix 的平均绝对激活值、ResQ 的 PCA/方差选择。这里统一格式、配额等条件后只是策略适配，不是 Atom/MicroMix/ResQ 的完整算法复现；尤其没有复现 MicroMix 用阈值自定比例的步骤。

参考（用于定义来源，执行前可核对对应代码）：
- Atom §5.1：https://arxiv.org/html/2310.19102v3
- MicroMix §3.1：https://arxiv.org/html/2508.02343v1
- ResQ：https://arxiv.org/html/2412.14363v2

### 分数与打包顺序要分开

按分数决定哪些通道入选；选定后，两个精度分区内都按该表示的稳定通道 ID 升序排列。不能 SS 按分数排序打包、Random 按随机序打包，额外混入不同 scale block 组成规则。PCA 通道 ID 固定为基构造时的方差顺序并保存映射。

RND 为每个模型/层/seed 一次性生成优先级，所有 r 用同一排列的前缀，避免每个比例重新随机造成曲线噪声。分数并列按稳定 layer ID/channel ID 打破。

## 5. 比例、通道粒度与真实 scale

目标网格：r_target ∈ {0.0,0.1,...,1.0}。横轴使用实际 r_actual，不拿请求比例冒充实现比例。

精度选择以通道为单位，**不是 ring512/64，也不是每通道一个 scale**。先选择并打包同精度通道，再按真实格式分块计算 scale。当前函数 NVFP4 默认块长16、MXFP8默认块长32；执行时确认实际配置、全局/两级 scale、clipping 与 GPTQ 设置。

为兼容现有无 padding fake-quant，逐投影可行 FP8 数量取：F_l={h: 0≤h≤d_l, h mod32=0, (d_l−h) mod16=0}（以核实后的格式块长为准）。这限制的是**配额数量**，不要求被选通道在原始布局中连续或成组。若宽度导致集合为空，单独报告，不擅自改 scale 粒度。

M31-A 的配额 h_l(r) 取离 r*d_l 最近的可行值，平局取较小值。所有策略共用该配额。令 H(r)=sum_l h_l(r)，r_actual=H/sum_l d_l。M31-B 必须使用同一 H。

纯端点使用真实纯 NVFP4/MXFP8 fake-quant 分支：空分区跳过，不能把 all-but-one-ring 标为 r=1。由于 AV 表示不同，不要求 AV 与原始表示端点完全一致；同一原始表示中 RND/SS/MA 的端点应一致，同一策略 within/across 端点应一致。

精度分区变化后必须重算其实际 block scale，不能把原始通道的预先量化误差相加当结果。按相同程序重做 GPTQ 是允许的，但记录超参数和校准数据；不能声称所有策略最终权重逐元素相同。采用已有主配方并冻结 RTN/GPTQ、act-scale 搜索与 weight-scale 设置，不能逐策略调优。

## 6. 实验矩阵与跨层选择

### M31-A：四种策略，固定逐层数量

每模型 × 11个目标比例 × RND/SS/MA/AV。每层选分数最高的 h_l(r) 个通道。RND 3个 assignment seeds，其他策略主矩阵 seed0。

### M31-B：三个标准，逐层与跨层配对

SS/MA/AV 各自比较 Within 与 Across；Within 复用 A 的结果，不重跑。Across 保持各自的表示/分数，选总 H 个通道，允许各层数量变化。

主比较跨层使用第4节定义的原始分数，不额外作层内归一化；这保留与 Within 相同的评分函数。不同层输出/激活尺度会影响此排名，必须报告各层 score mean/max/sum、入选比例及分配图。它是被测策略的性质，不代表全局最优或 task-loss 最优。

当需要满足每层数量对齐时，不能先全局 top-H 后逐层独立四舍五入造成总量变化。采用确定性的配额约束最大分数选择：对每层排序分数计算各个可行 h 的前缀分数和，在 sum h_l=H 约束下选择配额，再取各层 top-h_l。对 d 为32倍数、可行步长32的常见情况，可按每层连续32个已排序通道的增量分数进行全局选择；层内增量非增，平局优先较早前缀。这只是对齐配额的实现，不是把原始通道改为固定组。非等步长情形用正确约束求解或报告暂不支持。

若补做 layer-normalized 分数，只作为另命名的敏感性实验；Within/Across 两侧使用相同变体，不得仅修改全局一侧并作为唯一结果。不得依据测试PPL挑归一化方案。

实验数量：每个模型主 seed 为4个 Within策略+3个 Across策略，共7×11=77配置；Random额外2×11=22配置，共99个逻辑配置。端点、重复实际配额或相同mask可以在验证配置完全相同时复用，仍保留逻辑映射。两个主模型为198个逻辑配置，不代表198次必须独立量化。

## 7. 指标：PPL 与逐层输出误差同时报告

### 7.1 模型 PPL

WikiText-2 test、C4 validation 分别报告；对有效目标 tokens 累积 NLL 后取 exp，不平均窗口 PPL。保存 NLL sum、token count、数据 hash、模型/量化配置。浮点参考采用共同 dtype 和统一协议，不混合 BF16/FP16 基线。

### 7.2 相同 teacher 输入下的局部输出误差

“层”在本任务指一个线性投影模块（如 block7.q_proj），不是把整个 Transformer block 合成一个数。

对 diagnostic 集合中的同一浮点 teacher 输入 X_l，计算 Y_l=X_l W_l^T（存在 bias 则两侧一致计入），以及该配置实际量化路径的输出 Yhat_l。不要用量化模型传播到这一层的不同输入替换 X_l，否则混入上游误差。

PCA/重排候选应把相同的语义输入经其正确变换送入候选算子，并把候选输出映射回同一输出坐标再比较。teacher 不量化；候选权重来自该完整配置的 PTQ。若最终实现对输出也做了基变换，必须还原后比较，不能直接比较不同坐标。

跨所有诊断窗口流式累积：

    SSE_l = sum_windows ||Y_l - Yhat_l||_F²
    ENERGY_l = sum_windows ||Y_l||_F²
    MSE_l = SSE_l / num_output_elements_l
    E_l = SSE_l / (ENERGY_l + epsilon)

E_l 是 relative squared output error，不开平方，不叫 relative L2。主跨层统计用 E_l；同时保留原始 MSE_l。epsilon固定1e-12；累计建议float64，输出/差值至少float32。记录零或近零 ENERGY 的层，不能让epsilon掩盖异常；补充其绝对误差。浮点变换后的无量化对照也输出误差作为数值底噪。

不要先对每窗口归一化再平均来替代上述总SSE/总ENERGY，避免短窗口/低能量窗口改变权重。

### 7.3 mean ± std 的精确定义

对共同 eligible 集合中 L 个线性层：

    mean_layers = sum_l E_l / L
    std_layers = sqrt(sum_l (E_l - mean_layers)^2 / L)

采用总体标准差（ddof=0），每个线性层等权。这里 std 表示**层间差异**，不是置信区间、校准波动或随机seed波动。所有配置使用同一层集合；出现坏层不能从该配置独自删掉再比较。

附加输出按 projection type 的 mean/std，以及 p50/p90/max 和完整 E_l，防止整体均值掩盖少数层恶化。可另给能量加权汇总，但不得替换主层等权指标。

Random：每个 seed 先独立算 mean_layers/std_layers。主图的均值可使用 seed平均；层间带先对每层 E_l 做 seed平均，再在层间求std，并在图注明确该口径。另表报告各seed的mean_layers及其seed std。不能把3×L样本混成一个std。

如果只有PPL改善而局部误差不改善，或反之，两者都原样报告，不强行解释为因果一致。

## 8. 图与输出文件

默认输出到 `docs/motivation_3_1/`，避免覆盖正式 evaluation 结果。

- `inventory.md`：环境、复用情况、实现缺口。
- `manifest.json`：全部固定条件、模型revision、代码/数据/basis hash、格式和量化参数、eligible/shared约束、分数及归一化定义、seed语义。
- `runs.csv`：model,strategy,scope,representation,target_ratio,actual_ratio,seed,H,total_channels,weight_bytes,scale_bytes,padding_bytes,metadata_bytes,wiki_ppl,c4_ppl,error_mean_layers,error_std_layers,status,run_id。
- `layer_errors.csv`：run_id,module_id,projection_type,n_tokens,n_output_elements,SSE,ENERGY,MSE,relative_squared_error。
- `allocations.jsonl`：run_id,module_id,d,h,stable_selected_channel_ids,packed_order,score_summary,transform_id。另保存每通道完整分数张量。
- `failures.csv`：配置、错误、原因、是否可重试。坏点不阻塞其他点，但必须保留，不能静默插值。
- `report.md`：设定、观察、条件性解释、缺失项；不写预设胜负。
- `plot.py` 与绘图所需CSV：真实数据可复现绘图，不用image-gen生成数值曲线。

图A：Within策略对比，横轴actual FP8 channel fraction；两个PPL数据集分别成panel，另一个panel为layer relative squared output error mean ± std。AV图例/图注明确使用PCA表示。

图B：SS/MA/AV分别成配对小面板；颜色表示criterion，实/虚线表示Within/Across，横轴同A；同时给PPL与层误差。不要把7条曲线及所有误差带挤在同一面板。

图C（诊断附图）：在预设r=0.1/0.5/0.9附近的实际点展示各层误差与FP8比例，不能事后只选最漂亮比例。

mean−std可能为负，主误差图默认线性轴并保留实际带，不把负下界当物理负误差；如需限制可视区，在图注明确。不要在log轴静默裁剪误差带；可以另画分布图。

导出矢量PDF和PNG预览；论文集成时统一放到 `fig/motivation-allocation-within.pdf`、`fig/motivation-allocation-across.pdf`。图注写明层间std、seed口径、实际比例、格式、模型和数据，不声称iso-memory或iso-latency。

## 9. 执行顺序与验收

1. Inventory：核对模型、配置、数据、PCA及原始表示的合法路径，冻结manifest。
2. 实现四种通道策略和Within/Across配额选择；先检查分数、mask、布局和scale。不把 `--ring_size 1` 当作自动得到通道级实验的办法。
3. 1B或3B小样本验证：无量化等价性、W/X一致重排、纯端点、配额总数、共享输入约束、teacher-input误差hook。
4. 3B seed0先跑r=0/0.1/0.5/1，核对两类指标与结果文件；之后完成两主模型11点，不根据趋势好坏决定是否继续。
5. 补Random seeds1/2，生成主图和完整逐层数据。端点理论等价的策略应数值一致，差异超容差先排查实现，不当作方法差异。
6. 每个坏点记录原因后继续其他配置。不要把缺点当成功结果，不改请求比例凑更好曲线。复用数据需匹配完整manifest，不仅匹配模型和r。
7. 汇报两组问题的实际答案、所有不利结果和局限，实验结论回填论文前再审核。

本文件只授权描述实验任务，当前撰写工作未运行GPU实验、未更改量化实现、未修改 `paper_experiments.md` 中既有进度。
