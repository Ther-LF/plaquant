# Motivation §3.2：固定精度分配下的表示变换实验

日期：2026-11-24  
论文：AdaMix（代码可能仍使用 ProMix）  
状态：实验设计，尚未运行或填入新结果。  
目的：交给实验 agent 执行；与 `paper_motivation_3_1_experiments.md` 配套，不改正式 evaluation 矩阵。

## 1. 要回答的问题

**固定分配规则和实际精度分配，改变表示是否还能改善量化质量？**

问题本身不预设 PCA 或某种旋转最优。具体实验采用 activation-variance-based 的层内分配，参照 ResQ 用 PCA 构造通道表示，然后比较直接量化与精度组内正交变换后量化。

本任务只有两个实验条件：

| ID | 配置 | 被改变的因素 |
|---|---|---|
| Direct | 方差组织的表示，按固定精度划分后直接量化 | 无额外精度组内旋转 |
| Rotated | 相同表示和精度划分，在两个精度组内分别正交变换后量化 | 仅增加组内正交变换 |

**每个线性层内只有高、低精度两个逻辑组。**本实验不是固定宽度 ring512/64 分组，不比较原始表示，不比较 rotation-only，不进行 across-layer 分配，也不加入 measured-error allocator。量化 scale block 与这两个逻辑组不是同一概念。

不能从这个实验推出：PCA 优于原始表示、全局分配最优、固定宽度组最优或变换一定改善 PPL。实验观察到哪些变化就报告哪些变化。

## 2. 固定条件与模型范围

- 主模型：Llama-3.2-3B、Llama-3-8B；1B 可作实现检查，Qwen3-8B 为可选扩展。
- 候选格式：低组 NVFP4、高组 MXFP8；KV16。采用真实格式的 scale、舍入、clipping 和已有主 PTQ 配方，不用理想 INT4/INT8 替代。
- 采用 §3.1 的模型 revision、tokenizer、fit/diagnostic 数据和 eligible 投影清单。默认 q/k/v/gate/up/down；如因合法变换路径限制范围，须在 Direct/Rotated 两侧一致并报告。
- fit 默认 WikiText-2 train 512×2048、seed0。PCA 基在所有比例间冻结。局部诊断使用与 fit 不重叠的32个2048-token窗口。PPL 用 WikiText-2 test 和固定 C4 validation 256窗口，seqlen2048。
- PCA 基构造、共同预处理、成员排序、每层配额、权重/激活量化算法、GPTQ 超参数与校准数据均匹配。候选权重按同一 PTQ 程序分别生成；不要声称旋转后量化权重逐元素相同。
- 不复用量化模型自身的层输入作为局部诊断输入。两侧使用相同 teacher 语义输入；PPL 则正常运行完整量化模型。
- 不用测试 PPL 搜索旋转矩阵或决定保留哪个 seed。

## 3. 层内划分与比例

对投影 l，令共同的 PCA/ResQ 式基为 U_l，Z_l=X_l U_l。在这一表示上统计 activation variance，按固定顺序确定高方差通道。PCA 是构造表示的工具；图例使用 Direct / Within-group rotation，正文说明 activation-variance-based assignment。

沿用 §3.1 对基构造的审计要求：现有 `basis.py` 累积未中心化 X^T X，记录其定义、damping、共享或分块方式；测量 Z 的实际中心化方差，不把 second-moment eigenvalue 自动称为 variance。

每层按预先规定比例 r 选出 h_l 个高方差通道组成 H_l，其余为 L_l；H_l 使用 MXFP8，L_l 使用 NVFP4。Direct 与 Rotated **保存并复用同一个 precision mask**。旋转后不重新计算排序、不调整配额、不重选精度。

默认非退化比例网格：r_target=0.1,0.2,...,0.9。与 §3.1 同样采用格式对齐后的实际配额，横轴报告 actual ratio。该范围对应两个非空组；r=0/1 可作为单格式附加对照，不标为“两组混合精度”。

配额遵循实际量化函数要求：NVFP4 K_low 能被16整除、MXFP8 K_high 能被32整除（以运行版本核验）。从可行配额中选最接近 r*d_l 的数量，平局取较小值。不因旋转算法要求偷偷移动组边界。任意维度正交矩阵应能覆盖合法配额。

精度分区内使用确定的基础通道顺序，两配置一致。每个逻辑组内部仍有多个真实 scale blocks；不设每通道一个 scale，也不让整组共享一个人为大 scale。

## 4. 正交变换与正确计算

对 g∈{H_l,L_l}：

    Direct:  Z_g, V_g
    Rotated: Z_g R_g, V_g R_g
    R_g R_g^T = I

其中 V=W U（采用行向量激活和 Y=XW^T 约定），两组乘积之和保持同一浮点输出。偏置如存在，两侧保持一致。

主变换建议采用确定性随机正交矩阵：Gaussian matrix → QR → 对角符号校正，float64构造，再使用共同计算dtype。种子由model/module-or-shared-unit/ratio/group/rotation_seed稳定派生。使用未经训练的旋转，不追加旋转优化。若使用ResQ已有矩阵，核验其维度、随机或训练来源及组边界，写入manifest。

必须作用于**整个高精度组和整个低精度组**，不能以ring512替代；若现有快速Hadamard只支持特定维度，可使用正确的一般维度离线/显式变换路径，或披露缺失点。不能用不正交的截断Hadamard。

完整网络有残差、非线性和共享基约束，不能把每层任意旋转直接吸收到所有权重后假设等价：
- 优先使用经过无量化等价检查的网络变换路径；共享矩阵按真实共享单元生成。
- 若为了隔离实验采用显式 Z_g→Z_g R_g 操作，必须正确配对权重，披露其为算法级参考实现；本节不声称零运行开销。
- 局部层输出比较要在同一输出坐标系下；如输出也换基，先恢复后比较。
- 无量化时检查正交残差、组内能量保持、层输出、完整网络logits与PPL；记录计算dtype、数值容差和实测误差。不要求低精度计算bit-exact，不把变换实现错误当量化结论。

## 5. 四类指标

统一在 held-out diagnostic 输入上测前三类，第四类使用独立PPL评估集。所有局部输出使用teacher输入，从而不混入前层传播误差。

### 5.1 Activation magnitude：数值分布

分别对H/L展示 Direct和Rotated的：
- 绝对激活值分布：分位数p50/p90/p99/p99.9/max，以及CDF或分位数曲线。
- 实际scale block内 `max(abs(z))/RMS(z)` 的分布，RMS按该block元素计算；全零block单独计数、比值记0。
- 每组总平方幅度/能量的变换前后比值，作为正确性检查，不作为待追求的下降指标。

正交变换保持整组能量；预期检查的是幅度集中程度是否变化，不能宣称旋转降低总能量，也不要求每个block的能量不变。

代表层预先选浅/中/深block，各选q_proj与down_proj；无对应投影则在manifest记录替代规则。全层统计必须保留，不事后挑最漂亮的层。不同层原始幅度不能直接混合成一个分布；全局摘要先按层计算指标，再层等权汇总。H/L分开。

### 5.2 Activation quantization error：激活本身

对同一层的每个精度组：

    Z'_lg = Z_lg             (Direct)
    Z'_lg = Z_lg R_lg        (Rotated)
    A_lg = sum ||Z'_lg - dequant(quant_f(Z'_lg))||_F²
           / (sum ||Z'_lg||_F² + epsilon)

sum跨诊断窗口累计后再相除。真实scale按各自数值重新计算；固定的是scale算法及超参数，不是强迫两侧使用相同scale值。epsilon=1e-12；同时保留SSE、ENERGY、元素数、原始MSE。累计float64，差值至少float32。

分别对H与L报告跨线性层mean±std（总体标准差，ddof=0）。不要混合不同格式的两个组后只给一个平均值。A是normalized squared error，不开平方、不叫relative L2。

### 5.3 Linear-layer output quantization error：实际层计算

    E_l = sum ||Y_l - Yhat_l||_F² / (sum ||Y_l||_F² + epsilon)

Y_l为共同浮点teacher线性输出，Yhat_l包含两精度分区、量化权重和量化激活的完整输出，正确处理scale和坐标变换。使用同一X_l；与§3.1同一口径。

对共同eligible集合的各个线性投影等权报告mean±std（ddof=0），保留每层值、projection-type分层统计、p50/p90/max。std表示层间差异，不表示seed不确定性。零/近零输出能量单独记录，并报告绝对误差。

H/L的激活误差与完整输出误差分别展示，不能假设后者等于前者相加；权重及交叉项也影响输出。

### 5.4 Model perplexity

WikiText-2和C4分别报告，以总NLL/有效token数取exp。不平均窗口PPL。报告相同dtype的浮点参考线，不用另一个模型revision或KV位宽的旧基线拼接。

数值误差下降但PPL不变、某精度组恶化、PPL上升等都保留，不预设四种指标单调一致。

## 6. 实验矩阵与seed

- 主矩阵：2模型 × 9比例 × Direct/Rotated，rotation_seed=0，fit_seed固定0。
- Direct可在完全匹配协议后复用§3.1的AV-Within结果；核对权重PTQ、mask、表示、数据与量化布局，不仅核对PPL数字。
- Rotated在预设r=0.1/0.5/0.9补rotation_seed=1,2，Direct不重复运行假装三个独立样本。
- 主曲线使用seed0配对结果；另表给三个旋转seed的指标及seed mean/std。若汇总曲线，先明确逐层跨seed平均后计算层间std，不能把seed×layer混成同一误差棒。
- 主矩阵36个逻辑配置，附加旋转seed12个配置；Direct复用可减少运行。首先检查一个模型/中间比例的正确性，再跑完整矩阵，失败点记录后继续其他点。

## 7. 图设计

不要强行把四类数据塞到一幅拥挤的图。建议两个图：

**图A：分布和激活误差诊断。** 固定预设r=0.1（或其实际对齐点），代表层H/L分别展示Direct/Rotated的幅度分布；配套全层block max/RMS摘要与H/L激活误差。r=0.5/0.9诊断放附图。颜色始终表示Direct/Rotated，不改变语义。

**图B：随比例变化的误差与模型质量。** 横轴actual FP8 channel fraction，分panel显示H/L的A均值±层间std、E均值±层间std、Wiki/C4 PPL。必要时拆分，不能通过缩小字号强行合并。caption明确两组是每层高/低精度分区，不是固定宽度ring。

保留完整逐层结果供检查局部恶化。mean−std为负不等于负误差；默认线性轴，限制显示范围须注明，不在log轴静默截断带。

使用真实数据与绘图脚本，导出PDF及PNG。论文集成路径为 `fig/motivation-representation-distribution.pdf` 和 `fig/motivation-representation-quality.pdf`；本任务先保存在独立数据目录。

## 8. 当前实现陷阱与复用入口

已检查 `promix/quantize/rotation.py`：
- `build_main_transform(..., identity_main_rotation=True)` 仅关闭main rotation；不等于关闭所有R2/key/value/down_proj变换。
- `ring_r1=True` 优先级更高，可能覆盖identity设置。本任务必须禁用ring路径。
- dense路径使用`R1_1/R1_2`（可另有R1_0），须检查是否真的只有本实验两个分区，矩阵维度是否匹配每个r；不能复用固定r=1/8的旧矩阵扫所有比例。
- `fuse_basis_to_model` 使用共享残差流基；不能假定支持每个线性投影独立的任意组旋转。

每次run输出transform manifest，逐项列出哪些变换在两侧共同开启、哪些被实验改变。若实际只比较main R1，则结论范围也限于main R1，不能写成所有线性层全部旋转的效应。

现有 `rd_validity.py`、`rd_mmlu.py` 默认ring/measured设置不能直接作为本实验命令。这里不提供未经验证的CLI；执行agent应实现/适配明确的Direct与Rotated入口并记录可复现命令。

## 9. 输出与验收

输出到 `docs/motivation_3_2/`：

- `inventory.md`：环境/代码commit/数据/模型、可复用结果、合法变换路径及缺口。
- `manifest.json`：fixed config、fit/diagnostic/eval数据hash、精度mask、实际比例、PCA与R hash、seed定义、scale配置、变换范围。
- `runs.csv`：run_id/model/ratio_target/ratio_actual/variant/rotation_seed/wiki_ppl/c4_ppl/A_high_mean/std/A_low_mean/std/E_mean/std/status。
- `layer_metrics.csv`：逐投影各组的元素数/SSE/ENERGY/A、完整输出的SSE/ENERGY/E、原始MSE、组大小和格式。
- `magnitude_stats.csv`：逐投影逐组的分位数、block max/RMS统计、零block数量、组能量比；分布采样若使用固定reservoir需记录seed和样本数。
- `equivalence.json`：无量化的正交性、层输出/logits/PPL桥接检查。
- `failures.csv`：坏点和原因，不补造数据或静默插值。
- `report.md`：四类指标分别回答了什么、哪些趋势一致/不一致、适用范围和未完成项。
- 绘图脚本、PDF与PNG，数据可追溯到run_id。

验收：两侧mask与初始basis一致；变换只在每层各自精度组内；没有ring/group allocator混入；scale按真实格式工作；无量化等价性通过；四类指标可从原始汇总数重算；层间std与seed std明确区分；不以“Rotated胜出”作为代码或实验通过条件。

本任务书未启动GPU实验，也未修改原实验文档或量化实现。
