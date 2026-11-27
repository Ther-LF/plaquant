# Kernel 代码与优化总结（kernels/mixed_gemm_sm120）

面向论文 kernel 章节写作与新人读代码。所有 file:line 均经打开文件核实（2026-11-27，
feat/rd-alloc 工作区现状，含未提交的 fk_spg_ext.cu）。文件路径均相对
`kernels/mixed_gemm_sm120/`，Python 路径相对仓库根。收益数字引自
`docs/rd_oppareto_data/kda_gb10_log.md`（下称 kda log）与 `docs/results_tracker.md`。

---

## 1. 一页总览：kernel 家族与 dispatch 路由

### 1.1 家族图

```
                      ┌─ 生产主线（CUTLASS fork，共享累加器）────────────────────┐
                      │ codesign_fused_kernel.hpp   双 phase 串行 glue（根）      │
                      │ codesign_sm120_bs_mma.hpp   block-scaled mma collective   │
                      │ codesign_bias_epilogue.hpp  castfold bias epilogue        │
入口 ext：            │ codesign_splitk_kernel.hpp  stream-K/split-K glue         │
 codesign_fused_ext ──┤ 变体入口: codesign_fused_bias_ext / cd_smallm_ext /       │
 (+5 个变体 ext)      │           cd_splitk_ext                                   │
                      └───────────────────────────────────────────────────────────┘
                      ┌─ ForgeKernel 进化线（手写 PTX kernel，GB10 专用窗口）─────┐
                      │ fk_smallk_ext.cu  双 phase + 整 tile smem 输出暂存        │
                      │ fk_spg_ext.cu     单 phase NVFP4（r=0 列）                │
                      └───────────────────────────────────────────────────────────┘
                      ┌─ decode 线（带宽型，非 tensor core GEMM）─────────────────┐
                      │ decode_gemv_ext.cu  W4A16/W4A4 GEMV + Marlin 配套         │
                      └───────────────────────────────────────────────────────────┘
producer 线：quant_pack_float / quant_pack_fused / hadamard_pack(28) /
             add_rmsnorm_pack / vllm_rms_quant_pack（激活量化+pack 单趟融合）
历史/探针线：fused_gemm_ext（pre-codesign 双 CUTLASS GEMM）、b1_fused/b1_tma
             （手写单 kernel 前身）、codesign_overlap/optc（PoC）、probe 族
```

### 1.2 Python 路由表（与 `promix/quantize/mixed_linear.py` 对账）

`MixedQuantLinear.forward`（mixed_linear.py:759-798）按 M 自上而下路由：

| 优先级 | 条件 | 路由 | 窗口表 (file:line) |
|---|---|---|---|
| 1 | `marlin_route(M)`，M=2..48 | Marlin W4A16 低段 + hi companion | `_MARLIN_TABLE` mixed_linear.py:282-291、`_MARLIN_HI_TABLE` :302-311 |
| 2 | `gemv_route(M)`，M=1 或 2..16 | decode_gemv（W4A16 默认 / W4A4 legacy） | 判定 mixed_linear.py:836-848；per-shape M 上限 :472-476 |
| 3 | GEMM 路（pack + `_gemm`） | 下表 | mixed_linear.py:662-757 |

`_gemm`（mixed_linear.py:662-757）内部再分（首个命中生效）：

| 分支 | 条件 | kernel 入口 | 窗口表 |
|---|---|---|---|
| r=0 fk 单相 | `is_pure_nvfp4` 且 `_fk_spg_cfg(M)` | fk_spg_ext `run_prepacked`（:666-674） | `_FK_SPG_TABLE_GB10` mixed_linear.py:209-219，GB10-only |
| r=0 castfold | `is_pure_nvfp4` 且 cd_bias 可用、fp16 输出 | `single_nvfp4_gemm_prepacked_bias`（:683-688） | 无窗口，默认 |
| r=0 fallback | 其余 | `single_nvfp4_gemm_prepacked` + torch cast（:690-696） | — |
| fk 双相 | `_fk_cfg(M)` 命中 | fk_smallk_ext `run_prepacked`（:697-707） | `_FK_TABLE_GB10` mixed_linear.py:165-197（(m_lo,m_hi,kh_max,swz,along_m) 带，首中生效；swz=0 为实测亏损排除带 :645-646） |
| bias castfold | 有真 bias 且 cd_bias | `fused_gemm_prepacked_bias`（:708-714） | 无窗口 |
| split-K | `_splitk_cfg(M)` | cd_splitk_ext `fused_gemm_prepacked_sk`（:715-726） | `_SPLITK_TABLE` :106-110（Pro6000）/ `_SPLITK_TABLE_GB10` :221-226 |
| small-M | `_smallm_cfg(M)` | cd_smallm_ext `fused_gemm_prepacked_cfg`（:727-734） | `_SMALLM_TABLE` :77-84（Pro6000 cfg1）/ `_SMALLM_TABLE_GB10` :128-131（cfg2/3） |
| 默认 castfold | cd_bias 可用 | `fused_gemm_prepacked_bias`（零 bias buffer，:736-748） | 全 prefill M≥1024 的主路 |
| 裸 GEMM | 以上皆无 | `fused_gemm_prepacked` + torch cast（:749-757） | 仅 ext 未加载时 |

平台切换开关：`_low_sm_device()`（mixed_linear.py:229-237，SM 数 ≤72 判 GB10/SM121；
Pro6000=188 走原表）。fk/fk_spg/GB10 表均经它门控（:637, :653, :609, :627）。

ext 加载（`promix/vllm_integ/kernels.py`）：cd+qpf 为必载对（kernels.py:44-68），其余
按 env 开关懒加载——smallm :90-96、qpfm :99-106、splitk :109-118、norm_pack（默认关）
:121-138、cd_bias :141-154、fk_smallk :157-174（`-lcuda`、编译失败返回 None 不影响主路）、
fk_spg :177-192、had_pack（默认关）:195-209、decode_gemv :212-228。torch≥2.13 自动切
`-std=c++20`（:36-41）。

---

## 2. 逐优化条目

### 2.1 双 phase 共享寄存器累加器（单 launch 融合的根）

**机理**：把 NVFP4(K_low) 与 MXFP8(K_high) 两个 block-scaled mainloop collective 装进
同一个 CUTLASS SM90 cooperative glue，对每个输出 tile 串行执行 Phase1→Phase2，两相写
**同一个 FP32 寄存器 fragment**，累加器全程不出寄存器，epilogue 只跑一次。机制源头只有
一行：fork 的 collective `CodesignBlockScaledMma` 加 `bool ClearAccum` 模板位
（codesign_sm120_bs_mma.hpp:90-91）门控 mma() 顶部唯一的 `clear(accum)`
（codesign_sm120_bs_mma.hpp:727）——NV 相 ClearAccum=true 先清零，MX 相 false 直接累加。

**代码位置**：
- glue `CodesignFusedKernel`（codesign_fused_kernel.hpp:60）；共享 fragment 分配
  codesign_fused_kernel.hpp:557（注释 "SHARED fragment"），Phase1 mma :559-564，
  Phase2 mma :568-587（同一 `accumulators` 传入）。
- producer 侧两相 load：NV :484-489，phase_bar1 后 MX :496-515（k_tile_iter 重置为 0，:502）。
- 共享可行性在编译期锁死：NV/MX 的 `partition_fragment_C` 布局/元素类型/size/shape 四项
  static_assert（codesign_fused_kernel.hpp:98-115）；该检查最早由去风险 PoC
  codesign_nvfp4_ext.cu:224-234 建立。
- SMEM 用 **union** 复用：`union MainloopUnion { nv; mx; }`
  （codesign_fused_kernel.hpp:168-172），注释明说 SM120 仅 ~101 KB/block，union 是
  BLK_K=128 双格式能装下的前提（:164-167）；正确性由两个 phase NamedBarrier 保护
  （producer :497/:511，consumer :571/:583；barrier user id 6/7 = effective 14/15，
  :385-386——注意 :381-382 的注释写 "IDs 8/9" 与代码不符，系注释笔误）。
- `DualPhase` 模板 bool（:59）使同一 glue 编译出单相 NVFP4 kernel（GemmSingle，
  codesign_fused_ext.cu:339），即 r=0 路复用双相 kernel 调优过的 Phase-1
  （codesign_fused_ext.cu:733-740）。
- host 端 MX tile 数：`mx_k_tiles = ceil(K_high/blk_k)`（codesign_fused_kernel.hpp:232）。

**收益**：单 launch 取代 2-launch+add；对照 docs/motivation_3_3_ext 的 naive 2-launch
r-sweep，mix2（双 launch 税）在 8B qkv r=0.125 M8192 为 0.568ms vs 融合路 ~理论线
（results_tracker.md M33-ext 节）。torch.profiler 验证每 `_gemm` 恰 1 个 CUDA kernel
（kda log Round 6/11/13 launch_verify）。

**适用窗口**：所有 GEMM 路 M（prefill 主力）；decode 小 M 另有 GEMV/Marlin 路。

### 2.2 warp specialization + multi-stage TMA pipeline（(4,2) 与 ~101 KB SMEM 约束）

**机理**：1 个 producer warpgroup（128 线程）+ 2 个 cooperative MMA warpgroup（256 线程）
= 384 线程（codesign_fused_kernel.hpp:126-128）；寄存器重分配 producer dealloc 40 /
consumer alloc 232（:135-140 规则，:465/:546 调用），与 GB10 SASS 实测 `setmaxnreg 40/232`
一致（docs/rd_oppareto_data/fk_sm121_recon_report.md §1）。NV/MX 各自独立 TMA pipeline，
stage 数由各自 DispatchPolicy 携带（:418-420），在入口 ext 以宏设定：`CD_NV_STAGES=4`、
`CD_MX_STAGES=2`（codesign_fused_ext.cu:82-87），tile 固定 128×128×128、Cluster 1×1×1
（:295-296）。

**SMEM 约束**：SM120 opt-in smem ~101376 B/CTA（config_dispatch.hpp:20-26，
codesign_fused_ext.cu:293-294 注释；`compat_report` 运行时打印实测 SharedStorageSize，
codesign_fused_ext.cu:774-776）。加深 stage 全部撞墙：GB10 实测 nv5mx2=106KB、
nv4mx3/nv5mx3=116KB 均 initialize 失败，仅 nv3mx2(82KB) 可装但全面亏 0.94-1.01x
（kda log Round 4a）；deep-MX（nv2mx3 等）同样超限（Round 14，+33.8KB/stage）。
(4,2) 因此是被约束"钉死"的配置，论文可直接写成 SMEM-constrained design point。

**SF 通路**（codesign_sm120_bs_mma.hpp）：SFA/SFB 走独立 TMA 通道（:326-327），与主操作数
同 pipeline stage 入 smem（:680-681），mma() 内 copy 进寄存器 fragment 后
`make_zip_tensor(tCrA,tCrSFA)` 喂给 block-scaled MMA atom（:839-845）；FP4 操作数入寄存器
后做 `fp4_shift_A/B`（:834-835）。mainloop 内嵌寄存器级双缓冲（先 copy k_block+1 再 gemm
k_block，:874-875），每 stage 末用保留 barrier `Sm120MainloopBarrier` 同步后 release
（:862, :889）。

### 2.3 相间 overlap（MX TMA fill 提前）——PoC，未入生产路由

**机理**：串行 kernel 中 Phase2 的 TMA fill 要等 phase barrier；overlap 变体把 smem union
拆成 SEPARATE `alignas(128) nv; mx;`（codesign_overlap_kernel.hpp:156-161），删除两个
phase barrier（:360-362 注释），producer 发完 NV load 后**立即**发 MX load（:466-478），
使 MX pipeline fill 与 NV 尾部 compute 重叠；MMA 仍串行（共享累加器，:523-538）。代价是
tensor smem 翻倍 → stage 必须变浅才装得下 ~101 KB（:156-158）。

**状态核实**：入口 codesign_overlap_ext.cu:1-12 自述为 "round-20 STAGE-1 PoC harness"，
mixed_linear.py 无任何 overlap 路由分支——**该优化是探索产物，未 shipping**（r20_overlap*.csv
为其数据）。写论文时只能作为设计空间讨论，不能计入生产收益。[与任务骨架"核心融合 kernel
家族"的描述有出入：它是家族成员但非生产成员。]

### 2.4 castfold epilogue（bias/cast 折叠；r>0 与 r=0 两个入口）

**机理**：validated 基线存 bf16 再在 Python 侧 `.to(fp16)`(+bias)，每层多两次 [M,N]
elementwise 往返。castfold 把 `half(float(bf16(acc)) + float(bias[n]))` 整链折进 GEMM
epilogue——必须先 round 到 bf16 再 fp32 加 bias 再 round fp16，才与参考算术逐位一致；
直接 fp32→fp16（codesign_fused_fp16_ext.cu 那条路）数值不过门（logit cos 0.983、
PPL +0.0645），100k 随机对验证直 round 错 81%（codesign_bias_epilogue.hpp:5-23）。

**代码位置**：融合算子 `LinCombBiasBFRoundHalf` + 叶子回调核心一行
`out[i] = Elem( float(bf16(alpha*acc[i])) + b )`（codesign_bias_epilogue.hpp:128-151）；
EVT 组装 `Sm90EVT<Callbacks, Sm90AccFetch, Sm90RowBroadcast<bias>>`（:173-191）；无 bias 层
用零 buffer（:212-218；Python 缓存 `_zb`，mixed_linear.py:487-492）。入口 ext 用
`#define CD_OUT_FP16/CD_EPI_BIAS` 后 `#include "codesign_fused_ext.cu"` 生成
（codesign_fused_bias_ext.cu:1-30）：
- **r>0 双相入口** `fused_gemm_prepacked_bias`（codesign_fused_bias_ext.cu:169-174）。
- **r=0 单相入口** `single_nvfp4_gemm_prepacked_bias`（:108-161；只建 NV mainloop args、
  显式 `arguments.K_high = 0`（:137）、用 GemmSingle 编译期裁掉 Phase2）。KDA Round 1 的
  最小改动即此入口 + mixed_linear 分派（kda log :31-38）。

**收益**：Pro6000 侧：prefill 每步 112/128 次 direct_copy、3.3/6.8ms（3B/8B L2048）消除
（kernels.py:141-154 注释）；GB10 侧 r=0 列 1.03-1.90x（kda log Round 1 表，r=0 中位
0.548→0.951 vs cuBLASLt FP4，Round 6 表）。**适用窗口**：所有非 smallm/splitk 窗口的
GEMM 路（_gemm 默认分支）。

### 2.5 两级 weight scale 的 kernel 侧消费

**机理**：一级 = per-block scale factor（NVFP4 block-16 UE4M3 / MXFP8 block-32 UE8M0），
在 mainloop 内随操作数 zip 进 block-scaled MMA（§2.2 的 SF 通路）；二级 = per-tensor 全局
`g_w`（mixed_linear.py:396），作为 **epilogue alpha** 在 bf16 round **之前**乘 FP32 累加器
（codesign_bias_epilogue.hpp:143-148；裸路 epilogue args `{alpha, 0.0f}`
codesign_fused_ext.cu:594-600；所有 GEMM 入口带 `alpha=1.0` 默认参 :794-807）。
fk_smallk 用 `fmaf(acc, p.alpha, bias)`（fk_smallk_ext.cu:566-569）。decode GEMV 不走
epilogue，把 g 折进 FP32 scale 买断或 sf8 变体在 kernel 内折（`float(E4M3)*g`、
`ldexpf(g, byte-127)`，decode_gemv_ext.cu:826-832，位精确论证 :812-825）。

### 2.6 K_low/K_high 128 对齐零填充（pad 同 launch 写零）

**机理**：K%128==0 时 K_high%128==64 ⇔ K_low%128==64，packed A_lo/B_lo 行仅 64B 对齐且
NV 相以半 k-tile 收尾，实测平坦罚 +11-19%（B-heavy 形状）；对**低段**补 64 列零即可恢复
（kda log Round 3 探针解剖：罚与 tile 数无关、补高段无效、补低段 70B down 1.16x）。
KDA-3 再镜像到**高段**：目的不是性能而是让 fk_smallk 的硬约束 K_high%128==0 在
r=1/32、1/12 列可满足（kda log Round 12）。

**代码位置**：激活侧 pad 在量化 pack 的**同一 launch** 内发射——
`quant_pack_float_fp4` 的 `K_valid` 分支写零 code+零 SF（quant_pack_float_ext.cu:194-204，
E2M1/SF 全零对 fp32 累加贡献精确 0.0）；fp8 对应 :157-164。权重侧 pad 在
`promix/quantize/weight_pack.py`（PackedWeight.K_low_pad/K_high_pad + `klow_pad_enabled`/
`khigh_pad_enabled` 门控）。GEMM 入口一律用物理宽度 K_low_pad/K_high_pad
（mixed_linear.py:379-384，_gemm 各 call site），逻辑切片/decode/marlin 仍用逻辑宽度。
注意 GEMM 入口 **没有** %128 的 TORCH_CHECK——对齐断言只有 SFVecSize 级
`K_lo%16==0 && K_hi%32==0`（codesign_fused_ext.cu:665 等）；%128 约束仅 fk 路有，且在
Python `_fk_cfg` 检查（mixed_linear.py:639-641）。

**收益**：K_low pad：70B down KH2368 1.16x、其余 1.02-1.03x，pack 时间不变、int16 逐位
相等（kda log Round 3）；K_high pad：解锁 fk 后相应列 gap 1.28-1.36→1.02-1.15
（kda log Round 12 + tracker KDA-3）。**窗口**：klow pad 自动开于 SM≤72；khigh pad 仅
FK 窗口形状（weight_pack 门控）。serve 产出器（qpfm/hpk/gather/norm）尚未发射 pad，
激活 pad 时被守卫绕开（mixed_linear.py:565-566, :947-948；ring512 分配 KH%128==0 不触发）。

### 2.7 scheduler raster/swizzle heuristic（SM 数 + L2 窗口自适应，v1/v2）

**机理**：CUTLASS StaticPersistentScheduler 默认（Heuristic raster、swizzle 1）在 188-SM
Pro6000 上隐式调优，48-SM LPDDR GB10 上大 grid tile 顺序 thrash DRAM/L2——M=8192 大形状
崩到 0.2-0.6x vs FP8（kda log Round 0/2）。修复为 host 端启发式 `cd_apply_sched_env`
（codesign_fused_ext.cu:385-415，单一定义、bias ext include 复用，共 5 个入口生效）：
- 仅当无 `CD_SCHED_SWIZZLE/RASTER` env 覆盖且 `cd_sm_count() <= 72`（:389）才启用 AUTO
  （Pro6000 不受影响）。
- v1（Round 2）：默认 swizzle=4；大 grid 大权重点升 swizzle=8。
- v2（Round 8）：tile-parallel 分支 key 改为 `nkw = N*(K_low + 2*K_high)`（MXFP8 按 2 字节
  计权，:394-402；legacy 巨权重分支保留旧 `N*K_low` key）；新增 **L2 窗口上限**：swizzle-8
  组流 `8*128*(K_low/2+K_high)` 字节 > 24MiB（GB10 L2）则回退 4（:403-409，边界实测
  ~22MB 赢 s8 / ~25.7MB 翻 s4）。raster 仅 env 可强制 AlongM/AlongN（:411-414）。

**收益**：schedule-only、bit-exact（kda log Round 2 gate）；M=8192 崩溃消除——70B qkvo
13236→3669us（3.5x），8B down 0.79→2.02x vs fp8（Round 2）；v2 把 r≥3/4 down 列 gap
1.52→1.14（Round 8 表）。**窗口**：GB10（SM≤72）全 GEMM 路 per-launch 常量，capture-safe。

### 2.8 small-M tile 配置变体（cd_smallm_ext）

**机理**：shipping 128×128 在 GEMM-routed decode/chunked-prefill（M=8..512）只铺 N/128 个
CTA，离 weight-traffic 下限 2.5-5x（cd_smallm_ext.cu:1-13）；128×64 tile 翻倍 CTA 数。
CUTLASS 交织 SF atom 只依赖 SFVecSize 不依赖 tile，故全部 cfg 共享同一 prepacked
physSFA/physSFB（:23-29）——换 tile 零重打包成本。cfg0-3 = <128,128,4,2>（=shipping
对照）/<128,64,4,2>/<128,64,6,2>/<128,64,6,3>（:91-102, :215-218），入口
`fused_gemm_prepacked_cfg` 按 cfg switch（:220-241），无内置启发——窗口全在 Python 实测表。

**收益/窗口**：Pro6000 `_SMALLM_TABLE`（mixed_linear.py:77-84，cfg1 恒选，1.05-1.63x，
窗口符合 CTA-wave 算术但编码实测值）；GB10 重测后窗口大幅收缩（48 SM 下 128×128 已填满，
cfg1 在 M≤64 反亏 30-70%），仅存 2 条 cfg2/cfg3 窗口（`_SMALLM_TABLE_GB10`
mixed_linear.py:128-131；kda log Round 5）。

### 2.9 split-K / stream-K 变体（cd_splitk_ext + codesign_splitk_kernel.hpp）

**机理**：M≤16 时串行 K mainloop（24-64 个 k-tile）是结构性延迟地板。fork glue 直接实例化
`PersistentTileSchedulerSm90StreamK`（codesign_splitk_kernel.hpp:123，绕过 SM120 selector
会选的 CLC 动态调度），把 NV+MX 合并为 **union k-tile 域**：调度器看到的 K =
`(nvT+mxT)*blk_k`（:205-213），设备端 `phase_ranges` 把每个 work unit 的 k 区间映射回两相
子区间（:478-487），NV 从 nv_start、MX 从 mx_start 起（:511-531，不同于 fused 恒从 0）。
归约用调度器自带 workspace fixup（:609）+ 保留 StreamkBarrier，Deterministic 模式
（cd_splitk_ext.cu:179）。特例：纯 MX work unit 无 ClearAccum=true 的 NV 相，glue 手动
`clear(accumulators)`（codesign_splitk_kernel.hpp:586-590）。入口参数 (cfg, splits, decomp
∈{heuristic,splitK,streamK,dataParallel})（cd_splitk_ext.cu:178-186, :220-240），
dual-phase only（`TORCH_CHECK(K_hi>0)` :227）。

**数值**：FP32 部分和归约顺序不同 → 非 torch.equal、cosine~1（mixed_linear.py:101-105）。
**收益/窗口**：Pro6000 8B down M8-16 splits=4 达 1.48-1.66x、M32-512 stream-K 1.07-1.35x
（`_SPLITK_TABLE` mixed_linear.py:106-110）；GB10 窗口另测（`_SPLITK_TABLE_GB10`
:221-226；kda log Round 5）。split-K 在窗口内优先于 small-M 表（_gemm :715-726）。

### 2.10 fk_smallk：整 tile smem 输出暂存 + bulk TMA store 跨 tile 在飞 + per-call walk

**机理**（ForgeKernel 进化合入，kda log Round 10）：SASS 级 recon 证明短 K 形状的失效模式
是 **输出写与 mainloop 串行化**——fp16 输出占强制流量 81-87%，CUTLASS cooperative epilogue
在下一 tile mainloop 前把 TMA store 排空到 ≤1 outstanding（fk_sm121_recon_report.md §3；
single-phase K=128..512 平坦在纯写时间 0.685ms）。fk kernel 的两点结构改动：
1. **整 128×128 fp16 tile 先 stmatrix.x2 进 smem staging**（STAGING 32KB，
   fk_smallk_ext.cu:72-74；写入 :558-575），对 8 个 64×32 子块发
   `cp.async.bulk.tensor.2d...bulk_group` store + 一次 `commit_group`（:581-588，封装
   :174-177），**仅在下一 tile 的 epilogue 入口**才 `cp.async.bulk.wait_group.read 0`
   （:552-556）——一个 bulk store group 跨 tile 边界在飞，写流与下一 tile mainloop 重叠；
   kernel 收尾 `wait_group 0`（:590-592）。
2. NV 3-stage / MX 2-stage 流水（:69-70）+ k-step j+1 fragment 预载跨 tile/段边界
   （:459-497）。
宏结构与生产 kernel 同构：持久 48 CTA、384 线程 = 1 producer WG + 2 MMA WG
（:38-39, :65-67, launch_bounds :314）；NV/MX 相间 named barrier 交接（:180-184）。
**per-call tile-walk**（KDA-3 Round 13）：GemmParams 带 `swz`/`along_m`（:109-110），
`tile_coords` 线性 id→(mt,nt)（:115-122），`run_prepacked` 每次调用可覆盖（:777-778,
:819/:834）；`run_prepacked` 直接消费生产路的物理 SF buffer，稳态单 launch、可 graph
capture（:811-835）。硬约束 K_high%128==0 等六项在 Python `_fk_cfg` 检查
（mixed_linear.py:639-641）。

**收益**：合入时 e2e geomean 1.146x；worst-list 点 1B qkvo M2048 gap 1.21-1.29→0.98-1.04、
1B gate_up M8192→1.03-1.16（kda log Round 10）；**Round 13 翻案**：Round-10 全部"fk 亏损"
排除点系默认 s1/AlongM walk 伪象，换 walk 后全部反转为赢（3B down M8192 0.30x→1.05-1.29x），
v3 曲线 GB10 中位 gap 1.081→1.039、worst 1.232（kda log Round 13；tracker KDA-3）。
cosine 每点 +1.2e-6..+1.6e-6 **优于** shipping（其 epilogue 无 bf16 round-trip）。
**窗口**：`_FK_TABLE_GB10`（mixed_linear.py:165-197）11 组 (N,K)，M∈{2048, ≥8192}，
kh_max 分带兼作 walk crossover。

### 2.11 fk_spg：r=0 列的单相 ForgeKernel（新，未提交）

**机理**：双相 fk 要求 K_high≥128，覆盖不了 r=0 列（KDA-3 residual #3）。fk_spg =
Single-Phase GEMM：纯 NVFP4、4-stage ring、无相切换（fk_spg_ext.cu:1-9, :47），沿用
整 tile smem 暂存 + 在飞 bulk store（:355-407，TMA store :396-402）；walk 参数改为
`swz_w`（m-block 组宽）+ `serpentine`（蛇形 n 方向）（:209-230）；`run_prepacked` 直读
物理 32_4_4 SF 布局（:232-240, :501-520）。约束 M/N/K 均 %128（:443）。

**收益/窗口**：`_FK_SPG_TABLE_GB10`（mixed_linear.py:209-219）7 组形状 1.05-1.27x
（eager+graph 双赢规则，kernels.py:177-192 注释：评估 run 确认 e2e 1.07-1.19x）。
正在收尾（fk_spg_ext.cu 为工作区未提交文件）。

### 2.12 producer 融合（RMSNorm / Hadamard / gather 与量化打包单趟）

共性：所有 producer 输出与 `cd.quant_pack_*` 位一致的 packed codes + CUTLASS 交织 SFA
（atom layout 经 stock CollectiveBuilder 取得，quant_pack_float_ext.cu:49-75）；per-block
scale 数学统一——NVFP4 block-16 E4M3 round-UP scale（`e4m3_ceil`，
quant_pack_float_ext.cu:106-111, :209-211）、MXFP8 block-32 UE8M0 ceil(log2) 精确
（:168-177）；可选 act-MSE scale 搜索编译期注入（act_mse.cuh:47-55，只改 scale 不改
code 写出 :5-7；各 producer 在 `#if PROMIX_ACT_MSE` 下调用）。

| producer | 融合内容 | 位置 | 状态/收益 |
|---|---|---|---|
| quant_pack_float（qpf） | float→codes+交织 SF 单 pass，row_stride/col_off 原位读列段，K_valid 零填充 | quant_pack_float_ext.cu:147-164, :194-204, :258/:281 | 生产必载；替代 93-98% e2e 的 torch 量化（mixed_linear.py:343-349 注释） |
| quant_pack_fused（qpfm） | 一次 launch 同产 NVFP4 低段+MXFP8 高段（grid.z 选段），+GATHER 变体把 o_proj 静态列重排折进 load | quant_pack_fused_ext.cu:120-201, :143-144, :249-254 | pack launch 减半；gather 省独立 kernel + fp16 [M,K] 往返（vllm-perf r3 lever 3） |
| hadamard_pack / hadamard28_pack | SwiGLU + 在线 Hadamard（8192 二幂 / 14336=28×512 分组+had28 常量矩阵）+ pack，fp32 只驻 reg/SMEM | hadamard_pack_ext.cu:1-14, :330-449；hadamard28_pack_ext.cu:97-105, :234-260, :342-344 | SwiGLU 版由 `forward_swiglu_pack` 用（mixed_linear.py:811-834）；无 SwiGLU 版实测**负收益**默认关（FHT 专用 fp16 kernel 快 ~4x，kernels.py:195-209） |
| add_rmsnorm_pack（arn） | residual add + ATen LlamaRMSNorm + pack 单 launch | add_rmsnorm_pack_ext.cu:1-41, :216-315 | promix 参考版；vLLM 路被 vrq 取代（ATen reduction 顺序 cos 0.9889 不过门，vllm_rms_quant_pack_ext.cu:14-18） |
| vllm_rms_quant_pack（vrq） | vLLM fused_add_rms_norm **位精确复刻**（vendored _f16Vec/cub BlockReduce/block-size 规则）+ pack，另出 fp16 xn 供 GEMV 路 | vllm_rms_quant_pack_ext.cu:17-31, :57-126, :262-282, :372-376 | serve A/B 实测 decode 净负（inductor 原生融合更强），默认关（kernels.py:121-138） |

### 2.13 decode GEMV / Marlin 域（M=1..48 带宽型路）

**机理**：M=1 时融合 TC kernel 利用率 ~1/128、仅 1-14% HBM 带宽（decode_gemv_ext.cu:1-21），
改为权重流式 GEMV：每行一 warp/block 流读 packed 4/8-bit 权重、in-register 反量化
（fp4 查表 :35-39、fp8 硬件 cast）、fp32 累加、`__shfl_down_sync` warp 归约（:223-225 等）。
域内独立优化：
- **W4A16 模式**（默认）：decode 带宽受限，不量化激活、只反量化权重——严格更准
  （mixed_linear.py:37-43；kernel :498-559）；W4A4 为 legacy 单 launch 融合量化版
  （quant_act_to_shared :228-300）。
- **批量 mb 变体**（M=2..16，vLLM captured decode batch）：warp-per-row + K-tiled shared
  暂存 + float4 LDS.128（:601-770）。
- **sf8 变体**：直接流 1 字节原始 scale（E4M3/UE8M0），g 在 kernel 内折，位精确，低段
  权重流省 ~18%（:812-832, :826-832）。
- **Marlin 配套**：高段 companion `gemv_w8a16_hi_add` 原位加进 Marlin fp16 输出
  （:1100-1234）；融合 `axpy_bf16_into_half`（g+cast+add 单 kernel，:1242-1281）。
  Marlin 低段 GEMM 本体在 `promix/vllm_integ/marlin_w4a16.py`（本目录外）。

**窗口**：M=1 单行 GEMV；2≤M≤16 批量 GEMV（per-shape 上限 mixed_linear.py:472-476）；
Marlin 窗口 M=2..48 优先于 GEMV（`_MARLIN_TABLE` :282-291，8B serve 实测修正 m_hi 的
诚实记录见 :260-273）。注意 Marlin/decode 窗口表仍为 Pro6000 实测，GB10 未重扫
（kda log Round 5 note）。

### 2.14 其余值得一提的点

- **prepacked SF + cached workspace + capture-safe host barrier**：权重 SF 一次性交织
  （prepack_sf_* 族，codesign_fused_ext.cu:470-510），GEMM 每调用零 SF scatter；stream
  capture 中跳过 `cudaStreamSynchronize`（:626-633）。SF 交织 pad 槽按需 `torch::zeros`
  确定性零填充（:425-431, :453-455，修 exporter 自检 flaky，git 3099103/2372c96）。
- **lm_head 纯 MXFP8**：W4 混合在未旋转 lm_head 上 PPL +0.295 不过门，W8A8 MXFP8 保持
  +0.048（lmhead_mxfp8_ext.cu:1-6），双侧 SF 预交织 + capture 保护（:140-145）。
- **config_dispatch.hpp**：round-24 网格搜索结晶的 if-ladder（classify N≥2K/K≥2N/方形
  :47-51；`select_codesign` M≤512 非 GATE→128×64 否则 128×128，K_high 参数被显式忽略
  ——config 最优近乎 r-independent，:112-120）。注意它**不做 SM 数切换**；per-SM 切换
  全在 Python 表 + cd_apply_sched_env。

---

## 3. 平台适配：SM120（Pro6000, 188 SM）vs SM121（GB10, 48 SM）

1. **运行时 SM 数门控**（代码中唯一的 per-SM 分支点有二）：
   - C++ 侧：`cd_apply_sched_env` 的 `cd_sm_count() <= 72`（codesign_fused_ext.cu:389）
     ——GB10 启用 swizzle AUTO，Pro6000 保留 validated 默认。
   - Python 侧：`_low_sm_device()`（mixed_linear.py:229-237）切换
     `_SMALLM_TABLE(_GB10)`/`_SPLITK_TABLE(_GB10)`，并门控 `_FK_TABLE_GB10`/
     `_FK_SPG_TABLE_GB10`/`klow_pad_enabled`（fk 与 pad 均 GB10-only）。
2. **窗口表不可移植的实证**：Pro6000 的 128×64 small-M 窗口在 GB10 反亏 30-70%
   （48 SM 下 128×128 已填满机器，kda log Round 5）——per-SM 实测表是必需品而非工程洁癖。
3. **编译**：GB10 侧 `TORCH_CUDA_ARCH_LIST=12.1a`（kda log Round 10 setup）；本地加载器
   默认 `12.0a`（kernels.py:48）；torch≥2.13（GB10 2.14）强制 `-std=c++20`
   （kernels.py:36-41，Pro6000 2.11 保持 c++17 使 build cache 字节稳定）。fk ext 需
   `-lcuda`（kernels.py:172），非 SM12x 工具链编译失败时静默返回 None 不影响主路。
4. **GB10 专属 heuristic 的物理依据**：24MiB L2（recon 实测确认）、LPDDR ~273GB/s、
   DRAM write 197GB/s（fk_sm121_recon_report.md）——L2 窗口上限与 castfold 的收益都
   直接由这两个数决定。
5. **遗留**：Pro6000 sq16384 全 r 抬高疑似同类巨网格问题，但 `sm_count<=72` 门把
   heuristic 挡在外面，待空闲机台复测（kda log Round 9）。

---

## 4. 调优方法论（KDA 循环 + gap-to-theory + ForgeKernel 进化）

- **KDA 循环**（kda_gb10_log.md Round 0-14）：每轮 = 假设 → 最小改动 → 探针矩阵 →
  位一致 gate（torch.equal / int16 视图逐位）→ 窗口化 shipping → 全曲线复测 + 单 launch
  profiler 复验。硬门（log 头部）：单 launch 保持、cosine_vs_fp32 不降。
- **gap-to-theory 验收**（Round 7 起）：`t_theory(r) = t_fp4*(K_low/K) + t_fp8*(K_high/K)`
  两点线性插值，gap→1.0 为目标；GB10 收官中位 1.039 / worst 1.232 / r=1 列 1.058
  （Round 13 v3 曲线），方阵 K≥8192 到理论线（16384 gap 0.990，Round 9）。
- **ForgeKernel 进化**（Round 10）：teacher GPU recon（SASS/CUPTI，NCU 被禁时降级）定位
  结构瓶颈 → student 以生产 kernel 为 source-mode reference 迭代（4 代留 1，e2e geomean
  1.146x）→ 合并时对齐物理 SF 布局使 `run_prepacked` 零拷贝接入。重要教训（Round 13）：
  进化 kernel 的"亏损排除点"必须先排除 schedule 伪象（tile-walk 参数化后全部翻案）——
  新 kernel generation 不是第一选项。
- 失败也入账并 default-off：hadamard_pack 全融合（kernels.py:195-209）、norm fuse
  （:121-138）、deep-stage（Round 4a/14）、fp16 直出（codesign_fused_fp16_ext.cu 数值
  不过门）。

---

## 5. 文件索引（37 个 .cu/.hpp，+1 个 .cuh）

| 文件 | 行数 | 职责 |
|---|---|---|
| codesign_fused_kernel.hpp | 612 | 双 phase 共享累加器串行 glue（生产 kernel 本体，§2.1/2.2） |
| codesign_sm120_bs_mma.hpp | 913 | block-scaled mma.sync collective fork（ClearAccum 位，§2.1） |
| codesign_bias_epilogue.hpp | 219 | castfold bias epilogue（LinCombBiasBFRoundHalf，§2.4） |
| codesign_splitk_kernel.hpp | 630 | stream-K/split-K glue（union K 域 + fixup 归约，§2.9） |
| codesign_overlap_kernel.hpp | 563 | 相间 overlap PoC glue（de-union smem，未 shipping，§2.3） |
| codesign_optc_kernel.hpp | 617 | Option-C 双累加器寄存器探针 glue（TwoAccum，round-23） |
| codesign_fused_ext.cu | 810 | 生产主入口：prepack_sf_*/quant_pack_*/fused_gemm_prepacked/single_nvfp4_* + swizzle heuristic（§2.7） |
| codesign_fused_bias_ext.cu | 176 | castfold 入口（r>0 双相 + r=0 单相，宏复用母本，§2.4） |
| codesign_fused_fp16_ext.cu | 17 | fp16 直出副本（数值不过门的反例，保留作证据） |
| cd_smallm_ext.cu | 253 | 小 M tile 配置变体 cfg0-3（§2.8） |
| cd_splitk_ext.cu | 252 | split-K/stream-K 入口（cfg/splits/decomp，§2.9） |
| codesign_overlap_ext.cu | 351 | overlap PoC harness（stage 扫描 + compat_report） |
| codesign_optc_ext.cu | 689 | Option-C 入口（ForkMXclear + fused_gemm_prepacked_optc） |
| codesign_nvfp4_ext.cu | 267 | STAGE-1 去风险：纯 NVFP4 走 stock glue + 累加器兼容 static_assert |
| config_dispatch.hpp | 515 | round-24 配置 if-ladder + launch_best_*（§2.14） |
| cd_cfg_ext.cu | 340 | `-DCDCFG_*` 宏参数化 config 网格搜索 harness（round-19） |
| config_dispatch_test.cu | 26 | config_dispatch 测试 TU |
| rd_cd_probe.cu | 15 | RD 诊断：显式暴露 128×128/128×64 两 tile 单测 |
| fk_smallk_ext.cu | 871 | ForgeKernel 双相 kernel（整 tile 暂存 + 在飞 bulk store + per-call walk，§2.10） |
| fk_spg_ext.cu | 537 | ForgeKernel 单相 NVFP4（r=0 列，swz_w+serpentine walk，§2.11，未提交） |
| decode_gemv_ext.cu | 1366 | decode GEMV 族（W4A16/W4A4/mb/sf8/hi_add/axpy，§2.13） |
| b1_fused_ext.cu | 1007 | 手写单 kernel 前身（cp.async 流水 + cfg 菜单 + WS cfg10） |
| b1_tma_ext.cu | 1466 | b1 的 TMA 进化（expect_tx、PERSIST/KPIPE/PINGPONG/SFILV 模板轴） |
| fused_gemm_ext.cu | 800 | pre-codesign 融合入口（两次 CUTLASS GEMM + combine，每调用 sync，对照用） |
| tile_probe_ext.cu | 800 | fused_gemm_ext 的 128×64 tile 副本（仅 :65/:114 两行不同，tile 对照探针） |
| tma_probe_ext.cu | 108 | TMA+mbarrier+swizzle smem 位一致 round-trip 连通性探针 |
| quant_pack_float_ext.cu | 317 | qpf：float→codes+交织 SF 单 pass + K_valid 零填充（§2.6/2.12） |
| quant_pack_fused_ext.cu | 297 | qpfm：单 launch 双段 pack + gather 折叠（§2.12） |
| hadamard_pack_ext.cu | 567 | SwiGLU+Hadamard(8192)+pack 融合 producer |
| hadamard28_pack_ext.cu | 496 | 同上，I=14336（28×512 分组 + had28 常量矩阵） |
| add_rmsnorm_pack_ext.cu | 474 | residual+ATen RMSNorm+pack（promix 参考版） |
| vllm_rms_quant_pack_ext.cu | 423 | vLLM 位精确 fused_add_rms_norm+pack（serve 版，默认关） |
| pure_presf_ext.cu | 214 | 单格式 full-K GEMM 基线（权重 SF 预交织，公平下界） |
| presf_cfg_ext.cu | 282 | pure_presf 的 `-DPSCFG_*` config 搜索版 |
| lmhead_mxfp8_ext.cu | 163 | lm_head 纯 MXFP8 W8A8 GEMM（双侧 prepack + capture 保护） |
| mxfp8_gemm_ext.cu | 194 | MXFP8×MXFP8→bf16 单格式基线（CUTLASS ex79c） |
| nvfp4_gemm_ext.cu | 193 | NVFP4×NVFP4→bf16 单格式基线（ex79a） |
| act_mse.cuh | 196 | producer 共享的 per-block NVFP4 scale MSE 搜索（编译期开关） |

---

## 6. 核实中发现的笔记/代码不一致（写论文前注意）

1. **adamix_code_design.md §4.2 行号已漂移**：fused_gemm_prepacked 写 :596（现 :662）、
   single_nvfp4_gemm_prepacked :674（现 :741）、prepack_sf_fp4_b :405（现 :470）——
   KDA 轮次后 codesign_fused_ext.cu 增长所致，本文行号以现状为准。
2. **phase barrier 注释笔误**：codesign_fused_kernel.hpp:381-382（及 optc 对应处）注释写
   "IDs 8/9"，代码实际 user id 6/7（effective 14/15，:385-386）。
3. **overlap 不是生产优化**：任务骨架把 codesign_overlap 列为核心融合家族，核实为
   round-20 PoC harness（codesign_overlap_ext.cu:1-12），mixed_linear 无路由。
4. **tile_probe_ext.cu 名不副实**：无任何 raster/walk 探测逻辑，是 fused_gemm_ext.cu 的
   副本仅改 ThreadBlockShape 128×128→128×64（tile_probe_ext.cu:65, :114）。
5. **"100KB SMEM 约束"口径**：kda log 写 100KB/block，代码实测口径为 ~101376 B/CTA
   （config_dispatch.hpp:21-23），量级一致、论文建议用 101376 B。
6. **128 对齐不在 kernel 入口强制**：GEMM ext 只断言 %16/%32（codesign_fused_ext.cu:665），
   %128 是 fk kernel 约束 + Python 侧 pad/路由共同实现（§2.6）——骨架中"K_low/K_high 128
   对齐"是生产策略而非入口硬断言。
7. **config_dispatch.hpp 无 SM 数切换**：per-SM 自适应全部在 cd_apply_sched_env（C++）与
   mixed_linear 窗口表（Python）；config_dispatch 仅按 (M,N,K) 分类且显式忽略 K_high。
8. **两级 scale 的第二级不在 mainloop**：per-block SF 在 collective 内消费，per-tensor
   g_w 全部走 epilogue alpha / GEMV scale 折叠（§2.5）——与"kernel 侧消费"骨架一致但
   位置需写准。
