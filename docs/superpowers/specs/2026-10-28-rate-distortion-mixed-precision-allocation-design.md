# 率失真最优混合精度分配(实测 MSE + e2e cost) — 设计 spec

> 2026-10-28。ProMix 的**核心算法创新**:把"每个单元用 FP4 还是 FP8"表述成一个
> **率失真最优比特分配**问题——给定一个精度/性能预算,算法用**实测量化误差**和**真实
> 端到端成本**,为每个单元求率失真最优格式,自动反推出"用多少高精度、放哪些单元"。
> 取代"拍脑袋固定 1/8"。
>
> **旋转底座**沿用 `docs/superpowers/specs/2026-10-28-nested-r1-per-layer-allocation-design.md`
> 的**统一 64-环块对角 R1**(本 spec 取代那份的 §4 能量注水分配,旋转结构不变)。

## 1. 目标与底线(可行域)

- **精度底线**:每模型 PPL **≤ 当前 shipped r=1/8 uniform**(1B 14.56 / 3B 8.30 / 8B 6.60)。
- **低精度底线**:全模型平均高精度(FP8)比例 ≤ 1/8(平均低精度 ≥ 7/8)。
- **目标**:在可行域内,按选定的操作点(§8)最优化精度-性能权衡;默认追求"等精度下更快"。
- **不变量**:旋转共享、embedding bake 一次、零在线开销;down_proj 在线 Hadamard;分配**离线静态**(每模型一套,kernel 每层固定 K)。

## 2. 格式菜单(锁定两种)

- **低 = NVFP4**(E2M1,block-16,E4M3 微缩放,~4.5 bit);
- **高 = MXFP8**(E4M3,block-32,UE8M0 缩放,~8.5 bit)。
- **FP16 不入菜单**(本期):它需要融合 kernel 加第三个 phase/单独 launch,成本不划算且是明确的 kernel 工作量 → 列为 future(§10 风险/未来)。
- 两格式 = 融合 kernel 现成(NVFP4+MXFP8 两 phase 共享累加器),无需改 kernel。

## 3. 分配单元:64-通道环(原子)

- 旋转 `T = U @ R1`,R1 = 统一 64-环块对角(见旋转底座 spec)。量化发生在 `U@R1` 全转之后的值上。
- **R1 把环内 64 通道混死 → 环内不可再分,必须整环用同一格式**;且 64 满足硬件对齐(K_high%32=0、K_low%64=0)。
- → 环 = 分配/度量/量化的最小粒度:每环量一次误差、分一个格式。

## 4. 率失真形式化

每环 j、每格式 f∈{FP4, FP8}:
- **distortion** `D_f(j)` = 该环用 f 量化的**实测 MSE**(§5);
- **cost** `C_f(j)` = 该环用 f 的**真实 e2e 边际成本**(§6)。

拉格朗日:给定价格 μ,每环独立选 `argmin_f [ C_f(j) + μ·D_f(j) ]`。两格式下等价于:
```
环 j 选 FP8  ⟺  ΔD(j)/ΔC(j) > 1/μ
  ΔD(j) = D_fp4(j) − D_fp8(j)   (升级省掉的误差,收益)
  ΔC(j) = C_fp8(j) − C_fp4(j)   (升级多花的成本,价格)
```
扫 μ → 画出整条 accuracy-vs-cost Pareto。μ 是统一的"每成本换多少误差"的价格,也是单旋钮(§8)。

## 5. distortion = 实测量化 MSE(不是方差代理)

- **做法**:离线校准时,对每环取其 `U@R1` 转后的真实激活值,分别用 FP4、FP8 量化→反量化,各算 `MSE = mean((原值−反量化值)²)`,得 `D_fp4(j)`、`D_fp8(j)`。
- **为何不用方差代理**:`误差∝方差` 是一阶假设,**漏掉重尾/outlier**——方差中等但重尾的环,NVFP4 块缩放被 outlier 撑崩,实测 MSE 才看得出。实测 MSE 去掉假设、天然抓 clipping。
- **成本**:全离线、校准集上,每环每格式各量一次,分钟级;不训练、不多跑模型。

## 6. cost = 真实端到端边际成本(不是 GEMM 微基准)

- **不能用孤立 GEMM 延迟当 cost**(会重犯"GEMM 快但 e2e 不涨"的坑)。cost 必须反映**整网实测**的 e2e 影响。
- 一个环升 FP8 的 e2e 边际成本分两块,按部署场景加权:
  - **Prefill(TTFT,compute-bound)**:融合 GEMM 的计算增量 × GEMM 占 prefill 份额(从整网实测拿,含非-GEMM 开销);
  - **Decode(TPOT,bandwidth-bound)**:权重字节增量(FP4 4.5b→FP8 8.5b),decode 延迟 ∝ 总权重字节。
- `C_f(j) = w_prefill · compute_cost_f(j) + w_decode · byte_cost_f(j)`,`(w_prefill, w_decode)` = 部署场景权重(prompt 长/生成长决定),是**部署输入**不是超参。
- **校准**:GEMM 峰值表(`docs/hardware_format_peak.md`:NVFP4 1469 / FP8 807 TFLOPS)只用来**初始化** compute 系数;最终以**整网实测**的 r-相关边际延迟为准。r-无关部分(attn/norm/embedding/固定开销)是常数,不影响 argmin。
- **两格式等环**:ΔC 近似常数 → 实操可先按 ΔD 单独排序(§7),cost 非线性(kernel tile 效应)作二阶校正。

## 7. 判定规则(单阈值切线)

1. 每环算 `ΔD(j)=D_fp4−D_fp8`(和 `ΔC(j)`,等环时近常数)。
2. 按 `ΔD/ΔC`(等环时即 ΔD)从大到小排所有环。
3. **单阈值 θ(=1/μ)切一刀**:上方 → FP8,下方 → FP4。
4. θ 的位置由操作点(§8)决定。
- **要点**:排序量是 **ΔD(升级省掉多少),不是原始 D_fp4**。FP4 很烂但 FP8 也救不了的环(ΔD 小)不值得升(本该 FP16,已砍 → 将就 FP8)。两格式下 FP8 通常能修好,故 ΔD≈D_fp4,近似"按 FP4 误差排"。

## 8. 操作点选择(同一条曲线,三种模式)

同一条 μ/θ 曲线上取点,底层都是"排序 + 一条切线":
1. **膝盖(默认,近无旋钮)**:取曲线拐点(marginal 收益/成本塌陷处)→ 精度近满、低精度区天然大、性能不差。**诚实:膝盖检测需定义规则(最大曲率/marginal 阈值),可能不是真·无旋钮;不 robust 时回退模式 2。**
2. **精度目标**:给可接受 PPL(或 Δ%),从高往低降级到刚好触线 → 可行域内最省点。
3. **性能目标**:给平均比特/延迟预算,找预算内精度最高点(= 旧 spec 的"平均 high ≤ 1/8")。
- 三种都受 §1 底线约束为**硬可行域**。

## 9. 效度风险 + 兜底(整个创新的命门)

- **核心假设**:最小化"总实测局部 MSE" ≈ 最小化 PPL。但局部 MSE 是**每层局部**的,误差对最终 PPL 的影响**各层不同**(早/晚层、attn/mlp 传播不同)→ 两者不严格等价。
- **验证**:先用局部实测 MSE,**实测它是否 track PPL**(在几个操作点上跑真 PPL,看 MSE-排序给的分配是否在同成本下 PPL 最低)。
- **兜底升级路径**:若不 track,distortion 升级为 **loss 加权**——用梯度/Hessian 代理(对最终 loss 的敏感度)加权每环 MSE,`D_f(j) ← sensitivity(j)·MSE_f(j)`。round-11 试过敏感度方向但在 R1-off 下;本设计 R1-on + 实测 MSE 为基,敏感度仅作兜底加权。
- spec 立场:**这是待验证的核心假设,不是既定事实**;对外讲方法时以"验证通过"为前提,先做这个数。

## 10. 验证 / 验收

1. **精度**:3B/1B/8B PPL ≤ 现状(同协议 `docs/baselines_on_hardware.md`,FP16 bridge 复现 13.16/7.82/6.14)。
2. **效度(§9)**:实测 MSE 排序给的分配,在同成本下 PPL ≤ "方差代理"与 "uniform" 两个对照 → 确认 MSE track PPL;否则启用 loss 加权。
3. **性能**:用整网实测 e2e(prefill TTFT + decode TPOT,按场景权重)对比现状 uniform 1/8,报告真实加速(不用孤立 GEMM 数)。
4. **预算/可行域**:平均 FP8 比例 ≤ 1/8,逐层/逐环 K 全硬件对齐(K_high%32、K_low%64)。
5. **对照组**:现状 uniform 1/8;uniform 1/16;方差代理分配;本设计(实测 MSE + ΔD)。本设计须 ≥ 最优 uniform 精度且平均 FP8 更少或等。

## 11. 落到哪些文件

- `promix/quantize/adaptive.py` — 取代能量注水:新增 `ring_mse(x_rot_ℓ, ring_size, fmt)`(每环每格式实测 MSE)+ `allocate_rd(D_fp4, D_fp8, C, mode, target)`(ΔD/ΔC 排序 + 按操作点定 θ)。
- `promix/quantize/basis.py` / `rotation.py` — 旋转底座(统一 64-环 R1)见另一份 spec;本 spec 不改旋转,只消费 `U@R1` 转后的激活做 MSE 度量。
- `promix/eval/ptq.py` — 接线:校准前向收每环转后激活 → 调 `ring_mse` 得 D_fp4/D_fp8 → `allocate_rd` 得逐层每环格式 → 按格式分组 reorder + 量化。
- `promix/quantize/cost_model.py`(新) — e2e cost 系数:峰值表初始化 + 整网实测校准 + 场景权重。
- `promix/configs/*mxfp8-nvfp4*.yaml` — opt-in `rd_alloc`(默认 off=现状 uniform,零回归)、`ring_size`、`mode`(knee/accuracy/perf)、`target`。
- `tests/` — 实测 MSE 正确性、ΔD 排序/阈值、硬件对齐、可行域、旋转一致性探针。
- kernel:**不改**(两格式两 phase 现成;每层变 K 已支持)。

## 12. 风险 / 未来

- **§9 效度**是最大风险(局部 MSE vs PPL)——已给兜底。
- **膝盖检测**可能偷带阈值 → 以"精度目标"为可靠默认。
- **cost 非线性**(kernel tile 效应)→ 用实测 cost 曲线而非单斜率。
- **FP16 tier(future)**:极端 outlier 环只有 FP16 能救;入菜单需融合 kernel 加第三 phase(明确工作量)+ 计入 launch 开销。由届时的 ΔD/ΔC 决定是否值得。
- **跨层 MSE 可比性**:实测 MSE 已比方差好,但不同层激活尺度仍可能偏置排序 → 若 §9 验证暴露此问题,并入 loss 加权解决。

## 13. 成功的样子(一句话)

把"用 FP4 还是 FP8"表述成**率失真最优分配**:每环的失真用**实测量化 MSE**(抓 outlier)、成本用**整网实测 e2e**(不被 GEMM 数骗),按单一价格 μ 独立求最优格式,单阈值切线、三种操作点选预算——于是"用多少高精度、放哪些环"由**误差和真实性能**决定,而非魔数 1/8;算法与格式解耦、跨硬件可迁移。
