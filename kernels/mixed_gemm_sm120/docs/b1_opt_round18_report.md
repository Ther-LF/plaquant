# B1 优化第十八轮 — 报告（把 activation-SF interleave 折进 quant/pack → 真实天花板下的公平 A/B）

> 全程 remote-only（10.208.130.90；luofan-plaquant 建/跑；RTX PRO 6000 Blackwell **SM120**，GPU 3 空闲）。
> 本轮**无任何 mainloop 逻辑改动**：`codesign_sm120_bs_mma.hpp` / `codesign_fused_kernel.hpp` / `b1_tma_ext.cu` / 全部受保护文件零改动。
> 只在 **pack/quant 层**新增一个融合 kernel（`cd_quant_pack_kernel` + `quant_pack_fp4/fp8` 入口，加进非受保护的 wrapper `codesign_fused_ext.cu`）+ 新 harness `r18_lat.py`。
> 正确性：fold 的 **buffer bit-exact 硬门**（ILV 输出 == r17 独立 interleave，逐 shape `torch.equal` 全 True）+ **24/24 cell cosine = 0.999999** 先过，再计时。
>
> ## 一句话结论（verdict）
> **把激活-SF interleave 折进「每次调用都要跑的 quant/pack 步」后，co-design 在真实天花板下近乎全胜。**
> r=1/8（生产点）**赢 5/6 shape**（qkv·8192 0.96、down·2048 0.95、down·8192 0.91、gate·2048 0.95、gate·8192 0.88），只输 qkv·2048（1.03，边际）；
> r=1/2 **赢 6/6**（0.90–0.97）。这比 round-17 的「只赢 down/gate·8192」**又翻过来三条**（qkv·8192 1.12→0.96、down·2048 1.05→0.95、gate·2048 1.03→0.95）。
> **诚实边界**：fold 不是**完全免费**——在同一个 pack kernel 里，写**交错**布局的 SF 比写**自然**布局贵一点点（散写 vs 合并写），这笔 `delta_q` 随 M×Kl 增长（低-Kl 的 qkv/gate ≈0–6us，高-Kl 的 down·8192 最多 +22us）。但它被折进**本就每次要跑的 quant/pack**，**没有单独的 interleave 发射**，且这点额外开销被 co-design 的 GEMM 优势在 5/6 shape 上盖过。唯一还输的 qkv·2048 是最小 GEMM、GEMM 优势最薄的一条。

---

## Part 0 — 本轮依据（round-17 留下的问题）

round-17 把 wrapper 的冗余 scatter 修成高效单遍 interleave 后，公平 A/B 翻盘为**条件 SHIP**：
- **GEMM-only 天花板**（activation SF 预交错）：co-design **赢 5/6**（0.92–0.97×），只输 qkv·2048。
- 但 `cd_full`（真实 per-call = 高效 interleave **独立发射** + GEMM）还带着 **~9–20us 的残余 interleave launch**，只在大 GEMM（down/gate·8192）净赢，小 shape（qkv、M2048）仍输。
- round-17 明确指出：**要把天花板真正变成「免费」，得把 interleave 折进量化/pack 核**——量化已遍历每个元素产出 SF，可直接按交错布局 emit，省掉独立 interleave 发射。

本轮就做这件事，并做真实天花板下的公平 A/B。

---

## Part 1 — fold 怎么做的（pack 层，零 mainloop 改动）

### 新增：融合 quant/pack kernel（`cd_quant_pack_kernel`，`codesign_fused_ext.cu`）
真实推理里激活是**每次调用现量化/pack** 的——这一步本就遍历每个元素、产出 block SF。本轮新增一个 kernel 模拟这一步：
**一遍 over (row, block)**，每线程同时做两件事：
1. **data pack**：fp4 把 16 个 code 打成 8 字节（偶 k 低 nibble，与 `nv.pack_fp4` 逐位一致）；fp8 把 32 字节原样拷贝。
2. **SF 落位**：把该 block 的 SF 字节写到**自然连续**布局（`ILV=false`）**或** CUTLASS 交错 atom 布局（`ILV=true`，`layout(make_coord(r, b*VEC, 0))`）。

**关键**：两种变体的 **data pack 路径逐字节相同**，唯一区别是 **SF 的写地址**（一次连续写 vs 一次散写）。因此
- `ILV=true` 的输出喂给 `fused_gemm_prepacked` → **没有任何单独的 interleave 发射**，co-design 直达 GEMM-only 天花板；
- NAT vs ILV 的计时差 `delta_q` = **在已经遍历每个元素的那一步里，emit 交错 SF 相对 emit 自然 SF 的纯边际代价**——直接回答「fold 免不免费」。

### bit-exact（硬门，GATE A）
`torch.equal` 逐 shape 验证（M=2048）：
- fp4/fp8 的 **packed data** == `nv.pack_fp4(codes)` / `Ah_bytes`（ILV、NAT 两变体都 True）；
- **ILV 的 SF** == r17 已证的独立 interleave（`prepack_sf_*_a`）→ **fold 是布局置换，不改数值**；
- **NAT 的 SF** == 自然连续 SF。
全部 True（`FOLD_BITEXACT_PASS`）。量化**数学**仍走 torch（与 r17 同一条已 gate 路径，两核共享、在 A/B 中抵消），本轮只把 **pack + SF 落位**放进 CUDA 单遍核；所以 cosine 与 r17 一致（见 Part 2）。

### 产物（pack 层）
`codesign_fused_ext.cu` 新增：`cd_quant_pack_kernel<LayoutSF,VEC,PACK,ILV>`、`quant_pack_impl`、`quant_pack_fp4` / `quant_pack_fp8`（返回 `{packed_data, SF}`）。**所有 .hpp mainloop 零改动。**

---

## Part 2 — 真实天花板公平 A/B（`r18_lat.csv`，24 cell，warm，median-of-50，GPU 3 独占同会话）

两核都**从 quant/pack 步起步**（真实推理每次都要跑）：
- **SHIP** `b1_fused_tma_dual`（round-12 生产配置）消费 **NAT**（自然连续）SF，原生零 reformat。
- **CD** 消费 **ILV**（交错）SF（由同一 pack 核折出，**无单独 interleave 发射**）→ `fused_gemm_prepacked`（天花板）。
- `t_q_nat` / `t_q_ilv` = pack 核 NAT / ILV 两变体（lo fp4 + hi fp8）；`delta_q = t_q_ilv − t_q_nat`。
- **真实 e2e**（pack+GEMM 一个计时区、依赖串行无重叠）：`ship_e2e = pack_nat ; ship`；`cd_e2e = pack_ilv ; fused_gemm_prepacked`。
- excess vs `t_ideal(r)=t_fp4(1−r)+t_fp8·r`（同会话 CUTLASS pure-presf 端点）。

**全 24 cell cosine = 0.999999**（含 r=0 单格式 PoC）；`R18_LAT_ALL_COSINE_PASS`。

### 生产点 r = 1/8（关键）

| shape | M | cos | **delta_q** (fold 额外) | ship_gemm | cd_gemmonly | **ship_e2e** | **cd_e2e** | ideal | **cd_e2e/ship_e2e** | cd_gemmonly/ship_gemm |
|--|--|--|--|--|--|--|--|--|--|--|
| qkv  | 2048 | 0.999999 | **−0.13** | 57.55 | 62.83 | 77.89 | 79.92 | 63.39 | **1.03** | 1.09 |
| qkv  | 8192 | 0.999999 | 4.78 | 123.28 | 118.08 | 159.70 | 153.92 | 145.50 | **0.96** ✅ | 0.96 |
| down | 2048 | 0.999999 | 4.48 | 128.53 | 121.81 | 166.93 | 158.67 | 154.28 | **0.95** ✅ | 0.95 |
| down | 8192 | 0.999999 | 22.29 | 329.39 | 290.90 | 459.38 | 419.73 | 444.50 | **0.91** ✅ | 0.88 |
| gate | 2048 | 0.999999 | 0.00 | 124.82 | 117.86 | 141.52 | 133.90 | 110.24 | **0.95** ✅ | 0.94 |
| gate | 8192 | 0.999999 | 5.47 | 382.90 | 319.10 | 398.54 | 349.54 | 341.46 | **0.88** ✅ | 0.83 |

### r = 1/2

| shape | M | delta_q | ship_e2e | cd_e2e | **cd_e2e/ship_e2e** | cd_gemmonly/ship_gemm |
|--|--|--|--|--|--|--|
| qkv  | 2048 | −0.40 | 102.78 | 99.55 | **0.97** ✅ | 1.06 |
| qkv  | 8192 | 1.78 | 247.87 | 235.63 | **0.95** ✅ | 0.94 |
| down | 2048 | 0.93 | 260.54 | 247.39 | **0.95** ✅ | 0.94 |
| down | 8192 | 17.82 | 792.10 | 724.98 | **0.92** ✅ | 0.83 |
| gate | 2048 | 0.16 | 183.97 | 165.86 | **0.90** ✅ | 0.92 |
| gate | 8192 | 6.03 | 536.38 | 481.66 | **0.90** ✅ | 0.85 |

**r = 0（退化纯 NVFP4，co-design 走 single_nvfp4_gemm PoC，SF 内部交错，无 fold 天花板变体）**：cos=0.999999；fp4 fold delta_q = qkv2048 +0.08 / down8192 +23.15 / gate8192 +7.81（同 M×Kl 规律）。非生产点。
**r = 1（纯 MXFP8，K_low=0）**：co-design 无暴露路径，记 n/a（出厂核覆盖）。

### 读数
1. **fold 把 round-17 的「只赢 2/6」直接推到「赢 5/6（r=1/8）/ 6/6（r=1/2）」**：r=1/8 又翻过来三条——qkv·8192 (r17 cd_full 1.12→**0.96**)、down·2048 (1.05→**0.95**)、gate·2048 (1.03→**0.95**)；原本就赢的 down·8192 (0.95→**0.91**)、gate·8192 (0.94→**0.88**) 再拉大。只剩 qkv·2048（1.03）还边际输。
2. **真实 e2e 基本达到 GEMM-only 天花板**：`cd_e2e/ship_e2e`（0.88–1.03）与 `cd_gemmonly/ship_gemm`（0.83–1.09）同向、同量级。两者都把**共享的 pack 代价**计进分子分母，故 e2e 比值比纯 GEMM 天花板**略靠近 1**（稀释），但 co-design 的 GEMM 优势仍主导。
3. **excess vs ideal**：r=1/8，co-design e2e 的 excess = qkv·2048 +26% / qkv·8192 +6% / down·2048 +3% / **down·8192 −5.6%**（快过理想端点插值！）/ gate·2048 +22% / gate·8192 +2%。出厂核对应 +9%…+28%。大 GEMM（down·8192）co-design e2e 已压到理想线下。

---

## Part 3 — fold 到底免不免费？（诚实边界，`delta_q` 列）

**结论：不是完全免费，但接近，且被 GEMM 优势盖过。** 在同一个遍历每个元素的 pack 核里：
- 写**自然连续** SF = 一次合并写；写**交错 atom** SF = 一次**散写**（输出偏移按 atom 布局散列，不可平凡合并）。二者线程数、读模式、data-pack 全同，唯一差别是 SF 的写地址。
- 实测 **`delta_q` 随 fp4 SF 数（M×Kl/16）线性增长**：
  - 低-Kl（qkv/gate，Kl=1792）：M2048 ≈ **0us**（−0.13 / 0.00），M8192 ≈ **5us**；
  - 高-Kl（down，Kl=7168）：M2048 **4.5us**，M8192 **22us**（r=1/2 时 18us）。
- 这比 round-17 的「**独立** interleave 发射」残余（~9–20us，含 launch+sync 开销）更小/同量级，但更重要的是**它折进了本就每次要跑的 pack 步、没有额外发射**，并且**两核都付 pack**，所以在 e2e 里它只以「ILV 比 NAT 多的那点 `delta_q`」出现，被 co-design 的 GEMM 省时（down·8192 GEMM 省 38us、gate·8192 省 64us）净盖过。
- **唯一盖不过的是 qkv·2048**：GEMM 最小（cd_gemmonly 62.8 vs ship 57.6，天花板本就 1.09 输），fold 的 −0.13us 也救不回来 → e2e 1.03，出厂核保持最优。

> 方法学诚实：本轮 pack 核读 **uint8 codes**（不是 float），量化**数学**仍走 torch（两核共享、A/B 中抵消）。这让 pack 的绝对时间比「读 float 的真实量化核」略偏小，从而 **SF 写占比偏大**——即 `delta_q` 若有偏差是**偏大（对 co-design 保守）**。真实 float-in 量化核里 SF 只占总遍历的 1/16（fp4）/1/32（fp8），交错散写的边际占比只会**更小**，fold 只会**更接近免费**。所以本轮的 5/6 赢是**下界**。

---

## Part 4 — 判定与最终 per-shape 分派

| 判据 | 目标 | 实测 | 判定 |
|--|--|--|--|
| fold buffer bit-exact（ILV == 独立 interleave；data == torch pack） | 完全相同 | `torch.equal` 全 True（`FOLD_BITEXACT_PASS`） | ✔ |
| 双格式 cosine（fold 后重门，24 cell） | =1.0 | 0.999999（24/24，含 r=0 PoC） | ✔ |
| 激活-SF interleave 的 per-call 代价 | 折进 quant/pack、无单独发射 | 已折进；边际 `delta_q`=0–22us（随 M×Kl） | ✔ |
| **真实 e2e 延迟 ≤ 出厂核** | ≤1.0× | **r=1/8 赢 5/6（0.88–0.96），r=1/2 赢 6/6（0.90–0.97）**；只输 qkv·2048（1.03） | **◕ 近乎全胜** |

### **SHIP 判定：co-design 上位用于除 qkv·2048 外的全部 shape**

**最终 per-shape 分派（生产 r=1/8）：**

| shape / M | 推荐 kernel | cd_e2e/ship_e2e | 理由 |
|--|--|--|--|
| qkv · M2048 | **出厂** `b1_tma_ext` | 1.03 | 最小 GEMM，GEMM 天花板本就 1.09 输，fold 救不回 |
| qkv · M8192 | **co-design** `fused_gemm_prepacked`（+ 折入 pack） | **0.96** | fold 后翻盘（r17 曾 1.12） |
| down · M2048 | **co-design** | **0.95** | fold 后翻盘（r17 曾 1.05） |
| down · M8192 | **co-design** | **0.91** | 大 GEMM，e2e 已压到理想线下（excess −5.6%） |
| gate · M2048 | **co-design** | **0.95** | fold 后翻盘（r17 曾 1.03） |
| gate · M8192 | **co-design** | **0.88** | 最大净赢 |

> **r=1/2 时全部 6/6 切 co-design**（qkv·2048 也翻为 0.97）。r↑ → MXFP8 占比↑ → co-design 的 block-scaled 硬件 SF 优势越明显，赢面越大。

### 一句话整体 verdict
**在真实天花板下（把 activation-SF interleave 折进本就每次要跑的 quant/pack 步），co-design 近乎全胜——r=1/8 赢 5/6（4–12% 更快）、r=1/2 赢 6/6，唯一例外是最小的 qkv·2048（+3%，出厂核保持最优）。** round-17 的「条件 SHIP、只赢 down/gate·8192」收紧为「**除 qkv·2048 外全切 co-design**」。fold 并非字面免费（交错 SF 散写比自然 SF 合并写贵 0–22us，随 M×Kl），但折进每次调用的 pack 步、无单独发射，且这点代价被 GEMM 优势在 5/6 shape 上净盖过；本轮的赢面还是**保守下界**（pack 读 codes 而非 float，夸大了 SF 占比）。

---

## Part 5 — 对 round-17 的收口（诚实双向）

| round-17 结论 | 本轮修正 |
|--|--|
| 「条件 SHIP：只有 down·8192 / gate·8192 净赢，qkv/M2048 因残余 ~9–20us 独立 interleave 略慢」 | **把 interleave 折进 quant/pack 后**：qkv·8192 / down·2048 / gate·2048 **全翻为赢**；r=1/8 赢 5/6、r=1/2 赢 6/6。残余不再是独立发射，只剩 `delta_q` 边际差。 |
| 「GEMM-only 天花板赢 5/6（0.92–0.97），折进 quant 可兑现」 | **兑现**：真实 e2e `cd_e2e/ship_e2e` 0.88–0.96（r1/8，5/6）贴着天花板。 |
| 「能否降到零（免费 fold）未验」 | 已验：**不完全免费**，交错 SF 散写比自然 SF 合并写贵 `delta_q`=0–22us（随 M×Kl），但不改分派结论（5/6 仍赢）。 |

**净判定**：co-design 作为 **down/gate（全 M）+ qkv·8192** 的生产核（r=1/8），qkv·2048 保留出厂核；r=1/2 全切 co-design。GEMM 优势已在真实、每次调用都要跑的 quant/pack 折叠模型下兑现为端到端延迟优势。

---

## 产物
- **pack 层改动（零 mainloop 改动）**：`codesign_fused_ext.cu` 新增 `cd_quant_pack_kernel` + `quant_pack_fp4/fp8`（融合 data-pack + SF 落位，NAT/ILV 两变体，返回 `{packed, SF}`）。
- **公平 A/B harness**：`r18_lat.py`（fold buffer bit-exact 门 + 24-cell cosine 门 + delta_q + ship/cd GEMM-only + 真实 pack+GEMM 组合 e2e）。
- 数据：`r18_lat.csv`（24 cell，全 cos=0.999999，含 t_q_nat/ilv/delta_q + GEMM-only + e2e + 双向 excess）；日志 `logs/_r18_lat.log`（`FOLD_BITEXACT_PASS` + `R18_LAT_ALL_COSINE_PASS` + `R18_LAT_DONE` + `EXIT=0`）。
- **未改动（保护）**：`codesign_sm120_bs_mma.hpp` / `codesign_fused_kernel.hpp` / `b1_tma_ext.cu` / `b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` / `mixed_gemm.py` / `pure_presf_ext.cu` 全部零改动。
- 复用已缓存 .so（仅 codesign_fused_ext 因新增 pack 核重编一次）。
- `candidates.jsonl`：+2 条（fold bit-exact + delta_q 边界；真实天花板 e2e 赢 5/6（r1/8）/6/6（r1/2） + 最终 per-shape 分派）。GPU 3 已释放。
