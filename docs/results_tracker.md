# 实验结果追踪(serving + 精度)

> 建立 2026-11-05。每格状态:⬜ 待跑 / 🔄 跑中 / ✅ 完成(带已验证的结果路径)。
> 路径经验证 = 该 commit/文件在 origin 分支上确实存在(controller 已 `git show`/`ssh ls` 核过)。
> 四路后台:densesweep(feat/rd-densesweep)、accmatrix(feat/rd-accmatrix)、peers-70b(feat/rd-peers70b)、qwenmid(feat/rd-qwenmid)。

---

## 一、精度矩阵(方法 × 指标 × 尺寸)

指标:WikiText-2 PPL · C4 PPL · 0-shot 6任务 · MMLU 5-shot。协议 = `docs/baselines_on_hardware.md`。

### 已验证落地(✅)

| 方法 | 指标 | 尺寸 | 结果路径(已验证) |
|---|---|---|---|
| ProMix(新锚点) | wiki+C4 | 1B/3B/8B | ✅ `docs/rd_accuracy_tuning.md` |
| ProMix | wiki+C4 | 13B | ✅ `docs/rd_metrics_report.md` |
| ProMix | wiki+C4 | 70B | ✅ `docs/rd_scale_report.md` |
| ProMix | 0-shot | 1B/3B/8B/13B | ✅ `docs/rd_metrics_report.md` + `docs/rd_mmlu_report.md`(新配方0-shot)|
| ProMix | MMLU | 3B/8B | ✅ `docs/rd_mmlu_report.md`(54.12 / 62.85)|
| ProMix KV4 | wiki+C4 | 1B/3B/8B | ✅ `docs/rd_kv4_fix.md` |
| NVFP4 | wiki+C4 | 1B/3B/8B/13B/70B | ✅ `docs/rd_newpeers_accuracy.md` + `docs/rd_scale_report.md` |
| NVFP4 | MMLU | 3B/8B | ✅ `docs/rd_mmlu_report.md`(52.63 / 62.34)|
| ARCQuant | wiki+C4 | 1B/3B/8B | ✅ `docs/rd_newpeers_accuracy.md`(8.54/6.81…)|
| ARCQuant | MMLU | 3B/8B | ✅ `docs/rd_mmlu_report.md`(52.64）|
| SharQ | wiki+C4 | 1B/3B/8B | ✅ `docs/rd_newpeers_accuracy.md`(8.39/6.68…)|
| SharQ | MMLU | 3B/8B | ✅ `docs/rd_mmlu_report.md`(53.42）|
| MicroMix | wiki+C4 | 3B/8B | ✅ `docs/rd_newpeers_accuracy.md`(9.05/7.39）|
| CoQuant | wiki+C4 | 1B/3B/8B | ✅ `docs/rd_newpeers_accuracy.md`(9.56…)|
| QuaRot/SpinQuant/Atom/ResQ | wiki | 1B/3B/8B | ✅ `docs/scoreboard.md` 主对比组 |
| GPTQ/AWQ W4A16 / FP8 | wiki(+部分C4) | 3B/8B(+1B) | ✅ `docs/scoreboard.md` 参考组 |
| ProMix 精度曲线 | wiki+C4 多比例 | 1B/3B/8B | ✅(旧版6点)`docs/vllm_final_fiveway.md` §4 |

### 待补 / 跑中(🔄/⬜)—— accmatrix + qwenmid + peers-70b 负责

| 方法 | 指标 | 尺寸 | 负责 | 状态 | 结果路径(落地后填) |
|---|---|---|---|---|---|
| **精度矩阵 BOPs join**(理论+门后真实) | 全方法 | — | accmatrix | ✅ | `docs/rd_accuracy_matrix.md`(已验证;SharQ 70B 赢精度但 ~2× 真实BOPs)|
| NVFP4/ARC/SharQ/MicroMix/CoQuant | **0-shot(实测)** | 3B/8B | accmatrix | ✅(CoQuant-8B 1 格在跑)| `docs/rd_accuracy_matrix.md`(已验证)|
| MicroMix/CoQuant | **MMLU(实测)** | 3B/8B | accmatrix | ✅ | 同上 |
| NVFP4 | **0-shot+MMLU @70B(TP1)** | 70B | accmatrix | ✅ | 同上(0-shot 77.94 / MMLU 78.99)|
| MicroMix/CoQuant | **MMLU** | 3B/8B | accmatrix | 🔄 | 同上 |
| **FP-Quant/QuTLASS** | 全4指标 | 3B/8B | accmatrix | 🔄 | 同上 |
| ProMix | **0-shot+MMLU 多比例**(5点)| 3B/8B | accmatrix | 🔄 | 同上 |
| ProMix / NVFP4(+peers) | **0-shot+MMLU @70B(TP)** | 70B | accmatrix | 🔄 | 同上 |
| 老四样 | **C4+0-shot+MMLU** | 3B/8B | accmatrix | 🔄(低优)| 同上 |
| ProMix | **MMLU** | 1B/13B | accmatrix | 🔄 | 同上 |
| ARCQuant/SharQ | wiki+C4 | **70B** | peers-70b | ✅ | `docs/rd_peers70b.md`(已验证:SharQ 5.769/9.659、ARCQuant 6.183/10.049)|
| ProMix 精度曲线 | wiki+C4 | **1B/3B/8B/70B @ 20点(5%)** | densesweep | 🔄 自维持 | 1B 0.05-0.40 已落(wiki 15.07→13.91 单调);70B 曲线跑中(~12h);3B/8B 尾在跑 |
| Qwen 14B/32B 旋转闭合+架构解锁 | — | 14B/32B | qwenmid | ✅ | `docs/rd_qwen_midsize.md`(已验证:14B/32B 均 GO,cos 1.0,had68/had100 验正交)|
| Qwen 14B/32B PPL+曲线 | wiki+C4 多比例 | 14B/32B | qwenmid | 🔄 | 14B basis 重算中(~3.6h)→ chained campaign 跑 bf16+20点;JSON `/home/luofan/qwenmid_work/` 待收 |

---

## 二、serving 矩阵(方法 × 指标 × 负载)

指标:TTFT(prefill延迟↓) · TPOT↓ · 吞吐 tok/s↑ · 并发吞吐↑ · 最大可服 batch/KV 容量。
负载:prefill-heavy ISL/OSL 1024/128 · decode-heavy 128/1024 · 并发 1024/512@256 · **长上下文 ISL 2048/4096** · **batch→512** · **ShareGPT 真实分布**。

### 已验证落地(✅)

| 项 | 结果路径(已验证) |
|---|---|
| 五方同栈(BF16/FP8/GPTQ-Marlin/NVFP4/ProMix)× batch16-512 × prefill/decode/并发 | ✅ `docs/vllm_final_fiveway.md` |
| ProMix r-sweep serving(3B 6点/8B 4点,旧版)| ✅ `docs/vllm_final_fiveway.md` §4 + `docs/vllm_final_data/` |
| SharQ e2e(自带 vLLM 集成,同栈)| ✅ `docs/vllm_final_fiveway.md` §6 |
| 70B:NVFP4 单卡 vs BF16-TP2 | ✅ `docs/rd_scale_report.md` §5 |
| 跨论文相对表(QServe 口径,已修列名)| ✅ `docs/vllm_final_fiveway.md` 跨论文节 |

### 待补 / 跑中(🔄)—— densesweep 负责

| 项 | 状态 | 结果路径(落地后填)|
|---|---|---|
| ProMix serving 曲线 @ 20点(5%)× 1B/3B/8B/70B | 🔄 | `docs/vllm_rsweep_curve.csv`(待验证)|
| 长上下文 ISL 2048/4096 × batch512 × 全方法 + OOM 上限 | 🔄 | `docs/vllm_serving_extended.md`(待验证)|
| ShareGPT 真实分布吞吐 × 全方法 | 🔄 | 同上 |
| AWQ-Marlin serving / QuTLASS serving | 🔄 | 同上 / `rd_accuracy_matrix.md` |

---


## serving batch 对账(手册规定 vs 实际,2026-11-08)
手册(本文件 L59)规定:prefill 1024/128 · decode 128/1024 · 并发 1024/512@256 · 长上下文 ISL 2048/4096 · **batch→512** · ShareGPT。实际:
| 项 | 手册 | 实际落地 | 缺口 |
|---|---|---|---|
| 主五方 batch | 16-512 | **只 16/64/128/256** | ❌ **batch 512 待run**(主 JSON 无 b512)|
| ProMix 曲线 batch | ≤512 | ≤256 | ❌ b512 待run |
| 长上下文 2048/4096×b512 | 全方法 | **仅 3B 片段**(ext_{bf,fp8,nv,pm}3b) | ❌ **8B 长上下文全缺 + b512 待run** |
| ShareGPT 真实分布 | 全方法 | 3B/8B 零散(sgpt_*) | 🔄 部分,待补齐 |
| 最大可服 batch / OOM 上限 | 要测 | 未测 | ❌ 待run |
**诊断**:prefill 吞吐 PM/FP8 随 batch 单调升(3B b256=0.99、8B b256=1.12;b16→b256 均升)→ **"3B 不如 FP8"部分因 batch 没测到 512**(b512 大概率反超)。8B+大batch 是最占优组合却几乎没测。**卡空优先补 batch512(3B/8B prefill+decode)+ 8B 长上下文,而非补 1B serving。**


## 70B 高尾补点:resume 隐患 + 修正(2026-11-09)
- **发现**:feat/rd-alloc 里 `70b_r0.70.json`=wiki 5.796 但**无 c4**(旧 campaign killed 在 wiki↔c4 间);`70b_r0.80.json`=纯 stub(只 meta)。rd_scale **启动即写 meta stub**,wiki、c4 分段写,**c4 最后写**。
- **stub 守卫 bug**:lane 脚本原判据=`"wikitext" in d`,但半成品 r0.70 有 wikitext 缺 c4 → 不会被删 → acc_point `[ ! -f OUT ]` 误跳,c4 永缺。**正确判据 = `"c4" in d`**(完整点的最后字段)。
- **ssh 恢复后必做**:(1) 把 lane 守卫改成 `"c4" in d` 否则删;(2) 删 remote+local 的半成品/ stub(r0.70 无c4、r0.80 stub)强制重跑;(3) 本次见缝插针起的 4 lane(0.70/0.75/0.80/0.85)若已完成则 harvest 完整版覆盖半成品;(4) sanity-gate 看 0.70-0.85 是否跌破 SharQ 5.769(参照 r0.65 wiki 5.827 / bf16 5.316)。
- **当前阻塞**:远端 host 自 2026-11-08 ~11:3x 起 ssh port22 超时,跨夜未恢复;detached lane 命运未知(host 若重启则被杀,幂等重跑补缺)。

## 验证记录(controller 每次收割时追加)
- 2026-11-05 建表:上述 ✅ 行均已 `git show origin/feat/rd-alloc:<path>` 核过存在。
- 2026-11-05 peers-70b 收割:`git cat-file -e docs/rd_peers70b.md` PASS;SharQ 70B 领先全 W4A4 场、ARCQuant 70B 翻转劣于 NVFP4。
- 2026-11-05 opus协调收割:densesweep 自维持(8 GPU 满载,逐点 auto-commit);accmatrix BOPs join 落地(`git cat-file -e` PASS);实测 0-shot/MMLU/FP-Quant/old-four 未启动,待派;qwenmid 闭合未收敛(cos 0.9937)。_khigh_for 非bug(acc 点即 fakequant 参考,不走 kernel 路由)。
- 2026-11-05 qwenmid 闭合收割:14B cos 0.99999994 / 32B cos 1.0,argmax 全对;0.9937 是 fuse_layer_norms 均值中心化在随机小模型上的表象,真模型近似无损(1B 与 ResQ 逐位吻合)。两 bug 修复(basis.py tf5.x tuple、argparse 重复)。PPL 曲线 campaign 跑中(非伪造,JSON 待落)。
- 2026-11-05 accmatrix 实测收割(含 -f 误删→从 /tmp/accmatrix-push 恢复 57515e5):0-shot 3B/8B 全 peer + MMLU(MicroMix/CoQuant)+ 70B-NVFP4(单卡TP1,0-shot77.94/MMLU78.99)。排序 SharQ≥NVFP4≈ARC>MicroMix>CoQuant,全在 ProMix 下,与 PPL 一致。未做:ProMix-70B 0-shot(需多卡驻留,densesweep 占满)、FP-Quant、ProMix 多比例0shot/MMLU、old-four C4/0shot/MMLU、CoQuant-8B-0shot(跑中)。

## Sanity-gate 验证日志(跑完≠对)
- 2026-11-06 精度曲线 gate:**3B 20pt ✅**(严格单调、全>bf16、最大跳 0.002)、**8B 14pt ✅**(严格单调)。**1B ⚠️**:r0.75→0.80 非单调(13.681→13.703,+0.021,疑跨-lane 噪声)→ 已派同卡复测,复测前不标 ✅。70B 9/20 未齐,待补齐再 gate。
- 2026-11-06(续)gate 复验:**8B 曲线 20/20 ✅ 严格单调**(补点根因=GPU共享OOM,非recipe);**1B r0.75→0.80 反转经同卡复测=真实确定性 artifact(非噪声/非污染)**:345→373环 wiki+0.021,ΔD≠loss 的曲线实例,诚实写入(幅度0.02不改结论)。扩展serving(ShareGPT+长上下文)在落。**Qwen14B bridge 崩=Qwen3 head-dim reshape bug(40头×128),插件需修**,PPL 未落。
- 2026-11-06 70B 曲线 gate(10/20 有效,0.70/0.75 失败None 待重跑):**单调无违反**;已落 r0.35-0.65 全 < NVFP4(5.928),最好 r0.60=5.8245;**尚未追上 SharQ 5.769(差 0.055,曲线高比例端已趋平,预判追不上)** → 诚实:70B 精度让位 SharQ 是整条曲线结论,非单点;ProMix 全程压 NVFP4 + 可调 + serving 远快于 SharQ(各有千秋)。待补 r0.05/0.10/0.70-1.0。
- 2026-11-07 70B 曲线续(11点,单调✅):高比例段逼近 SharQ——r0.60=5.8245(+0.056)→r0.65=5.8271→**r0.70=5.7960(距 SharQ 5.769 仅 +0.027,仍在降)**。修正上轮预判:高比例 ProMix 可能追平/反超 SharQ,待 r0.75-1.0 落地定论(0.80 当前失败需重跑)。若 r≥0.8 跨过 5.769 → "70B 让位"结论推翻,变成"高比例可反超、但付更多bit"。
