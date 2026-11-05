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
| NVFP4/ARCQuant/SharQ/MicroMix/CoQuant | **0-shot(实测)** | 3B/8B | accmatrix | ⬜ 未启动 | 待派 |
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

## 验证记录(controller 每次收割时追加)
- 2026-11-05 建表:上述 ✅ 行均已 `git show origin/feat/rd-alloc:<path>` 核过存在。
- 2026-11-05 peers-70b 收割:`git cat-file -e docs/rd_peers70b.md` PASS;SharQ 70B 领先全 W4A4 场、ARCQuant 70B 翻转劣于 NVFP4。
- 2026-11-05 opus协调收割:densesweep 自维持(8 GPU 满载,逐点 auto-commit);accmatrix BOPs join 落地(`git cat-file -e` PASS);实测 0-shot/MMLU/FP-Quant/old-four 未启动,待派;qwenmid 闭合未收敛(cos 0.9937)。_khigh_for 非bug(acc 点即 fakequant 参考,不走 kernel 路由)。
- 2026-11-05 qwenmid 闭合收割:14B cos 0.99999994 / 32B cos 1.0,argmax 全对;0.9937 是 fuse_layer_norms 均值中心化在随机小模型上的表象,真模型近似无损(1B 与 ResQ 逐位吻合)。两 bug 修复(basis.py tf5.x tuple、argparse 重复)。PPL 曲线 campaign 跑中(非伪造,JSON 待落)。
