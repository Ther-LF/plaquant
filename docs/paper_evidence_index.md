# 论文证据索引(claim → 数据文件 → 关键数字)

> 2026-11-03。写论文时按 claim 查数据的总目录。全部为 RTX Pro 6000(SM120)实测,协议 = `docs/baselines_on_hardware.md`(bridges 13.16/7.82/6.14 逐位复现为前提)。

## 1. 精度 claims

| claim | 关键数字 | 证据文件 |
|---|---|---|
| **W4A4 同协议精度全场第一**(新锚点) | 1B **14.169**/3B **8.218**/8B **6.564**(wiki);C4 同号 | `rd_accuracy_tuning.md`;配方与 gate 在内 |
| 老锚点(发货 r=1/8,dense) | 14.472 / 8.268 / 6.588 | `rd_ron_report.md`、`rd_metrics_report.md` |
| vs 新 W4A4 对手(我们协议实测) | ARCQuant 8.54/6.81;SharQ 8.39/6.68;MicroMix 9.05/7.39;CoQuant 9.56(其 wa_cov 主张未复现) | `rd_newpeers_accuracy.md` |
| vs 老 W4A4 对手(同协议) | QuaRot 22.1/10.7/8.65;SpinQuant(学习)16.14/9.25;Atom 18.4/9.70;ResQ 15.98/8.71/7.06 | `scoreboard.md` 主对比组 + `/home/luofan/baselines/peers/` |
| 多指标:C4 同号、0-shot 噪声内(不 claim) | 1B C4 −0.129、3B −0.072;0-shot ±1.4-3.3pp | `rd_metrics_report.md` |
| 跨族(Qwen3-8B) | uniform gap +0.242(比 Llama-8B 的 +7.4% 健康);分配 break-even 落在 scaling 趋势上 | `rd_qwen_report.md` |
| 校准鲁棒(wiki↔C4 2×2) | 交叉格 ≤+0.12(~1%),排名不变;分配增益四格全活;我们的 ΔD 环节最鲁棒;U 子空间重叠 0.66-0.69、环选择 81.4%(随机 12.5%) | `rd_calib_robustness.md` |

## 2. 分配算法 claims

| claim | 关键数字 | 证据文件 |
|---|---|---|
| 实测-MSE 分配 > 方差代理 > uniform(命门 gate) | 8.3124 < 8.3924 < 8.4711(3B R1-off 同预算) | `rd_validity_report.md` |
| "uniform 近最优"被推翻(模型相关) | 1B −0.077 / 3B −0.065(ring512)推翻;8B/13B 持平(本质:冗余) | `rd_ron_report.md`、`rd_whydecay_report.md` |
| ε/gain 环宽分解(机制图) | S=64/128/256/512:net +0.012/−0.008/−0.031/−0.065(3B) | `rd_ron_report.md` |
| 增益随规模归零的机理(两段式) | 1B→3B=headroom 缩(gain/headroom 恒 ~6%);3B→8B=每环敏感度塌 6.2×(flip16 实验) | `rd_whydecay_report.md` |
| 悬崖 = outlier 集中;平滑 S 坡,膝 ~1/96 | top-1/32 环 = 42% ΔD(≈13× 均匀);1/256→1/32 恢复 13→86% | `rd_accpuzzle_report.md`;谱图 `rd_deltaD_spectrum.html` |
| loss 加权:诊断性(修 8B mispick 73%,不翻正;3B 反伤) | 8B +0.019→+0.005;3B +0.0066 | `rd_lossweight_report.md` |
| down_proj 全 FP4 实测最优(叙事防御) | 强塞等预算 +0.056 C4(=78% 分配收益);白送 +35% 预算=噪声 | `rd_down_fp4_control.md` |

## 3. kernel / 硬件 claims

| claim | 关键数字 | 证据文件 |
|---|---|---|
| Blackwell 格式峰值:FP4:INT8=1.49-1.80×;INT4=INT8(零通路) | NVFP4 1469 / INT8 814 / FP16 406;s4→IMMA.S8 SASS 证据 | `hardware_format_peak.md` |
| 我们融合 GEMM @M=8192 最快梯队 | vs vLLM-NVFP4 1.0-1.53×、SharQ 1.12-1.51×、MicroMix 1.4-1.85×、ARCQuant 平手、fp16 2.3-3.1× | `rd_baselines_m8192.md` |
| 融合的独立价值(vs 同算法朴素 2-launch) | 1.33×(3B)/1.28×(8B)geomean,随 M→1.7× | `rd_torchfp4_report.md` |
| INT4 系对手在 Blackwell 不可行 | QuaRot 慢 FP16 4.2-6.8×(所有 M);QServe 编不过;Atom s4→INT8 | `rd_largem_report.md`、peers RESULTS |
| 单 launch 双格式 kernel 达 CUTLASS TC 平价 | alu/tc≈1、ACTIVE 入 CUTLASS 带(co-design 分析) | `b1_opt_round13/14/15/16_report.md` |

## 4. serving claims(vLLM 0.32 同栈,FINAL)

| claim | 关键数字 | 证据文件 |
|---|---|---|
| **五方同栈终表**(BF16/FP8/W4A16/NVFP4/ProMix,b16-512) | ProMix 8.370/6.640 PPL;位置介于 FP8 与 NVFP4 | `vllm_final_fiveway.md` |
| 赢 BF16 全线 | decode 1.09-1.68×、prefill 1.26-1.82× | 同上 |
| **赢 weight-only W4A16-Marlin**(serving 区间) | prefill 1.3-2.0×;decode b≥128 1.03-1.24×;QServe 协议 c256 1.17×/1.34×(诚实:b1-64 decode 输 0.66-0.85×) | 同上 |
| 平 FP8 W8A8(大 batch,4.6-5.4bit vs 8.5bit) | 0.90-1.09 | 同上 |
| **r-sweep serving 曲线**(修正后干净) | 3B 6 点/8B 4 点全部精度压 NVFP4(连 4.62bit 的 r=1/32:6.817<6.87);decode 对 r 近乎全平(b256 18.1-19.1ms) | `vllm_final_fiveway.md` §4 + `rd_r14_anomaly.md` |
| **W4A4 论文方法 e2e**(reviewer 必查) | SharQ(自带 vLLM 集成,同栈):精度平 8.385,serving 我们 prefill 2.41×/c256 1.40×;MicroMix 同卡复现其论文 2.00× prefill 但 decode 慢 FP16 4.4×、PPL 9.045;Quartet as-shipped 不可运行(sm100 cubin),修后 RTN 9.62+ | `vllm_final_fiveway.md` §6 |
| 性能 loop 轨迹(R1-R5) | pm/nv decode 0.52→0.85-0.93(b256-512);精度净升 −0.32;4 个诚实负收益记录 | `vllm_perf_round1..5.md` |
| 结构地板分解 | 双段 GEMM 1.33×+act-MSE+Hadamard+host;离地板 ~4% | `vllm_perf_round5.md` §3c |
| 数值对齐方法论(可复现卖点) | T0 torch.equal / T1 bit-exact / T2 PPL\|Δ\|≤0.1 + 签名 ΔPPL gate,全程 byte 级 | `vllm_integ_phase1.md`、`vllm_final_fiveway.md` gates |
| 跨论文相对比(QServe 式口径) | 各家 vs 各自强栈 FP16,ISL/OSL 1024/512,锚点强弱注记 | `vllm_final_fiveway.md` 跨论文节 |

## 5. 待收尾(进行中)

- ring512 基座 → vLLM serving 导出(新锚点进 serving 表)— `feat/vllm-integ` 跑中
- KV4 16ch 修复 + W4A4KV4 公平行 — `feat/rd-kv4` 跑中

## 总定位

见 `positioning.md`(一句话 thesis、四件套、能/不能 claim、旋转防御、limitation)。
