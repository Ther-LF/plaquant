# B1 优化第一轮 — 报告(controller 收割版)

> 本轮由两个 agent 接力完成(多次基础设施 stream 中断,工作均落盘无损);最终数字由 controller 从 r3 日志收割。正确性 gate:**r3 版 cosine=1.000000 全 shape 全 M(NVFP4 阶段 bit-exact,rel=0)**。

## 演进:b1_single(naive)→ b1_mw(多warp)→ b1_mw_pipe → b1_opt_r1/r2/r3

本轮落地的优化:多 warp CTA tile、cp.async 流水线、ldmatrix(B 侧 SM75_U32x2_LDSM_N 适配 8B/线程 fragment)、K-loop 调优。r2→r3 的增量主要来自 ldmatrix/流水收尾。

## 成绩单(计算饱和区 M=2048,主指标 band:0%=纯FP4上界,100%=纯FP8下界;bounds 均 presf 同场)

| shape | 轮前 band | **r3 band** | r3 延迟 | 纯FP4 | vs presf |
|---|---|---|---|---|---|
| down_proj_1B | 71-74% | **42%** | 0.156ms | 0.129ms | 0.94× |
| down_proj_3B | 123% ❌ | **52%** | 0.218ms | 0.160ms | 1.06× |
| q_proj_3B | 133% ❌ | **62%** | 0.100ms | 0.078ms | 0.87× |
| up_proj_1B | ~127% ❌ | **84%** | 0.132ms | 0.088ms | 0.81× |
| gate_proj_1B | 127% ❌ | **95%** | 0.135ms | 0.090ms | 0.81× |

**底线目标达成:不及格四兄弟全部压回 band<100%**(此前比纯 FP8 还慢);down 系列已达 <55% 的本轮冲刺线。gate/up(84-95%)离上界仍远,是下一轮主攻。

decode(M=1/16):band 深负(-77%~-172%),B1 比 CUTLASS 纯格式 collective 快 2-3×(轻量 kernel 的 launch 优势;按负责人指令不再投入、不作卖点)。

## 新发现的诚实问题:M=256 中档塌陷

r3 暴露 M=256 是新弱区:q_proj_3B band **162%**、down_proj_3B **232%**(比纯 FP8 慢)。推测:tile/warp 配置为 M=2048 调的,中档 M 下波次/占用率失配。下一轮需 M-adaptive 配置或中档专用 tile。

## 剩余差距排序(下一轮弹药)

1. **gate/up_proj M=2048(band 84-95%)**:N=8192 大 N shape,vs 纯 FP4 还有 ~1.5×——ncu 对比(profile/b1_round1/gate2048_b1_r2.ncu-rep 等)指向流水深度/SMEM 带宽利用仍逊于 CUTLASS collective;
2. **M=256 中档塌陷(band 162-232%)**:M-adaptive tile 派发;
3. down_proj_3B 的 presf 对比仍 1.06×(唯一没赢 presf 的 M=2048 shape);
4. TFLOPS%峰值的系统化表格未完成(ncu SOL 数据在 rep 文件里,待下轮统一出)。

## 产物

- `b1_fused_ext.cu`(49.5KB,r3 版,本地远程已同步)+ `b1_fused.py`
- `pure_presf_ext.cu`(presf 版纯格式 bounds,干净标尺)
- benchmark.csv:+288 行(r2/r3 各 144);profile/b1_round1/*.ncu-rep
- 日志:b1_opt_r2.log / b1_opt_r3*.log(含逐 shape cosine PASS 记录)
