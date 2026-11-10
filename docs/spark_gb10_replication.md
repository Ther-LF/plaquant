# ⏸️ SPARK 暂停(2026-11-11,用户主动关机)

> Spark box(111.229.66.102)由用户主动关闭,**暂停**。非故障。重启后一连上即可续跑(环境/模型/artifacts/JIT .so/staged 脚本全在盘;两 agent context 保留可 SendMessage 唤醒)。下方为暂停时的进度快照。

## Spark 复刻进度快照(暂停点)
| Track | 状态 | 产出/剩余 |
|---|---|---|
| **0 环境** | ✅ 完成 | torch2.14.0+cu130 / sm_121 / fht(源码编)/ vLLM0.33 / transformers5.20 / datasets 全验证 |
| **A 临时kernel** | ✅ 完成 | 融合 kernel 移植 sm_121,cosine **0.999999**(仅需 12.1a+c++20+python-dev)|
| **C 精度** | ✅ bridge gate 过 / ⏸️ 发货点在跑时被关 | **bf16 bridge wiki 7.8146 逐位复现锚点 7.8148**;默认配方量化 8.2592;发货配方(ring512/gptq512/..)复现 8.218 的 run 被关机中断,脚本`fire_ship.sh`+stats 已就位,重启重跑 |
| **B KDA kernel** | ✅ Round0 / ⏸️ Round1 待净卡 | oracle cosine 1.0;**关键发现:kernel 对 GB10 48-SM 失配(vs Pro6000 188 SM)=跨平台 config 重调洞察**;Round1 cfg 网格 poller 已挂,artifacts 在 gb10_kda/ |
| **D vLLM插件** | ✅ 移植+注册+kernel编译全通 / ⏸️ e2e smoke 待跑 | vLLM0.33 API 未破,entry-point 自动注册,codesign 融合 kernel sm_121 编译通过;smoke 脚本 `/home/luofan/smoke_promix.py` staged |

## 关机前的事故教训(重启后遵守)
GB10 单卡 **121GB 统一内存容不下并发双 GPTQ + sglang** → 曾把 unified mem 撑爆、sshd 饿死。**重启后重 job(GPTQ/export/smoke-perf)必须串行,一次一个**;correctness 可共享卡但也别叠双 GPTQ。

## 重启后续跑清单(串行)
1. Track C 发货配方 `bash /home/luofan/fire_ship.sh` → 量化 wiki 对 8.218;
2. Track D export(--no-gptq 功能版)→ e2e smoke(smoke_promix.py)→ T0/T1/T2;
3. Track B Round1 cfg 网格(等 sglang 走的净卡窗口)→ ncu → GB10 dispatch 重调;
4. 两 agent 可 SendMessage 唤醒(context 在)。

---

# Spark/GB10 复刻编排方案(kernel→算法→框架→实验,4-track 并行)

> 2026-11-09 定稿(brainstorming 产出)。目标:在 NVIDIA DGX Spark(GB10 / SM121 / aarch64)上**从头到尾复刻** ProMix 全套工作,作为论文的**第二硬件点**(跨平台泛化证据)。核心策略:4 条同等重要的 track 并行,用 agent team 推进,单卡 GPU 按"只有测量步才需独占"调度。

---

## 0. 平台事实(约束一切)

- **访问**:`ssh -p 42762 -i ~/Downloads/id_ed25519 luofan@111.229.66.102`(luofan 账户已建,passwordless sudo + docker 组)。
- **硬件**:GB10 Grace-Blackwell,**compute cap 12.1 = SM121**(≠ Pro 6000 的 SM120 → CUTLASS kernel 必须 sm_121 重编)。**aarch64/ARM64**(整栈要 ARM 构建)。**121GB 统一内存**(Grace+Blackwell 共享,无独立 VRAM)。20 CPU 核。3.7TB 盘。CUDA 13.0。系统 python 3.12.3 无 torch。联网通(pypi/github 200)。
- **单 GPU + 共用机**:只有 1 块 GB10;sudo 组里有 yuqiuchu/donghanyuan/zhangshuoming/xuruiyuan,当前被别团队 **sglang(Qwen3-27B)占 60GB/96% util**。→ 和 Pro 6000 一样要**让路**,无法硬抢独占。

**决定性洞察**:
1. **精度(PPL)是硬件无关的确定性 fake-quant 数学** → 不同硬件上数值相同。Spark 上 Track C 只需跑**少量点确认逐位复现**,全网格可引用已有结果(点数多少本文档不决定,留后)。
2. **只有 kernel perf(Track B profile)和 serving perf(Track D 吞吐)是硬件特有且需要"干净卡"的** → GPU 稀缺性只压在这两类"测量步"上。

---

## 1. 四 Track 架构 + 依赖 DAG

```
Track 0 — 环境底座 (全局唯一阻塞,最先串行打通)
  ARM-torch 装通 + GB10 冒烟(cuda.is_available/小GPU op)
  + clone 仓库 + 拉模型 + 跑通 bf16 bridge PPL(逐位对 Pro6000 锚点)
         │
   ┌─────┼──────────────────┬─────────────────────┐
   ▼     ▼                  ▼                     ▼
Track A         Track B(KDA)       Track C              Track D
移植kernel       从零构建kernel      算法/精度             框架/serving
(临时替代品)     (agent loop,真活)  (fake-quant,独立!)   (vLLM-ARM+插件)
   │             │ 独占卡做ncu/perf   basis→ringR1→         依赖 A 的临时kernel
   │             │ 慢、多轮探索        measured-alloc→       做功能联调
   │             │ B0→B1→B2→ncu闭环   GPTQ→两级scale→       │
   │             │                   actMSE→PPL曲线         │
   ▼             ▼                        │                ▼
   └─→ 下游(D)先用 A 跑功能 ───────────────┼───────→ e2e serving + 算子实验
              B 产出真 kernel 后 swap 掉 A ┘         (A→B 换装后重测 perf)
```

**同等重要,但角色不同**:
- **Track A = 临时解锁器**:现有 SM120 融合 kernel 移植到 sm_121,**只求能跑对(cosine≥0.999),不追极致**。让 Track D 不必等 kernel 调优就能开工。
- **Track B = kernel 本体工作**:用 **KDA(NVlabs/kda)agent loop** 为 sm_121 从零构建+逐轮优化融合 kernel,复刻"当初 kernel 优化那件事"。独立长跑,成熟后回灌 D。
- **Track C = 精度**:fake-quant,**完全不依赖 kernel**,环境一通即可全速跑,最快出"SM121 方法可复现"结论。
- **Track D = 框架/serving**:vLLM-ARM + 我们的量化插件;功能联调用 A,最终 perf 用 B。

---

## 2. GPU 优先级与调度(单卡,目标整体最快)

**原则**:GPU 是单卡瓶颈 → 把"需要干净卡"的工作量压到最小(只有 B-profile / D-perf),其余全部在"被别人占着的卡"上并发(统一内存可塞多个小 job)。

### flock GPU 锁(协调我们自己的 track)
```
锁文件: /home/luofan/gpu.lock
  B-profile / D-perf      →  flock -x  (独占)  最高优先,但只在"测量步"短暂持有
  A/C/D correctness/eval  →  flock -s  (共享)  自由并发;B 请求独占时礼让
```

### 调度规则
1. **不需干净卡的活 → 立刻全速并行**(即使别团队 sglang 占着卡):
   - Track 0 环境搭建、Track C fake-quant 复现、**Track B 的 KDA 生成/编译/分析阶段(纯 CPU+LLM,不占 GPU)**、Track A/D 的代码移植。
2. **稀缺"干净卡窗口" → 只给 B-profile 和 D-perf**,排队独占,挑别团队空闲时跑(单卡无法硬抢 → 挑时机 / 人工协调 / off-hours)。
3. **B 的 ncu 闭环里,GPU 独占只发生在 profile 那一步**(generate/compile/diagnose 都不占卡)→ 窗口之间穿插 A/C/D 的 correctness burst。
4. **让路纪律**(同 Pro 6000):别团队占卡时不硬抢;correctness job 可共享但接受变慢;perf 测量必须等干净窗口,否则数据作废(有干扰的 perf 不记)。

**净效果**:4 track 名义同等重要,但 B 的"独占特权"只在测量瞬间;长期是全并发。整体 wall-clock = max(各 track),而非 sum。

---

## 3. 各 Track 逐步任务

### Track 0 — 环境底座(串行,阻塞全局,~0.5 天)
- [ ] 0.1 venv + 装 ARM torch(`pip install torch --index-url .../whl/cu128`;cu13 若无轮子则试 NGC pytorch ARM 容器);`cuda.is_available()` + 小 GPU matmul 冒烟。
- [ ] 0.2 装 `fast_hadamard_transform`(ARM 源码编,需 nvcc on PATH:`export PATH=/usr/local/cuda/bin:$PATH`)。
- [ ] 0.3 clone 仓库到 `/home/luofan/plaquant`(从 GitHub origin,feat/rd-alloc 分支;需配 GH token 或从 Mac rsync)。
- [ ] 0.4 拉模型(Llama-3.2-1B/3B、Llama-3-8B;大模型后置)到本地 HF cache。
- [ ] 0.5 **bridge PPL 逐位复现**:bf16 原模型 wiki PPL 对 Pro6000 锚点(1B 13.16 / 3B 7.82 / 8B 6.14)—— 环境正确性的硬 gate。
- **Gate**:torch 见 GB10(cap 12.1)+ bridge PPL 逐位一致 → 解锁 A/B/C/D。

### Track A — 临时移植 kernel(下游解锁器,门槛低)
- [x] A.1 ✅ arch via TORCH_CUDA_ARCH_LIST=12.1a。原: setup.py 改 arch:`sm_120a`→`sm_121a`(或 `-arch=sm_121`);CUDA13 编译。
- [ ] A.2 修 aarch64/CUDA13 编译错(头文件/intrinsic 差异;CUTLASS 4.5 对 sm_121 的支持确认)。
- [x] A.3 ✅ correctness cosine 0.999999(sm_121,2026-11-10)。
- [ ] A.4 导出可被 vLLM 插件调用的 op(交付给 D)。
- **Gate**:cosine≥0.999 → D 可用;**不测 perf**(perf 是 B 的事)。

### Track B — KDA 从零构建 kernel(真 kernel 工作,agent loop)
- [ ] B.1 Spark 上 clone `NVlabs/kda`,装依赖(Claude-Code 驱动 + KernelWiki + ncu-report-skill);确认 ncu 在 GB10 可用(`ncu --version`,可能需 sudo/权限)。
- [ ] B.2 写 kernel spec 喂 KDA:NVFP4(low,E2M1 block16)+MXFP8(high,E4M3 block32)**单 launch 双 phase 共享 FP32 累加器**融合 GEMM,target sm_121,输入 shape = 我们的 LLM GEMM(q/k/v/o/gate/up/down,K_high/K_main 见 CLAUDE.md)。
- [ ] B.3 KDA 闭环:B0(正确性)→B1(朴素单 kernel)→B2(cp.async 流水+SMEM union)→ncu 驱动逐轮(Small-Tile-N / 寄存器 spilling / 小 M decode 路径)。**注意 SM121≠B200**:用 mma.sync 无 TMEM/tcgen05;metric 名多数通用,少数查 b200 对照表。
- [ ] B.4 每轮 gate:cosine≥0.999 且延迟单调降;GPU 独占仅 profile 步(flock -x)。
- **交付**:sm_121 融合 kernel,perf 达 FP16 cuBLAS 基准之上;回灌 D 重测。**与 A 的对照**:A 做 correctness oracle 校验 B 的数值。

### Track C — 算法/精度(fake-quant,独立于 kernel)
- [ ] C.1 跑 PCA basis(`basis.py`,CPU-offload 可行)→ ring512 R1 → measured-ΔD 分配 → GPTQ+两级scale+actMSE 全链(`rd_acctune.py`/`rd_scale.py`,与 Pro6000 同配方)。
- [ ] C.2 **数值复现确认**:Spark 上 1B/3B/8B 的发货锚点 PPL 对 Pro6000(14.169/8.218/6.564 wiki)—— 证明方法在 SM121+ARM+CUDA13 上逐位可复现。
- [ ] C.3 (点数多少 = **本文档不决定,留后**:全 20 点 grid 跑不跑在 Spark,取决于"是否要全在 GB10 实测"的说法,届时定。)
- **Gate**:锚点 PPL 复现(硬件无关,应逐位一致)→ 精度故事在 Spark 成立。
- GPU:共享卡即可(fake-quant 对干扰不敏感,只是慢)。

### Track D — 框架/serving(vLLM-ARM + 插件)
- [ ] D.1 装 vLLM(ARM;可能需源码编或 NGC);确认 GB10 上起得来。
- [ ] D.2 移植我们的量化插件(`promix/vllm_integ/`:plugin/ops/kernels/shared_g/export),ops 指向 **Track A 的临时 kernel**。
- [ ] D.3 功能联调:T0/T1/T2 gate(torch.equal / bit-exact / PPL|Δ|≤0.1);共享卡跑通 correctness。
- [ ] D.4 serving perf(TTFT/TPOT/吞吐):**等干净卡窗口**(flock -x),五方同栈(BF16/FP8/W4A16/NVFP4/ProMix)。
- [ ] D.5 **B 成熟后 swap**:ops 从 A 换成 B 的真 kernel,perf 重测。
- **Gate**:功能 T2 通过(用 A)→ 最终 perf 数(用 B,干净窗口)。

---

## 4. Agent-team 映射与并行

| 阶段 | 并行度 | agent 分工 |
|---|---|---|
| **P0** 环境 | 串行 | 1 个 bring-up agent 打通 Track 0(阻塞,必须先完成) |
| **P1** 主体并行 | **3-4 路** | Track C agent(精度,可再按 1B/3B/8B 分子 agent)‖ Track A agent(移植 kernel)‖ Track B = **KDA 自身的 agent loop**(独立 Claude-Code 跑在 Spark)‖ (Track D 待 A 的 A.3 gate 过后起) |
| **P2** 收敛 | 2 路 | Track D 用 A 跑完功能+初步 perf ‖ Track B 继续压 kernel;B 成熟→D swap 重测 |

**编排纪律**(沿用 Pro6000 教训):
- **不并行派两个改同一文件的写 agent**(A 和 D 碰 vLLM ops 接口 → 串行或明确分文件)。
- 每个 GPU 测量 job:detached + 每点/每轮 commit + 幂等(可中断恢复,别团队占卡随时被 kill)。
- HARVEST 保命:Spark 的产出同样 fetch→push origin(Spark 若无 GH 凭证,走 ssh-fetch→Mac push,同 Pro6000 模式)。
- B 的 KDA loop 作为**独立 agent 进程**跑在 Spark,我方 controller 定期巡检其产出/ncu 报告,不干预其内部循环。

---

## 5. 同步点与收敛

1. **Track 0 完成** → 广播解锁 A/B/C/D。
2. **A.3(cosine gate)完成** → D 可开工(有临时 kernel)。
3. **C.2(锚点复现)完成** → 精度故事成立(可独立于 B/D 先交付)。
4. **B 交付真 kernel** → D swap(A→B)→ 最终 perf 重测 → kernel+serving 故事闭合。
5. **全齐** → Spark 作为第二硬件点的 kernel/serving/精度 三块齐 → 并入论文(与 Pro6000/SM120 并列,跨平台泛化)。

---

## 6. 风险与待定

- **torch-on-GB10/cu13 轮子**:cu128 轮子未必认 sm_121;可能要 NGC ARM pytorch 容器或源码编。(Track 0.1 先验)
- **CUTLASS 4.5 对 sm_121 支持**:A/B 都踩;可能需升 CUTLASS 或打 sm_121 patch。
- **ncu 权限**:GB10 上 ncu 需 `--target-processes` + 可能 sudo/驱动计数器权限。(B.1 先验)
- **vLLM-ARM**:可能无现成轮子,源码编耗时。(D.1 风险)
- **单卡共用**:B-profile/D-perf 的干净窗口依赖别团队空闲 → wall-clock 不可控,需人工协调或 off-hours。
- **待定(用户明示留后)**:Track C 在 Spark 上跑多少精度点(全 20-grid vs 少量复现确认)——取决于论文是否要"全在 GB10 实测"的说法。

---

- **CN 网络**:huggingface.co 不可达,所有 HF 下载走 `HF_ENDPOINT=https://hf-mirror.com`;模型用 ungated `unsloth/Llama-3.2-1B|3B`、`unsloth/llama-3-8b`(配置里就是这些,无需 token)。


## ✅ Track A 完成(2026-11-10):sm_121 融合 kernel 移植成功
- `kernels/mixed_gemm_sm120/b1_fused.py`(b1_fused_ext.cu)**在 GB10/sm_121 编译通过**:仅需 `TORCH_CUDA_ARCH_LIST=12.1a`(12.0a→12.1a,nvcc 13.0 接受 `sm_121a`),+ 装 `python3.12-dev`(Python.h)+ cutlass 4.4.2@cb37157(github clone)。
- **correctness PASS**:N/K/M 多形状(2048×8192×2048、3072×3072×{1,16,2048})**fp4_presf=0.999999 / fp8_presf=0.999999 [PASS]**,BOUNDS PRESF ALL PASS,EXIT rc=0 → **cosine≥0.999 gate 过**。
- 结论:SM120 融合 kernel **几乎零改动**移植到 sm_121(arch flag + headers,无代码改动)。**Track D 的临时 kernel 就绪**;Track B(KDA 从零优化)继续独立推进 perf。
- CUDA 版本:torch cu128(12.8)+ 系统 nvcc 13.0,JIT cpp_extension **不卡版本不匹配**(与 fast_hadamard 的严格检查不同)。


## ✅ Spark 栈命门解除(2026-11-10):fht + vLLM 装成
- **fast_hadamard_transform** 源码编译成功(Dao-AILab repo,setup.py 原生含 `sm_121` gencode);卡点=torch12.8 vs nvcc13 版本检查 → patch `cpp_extension._check_cuda_version` 跳过(前向兼容,简单扩展 OK)。**bridge PPL + 全精度 eval 解锁**(rd_acctune 顶层 import fht,故 bridge 也需它)。
- **vLLM 0.33** aarch64 轮子装成(`import vllm` OK)→ Track D 解锁。
- 栈齐(torch 2.14 note):**vLLM 装机把 torch 升到 2.14.0+cu130**(原 2.11+cu128)→ torch 与系统 nvcc 都 CUDA13,版本一致更干净,不再需版本检查 hack。新栈实测:GB10 matmul OK + **fht 实际 hadamard 计算通过**(非仅 import)。栈齐:torch2.14/cu130/sm_121 + transformers5.20 + datasets5.1 + fht(functional) + vllm0.33。**注意:Track A 的 b1_fused .so 是 torch2.11 下编的,Track D 用前可能需按 2.14 重编**。待:3B 模型传完(~7.3G/12G)→ bridge PPL 自动发射(watcher 694166)。


## 🎯 Track C/0 Gate PASS(2026-11-11):bridge PPL 逐位复现
- **Spark(GB10/SM121/aarch64/torch2.14)3B bf16 bridge wiki = 7.8146** vs **Pro6000 锚点 7.8148** → **Δ=−0.0002(浮点舍入级,逐位复现)**。
- sanity-gate:复现✅ / 无NaN✅ / 合理✅。**证明 ProMix 方法+环境在 GB10 上逐位可复现**(精度硬件无关的实证)。C4 同轮计算中。
- 前置链(本轮逐个解除):模型(Pro6000→Mac→Spark 中转,12G)+ fht(源码编+torch版本检查patch)+ rotation artifacts(U/R/E deref 传输)+ wikitext-2(镜像下载)+ `--point` 参数 + HF_ENDPOINT 镜像(替 OFFLINE)。
- **Track 0 的 bridge gate(3B)通过** → Track C 精度在 Spark 成立。
- **补充(C4 同轮 BRIDGE_RC=0 完成)**:`--bridge` 连带跑了默认配方量化管线 → **量化 wiki=8.2592(dense/gptq128 轻配方,非发货配方)**,证明 GPTQ+ring分配+act-MSE **全链在 GB10 端到端跑通**(sel_rings 1288/10304)。**要复现发货锚点 8.218 需用发货 knobs(ring512+gptq512+percdamp0.025+weight_mse7+act_mse_up2)重跑**。bf16 bridge 7.8146 的逐位复现才是环境验证 gate(已过)。


## Track B Round-0(2026-11-11):KDA 发现 kernel 对 GB10 失配
- **KDA setup 完成**:NVlabs/kda skill 子模块(KernelWiki@76d27b5 / ncu-report-skill@d188794)初始化 + 链入 ~/.claude/skills;KDA 是 prompt/skill 驱动 workflow(非二进制),agent 亲自驱动闭环。
- **Round 0 baseline**:b1_fused oracle 在 sm_121 编译通过 + **cosine 1.000000 全 shape×M×cfg** → 有效 oracle + warm-start。
- **🔑 核心发现:kernel 对 GB10 失配** —— **GB10 仅 48 SM vs Pro6000 的 188 SM(3.9× 少)**,+1536 线程/SM、100KB smem/SM。此前 25 轮 auto-dispatch(cfg=0)是在 188-SM 上固化的,48 SM 上 tile/wave 假设全错(decode tile CTA 数 < SM 数=underfill;prefill tile wave-quantization 尾部浪费)。**纯 config 级优化目标(不动 mainloop)** → 这是跨平台泛化的论文级洞察:相同格式/算法,dispatch 需按目标 arch SM 数重调。
- Round 1(cfg×stages 网格扫 + cosine-gate + 中位计时 → GB10 最优 dispatch)已挂 **GPU-free poller**(wait_and_run.sh,sglang<40GB 时 flock-x 自动触发);ncu 就绪(sudo 计数器)。
- 当前让路:sglang 96%util/61GB 实际服务中 → perf/ncu 正确 yield,等干净窗口。artifacts: kernels/mixed_gemm_sm120/gb10_kda/。


## Track D 插件移植(2026-11-11):vLLM 0.33 移植实质完成(差 e2e smoke)
- **API audit**:vLLM 0.33 **未破** promix 插件 —— LinearBase/LinearMethodBase/register_quantization_config/parameter.* /direct_register_custom_op 全在,签名不变;仅 torch2.14 非致命 deprecation 警告。
- **插件导入+quant method 实例化 CONFIRMED**;`vllm.general_plugins` entry point `promix_quant` **pip -e 装好,vLLM 启动自动注册**(无需手动 import)。
- **kernel 路径 sm_121 编译通过**:成熟融合 kernel `codesign_fused_ext.cu`(W4A4 双格式:fused_gemm_prepacked/prepack_sf/quant_pack)**c++20+12.1a 干净编译** → 功能 bring-up **不需 b1_fused adapter**(真实管线直接 build;b1_fused 接口不同,deferred)。
- 改动:kernels.py(12.1a/c++20)、full_network_realkernel.py(预授权,12.1a/c++20)、pip install -e。
- **差最后一步 e2e smoke**(`/home/luofan/smoke_promix.py` 已 staged,vLLM offline LLM + fp16 + eager)——阻塞于 box 不可达,恢复后串行跑(先 Track C 释放内存)。
- **两 agent(B KDA / D 插件)均完成 box-independent 工作,parked 于 Spark 停机,context 保留可 SendMessage 唤醒续跑。**

## 7. 下一步

Track 0 先行(bring-up agent)。0.1 的第一步 = 装 ARM-torch 验证 GB10 可见 —— 这是整个移植的命门,上一轮正要跑被打断,环境一确认即展开 P1 三路并行。
