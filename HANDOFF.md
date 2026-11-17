# HANDOFF — ProMix / PLAQuant 项目交接总纲

> 本文档是换机器 / 换人 / 换 Claude 时的单一入口。
> 写于 2026-11-17。实际密钥(SSH key / 服务器 token / sudo 密码)**不在本文档也不在 repo 里**,在私有迁移包 `plaquant-migration-<date>.tar.gz` 的 `credentials/` + `RESTORE.md`。
> ⚠️ `CLAUDE.md` 顶部的"远程服务器"段落是**旧的 H20/SM90 平台(已废弃)**。当前实际用的是下面两个平台(Pro6000 + Spark)。以本文档为准。

---

## 0. 一句话项目是什么

ProMix(旧名 PLAQuant):LLM 混合精度量化推理。PCA basis(U) + ring512 旋转 R1 + 实测-ΔD 误差驱动分配 + GPTQ + 两级 weight scale + act-MSE。核心创新 = **NVFP4(低位,E2M1 block16) + MXFP8(高位,E4M3 block32) 双格式单launch融合 GEMM**,发货比例 high_fraction = 1/8 (r=0.125)。目标:同强栈下 serving 真实 2-3× 加速,精度压 NVFP4、可调、与 SharQ 各有千秋(70B 高比例尾双指标反超 SharQ)。

---

## 1. 当前在飞的工作(换机后要续上的)

有**两个 standing loop**(每轮唤醒推进一步、报一句进度)。换机后让新 Claude 从迁移包的 transcript + 本节恢复:

### Loop A — Pro6000 巡检(rebuttal 期,保命 + 见缝插针)
每 tick:
1. **HARVEST 保命**:`export GIT_SSH_COMMAND="ssh -i $HOME/Downloads/id_ed25519 -o BatchMode=yes"`;对三个远程 worktree 各 `git fetch ssh://luofan@10.208.130.90/home/luofan/<dir> <branch>` → `merge --no-edit FETCH_HEAD` 进 `feat/rd-alloc` → `push origin`:
   - `plaquant-densesweep` : `feat/rd-densesweep`
   - `plaquant-accmatrix`  : `feat/rd-accmatrix`
   - `plaquant-qwenmid`    : `feat/rd-qwenmid`
2. **ssh nvidia-smi**:多卡 >40GB 被占 = 别团队在跑 → **YIELD**(只 harvest 不启动);有卡空闲(<2GB)→ 见缝插针补缺点。**绝不 kill 别团队进程**。
3. 每个落地点 **sanity-gate**(复现/单调/>bf16/无NaN/不离谱),合理才标 ✅+日期 到 `docs/results_tracker.md`。
4. 全齐 → 更新 `docs/vllm_final_fiveway.md §4` + `docs/paper_evidence_index.md`。

**Loop A 当前状态(2026-11-17)**:
- 70B 高比例尾(r0.75-1.00,7点)**已全落地**,反超 SharQ 结论坐实(见 results_tracker.md 末)。
- 在跑:70B 精度**低端补点 r0.05/0.10/0.20**(GPU0/1/2,CPU-offload,RAM 3并发上限) + **E3: 8B serving 补密 r0.09375/0.125/0.1875**(GPU6,`serve_point.sh`)。
- 关键脚本(容器 `luofan-plaquant` 内):`/home/luofan/lane70b.sh`(70B精度,幂等) · `chk70b.py`(查70B点完成度,"c4" key=完整) · `promix/scripts/densesweep/{acc_point,serve_point}.sh`。
- **venv 坑**:必须用系统 `/usr/bin/python`(TF4.48.1);`baselines/venv` 被别团队升级到 TF5.17 会崩 offload eval。`common.sh` 里 `VENV_PY=/usr/bin/python` 已 pin。
- **tf5.x 坑**:transformers≥5 的 decoder layer 返回裸 Tensor 不是 tuple → 需 `outs[j] = _layer_out[0] if isinstance(_layer_out, tuple) else _layer_out`(evaluator.py 已修)。

### Loop B — Spark/GB10 复刻
按 `docs/spark_gb10_replication.md` 的 4-track 推进。
- 访问:`ssh -p 42762 -i ~/Downloads/id_ed25519 luofan@111.229.66.102`,venv `/home/luofan/.venv`。
- 单卡,被别团队 sglang 占 ~60GB;correctness 共享卡,只有 B-profile/D-perf 需干净卡(flock -x)。
- **当前状态**:功能 4-track 全绿(环境✅ / kernel sm_121 cosine 0.999999✅ / bridge PPL 逐位✅ / C精度发货锚点 wiki 8.2167✅ / D e2e smoke✅)。剩:perf 需干净卡(B.Round1 KDA cfg grid+ncu、D.4 five-way serving)。

---

## 2. 怎么登陆(两个平台)

> 实际 SSH 私钥在迁移包 `credentials/id_ed25519`;放到 `~/.ssh/` 和 `~/Downloads/` 后 `chmod 600`。

### Pro6000(主平台,精度主实验)
- SM120, x86, 8×96GB 卡, 128 CPU核, 503GB RAM, CUDA12/13。
- 直连:`ssh -i ~/Downloads/id_ed25519 luofan@10.208.130.90`
- 工作在 **Docker 容器** `luofan-plaquant` 内:`docker exec luofan-plaquant bash -lc "..."`。容器若 Exited:`docker start luofan-plaquant`(数据/worktree 都在,2天前被OOM杀过一次,start即恢复)。
- 远程 git worktree:`/home/luofan/plaquant-densesweep`(feat/rd-densesweep)、`plaquant-accmatrix`、`plaquant-qwenmid`;主 `/home/luofan/plaquant`。
- **坑**:嵌套 `docker exec bash -lc "..."` 里 `echo` 带括号 `()` 会破坏解析 —— 避开括号。

### Spark(第二平台,GB10 复刻)
- GB10 Grace-Blackwell, **SM121(cap 12.1,不是 SM120,融合 kernel 要 sm_121 重编)**, aarch64/ARM64, 121GB 统一内存, 单卡。
- 访问:`ssh -p 42762 -i ~/Downloads/id_ed25519 luofan@111.229.66.102`
- passwordless sudo + docker 组。venv `/home/luofan/.venv`(torch2.14/cu130/sm_121)。系统 python3.12 无 torch。
- 共享机:别团队(yuqiuchu/donghanyuan/zhangshuoming/xuruiyuan)也在用,不硬抢卡。
- CN 网络:`HF_ENDPOINT=https://hf-mirror.com`。

---

## 3. 怎么构建 / 测试

### Kernel(在服务器,base 环境有 torch)
```bash
# Pro6000 SM120 / Spark SM121 —— setup.py 里改 arch (sm_120a / sm_121a)
cd kernels/mixed_gemm_l20   # (或当前 kernel 目录)
rm -rf build *.so
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python setup.py build_ext --inplace
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python benchmark.py
```

### PTQ 精度管线(用 .venv,有 fast_hadamard_transform)
```bash
# Step 0 PCA basis(~20min) → Step 1 旋转优化(可选,1B无效果) → Step 2 量化+eval(~3min)
# 见 CLAUDE.md "Build Commands" 节的 torchrun 命令(配置 promix/configs/*.yaml)
```

### 单元测试
```bash
pytest tests/test_mixed_gemm.py -v            # 需 gemm_data/ + GPU
pytest tests/test_mixed_gemm.py -v -k "q_proj"
```

---

## 4. 实验状态(论文 eval 章)

权威 spec = `docs/paper_experiments.md`。状态(2026-11-17):
- **E1**(MMLU+0-shot 新配方)✅ `rd_mmlu_report.md`
- **E2**(32B/70B 规模扩展)✅ `rd_scale_report.md` + 70B 高比例尾反超(results_tracker.md 末)
- **E3**(8B serving r-sweep 补密至6点)🔄 **正在 GPU6 跑(本次收尾)** → 跑完三档闭合
- **E4**(算子 Pareto 全曲线)✅ `rd_operator_pareto.md`
- **E5**(组件累计消融 build-up)✅ `rd_buildup_ablation.md`
- **E6**(3-seed 误差棒)✅ 同上
- **KV4** ✅ `rd_kv4_fix.md`

结果追踪总表:`docs/results_tracker.md`(精度矩阵 + serving 矩阵 + 验证记录)。
证据索引:`docs/paper_evidence_index.md`。

---

## 5. 分支结构

- **`feat/rd-alloc`** = 当前主集成分支(所有 harvest 汇入这里,push origin)。换机后在这个分支上工作。
- `main` = 稳定基线。
- 远程 worktree 分支:`feat/rd-densesweep` / `feat/rd-accmatrix` / `feat/rd-qwenmid`(在 Pro6000 上各自跑后台 campaign,controller 从 Mac harvest 进 rd-alloc)。
- 其余 `feat/rd-*` 为历史阶段分支(独有 commit 均已收编进 rd-alloc,可忽略)。
- 子模块:`project-resq`(参考实现)、`third_party/cutlass`(4.5)、`ResComp`/`comet-25`/`mixed_tensor`。clone 后 `git submodule update --init --recursive`。

---

## 6. 换机检查清单

见迁移包里的 `RESTORE.md`(逐步:装 Claude Code → 放配置 → 放 key → clone repo + init 子模块 → 恢复 transcript 让新 Claude 续上 → 重启两个 loop)。
