# 率失真混合精度分配 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用"每64-环实测量化MSE + ΔD/ΔC率失真排序 + 单阈值"自动决定每个单元用 FP4 还是 FP8,取代固定 1/8;先以最便宜的方式验证"实测MSE排序是否track PPL"这个命门,通过后再建完整系统。

**Architecture:** 两个纯函数(每环实测 MSE;ΔD 排序+预算阈值)+ 一个验证实验(复用现有 R1-off 评测路径,证明 MSE-排序在同预算下 PPL ≤ 方差代理 ≤ uniform)+ 命门通过后才投入的 e2e cost 模型、三操作点、与 64-环旋转底座的整合。研究性管道,早期任务在本地纯函数 TDD,实验/整合任务在远端 Pro 6000 跑真 PPL。

**Tech Stack:** Python/PyTorch;promix PTQ 管道(basis.py/ptq.py/quant_utils.py/adaptive.py);远端 docker `luofan-plaquant`(torch 2.11+cu130,Pro 6000 SM120);pytest。

**Spec:** `docs/superpowers/specs/2026-10-28-rate-distortion-mixed-precision-allocation-design.md`(旋转底座见 `docs/superpowers/specs/2026-10-28-nested-r1-per-layer-allocation-design.md`)

## Global Constraints

- **精度底线**:3B/1B/8B PPL ≤ 当前 shipped r=1/8 uniform(14.56 / 8.30 / 6.60),同协议(`docs/baselines_on_hardware.md`:wikitext-2 test、bf16、`add_special_tokens=False`、2048 非重叠 141 chunk、token-pooled exp(mean NLL))。任何改动的 FP16 bridge 必须复现 13.16/7.82/6.14。
- **低精度底线**:全模型平均 FP8 比例 ≤ 1/8(平均 FP4 ≥ 7/8)。
- **硬件对齐**:任何分配结果逐层 K_high%32=0、K_low%64=0;分配单元 = 64 通道一环。
- **格式菜单**:只用 NVFP4(低)+ MXFP8(高);不引入 FP16(本期)。
- **零回归**:所有新路径 opt-in(默认关 = 现状 uniform,byte-identical)。
- **分配离线静态**:每模型一套,kernel 每层固定 K;不改 kernel。
- **远端同步**:本地改动先 `git push`,远端 `git pull`(HTTPS);GPU 跑用 detached `docker exec -d` + 日志文件,勿交互式长跑。

---

### Task 1: 每环实测量化 MSE(纯函数)

**Files:**
- Modify: `promix/quantize/adaptive.py`
- Test: `tests/test_rd_alloc.py`

**Interfaces:**
- Consumes: 现有 `promix/quantize/quant_utils.py` 的 `fake_quantize_nvfp4(W, ...)` 与 `fake_quantize_mxfp8(W, ...)`(对张量做量化→反量化,返回同形状反量化张量)。
- Produces:
  - `ring_mse(x_rot, ring_size, fmt) -> torch.Tensor`:输入 `x_rot`(M, d)=旋转后激活(或权重列),`fmt∈{'nvfp4','mxfp8'}`;返回 `(d//ring_size,)` 每环 MSE。
  - `ring_deltaD(x_rot, ring_size) -> torch.Tensor`:返回 `(d//ring_size,)` 的 `D_fp4 − D_fp8`(升级收益)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rd_alloc.py
import torch
from promix.quantize.adaptive import ring_mse, ring_deltaD

def test_ring_mse_shape_and_monotonic():
    torch.manual_seed(0)
    d, M, ring = 256, 512, 64
    x = torch.randn(M, d)
    mse_fp4 = ring_mse(x, ring, 'nvfp4')
    mse_fp8 = ring_mse(x, ring, 'mxfp8')
    assert mse_fp4.shape == (d // ring,)
    assert mse_fp8.shape == (d // ring,)
    # FP8 (8-bit) must be at least as accurate as FP4 (4-bit) per ring
    assert torch.all(mse_fp8 <= mse_fp4 + 1e-8)

def test_ring_deltaD_matches_difference():
    torch.manual_seed(1)
    d, M, ring = 128, 256, 64
    x = torch.randn(M, d)
    dd = ring_deltaD(x, ring)
    expected = ring_mse(x, ring, 'nvfp4') - ring_mse(x, ring, 'mxfp8')
    assert torch.allclose(dd, expected, atol=1e-10)
    assert torch.all(dd >= -1e-8)  # upgrading never hurts

def test_ring_mse_catches_outlier():
    # A heavy-tailed ring must show larger FP4 MSE than a Gaussian ring of equal variance.
    torch.manual_seed(2)
    ring = 64
    gauss = torch.randn(1024, ring)
    heavy = torch.randn(1024, ring); heavy[:, 0] *= 50.0  # inject outlier channel
    gauss = gauss / gauss.std() ; heavy = heavy / heavy.std()  # equalize variance
    x = torch.cat([gauss, heavy], dim=1)  # ring0=gauss, ring1=heavy
    mse = ring_mse(x, ring, 'nvfp4')
    assert mse[1] > mse[0]  # outlier ring hurts more under FP4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rd_alloc.py -v -k "ring_mse or deltaD"`
Expected: FAIL (ImportError: cannot import name 'ring_mse').

- [ ] **Step 3: Implement**

```python
# promix/quantize/adaptive.py  (append)
import torch
from .quant_utils import fake_quantize_nvfp4, fake_quantize_mxfp8

_FMT = {'nvfp4': fake_quantize_nvfp4, 'mxfp8': fake_quantize_mxfp8}

def ring_mse(x_rot, ring_size, fmt):
    """Per-ring measured quantization MSE on rotated activations/weights.
    x_rot: (M, d). Returns (d//ring_size,) float tensor."""
    if fmt not in _FMT:
        raise ValueError(f"fmt must be one of {list(_FMT)}, got {fmt}")
    x = torch.as_tensor(x_rot, dtype=torch.float32)
    M, d = x.shape
    if d % ring_size != 0:
        raise ValueError(f"d={d} not a multiple of ring_size={ring_size}")
    q = _FMT[fmt](x)                       # dequantized, same shape
    err2 = (x - q).pow(2)                  # (M, d)
    per_ring = err2.reshape(M, d // ring_size, ring_size).mean(dim=(0, 2))
    return per_ring                        # (d//ring_size,)

def ring_deltaD(x_rot, ring_size):
    """D_fp4 - D_fp8 per ring (error reduced by upgrading)."""
    return ring_mse(x_rot, ring_size, 'nvfp4') - ring_mse(x_rot, ring_size, 'mxfp8')
```

> NOTE: confirm the real `fake_quantize_nvfp4/_mxfp8` call signature (they may require extra args like block size / scale selection). If they need a scale arg, select scales with the existing `select_block_scales_nvfp4/_mxfp8` first and pass through, keeping the pipeline's real (MSE-searched) scales so the measured MSE matches deployment.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rd_alloc.py -v -k "ring_mse or deltaD"`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add promix/quantize/adaptive.py tests/test_rd_alloc.py
git commit -m "rd-alloc: per-ring measured quantization MSE + deltaD"
```

---

### Task 2: ΔD 排序 + 单阈值预算分配(纯函数)

**Files:**
- Modify: `promix/quantize/adaptive.py`
- Test: `tests/test_rd_alloc.py`

**Interfaces:**
- Consumes: Task 1 `ring_deltaD`.
- Produces:
  - `allocate_by_budget(deltaD_per_layer, target_mean_fp8, deltaC_per_layer=None) -> list[torch.BoolTensor]`:输入每层的 `ΔD`(list of `(n_ring_ℓ,)` 张量),目标平均 FP8 比例;返回每层 `(n_ring_ℓ,)` 布尔掩码(True=FP8)。等环时 `deltaC=None`→按 ΔD 排;否则按 ΔD/ΔC 排。全局单阈值,使平均 FP8 环比 = target。
  - `masks_to_khigh(masks, ring_size) -> list[int]`:每层 FP8 掩码 → K_high(= FP8 环数×ring_size),已天然 %ring_size 对齐。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rd_alloc.py  (append)
from promix.quantize.adaptive import allocate_by_budget, masks_to_khigh

def test_allocate_hits_budget_and_picks_highest_deltaD():
    # layer0: deltaD [0.08,0.02,0.005,0.001]; layer1: [0.015,0.009,0.004,0.002]
    dd = [torch.tensor([0.08,0.02,0.005,0.001]),
          torch.tensor([0.015,0.009,0.004,0.002])]
    # 8 rings total, target mean fp8 = 0.25 -> 2 FP8 rings total
    masks = allocate_by_budget(dd, target_mean_fp8=0.25)
    total_fp8 = sum(int(m.sum()) for m in masks)
    assert total_fp8 == 2
    # the 2 globally-highest deltaD rings are layer0-ring0 (0.08) and layer0-ring1 (0.02)
    assert masks[0].tolist() == [True, True, False, False]
    assert masks[1].tolist() == [False, False, False, False]

def test_masks_to_khigh_alignment():
    masks = [torch.tensor([True, True, False, False]),
             torch.tensor([False, False, False, False])]
    kh = masks_to_khigh(masks, ring_size=64)
    assert kh == [128, 0]
    assert all(k % 64 == 0 for k in kh)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rd_alloc.py -v -k "allocate or khigh"`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# promix/quantize/adaptive.py  (append)
def allocate_by_budget(deltaD_per_layer, target_mean_fp8, deltaC_per_layer=None):
    """Global single-threshold rate-distortion allocation.
    Rank all rings by deltaD/deltaC (deltaC=None -> equal cost -> by deltaD),
    upgrade the top ones until the average FP8 ring-fraction == target_mean_fp8."""
    flat = []  # (score, layer_idx, ring_idx)
    for li, dd in enumerate(deltaD_per_layer):
        dd = torch.as_tensor(dd, dtype=torch.float64)
        if deltaC_per_layer is not None:
            dc = torch.as_tensor(deltaC_per_layer[li], dtype=torch.float64).clamp_min(1e-12)
            score = dd / dc
        else:
            score = dd
        for ri in range(score.numel()):
            flat.append((float(score[ri]), li, ri))
    total_rings = len(flat)
    n_fp8 = int(round(target_mean_fp8 * total_rings))
    flat.sort(key=lambda t: t[0], reverse=True)
    masks = [torch.zeros(len(dd), dtype=torch.bool) for dd in deltaD_per_layer]
    for _, li, ri in flat[:n_fp8]:
        masks[li][ri] = True
    return masks

def masks_to_khigh(masks, ring_size):
    return [int(m.sum()) * ring_size for m in masks]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rd_alloc.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add promix/quantize/adaptive.py tests/test_rd_alloc.py
git commit -m "rd-alloc: deltaD-ranked single-threshold budget allocation + khigh mapping"
```

---

### Task 3: 【命门验证实验】实测-MSE 分配 vs 方差代理 vs uniform,同预算比 PPL

> 这是 spec §9 的核心假设验证,也是整个方向的 go/no-go。Task 1/2 是它的最小前置。
> **在最便宜的设置(R1-off 纯 PCA,复用现有 `round11_sensitivity.py --r1_off` 评测路径)** 跑,
> 先只验证**信号**:实测-MSE 排序在同预算下是否 PPL ≤ 方差代理、≤ uniform。信号成立才投入后续。
> 这是测量实验,不是单元测试——没有 pytest 断言,验收 = PPL 数值比较。

**Files:**
- Create: `promix/scripts/rd_validity.py`(实验 harness,收集每层旋转后激活 → ring_mse → allocate_by_budget → 按掩码 reorder+量化 → 评 PPL)
- Modify: `promix/eval/ptq.py`(接受外部传入的"逐层 K_high map",复用现有 `adaptive_high_fraction` + `reorder_before_quant` 分支;不新增量化逻辑)
- Reference: `promix/scripts/round11_sensitivity.py`(R1-off 构建法)、`promix/scripts/ablation_acc_per_bit.py`(PPL 评测)

**Interfaces:**
- Consumes: Task 1 `ring_mse`,Task 2 `allocate_by_budget`/`masks_to_khigh`;`ptq.py` 的 `main_path_khigh_map`-风格逐层 K_high 注入点。
- Produces: `promix/scripts/rd_validity.py` 可执行脚本,打印一张表:{uniform-1/8, 方差代理-alloc, 实测MSE-alloc} × {平均FP8比例, 3B PPL}。

- [ ] **Step 1: 收集旋转后激活**:在校准前向里对每层目标 Linear 的输入(R1-off = `U` 转后)收集 `(M, d)` 激活(复用 basis.py 的 hook 风格 / ablation 已有的前向)。写入 `rd_validity.py` 的 `collect_rotated_acts(model, calib, layers) -> dict[name -> (M,d) tensor]`。

- [ ] **Step 2: 三种分配**在同一平均 FP8 比例(=1/8)下生成逐层 K_high:
  - `uniform`: 每层 K_high = 1/8·d(现状);
  - `variance`: 用现有 `adaptive.py:derive_khigh`(能量/方差代理)在 τ 下使平均=1/8;
  - `measured_mse`: `allocate_by_budget([ring_deltaD(acts[name],64) for name...], target_mean_fp8=1/8)` → `masks_to_khigh`。
  三者平均 FP8 比例必须都 =1/8(打印核对)。

- [ ] **Step 3: 评 PPL**:对三种 K_high map 各跑一次 R1-off PTQ + wikitext-2 PPL(3B,同协议)。复用 `ablation_acc_per_bit.py` 的评测;R1-off 构建复用 `round11_sensitivity.py --r1_off`。

- [ ] **Step 4: 运行(远端 Pro 6000)并核对**

```bash
# 本地 push 后,远端:
docker exec -d -w /home/luofan/plaquant -e CUDA_VISIBLE_DEVICES=6 \
  -e PYTHONPATH=/home/luofan/plaquant -e HF_HOME=/home/luofan/hf_cache \
  luofan-plaquant bash -c 'python promix/scripts/rd_validity.py \
  --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml --r1_off \
  --target_mean_fp8 0.125 > /home/luofan/RD_VALIDITY.log 2>&1'
# 轮询日志
```

Expected(验收/gate):
- 三种平均 FP8 比例都 = 0.125(±一个环);
- **PPL 排序:measured_mse ≤ variance ≤ uniform(R1-off 同预算)**。
- **GATE 判定**:若 `measured_mse < variance`(实测 MSE 比方差代理更会分)→ 信号成立,方向继续(Task 4+)。若 `measured_mse ≈ variance ≈ uniform`(分配毫无增益)→ **命门不过**,停下来,按 spec §9 启用 loss 加权(见 Task 3b),再不行则整个"按误差分配"方向存疑,如实上报。

- [ ] **Step 5: Commit**(把脚本 + 结果表写入 `docs/rd_validity_report.md`)

```bash
git add promix/scripts/rd_validity.py promix/eval/ptq.py docs/rd_validity_report.md
git commit -m "rd-alloc: validity gate — measured-MSE vs variance vs uniform PPL at matched budget (R1-off)"
```

---

### Task 3b:(仅当 Task 3 gate 不过时)loss 加权兜底

**Files:** Modify `promix/quantize/adaptive.py`、`promix/scripts/rd_validity.py`

**Interfaces:**
- Produces: `ring_deltaD_weighted(x_rot, ring_size, sensitivity) -> tensor`,其中 `sensitivity`=每环对最终 loss 的梯度/Hessian 代理(每环 `(grad·Δ)²` 或 Fisher 对角近似)。`D_f(j) ← sensitivity(j)·MSE_f(j)`。
- 重跑 Task 3 的验收;若加权后 measured 仍不 track PPL,则按 spec §9 如实上报"按误差分配无增益",方向存疑。

- [ ] **Step 1-5**:实现 Fisher-对角/梯度代理敏感度(校准集一遍反传累积每环梯度平方)→ 加权 ΔD → 重跑 Task 3 对照 → commit 到 `docs/rd_validity_report.md`。

> (此任务仅在 Task 3 gate 不过时执行;过了直接跳到 Task 4。)

---

### Task 4:(gate 通过后)e2e cost 模型

**Files:**
- Create: `promix/quantize/cost_model.py`
- Test: `tests/test_rd_alloc.py`

**Interfaces:**
- Produces: `ring_cost(ring_size, fmt, w_prefill, w_decode, compute_coeff, n_total_rings) -> float`:返回一个环在格式 `fmt` 下的 e2e 边际成本 = `w_prefill·compute_coeff[fmt] + w_decode·bytes[fmt]`;`bytes['nvfp4']=4.5*ring_size/8`,`bytes['mxfp8']=8.5*ring_size/8`;`compute_coeff` 初始化自峰值表(NVFP4 1469 / FP8 807 TFLOPS 的倒数比),可被整网实测覆盖。`deltaC = ring_cost(fp8) − ring_cost(fp4)`。

- [ ] **Step 1: 失败测试**

```python
# tests/test_rd_alloc.py (append)
from promix.quantize.cost_model import ring_cost

def test_fp8_costs_more_than_fp4():
    cc = {'nvfp4': 1.0/1469, 'mxfp8': 1.0/807}  # ~inverse TFLOPS
    c4 = ring_cost(64, 'nvfp4', w_prefill=1.0, w_decode=0.0, compute_coeff=cc, n_total_rings=48)
    c8 = ring_cost(64, 'mxfp8', w_prefill=1.0, w_decode=0.0, compute_coeff=cc, n_total_rings=48)
    assert c8 > c4
def test_decode_cost_is_bytes():
    cc = {'nvfp4': 0.0, 'mxfp8': 0.0}
    c4 = ring_cost(64, 'nvfp4', 0.0, 1.0, cc, 48)
    c8 = ring_cost(64, 'mxfp8', 0.0, 1.0, cc, 48)
    assert abs(c4 - 4.5*64/8) < 1e-6 and abs(c8 - 8.5*64/8) < 1e-6
```

- [ ] **Step 2-5**:实现 `ring_cost`、跑测试通过、commit。整网实测校准(用真 e2e 前向替换 compute_coeff)作为脚本 `promix/scripts/calibrate_cost.py` 的一步,数据写 `docs/rd_cost_calib.md`。

---

### Task 5:(gate 通过后)三种操作点模式

**Files:** Modify `promix/quantize/adaptive.py`;Test `tests/test_rd_alloc.py`

**Interfaces:**
- Produces: `select_operating_point(deltaD_per_layer, deltaC_per_layer, mode, target, eval_ppl_fn, floor) -> (masks, info)`;`mode∈{'perf','accuracy','knee'}`:
  - `perf`: 给 `target`=平均FP8预算 → 直接 `allocate_by_budget`;
  - `accuracy`: 给 `target`=可接受 PPL → 从全FP4起逐步降阈值(加 FP8)直到 `eval_ppl_fn ≤ target`,取最省;
  - `knee`: 扫预算画 ΔPPL/Δcost 曲线,取最大曲率点(规则:相邻斜率比 < 0.5 的首点);不 robust 则回退 `accuracy`。
  - 所有模式结果受 Global Constraints 的可行域(PPL ≤ 现状、平均FP8 ≤ 1/8)裁剪。

- [ ] **Step 1-5**:`perf` 模式有纯函数测试(= Task 2);`accuracy`/`knee` 用 mock `eval_ppl_fn`(单调函数)做纯逻辑测试;commit。

---

### Task 6:(gate 通过后)整合 64-环旋转底座 + config 接线 + 多模型验收

**Files:** Modify `promix/quantize/rotation.py`(统一64-环 R1,依赖另一份旋转 spec)、`promix/eval/ptq.py`、`promix/configs/*mxfp8-nvfp4*.yaml`;Create `docs/rd_final_report.md`

**Interfaces:**
- Consumes: Task 1/2/4/5 全部;旋转底座 spec 的统一64-环 R1 构建。
- Produces: config 开关 `rd_alloc: {enabled, ring_size: 64, mode, target}`(默认 `enabled: false` = 现状 uniform,byte-identical 零回归);端到端:校准→ring_mse→select_operating_point→逐层K_high→64-环 R-on 量化→PPL+性能。

- [ ] **Step 1: 零回归测试**:`rd_alloc.enabled=false` 时逐层 K_high 与现状 uniform 1/8 完全一致(`test_rd_disabled_matches_uniform`,对比 `main_path_khigh_map` 输出)。
- [ ] **Step 2: R-on 整合**:在统一64-环 R1 上跑 measured-MSE 分配(R-on,非 R1-off),3B PPL ≤ 现状 8.30 且平均FP8 ≤ 1/8。
- [ ] **Step 3: 多模型 + 性能**:1B/8B 同样验收(≤ 14.56/6.60);用整网实测 e2e(prefill+decode)报告 vs uniform 1/8 的真实加速,写 `docs/rd_final_report.md`。
- [ ] **Step 4: 对照组齐全**:uniform-1/8、uniform-1/16、方差代理、measured-MSE 四条,accuracy-per-bit + accuracy-per-latency 两张 Pareto。
- [ ] **Step 5: Commit**。

---

## Self-Review

**Spec coverage:** §1 底线→Global Constraints+Task3/6 验收;§2 两格式→Task1/全程;§3 环→Task1/2;§4 率失真→Task2/4;§5 实测MSE→Task1;§6 e2e cost→Task4;§7 判定→Task2;§8 三操作点→Task5;§9 效度命门→Task3+3b(前置);§10 验收→Task3/6;§11 文件→各 Task Files;§12 风险(膝盖/cost非线性/FP16)→Task5/4/未来。全覆盖。

**Placeholder scan:** 纯函数任务(1/2/4/5)含真实测试+实现代码;实验任务(3/3b/6)为测量/整合,给了确切命令+验收数值+go/no-go,不是占位(研究实验无法单元断言,已显式标注)。

**Type consistency:** `ring_mse`/`ring_deltaD`(Task1)→`allocate_by_budget`(Task2,吃 deltaD list)→`masks_to_khigh`(Task2)→`ptq` K_high 注入(Task3/6);`ring_cost`/`deltaC`(Task4)→`allocate_by_budget` 的 `deltaC_per_layer`(Task2 已留参)→`select_operating_point`(Task5)。签名一致。

**关键排序说明**:Task1/2 是命门 Task3 的最小前置(要先能算 MSE、能分配,才能测它是否 track PPL)。Task3 是 go/no-go gate;Task 4/5/6 全部 gated on Task3 通过。Task3 在最便宜的 R1-off 设置验证信号,不需先建 64-环旋转(推迟到 Task6)。
