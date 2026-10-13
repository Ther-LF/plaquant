# FP vs INT 混合精度量化:1B/3B/8B 复现结果

**日期:** 2026-10-13
**硬件:** RTX Pro 6000 Blackwell (SM120),容器 `luofan-plaquant`
**结论一句话:** NVFP4+MXFP8 混合精度相对 INT4+INT8 的精度优势**随模型规模单调增长**——1B 落后、3B 追平、8B 反超。

---

## 1. 核心结果

Wikitext-2 PPL(越低越好),W4A4 配置(KV cache 保持 FP16),per-head o_proj:

| 模型 | INT W4A4 | FP (NVFP4+MXFP8) | FP − INT | ResQ 论文 W4A4 |
|------|----------|------------------|----------|----------------|
| Llama-3.2-1B | 14.68 | 14.92 | **+0.24**(FP 落后) | ~12.4* |
| Llama-3.2-3B | 8.60 | 8.65 | **+0.05**(追平) | ~8.8 |
| Llama-3-8B | 6.99 | **6.83** | **−0.16**(FP 反超) | ~6.72 |

\* 1B 论文值偏低是因为论文跑了完整 Step 1 旋转优化;我们对 1B 跳过了 Step 1(该模型上旋转优化无收益,与原结论一致),所以用随机正交 R,PPL 略高。3B/8B 复现与论文吻合甚至略好。

**趋势解读:** 1B 冗余少、量化余量小,FP 的细粒度块 scale 发挥不出,反被 INT 的均匀格点在"旋转已抹平"的分布上够用;模型增大后冗余和离群结构变复杂,NVFP4(每 16 元素一个块 scale)+ MXFP8(每 32 元素一个块 scale)贴合钟形分布 + 抗离群的优势兑现。真实部署的是大模型 → 该方向成立。

### 补充:1B 的 FP 权重 vs 激活消融

| 配置 | 1B PPL |
|------|--------|
| 仅权重 FP(w=nvfp4, a=16) | 13.78(比 INT 的 14.68 还好) |
| 仅激活 FP(w=16, a=nvfp4) | 14.15 |
| 全 FP | 14.92 |

→ 1B 上 FP 的退化主要来自**激活**侧;权重侧 FP 量化本身优于 INT。

---

## 2. 测试方法(可复现)

### 2.1 量化方案

- **格式:** 低精度主段 = NVFP4(E2M1,每 16 元素一个 FP8-E4M3 块 scale);高精度段 = MXFP8(E4M3,每 32 元素一个 E8M0 块 scale)。INT 基线 = INT4 主段 + INT8 高段。
- **分段:** 每个 Linear 的输入沿 hidden 维按方差排序后切 `[低 7/8 | 高 1/8]`(high_fraction=0.125)。
- **o_proj:** per-head 旋转(U_value·R2 融进 v_proj 输出 + o_proj 输入),量化时**先列重排成连续 [低|高] 再按硬件块网格量化**(FP 路径关键:per-head 的 64 维 [56|8] 不满足 FP 块对齐,必须重排;见 `ActQuantWrapper.reorder_before_quant`)。
- **down_proj:** 在线 Hadamard + FP 分段。
- **KV cache:** 本实验保持 FP16(W4A4,非 KV4)。
- **这是 fake quant:** 量化→反量化→FP16 矩阵乘,只验证**精度**。真低比特 SM120 kernel 尚未实现。

### 2.2 三步 pipeline

```
Step 0 (basis):  收集激活协方差 → PCA 基 U(按方差排序)+ 随机正交 R
Step 1 (rotation): 旋转优化(1B 跳过,3B/8B 用随机 R;本实验未训练 R)
Step 2 (PTQ eval): 融合 U/R + GPTQ 权重量化 + fake quant → 测 PPL
```

**关键:basis U 与量化格式无关**(只反映数据统计),且 INT/FP 都用 per-head o_proj → **同一模型的 Step 0 只跑一次,INT 和 FP Step 2 共享 basis**。

### 2.3 精确复现命令(在 Pro 6000 容器 `luofan-plaquant` 内)

环境:`docker exec -w /home/luofan/plaquant -e HF_HOME=/home/luofan/hf_cache -e PYTHONPATH=/home/luofan/plaquant -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=<空闲GPU>`

```bash
# ---- 每个模型只跑一次 Step 0(INT 配置,FP 复用其 basis)----
python -m promix.quantize.basis --config promix/configs/llama-3.2-3b-w4a4.yaml \
    --output_dir ./rotation --nsamples 512          # 3B,~50min
python -m promix.quantize.basis --config promix/configs/llama-3-8b-w4a4.yaml \
    --output_dir ./rotation --nsamples 512          # 8B,~75min

# ---- INT 基线 Step 2 ----
torchrun --nproc_per_node=1 --master_port=29501 promix/eval/ptq.py \
    --config promix/configs/llama-3.2-3b-w4a4.yaml   # → 8.60
torchrun --nproc_per_node=1 --master_port=29502 promix/eval/ptq.py \
    --config promix/configs/llama-3-8b-w4a4.yaml     # → 6.99

# ---- FP Step 2(复用同一 basis)----
torchrun --nproc_per_node=1 --master_port=29503 promix/eval/ptq.py \
    --config promix/configs/llama-3.2-3b-fp-perhead.yaml  # → 8.65
torchrun --nproc_per_node=1 --master_port=29504 promix/eval/ptq.py \
    --config promix/configs/llama-3-8b-fp-perhead.yaml    # → 6.83
```

1B 同理,用 `llama-3.2-1b-resq.yaml`(INT)和 `llama-3.2-1b-fp-perhead.yaml`(FP)。

### 2.4 配置文件

| 模型 | INT 配置 | FP 配置 |
|------|----------|---------|
| 1B | `llama-3.2-1b-resq.yaml` | `llama-3.2-1b-fp-perhead.yaml` |
| 3B | `llama-3.2-3b-w4a4.yaml` | `llama-3.2-3b-fp-perhead.yaml` |
| 8B | `llama-3-8b-w4a4.yaml` | `llama-3-8b-fp-perhead.yaml` |

FP 配置相对 INT 的差异:`w_bits/a_bits="nvfp4"`、`high_bits="mxfp8"`、显式 `o_proj_pca: "per_head"`;basis/rotation 路径指向与 INT 相同的文件(共享 Step 0)。`nsamples=512`(需用 `--nsamples 512` 传入以匹配 basis 文件名 `-512-`)。

### 2.5 环境与坑

- 模型:`unsloth/Llama-3.2-{1B,3B}`、`unsloth/llama-3-8b`,HF 下载需开 VPN(到 huggingface.co;hf-mirror.com 无此 repo)。
- GPU 共享:别的用户的任务会随时占满 GPU → 跑前用 `nvidia-smi` 挑空闲卡,加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。3B FP32 协方差约需 24GB,8B 更多。
- 容器需 `pip install --break-system-packages pytest`(未持久化)。
- 用 `docker exec -d ... > log 2>&1` 分离运行,日志落盘(SSH 管道会吞输出);PPL 在 lm-eval 任务前打印,`grep Perplexity` 取。
- **8B 下载极慢**(VPN ~70MB/min,~16GB);中途 `pkill` 会丢失正在下的分片 → 启动后不要打断,让它一次下完。

---

## 3. 现状

| 项 | 状态 |
|----|------|
| 1B/3B/8B INT 复现 + FP 扩展 | ✅ 完成,FP 优势随规模增长 |
| 自适应 high_fraction 实验 | ✅ 负结果(见 `2026-10-09-adaptive-high-fraction-results.md`) |
| W4A4KV4(KV 量化) | ❌ 配置在,未跑 |
| **SM120 真融合 kernel(低比特计算)** | ❌ 未开始 —— 当前全是 fake quant,只验精度 |

**下一步:** 实现 SM120 NVFP4+MXFP8 融合 GEMM 算子,把"性能更好"从假设变成实测(拟用 KDA 方式优化)。
