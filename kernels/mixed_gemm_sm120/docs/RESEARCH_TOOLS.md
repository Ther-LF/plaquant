# 查资料渠道(每轮 kernel agent 必读)

实现任何优化前,先查前人怎么做,别闭门造车。三个来源:

## 1. DeepSeek Web Search / Fetch(实时网页,最广)
本机(Mac)可直接跑,key 走环境变量 DS_KEY(勿写入文件/提交):
```
DS_KEY=<见下> python3 kernels/mixed_gemm_sm120/tools/websearch.py search "<query>" [count]
DS_KEY=<见下> python3 kernels/mixed_gemm_sm120/tools/websearch.py fetch  "<url>"
```
key:见 ~/.claude 或向 controller 索取;勿写入任何被提交的文件(建议轮换)。
endpoint:api.deepseek.com/v1/tools/{search,fetch},按 query 返回结构化结果+摘要,fetch 返回 markdown。
查询技巧:带 site:github.com、精确词组、lang:en、after:2024 等。

## 2. KernelWiki(本地,Blackwell/Hopper kernel 知识库 + CUTLASS/SGLang/vLLM/FlashInfer/TRT-LLM PR)
```
cd ~/.claude/skills/KernelWiki && python3 scripts/query.py "<natural language>"
python3 scripts/get_page.py <page-id> --body-only
```

## 3. CUTLASS 源码(远程 third_party/cutlass,权威实现)
include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp、kernel/sm120_*、
tile_scheduler*.hpp、pipeline/*、cute/atom/mma_traits_sm120.hpp。

## 已证高价值来源(本工程查到的)
- LeetCUDA(xlite-dev):寄存器乒乓 ldmatrix(k+1)⊗MMA(k) 的确切写法
- TensorRT-LLM tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dense_gemm_persistent.py:persistent + StaticPersistentTileScheduler
- DeepWiki "L2-aware tile schedulers" / "Blackwell GEMM labs"
- CUTLASS SM120 collective:persistent-by-default、cooperative atom Shape<4,2,1>

## 规矩:每个优化杠杆,先搜(至少 KernelWiki + 一次 web search),把"借鉴了谁的什么模式"写进 report。
