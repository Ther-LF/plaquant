# SM121 (GB10 / DGX Spark, compute capability 12.1) interface index — CUDA mode

The sm121 tree of CUDA mode (single-file CUDA C++ student). Opened empty following
`../README.md` ("Opening a tree for a new architecture = create the directory + write
`index.md`"). Demand-driven: a recipe is written when the student gets stuck on that
interface, following `../TEMPLATE.md`, and is marked ✅ only on a GPU probe returning
`PROBE PASS` on THIS architecture, or a shipped KEEP whose own NCU census shows the
contract axis moved. Readings from `../sm100/` (B200 tcgen05/TMA-centric) are
hypotheses at best here: SM 12.1 is a Blackwell-consumer-class part (48 SMs, ~100 KB
smem/block opt-in, LPDDR unified memory, SM120-style `mma.sync` block-scaled tensor
cores, **no tcgen05, no cta_group::2 pairs, no tmem**), so most sm100 entries do not
transfer; what does transfer must be re-certified here before it earns ✅.

## Look up by symptom first

| What you see / what you are about to do | Look first at |
|---|---|
| **Writing a multi-contract GPU probe**; or a round comes back all FAIL with only one fact; or "no verdict" with the reason already scrolled out of the log tail | [`gpu_probe_ladder`](gpu_probe_ladder.md): one rung per process, self-contained digest, verdicts and facts on separate channels |
| **Kernel hangs / never returns after launch** | [`gpu_probe_ladder`](gpu_probe_ladder.md) §1: one rung per process, each with its own budget |
| **smem traffic over the line** — unsure whether to tune the swizzle or inspect the feed path | [`smem_traffic_triage`](smem_traffic_triage.md) |

## Certified entries

(none yet — this tree was opened for the first SM121 run; methodology entries above
are transcribed reading recipes, not mechanism claims about this GPU)
