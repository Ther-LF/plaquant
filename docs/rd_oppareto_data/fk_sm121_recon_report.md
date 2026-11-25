# promix_dualphase_gemm — reference recon report (source run)

Job index (every fact below cites one of these):

| run | job | script | what it measured |
|---|---|---|---|
| run0 | 1795610689 | notes/p1_sass_launch.py | CUPTI launch record, cuobjdump SASS, seeded sass/orch/sched/live/struct tools, compat_report |
| run1 | 1795611670 | notes/p2_timing_sweep.py | crashed (script bug); only the permission probe (ERR_NVGPUCTRPERM) is usable |
| run2 | 1795612103 | notes/p2_timing_sweep.py | DRAM bandwidth, GEMM-only CUPTI duration sweep over k_lo/k_hi/n/m, linear fit, NCU permission state |
| run3 | 1795612722 | notes/p3_phase_residency.py | single-phase vs dual-phase contrasts, tile-order (swizzle x raster) residency grid |
| run4 | 1795613929 | notes/p4_validate.py | tile order de-confounded (3 rotated rounds), per-call SF re-layout cost + bit-equality, expressibility ladder (rung A / rung B) |
| run5 | 1795614858 | notes/p5_switch_order.py | phase-switch excess vs tile order and n (interleaved) |
| run6 | 1795615894 | notes/p6_waiver_channels.py | waiver channels: seeded `stall_price_run.py --pm`, `pm_timeline_run.py`, ncu SourceCounters, CUPTI PC-sampling activity kinds (all blocked) |
| driver | 1795609703 | sealref | REF_BASELINE device 1.7402 ms, "NCU: no collection path" |

Local (no-GPU) derivations below were made off the run0 SASS dump (`notes/ref_gemm.sass`, decoded from job 1795610689) using `notes/smem_footprint.py` / `notes/smem_regions.py`. Those aren't new measurements; they re-read the same binary, so they cite job 1795610689.

## Disclaimer

- **Toolchain.** The reference binary was built with nvcc 13.0, CUTLASS 4.4.2 (a fork) and `-O3 --use_fast_math -arch=sm_121a` [MEASURED, job 1795613929]. A student build with a different nvcc/CUTLASS or different flags can get different register allocation, a different instruction order and a different LDSM/LDS mix. Every SASS count in §2 is a property of *this* binary, not of the algorithm.
- **Per-shape heuristic choice.** The tile order (swizzle 4, raster AlongN) comes from a host heuristic (`cd_apply_sched_env` in the source) that keys off SM count, tile counts and K. All three graded workloads land on the same choice. The measured best order differs per workload (§3b, job 1795613929), so the reference's order is a heuristic outcome, not an optimum.
- **Compiler behaviour the student may lack.**
  - The `setmaxnreg` 40/232 split, the interleaved LDSM/MMA software pipeline (copy k-block n+1 while MMA k-block n), and the bf16 round-trip in the epilogue are all visible in SASS.
  - A hand-written kernel won't get them for free. In particular the compiler hoisted address-source loads out of the producer loops (ORCH supply axis, job 1795610689).
- **Profiler access.** Performance counters are blocked on this machine (`RmProfilingAdminOnly: 1`, ncu `ERR_NVGPUCTRPERM`) [MEASURED, jobs 1795611670, 1795612103; driver job 1795609703]. Nothing in this report comes from NCU metrics or PC sampling. Durations come from CUPTI activity records; structure comes from cuobjdump SASS.

## §1 Launch config

`LAUNCH-OK kernel=_ZN7cutlass13device_kernelIN8codesign19CodesignFusedKernel... grid=(1,48,1) block=(384,1,1) ctas=48 cluster=(0,0,0) cluster_max_active=n/a regs=168 smem_dyn=88064` [MEASURED, job 1795610689]

- **Grid and residency.** There are 48 CTAs, one per SM (GB10: 48 SMs, L2 25165824 B [MEASURED, job 1795612103]). The kernel is persistent: 4096 tiles of 128×128 for M=N=8192, i.e. 85.3, so 86 waves [MEASURED arithmetic in job 1795612103 output].
- **Cluster.** Launched without a cluster attribute (cluster=(0,0,0)). The kernel template has ClusterShape 1×1×1.
- **Block.** 384 threads = 3 warpgroups (1 producer + 2 consumer/MMA) [compat_report, MEASURED, job 1795610689].
- **Registers.** 168 per thread at launch. `USETMAXREG.DEALLOC 0x28` (40) for the producer warpgroup and `USETMAXREG.TRY_ALLOC 0xe8` (232) for the MMA warpgroups [MEASURED, job 1795610689]. Check: 128×40 + 256×232 = 64512 ≤ 65536, and 384×168 = 64512. It's the same pool.
- **Shared memory.** Dynamic smem is 88064 B = SharedStorageSize [compat_report, MEASURED, job 1795610689]. Static is 0; res-usage reports SHARED 1024, which is the runtime reserve.
- **Launches per reference call.** There are 9: 8 `at::native direct_copy` elementwise kernels plus 1 GEMM [MEASURED, job 1795610689; CUPTI over 20 calls: aux kernels/call = 8.0, job 1795613929].

  | workload | aux kernels (sum/call) | GEMM (median) |
  |---|---|---|
  | gu1b_r8 | 53.2 µs | 1442.4 µs |
  | gu1b_r2 | 41.1 µs | 1689.5 µs |
  | gu3b_r8 | 68.8 µs | 1914.5 µs |

  [MEASURED, job 1795613929]

## §2 SASS

`SASS-OK kernel=_ZN7cutlass13device_kernelIN8codesign19CodesignFusedKernelIN4cute5tupleIJiiiiEEE instructions=2528 opcodes=115` [MEASURED, job 1795610689]

### Histogram (load-bearing families) [MEASURED, job 1795610689]

| family | count | role |
|---|---|---|
| QMMA.SF.16832.F32.E4M3.E4M3.E8 | 128 | MXFP8 phase MMA (m16n8k32, UE8M0 scale, 1X) |
| OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X | 64 | NVFP4 phase MMA (m16n8k64, UE4M3 scale, 4X) |
| LDSM.16.M88.4 | 72 | A/B operand fragments smem→reg |
| LDS (32b) | 48 | NV scale factors (mainloop) + epilogue bias reads |
| LDS.U8 | 80 | MX scale factors (bytes) |
| LDS.128 | 8 | NV scale factors (vector) |
| STSM.16.M88.2 | 16 | epilogue D staging reg→smem |
| STS.U16 | 2 | epilogue bias staging |
| LDG.E.U16 | 1 | bias global load |
| UTMALDG.3D | 4 | TMA load A, B (×2 phases) |
| UTMALDG.4D | 4 | TMA load SFA, SFB (×2 phases) |
| UTMASTG.3D | 8 | TMA store of D subtiles |
| UTMACMDFLUSH | 8 | TMA store commit |
| UTMACCTL.PF | 10 | TMA descriptor prefetch |
| DEPBAR.LE | 9 | TMA store drain (SB0 ≤1 ×8, SB0 ≤0 ×1) |
| BAR.SYNC | 26 | named barriers (see §2g) |
| SYNCS.EXCH.64 | 20 | mbarrier init |
| SYNCS.PHASECHK.TRANS64.TRYWAIT | 12 + 12 | mbarrier waits (spin) |
| NANOSLEEP | 12 | spin back-off in mbarrier waits |
| USETMAXREG (DEALLOC 0x28 / TRY_ALLOC 0xe8) | 1 + 1 | 40/232 register split |
| ELECT | 10 | single-thread elect for TMA issue |
| F2F.BF16.F32 | 64 | epilogue: acc → bf16 round |
| F2F.F16.F32 | 64 | epilogue: → fp16 output |
| FADD | 64 | epilogue: + bias |
| FMUL | 64 | epilogue: × alpha (alpha = 1.0, kept by the compiler) |
| HADD2.F32 | 16 | epilogue: half bias → f32 |

**Cache-policy bits:** the UTMALDG/UTMASTG appear plain, with no L2 cache-hint operand in the census [MEASURED, job 1795610689]. Caveat: the rung-B probe with an L2 hint also produced an opcode-level footprint of plain `UTMALDG.3D` [MEASURED, job 1795613929]. So the opcode census can't tell hint from no hint. See the candidates `[VERIFY: inferred]` row.

### smem access footprint (mandatory) [MEASURED off SASS, job 1795610689]

Layout derived from the immediate offsets. It's an inference; the derivation and falsifier are under the buffer map.

- epilogue region = [0x0000, 0x3800) = 14336 B
- mainloop union = [0x3800, 0x15800) = 73728 B
- 14336 + 73728 = 88064 = smem_dyn ✓

**Write side** (who writes smem, into which buffer):

| family | count | offsets | buffer |
|---|---|---|---|
| UTMALDG.3D (TMA) | 4 (2 per phase: A, B) | register/descriptor-addressed | NV: A stages 0x3800–0xB800 (4×8 KiB), B 0xB800–0x13800 (4×8 KiB). MX: A 0x3800–0xB800 (2×16 KiB), B 0xB800–0x13800 (2×16 KiB) |
| UTMALDG.4D (TMA) | 4 (2 per phase: SFA, SFB) | descriptor-addressed | NV: SFA 0x13800–0x14800 (4×1 KiB), SFB 0x14800–0x15800. MX: SFA 0x13800–0x13C00 (2×512 B), SFB 0x13C00–0x14000 |
| STSM.16.M88.2 | 16 (all in epilogue) | 0x0 ×8, 0x400 ×8 (+ register base) | epilogue D staging (2 D buffers, SM90_U32x2_STSM_N) |
| STS.U16 | 2 (epilogue) | 0x3400 | bias staging (128 halves) |
| SYNCS.EXCH.64 | 20 | UR6+0x0..0x88, UR5+0x90/0x98 | mbarrier objects (control, not data) |

**Read side** (who reads smem, from which buffer):

| family | count | per region | buffer |
|---|---|---|---|
| LDSM.16.M88.4 | 72 | NV mainloop 24, MX mainloop 48 | A/B operand stages, offsets 0x3800–0xE800 immediate + stage register base |
| LDS.128 | 8 | NV mainloop 8 | NV SFA/SFB (0x13800–0x14A80) |
| LDS (32b) | 48 | NV mainloop 8, epilogue 40 | NV SF (0x13800+), epilogue bias 0x3400–0x34D0 and register-based 0x0 (×32) |
| LDS.U8 | 80 | MX mainloop 80 | MX SFB 0x13C00–0x13C8C immediate, MX SFA via register base 0x0–0x8F |
| UTMASTG.3D (TMA reads smem) | 8 | epilogue | D staging buffers 0x0 / 0x400 region |

**Per-region op counts** (address regions as in `smem_regions.py`) [MEASURED off SASS, job 1795610689]:

| region | address | smem ops | MMAs |
|---|---|---|---|
| NV mainloop | 0x37e0– | LDSM 24, LDS 8, LDS.128 8 | OMMA 48 |
| NV→MX switch, around bar 0xe | 0x4710– | — | OMMA 16 (the drained last k-block) |
| MX mainloop | 0x4d20– | LDS.U8 80, LDSM 48 | QMMA 128 |
| MX→epilogue switch, bar 0xf | 0x6900– | — | — |
| epilogue | 0x6cc0– | LDS 40, STS.U16 2, STSM 16 | — |

**Buffer-map inference.**
- **Basis:** immediate offsets plus the stage sizes the source implies.
  - NV stage = 8 KiB A + 8 KiB B + 1 KiB SFA + 1 KiB SFB = 18 KiB. ×4 = 72 KiB.
  - MX stage = 16 + 16 + 0.5 + 0.5 = 33 KiB. ×2 = 66 KiB ≤ 72 KiB.
  - So the union is sized by NV.
- **Chain:** the union size 73728 B plus the epilogue's 14336 B equals smem_dyn exactly. The LDS SF offsets 0x13800/0x13C00 land exactly at 0x3800 + 64 KiB (= 0x3800 + 4×16 KiB for NV A+B), which is the SF block in both phases.
- **Confidence:** high for the epilogue/union split and for the SF placement. Medium for the epilogue's internal split (D staging vs bias vs the C stages that are allocated but unused).
- **Falsify:** a run with a different `CD_NV_STAGES` that does not move smem_dyn by 18 KiB per stage.

### §2b Question-driven checklist

| question | answer | evidence |
|---|---|---|
| Do both phases share one FP32 accumulator? | Yes. The same base registers (R16, R24, R28, R32, …) head both OMMA chains (chain2–14) and QMMA chains (chain18–33). No register move or add between the phases. | [MEASURED, job 1795610689] SCHED chains |
| Is the accumulator cleared only once per tile? | Yes, by inference. MX chains start without a zeroing move on the shared registers, and the NV phase is the one with ClearAccum (source). | sass-derived [job 1795610689]; intent from source |
| MMA instruction shapes? | NV m16n8k64 OMMA.SF …UE4M3.4X; MX m16n8k32 QMMA.SF …E8 | [MEASURED, job 1795610689] |
| MMA counts per k-tile per warp? | Atom layout 4×2×1 warps (kernel name `Shape<_4,_2,_1>`), so the warp tile is 32×64, i.e. 2×8 = 16 m16n8 MMAs per k-step. NV: k64 steps, 2 per 128-wide k-tile, 32 OMMA per k-tile; the 64 static OMMA = 48 in the NV loop body + 16 in the switch region. MX: k32 steps, 4 per k-tile, 64 QMMA per k-tile; the 128 static QMMA = 2 k-tiles (2 stages) unrolled. | sass-derived [job 1795610689] |
| Operand path? | TMA → smem (swizzled) → LDSM.x4 → registers → mma.sync family. FP4 operands get an in-register shift fixup (source `fp4_shift_A/B`). No warpgroup MMA or tcgen05 (wgmma=0, tcgen05=0). | [MEASURED, job 1795610689] |
| Scale-factor path? | TMA 4D → smem → LDS/LDS.128 (NV, UE4M3 ×4 per 64-K) or LDS.U8 (MX, UE8M0). Passed in registers to the MMA. | [MEASURED, job 1795610689] |
| Scale-factor layout in global? | 128×4 atom, "32_4_4" interleave, built per call by 8 torch copy kernels from the row-major SF (§3c). | [MEASURED, jobs 1795610689, 1795613929]; layout from run.py |
| Pipeline depth? | NV 4 stages, MX 2 stages (compat_report). Ring stride 1, wrap by SEL. | [MEASURED, job 1795610689] |
| Producer/consumer split? | Warp 0 of WG0 issues mainloop TMA (ELECT). Warp 2 of WG0 would be the epilogue-load warp, but there is no C source: no epilogue UTMALDG, and all 8 UTMALDG are mainloop. WG1+WG2 do the MMA and the epilogue (cooperative, not ping-pong). | [MEASURED, job 1795610689] |
| Phase handoff? | Named barriers 0xe (NV→MX) and 0xf (MX→epilogue/next tile), width 288 = 32 producer + 256 consumer threads, SYNC on both sides. | [MEASURED, job 1795610689] |
| Epilogue arithmetic? | out = fp16( f32(bf16(acc × alpha)) + f32(bias[n]) ). F2F.BF16.F32 → FADD → F2F.F16.F32, 64 each (64 acc regs/thread). | [MEASURED, job 1795610689] |
| Output path? | STSM to smem (2 D buffers), then UTMASTG.3D + UTMACMDFLUSH, then DEPBAR.LE SB0 0x1. 8 store subtiles per tile, drained to 0 at the end. | [MEASURED, job 1795610689] |
| Epilogue/mainloop overlap? | Not on the same warps: the consumer warps run the epilogue after bar 0xf. The producer warp can prefetch the next tile's NV stages while the epilogue runs (it passed bar 0xf). Measured effect: see §3b. | sass-derived [job 1795610689]; timing [job 1795612103] |
| Tile order? | StaticPersistentScheduler, grid (1,48,1) consistent with raster AlongN. The heuristic picks swizzle 4; the env-forced sw4/AlongN geomean is 0.996× auto. | [MEASURED, jobs 1795610689, 1795613929] |
| Does the phase switch cost time intrinsically? | No. The excess disappears under sw1/AlongN (−15.8 ns/wave at the floor) and at n=4096. It's a tile-order/memory effect. | [MEASURED, job 1795614858] |

### §2g Sync topology and supply orchestration

`ORCH-OK regions=41 loops=39 nbar_ids=5 mbar_objs=38 resolved=1 edges=0 itc_candidates=0` [MEASURED, job 1795610689; dossier banked as `ref_orch_dossier.txt`]

**Dossier event table** [MEASURED, job 1795610689]:

| region | kind | address span | instrs | events |
|---|---|---|---|---|
| r0 | straight | 0000–1270 | 296 | prologue, SHFL ×6, mbarrier INIT ×20, bar 0x0 |
| r4 | loop copy-producer | 1400–16f0 | 48 | UTMALDG ×4 (NV phase: A, B, SFA, SFB) |
| r5 | loop copy-producer | 1700–18d0 | 30 | bar 0xe SYNC (producer side) |
| r8 | loop copy-producer | 1980–1cb0 | 52 | UTMALDG ×4 (MX phase) |
| r9 | loop copy-producer | 1cc0–24a0 | 127 | bar 0xf SYNC (producer side), arrive R8+0x98 |
| r28/r29 | loop | 37e0–4090 | 54/86 | NV k-loop: LDS ×4, LDSM ×6 each; bar 0x7 |
| r31/r32 | loop | 40e0–4cc0 | 47/144 | NV tail + bar 0x7 + bar 0xe (consumer side) |
| r34/r35 | loop | 4d10–59c0 | 27/177 | MX k-loop: LDS 10/30, LDSM 6/18; bar 0x7 |
| r37 | loop | 5a10–5d30 | 51 | MX k-loop part |
| r38 | loop mma-consumer | 5d40–9ba0 | 999 | MX tail + bar 0xf + epilogue: UTMASTG ×8, LDS 70, STS 2, LDSM 18, STSM 16, LDG 1 |
| r39/r40 | scalar/straight | 9bb0–9df0 | 25/12 | tile-loop tail, exit |
| — | spin-wait micro-loops | ×20 | | r2, r3, r6, r7, r11–r16, … (mbarrier TRYWAIT + NANOSLEEP) |

**Summary axes** [MEASURED, job 1795610689]:
- cache_ops = [UTMALDG.3D, UTMALDG.4D, UTMASTG.3D]
- commit = []
- supply = "addr-source loads hoisted; prefetched"
- supply_counts = "copies 4, no in-loop address-source loads (hoisted); tight-chained copies 0" ×2 (one per phase loop)
- rings = [stride=0x1, wrap=sel]
- mbar_tiers = {RING: 1}; mbar_unresolved = 37
- spin_loops = 20
- merge: scalar_sts_in_loop = 2, shfl_out_of_loop = 6

**Per-barrier topology** [MEASURED, job 1795610689]:

| id | width (threads) | ARV | SYNC | regions | verdict | role (inferred from position + source) |
|---|---|---|---|---|---|---|
| 0x0 | (block) | 0 | 1 | r0 | symmetric rendezvous | prologue: whole CTA after mbarrier init |
| 0x1 | 256 | 0 | 17 | r38 | symmetric rendezvous | epilogue: consumer WGs around each STSM → TMA-store hand-off and buffer reuse |
| 0x7 | 256 | 0 | 4 | r29, r32, r35, r38 | symmetric rendezvous | Sm120 mainloop barrier: all MMA threads done reading a stage before consumer_release |
| 0xe | 288 | 0 | 2 | r5 (producer), r32 (consumer) | symmetric rendezvous | NV→MX phase switch (producer warp + 256 MMA threads) |
| 0xf | 288 | 0 | 2 | r9 (producer), r38 (consumer) | symmetric rendezvous | MX→epilogue / next-tile switch |

**Blocking rendezvous per loop body** [MEASURED, job 1795610689]:

| loop | rendezvous per body | made of |
|---|---|---|
| r5 | 1 | 0xe |
| r9 | 1 | 0xf |
| r29 | 1 | 0x7 |
| r32 | 2 | 0x7 + 0xe |
| r35 | 1 | 0x7 |
| r38 | 19 | 17×0x1 + 0x7 + 0xf |

- **Per-tile total, consumer side** (derived): 1 (0x7) per NV k-tile, 1 per MX k-tile, plus 0xe, plus 0xf, plus 17 epilogue 0x1.
  - gu1b_r8 (KT_lo = 14, KT_hi = 2): 14 + 2 + 1 + 1 + 17 = 35 blocking rendezvous per tile.
  - This is a derivation: it assumes one 0x7 per k-tile, as the source loop shows. Falsifier: per-PC exec counts (not available, §2h).
- **mbarrier objects:**
  - Ring offsets: NV full +0x0, NV empty +0x20, MX full +0x40, MX empty +0x50.
  - Epilogue / load-order barriers at +0x90 and +0x98: init at UR5+0x90/+0x98; producer arrives on R8+0x98 in r9.
  - Only one object resolved as RING (WAIT@r30 ×2, WAIT@r36 ×2, via SEL wrap); the other 37 are UNRESOLVED by the tool.
  - The NV/MX full/empty assignment above is an inference from offsets and the producer acquire/arrive order in r4/r8. Medium confidence. Falsifier: a probe build that changes the stage count and moves those offsets differently.

### §2d Schedule order

`SCHED-OK isa=mma_sync loop=backedge mma_runs=192 barriers=25 ldsm=53 depbars=9 cp_async_groups=0 mbar_waits=22 ew_runs=34` [MEASURED, job 1795610689]

**Event table** (condensed from the 399-event transcription of the backedge body 0x1750–0x9c30) [MEASURED, job 1795610689]:

| events | content |
|---|---|
| 0 | BAR.SYNC 0xe, 0x120 (producer side, previous tile) |
| 3–12 | producer: TRYWAIT MX empty (+0x50), arrive MX full (+0x40, expect-tx), UTMALDG.3D ×2 + UTMALDG.4D ×2 |
| 13–14 | BAR.SYNC 0xf, 0x120; arrive +0x98 |
| 15–37 | producer next tile: NV empty/full waits (R5, R7), ELECT |
| 38–40 | consumer entry: USETMAXREG.TRY_ALLOC 0xe8 |
| 41–47 | TRYWAIT stage full (R2); LDSM ×6 + LDS (k-block 0 prefetch) |
| 48–67 | OMMA chain0–15 interleaved with LDSM ×2 groups; BAR.SYNC 0x7 at event 57 (last k-block of the stage) |
| 68 | mbarrier.arrive [R0] (consumer_release of the NV stage) |
| 73–106 | next NV k-tile: OMMA interleaved with LDSM (copy k-block n+1 before MMA k-block n) |
| 114, 116 | BAR.SYNC 0x7; BAR.SYNC 0xe, 0x120 (consumer side, NV→MX) |
| ~117–316 | MX k-loop: TRYWAIT +0x40, LDS.U8 / LDSM, QMMA chains 18–34; BAR.SYNC 0x7 at 204, 299 |
| 317 | BAR.SYNC 0xf, 0x120 (consumer side, MX→epilogue) |
| 319–396 | epilogue ×8 subtiles: BAR.SYNC 0x1 → F2F/FADD/HADD2/STSM → BAR.SYNC 0x1 → UTMASTG.3D → UTMACMDFLUSH → DEPBAR.LE SB0 0x1 |
| 398 | DEPBAR.LE SB0 0x0 (drain at tile end) |

**Hand-off before each element-wise segment** (34 ew runs), verbatim from the tool [MEASURED, job 1795610689]:

```
BAR.SYNC 0xe,0x120 | arrive +0x40 ×3 | arrive R8+0x98 ×2 | trywait R7 | trywait R5 ×2 | trywait R2 ×2 |
arrive R0 | MMA:chain16 | trywait R4+0x40 | MMA:chain18 | arrive R0 | MMA:chain19 | MMA:chain34 | MMA:chain18 |
BAR.SYNC 0xf,0x120 | BAR.SYNC 0x1 | DEPBAR.LE SB0 0x1 | BAR.SYNC 0x1 ×2 | DEPBAR.LE SB0 0x1 | BAR.SYNC 0x1 ×2 |
DEPBAR.LE SB0 0x1 | BAR.SYNC 0x1 ×2 | DEPBAR.LE SB0 0x1 | BAR.SYNC 0x1 | DEPBAR.LE SB0 0x1 | BAR.SYNC 0x1
```

Read:
- **Epilogue segments.** All epilogue ew segments (bf16 round, bias add, fp16 convert) sit behind BAR.SYNC 0x1 or DEPBAR.LE SB0 0x1. Each conversion block waits until the TMA store from two subtiles back has drained its smem buffer.
- **Mainloop segments.** These sit behind mbarrier waits or MMA chains. There's no DEPBAR in the mainloop: cp_async_groups = 0, and TMA completion goes through mbarriers.

**Wait-target alignment** (inference):
- **Basis:** `DEPBAR.LE SB0 0x1` after each of the 8 UTMASTG, and STSM writes alternate between 2 buffers (0x0 / 0x400 + base). So a wait-to-≤1-outstanding matches a 2-deep D ring exactly.
- **Chain:** before STSM into buffer b, the store that last read b, two subtiles earlier, must be complete. ≤1 outstanding means only the immediately previous store is still in flight. It's aligned, with no over-wait and no under-wait.
- **Mainloop:** consumer_release (arrive R0) comes right after BAR.SYNC 0x7, so a stage is released only once all 256 MMA threads have finished their LDSMs. Each k-tile pays a rendezvous.
- **Confidence:** high for the epilogue (direct from SASS), medium for the mainloop (positions only, no exec counts).
- **Falsify:** a SASS reading showing STSM to a buffer with 2 stores outstanding.

### §2e Registers

`LIVE-OK peak=145 guessed_frac=0.02` on the doubled backedge body; the seeded single-body parse gave `LIVE-OK peak=123 guessed_frac=0.11` [MEASURED, job 1795610689].

- **Peak live registers:** 145, at body instr 688 (an OMMA in the NV phase) [MEASURED, job 1795610689].
- **Accumulator:** 64 fp32 registers per MMA thread (128×128 / 256 threads); 16 chains × 4 registers [MEASURED chains, job 1795610689].
- **Non-accumulator working set:** 145 − 64 = 81 registers. The tool's peak breakdown: smem/const-load = 32 (A/B fragments + SF), addr/temp = 26, undef/param = 14, plus the accumulator chains [MEASURED, job 1795610689].
- **Headroom:** 232 allocated − 145 peak = 87 registers of headroom on the MMA warps. The 40-register producer cap is tight but sufficient: only TMA issue and spin-waits run there.
- **Inference:** the register split isn't binding at this tile. A 128×256 tile would double the accumulator to 128 regs, putting the peak around 209 < 232. That's not measured.

### §2f STALL

**STALL-WAIVED: no per-instruction stall ledger.**
- Warp-stall sampling requires NVIDIA performance counters, and the device reports `RmProfilingAdminOnly: 1` [MEASURED, job 1795612103].
- Channels tried:
  - ncu launch metrics (`launch__grid_dim_x, launch__registers_per_thread`): rc=1, `ERR_NVGPUCTRPERM` [MEASURED, jobs 1795611670, 1795612103].
  - ncu DRAM/L2 metrics (`dram__bytes_read.sum, dram__bytes_write.sum, lts__t_sector_hit_rate.pct`): rc=1, `ERR_NVGPUCTRPERM` [MEASURED, jobs 1795611670, 1795612103].
  - Driver sealref NCU collection: "NCU: no collection path" [driver job 1795609703].
  - The seeded `stall_price_run.py --pm`, the standalone PM-timeline recipe, the ncu SourceCounters page and CUPTI PC-sampling activity kinds are submitted as `notes/p6_waiver_channels.py`; all four failed [MEASURED, job 1795615894] (see the addendum at the end of this report).
- CUPTI *activity* records (kernel durations, launch config) do work [MEASURED, job 1795610689]. They carry no stall reasons.

Stand-in evidence (timing, not stalls): per-wave cost decomposes as 7.37 µs fixed + 545 ns per NV k-tile + 978 ns per MX k-tile [MEASURED, job 1795612103]. See §3b.

### §2g Overlap ledger (ROLE / PMTL)

- **ROLE-WAIVED: the role exposure ledger needs the stall ledger plus the PM-sampling tensor plateau.** Both come from NCU counter collection, which is blocked (`ERR_NVGPUCTRPERM`, `RmProfilingAdminOnly: 1`) [MEASURED, jobs 1795611670, 1795612103]. The channels tried are the same as for STALL, and the p6 addendum reports the seeded recipe's verdict.
- **PMTL-WAIVED: no PM-sampling timeline.** The `PmSampling` / `PmSampling_WarpStates` sections are NCU counter sections and are blocked by the same permission. The channel tried is `notes/pm_timeline_run.py` in p6 (addendum).

Stand-in overlap evidence, from timing only:

**(a) Epilogue vs mainloop overlap.**
- Single-phase K=128/256/512 takes 683.9/687.8/703.1 µs [MEASURED, job 1795612722]. That's flat and equal to the 0.685 ms 8192² fp16 output-sized DRAM write [MEASURED, job 1795612103].
- At small K the kernel is therefore output-write-bound. Mainloop work hides under the store stream until the per-wave mainloop exceeds ~8 µs.

**(b) Long-K behaviour.**
- At long K, time ≈ intercept + k-proportional. The intercept is 633.7 µs total, 7.37 µs/wave [MEASURED, job 1795612103], which is ≈ the output-write time per wave (685/86 = 7.97 µs).
- Inference: write and mainloop overlap only partially at the graded shapes.
- Basis: for gu1b_r8, predicted compute 0.657 + 0.168 = 0.825 ms (marginal rates) + 0.685 ms write = 1.51 ms if fully serialized, vs 1.44 ms measured, vs max() = 0.825 ms if fully overlapped.
- Confidence medium. Falsifier: a PM timeline showing the tensor pipe busy during the store bursts.

### §2h STRUCT

**STRUCT-WAIVED: no per-PC execution counts.**
- The struct tool's self-check failed with "no per-PC execution counts — collect the source page as CSV (ncu --import rep --page source --print-source sass --csv)" [MEASURED, job 1795610689].
- The source page needs an NCU report, and NCU collection is blocked (`ERR_NVGPUCTRPERM`) [MEASURED, jobs 1795611670, 1795612103]. Channel retried: ncu `--section SourceCounters` in p6 (addendum).
- The static table, without exec counts, is below [MEASURED, job 1795610689]:

```
region 1: BAR.SYNC 0x7,0x100 -> 8 warps; 4 barriers -> 4 segments; copies=1
  seg 1 |  893-970  | LDSM.16.M88.4 x12  LDS x8  OMMA x6
  seg 2 |  971-1125 | OMMA x42  LDSM.16.M88.4 x12  LDS x8
  seg 3 | 1126-1399 | QMMA x48  LDS x40  LDSM.16.M88.4 x24  OMMA x16
  seg 4 | 1400-1627 | QMMA x64  LDS x40  LDSM.16.M88.4 x24
  smem addressing: seg1/2 granule 0x800 strides [0x8,0x80,0x180,0x1f8,0x800,0xdf8];
                   seg3/4 granule 0x1000 strides [0x1,0x2,0x4,0x72,0x74,0x1000]
  mma input types (tool label): {'fp16': 64, 'e4m3': 112}   <- tool mislabels E2M1 OMMA as fp16; the SASS says E2M1
region 2: BAR.SYNC 0x1,0x100 -> 8 warps; 17 barriers -> 17 segments
  seg 1          : STS x2                                     (bias staging)
  seg 2,6,10,14  : F2F x16 FADD x8 LDS x6 HADD2 x4 STSM x2     (offset strides 0x10, 0x3000/0x3040/0x3080/0x30c0)
  seg 3,7,11,15  : F2F x1
  seg 4,8,12,16  : F2F x15 FADD x8 LDS x4 STSM x2
  seg 5,9,13,17  : -
  hand-off: seg 1 stores -> seg 2 loads at 0x3400  (bias STS -> LDS)
```

Note: seg 3 (QMMA ×48 + OMMA ×16) is the NV→MX switch region. The last NV k-block's OMMAs drain under the same 0x7 segment as the first MX k-blocks, so the switch is software-pipelined across the phase boundary [sass-derived, job 1795610689].

## §3 DRAM traffic and kernel count

**DRAM bytes: not measured.** `dram__bytes_*` is blocked (`ERR_NVGPUCTRPERM`) [MEASURED, job 1795612103]. Structural arithmetic instead (compulsory, each byte once):

| workload | A+B lo (fp4) | A+B hi (fp8) | SF lo (ue4m3) + SF hi (ue8m0) | inputs total | output fp16 | bias |
|---|---|---|---|---|---|---|
| gu1b_r8 (1792 / 256) | 14.68 MB | 4.19 MB | 1.84 + 0.13 MB | ≈ 20.8 MB | 134.2 MB | 16 KB |
| gu1b_r2 (1024 / 1024) | 8.39 MB | 16.78 MB | 1.05 + 0.52 MB | ≈ 26.7 MB | 134.2 MB | 16 KB |
| gu3b_r8 (2688 / 384) | 22.02 MB | 6.29 MB | 2.75 + 0.20 MB | ≈ 31.3 MB | 134.2 MB | 16 KB |

- Output is 8192 × 8192 × 2 B = 134.2 MB. It's 81–87% of compulsory traffic on every workload.
- Re-read traffic, A panels across N-tiles and B panels across M-tiles beyond L2 hits, is not measured.
- Per call, the 8 SF re-layout copy kernels additionally read and write each SF tensor (≈ 2× SF bytes in, 2× out, because each `_blocked` produces 2 copies) [kernel count MEASURED, job 1795613929; bytes are arithmetic].

**Kernel count:** 9 per reference call (8 aux + 1 GEMM) [MEASURED, jobs 1795610689, 1795613929].

**Measured bandwidth** [MEASURED, job 1795612103]:
- DRAM write 197.0 GB/s
- copy 222.6 GB/s (r+w)
- 8192² fp16 write 0.685 ms (196.0 GB/s)
- read via torch sum 45.8 GB/s. This is reduction-kernel-bound, not a DRAM read ceiling.

## §3b Bottleneck / residency verdict

**(a) Roofline from structural arithmetic.**
- *Output-write floor:* 134.2 MB / 196 GB/s = 0.685 ms [MEASURED, job 1795612103].
- *Compute at the measured marginal rates:* NV 366.4 TFLOP/s and MX 204.3 TFLOP/s [MEASURED fit, job 1795612103].

  | workload | NV | MX | compute total |
  |---|---|---|---|
  | gu1b_r8 | 240.5 GFLOP → 0.656 ms | 34.4 GFLOP → 0.168 ms | 0.825 ms |
  | gu1b_r2 | — | — | 0.375 + 0.673 = 1.048 ms |
  | gu3b_r8 | — | — | 0.985 + 0.252 = 1.237 ms |

- *Measured GEMM:* 1440.7 / 1695.4 / 1890.3 µs [MEASURED, job 1795612103]. That sits between max(write, compute) = 0.825 / 1.048 / 1.237 ms and the serialized sum 1.51 / 1.73 / 1.92 ms, close to the sum.
- *Verdict: serialized write + mainloop.* The write stream (DRAM ~196 GB/s) and the tensor mainloop overlap poorly in this machine. The consumer warps own both mainloop and epilogue (cooperative, not ping-pong), and the TMA store keeps only 1 subtile in flight (DEPBAR.LE SB0 0x1).
  - Basis: the fit intercept (7.37 µs/wave) ≈ the write time per wave (7.97 µs), plus the flat single-phase floor (683.9 µs at K=128 ≈ 0.685 ms write) [MEASURED, jobs 1795612103, 1795612722].
  - Confidence medium. The marginal NV/MX rates come from a fit, not from a peak measurement, so the true tensor peak is not measured.
  - Falsify: a variant whose epilogue overlaps the next tile's mainloop (ping-pong, or a separate store warp) and still lands near the serialized sum.

**(b) Dominant operand and working set vs L2.**
- The output dominates bytes (134 MB, written once, streaming through a 24 MiB L2).
- Inputs: the whole B for gu1b_r8 is 7.34 MB lo + 2.10 MB hi + SF ≈ 10.4 MB, and A is the same size. Each operand fits in L2 alone, but A+B+SF ≈ 20.8 MB together with the 134 MB output stream does not stay resident.
- Per-wave working set (derived): 48 concurrent tiles under swizzle 4 cover about 4 M-panel rows × 12 N-panel columns. That's 16 panels × (128 × 1792 / 2 + SF) ≈ 16 × 129 KB ≈ 2 MB for the NV phase, which is small vs L2.
- Measured residency sensitivity [MEASURED, job 1795612103]: per-wave cost drops from 16.75 µs (n = 8192) to 13.99 µs (n = 4096) and 13.61 µs (n = 2048) at the same K.
- The phase-switch excess (1.67 µs/wave at the floor, n = 8192, auto order) is 0.07 µs/wave at n = 4096 and −0.016 µs/wave under sw1/AlongN [MEASURED, job 1795614858].

**(c) Is residency load-bearing? Yes.**

| order | gu1b_r8 | gu1b_r2 | gu3b_r8 | GEMM geomean |
|---|---|---|---|---|
| auto (reference choice) | 1.000 | 1.000 | 1.000 | 1655.7 µs |
| sw4 / AlongN (env-forced) | 0.996 | 0.988 | 0.998 | 1648.9 µs (0.996× auto) |
| sw1 / AlongN | 0.863 | 0.969 | 1.036 | 1574.6 µs (0.951× auto) |
| sw1 / AlongM | 0.903 | 0.961 | 1.000 | 1569.2 µs (0.948× auto) |
| sw2 / AlongN | — | — | — | 1596.6 µs (0.964× auto) |

[MEASURED, 3 rotated rounds, job 1795613929]

- The tile order alone moves the GEMM by up to 13.7% on gu1b_r8 and by 3.6% in the opposite direction on gu3b_r8 [MEASURED, jobs 1795612722, 1795613929]. The reference's heuristic order is not the per-shape best on any of the three workloads.
- Mechanism (inference, low confidence): it's not panel-capacity reuse, since the per-wave panels are ~2 MB. Candidates are DRAM read/write interleaving with the output stream, or L2 set pressure from the output stream evicting the B panels between M-rows.
- Falsify: `lts__t_sector_hit_rate` per order. That's not measurable here.

## §3c Inter-kernel dataflow ledger

There are 9 kernels per reference call [MEASURED, jobs 1795610689, 1795613929].

| # | producer kernel | buffer written | consumer | initialisation | ordering |
|---|---|---|---|---|---|
| 1–2 | at::native direct_copy ×2 | SFA_lo in blocked 128×4 / 32_4_4 layout (via permute→reshape→transpose→contiguous) | GEMM UTMALDG.4D (NV SFA) | freshly torch-allocated, fully overwritten, not zeroed | same stream, launch order |
| 3–4 | direct_copy ×2 | SFB_lo blocked | GEMM UTMALDG.4D (NV SFB) | same | same |
| 5–6 | direct_copy ×2 | SFA_hi blocked | GEMM UTMALDG.4D (MX SFA) | same | same |
| 7–8 | direct_copy ×2 | SFB_hi blocked | GEMM UTMALDG.4D (MX SFB) | same | same |
| 9 | GEMM | D (fp16, M×N) | caller | ptr_C = D, beta = 0, so D is never read: no epilogue UTMALDG [MEASURED, job 1795610689] | — |

- **Cost.** Replacing the per-call re-layout with already-blocked SF is bit-equal on all three workloads and saves 80.2 / 60.1 / 97.2 µs end to end; geomean ratio 1.0471 [MEASURED, job 1795613929].
- **Padding.** With M = N = 8192 and K multiples of 128, the 128×4 atom tiles exactly, so no padding/zero-fill kernel is needed. The arithmetic is from run.py's `_blocked`; the absence of a fill kernel is in the CUPTI census [MEASURED, job 1795610689].

## Waiver summary

- `SASS-WAIVED:` none. SASS obtained via cuobjdump [job 1795610689].
- `SCHED-WAIVED:` none. SCHED-OK [job 1795610689].
- `LIVE-WAIVED:` none. LIVE-OK [job 1795610689].
- `STALL-WAIVED:` performance counters blocked (`ERR_NVGPUCTRPERM`, `RmProfilingAdminOnly: 1`).
  - Channels tried: ncu launch and DRAM/L2 metrics [jobs 1795611670, 1795612103], driver sealref NCU [job 1795609703].
  - p6 (`stall_price_run.py --pm` → `STALL-FAIL reason=no-report`; CUPTI PC sampling → `NotImplementedError`) [job 1795615894].
- `ROLE-WAIVED:` depends on the stall ledger plus the PM-sampling plateau; both are blocked as above. Channel: `stall_price_run.py --pm` in p6 → `ERR_NVGPUCTRPERM` [job 1795615894].
- `PMTL-WAIVED:` the PM-sampling sections are NCU counter sections and are blocked. Channels: `pm_timeline_run.py` and `stall_price_run.py --pm` in p6 → both `PMTL-FAIL reason=no-report` [job 1795615894].
- `STRUCT-WAIVED:` no per-PC exec counts because the NCU source page is blocked. STRUCT-FAIL is in job 1795610689; the static table is given in §2h. Channel: ncu `--section SourceCounters` in p6 → rc=1 `ERR_NVGPUCTRPERM` [job 1795615894].
- `DATAFLOW-WAIVED:` none. The ledger is in §3c.
- DRAM bytes / L2 hit rate: not measured (counters blocked [job 1795612103]). Arithmetic is given in §3.

## Addendum — p6 waiver-channel run

Job 1795615894 (`notes/p6_waiver_channels.py`; log `notes/p6_waiver_channels.run6.log`) [MEASURED, job 1795615894]:

| channel | command | result |
|---|---|---|
| A. permission state | `/proc/driver/nvidia/params`, euid | `RmProfilingAdminOnly: 1`, euid 1005 (not root); ncu 2025.3.1.0 present |
| B. seeded overlap recipe | `python3 notes/stall_price_run.py --pm` | `STALL-FAIL reason=no-report`, `PMTL-FAIL reason=no-report`; `==ERROR== ERR_NVGPUCTRPERM` |
| C. standalone PM timeline | `python3 notes/pm_timeline_run.py` | `PMTL-FAIL reason=no-report`; `ERR_NVGPUCTRPERM` |
| D. per-PC exec counts | `ncu --profile-from-start off --launch-count 1 --section SourceCounters ... eval/ncu_run.py` | rc=1, `ERR_NVGPUCTRPERM`. The `eval/ncu_run.py` entry point itself works: the process connected and the extension loaded (`ninja: no work to do`) before the permission error. |
| E. CUPTI PC sampling (non-NCU path) | `cupti.activity_enable(...)` for INSTRUCTION_CORRELATION / INSTRUCTION_EXECUTION / PC_SAMPLING / PC_SAMPLING_RECORD_INFO | all four `NotImplementedError: Activity Kind ... not supported`. cupti-python exposes only `ActivityPCSamplingPeriod` / `ActivityPCSamplingStallReason` (enums), and no `cuptiPCSampling*` API |

**Conclusion.**
- Every channel that could give per-instruction stalls, per-PC execution counts or a PM timeline is blocked on this node. The blocks are the driver-level counter permission (B, C, D) and the missing CUPTI activity support (E).
- The STALL / ROLE / PMTL / STRUCT waivers above therefore stand, with every channel exhausted.
- The driver's automatic `stall_price_run.py --pm` at wrap-up will hit the same `ERR_NVGPUCTRPERM`. The entry point `eval/ncu_run.py` is valid; the failure is permission, not the entry file.
- **Downstream consequence (for distill).** There is no reference-side overlap ledger. The [SYNC:]/[GATHER:]/[PHASE:] verdicts can only be written as candidates. They should rest on:
  - the static ORCH dossier (`ref_orch_dossier.txt`, job 1795610689);
  - the SCHED event stream (§2d);
  - the timing-only overlap evidence in §2g (jobs 1795612103, 1795612722).
