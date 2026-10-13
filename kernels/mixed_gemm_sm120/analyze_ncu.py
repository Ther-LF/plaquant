"""Parse ncu .ncu-rep via the ncu_report Python API and emit the six-dimension
analysis per kernel. Run inside luofan-ncu.

Usage: python analyze_ncu.py <full.ncu-rep> [source.ncu-rep]
"""
import sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.3.1/extras/python")
import ncu_report

def safe(a, n, d=None):
    try:
        return a[n].value()
    except Exception:
        return d

METRICS = [
    # Dim1 occupancy / launch
    "launch__grid_size", "launch__block_size", "launch__waves_per_multiprocessor",
    "launch__registers_per_thread", "launch__shared_mem_per_block",
    "launch__occupancy_limit_blocks", "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # SOL / throughput
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
    "sm__inst_executed.avg.per_cycle_active",
    # Dim4 tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    # Dim6 memory
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    # Dim3 stall ratios
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
]

def short(n):
    n = n.split("(")[0]
    return n[-70:]

def main():
    rep = ncu_report.load_report(sys.argv[1])
    print(f"# ranges={rep.num_ranges()}")
    for ri in range(rep.num_ranges()):
        rng = rep.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            a = rng.action_by_idx(ai)
            print("\n" + "=" * 100)
            print(f"KERNEL[{ri}.{ai}]: {short(a.name())}")
            print("-" * 100)
            for m in METRICS:
                v = safe(a, m)
                if v is None:
                    continue
                try:
                    print(f"  {m:72s} {v:>14.4g}")
                except Exception:
                    print(f"  {m:72s} {str(v):>14}")
            # NCU rule suggestions
            try:
                rules = a.rule_results_as_dicts()
                sug = [(r.get("estimated_speedup_pct"), r.get("rule_name"),
                        (r.get("message_for_display") or "")[:160]) for r in rules]
                sug = [s for s in sug if s[0] is not None]
                sug.sort(key=lambda x: (x[0] if isinstance(x[0], (int, float)) else 0), reverse=True)
                if sug:
                    print("  --- NCU rules (top) ---")
                    for sp, rn, msg in sug[:5]:
                        print(f"    [{sp}%] {rn}: {msg}")
            except Exception as e:
                print("  (no rules)", e)

    # source pass: aggregate per-reason pcsamp if provided
    if len(sys.argv) > 2:
        print("\n\n########## SOURCE PASS (pcsamp stall totals per kernel) ##########")
        sr = ncu_report.load_report(sys.argv[2])
        reasons = ["long_scoreboard", "short_scoreboard", "wait", "barrier",
                   "lg_throttle", "mio_throttle", "math_pipe_throttle",
                   "not_selected", "selected", "no_instructions", "drain"]
        for ri in range(sr.num_ranges()):
            rng = sr.range_by_idx(ri)
            for ai in range(rng.num_actions()):
                a = rng.action_by_idx(ai)
                tot = safe(a, "smsp__pcsamp_sample_count")
                if not tot:
                    continue
                print(f"\nKERNEL[{ri}.{ai}]: {short(a.name())}  samples={tot:.0f}")
                for rr in reasons:
                    v = safe(a, f"smsp__pcsamp_warps_issue_stalled_{rr}")
                    if v:
                        print(f"    {rr:20s} {v:>12.0f}  ({100.0*v/tot:5.1f}%)")

if __name__ == "__main__":
    main()
