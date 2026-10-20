import csv, glob, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else "logs/occ"
MET = {"tcE": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
       "tcA": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
       "warp": "sm__warps_active.avg.pct_of_peak_sustained_active",
       "limSmem": "launch__occupancy_limit_shared_mem",
       "limReg": "launch__occupancy_limit_registers",
       "regs": "launch__registers_per_thread",
       "smem": "launch__shared_mem_per_block_static",
       "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
       "dur": "gpu__time_duration.sum"}
print(f"{'label':26} {'regs':>5} {'smemB':>7} {'limSm':>6} {'limRg':>6} {'warp%':>6} {'TCela':>6} {'TCact':>6} {'wait':>5} {'dur(us)':>8}")
for f in sorted(glob.glob(os.path.join(D, "*.csv"))):
    lab = os.path.basename(f)[:-4]; d = {}
    for r in csv.DictReader([l for l in open(f) if not l.startswith("==PROF")]):
        if r.get("Metric Name"): d[r["Metric Name"]] = r.get("Metric Value")
    def g(k):
        try: return float(str(d.get(MET[k], "nan")).replace(",", ""))
        except: return float("nan")
    print(f"{lab:26} {g('regs'):5.0f} {g('smem'):7.0f} {g('limSmem'):6.0f} {g('limReg'):6.0f} {g('warp'):6.1f} {g('tcE'):6.1f} {g('tcA'):6.1f} {g('wait'):5.2f} {g('dur')/1000:8.1f}")
