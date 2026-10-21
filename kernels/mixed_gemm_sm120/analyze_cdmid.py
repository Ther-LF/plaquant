import csv, os, sys

WANT = {
 "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "TCelap%",
 "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "TCact%",
 "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": "SMact%",
 "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM%",
 "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2%",
 "l1tex__throughput.avg.pct_of_peak_sustained_elapsed": "L1%",
 "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue%",
 "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "barrier",
 "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio": "membar",
 "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "longsb",
 "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "shortsb",
 "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "wait",
 "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "mio",
 "launch__registers_per_thread": "regs",
 "gpu__time_duration.sum": "dur",
}
LABELS = ["DUAL_down", "FP4_down", "FP8_down", "DUAL_gate", "FP4_gate", "FP8_gate"]
d = os.path.dirname(os.path.abspath(__file__)) if len(sys.argv) < 2 else sys.argv[1]
rows = {}
units = {}
for lab in LABELS:
    fn = os.path.join(d, lab + ".csv")
    if not os.path.exists(fn):
        continue
    with open(fn) as f:
        lines = [l for l in f if l.strip()]
    hi = None
    for i, l in enumerate(lines):
        if 'Metric Name' in l and 'Metric Value' in l:
            hi = i
            break
    if hi is None:
        continue
    rdr = csv.DictReader(lines[hi:])
    cur = {}
    for r in rdr:
        mn = r.get("Metric Name")
        mv = r.get("Metric Value")
        mu = r.get("Metric Unit", "")
        if mn in WANT:
            try:
                v = float(str(mv).replace(",", ""))
            except Exception:
                v = mv
            cur[WANT[mn]] = v
            units[WANT[mn]] = mu
    rows[lab] = cur
cols = ["dur", "TCelap%", "TCact%", "SMact%", "issue%", "DRAM%", "L2%", "L1%",
        "barrier", "membar", "longsb", "shortsb", "wait", "mio", "regs"]
print("units:", {c: units.get(c, "") for c in cols})
hdr = "%-11s" % "label" + "".join("%9s" % c for c in cols)
print(hdr)
for lab in LABELS:
    cur = rows.get(lab, {})
    def g(c):
        v = cur.get(c)
        if v is None:
            return "-"
        return "%.2f" % v if isinstance(v, float) else str(v)
    print("%-11s" % lab + "".join("%9s" % g(c) for c in cols))
