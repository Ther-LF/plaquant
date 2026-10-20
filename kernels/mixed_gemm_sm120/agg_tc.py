import csv, glob, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else "."
rows = {}
for f in glob.glob(os.path.join(D, "*.csv")):
    lab = os.path.basename(f)[:-4]
    d = {}
    try:
        for r in csv.DictReader([l for l in open(f) if not l.startswith("==PROF")]):
            mn = r.get("Metric Name"); mv = r.get("Metric Value")
            if mn:
                d[mn] = mv
    except Exception as e:
        d["err"] = str(e)
    rows[lab] = d

M = {
    "tcE": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tcA": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm":  "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram":"dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lsb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "ssb": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "wait":"smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "barr":"smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "mio": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "dur": "gpu__time_duration.sum",
}
hdr = ["label", "TCela", "TCact", "SM", "DRAM", "lsb", "ssb", "wait", "barr", "mio", "dur_ns"]
print("  ".join(f"{h:>8}" if h != "label" else f"{h:22s}" for h in hdr))
for lab in sorted(rows):
    d = rows[lab]
    vals = [lab] + [d.get(M[k], "-") for k in ["tcE","tcA","sm","dram","lsb","ssb","wait","barr","mio","dur"]]
    print("  ".join(f"{v:>8}" if i else f"{v:22s}" for i, v in enumerate(vals)))
