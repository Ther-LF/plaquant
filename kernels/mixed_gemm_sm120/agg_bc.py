import csv, glob, os, re, sys
D = sys.argv[1] if len(sys.argv) > 1 else "logs/bc"
MET = {"lsb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
       "ssb": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
       "bcld": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
       "bcst": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
       "tcA": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"}
data = {}
for f in glob.glob(os.path.join(D, "*.csv")):
    lab = os.path.basename(f)[:-4]
    m = re.match(r"(.+)_sfa(\d)$", lab)
    if not m: continue
    cell, sfa = m.group(1), int(m.group(2))
    d = {}
    for r in csv.DictReader([l for l in open(f) if not l.startswith("==PROF")]):
        if r.get("Metric Name"): d[r["Metric Name"]] = r.get("Metric Value")
    data[(cell, sfa)] = d
def gv(d, k):
    try: return float(str(d.get(MET[k], "nan")).replace(",", ""))
    except: return float("nan")
print(f"{'cell':22} | {'lsb 0>1':>13} {'ssb 0>1':>13} {'bcLD 0>1':>15} {'bcST 0>1':>15} {'TCact 0>1':>14}")
for cell in sorted(set(c for (c, s) in data)):
    d0, d1 = data.get((cell, 0), {}), data.get((cell, 1), {})
    print(f"{cell:22} | {gv(d0,'lsb'):5.2f}>{gv(d1,'lsb'):5.2f} {gv(d0,'ssb'):5.2f}>{gv(d1,'ssb'):5.2f} "
          f"{gv(d0,'bcld'):6.0f}>{gv(d1,'bcld'):6.0f} {gv(d0,'bcst'):6.0f}>{gv(d1,'bcst'):6.0f} "
          f"{gv(d0,'tcA'):5.1f}>{gv(d1,'tcA'):5.1f}")
