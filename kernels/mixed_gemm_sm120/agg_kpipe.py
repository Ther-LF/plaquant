import csv, glob, os, re, sys
D = sys.argv[1]
MET = {
    "tcE": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tcA": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "lsb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "dur": "gpu__time_duration.sum",
}
data = {}
for f in glob.glob(os.path.join(D, "*.csv")):
    lab = os.path.basename(f)[:-4]
    m = re.match(r"(\w+?)_M(\d+)_Kh(\d+)_kp(\d)", lab)
    if not m: continue
    shape, M, Kh, kp = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    d = {}
    for r in csv.DictReader([l for l in open(f) if not l.startswith("==PROF")]):
        if r.get("Metric Name"): d[r["Metric Name"]] = r.get("Metric Value")
    data[(shape, M, Kh, kp)] = d

def gv(d, k):
    try: return float(d.get(MET[k], "nan"))
    except: return float("nan")

print(f"{'shape':5} {'M':>5} {'Kh':>5} | {'TCela kp0>kp1':>16} {'TCact kp0>kp1':>16} {'wait0>1':>12} {'dur0>1(us)':>16}")
keys = sorted(set((s, M, Kh) for (s, M, Kh, kp) in data))
order = {"qkv": 0, "down": 1, "gate": 2}
keys.sort(key=lambda x: (x[1], order.get(x[0], 9), x[2]))
for (s, M, Kh) in keys:
    d0 = data.get((s, M, Kh, 0), {}); d1 = data.get((s, M, Kh, 1), {})
    e0, e1 = gv(d0, "tcE"), gv(d1, "tcE")
    a0, a1 = gv(d0, "tcA"), gv(d1, "tcA")
    w0, w1 = gv(d0, "wait"), gv(d1, "wait")
    u0, u1 = gv(d0, "dur")/1000, gv(d1, "dur")/1000
    print(f"{s:5} {M:>5} {Kh:>5} | {e0:6.1f}>{e1:6.1f} ({e1-e0:+5.1f}) {a0:6.1f}>{a1:6.1f} ({a1-a0:+5.1f}) "
          f"{w0:4.2f}>{w1:4.2f} {u0:7.1f}>{u1:7.1f}")
