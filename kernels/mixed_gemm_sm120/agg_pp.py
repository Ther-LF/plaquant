import csv, glob, os, re, sys
D = sys.argv[1]
MET = {
    "tcE": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tcA": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "warp": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "lsb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "ssb": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "regs": "launch__registers_per_thread",
    "limR": "launch__occupancy_limit_registers",
    "limS": "launch__occupancy_limit_shared_mem",
    "dur": "gpu__time_duration.sum",
}
data = {}
for f in glob.glob(os.path.join(D, "*.csv")):
    lab = os.path.basename(f)[:-4]
    m = re.match(r"(\w+?)_M(\d+)_Kh(\d+)_pp(\d)", lab)
    if not m: continue
    shape, M, Kh, pp = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    d = {}
    for r in csv.DictReader([l for l in open(f) if not l.startswith("==PROF")]):
        if r.get("Metric Name"): d[r["Metric Name"]] = r.get("Metric Value")
    data[(shape, M, Kh, pp)] = d

def gv(d, k):
    try: return float(str(d.get(MET[k], "nan")).replace(",", ""))
    except: return float("nan")

print(f"{'shape':5} {'M':>5} {'Kh':>5} {'pp':>2} | {'TCela':>6} {'TCact':>6} {'warp%':>6} {'wait':>5} {'lsb':>5} {'ssb':>5} {'regs':>5} {'limR':>4} {'limS':>4} {'dur(us)':>8}")
keys = sorted(set((s, M, Kh) for (s, M, Kh, pp) in data))
order = {"qkv": 0, "down": 1, "gate": 2}
keys.sort(key=lambda x: (x[1], order.get(x[0], 9), x[2]))
for (s, M, Kh) in keys:
    for pp in (0, 1, 2):
        d = data.get((s, M, Kh, pp))
        if not d: continue
        print(f"{s:5} {M:>5} {Kh:>5} {pp:>2} | {gv(d,'tcE'):6.1f} {gv(d,'tcA'):6.1f} {gv(d,'warp'):6.1f} "
              f"{gv(d,'wait'):5.2f} {gv(d,'lsb'):5.2f} {gv(d,'ssb'):5.2f} {gv(d,'regs'):5.0f} "
              f"{gv(d,'limR'):4.0f} {gv(d,'limS'):4.0f} {gv(d,'dur')/1000:8.1f}")
    print()
