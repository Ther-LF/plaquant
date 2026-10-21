import csv, glob, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/home/luofan/plaquant/kernels/mixed_gemm_sm120/logs/tcR16"
M = {
 "TCela": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
 "TCact": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
 "tcyc":  "sm__pipe_tensor_cycles_active.sum",
 "ecyc":  "sm__cycles_elapsed.sum",
 "acyc":  "sm__cycles_active.sum",
 "SM":    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
 "DRAM":  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
 "dbytes":"dram__bytes.sum",
 "L2":    "lts__t_sector_hit_rate.pct",
 "L1":    "l1tex__t_sector_hit_rate.pct",
 "issue": "smsp__issue_active.avg.pct_of_peak_sustained_active",
 "tc":    "sm__inst_executed_pipe_tensor.sum",
 "alu":   "sm__inst_executed_pipe_alu.sum",
 "regs":  "launch__registers_per_thread",
 "dur":   "gpu__time_duration.sum",
}
INV = {v: k for k, v in M.items()}

def tonum(s):
    if s is None: return None
    s = s.strip().strip('"').replace(",", "")
    try: return float(s)
    except: return None

def parse(fn):
    d = {}
    with open(fn) as f:
        rows = list(csv.reader(f))
    # find header row containing "Metric Name"
    hi = None
    for i, r in enumerate(rows):
        if "Metric Name" in r and "Metric Value" in r:
            hi = i; break
    if hi is None: return d
    hdr = rows[hi]
    mn = hdr.index("Metric Name"); mv = hdr.index("Metric Value")
    for r in rows[hi+1:]:
        if len(r) <= max(mn, mv): continue
        name = r[mn].strip().strip('"')
        if name in INV:
            v = tonum(r[mv])
            if v is not None: d[INV[name]] = v
    return d

rows = []
for fn in sorted(glob.glob(D + "/*.csv")):
    lbl = os.path.basename(fn)[:-4]
    d = parse(fn)
    if "TCela" not in d: continue
    rows.append((lbl, d))

hdr = (f"{'label':20s} {'TCela':>6} {'TCact':>6} {'SM%':>5} {'DRAM%':>6} {'L2%':>5} "
       f"{'L1%':>5} {'issue':>6} {'alu/tc':>6} {'tcyc(M)':>8} {'ecyc(M)':>8} "
       f"{'chk_ela':>7} {'regs':>4} {'dur_us':>8}")
print(hdr)
for lbl, d in sorted(rows):
    tc = d.get('tc', 1) or 1
    tcyc = d.get('tcyc', 0); ecyc = d.get('ecyc', 1) or 1
    chk = 100.0 * tcyc / ecyc  # should ~= TCela (sanity: abs cycles vs pct rollup)
    print(f"{lbl:20s} {d.get('TCela',0):6.1f} {d.get('TCact',0):6.1f} "
          f"{d.get('SM',0):5.1f} {d.get('DRAM',0):6.1f} {d.get('L2',0):5.1f} "
          f"{d.get('L1',0):5.1f} {d.get('issue',0):6.1f} {d.get('alu',0)/tc:6.2f} "
          f"{tcyc/1e6:8.1f} {ecyc/1e6:8.1f} {chk:7.1f} {d.get('regs',0):4.0f} "
          f"{d.get('dur',0)/1e3:8.1f}")
