import csv, glob, os, sys, statistics

D = sys.argv[1] if len(sys.argv) > 1 else "/home/luofan/plaquant/kernels/mixed_gemm_sm120/logs/traceR16b"
DUR = "gpu__time_duration.sum"

def tonum(s):
    s = s.strip().strip('"').replace(",", "")
    try: return float(s)
    except: return None

def parse_ordered(fn):
    with open(fn) as f:
        rows = list(csv.reader(f))
    hi = None
    for i, r in enumerate(rows):
        if "Kernel Name" in r and "Metric Name" in r and "Metric Value" in r:
            hi = i; break
    if hi is None: return []
    hdr = rows[hi]
    ki = hdr.index("Kernel Name"); mn = hdr.index("Metric Name"); mv = hdr.index("Metric Value")
    out = []
    for r in rows[hi+1:]:
        if len(r) <= max(ki, mn, mv): continue
        if r[mn].strip().strip('"') == DUR:
            out.append((r[ki].strip().strip('"'), tonum(r[mv])))
    return out

print(f"{'label':20s} {'scat_lo_us':>10} {'scat_hi_us':>10} {'scat_tot':>9} {'gemm_us':>9} "
      f"{'ncalls':>6} {'scat%dev':>9} {'dev_tot_us':>10}")
for fn in sorted(glob.glob(D + "/*.csv")):
    lbl = os.path.basename(fn)[:-4]
    kers = parse_ordered(fn)
    if not kers: continue
    # walk in order: accumulate scatter(fill_sf) durs; a cutlass device_kernel closes a call
    calls = []   # list of (scat_list, gemm)
    pend = []
    for name, dur in kers:
        if dur is None: continue
        if "fill_sf" in name:
            pend.append(dur)
        elif "device_kernel" in name or ("cutlass" in name and "gemm" in name.lower()):
            calls.append((pend, dur)); pend = []
    # keep calls that have exactly 2 scatters (steady dual-format calls); drop warmup partials
    full = [(s, g) for s, g in calls if len(s) == 2]
    if not full:
        # shipping: no scatter, just GEMM calls
        gemms = [g for s, g in calls]
        if gemms:
            g = statistics.median(gemms[2:] or gemms)
            print(f"{lbl:20s} {'-':>10} {'-':>10} {0.0:9.2f} {g/1e3:9.2f} "
                  f"{len(gemms):6d} {0.0:9.1f} {g/1e3:10.2f}")
        continue
    use = full[2:] if len(full) > 4 else full   # drop first 2 as warmup-in-window
    slo = statistics.median(s[0] for s, g in use)
    shi = statistics.median(s[1] for s, g in use)
    gm = statistics.median(g for s, g in use)
    scat = slo + shi
    dev = scat + gm
    print(f"{lbl:20s} {slo/1e3:10.2f} {shi/1e3:10.2f} {scat/1e3:9.2f} {gm/1e3:9.2f} "
          f"{len(use):6d} {100.0*scat/dev:9.1f} {dev/1e3:10.2f}")
