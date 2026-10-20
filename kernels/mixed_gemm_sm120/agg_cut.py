import csv, glob, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else "logs/tcC"
E = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
A = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
U = "gpu__time_duration.sum"
for f in sorted(glob.glob(os.path.join(D, "*.csv"))):
    lab = os.path.basename(f)[:-4]; d = {}
    for r in csv.DictReader([l for l in open(f) if not l.startswith("==PROF")]):
        if r.get("Metric Name"): d[r["Metric Name"]] = r.get("Metric Value")
    def g(k):
        try: return float(str(d.get(k, "nan")).replace(",", ""))
        except: return float("nan")
    print(f"{lab:20} TCela={g(E):6.1f} TCact={g(A):6.1f} dur={g(U)/1000:8.1f}us")
