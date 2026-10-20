import csv, glob, os, sys
D = "/home/luofan/plaquant/kernels/mixed_gemm_sm120/logs/tcSFILV"
KEYS = {
 "gpu__time_duration.sum":"dur_ns",
 "launch__registers_per_thread":"regs",
 "sm__inst_executed.sum":"inst",
 "sm__inst_executed_pipe_alu.sum":"alu",
 "sm__inst_executed_pipe_fma.sum":"fma",
 "sm__inst_executed_pipe_lsu.sum":"lsu",
 "sm__inst_executed_pipe_tensor.sum":"tc",
 "sm__inst_executed_pipe_xu.sum":"xu",
 "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active":"TCact",
 "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed":"TCela",
 "smsp__issue_active.avg.pct_of_peak_sustained_active":"issue",
 "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio":"lsb",
 "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio":"ssb",
 "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio":"wait",
}
def parse(fn):
    d={}
    with open(fn) as f:
        for row in csv.reader(f):
            if len(row)<15: continue
            m=row[-3]; v=row[-1]
            if m in KEYS:
                try: d[KEYS[m]]=float(v)
                except: pass
    return d
rows=[]
for fn in sorted(glob.glob(D+"/*.csv")):
    lbl=os.path.basename(fn)[:-4]
    d=parse(fn)
    if not d or "inst" not in d: continue
    rows.append((lbl,d))
hdr=f"{'label':24s} {'TCact':>6} {'TCela':>6} {'issue':>6} {'alu/tc':>7} {'lsu/tc':>7} {'fma/tc':>7} {'inst(M)':>8} {'alu(M)':>8} {'lsu(M)':>8} {'tc(M)':>7} {'regs':>5} {'wait':>5} {'lsb':>5} {'dur_us':>8}"
print(hdr)
for lbl,d in rows:
    tc=d.get('tc',1) or 1
    print(f"{lbl:24s} {d.get('TCact',0):6.1f} {d.get('TCela',0):6.1f} {d.get('issue',0):6.1f} "
          f"{d.get('alu',0)/tc:7.2f} {d.get('lsu',0)/tc:7.2f} {d.get('fma',0)/tc:7.2f} "
          f"{d.get('inst',0)/1e6:8.1f} {d.get('alu',0)/1e6:8.1f} {d.get('lsu',0)/1e6:8.1f} {tc/1e6:7.1f} "
          f"{d.get('regs',0):5.0f} {d.get('wait',0):5.2f} {d.get('lsb',0):5.2f} {d.get('dur_ns',0)/1e3:8.1f}")
