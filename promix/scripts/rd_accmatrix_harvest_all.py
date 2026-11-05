import json,glob,os
O="/home/luofan/accmatrix"
CONV={"piqa":"acc_norm","arc_easy":"acc_norm","arc_challenge":"acc_norm","hellaswag":"acc_norm","winogrande":"acc","lambada_openai":"acc"}
def avg6(res):
    vals={}
    for t,m in CONV.items():
        if t not in res: return None
        d=res[t]; vals[t]=d.get(m+",none", d.get(m))
    if any(v is None for v in vals.values()): return None
    return sum(vals.values())/6
def peer_zs(p):
    r=json.load(open(p)); zs=r.get("lmeval",{}).get("zeroshot",{})
    res={t:{"acc_norm,none":d.get("acc_norm"),"acc,none":d.get("acc")} for t,d in zs.items() if isinstance(d,dict)}
    return avg6(res)
def peer_mmlu(p):
    r=json.load(open(p)); mm=r.get("lmeval",{}).get("mmlu",{})
    g=mm.get("groups",{}).get("mmlu"); return (g, mm.get("macro_avg"))
def vllm_zs(d):
    fs=glob.glob(d+"/*/results_*.json")
    if not fs: return None
    return avg6(json.load(open(fs[0]))["results"])
def vllm_mmlu(d):
    fs=glob.glob(d+"/*/results_*.json")
    if not fs: return None
    r=json.load(open(fs[0]))["results"]; return (r.get("mmlu",{}).get("acc,none"),)
def coq_zs(p):
    return avg6(json.load(open(p)))
def coq_mmlu(p):
    r=json.load(open(p)); return (r.get("mmlu",{}).get("acc,none"),)
JOBS=[("arc_8b 0-shot","peer_zs",O+"/arc_8b_zs.json"),
      ("sharq_8b 0-shot","peer_zs",O+"/sharq_8b_zs.json"),
      ("mm_8b 0-shot","peer_zs",O+"/mm_8b_zs.json"),
      ("mm_8b MMLU","peer_mmlu",O+"/mm_8b_mmlu.json"),
      ("coq_3b 0-shot","coq_zs",O+"/coq_3b_zs.json"),
      ("coq_8b 0-shot","coq_zs",O+"/coq_8b_zs.json"),
      ("coq_3b MMLU","coq_mmlu",O+"/coq_3b_mmlu.json"),
      ("coq_8b MMLU","coq_mmlu",O+"/coq_8b_mmlu.json"),
      ("nv_70b MMLU","vllm_mmlu",O+"/nv_70b_mmlu")]
for name,fn,path in JOBS:
    ok = os.path.isdir(path) and glob.glob(path+"/*/results_*.json") if fn.startswith("vllm") else os.path.isfile(path)
    if not ok: print("%-18s : pending"%name); continue
    try:
        v=globals()[fn](path)
        if v is None: print("%-18s : incomplete"%name)
        elif isinstance(v,tuple): print("%-18s : %s"%(name, " ".join("%.2f"%(x*100) if x is not None else "NA" for x in v)))
        else: print("%-18s : AVG6=%.2f"%(name, v*100))
    except Exception as e: print("%-18s : ERR %s"%(name,str(e)[:60]))
