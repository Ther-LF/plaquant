import json,glob,sys
CONV={"piqa":"acc_norm","arc_easy":"acc_norm","arc_challenge":"acc_norm","hellaswag":"acc_norm","winogrande":"acc","lambada_openai":"acc"}
def avg6(res):
    vals={}
    for t,m in CONV.items():
        if t not in res: vals[t]=None; continue
        d=res[t]; vals[t]=d.get(m+",none", d.get(m))
    if any(v is None for v in vals.values()): return None,vals
    return sum(vals.values())/6, vals
def show(tag,avg,vals):
    print("[%s] AVG6=%.2f"%(tag,avg*100) if avg is not None else "[%s] INCOMPLETE"%tag)
    for t in CONV:
        v=vals.get(t); print("   %s: %.2f"%(t,v*100) if v is not None else "   %s: NA"%t)
mode=sys.argv[1]; path=sys.argv[2]
if mode=="vllm":
    f=glob.glob(path+"/*/results_*.json")[0]; r=json.load(open(f))["results"]; a,v=avg6(r); show(path,a,v)
elif mode=="peer":
    r=json.load(open(path)); zs=r.get("lmeval",{}).get("zeroshot",{})
    res={t:{"acc_norm,none":d.get("acc_norm"),"acc,none":d.get("acc")} for t,d in zs.items() if isinstance(d,dict)}
    a,v=avg6(res); show(path,a,v); print("   peer-avg field:",zs.get("average"))
    mm=r.get("lmeval",{}).get("mmlu",{})
    if mm:
        g=mm.get("groups",{}).get("mmlu")
        print("   MMLU micro=%s macro=%s group=%s"%(mm.get("micro_avg"),mm.get("macro_avg"),g))
elif mode=="coq":
    r=json.load(open(path)); a,v=avg6(r); show(path,a,v)
    if "mmlu" in r: print("   MMLU group acc =",r["mmlu"].get("acc,none"))
