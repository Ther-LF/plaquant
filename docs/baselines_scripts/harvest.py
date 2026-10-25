#!/usr/bin/env python3
"""Build a markdown summary table from /home/luofan/baselines/results/*.json."""
import glob, json, os, re, sys

RES = "/home/luofan/baselines/results"
LOGS = "/home/luofan/baselines/logs"


def backend_for(tag):
    """Pull the kernel/attention backend lines out of the run log."""
    p = os.path.join(LOGS, tag + ".log")
    if not os.path.exists(p):
        return ""
    txt = open(p, errors="ignore").read()
    pats = [
        r"Using (\S+) attention backend",
        r"Using (FlashInfer\S*|\S*Marlin\S*|\S*NvFp4\S*|\S*Cutlass\S*|\S*FP8\S*) for",
        r"Using (\S+) for mixed-precision linear",
    ]
    out = []
    for pat in pats:
        for m in re.findall(pat, txt):
            if m not in out:
                out.append(m)
    return ", ".join(out)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
        d = json.load(open(f))
        m = d.get("meta", {})
        tag = m.get("tag") or os.path.basename(f)[:-5]
        if "ttft8k" in tag:
            continue
        ppl = d.get("ppl", {}).get("ppl")
        dec = d.get("decode", {})
        tt = d.get("ttft", {})
        mem = d.get("memory", {})
        rows.append({
            "tag": tag,
            "ppl": ppl,
            "ttft_2048_b1": tt.get("L2048_b1", {}).get("median_s"),
            "ttft_4096_b1": tt.get("L4096_b1", {}).get("median_s"),
            "ttft_8192_b1": tt.get("L8192_b1", {}).get("median_s"),
            "ttft_8192_b8": tt.get("L8192_b8", {}).get("median_s"),
            "dec_b1": dec.get("b1", {}).get("tok_per_s"),
            "dec_b8": dec.get("b8", {}).get("tok_per_s"),
            "reserved_gib": mem.get("max_memory_reserved_gib"),
            "weight_bytes": d.get("weight_bytes"),
            "backend": backend_for(tag),
            "quant": m.get("quantization_resolved"),
        })
    hdr = ("| run | PPL | TTFT 2048 b1 (ms) | TTFT 4096 b1 (ms) | TTFT 8192 b1 (ms) | "
           "TTFT 8192 b8 (ms) | dec b1 (tok/s) | dec b8 (tok/s) | weights (GB) | backend |")
    print(hdr)
    print("|" + "---|" * 11)
    for r in rows:
        def ms(x):
            return f"{x*1000:.0f}" if x else "-"
        def ts(x):
            return f"{x:.0f}" if x else "-"
        def gb(x):
            return f"{x/1e9:.2f}" if x else "-"
        print(f"| {r['tag']} | {r['ppl']:.2f} | {ms(r['ttft_2048_b1'])} | {ms(r['ttft_4096_b1'])} | "
              f"{ms(r['ttft_8192_b1'])} | {ms(r['ttft_8192_b8'])} | {ts(r['dec_b1'])} | "
              f"{ts(r['dec_b8'])} | {gb(r['weight_bytes'])} | {r['backend']} |")


if __name__ == "__main__":
    main()
