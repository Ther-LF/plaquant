"""P0-2 sub-task B: per-checkpoint tensor byte statistics from safetensors headers."""
import argparse, json, os, struct, glob

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "qkv_proj", "gate_up_proj")
DTYPE_BITS = {"F64": 64, "F32": 32, "F16": 16, "BF16": 16, "I64": 64, "I32": 32,
              "I16": 16, "I8": 8, "U8": 8, "F8_E4M3": 8, "F8_E5M2": 8,
              "F8_E8M0": 8, "BOOL": 8, "F4": 4, "U32": 32}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def suffix_of(name):
    parts = name.split(".")
    return parts[-1]


def classify(name):
    if ".layers." in name and any(p in name for p in PROJ):
        s = suffix_of(name)
        if s in ("weight", "qweight", "weight_packed", "weight_lo_packed",
                 "weight_hi_bytes"):
            return "lin_weight", s
        if "scale" in s or s == "scales":
            return "lin_scale", s
        if s in ("qzeros", "weight_zero_point", "zeros"):
            return "lin_zeros", s
        return "lin_meta", s
    return "other", suffix_of(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    out = {"tag": args.tag, "dir": args.dir, "by_cat_bytes": {},
           "by_suffix_bytes": {}, "by_suffix_count": {}, "dtypes": {},
           "lin_weight_elems_bf16like": 0, "other_tensors": {},
           "files_bytes": {}, "disk_total_bytes": 0}
    sts = sorted(glob.glob(os.path.join(args.dir, "*.safetensors")))
    assert sts, f"no safetensors in {args.dir}"
    for st in sts:
        hdr = read_header(st)
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            b = info["data_offsets"][1] - info["data_offsets"][0]
            cat, suf = classify(name)
            out["by_cat_bytes"][cat] = out["by_cat_bytes"].get(cat, 0) + b
            key = cat + ":" + suf + ":" + info["dtype"]
            out["by_suffix_bytes"][key] = out["by_suffix_bytes"].get(key, 0) + b
            out["by_suffix_count"][key] = out["by_suffix_count"].get(key, 0) + 1
            if cat == "lin_weight" and suf == "weight":
                n = 1
                for d in info["shape"]:
                    n *= d
                out["lin_weight_elems_bf16like"] += n
            if cat == "other":
                short = name if "layers." not in name else "layers_other"
                e = out["other_tensors"].setdefault(
                    short, {"bytes": 0, "dtype": info["dtype"]})
                e["bytes"] += b
    # disk: all files in dir
    for root, _, files in os.walk(args.dir):
        for fn in files:
            p = os.path.join(root, fn)
            out["files_bytes"][os.path.relpath(p, args.dir)] = os.path.getsize(p)
            out["disk_total_bytes"] += os.path.getsize(p)
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("tag", "by_cat_bytes", "disk_total_bytes")}))


if __name__ == "__main__":
    main()
