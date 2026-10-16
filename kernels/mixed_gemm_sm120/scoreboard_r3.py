"""Round-3 scoreboard: correctness gate (cosine) then full bands for all shapes
at M in {1,16,256,2048} via b1_fused.benchmark_b1 (b1 auto vs rungB_presf vs
pure fp4/fp8 presf bounds -> band%). Confirms the M=256 cfg10/6/7 dispatch (T2)
and no M=2048/decode regression. Appends to benchmark.csv.
Run inside luofan-plaquant on a FREE/QUIET GPU.
"""
import b1_fused as b1

if __name__ == "__main__":
    b1.get_ext()
    b1.get_presf_ext()
    ok = b1.validate_fused(stages=0)
    if not ok:
        print("GATE FAILED — aborting scoreboard")
        raise SystemExit(1)
    b1.benchmark_b1("b1_r3", stages=0, ref_paths=True, bounds=True)
    print("SCOREBOARD DONE")
