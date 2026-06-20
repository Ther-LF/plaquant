#!/usr/bin/env python3
"""Lightweight GPU keep-alive: runs a small matmul on every visible GPU
every few seconds so the cluster does not reap the node for being idle.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/keepalive_matmul.py
    # or just:
    python scripts/keepalive_matmul.py

Stop with Ctrl-C, or `pkill -f keepalive_matmul.py`.
"""
import argparse
import os
import threading
import time

import torch


def worker(device_id: int, size: int, sleep_s: float, log_every: int) -> None:
    torch.cuda.set_device(device_id)
    dev = f"cuda:{device_id}"
    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)
    iters = 0
    while True:
        c = a @ b
        torch.cuda.synchronize(device_id)
        del c
        iters += 1
        if iters % log_every == 0:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] gpu{device_id} iter={iters}", flush=True)
        time.sleep(sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048,
                        help="matmul side length (default 2048)")
    parser.add_argument("--sleep", type=float, default=5.0,
                        help="seconds between matmuls per GPU (default 5)")
    parser.add_argument("--log-every", type=int, default=60,
                        help="log heartbeat every N iters per GPU (default 60)")
    args = parser.parse_args()

    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("no CUDA devices visible — nothing to keep alive")

    print(f"keepalive: {n} GPUs, matmul {args.size}x{args.size} fp16 "
          f"every {args.sleep:.1f}s", flush=True)

    threads = []
    for i in range(n):
        t = threading.Thread(
            target=worker,
            args=(i, args.size, args.sleep, args.log_every),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("keepalive: stopping", flush=True)


if __name__ == "__main__":
    main()
