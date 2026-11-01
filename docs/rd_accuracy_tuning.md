# rd-acctune: Accuracy Tuning Campaign (push PPL below the shipped anchors)

**Branch:** `feat/rd-acctune` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant`, remote dir
`/home/luofan/plaquant-acctune` · **Date:** 2026-11-01 · **Status: IN PROGRESS**

## Mission

Shipped accuracy anchors (dense-R-on, measured-MSE @1/8, our protocol):

| model | wiki (shipped) | C4 (shipped) | bf16 bridge |
|---|---|---|---|
| 1B | 14.4717 | 20.1864 | 13.1569 |
| 3B | 8.2678 | 11.3194 | 7.8148 |
| 8B | 6.5882 (uniform; measured 6.6072) | 9.6355 | 6.1368 |

Measure each lever ISOLATED on 3B, stack winners, validate the stack on
1B/3B/8B (wiki + C4). Protocol gates: bridges 13.1569 / 7.8148 / 6.1368 must
reproduce (PASS, see below).

## Protocol gates

| gate | expected | measured | verdict |
|---|---|---|---|
| 3B bf16 bridge | 7.8148 | 7.8148 | PASS |
| 1B bf16 bridge | 13.1569 | 13.1569 | PASS |
| 3B dense+measured @1/8 | 8.2678 | TBD | — |
| 1B dense+measured @1/8 | 14.4717 | TBD | — |

## Levers (isolated, 3B wiki unless noted)

(TBD — per-lever table)

## Best stack (1B/3B/8B × wiki/C4) vs shipped anchors

(TBD)

## Reproduce

```bash
# one point per invocation (env knobs isolate per process)
python promix/scripts/rd_acctune.py --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
    --point 3b_ring512_measured --substrate ring --ring_size 512 --alloc measured \
    --out /home/luofan/acctune/P.json
```
Raw JSONs: `/home/luofan/acctune/*.json` (remote).
