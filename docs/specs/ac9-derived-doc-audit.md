# AC-9 Derived-Doc Consistency Audit

**Status**: derived-doc clause fully met (zero stale instances).
Plan-source clause is plan-immutability-blocked and tracked
separately as a blocking side issue.

**Reproducible**: re-run the three documented `grep` commands
below; the test
`tests/test_fake_quant_mxfp8_nvfp4.py::test_no_stale_ac9_terms_in_derived_docs`
codifies the same audit.

## Acceptance Criterion (verbatim)

> **AC-9**: Draft documentation consistency cleanup — derived
> docs use MXFP8=E8M0 (not E4M3), atom name
> `SM100_MMA_MXF8F6F4_*` (not F8F6F4), corrected β-4 alignment
> example pairs.

## Audit Method

### Scope

**In scope** (derived docs subject to AC-9 cleanup):

- `docs/` and `docs/specs/` — all `*.md` files except the binary
  `*.pdf` artifacts.
- `docs/superpowers/` — historical design specs and plans.
- `kernels/**/*.md`, `kernels/**/README*` — kernel-side docs.
- `promix/**/*.py` — docstrings and inline comments.
- `promix/configs/*.yaml` — config comments.
- `tests/**/*.py` — test docstrings and inline comments.
- Root-level `*.md` (`README.md`, `CLAUDE.md`).

**Out of scope** (plan source / idea draft, plan-immutability-blocked):

- `.humanize/plans/plaquant-sm100-fp8-nvfp4.md`
- `.humanize/ideas/resq-mixed-precis-20260621-154239.md`
- All other `.humanize/` files.

**Out of scope** (vendored third-party):

- `third_party/`
- `project-resq/`
- `sglang/`
- `.git/`

### Search Terms

Three audit terms, one per AC-9 clause:

1. **MXFP8 scale = E4M3** (incorrect; should be E8M0):
   ```
   grep -rinE 'MXFP[ -]?8.*E4M3|E4M3.*MXFP[ -]?8' \
        docs/ kernels/ promix/ tests/ README.md CLAUDE.md
   ```

2. **Atom name `SM100_MMA_F8F6F4_*` without MX prefix**
   (incorrect in PRIMARY context; PRIMARY uses microscaled
   `SM100_MMA_MXF8F6F4_*`):
   ```
   grep -rnE 'SM100_MMA_F8F6F4|\bF8F6F4_S?S?\b' \
        docs/ kernels/ promix/ tests/ README.md CLAUDE.md \
     | grep -v MXF8F6F4
   ```

3. **β-4 alignment example pair `(K_high=32, K_low=1984)`**
   (incorrect; sums to 2016 not 2048; correct minimum is
   `K_high=64`):
   ```
   grep -rnE '\(32,\s*1984\)|32\s*\+\s*1984|K_high\s*=\s*32' \
        docs/ kernels/ promix/ tests/ README.md CLAUDE.md
   ```

### Classification Rules

Each hit is classified into one of four categories:

- **(a) corrective citation** — text identifying the wrong value
  in another document and stating the correct one (e.g. spec's
  "the original idea draft mistyped MXFP8 scale as E4M3"). KEEP.
- **(b) correct canonical usage** — uses the correct value (e.g.
  "MXFP8 element dtype is E4M3" — element IS E4M3; only the
  scale is E8M0). KEEP.
- **(c) stale derived-doc instance** — derived doc using the
  wrong value as if canonical. FIX.
- **(d) plan-source instance** — hit in the immutable plan or
  idea draft. DEFER to plan-cleanup session.

## Audit Results

### Term 1: MXFP8 scale = E4M3

| File:Line | Excerpt | Class |
|-----------|---------|-------|
| `docs/specs/cutlass-sm100-atom-references.md:103` | "the original idea draft had several lines saying MXFP8 scale was E4M3 (incorrect). Per the trait excerpts above, MXFP8 scale is **E8M0**" | (a) |
| `docs/specs/spec-mxfp8-nvfp4.md:50` | "the original idea draft had mistyped (some lines said MXFP8 scale was E4M3 — that is wrong; MXFP8 = E8M0, NVFP4 = E4M3)" | (a) |
| `docs/specs/spec-mxfp8-nvfp4.md:103` | "the block scales (E8M0 for MXFP8, E4M3 for NVFP4) are already inside `acc_high` and `acc_low`" | (b) — explicit correct distinction |
| `docs/specs/spec-mxfp8-nvfp4.md:120` | "max_format_value (MXFP8 E4M3: 448; NVFP4 E2M1: 6)" | (b) — element dtype context |
| `docs/specs/spec-mxfp8-nvfp4.md:136` | "MXFP8 E4M3 there is no Inf either (max is 448 finite)" | (b) — element dtype context |
| `docs/specs/spec-mxfp8-nvfp4.md:143` | "MXFP8 E4M3 does not represent Inf either" | (b) — element dtype context |
| `docs/specs/spec-mxfp8-nvfp4.md:215` | "analogous set with E4M3 representable values for MXFP8" | (b) — element dtype context |
| `promix/quantize/quant_utils.py:635` | docstring: "fmt: 'mxfp8' (block_size=32, FP8 E4M3 element grid)" | (b) — element dtype context |
| `promix/quantize/quant_utils.py:665` | docstring: "MXFP8 (block_size=32, FP8 E8M0 block scale, FP8 E4M3 elements)" | (b) — explicit correct distinction |
| `promix/quantize/quant_utils.py:684` | "MXFP8 E4M3 only; E5M2 is a future option" | (b) — element dtype context |

**Classes (c) found: 0.**

### Term 2: Atom name `SM100_MMA_F8F6F4_*` (without MX prefix)

| File:Line | Excerpt | Class |
|-----------|---------|-------|
| `docs/specs/cutlass-sm100-atom-references.md:105` | "the original draft β-2 phase plan said `SM100_MMA_F8F6F4_SS` (dense, no microscaling). The PRIMARY direction throughout the rest of the plan calls for `SM100_MMA_MXF8F6F4_*` (microscaled). ... β-2's F8F6F4 mention is a textual bug that AC-9 cleanup must fix." | (a) — corrective citation contrasting the two atom families |

**Classes (c) found: 0.**

### Term 3: β-4 alignment pair `(K_high=32, K_low=1984)`

| File:Line | Excerpt | Class |
|-----------|---------|-------|
| `docs/specs/cutlass-sm100-atom-references.md:107` | "**β-4 alignment example pair `(K_high=32, K_low=1984)`**: 32 + 1984 = 2016 ≠ 2048. ... The smallest valid K_high for K_total=2048 is K_high=64 (K_low=1984). The example pair must be removed or corrected." | (a) — corrective citation |

**Classes (c) found: 0.**

## Conclusion

| Clause | Status | Action |
|--------|--------|--------|
| Derived docs use MXFP8=E8M0 (not E4M3) | **MET** | 0 (c)-class hits; 7 (b)-class correct uses; 2 (a)-class corrective citations |
| Atom name `SM100_MMA_MXF8F6F4_*` (not F8F6F4) | **MET** | 0 (c)-class hits; 1 (a)-class corrective citation contrasting the dense F8F6F4 vs the microscaled MXF8F6F4 family |
| Corrected β-4 alignment example pairs | **MET** | 0 (c)-class hits; one (a)-class corrective citation in `cutlass-sm100-atom-references.md` |

**AC-9 derived-doc clause: fully met.** Audit ran on commit `c8cde4d` and re-verified on the round-17 working tree.

## Plan-Source Items (Out-of-Scope, Tracked Separately)

The following plan-immutability-blocked items remain open and
are tracked in the goal-tracker's Blocking Side Issues:

1. `.humanize/plans/plaquant-sm100-fp8-nvfp4.md:29` —
   "Block scale itself is computed by `block_scale = max_abs(block) / max_format_value`, then **rounded down** to the nearest representable scale value"
   — should be **rounded UP** per
   `BL-20260623-block-scale-rounding-direction`. The corrected
   spec at `docs/specs/spec-mxfp8-nvfp4.md` §6 has the right
   behavior; the plan source is the only file disagreeing.

2. `.humanize/plans/plaquant-sm100-fp8-nvfp4.md:484-486` —
   "Draft Inconsistency Reconciliation (AC-9 detail)" section
   listing the original idea draft's three textual issues by
   line number. These line citations point at the idea draft
   (`.humanize/ideas/resq-mixed-precis-20260621-154239.md`), not
   at the plan or derived docs. The citations are accurate
   historical references; they do not introduce a stale value
   into the plan itself.

3. `.humanize/plans/plaquant-sm100-fp8-nvfp4.md:581` —
   atom-comparison table row "FP8/FP6/FP4 (dense, no microscaling) | `kind::f8f6f4` | ✓ | SM100_MMA_F8F6F4_*"
   — this is a (b)-class correct usage (table compares the dense
   F8F6F4 family vs the microscaled MXF8F6F4 family explicitly).
   No fix needed.

The only plan-source content that materially conflicts with the
spec is item #1 (line 29's "rounded down"). Resolution requires
a permitted plan-cleanup session per the RLCR plan-immutability
rule. Until that session occurs, the BitLesson note in
`docs/specs/spec-mxfp8-nvfp4.md` §6 (Reconciliation Note) plus
this audit doc serve as the authoritative override.

## Idea-Draft Items (Out-of-Scope, Historical)

`.humanize/ideas/resq-mixed-precis-20260621-154239.md` is the
original loose-input draft; per the gen-idea workflow it is
superseded by the plan and the spec. The plan's
"Draft Inconsistency Reconciliation" section (lines 482-486)
explicitly inventories the three textual issues in this draft.
No fix needed — the draft is historical input, not a current
source of truth.

## Regression Prevention

`tests/test_fake_quant_mxfp8_nvfp4.py::test_no_stale_ac9_terms_in_derived_docs`
re-runs the three audit greps over the in-scope files and fails
if any new derived doc reintroduces a (c)-class stale instance.
This catches regressions in future rounds.
