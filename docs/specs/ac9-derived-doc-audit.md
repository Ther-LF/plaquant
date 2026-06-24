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

The audit and the regression test scan the same universe: every
**tracked** file (per `git ls-files`) whose path ends in one of
the AC-9-relevant extensions and whose path is not under an
excluded prefix.

**In scope — extensions** (derived docs + code comments + configs
+ tests + kernel sources):

- `.md`, `.txt` — derived docs, READMEs, design notes
- `.py` — Python source / docstrings / inline comments
- `.yaml`, `.yml` — yaml configs
- `.cu`, `.cuh`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.inl` —
  CUDA / C++ kernel sources and headers (covers current and
  future M2 work in `kernels/mixed_gemm_sm100/`)

**Out of scope — path prefixes**:

- `.humanize/` — plan source + idea draft + RLCR loop state
  (plan-immutability rule; tracked separately as a blocking
  side issue requiring user-side action).
- `third_party/` — vendored CUTLASS, DeepGEMM, flashinfer,
  comet-25, ResComp, mixed_tensor, quack, LLM-Infra-Reference.
- `project-resq/` — vendored ResQ submodule.
- `sglang/` — vendored sglang submodule.

**Out of scope — self-references** (excluded because they name
the audit terms by definition, not as canonical or stale usage):

- `docs/specs/ac9-derived-doc-audit.md` (this doc)
- `tests/test_fake_quant_mxfp8_nvfp4.py` (the regression test)

Untracked files — `__pycache__`, `build/`, `.venv/`, generated
caches — are automatically excluded by `git ls-files`.

### Search Terms

Three audit terms, one per AC-9 clause. Each command uses
`git ls-files` so the scan universe matches the regression test
exactly.

A reusable filter pipeline is defined once:

```bash
ac9_files() {
    git ls-files \
      | grep -E '\.(md|txt|py|yaml|yml|cu|cuh|cpp|cc|cxx|h|hpp|inl)$' \
      | grep -vE '^(\.humanize|third_party|project-resq|sglang)/' \
      | grep -vE '^(docs/specs/ac9-derived-doc-audit\.md|tests/test_fake_quant_mxfp8_nvfp4\.py)$'
}
```

1. **MXFP8 scale = E4M3** (incorrect; should be E8M0):
   ```
   ac9_files | xargs grep -inE 'MXFP[ -]?8.*E4M3|E4M3.*MXFP[ -]?8'
   ```

2. **Atom name `SM100_MMA_F8F6F4_*` without MX prefix**
   (incorrect in PRIMARY context; PRIMARY uses microscaled
   `SM100_MMA_MXF8F6F4_*`):
   ```
   ac9_files | xargs grep -nE 'SM100_MMA_F8F6F4|\bF8F6F4_S?S?\b'
   ```

   Note: this command does **not** include a `grep -v MXF8F6F4`
   filter. The corrective citation at
   `docs/specs/cutlass-sm100-atom-references.md:105` legitimately
   contains both `SM100_MMA_F8F6F4_*` and `SM100_MMA_MXF8F6F4_*`
   on the same line (it explicitly contrasts the dense vs
   microscaled atom families); filtering it out would suppress
   the audit's primary corrective evidence. Classification
   (a/b/c/d) is a separate step from the scan; the raw scan must
   surface every match so the classifier can see all of them.

3. **β-4 alignment example pair `(K_high=32, K_low=1984)`**
   (incorrect; sums to 2016 not 2048; correct minimum is
   `K_high=64`):
   ```
   ac9_files | xargs grep -nE '\(32, ?1984\)|32 ?\+ ?1984|K_high ?= ?32\b'
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
| `docs/specs/cutlass-sm100-atom-references.md:160` | "scale dtype = E8M0 for MXFP8 / MXFP4-VS-32 (asserted in trait) and E4M3 for NVFP4-VS-16 (downstream-pinned)" — round-19 verification checklist | (b) — explicit correct distinction (same pattern as `spec-mxfp8-nvfp4.md:103`) |

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
| Derived docs use MXFP8=E8M0 (not E4M3) | **MET** | 0 (c)-class hits; 8 (b)-class correct uses; 2 (a)-class corrective citations |
| Atom name `SM100_MMA_MXF8F6F4_*` (not F8F6F4) | **MET** | 0 (c)-class hits; 1 (a)-class corrective citation contrasting the dense F8F6F4 vs the microscaled MXF8F6F4 family |
| Corrected β-4 alignment example pairs | **MET** | 0 (c)-class hits; one (a)-class corrective citation in `cutlass-sm100-atom-references.md` |

**AC-9 derived-doc clause: fully met** (zero (c)-class hits across
the corrected scan universe). Audit method first authored in
round 17; the documented commands and the regression test were
unified onto a single tracked-files-only `git ls-files` pipeline
in round 18 so the documented commands reproduce the documented
results bit-for-bit (round-17 review found the original Term 2
command included a `grep -v MXF8F6F4` filter that suppressed the
corrective citation at line 105 and the audit-doc-vs-test scan
universes diverged on `__pycache__`).

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

## Audit History

The audit method and result tables were built up over four
RLCR rounds. Tracking the history here so future readers can
trace which round introduced or modified each entry.

- **Round 17** (initial authoring). Three audit terms defined;
  scan ran via `os.walk` over `docs/`, `kernels/`, `promix/`,
  `tests/`, root `*.md`. Allowlist seeded with 12 (a)/(b)-class
  entries: 10 in Term 1, 1 in Term 2, 1 in Term 3. Codex
  round-17 review found two operational defects: the
  documented Term 2 grep used `grep -v MXF8F6F4` which
  filtered out the corrective citation the audit doc claimed,
  and the audit doc and the regression test scanned different
  universes (raw `grep` vs `os.walk` with `__pycache__`
  exclusion).
- **Round 18** (universe unification). Documented commands
  and the regression test were both rewritten onto a shared
  tracked-files-only `git ls-files` pipeline (same extension
  list, same excluded path prefixes, same self-exclusion).
  Term 2 filter dropped. Extension list expanded to also
  cover `.cu`, `.cuh`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`,
  `.inl` (for future M2 kernel sources). Codex round-18
  review independently reproduced the documented 10/1/1 hit
  split and accepted the derived-doc clause as MET.
- **Round 19** (verification side-effect addition). Round 19
  added a Verification section to
  `docs/specs/cutlass-sm100-atom-references.md` that includes
  the audit term "MXFP8 / E4M3" in a (b)-class correct-
  distinction context (line 160). The allowlist was extended
  with this entry. The regression test caught the new hit
  immediately at edit time, proving the round-18 expanded
  scope works.
- **Round 20** (no audit-doc edit). Round 20 fixed the NVFP4
  trait-table M-dimension row in
  `docs/specs/cutlass-sm100-atom-references.md` but the
  edited rows (32, 67-68) are above the audit-cited section.
  Audit-cited line numbers (103, 105, 107, 160) were
  unchanged; allowlist did not need updating.
- **Round 21** (this provenance section). Audit method and
  result tables unchanged. This section added so future
  readers can trace the chain.

## Regression Prevention

`tests/test_fake_quant_mxfp8_nvfp4.py::test_no_stale_ac9_terms_in_derived_docs`
uses the same tracked-files-only `git ls-files` pipeline as the
documented audit commands above (same extension list, same
excluded path prefixes, same self-exclusion list) and fails
if any new derived doc reintroduces a (c)-class stale instance.
This catches regressions in future rounds.
