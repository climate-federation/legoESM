# SI3 Ladder Round 4 (`si3_r4`) Receipt — Rung O1

## Outcome

O1 is **LANDED**. The kept completed run at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round4/refused/oracle_o1_orca2_sas_ice_admission_refused_20261024T161410Z_1010784`
was re-admitted with `--admit-existing`; NEMO was not rebuilt or rerun. Admission
was pinned to producer commit `4fefbe7cba0afbc76f26bc94a744b5650b26285f`, which was a pre-rebase commit and is no longer reachable (Amendment, 2026-10-29 provenance audit),
the recorded inputs, and both recorded executable digests. The repaired gate is
intentionally newer than the producer (`gate_changed_since_run=true`).

The operator's real same-program proof is unchanged: `303` object files and
`291` preprocessed sources are identical; only observer units
`icedyn_adv_pra`, `icestp`, and writer unit `o1_si3_state_writer` differ. The
reference executable SHA-256 is
`dba67f3c81a08dced5c975d26317c637a57b5362237b9121606d1c8f578f7a3e` and
the observer executable SHA-256 is
`bfaf323f41d10b5bf959769d7586d3c2021cafa4dc5be0165cd8606ba7aa56f7`.
Admission checked `2` binaries and `7` pinned inputs.

## Diagnosis — CONFIRMED versus PLAUSIBLE

**CONFIRMED:** the real `kt=1` and `kt=2` `sss_m` arrays are not bit-identical.
Both have shape `(184, 152)` and contain `27,968` binary64 values. Exactly
`16,849` elements differ bit-for-bit; `max|kt2-kt1|` is
`0.20845785140991246`. Their payload SHA-256 values are, respectively,
`c7dbd409805477751751b9c07438f7f3ce9a5d7f8be8b6c50c0f97046949fee0`
and `28ec7cae3d892eb21a086e2da36ad5f47f3d696b5411da8ac0f6102baf9f64d7`.
The committed admission gate measures and records this comparison
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1770-1802`,
`:1970-1977`).

**CONFIRMED:** the old `kt2-sss-step1` substitution was vacuous on this real
record even though the fields differ. O1 resolves `nn_icesal=4`,
`rn_simin=0.1`, and `rn_sinew=0.75`. At an icy category NEMO computes `sz_i`
from `szv_i / (v_i * r1_nlay_i)` and does not consume `sss_m`; at an ice-free
category it computes `MIN(rn_simin, rn_sinew * sss_m)`
(`src/ICE/icevar.F90:362-375`; compiled
`ORCA2_SAS_ICE_O1_REC/BLD/ppsrc/nemo/icevar.f90:376-389`). The branch point is
therefore `0.13333333333333333`, far below both real forcing snapshots at the
scored cells, so both produce `0.1`. Substitution changed no scored bit.

**CONFIRMED:** the replacement `kt2-sss-perturb` plant changes the actual
`kt=2` field at the first interior wet cell with an ice-free category,
zero-based `(2, 21)`, to one ULP below that branch point. It fails on the kept
real record at the first compared statement, equivalent `s_i`, with
`wet_ULP=5`, `full_ULP=5`, and normalised error `6.939e-17`. This both proves a
live scored dependency and retains the selection check: if the step regresses
to the `kt=1` field, changing only `sss_step2` becomes vacuous and the
acquisition refuses (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1782-1802`).

**PLAUSIBLE:** none. No unmeasured explanation is used in the O1 verdict.

## Preregistered O1 score

The score ran on `cpu`, binary64, production `jax.jit`, with `XLA_FLAGS=""`;
the permanent admission JSON stamps `execution_backend="cpu"`,
`execution_x64=true`, and `execution_jit=true`. The gate scores every
prognostic field, Prather moment, equivalent field, and aggregate on wet
T/U/V cells, including NEMO-defined cyclic seam and north-fold halos; only
reduced `A2D(0)` diagnostics remain interior-only
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1300-1396`).

- Continuous bridge `kt=1`: maximum `0` ULP, maximum normalised error `0.0`.
- Continuous bridge `kt=2`: maximum `0` ULP, maximum normalised error `0.0`.
- Restart entry: maximum `0` ULP.
- Restart advance to `kt=2`: maximum `0` ULP, maximum normalised error `0.0`.
- Restart `kt=2` versus continuous `kt=2`: maximum `0` ULP.
- NetCDF restart variables: every registered wet-cell comparison is `0` ULP.
- First interior wet T cell: zero-based `(2, 21)`; first interior dry T cell:
  `(2, 2)`. Dry `-0.0`, dry nonzero, and halo fixture perturbations score
  `0` ULP; wet `-0.0` and wet one-ULP plants score `1` ULP.

There is no first failing production statement: all preregistered bridge,
restart, seam, and north-fold comparisons are exact. The first failure exists
only in the deliberate plant and is `s_i`, sourced by NEMO's ice-free
`sz_i` assignment and layer accumulation
(`src/ICE/icevar.F90:368-375`; compiled
`ORCA2_SAS_ICE_O1_REC/BLD/ppsrc/nemo/icevar.f90:382-389`).

The permanent `o1_admission.json` SHA-256 is
`c1c2ea1cb78fe9347e2db9096a39107b7b1c093b0676931aeeee0cff675ca963`.
The successful final `--admit-existing` log SHA-256 is
`4c3f1dde360726da4a91df65263bd94037776e537f073a927b89bf69de0e4b76`.
Preflight checklist items `1-7` and all `36` preflight plants pass; all `13`
admission plants exit nonzero. The focused O1 suite passes `15/15`; the
post-change real-admission unit passes `1/1` (`14` deselected). The default
receipt citation gate and private-import ratchet pass.

## GPU spot-check for the operator

Run only after the two `nvidia-smi` queries show an idle device; replace
`<idle-gpu-index>` without changing any scientific option:
Amendment (2026-10-29 provenance audit): the `--expect-commit` value below was a pre-rebase commit and is no longer reachable.
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
CUDA_VISIBLE_DEVICES=<idle-gpu-index> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_FLAGS='' \
PYTHONPATH=.:packages/core:packages/ocean:packages/ice:packages/atmosphere:packages/land:packages/coupler:packages/driver \
/home/dbalwada/legoESM/.venv/bin/python \
scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py \
--admit-evidence /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round4/refused/oracle_o1_orca2_sas_ice_admission_refused_20261024T161410Z_1010784 \
--expect-commit 4fefbe7cba0afbc76f26bc94a744b5650b26285f \
--json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round4/o1_gpu_spotcheck.json
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

## Choices and review

**CHOICES:** `ASKED` = diagnose the refused real record, replace the vacuous
plant, retain exact `kt=2` forcing selection, re-admit without rerunning NEMO,
and score O1 under the preregistered CPU/JIT/halo contract. `UNASKED` = none.
No model default, scientific selector, bar, window, mask, or record changed.

Search-before-build: searched the existing O1 admission path, plant loop,
direct test, and retained record reader; extended those paths and added no
duplicate implementation. Independent review was unavailable: the required
read-only Codex CLI failed to initialize its app-server client because the
review sandbox filesystem was read-only.

## Operator amendments after final review (Claude LAND, Argon pending/LAND)

- The admission run's logs and probe outputs lived only in /tmp; the operator copied them into
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round4/admission_evidence/` with `SHA256SUMS`.
- The jit scoring is NOT what the preregistration specified (it names jit only for gradients); the reviewer
  re-scored both eager and under jit and both are 0 ULP, so the result does not depend on that choice.
- The admitted record still sits under `refused/` (where the first, plant-refused admission parked it) and its
  input links point at the old path; it is the admitted O1 record nonetheless (pinned producer 4fefbe7cba,
  binary and input digests verified by the reviewer).
- Reviewer recomputation (CONFIRMED): kt 1, kt 2, restart entry, restart kt 2 and restart-vs-continuous kt 2
  all 0 ULP over 107 field arrays on 16,433 interior, 281 seam and 135 fold wet T cells; 40 restart variables at
  kt 1 and kt 2 at 0 ULP. The old kt2-sss plant was vacuous because all 13,145 wet cells with an ice-free category
  sit above the 0.1333 branch point at both steps; the replacement plant fails on the real record (s_i off by 5 ULP).
- Rebase conflict report (one file, three hunks): in the `FILES` hunk, kept the landed Round 5 phase2b-year entry and
  appended every Round 4 file entry; in the `CITATION_MAP` hunk, kept every landed Round 5 entry and its re-anchored
  Round 3 keys, appended every Round 4 entry, and omitted only the superseded duplicate Round 3 keys; in the `_NAME`
  matcher hunk, retained both Round 4 and Round 5 receipt names. At that conflict the combined map had 2,541 keys
  and zero duplicates; subsequent O1 commits bring the final map to 2,556 keys, still with zero duplicates.
- Post-rebase parent control: the O1 invariance test's parent pin was re-anchored from the old lane base `a572f4b25403`
  to the requested landed Round 5 tip `071dfa844369`; otherwise Round 5's intended one-ULP D5 `e_ice` change was
  incorrectly charged to O1. No assertion, plant, comparison field, tolerance, model, tool, or receipt claim changed.
