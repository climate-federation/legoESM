# Codex-internal Round-11 source review

Reviewer identity: `/root/round11_source_review`  
Commit reviewed: `b135e32ac73b5635f535faf3b74a2b7108db2bc4`  
Verdict: **SHIP**

## Findings

- No blocking findings.
- LOW: the receipt said the libm plant “changes 13,468 scored rows.”  The
  reproduced result is specifically 13,468 over-bar rows and 13,902 non-bit
  rows.  The final documentation corrects that ambiguity.

## Verified

- All five imported blobs are byte-identical to `61180a6776c`; subsequent
  commits do not alter them.
- Import commit `e2ad2c30629` names the source commit and contains only the
  requested paths.
- `nemo_source_round` has one implementation in `core/source_rounding.py` and
  all three callers use it.
- The only executing certified-path exponentials are the dry/melt albedo
  expressions corresponding to `icealb.F90:167-169`; both route through shared
  `transcendentals.exp`.
- The C1D card explicitly carries
  `PrecisionPolicy.fp64(transcendentals="libm")`; the gate installs it before
  tracing.
- The reviewer reproduced 271,560/271,560 bit-identical rows, zero non-bit,
  and zero over-bar under CPU/fp64/libm.  The JSON SHA-256 recomputed as
  `9c6040ef1824c5d1c1a9a261afd273358e55dbe07455bf17b31739e87d4e5712`.
- The poisoned-libm plant exits 1 and propagates into albedo, `qsr_ice`, and
  `qsr_tot`.
- Reviewer tests: `85 passed in 34.78s`; touched-package constants ratchet:
  `7 passed in 0.74s`.
- Receipt source, forcing, archive, stream, and accepted-artifact hashes
  recomputed correctly.
- The mandatory RGB input files are absent, only forbidden `CHLA_BATS.nc` is
  present, no runtime output is committed, and no shipped NEMO path changed.

## UNVERIFIED

- External page wording, expiry date, and remote URL availability were not
  browsed independently by this reviewer.
- The separate 21-test regression and repository-wide ratchet attempt were not
  rerun.
- Rung 3.6 is intentionally unmeasured pending its two mandatory real inputs.
