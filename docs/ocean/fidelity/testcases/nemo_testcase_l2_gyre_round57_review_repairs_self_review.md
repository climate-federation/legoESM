# Round 57 self-review — round-56 repair packet

Date: 2026-09-11. Frozen parent `21e85d25284d`.

## Before code

| independent finding | source/path audit | verdict before edit |
|---|---|---|
| z-star NEMO card raises without `is_active` | `_vmix_K_profiles` raised at the new branch while `build_nemo_rest_recipe` constructs `OceanZStarCoordinate`; NEMO multiplies the anchor by level-1 `tmask` at compiled `GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:614-619` | CONFIRMED blocker |
| legacy `nemo_v1` floor was omitted | `_nemo_tke_config` selects choice 3 and retained the `TKEConfig` default `mxl_min=1e-8`; compiled NEMO derives the floor at `zdftke.f90:824-826` and overwrites `rn_mxl0` at `:840-842` | CONFIRMED Rule-12 debt |
| reported focused tests were all green | the named RK3 pairing test is red on this branch; independent review reproduced it at `3a8d94ea1985` | CONFIRMED provenance correction; pre-existing red |
| shape plant reached the new array-shape guard | the reader changed stamped `jpi`, so the fixed `(32,22,31,30)` guard rejected it before array decoding | REFUTED claim; CONFIRMED plant defect |
| receipt parent/stamp state | branch history gives parent `03098e6fe91c`; producer stamp is `21e85d25284de001c575e23f351a45c426ba5e15` and acquisition exists | CONFIRMED stale receipt |
| writer/diagnostic lows | writer declaration is MY_SRC line 571; the explicit-shape `SIZE(zdiag,...)` check cannot fail; the two DINO diagnostics read raw `cfg.mxl_min` | CONFIRMED |

No code claim was inferred from the receipt. The compiled R56TKE branches and
executing recipe/model call paths were read before edits.

## After code

- The surface operand has one explicit owner: coordinate level-1 `is_active`
  when present, otherwise the state wet mask carried by every z-star card. No
  ones fallback exists. The modified recipe test asserts the no-`is_active`
  operand and then executes the full step: `24 passed in 275.93s`.
- The legacy `nemo_v1` move is registered as UNMEASURED-with-spec debt in the
  round-56 correction and round-57 preregistration; the recipe suite is named
  as construction/stepping coverage, not a bit score.
- The shape plant mutates `avt_pre_evd.shape` against unchanged stamped
  extents. Its decisive output is `STATUS FAIL: avt_pre_evd shape (31, 22, 31)
  disagrees with stamped extents`.
- Writer line/comment, tautological check and shared-floor diagnostic reads
  are corrected. Instrument/diagnostic tests: `15 passed in 0.51s`; Python
  compilation and `git diff --check` pass.
- LOCK_EXCHANGE and OVERFLOW resolve constant mixing at their own namelists,
  lines 131 and 129 respectively. The shared TKE path is not executed there.

Differentiability: the mask remains a JAX array and no host conversion or
data-dependent Python branch was introduced. Conservation: no tendency or
state update changed; only the NEMO surface-anchor operand can now reach the
previously broken flat-bottom card. Stability: the six formerly failing card
paths step finite in the 24-test recipe suite.

ASKED: every edit in this packet. UNASKED: none. No configuration choice,
carried state, NEMO source/data, year/reconciliation harness, freshwater pair,
or #1484 guard changed.

## Stop condition

The required preregistration commit failed because `.git/index.lock` cannot be
created on the read-only Git metadata. Under the fail-closed commit-stamp rule,
the kt=2 calibration/substitution score and all trajectory measurements remain
unrun until an operator commits this packet.
