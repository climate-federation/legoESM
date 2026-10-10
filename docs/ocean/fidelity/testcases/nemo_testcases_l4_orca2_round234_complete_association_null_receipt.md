# ORCA2 round 234 — complete external-mode association null

Date: 2026-10-10. Frozen base: `afb2daf6b`. Preregistration commit:
`b360d0e64`. Measurement commit: `8845bf856`. Restoration commit:
`4d1d153d0`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round234/`.

The temporary candidate completed NEMO's one post-substep seven-array
association and reintroduced round 233's private exact-geometry/vector-mask
control. It was removed after its owner prediction failed. The final tree has
no package, card, deck, carried-state, stabiliser, sea-ice selector, or
`unmeasured_features` change; `git diff afb2daf6b..HEAD -- packages/` is empty.
Every binding number is reported separately as **independent** and **given
NEMO's entry**. No NEMO acquisition was run.

## Compiled statement and exact local boundary

The executing build updates `hu_e/hur_e/hv_e/hvr_e` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:738-744`, then calls
one `lbc_lnk` over `ua_e`, `va_e`, `hu_e`, `hv_e`, `hur_e`, `hvr_e`, and
`ssha_e` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756`. Its
executing T-pivot T/U rules are
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:945-972`.

The candidate added the missing U-fold images for `hu_e/hur_e`, the T-fold
image for `ssha_e`, and carried associated SSH to the next substep. Against
the admitted rank-complete rung-0 record, all seven post-call arrays are
bit-exact:

| array | unequal cells | maximum absolute |
|---|---:|---:|
| `ua_e` | 0 | 0 |
| `va_e` | 0 | 0 |
| `hu_e` | 0 | 0 m |
| `hv_e` | 0 | 0 m |
| `hur_e` | 0 | 0 m-1 |
| `hvr_e` | 0 | 0 m-1 |
| `ssha_e` | 0 | 0 m |

The ordinary and traced state arrays are array-identical. This establishes a
source-exact statement, not a trajectory inference.

## Binding owner result — null

The exact association was then measured inside the complete round-217 unit,
with round 233's private raw face geometry and raw vector mask retained to keep
that independent geometry debt out of the attribution. The two labels give
identical results:

| label | `vn_adv` unequal / support | `vn_adv` max | `zvb` unequal | `zFv` unequal | fold S max | fold T max |
|---|---:|---:|---:|---:|---:|---:|
| independent | 8,589 / 8,589 | 24.341577728515905 | 8,589 | 226,637 | 3.2850941483738296 PSU | 0.17742816676640505 K |
| given NEMO's entry | 8,589 / 8,589 | 24.341577728515905 | 8,589 | 226,637 | 3.2850941483738296 PSU | 0.17742816676640505 K |

R234-P3 is **REFUTED**: `vn_adv` did not fall to at most 35 unequal cells; it
did not move at all from the round-233 8,589/8,589 census or its maximum.
Fold-band salinity also moves away from the preregistered endpoint
(3.283356343139289 to 3.2850941483738296 PSU). The complete association is a
real missing statement and is locally bit-exact, but it does not own the
independent external-mode correction debt. It therefore remains registered
debt and does not land in this round.

R234-P4 and P5 are **NOT ACTIVATED** because P3 is false. No trajectory,
Decision-96, GYRE-year, DINO-month, tank, or rung-10 landing claim is made.
Decision 114 remains pending; this round neither lands nor rejects the
independently exact raw face-thickness/mask geometry.

## Controls, tests, and review

Association JSON/log SHA-256 values are
`92e7a1f810e4feb5251a289019a1fb5ff085a6b051b73ec62dce686d5fe035c2`
and `9369ab90d04cf06e2911d7024a259b7b90ab064b93e1509c610bd5ea18fcc6ea`.
Independent/given endpoint JSON SHA-256 values are
`fa0c25245cea3508e86a13404163632598d767d00c2f52e677d78316de39a24e`
and `173e6305fb58b7bd62a3a6ab552aad5895c9874888756aac4cd8e8a30b500539`.
The classification JSON SHA-256 is
`2dedf9bd774ce48d93dbe8ac86a93db0a36f6dc535af2443c5be0a2c2e0e8639`.

The U-depth, SSH, label-coverage, false-owner, and endpoint plants all refuse.
The clean-tree citation gate passes both this receipt and the cumulative
default receipt with zero unmapped citations, endpoint failures, or map-audit
failures.  Its rigid two-line shift of the seven-array-call citation refuses
with exit 1, as required.
Focused round-233/234 tests pass 10/10 (log SHA-256
`af7630bd9ec7db520f9a83052dfab618c49360ae2a49414183fb767d84bbee42`).
The prescribed `tests/ocean/fidelity -n 12` battery collected 3,148 tests and
reached 99% before the registered compiler-limit tail loss: 3,108 PASS, seven
SKIP, four FAIL, 29 unclassified. The four failures are exactly the registered
pre-existing reds: GYRE round-129 record stamp, allow-dirty scope,
worktree-stamp ratchet, and SI3 scalar-math provenance. No round-234 test is
red. Log SHA-256:
`73ab424a86392a1e30d2dbc48791dca55fedaff22b9e4e2a662c2985ba21d765`.

Independent review was attempted with a separate
`codex exec --sandbox read-only` process. It exited before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Independent review unavailable in-sandbox. Log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## Preregistered predictions

| ID | disposition |
|---|---|
| R234-P1 | **CONFIRMED**: passive CPU/JIT/fp64 instrument; all plants refuse. |
| R234-P2 | **CONFIRMED**: all seven association outputs are bit-exact. |
| R234-P3 | **REFUTED**: `vn_adv` remains 8,589/8,589 unequal under both labels. |
| R234-P4 | **NOT ACTIVATED**: P3 is false. |
| R234-P5 | **NOT ACTIVATED**: no landing candidate qualifies. |

## OPEN

Do not try another statement blindly. Build the source-ordered per-substep
operand table on OMT-4 from the existing admitted round-222 streams
(`oracle_bt_substeps`, `oracle_bt_frames`, `oracle_bt_ordered_operands`,
`oracle_bt_drag_operands`, and `oracle_bt_advmean_operands`). For every
substep, compare SSH, U/V, face depths/reciprocals, EEN coefficients, slow
forcing, midpoint history, and transport accumulators against NEMO; name the
first unequal fold-row operand and only then test one cited statement. No new
NEMO acquisition is presently required.

Decision 114 is still required: land the independently exact raw
face-thickness/mask geometry now (**pick**), or retain it privately until the
separate `vn_adv` debt closes?

ASKED choices: Decisions 103, 109, 113, standing Decision 96, and pending
Decision 114. UNASKED choices: empty.
