# Preregistration — VORTEX_SMT round 10 (lane round 222): rung SMT-2, linear bottom drag

Frozen before any SMT-2 measurement.  Decision 93 (operator note CE) mini-ladder,
rung 2 of 4: **linear bottom drag exactly as ORCA2 rung 0 resolves it**, on the
seamount VECTOR deck.  One module's switches change against SMT-1.

## The switch set, cited

| knob | SMT-1 | SMT-2 | citation |
|---|---|---|---|
| `ln_drg_OFF` | `.true.` (deck) | `.false.` | rung-0 `namelist_cfg` leaves it unset, so it resolves from `namelist_ref:812` `.false.` |
| `ln_lin` | unset (`.false.`) | `.true.` | rung-0 `namelist_cfg:270` |
| `ln_non_lin` / `ln_loglayer` | `.false.` | `.false.` | `namelist_ref:814`, `:815`; rung 0 writes neither |
| `ln_drgimp` | `.true.` (ref) | `.true.` (ref) | `namelist_ref:817`; rung 0 writes neither |
| `rn_Cd0` (bot) | n/a | `1.e-3` | `namelist_ref:834` (`&namdrg_bot`); rung 0 writes no `&namdrg_bot` block |
| `rn_Uc0` (bot) | n/a | `0.4` | `namelist_ref:835` |
| `ln_boost` (bot) | n/a | `.false.` | `namelist_ref:839` |

NOTE / correction carried into this round: the round-222 brief cited `rn_Cd0 = 1.e-3 :823`,
`rn_Uc0 = 0.4 :824`, `ln_boost = .false. :828`.  Those three lines are `&namdrg_top`.
The `&namdrg_bot` block this rung resolves is `:834`, `:835`, `:839` — same VALUES,
different block.  Verified against the compiled `drg_init`, which reads
`namdrg_bot` for `cd_topbot == 'BOTTOM'`.

Resolved linear coefficient: `rCdU_bot = -rn_Cd0 * rn_Uc0 * ssmask = -4.0e-4 m/s`
(`zdfdrg.f90:532` `pCd0 = rn_Cd0 * zmsk_boost`, `:262` `pCdU = - pCd0 * rn_Uc0`).

## Predictions and falsifiers

* **R10-P1** The deck diff against SMT-1 is exactly ONE hunk, two lines, in `&namdrg`.
  REFUTED if any other namelist line differs.
* **R10-P2** The build's `ocean.output` echoes `ln_lin = T`, `ln_drg_OFF = F`,
  `ln_drgimp = T`, `rn_Cd0 = 1.0e-3`, `rn_Uc0 = 0.4`, `ln_boost = F`, and the
  linear-friction line `Cd0*Uc0 = 4.0e-4`.  REFUTED if any echoed value differs.
* **R10-P3** NEMO runs kt=1..10 and 100 days with no non-finite field; `max|ssh|`
  and `max|u|` at kt=10 stay within 2x of SMT-1's (0.80 m, 1.01 m/s) and the drag
  REDUCES them (a momentum sink).  REFUTED by non-finite output or by growth.
* **R10-P4** The plain and instrumented kt=10 restarts are byte-identical
  (additions-only instrument).  REFUTED if they differ.
* **R10-P5** The drag term is non-zero on the bottom level of every wet column
  and on no other level; the bottom level is NOT uniform over the seamount.
  REFUTED if the bottom-level map is constant or if a dry column carries drag.
* **R10-P6** BEFORE the legoESM landing, the SMT-2 card's ladder is WORSE than
  SMT-1's at the first-over-bar row (legoESM would be missing a momentum sink
  NEMO has).  REFUTED if the before-arm ladder matches SMT-1's rows.
* **R10-P7** The first non-bit statement is NOT the linear coefficient itself
  (a time-constant scalar), but one of its consumers over partial cells:
  `dynzdf.f90:306` / `:473` (the implicit tridiagonal diagonal, divisor
  `e3u_3d(iku)*(1+r3u(Kaa)*umask(iku))` at `iku = mbku`), `dynzdf.f90:166`/`:168`
  (the barotropic bottom-stress re-add), or `dynspg_ts.f90:1245-1246`/`1271-1272`
  (the barotropic mode's own drag).  REFUTED if the coefficient itself is the owner.
* **R10-P8** After the landing, no AT-BAR row of the SMT-2 registry leaves the bar
  and first-over-bar is never earlier than SMT-1's kt=2.  REFUTED otherwise (HOLD).
* **R10-P9** Every other certified card registry is 0/50 rows moved (the linear
  drag is a NEW scheme selection no existing card names), GYRE's ten-step ladder
  and certified year are bit-identical, DINO's month gate is unchanged.
  REFUTED by any moved row — which would be registered, not hidden.

## No hidden choices

Anything rung 0 does not pin is a DECISION_NEEDED, not a guess.  Rung 0 pins
`ln_lin` alone and takes every `&namdrg`/`&namdrg_bot` companion from the
reference namelist; this deck does the same and states the resolved values.
