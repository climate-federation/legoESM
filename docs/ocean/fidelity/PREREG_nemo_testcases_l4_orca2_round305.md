# PREREG — ORCA2 round 305: G5 top drag + G7 bottom drag under Decision 122

Frozen before measurement (2026-10-17).

## Finding before measurement (read off the cards, CONFIRMED by printing the resolved configs)

| card | NEMO deck `&namdrg` | card bottom-drag law today |
|---|---|---|
| rung 0 | `ln_lin = T`, `nn_ice = 0` | `legacy`, explicit `r = 4e-4` over a 100 m layer, no implicit matrix term, no substep drag |
| OMT-1 | `ln_drg_OFF = T` | none |
| OMT-2..4 | `ln_lin = T`, `nn_ice = 0` | `nemo_linear` bundle (matrix + substep + baroclinic split) |
| rung 10 (shipped) | `ln_lin = T`, `nn_ice = 2` (`ocean.output`: `ln_lin = T`, `ln_drgimp = T`, `ln_drgice_imp = T`) | `nemo_quadratic` (GYRE's `ln_non_lin` law, inherited from the shared GYRE config) |

So G7's two bottom statements can run unchanged only on OMT-2..4. On rung 0 and
rung 10 they need the card's drag law to change to NEMO's linear bundle first.
That is a scheme selection: it is measured here, and it is asked, not landed.

## Arms (one tree; `nemo_testcase_l4_orca2_round305_drag_d122_gate.py`)

- `B`: G7's bottom residual (`dynspg_ts.f90:1446-1447`, `:1462-1463`) and dyn_zdf
  bottom RHS (`dynzdf.f90:166-169`), no ice top drag. NEMO's nn_ice=0 program
  (`zdfdrg.f90:380` forces `ln_drgice_imp = F`). OMT-2..4.
- `L`: the linear bundle only (law change). rung 0, rung 10.
- `LB`: L + B. rung 0, rung 10 (bottom apart from top).
- `LBT`: L + G5 ice top drag (which also selects B): G7's certified composition. rung 10.

## Predictions

- R305-P1: on OMT-2..4, B loses no exact row and leaves the first over-bar row unchanged or toward.
  Falsifier: any exact row lost, or the first over-bar row moves away.
- R305-P2: on OMT-2..4, B moves no row at kt 1 before stage 3 (the bottom residual is zero from rest).
  Falsifier: a kt-1 stage-1 or stage-2 row moves.
- R305-P3: on rung 10 (both labels), L moves most rows and the majority moves toward NEMO.
  Falsifier: majority away, or the first over-bar row moves away.
- R305-P4: on rung 10, LBT vs LB moves rows (ice-covered faces) without losing an exact row.
  Falsifier: zero rows moved (top drag inert) or an exact row lost.
- R305-P5: GYRE ten-step residuals and day-30 snapshots byte-identical (every new selector defaults False).
  Falsifier: any byte differs.

## Landing rule

D122 (c)+(d) on every applicable pair. Only selections that need no unasked
choice may land. The rung-0 / rung-10 law change is a DECISION_NEEDED.

## Correction (appended 2026-10-17, after measurement; text above unchanged)

The bottom residual's `zu_i`/`zv_i` statements are `dynspg_ts.f90:1447-1448`
in the ORCA2_OMIP_L4 build, not `:1446-1447` (`:1446` is `ikbv = mbkv(ji,jj)`).
Found by the receipt citation gate's anchor check. No prediction depends on it.
