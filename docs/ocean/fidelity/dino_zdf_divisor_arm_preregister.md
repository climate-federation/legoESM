# Preregistration — collapsing the implicit-mixing divisor onto NEMO's `e3w(Kmm)`

Branch `fidelity/dino-zdf-divisor-scaling`, cut from `c0d34ca21` (the scaling
study). Written and committed **before** the change is written or any arm is
run. Predecessor documents, which are this one's spec:
`dino_zdf_divisor_scaling.md` (the measurement) and
`dino_zdf_divisor_scaling_preregister.md` (its own preregistration).

## What is being changed, and why it is a card-arm change

The scaling study measured that the certified DINO twin
(`nemo_dino_kamm_mlf`) divides the backward-Euler vertical-mixing gradient by

```
D_lego(i,j,k) = 0.5*(dz_k + dz_{k+1}) * (1 + eta_AFTER/H)
```

while NEMO divides by

```
D_nemo(i,j,k) = e3w_0(i,j,k) * (1 + r3t(i,j,Nnn))       e3w_0(k) = gdept_0(k) - gdept_0(k-1)
```

Oracle lines (Rule 0), from the executing DINO build
(`cpp_DINO.fcm: key_qco key_vco_3d`):

```
TRA/trazdf.F90:219-221              zwi = -p2dt*zwt(jk  )/e3w(ji,jj,jk  ,Kmm)
                                    zws = -p2dt*zwt(jk+1)/e3w(ji,jj,jk+1,Kmm)
                                    zwd = e3t(ji,jj,jk,Kaa) - (zwi + zws)
cfgs/DINO/MY_SRC/dynzdf.F90:200-203 zzwi = -zDt_2*(avm(ji+1,jj,jk)+avm(ji,jj,jk))
                                         / (e3u(ji,jj,jk,Kaa)*e3uw(ji,jj,jk,Kmm)) * wumask
cfgs/DINO/MY_SRC/zgr_lib.F90:111-112   pe3uw(:,:,:) = pe3w(:,:,:)     ! zco: e3uw_0 IS e3w_0
DOM/domzgr_substitute.h90:131          # define e3w(i,j,k,t)  (E3w_0(i,j,k) Time(r3t,i,j,t))
DOM/domzgr_substitute.h90:132          # define e3uw(i,j,k,t) (E3uw_0(i,j,k) Time(r3u,i,j,t))
DOM/domzgr_substitute.h90:108          # define E3w_0(i,j,k)  e3w_3d(i,j,k)      [key_vco_3d]
DOM/domzgr_substitute.h90:49           # define Time(r3,i,j,t) *(1._wp+r3(i,j,t)) [key_qco, no mask]
```

Time levels (Rule 1d), read at the call site, not inferred:

```
MLF  cfgs/DINO/MY_SRC/stpmlf.F90:551   CALL tra_zdf( kstp, Nbb, Nnn, Nrhs, ts, Naa )
MLF  cfgs/DINO/MY_SRC/stpmlf.F90:396   CALL dyn_zdf( kstp, Nbb, Nnn, Nrhs, uu, vv, Naa )
RK3  src/OCE/stprk3_stg.F90:598        CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts, Kaa )     [kstg==3]
RK3  src/OCE/stprk3_stg.F90:430        CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa ) [kstg==3]
RK3  src/OCE/stprk3.F90:207            CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )
```

so in BOTH identities the divisor's `Kmm` is the NOW level and the diagonal's
`Kaa` is the AFTER level. RK3's stage-3 `Kmm` is the stage-2 output, which is
the level legoESM's RK3 cards already thread as `eta_now`; the RK3 identity
therefore needs the same divisor object, not a different one.

**This is a card-arm change on a CERTIFIED twin.** The user asked for it. Every
number that moves is disclosed in the receipt.

## The shape of the change (written before the code)

One canonical function, `nemo_e3w_kmm`, next to `build_dz_half` in the implicit
solver, produces `e3w(...,jk,Kmm)`; the tracer solve and the momentum solve both
call it, at T points and at U/V faces respectively (`e3uw_0 == e3w_0`,
zgr_lib.F90:111-112). It resolves `e3w_0` in ONE place:

1. the card's raw NEMO mesh `e3w_0` when it carries one (`nemo_e3w_0` with
   `nemo_e3w_mesh_reference`) — the bridged DINO/GYRE cards;
2. otherwise `e3w_0 = diff(gdept_0)` where the card's T points ARE the cell
   midpoints, which is the midpoint of the live reference thickness — the
   expression the NEMO test-case cards run today, unchanged. Fails closed
   (raises) if a card's own `t_depth_ref` is NOT that midpoint ladder, so a
   stretched card without a NEMO mesh cannot silently take arm 2.

**Config fields.** `implicit_vmix_e3t_now_divisor` is REMOVED. NEMO has no such
switch; the divisor belongs to the NEMO identity, which on this routine is
`zdf_implicit_solver_evaluation="nemo_literal"` (the `dyn_zdf`/`tra_zdf` literal
program, S-33's own selector). Every card already on that identity — DINO
`nemo_dino_kamm_mlf`, LOCK_EXCHANGE, OVERFLOW, GYRE — gets `e3w(Kmm)`
unbranched. `implicit_vmix_dzw_slot` is KEPT: it is the Veros arm, selected by
the `veros_faithful_v1` recipe (`recipes.py:99`, cited to Veros
`thermodynamics.py:267`), and a referenced arm is not deleted. The legacy
midpoint stays the default for every card that is not on the NEMO identity
(ORCA1 and all catalog recipes, which run `shared_thomas`). No new public field
is added.

## PREREGISTERED PREDICTIONS

### P1 — the 5-day DINO twin, BEFORE code vs AFTER code

`kamm_twin_90d.py nemo_dino_kamm_mlf --days 5 --bridge-before`, CPU fp64,
`LEGOESM_NEMO_E3T=both`, 160 leapfrog steps from the day-180 bridged NEMO
restart, byte-identical invocation on both sides, one variable: the commit.

These are the SAME two arms the scaling study integrated
(`dino_zdf_divisor_scaling.md` §5b), so the study's numbers are the prediction:

| field | predicted day-5 max abs BEFORE-vs-AFTER | CONFIRM band | REFUTE |
|---|---|---|---|
| T | 8.13e-3 K | `[1e-3, 2e-2] K` | `< 3e-4 K` or `> 5e-2 K` |
| S | 1.02e-3 g/kg | `[1e-4, 3e-3]` | outside |
| u | 2.19e-2 m/s | `[3e-3, 6e-2]` | outside |
| v | 9.15e-3 m/s | `[1e-3, 3e-2]` | outside |

* `< 3e-4 K` REFUTES that the shipped change reaches the executing path — the
  fix would be sitting behind something the certified card does not select.
* `> 5e-2 K` REFUTES that the change is ONLY the divisor — something else moved
  with it, and the arm is no longer one variable.

NOTE, stated so a clean number is not over-read: the twin archive stores
`eta/sst/u/v` as **float32 surface slices** at 5 daily samples, so this
comparison resolves ~1e-7 relative and only at the surface. It is a delivery
check on the arm, not a climate result.

**It is NOT predicted that this number drops toward the roundoff floor.** The
8.13e-3 K IS the two arms' divergence; the after-code twin becomes the NEMO arm,
so the difference is expected to APPEAR at that size, not vanish. What would
drop to the floor is the scaling probe's own arm-vs-arm difference, since after
the change the executing arm and its NEMO arm are the same object — recorded as
P4 below.

### P2 — the `dino_1226` bit-exact gates

| gate | predicted | why |
|---|---|---|
| `nemo_geometry_gate.py` | UNCHANGED, still passes | pure mesh/coordinate coverage; the change touches no geometry construction |
| `fidelity_bar_gate.py` | UNCHANGED, same verdict | a recorded-measurement bookkeeping gate; it re-measures nothing |

A move in either REFUTES the "divisor only" claim.

### P3 — the RK3 cards (LOCK_EXCHANGE-zco, OVERFLOW-zps)

**Predicted BIT-IDENTICAL, exactly 0.0 movement, every stage row.** Measured
before writing the change, fp64, on the constructed cards:

```
LOCK_EXCHANGE-zco   diff(t_depth_ref) vs dz_half_ref   max|rel| 0.000000e+00   (n=19)
OVERFLOW-zps        diff(t_depth_ref) vs dz_half_ref   max|rel| 0.000000e+00   (n=99)
both                |z_full_ref| vs t_depth_ref        max|d|   0.0
both                nemo_e3w_0                          None  (no NEMO mesh e3w_0)
```

Both ladders are uniform, both cards take resolution arm 2, and arm 2 is
literally today's expression — so the stage-sweep gate at `kt=1..10` must be
byte-identical. **Any non-zero movement REFUTES arm 2's construction** and the
change goes back, not the gate.

GYRE is the one NEMO card that carries a NEMO mesh `e3w_0` on a stretched
ladder, so its divisor WILL move. It is currently unconstructible
(`test_nemo_gyre_card_constructs` is `xfail(strict=True)`, an open finding from
the L1 lane), so no gate covers it and no number can be produced here. Recorded
as disclosed, unmeasured reach.

### P4 — contamination controls (fields the change cannot touch)

| control | required |
|---|---|
| twin day-0 bridged state vs NEMO restart, both codes | max abs `0.0` on T, S, u, v, eta |
| twin `vertical_ladder_sha256`, both codes | identical string |
| LOCK/OVERFLOW stage-sweep rows | exactly `0.0` movement (P3) |
| the Veros `dzw` arm | `test_implicit_vmix_dzw_slot.py` still green, arm unchanged |
| the legacy midpoint arm (`shared_thomas`) | pinned by unit test to `build_dz_half(dz_cell)` exactly |
| scaling probe re-run after the change | executing-arm vs NEMO-arm difference collapses to `0.0` on T, S, u, v |

Anything the control moves is not the divisor, and the arm is not one variable.

### P5 — the synthetic unit test

A stretched column with non-zero ssh where the two divisors differ: the survivor
must reproduce `e3w_0 * (1 + eta/H)` to `1e-15` relative, and the test must FAIL
against the reverted (midpoint) expression. Non-vacuity is part of the test, not
a claim about it.

## What this does NOT establish

The year-1 climate battery and the 20-year ensemble member are **NOT run** —
GPU, the user's decision, explicitly out of scope. A 5-day surface diff cannot
speak to the water-mass census, the MLD cycle, the transports or the
variability families. Per Rule 4 this remains a scale-compatibility screen with
the arm now shipped, not an attribution.

## Choices

| choice | status |
|---|---|
| collapse the NEMO divisor onto the NEMO identity; remove `implicit_vmix_e3t_now_divisor`; keep the Veros `dzw` arm | **ASKED** (task brief, and the isomorphism map's own rank-4 collapse item names this exact fix) |
| bind the divisor to `zdf_implicit_solver_evaluation="nemo_literal"` rather than inventing a selector | **ASKED** in shape (task: "no new public field"); `nemo_literal` is the existing NEMO identity for `dyn_zdf`/`tra_zdf` (S-33) |
| retarget the mutual-exclusion raise (`implicit_vmix_dzw_slot` vs the NEMO divisor) onto `nemo_literal` instead of deleting it | **UNASKED** — offered for revert. It is the faithful translation of a guard the removed field owned, and no card in the tree selects both (grep: `veros_faithful_v1` runs `shared_thomas`), so it changes no run |
| delete the `DINO_NEMO_KMM_DIVISOR` env-var ablation hooks in the DINO probes | **UNASKED** — offered for revert. They cannot survive the field's removal; the before/after comparison is done across commits instead |
| do NOT change the momentum face stretch from the face-averaged `(1+r3t)` to NEMO's `r3u` (domqco.F90:164-167) | **UNASKED** — deliberately out of scope, recorded as an open row, not silently fixed |
