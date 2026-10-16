# Preregistration — NEMO-testcases grid ladder G6, round 1 (gridladder_r36)

**Frozen:** 2026-10-16, before any G6 record exists.  Nothing was measured
against NEMO for this rung; every statement below is read off the compiled
source (READ) or is a prediction (PREDICTED).

Builds cited: `REF` = `ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/ppsrc/nemo` (the
writer-free binary G6 reuses); Decision-122 pointer `ORCA2` =
`ORCA2_OMIP_L4/BLD/ppsrc/nemo`.

## 1. Claim and exact target (Decision 128, lane A, rung 1)

G6 = the certified G5 deck plus ORCA2's vector-form momentum advection and
nothing else.  The deck is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g6_orca2vec/namelist_cfg`
(SHA-256 `243fb531...b482b`); `namelist_ice_cfg` is G5's, byte-identical
(ORCA2's, `9d5ae3fa...3dd8`).

### Resolved-deck diff G5 -> G6 (assignments; a committed test pins it)

| key | G5 | G6 = ORCA2 |
|---|---|---|
| `namdyn_adv.ln_dynadv_OFF` | `.true.` | `.false.` |
| `namdyn_adv.ln_dynadv_vec` | `.false.` | `.true.` (ORCA2 `namelist_cfg:346`) |
| `namdyn_adv.nn_dynkeg` | not stated (unread under `ln_dynadv_OFF`) | `0` (ORCA2 `namelist_cfg:347`; C2 KE gradient) |

`ln_dynadv_cen2`/`ln_dynadv_up3` stay `.false.`, `ln_dynvor_een` stays
`.true.`, `ln_zad_Aimp` stays `.false.`, tracer advection and lateral
momentum diffusion stay OFF — all unchanged from G5.

### What NEMO derives from the switch (READ, REF build)

| derived item | G5 (`np_LIN_dyn`) | G6 (`np_VEC_c2`) | cite (REF) |
|---|---|---|---|
| advection selector | `n_dynadv = np_LIN_dyn` | `n_dynadv = np_VEC_c2` | `dynadv.f90:185-186` |
| vorticity in EEN | `ntot = np_COR` (planetary) | `ntot = np_CRV`, `nrvm = np_RVO` | `dynvor.f90:857-864` |
| stp_2D 3-D RHS | HPG + VOR | HPG + VOR + KEG + ZAD | `stp2d.f90:157-163` |
| stp_2D vertical average | 3-D RHS only | same branch (`np_VEC_c2, np_LIN_dyn`) | `stp2d.f90:175-179` |
| vertical velocity, stages 2-3 | `wzv` on transports (`np_transport`) | `wzv` on velocities (`np_velocity`) | `stprk3_stg.f90:287-302` |
| stage-1 advection | none | none (already in stp_2D) | `stprk3_stg.f90:316` |
| stages 2-3 advection | linear: KEG/ZAD absent | `dyn_adv` VIF (KEG + ZAD) | `stprk3_stg.f90:331-335` |
| stage 1-2 momentum update | thickness-weighted | applied on velocity | `stprk3_stg.f90:366` |
| stage-3 (`dyn_zdf`) update | thickness-weighted | applied on velocity | `dynzdf.f90:138` |
| external-mode update | flux form | vector form, `* ssvmask` | `dynspg_ts.f90:533-545` |
| external-mode time sum | transports | velocities | `dynspg_ts.f90:627` |
| post-loop depth recompute | executed | skipped | `dynspg_ts.f90:674` |
| tracer `zFw` transport | computed in the flux branch | `ll_Fw = .true.` in `tra_adv_trp`; sole consumer `tra_adv` is OFF | `traadv.f90:195` |

Readers of the switch that do not execute here: `dynatf_qco.f90:164`
(no `CALL dyn_atf` in any compiled file of the RK3 build), `dynzdf.f90:245`,
`:412` (inside `ln_zad_Aimp` = F), `diawri.f90` (output only),
`domqco.f90:263`, `:275` (commented out).  The `r_stb_*` thresholds set at
`dynadv.f90:186` are read only by `wAimp` (`sshwzv.f90:407`), not called.

## 2. Card (no new physics; G1b's statements made selectable on G6)

`build_orca2g6_vec_card` / case `ORCA2G6-vec` = G5's builder path with the
rung tables keyed for G6 (every G2-G5 selection inherited) and
`_ORCA2G_DYNADV_VEC = True`.  The existing vector block then selects, on G6,
exactly G1b's statements:

- vorticity `een_total` with `een_nemo_relative_vorticity` (np_CRV e1f*e2f
  area and NEMO's U fold halo on the top F row; G1b rounds 2-3);
- `rk3_vector_rhs_nemo_order` (Krhs order HPG, VOR, KEG and the KEG
  product source-rounding; G1b rounds 3 and 5);
- `barotropic_literal_fold_unit_vector_form` (fold unit outside the
  vector/flux branch; G1b round 2);
- momentum selectors `vector_invariant`, `c2`, `nemo_advective` (dyn_zad),
  inert `upwind` flux scheme;
- deselected because NEMO's vector branch never runs them: flux-form
  external update, entry-depth statement, flux-form stage-1 RHS option.

`rk3_stage_velocity_written_order` (G1b round 4: vector stage update in
written order) is already selected on every G rung and now acts on G6's
vector arm.  G5's own statements (implicit ice-ocean top drag, stage-2 stored
thickness, stage-3 salt flux) stay selected.  The card is fail-closed:
`ORCA2G6_UNMEASURED = (g6_record_not_admitted, vector_form_multilevel_step)`.

New versus G1b (never run on any ladder rung): 30 wet levels, so `dyn_zad`'s
interior loop (`dynzad.f90:105-114`), velocity-form `wzv` with partial
cells, KEG/np_CRV on partial-cell faces, the velocity-form `dyn_zdf` update
with barotropic removal over many levels, and the vector external update
with real depths and the ice top drag.

## 3. Acquisition

No build, no new cpp key: `ln_dynadv_vec` is a namelist switch
(`dynadv.f90:162`, `:186`), and G5's round-31 pair (writer-free
`R31REF_FIX1` e8ebf4ae..., record `R31FULLENTRY_FIX1` 41cd9a63...; keys
`key_RK3 key_qco key_si3 key_vco_1d3d`) is reused, pinned.  The record
writer set comes from the ORCA2 lane, whose deck is itself vector form.

Order: smoke (record binary, 2 steps) -> writer-free 100-step reference
(discovers NEMO's boundary) -> twins A/B/C to N = min(10, completed)
rounded down to `nn_fsbc` = 2 -> admission.  Restarts: A at N, B at
2,4,..,N, C at 3,5,..,N: every step 2..N.  Step 1 cannot be written
(a stock list cannot start at step 1; G1b round 2 found its step-1 file an
empty shell).  Each run's `ocean.output` must state the vector form and
total vorticity.  Admission = G5's: passivity (every restart variable of
the record twins byte-identical to the writer-free reference at common
steps; plant fires), twin bytes of every per-step frame, stage-3 frame =
next entry, restarts = stage-3 frames.

## 4. Frozen predictions

- **P1 (boundary).** NEMO's writer-free reference stops in `stp_ctl` while
  beginning kt 11 (G5: |V| 10.25 m/s at kt 11), so N = 10.  PLAUSIBLE: the
  added advection could move the stop earlier; any stop at kt <= 11 is
  admitted by the script, a stop before kt 3 refuses.
- **P2 (passivity).** Record twins' restarts equal the writer-free
  reference's at every common step (as on G5).
- **P3 (entry).** kt-1 entry state, every ocean-read surface field and the
  ice top-drag coefficient are bit-identical to the card's inputs at kt 1
  (velocities are zero at entry, so the switch cannot act before the
  external mode of kt 1).
- **P4 (first non-bit statement — the brief's prediction, stated as such).**
  The first unequal value is the fold-unit vector-form V update of the
  external mode — the statement the ORCA2 lane is walking (ORCA2 rounds
  289-295: `va_e` on the fold-image row, operands `hv_0` and `ssvmask`),
  `REF/dynspg_ts.f90:540-544`, at kt 1.  Counter-evidence written down now:
  the G builder reads `hv_0` and the masks from the raw domain file (the
  ORCA2 lane's two operand defects were in its reconstructed builder), and
  G1b's identical fold unit is at the bar on the GIVEN label with ORCA2's
  coastline.  So this is PLAUSIBLE, not likely-by-measurement.
- **P5 (fallback order, if P4 holds exact).** The first non-bit statement is
  then at kt 1 stage 2 in a term G1b never exercised on more than one
  level: `dyn_zad` (`REF/dynzad.f90:105-114`) or the velocity-form `wzv`
  feeding it (`stprk3_stg.f90:290`).

## 5. Confirm / refute

- P1 confirmed by the reference's `time.step` and stp_ctl line; refuted by
  any other boundary (reported exactly; the horizon follows NEMO).
- P2 refuted by any differing restart variable: the record is then not
  admissible.
- P3 refuted by any kt-1 entry or input difference.
- P4 confirmed only if the first unequal cell (stage-local, both labels) is
  a V face on the fold-image row produced by `va_e`; refuted by an earlier
  or different first statement, or by exact agreement through kt 1's
  external mode.
- P5 refuted by any other first statement.

No bar verdict is claimed this round.  ROUND_STATUS: STOPPED_FOR_RECORD.
