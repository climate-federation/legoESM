# ORCA2 round 82 — hierarchy rung-0 deck and acquisition handoff

Base: `178904859a1757c4e5d62f115d8d2c4552979cfc`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_round82.md`.  Claim label: **independent**.
No package file, shipped ORCA2 card, sea-ice debt tuple, shared default, NEMO
source, CPP key, carried state, or fidelity threshold changed.

## Verdict

**STOPPED_FOR_RECORD.**  The Decision-79/B21 rung-0 NEMO deck is rendered
deterministically from the admitted shipped deck and passes its exact-delta
gate.  A fail-closed operator launcher will produce two identical two-rank
ten-step runs with a self-describing restart at every step, plus the two-rank
240-step independent month.  No compatible rung-0 record exists on disk, so
the rung-0 card, ladders, month scores, and first non-bit statement remain
unmeasured and nothing in `packages/` lands this round.

## Exact rung-0 deck

The canonical generated namelist has SHA-256
`b627f4e2d94e4619dbfa27f39b811732497f032b73be86adcc57ddca2dee0e91`.
It preserves every unlisted shipped assignment and changes these existing
assignments:

| group | rung-0 changes |
|---|---|
| `namtsd` / `namtra_dmp` | `ln_tsd_dmp=F`, `ln_tradmp=F` |
| `namsbc` | `ln_blk=F`, `nn_ice=0`, `ln_traqsr=F`, `ln_ssr=F`, `ln_rnf=F`, `nn_fwb=0` |
| `namtra_qsr` / `namsbc_rnf` | `ln_qsr_rgb=F`, `ln_rnf_mouth=F` |
| `namagrif` | `ln_spc_dyn=F` |
| `nambbc` / `nambbl` | `ln_trabbc=F`, `ln_trabbl=F` |
| `namtra_mle` / `namtra_eiv` | `ln_mle=F`, `ln_ldfeiv=F` |
| `namzdf` | `ln_zdftke=F`, `ln_zdfddm=F`, `ln_zdfiwm=F`, `nn_havtb=0` |
| `namzdf_iwm` | `ln_tsdiff=F` |

It adds explicit mutually exclusive surface and closure selectors rather than
inheriting reference defaults:

| group | explicit rung-0 assignments |
|---|---|
| `namsbc` | `ln_usr=F`, `ln_flx=T`, `ln_abl=F`, `ln_cpl=F`, `ln_mixcpl=F`, `ln_dm2dc=F` |
| `namzdf` | `ln_zdfcst=T`; `ln_zdfric=ln_zdfgls=ln_zdfosm=ln_zdfnpc=ln_zdfmfc=ln_zdfswm=F` |
| `namsbc_flx` | five yearly climatological fields from one generated `rung0_zero_flux.nc` |

The constant coefficients remain the GYRE values `rn_avm0=1.2e-4 m2/s` and
`rn_avt0=1.2e-5 m2/s`; `nn_avb=0` is unchanged.  The source audit found that
the shipped ORCA2 deck had `nn_havtb=1`, which applies an equatorial shape even
under constant closure.  Rung 0 therefore changes it to GYRE's `nn_havtb=0`:
the compiled initialization takes the scalar coefficients at
`nn_avb=0`, applies the horizontal shape only at `nn_havtb=1`, and seeds the
closure arrays (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:206-228`).  This
required selector was absent from the preregistration's prose but follows its
frozen prediction of constant GYRE mixing; the omission is recorded here, not
silently edited into the committed preregistration.

NEMO requires exactly one surface-boundary formulation
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:299-306`).  The permitted B20 choice
is `ln_flx`: the compiled dispatcher calls the flux reader and no bulk routine
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:440-448`), while the flux routine
maps the five file fields directly to stress, total/solar heat, and freshwater
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcflx.f90:191-204`).  The launcher generates
those five arrays as fp64 exact zero on the 180x148 ORCA2 grid and the admission
gate rejects any nonzero value.  This avoids modifying the compiled
`usrdef_sbc` source and keeps CPP keys unchanged.

With `nn_ice=0`, the compiled step selector has no executable ice arm
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:499-503`); the unchanged
`namelist_ice_cfg` is therefore inert.  Constant closure is the unique selected
vertical scheme (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:264-270`) and uses
the initialized background arrays without calling TKE/GLS/RIC/OSM
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:334-343`).  Enhanced vertical
diffusion remains on as B21 requires.  The disabled geothermal, BBL and
damping switches guard their stage-3 calls directly
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:738-740`), and the disabled GM
and MLE switches guard their tracer-advection additions
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-260`).

## Record contract and acquisition

The committed launcher reuses the admitted scalar-math executable
`c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343`
and the shipped input manifest; it pins both plus the unchanged ORCA2 CPP card.
It creates new targets only:

| target | run control | required record |
|---|---|---|
| `orca2_rung0_10step_a_np2` | `nn_itend=10`, `nn_stock=1` | steps 1..10, ranks 0 and 1 |
| `orca2_rung0_10step_b_np2` | identical twin | bit-identical to A for T/S/u/v/ssh |
| `orca2_rung0_240step_np2` | `nn_itend=240`, `nn_stock=240` | finite terminal ranks 0 and 1 |

Each restart is NetCDF self-describing fp64.  Admission requires the resolved
run log to print `nn_ice=0`, `ln_zdfcst=T`, and `ln_zdftke=F`; requires no ice
restart; compares 100 field payloads across the ten-step twins with
`np.array_equal`; and permits only the two run-control changes between the
ten-step and month decks.  The launcher is preflight-clean and reports:

```text
ORCA2_ROUND82_RUNG0_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round82/acquisition/orca2_rung0_10step_a_np2
```

The four deck plants fire now: undeclared delta, changed mixing value, live
excluded module, and changed CPP.  The five record plants are committed and
will run after acquisition: one-ULP restart change, missing rank, wrong step,
non-finite payload, and hidden month-deck delta.  They cannot be truthfully
reported as fired before their target record exists.

## Prediction ledger

| preregistered prediction | result |
|---|---|
| exact enumerated namelist delta; no CPP/input drift | **CONFIRMED at preflight**; zero-flux file is the sole generated input addition |
| constant closure, GYRE coefficients, no ice | **PARTLY CONFIRMED** by exact deck; resolved-log half awaits acquisition |
| finite complete 10-step and 240-step records | **UNMEASURED — record absent** |
| explicit fail-closed rung-0 card | **UNMEASURED — card deferred until record admission** |
| ladders/month remain unmeasured without acquisition | **CONFIRMED** |
| disposition `STOPPED_FOR_RECORD` if absent | **CONFIRMED** |

No post-hoc physical prediction is added.

## Gates, review, and tests

The focused deck/record tests pass **8/8**; all four currently executable deck
plants fire.  The citation gate passes on this receipt (9 citations, zero
failures/unmapped) and the campaign default (274 citations, zero
failures/unmapped); the shifted `sbcmod` citation exits 1 and reports
`SYMBOL-NOT-AT-LINE`.

The prescribed `tests/ocean/fidelity -n 12` battery reached 98% with six
failures and six skips, then repeated the standing no-summary stall and was
stopped.  The six established failures were rerun in isolation: round-129
record certification, round-35 stamp scope, the worktree-stamp emitter,
missing case-board row, SI3 scalar-math provenance, and round-51 private trace
registry.  They reproduce as **6 failed / 40 passed** across those isolated
files; every round-82 test is green.  No package file changed, so the GYRE
trajectory, DINO, tanks, shipped ORCA2 card, and all sea-ice behavior are
byte-identical by construction.

The required read-only Codex review failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round82_rung0_acquisition/run.sh --run`.
2. Admit the record and show all five record plants firing.  If NEMO rejects
   the zero-flux file schema or does not retain every `nn_stock=1` restart,
   repair the acquisition under a new target; do not reinterpret missing data.
3. Build the explicit rung-0 legoESM card only after record admission, leaving
   the shipped card and its `unmeasured_features` tuple unchanged.
4. Run the given-entry and independent kt=1..10 ladders, then the independent
   rung-0 month; name the first non-bit statement and score every row.
5. Higher-rung placement remains Decision 80 and is not chosen here.

ASKED: the B20/B21 rung-0 deck and its operator-run record contract.  The
choice between authorized zero-flux expressions is `ln_flx`, because it keeps
the compiled source and CPP card unchanged and feeds literal zero operands.

UNASKED: no higher-rung placement, shipped-card/sea-ice change, stabilizer,
threshold, carried-state change, NEMO source edit, or result inferred from an
absent record.
