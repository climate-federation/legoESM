# Preregistration — VORTEX_SMT round 25 (lane round 237): rung SMT-4, lateral momentum diffusion

Frozen before extending the shared acquisition driver or reading any SMT-4
trajectory result. Base: lane tip `33c754c71` (round 236 / VORTEX_SMT round
24). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/`.

## Compiled branch and switch set read first

The ORCA2 rung-0 deck at
`phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:388-393`
selects `ln_dynldf_lap=.true.`, `ln_dynldf_lev=.true.`, and
`nn_ahm_ijk_t=-30`, whose coefficient is read from a 3-D file. Decision 93
explicitly replaces only that unavailable file-backed coefficient on this
idealised rung with the stated stand-in `nn_ahm_ijk_t=20`. The values the
rung leaves unset resolve from
`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref:1104-1134`: div-rot type 0,
`rn_Uv=0.1 m/s`, `rn_Lv=10.e3 m`, and `rn_ahm_b=0`. SMT-4 states all of them
explicitly. It does not copy VORTEX's pre-existing `rn_Lv=30.e3 m`.

The compiled SMT-3 base reads the reference namelist and then the card at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:177-185`.
For a z-partial-step grid, level Laplacian resolves to `np_lap` at
`:221-278`; mode 20 materialises `ahmt/ahmf` through `ldf_c2d` with
`zUfac=0.5*rn_Uv` at `:310-347`. The stage dispatch then calls
`dynldf_lev_lap` at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf.f90:81-90`.
The compiled div-rot operator forms the curl and divergence with live partial
thicknesses at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`.

Pre-implementation search found the shared VORTEX acquisition driver, its
SMT-3 deck layering, the existing stage-1/2/3 momentum-term writer and
self-describing parser, and the prior 100-day reuse path. This round extends
those paths. It creates no second driver, parser, writer format, physics
operator, card builder, or configuration field.

## Frozen predictions and falsifiers

* **R25-P1 — one-module deck.** SMT-4 differs from SMT-3 only in
  `&namdyn_ldf`: OFF becomes false; Laplacian and level directions become
  true; horizontal becomes false; type 0, mode 20, `rn_Uv=0.1`,
  `rn_Lv=10.e3`, and `rn_ahm_b=0` are stated. Any other resolved scientific
  difference REFUTES the controlled rung.
* **R25-P2 — resolved branch.** `ocean.output` must echo the complete tuple
  above, report the iso-level Laplacian operator, and report a fixed
  grid-scale viscosity with `0.1 m/s`. A different echo or dispatch REFUSES
  admission.
* **R25-P3 — passive record.** The plain and instrumented step-10 restarts are
  byte-identical. Every self-describing entry and stage-1/2/3 momentum-term
  record parses to EOF and contains its required named groups. Header,
  required-name, and truncation plants must each exit nonzero. A restart byte
  or plant failure REFUSES admission.
* **R25-P4 — run sanity.** The ten-step and shipped-length 100-day NEMO runs
  reach `STOP 0`, emit finite state, and write their requested restart
  cadence. Any non-finite field, missing frame, missing restart, or nonzero
  launch REFUTES the run.
* **R25-P5 — sufficient owner boundary.** The existing all-stage momentum
  writer records the accumulator before and after `dyn_ldf` at stages 1, 2,
  and 3, plus stage inputs and outputs. The first new non-bit boundary is
  predicted to be that post-LDF accumulator. A missing stage-1 record or a
  pre-LDF mismatch REFUTES this prediction and determines the next walk.
* **R25-P6 — acquisition-only disposition.** This round changes no legoESM
  physics or certified card. It stops `STOPPED_FOR_RECORD` with
  `ACQUISITION_NEEDED` unless an already-admitted record unexpectedly exists.
  No ladder value, owner, or fidelity direction is claimed from preflight.

The next round admits the record, creates the explicit SMT-4 card, proves its
geometry and initial state against the admitted oracle, scores kt=1..10 and
100 days, and walks the first non-bit momentum-LDF statement in compiled
order. Every moved row is registered. Any later production statement remains
subject to the GYRE ladder/year, all SMT and flat VORTEX registries, tanks,
generic GYRE, private DINO month, ORCA2 pointer, citation, plant, and review
gates.

## No hidden choices

Decision 93 authorises the `nn_ahm_ijk_t=20` stand-in and fixes this rung's
scope. The coefficient values and every companion field above come from the
compiled ORCA2 reference that rung 0 actually reads. Geometry, EOS, tracer
diffusion, vertical mixing, drag, momentum advection, vorticity, pressure
gradient, barotropic program, timestep, and run lengths remain the admitted
SMT-3 values. No default, threshold, stabiliser, carried state, or record
source is selected here.

## Binding Decision-95/96 addendum — frozen before production changes

At 2026-10-05 21:52 the external campaign ledger recorded Decisions 95 and
96 after round 236: land and register the already measured source-exact SMT-3
closed-bottom-W-mask plus live-Kmm-divisor pair. The aggregate result is a net
improvement (30 of 34 moved rows toward NEMO, first-over-bar unchanged, no
status loss); the only refused control was the two-ULP cellwise ratchet on
near-zero cells. Decision 96 makes that ratchet a registered quantity rather
than a veto in exactly this case. This binding decision supersedes R25-P6's
acquisition-only disposition before any production file is edited or new
landing result is read. The SMT-4 acquisition package remains a prepared next
step, not scientific evidence.

The landing prediction remains round 236's frozen one-variable table:
`kt2 T 1.020298116571876e-08 -> 6.957537701781045e-10`, 34 aggregate rows
moved, 30 toward and four away, first-over-bar still kt2, with all 40
cellwise ratchet violations registered. The exact production candidate is
the state immediately before round 236's restoration commit; no statement is
added to it.

Before calling the decision landed, the candidate must reproduce that SMT-3
registry and pass the blast-radius gates that round 236 stopped before: GYRE
ladder and year, generic GYRE, all certified VORTEX/SMT cards and tanks, and a
private DINO month run. A new row-status loss, earlier first-over-bar, year
movement beyond Decision 59, DINO beyond its bar, or unregistered card move
REFUSES the landing. The already disclosed near-zero-cell ratchet result is
not rerationalised and remains red in the receipt. No new configuration or
carried state is authorised.
