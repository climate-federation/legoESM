# PREREG — VORTEX card, round 1 (transcription + acquisition)

Date 2026-09-27. Lane tip `c09a9e111780`. Decisions 64/65 (operator note BF):
TEOS-10 is the campaign's EOS for every idealised card, and VORTEX — the
non-AGRIF half of NEMO 5.0.2 `tests/VORTEX` — is added as a NEW card that must
not move any existing certified number.

Frozen BEFORE any NEMO record exists. Round 1 transcribes the card, writes the
acquisition, and stops at ACQUISITION_NEEDED; the operator runs NEMO.

AMENDED TWICE, both times before any measurement; nothing has been measured
yet and nothing in this file was written after a number was seen.

1. Two adversarial reviews found the card inheriting the Courant-dependent
   implicit vertical advection from a sibling case's namelist. The
   resolved-case table grew that row and the additivity digests were widened
   and re-measured. No prediction changed.
2. Building a real model from the card — a tripwire added because of those
   reviews — surfaced a BLOCKING capability gap, and section 3.2 is WITHDRAWN
   as a result. See section 2b.

## 1. The case as resolved

Read from `tests/VORTEX/EXPREF/namelist_cfg` and `tests/VORTEX/MY_SRC`, not
from prose.

| quantity | value | source |
|---|---|---|
| domain | 63 x 63 x 11, closed box, flat 5000 m bottom | `usrdef_nam.F90:96-97,121`, `usrdef_zgr.F90:187-193` |
| horizontal | uniform 30 km Cartesian, 1800 km square, centred on a T point | `usrdef_nam.F90` namelist `rn_dx`/`rn_dy`, `usrdef_hgr.F90:83-84` |
| vertical | 10 wet levels of exactly 500 m, full-step zco, record 11 dry | `usrdef_zgr.F90:126` |
| rotation | beta-plane about 38.5 N, f from 7.413e-5 to 1.074e-4 s^-1 | `usrdef_hgr.F90:174-177` |
| time step | 2880 s, shipped run 3000 steps | namelist `rn_Dt`, `nn_itend` |
| barotropic | split-explicit, `nn_e = 48` PINNED (`ln_bt_auto = F`), dissipative forward-backward filter, alpha 0.07 | namelist `namdyn_spg` |
| momentum | flux form + UP3, `nn_dynkeg = 0` | namelist `namdyn_adv` |
| vorticity | **EEN, on a LIVE Coriolis** | namelist `namdyn_vor`, `dynvor.F90:874` |
| tracers | FCT2 (h2 v2), no lateral diffusion | namelist `namtra_adv`, `namtra_ldf` |
| pressure gradient | `hpg_sco` | namelist `namdyn_hpg` |
| vertical mixing | constant, `rn_avm0 = 1e-4`, `rn_avt0 = 0`, no EVD, no NPC | namelist `namzdf` |
| vertical momentum advection | Courant-dependent implicit scheme OFF: the namelist does not set `ln_zad_Aimp`, so it stays `.false.` | namelist `namzdf`, reference default |
| drag / forcing | none: `ln_drg_OFF`, user forcing writes zeros | namelist `namdrg`, `usrdef_sbc.F90:60-68` |
| initial state | analytic anticyclonic Gaussian eddy: T, S = 35, u, v, ssh | `usrdef_istate.F90:69-75,83-88,101-105,177-182` |

**One switch in the namelist is OUTSIDE the transcribed set, and the survey's
UNLOCKED verdict is refuted by it.** See section 2b. Everything else is the
certified `lane1_flux_up3` identity (LOCK_EXCHANGE and OVERFLOW).

Two things the card carries that need saying out loud:

* **The AGRIF zoom is OUT OF SCOPE.** `cpp_VORTEX.fcm` compiles `key_agrif`
  for a 1:3 nest. The card and the oracle build are the PARENT grid alone.
* **`ln_dynldf_hor = .true.` is inert**, because `ln_dynldf_OFF = .true.` is
  also set. The ladder survey listed this case as running a horizontal
  Laplacian; it does not. The card has zero lateral viscosity.

## 2b. THE BLOCKER: legoESM cannot express this case's Coriolis

The ladder survey called VORTEX UNLOCKED on the grounds that its switch set is
the tanks' certified composition. That is wrong, and this is the measurement
that refuses it: **building a real model from the card raises.**

NEMO's own dispatch, read rather than assumed. `namelist_cfg:193` selects EEN
vorticity and `dynvor.F90:874` routes it to the energy-and-enstrophy scheme.
`namelist_cfg:182` selects FLUX-FORM momentum, and NEMO's `dyn_vor` then takes
its `ln_dynadv_vec=.false.` arm, which calls the EEN routine on the PLANETARY
vorticity alone. On this case, therefore, **EEN is the Coriolis operator**: a
triad-weighted f x u, not a vorticity flux.

legoESM binds its EEN arm to VECTOR-INVARIANT momentum and refuses the pair
outright, because its flux-form branch never receives the vertex Coriolis field
and combining them would drop f x u entirely. Its flux-form branch uses a
4-point C-grid average instead. The two tanks never exposed this: they have
f = 0 and one wet row, so their rotation operator is provably dead and any
Coriolis discretisation gives the same zeros. VORTEX is the first case on this
identity where the operator is alive.

**What the card does about it.** It declares the gap, verbatim, in the field
the campaign already has for exactly this (`unmeasured_features`), so the
execution gate REFUSES the card. It does not select the 4-point average and
call it NEMO's. A card that drops the declaration is refused by the validator,
and a card that selects the EEN family is refused too.

**Consequence: VORTEX is NOT execution-ready, and section 3.2's ladder
predictions are WITHDRAWN.** Round 1's deliverable is the card, the declared
gap, and the acquisition. The acquisition is still worth running: the initial
state does not depend on the momentum scheme, so section 3.1 stands unchanged,
and the record is what a later round needs to measure the new arm the day it
exists.

## 2. The EOS deviation, and the one thing that makes it unusual here

Per decision 64, the oracle build runs TEOS-10 and the card selects
`nemo_teos10`, while the shipped namelist selects S-EOS with
`rn_a0 = 0.28, rn_b0 = 0` (a linear-in-T equation of state). The acquisition's
committed namelist patch makes the build agree with the card, exactly as
`LOCK_EXCHANGE_OMIP_L1` does.

**FINDING, reported here rather than carried silently.** VORTEX is not like the
tanks in one respect: its initial TEMPERATURE is DEFINED by inverting the
S-EOS. `usrdef_istate.F90:88` computes `T = 20 + (rho0 - rho)/rn_a0`, reading
`rn_a0` from the `nameos` namelist. `eosbn2.F90:1890-1895` reads `nameos`
whichever EOS is selected, so `rn_a0 = 0.28` is still available and the initial
state is UNCHANGED by the deviation — the bit-exactness prediction in §3 is
safe. What the deviation does change is the physics after step 1: the eddy was
constructed to be in geostrophic and hydrostatic balance under the linear EOS,
and TEOS-10 will not reproduce that balance, so the run is an OMIP-style
variant of VORTEX rather than the shipped case's balanced vortex. This is fine
for an oracle-match ladder (legoESM must reproduce whatever NEMO does) and is
NOT fine if anyone later reads the run as the published VORTEX experiment.
**One line for the operator: keep TEOS-10 (decision 64, the campaign default,
my pick) or run this one card on its shipped S-EOS?**

## 3. Predictions, frozen

### 3.1 Initial state — the round's primary claim

Against `oracle_step_entry_kt00000001.bin` (NEMO's Nbb state at step 1), on the
wet-cell and active-face masks:

| field | predicted cells unequal |
|---|---|
| T | 0 |
| S | 0 |
| u | 0 |
| v | 0 |
| ssh | 0 |

**FALSIFIER: any field with a non-zero count.** The two most likely causes, in
order: (a) the live ssh-stretched depth — `istate.F90:127-130` hands
`gdept(:,:,:,Kbb)`, which `domzgr_substitute.h90:139` stretches by `1 + r3t`
after `restart.F90:461` has already set ssh — would show as T, u and v wrong
while ssh is exact; (b) an operand-order difference in the km-to-metre
conversion would show as a scattered 1-ULP field.

Secondary, NOT claimed exact this round: the derived barotropic velocities
`uu_b`/`vv_b` (`istate.F90:149-154`). They are transcribed with NEMO's own
statement order, but the `(1 + r3u)` factor is applied in the accumulation and
removed in the divisor as two separate statements, and that has never been
measured on a card with a non-zero initial velocity. Registered as a watch row.

### 3.2 The kt = 1..10 ladder — WITHDRAWN

Withdrawn by section 2b: the card cannot execute, so no ladder row is
predictable and none would mean anything if it were. Everything from here to
the end of this subsection is kept only as the record of what was frozen
before the gap was found, and MUST NOT be scored against.

The tanks' recorded history is the reference (`nemo_testcases_l1_phase3_ref_*`):
both are AT-BAR at kt=1 on every field, and first-over-bar at kt=2 — LOCK on
{T, u}, OVERFLOW on {T, u, ssh}. Predicted for VORTEX, bar 1e-15 normalized:

| kt | T | S | u | v | ssh |
|---|---|---|---|---|---|
| 1 | AT-BAR | AT-BAR | AT-BAR | AT-BAR | AT-BAR |
| 2 | DEBT | UNINFORMATIVE | DEBT | DEBT | DEBT |
| 3..10 | DEBT | UNINFORMATIVE | DEBT | DEBT | DEBT |

Reasons, one per row:

* **S is UNINFORMATIVE from kt=2**, as on both tanks: the field is uniformly
  35 and the gate's own rule downgrades a spatially uniform reference row.
* **v is a MEASURED row for the first time.** On both tanks it is UNMEASURED —
  no active meridional face exists in a three-row box. VORTEX has 61 wet rows,
  so this is the first card where the meridional momentum equation is scored at
  all, and the first where the EEN vorticity operator is live.
* **ssh is informative from kt=1**, unlike the tanks, where ssh starts at zero
  and its early rows are downgraded. VORTEX starts with a 0.916 m surface high.
* **DEBT at kt=2 is the EXPECTED outcome, not a failure.** Every card on this
  lane has entered at kt=2 debt; round 1 establishes where VORTEX enters and
  which fields own it, so round 2 can name the first non-bit statement.

**FALSIFIER for the round's premise:** if kt=1 is not AT-BAR on all five
fields, the transcription is wrong and no ladder row means anything. If kt=2 is
AT-BAR on every field, the card is better than any existing one and that claim
gets a plant before it is believed.

### 3.3 Invariants — existing cards must not move

| invariant | value | how checked |
|---|---|---|
| GYRE ladder digest | `cf06a8fc7d0e90f2` (unchanged; newest receipt `..._round170_mpas_tke_bc_receipt.md:149`) | the code GYRE executes is untouched, proven by the row below |
| LOCK_EXCHANGE-zco card digest | `42d13c75ea8cbcc6` | measured at `c09a9e111780` and with the VORTEX card present: identical |
| OVERFLOW-zps card digest | `c2bca636ac2f14ef` | as above |
| GYRE-zco card digest | `f227194da309e66b` | as above |
| push-gate tests | unchanged pass count | the six files in `autopilot_max.sh` |

The three card digests cover the resolved model config, the full initial T, S,
u, v, ssh and barotropic-velocity arrays, all three masks, the T-, U-, V- and
F-point Coriolis fields, the horizontal metric, the vertical coordinate
(reference ladder, interfaces, partial thicknesses, bottom index, active mask
and NEMO's own reference thickness), the land mask, and every card field. They
are measured with the fp64 precision policy the oracle cards run at, which the
test pins: the same card hashes differently under the library's fp32 default,
and a long mixed battery caught that the hard way. They are pinned
in `tests/ocean/unit/test_nemo_vortex_card.py`, so a later edit to the shared
identity that leaks into a certified card turns a test red rather than moving a
certified number quietly.

## 4. Non-vacuity — the new tests were shown to fail

Two defects were planted in the transcription and both turned the new gate red;
the planted file was restored and `git status --porcelain` verified empty.

| planted defect | which test fired |
|---|---|
| evaluate the vortex on the 1-D reference depth ladder instead of the live stretched depth | `test_temperature_uses_the_ssh_stretched_depth_not_the_reference_ladder` (4.026327615881819 vs 4.023400736344101 degC) |
| flip the meridional velocity sign (cyclonic instead of anticyclonic) | `test_initial_state_is_the_anticyclonic_source_vortex` |

The acquisition's record checker was exercised on synthetic records: it admits
a well-formed set, and refuses a corrupted header (its own `--plant`), a
perturbed reference restart, and a missing step.

## 5. What this round does NOT do

* It does not run NEMO. MPI is refused in the agent's sandbox.
* It does not write the kt=1..10 scorer. That is round 2, once the record
  exists; the existing trajectory gate hard-codes the tanks' header tuples and
  must be generalised to parse each record's own header (note BD) before it can
  score VORTEX.
* It does not re-anchor citations onto compiled `ppsrc` paths, which only exist
  after the build. Round 1 cites the shipped `tests/VORTEX/MY_SRC/*.F90` with
  line numbers; the ppsrc re-anchor is round 2.
