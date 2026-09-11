# ORCA1 namelist vs our production tripole card — the rows that still differ

Written 2026-09-08 during the cluster maintenance window, when nothing can be
measured. Every row below is READ OFF THE CODE AND THE ORACLE'S NAMELIST, not
measured; each is a candidate one-variable arm once the machine returns.
Provenance: `cfgs/ORCA1/EXP00/namelist_cfg` (and `namelist_ref` where the cfg
does not override), against `scripts/cluster/omip_nemo/_trp_base2_d30.sbatch`
plus what `orca1_zdftke_config` / `NEMOMatchTripoleRecipeConfig` construct.

## Rows that DIFFER

| what | the oracle | ours | note |
|---|---|---|---|
| eddy-induced velocity coefficient | `ln_ldfeiv=T`, `nn_aei_ijk_t=21` (Treguier 1997), `rn_Ue=0.018`, `rn_Le=100e3`, i.e. aei0 = Ue·Le/2 = **900 m2/s**, spatially varying | constant `kappa_GM=600` | `--gm-treguier` EXISTS and the branch is named for it, but the production card does not pass it |
| lateral tracer diffusion | `nn_aht_ijk_t=21`, same Treguier form, `rn_Ud=0.018`, `rn_Ld=100e3` | constant `kappa_Redi=600` | same shape of gap as the row above |
| tracer advection | `ln_traadv_fct=T`, `nn_fct_h=2`, `nn_fct_v=2` (FCT2) | card passes `superbee` | our driver's own help calls `ppm_fct` "closest to NEMO's FCT2"; an earlier note records ppm_fct destabilising the Gulf Stream, so this row has a REASON, not just a gap |
| kinetic-energy gradient | `nn_dynkeg=1` (Hollingsworth) | `centered` | our recipe documents why: the Hollingsworth form is not fold-aware on the tripole seam. A real constraint, not an oversight |

### ★ The eddy row has the WRONG SIGN at the equator, and that matters here

Read off the oracle's own formula (`compute_treguier_kappa_gm`, transcribed
from `ldf_eiv`): the Treguier coefficient carries an explicit TROPICAL TAPER,
`min(1, |f/f_20|)` with `f_20 = 2*Omega*sin(20 deg)`. At the equator the
Coriolis parameter goes to zero, so the ORACLE'S EDDY COEFFICIENT GOES TO ZERO
THERE. Our production card uses a CONSTANT 600 m2/s at every latitude, and the
same is true of our Redi coefficient while the oracle tapers that one too.

So in the equatorial waveguide we are applying roughly 600 m2/s of eddy
transport and isopycnal mixing where the oracle applies essentially none. The
sign is the one that matters for every bias this campaign is chasing: spurious
eddy flattening of the equatorial thermocline weakens the zonal density
gradient, which weakens the undercurrent, which removes the shear that feeds
turbulence, which is the 20-60 m mixing collapse and the too-high Prandtl
number. Each link is individually plausible and the first two are measured.

This REVERSES the reason for running the eddy arm. It is not that we are
missing a flow-dependent coefficient somewhere useful; it is that we are
applying a large constant one exactly where the oracle applies none. Status:
the taper and our constant are CONFIRMED by reading both codes; the chain from
there to the biases is PLAUSIBLE and is what the arm tests.

(An independent reviewer ranked this row as irrelevant at the equator on the
grounds that eddy closures taper off in the waveguide. That is right about the
ORACLE and wrong about US, which is the whole point.)

### What `--gm-treguier` does and does not close (checked 2026-09-08)

The flag makes ONLY the eddy-induced coefficient flow-dependent. Our
implementation's own constant is `gm_aei0 = 900`, derived in the code as
half of the oracle's `rn_Ue * rn_Le` with the namelist provenance written next
to it, so row 1 above is exactly what the flag closes -- including the value.

There is NO equivalent path for the Redi/lateral-tracer coefficient: the module
exposes a Treguier kappa for GM only, and `kappa_Redi` stays the constant 600.
So row 2 survives the flag and would need its own work. Do not expect one arm
to close both.

## Rows that AGREE (checked, so they are not re-opened)

| what | value on both sides |
|---|---|
| momentum advection | vector-invariant (`ln_dynadv_vec=T` / `momentum_advection="vector_invariant"`) |
| hydrostatic pressure gradient | s-coordinate form (`ln_hpg_sco=T` / `--pgf-scheme smc03`) |
| penetrative solar | RGB with chlorophyll data (`ln_qsr_rgb=T`, `nn_chldta=1` / `--sw-rgb-chl`) |
| internal-wave mixing | on, and it forces the molecular backgrounds on BOTH sides (`ln_zdfiwm=T` / `--iwm`) |
| TKE closure | on, with Langmuir, sub-mixed-layer penetration at `rn_efr=0.08` and the latitude-dependent depth (`nn_etau=1`, `nn_htau=1`) |
| Prandtl number | Richardson-dependent, `nn_pdl=1`, same critical-Richardson slope |
| enhanced vertical diffusion | present on both, and separately shown not to matter here |
| tracer damping | off (`ln_tradmp=F`) |

## Reading

The first two rows are the same defect twice: the oracle makes both its eddy
transport and its lateral tracer mixing follow a Treguier coefficient that
varies with the flow, and we use a constant that is a third smaller at the
reference value. That bears directly on the equatorial thermocline, and unlike
the shear row it is a coefficient rather than a discretisation, so it is cheap
to test.

The last two rows have documented reasons and should NOT be flipped casually:
one destabilised the Gulf Stream when tried, the other is unsafe at the tripole
fold. They are recorded so nobody re-derives them.

### ★★ Below 20 m our momentum diffusivity is sitting on its FLOOR

Re-reading the corrected band table rather than running anything: our momentum
diffusivity is **2.95e-06 at 20-60 m and 2.95e-06 at 60-160 m** -- the same
number to three figures across two very different depth ranges. A closure does
not produce identical output in two regimes; a floor does. Under `--iwm` both
sides force the molecular backgrounds (momentum 1.4e-6, tracer 1e-10), and our
value sits just above the momentum one.

So the honest statement of the defect is NOT "our Prandtl number is 11.4 against
the oracle's 1.23". That ratio is what you get when BOTH diffusivities fall back
toward backgrounds whose own ratio is four orders of magnitude -- it is a
symptom of a dormant closure, not evidence of a limiter clamping. The real
statement is:

  **below the surface layer our TKE closure produces essentially nothing, while
  the oracle's is active there.**

That also means the Prandtl co-gate on the queued shear arm should be read as a
consequence, not as an independent test: if the closure wakes up, the ratio
falls on its own.

### What the eddy arm can and cannot be expected to fix

An adversarial review (GLM, 2026-09-08) retracted its earlier "irrelevant at the
equator" ranking once shown the taper, and sharpened the mechanism: at 1-3
degrees the cold-tongue front has slopes around 3e-4 over ~100 km, so a constant
600 m2/s gives a bolus streamfunction of order 0.2 m2/s and an eddy velocity of
a few mm/s -- a thermocline-flattening e-folding of roughly 200 days, against
roughly seven years at the tapered value. Material, not negligible. It added
that a constant coefficient across the equator, where the isopycnal slope
REVERSES SIGN, produces a spurious convergent vertical eddy velocity of order
1e-6 m/s parked in the undercurrent core -- 10-20% of the Ekman upwelling, and a
pathology rather than mere over-diffusion. It also notes the Gent-McWilliams
derivation itself fails inside the equatorial deformation radius, which is why
the oracle tapers at all.

BUT it named the weak link, and the point stands: the undercurrent is only about
2.5x too slow, so shear production (which goes as shear squared) is short by
roughly a factor of six -- nowhere near the orders of magnitude by which our
mixing at 20-60 m falls short. **The eddy arm should therefore NOT be
pre-registered to fix the mixing collapse.** It is a thermocline-and-undercurrent
test. The mixing collapse needs its own explanation, and the floor observation
above is the first hard evidence about it.

### ★★★ The profile, level by level: our turbulence dies at 18 m

Median over the 220 cold-tongue columns, our production baseline at day 30:

| depth | K_M | K_H |
|---|---|---|
| 1.02 m | 4.25e-03 | 4.25e-04 |
| 5.83 m | 6.23e-05 | 1.86e-05 |
| 10.77 m | 9.07e-06 | 1.12e-06 |
| 15.22 m | 3.02e-06 | 6.89e-07 |
| **17.93 m** | **2.9929e-06** | 3.49e-07 |
| 24.60 m | 2.9607e-06 | 2.47e-07 |
| 50.45 m | 2.9486e-06 | 2.44e-07 |
| 102.42 m | 2.9467e-06 | 1.70e-07 |
| 155.10 m | 2.9469e-06 | 1.60e-07 |

From 18 m to 155 m the momentum diffusivity is FLAT TO FOUR SIGNIFICANT FIGURES
across 140 m of ocean. No closure does that. It is the background (the molecular
1.4e-6 both sides force under internal-wave mixing, plus the wave field's own
contribution), and our TKE closure contributes essentially nothing below 18 m.
The oracle in the same band carries 1.9e-03, roughly five thousand times more.

The floors are NOT the difference and must not be blamed: our card takes them
from the oracle's own code path (internal-wave mixing forces the TKE floor to
1e-10 and the mixing-length floor to 1e-3, overriding the TKE namelist), and an
earlier campaign step showed that NOT applying them pinned our equatorial
turbulence a hundredfold too high and damped the undercurrent. Both sides run
the same floors. What differs is that the oracle's closure is PRODUCING
turbulence at these depths and ours is not.

That makes the queued shear arm the right next test: shear production is the
term that should be sustaining turbulence there, and it is the one row of the
closure still discretised differently from the oracle. It also explains why the
system is hard to shift -- with the diffusivity at background, the TKE equation's
own downward transport is proportional to that same tiny coefficient, so the
quiet state sustains itself.

## 2026-09-10 — the DINO NEMO-literal TKE bundle never reaches ORCA1

Codex's CRITICAL asked which diffusivity our shear production consumes. Answered
in code, no run:

`tke.py:2746` sets `_carried_coeffs = (_coeff_source == "carried_previous_step")`,
and `tke.py:3086-3088` then feeds shear production either the carried pair or the
freshly recomputed one:

    _K_M_pre = preclosure_K_M if _carried_coeffs else K_M_curr
    P_s_curr = _K_M_pre * shear_sq   (or the face-native functional of it)

`config.py:586` defaults the field to `current_subiteration`. A repo-wide grep
shows `orca1_zdftke_config` (`run_omip_core2.py:508`) never sets it, so **every
OMIP/ORCA1 run to date evaluates shear production with a coefficient it
recomputed in the same call, not with NEMO's carried `avm_k`.** NEMO's `tke_tke`
consumes the carried pair and overwrites it only after the solve, so codex's
mechanism is real and present: a low recomputed K makes low production, which
makes low K.

This is not a missing feature. `carried_previous_step` is implemented, seeded at
cold start (`ocean_model_latlon_cgrid.py:7255` reproduces `zdf_phy_init`'s
background-times-wmask construction), restart-bridged, unit-tested
(`tests/ocean/unit/test_tke_carried_coefficients.py`), and **selected by the
DINO NEMO-oracle preset** (`experiments/dino.py:1247`). It is simply not wired
into the ORCA1 card.

The same is true of the rest of the bundle DINO's preset carries and ORCA1 does
not: `tke_matrix_evaluation`/`tke_solver_evaluation` = `nemo_literal`, and the
`nemo_literal` evaluations of etau/htau/mxl/Langmuir. Two are BLOCKED rather
than unasked: `tke_surface_bc_level="nemo_z0"` was reverted on the ORCA1 card
for a real metric defect (`dz_surface` is the top cell's midpoint, half
`e3t(1)`, which doubles the virtual-surface coupling), and the literal matrix
raises unless `nemo_z0` is selected, so the literal solver is downstream of that
same defect.

SCOPE. The coefficient memory (`tke_avm`, `tke_avt`, `tke_avm_surface`,
`tke_dissl`) lives on `LatLonCGridOceanState` only (`state.py:558-566`). MPAS and
FESOM2 have no slot, so this row can be harmonized on the tripole today and
needs state fields on the other two grids before the three-grid card agrees.

STATUS: measured gap, no default changed. The flip is a scientific choice and is
being put to the user before any arm is submitted.

## 2026-09-10 (b) — the surface-wave terms cannot reach the deficit band, in
## either model

Two measurements and one code reading, all cheap, all on the matched window
(our day-30 snapshot against GATEWAY record 5).

MEASURED, and new: our turbulence stops at 17.9 m while the oracle's reaches
73.2 m (median depth of the last interface with K > 1e-4 m2/s in the cold-tongue
box). From temperature alone, which depends on no diffusivity convention, our
mixed layer is 2.7 m against the oracle's 5.1 m. So the deficit is not a uniform
scaling of the coefficient: our turbulent layer simply ENDS at the base of our
mixed layer, where the oracle's continues four times deeper.

READ OFF THE CODE, not measured: NEMO's sub-mixed-layer TKE penetration
(nn_etau) decays as exp(-z/h_tau) with h_tau = max(0.5, min(30, 45*|sin phi|))
under nn_htau=1 (tke.py:306, and orca1_zdftke_config sets etau_mode="below_ml",
etau_frac=0.08, etau_htau_mode="latitude" -- faithful to ORCA1's namelist). In
the cold-tongue box, |lat| <= 2 deg, that e-folding depth is 0.4-1.6 m. The term
is therefore confined to the top metre or two IN BOTH MODELS, and it cannot be
the source of the oracle's turbulence at 20-73 m. The same argument retires
Langmuir (ln_lc), which is surface-trapped by construction.

CONSEQUENCE, and it narrows the field rather than widening it: whatever sustains
the oracle's turbulence in the deficit band is either generated locally there,
or TRANSPORTED down from the surface layer by the TKE equation's own vertical
self-diffusion. That self-diffusion is proportional to the very coefficient
under test, which is the mechanism the carried-coefficient arm acts on. Local
generation by shear discretisation was already refuted by its own arm.

STATUS: etau and Langmuir eliminated for this band. Nothing changed in the model.

## 2026-09-10 (c) — RETRACTION: "above 20 m we match" was the wrong oracle file

The corrected band table (our day-30 snapshot against GATEWAY record 5, the
matched window, with the record-index bug fixed and the 0-20 m self-check
passing exactly):

| band | tracer ours | tracer NEMO | ratio | Prandtl ours | Prandtl NEMO |
|---|---|---|---|---|---|
| 0-20 m | 2.60e-05 (median 3.38e-05) | 1.423e-03 | 0.024 | 4.09 | 4.66 |
| 20-60 m | 2.595e-07 | 8.423e-04 | 0.0003 | 11.38 | 1.09 |
| 60-160 m | 1.684e-07 | 8.205e-07 | 0.205 | 17.49 | 2.66 |

RETRACTED: I have been saying the top 20 m MATCHES the oracle (ratio 1.02-1.04).
That number came from the 2001 hourly RUN_TRD2 file, a different year from our
run window. On the matched window the top 20 m is 0.024 of the oracle -- a
factor of 42 too weak, not a match. Every statement built on "the deficit is
confined to a narrow band below the mixed layer" is withdrawn with it.

REVISED SHAPE. The deficit spans the whole upper 60 m and only closes below it:
42x too weak at 0-20 m, ~3250x at 20-60 m, and within a factor of 5 (tracer) or
1.35 (momentum) at 60-160 m. That is consistent with the turbulent-layer depth
measured the same way -- ours ends at 17.9 m, the oracle's at 73.2 m.

WHAT SURVIVES, and it is informative: the Prandtl ratio AGREES at 0-20 m (4.09
against 4.66) and disagrees only where the coefficient has collapsed to
background (11.38 against 1.09 at 20-60 m). The closure's momentum-to-tracer
partition is therefore faithful where there is any turbulence at all; what is
missing is the AMPLITUDE, i.e. the energy. That is the same conclusion the
production/dissipation arithmetic reached from the other side.

REVISED PRE-REGISTRATION for the running carried-coefficient arm (job 9692852).
The primary threshold was written against the wrong oracle value:
  * PRIMARY, unchanged as a trigger: 20-60 m tracer K rises 10x from 2.595e-07.
    Below 2x still REFUTES.
  * But FULL CLOSURE of that band now needs ~3250x, not 10x, so a 10x rise is
    evidence the mechanism is live -- it is NOT an adoption, and must not be
    reported as closing the band.
  * ADDED, and cheaper to read than any ratio: the turbulent-layer depth must
    move from 17.9 m toward 73.2 m. It is a depth, so it cannot be inflated by
    unclamping a limiter the way a coefficient ratio can.

## 2026-09-10 (d) — the energy ARRIVES and is destroyed within a few metres

Read from the baseline day-30 snapshot, cold-tongue box, no simulation. Our own
turbulent energy by depth:

| depth | our TKE [m2/s2] |
|---|---|
| 1.02 m | 1.30e-03 |
| 4.49 m | 2.33e-05 |
| 21.04 m | 6.66e-08 |
| 57.40 m | 2.16e-08 |

Four orders of magnitude lost inside the top 20 m, and NEMO's rn_emin is 1e-10,
so this is not the floor binding -- the energy is genuinely gone.

THE SOURCE IS NOT THE PROBLEM (PLAUSIBLE, not confirmed). For a typical
cold-tongue trade stress of order 0.05 N/m2, NEMO's Dirichlet condition
rn_ebb*|tau|/rho0 with rn_ebb=67.83 gives about 3.3e-03 m2/s2. Our value one
metre down is 1.30e-03, the same order. The probe could not close this properly
because the snapshot carries no wind-stress field (INSTRUMENT DEBT: it saves T,
S, u, v, tke, the two diffusivity diagnostics and the grid, but no stress), so
the comparison rests on a typical stress rather than our own. Labelled PLAUSIBLE
until the stress is saved.

WHAT THAT LEAVES. Energy arrives at the surface at roughly the right size and is
destroyed before it reaches 5 m. The decay between 1 m and 4.5 m is a factor of
56. In this closure only two terms can do that: Kolmogoroff dissipation, which
goes as e^1.5/l, and the failure of the TKE equation's own vertical
self-diffusion to carry energy downward. That self-diffusion is proportional to
our momentum coefficient, which in the same box and the same 0-20 m band is
1.38e-04 against the oracle's 6.62e-03 -- 48x weaker. So the downward transport
channel is 48x weaker exactly where the energy is being lost, which is the loop
the running carried-coefficient arm acts on.

This is the pre-registered "source faithful, loss downstream" outcome of the
surface-TKE probe, and it supports the arm rather than replacing it.

## 2026-09-10 (e) — VERDICT: the coefficient lifetime is REFUTED

The carried-coefficient arm ran to completion (jobs 9693002 leg 1, 9695257
leg 2) and is scored against its pre-registration, matched protocol throughout
(our day-30 snapshot vs GATEWAY record 5), one variable against trp_base2_d30.

| metric | baseline | arm | oracle | pre-registered |
|---|---|---|---|---|
| 20-60 m tracer K | 2.595e-07 | 2.923e-07 | 8.423e-04 | rise 10x; <2x refutes |
| turbulent-layer depth | 17.9 m | 24.6 m | 73.2 m | move toward 73.2 |
| 20-60 m Prandtl | 11.38 | 10.19 | 1.09 | fall toward 1.09 |
| 0-20 m tracer K | 3.382e-05 | 4.722e-05 | 1.423e-03 | -- |
| SST rmse | 0.5003 | 0.5004 | -- | within 0.05 |
| nino3 bias | +1.12 | +1.12 | -- | see note |
| EUC core 220E | 0.199 | 0.200 | 0.546 | rise toward 0.546 |

**REFUTED.** The primary rose 1.13x where 10x was required and below 2x was
pre-registered as refuting. The mechanism codex ranked CRITICAL twice -- that
evaluating shear production against a freshly recomputed coefficient rather
than NEMO's carried avm_k self-starves the closure -- does not carry the
equatorial mixing deficit. Coefficient lifetime joins the eliminated list.

WHAT DID MOVE, and it is honest to report it: the turbulent layer deepened
17.9 -> 24.6 m and the 0-20 m coefficient rose 1.40x. Real but small, and both
are far short of the oracle. The Prandtl co-gate barely moved (11.38 -> 10.19),
which is consistent: with the coefficient still at background in that band, the
ratio has nothing to respond to.

NOTE ON THE nino3 GUARDRAIL, so it is not misread: it was written as
"<= +1.10", but the BASELINE is itself +1.12. That is a mis-set threshold from
an older run, not a breach caused by the arm -- the arm and the baseline agree
to three digits on every surface metric.

THE ARM WAS STILL WORTH RUNNING. It exposed a real defect: carried_previous_step
could not survive its own second timestep on any card except the nemo_z0 ones,
because the cold-start seeder required a surface coefficient the closure only
produces under nemo_z0. Fixed at 82cc4fe81 and completed here for the
nemo_z0 + mxl_choice 1/2 combination codex flagged. The option is now usable;
it simply is not the lever.

NEXT, and it needs the user's decision: the surface boundary PLACEMENT is now
the leading untested candidate, and it is blocked behind one metric defect.

## 2026-09-11 — surface PLACEMENT landed as faithfulness, not run as a lever

TWO RETRACTIONS OF MY OWN, both caught before they cost anything.

1. I told the user a dz_surface metric defect BLOCKED NEMO's surface placement
   and asked them to approve fixing it. That defect was already fixed by #1690:
   `tke_vertical_mixing` derives `dz_face_surface = dz_ref[0]*jacobian` for the
   virtual-surface face, takes e3t(1) from the frozen bundle under
   `n2_evaluation_stage="step_entry"`, and `k_profiles` already threads both
   operands. Only the ORCA1 card COMMENT still recorded the revert. Codex 9698860
   confirmed the full e3t(1) is used on the tripole path and the midpoint metric
   never reaches that face. The comment is now corrected in place.

2. My own precondition guard validated the card DEFAULTS rather than the
   requested values, because it ran before `surface_bc` and `tke_mxl_choice`
   were applied. Its own test failed on that, which is what the test was for.
   The block now runs last.

GLM'S MECHANISM REVIEW, and why no arm was run. Asked whether displacing the
surface boundary one level can cost three orders of magnitude in the
diffusivity 20-60 m below, GLM answered percent-level, with the reasoning:
a boundary perturbation decays over sqrt(K_e * tau_diss), of order 1-10 m
given eps ~ e^{3/2}/l, so at 20 m it is worth at most a few to ten percent and
by 60 m essentially nothing; and K ~ l*sqrt(e) halves the sensitivity again.
Below the pinned node the column still receives the same imposed energy, and
at 20-60 m the budget is set by local production-dissipation balance, not by
diffusive memory of the surface value.

GLM named exactly one route to orders of magnitude: a regime flip, where the
clamp pushes a level onto the rn_emin floor or trips a stability limit. THAT
ROUTE IS CLOSED HERE BY MEASUREMENT -- our energy sits 283x above its floor
and the length 124x above its own, both recorded earlier in this ledger. It
also agrees with the independent ~4% column-probe estimate from a prior
session.

DECISION (user, 2026-09-11): land the flag as a faithfulness item, default
unchanged, and do not spend the GPU-hours on a controlled arm. The option is
selectable for anyone who wants it and the stale prose no longer misleads.

STATUS OF THE EQUATORIAL DEFICIT: every named mechanism is now eliminated --
enhanced vertical diffusion, mixing length, shear discretisation, advection,
floors, backgrounds, the Prandtl form, nn_etau and Langmuir, the 1.5/0.5
dissipation split, the coefficient lifetime (measured null), and now the
surface placement (argued small by two independent estimates, not run).
The cause is OPEN and honestly unknown. Do not re-open the list above without
new evidence; the next move should be a fresh measurement, not another arm on
an eliminated lever.
