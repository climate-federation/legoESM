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
