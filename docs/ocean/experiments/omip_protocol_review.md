# OMIP-1° in legoESM — protocol review and gap analysis

**Author**: ocean-model-expert subagent
**Date**: 2026-05-02
**Inputs**: Griffies et al. (2016) OMIP/CMIP6 protocol paper (`docs/references/`,
gitignored, do not commit); current scoping plan
`omip_1deg_plan.md`; realistic-geometry plan + Phase 4(c) results;
`tripole_grid_plan.md`; `realistic_geometry_forcing_literature_review.md`.

This is a planning/research document. No source edits proposed here.

---

## 1. TL;DR

- **Polar grid**: do **not** invest in tripole for the OMIP-1 pilot. Run on
  regular 1° lat-lon with the existing cos²(lat) A_h scaling + ≥80°N polar
  cap and a modest zonal Fourier filter on tendencies poleward of ~78°N.
  This is what the GFDL CM2-class generation actually shipped, and it's
  defensible for *first* AMOC, BSF, MHT, MOC numbers. Tripole becomes
  mandatory only when the Arctic *itself* is the science target or you
  push past ~0.5° (`tripole_grid_plan.md` is the right plan when that
  time comes — keep it on the shelf).
- **Sea ice**: OMIP/CORE-II *requires* dynamic-thermodynamic sea ice.
  There is no acceptable shortcut for a "compliant" submission — the
  protocol's own salinity-restoring rationale (Sect. 2.3) and the ice
  diagnostic block (Wang et al. 2016a,b in the citation chain) assume
  it. **Recommendation: phase the experiment**. Phase A = 60°S–60°N
  "tropical-OMIP" pilot with prescribed ice (~6 months); Phase B =
  bring our existing EVP `ice/sea_ice.py` up to global validation
  (~3 months) and do the full OMIP cycle. Do not attempt full OMIP
  with prescribed ice and call it OMIP — call it a forced-ocean
  validation run.
- **Forcing**: `omip_1deg_plan.md` says "ERA5 *or* JRA55-do" — that
  ambiguity must close. **Use JRA55-do v1.4+** (it *is* OMIP-2). ERA5
  is fine for development but a final JRA55-do cycle is the deliverable.
  The plan is also missing the Large & Yeager (2009) bulk-formula
  modifications to the raw reanalysis fields — these are part of the
  protocol, not optional.
- **Spin-up**: the plan's "10-yr pilot then 60-yr cycle" is **not
  protocol-compliant** as written. OMIP-1 mandates 5–6 cycles of the
  full 1948–2009 (CORE-II) or 1958–2018 (JRA55-do) record (Griffies
  2016 Sect. 2.2; reproduced verbatim in 2.3 of OMIP-2). 5 cycles ≈
  300 yr, not 60. The plan as written produces a one-cycle weakly
  equilibrated pilot — fine as a milestone, but should be labelled
  "OMIP-style spinup, not OMIP-compliant".
- **Diagnostics**: the plan lists ~5 diagnostics (MOC, AMOC@26.5N,
  Drake, BSF, climatologies). OMIP Tables H, I, J, K, L, M, N
  request ~150 fields, ~16 strait transports, full heat/salt budgets,
  ideal age, and online MLD. The plan's diagnostic budget is
  ~10× too thin to claim OMIP compliance. Most of the gap is
  recoverable cheaply (online accumulation, native grid output) but
  it has to be designed in from day 1, not retrofitted.

The rest of this document explains each call.

---

## 2. Polar grid — recommendation: stay on regular 1° lat-lon for the pilot

### What the protocol actually requires

OMIP is grid-agnostic. Griffies (2016) Appendix A3.1 explicitly states
"spherical latitude–longitude grids are rarely used as the native grid
for global ocean simulations" but does *not* prohibit them — the only
horizontal-grid-related requirement is that diagnostics be archivable
on the model native grid *plus* be regriddable to the standard
WOA13/Levitus 1° spherical analysis grid (App. A3.2–A3.3). Any
correctly conservatively regridded native grid is acceptable for
submission.

So: lat-lon is *allowed*. The question is whether it's *defensible
science*.

### Can a regular 1° lat-lon get scientifically defensible results?

For the OMIP big-three diagnostics (AMOC, ACC/Drake, MHT) at
non-eddying resolution, **yes, with documented caveats**. The case:

- **Historically**, several CMIP-class submissions used displaced-pole
  lat-lon (POP/CESM `gx1v6` is *one* displaced pole, not a tripole;
  GFDL CM2.0 OM3 was Mercator with filtering). The fully-symmetric
  (no-displacement) version with a polar cap is rarer in CMIP but is
  exactly what the original Bryan-Cox-Semtner generation used for two
  decades, and it's what Wolfe & Cessi (2010), Saenko & Merryfield,
  and most analytical/process AMOC studies still use.
- The convergent-meridian problem at 1° gives Δx ≈ 0.5 km at 89.5°N.
  The barotropic CFL is solved by `barotropic_solver="implicit_cn"`
  (Phase 4(c) confirmed this works on real ETOPO). The remaining
  problem is **horizontal viscosity stability**: A_h * dt / Δx² ≤ 0.5
  blows up at the pole unless A_h scales as cos²(lat) — which it
  already does in this branch (`Realistic-geometry stability:
  cos²(lat) A_h scaling + polar cap`, commit c1fbb038).
- The dominant *scientific* failure mode of regular lat-lon at the
  pole is not stability — it's that the convergent-meridian noise
  pollutes Arctic dynamics. For OMIP, this matters for **Arctic sea
  ice extent and Arctic freshwater export through Fram/Davis**. It
  matters relatively little for AMOC@26.5N or Drake or global OHC
  trends.

### Concrete plan: lat-lon + cap + zonal Fourier filter

Adopt three layers in order of cost:

1. **cos²(lat) A_h scaling + ≥80°N polar cap with locked u,v=0** —
   already in tree (`Phase 4(c)` audit). This survives 100-yr
   integrations.
2. **Zonal Fourier (FFT) tendency filter poleward of ~78°N** — a
   1-line addition: take FFT of the zonal tendency, zero modes with
   wavelength shorter than the local Δx-equivalent, IFFT. This is
   the standard POP/MOM4 trick (Smith et al. 1995, *J. Geophys. Res.*
   for POP; standard in MOM3/MOM4 source). Cost: 1–2 days. Gain:
   suppresses the residual 2Δx Arctic noise that A_h scaling alone
   cannot reach because A_h ∝ cos² collapses to zero at the pole.
   Differentiability is fine — FFT/IFFT are linear and AD-clean.
3. **Document the Arctic as untrustworthy** in the diagnostics report.
   Specifically: do not claim OMIP-quality Arctic sea-ice volume or
   Fram Strait freshwater on this grid. AMOC, Drake, MHT, MOC, BSF,
   global OHC trends all OK.

### When does tripole become mandatory?

The decision tree:

| Trigger | Stay lat-lon | Move to tripole |
|---|---|---|
| Resolution ≤ 1° | yes | no |
| Resolution 0.5°–0.25° | marginal | yes |
| Arctic-focused science | no | yes |
| Eddying Arctic / Beaufort Gyre | no | yes |
| First-pass AMOC/MHT/Drake at 1° | yes | no |

The current task is in row 1 column "yes" — pilot first, port to
tripole when (a) resolution increases, (b) Arctic becomes a science
target, or (c) sea-ice dynamics validate at high lat (where the polar
cap kills them anyway).

### If we *did* port to tripole — implementation cost

`tripole_grid_plan.md` already scopes this honestly: ~7 weeks focused
work, ~1500 LOC, ~85% reuse of the lat-lon C-grid stack. The hard
parts in JAX:

- **Fold halo exchange** (1 week): the bipolar-cap seam is a north
  boundary where cell `(i, jmax)` is physically the same as
  `(N_i − i + 1, jmax)`. In JAX this is a static `lax.gather` —
  cheap, AD-clean. Vector fields need a sign-flip on both
  components across the seam; that's another `where` mask.
- **Vector rotation in the cap** (4 days): per-edge `(cos α, sin α)`
  arrays loaded from the grid file; rotate u,v components into the
  receiving cell's local frame on every halo crossing into the cap.
  Rotation arrays are static — no recompilation, no AD pain.
- **Metric-array refactor of `latlon_cgrid_operators.py`** (1 week):
  every `cos(φ)` factor becomes a per-cell `dx_T, dy_T, area_T,…`
  array. Mostly mechanical. The bit-exact regression test on
  regular lat-lon is the only stability gate.
- **The genuine risk**: GM/Redi triads + flux-form tracer advection
  schemes (SOM, DST-3, WENO) crossing the fold need orientation-aware
  stencils. SOM in particular carries 9 sub-grid moments per cell —
  folding those correctly is non-obvious. `tripole_grid_plan.md` Phase
  5 calls this out and recommends falling back to `slope_scheme=
  "centered"` on the cap row if triads-through-fold fails. That's a
  defensible degradation.

**Net**: tripole is real work but **bounded** work. The right
sequencing is "lat-lon OMIP pilot → tripole port → tripole OMIP
production". Doing tripole *before* OMIP brings up two big unknowns
in one branch and is exactly the anti-pattern
`realistic_geometry_lat_lon_plan.md` argued against.

### Intermediate options I considered and rejected

- **Displaced single-pole lat-lon (POP gx1v6 style)**: gives you the
  *worst* of both worlds for our use — needs a custom grid generator,
  needs metric arrays anyway, doesn't fix the symmetric problem at
  the displaced pole. If we're paying the metric-array tax we should
  pay the bipolar-cap tax once and be done.
- **Zonal FFT filter alone (no cap)**: works for stability but doesn't
  help the under-resolution issue (Δx → 0). Cap is structurally
  cleaner.
- **Polar cap big enough to swallow the entire Arctic (>70°N)**:
  drops a lot of real ocean. Don't.

---

## 3. Sea ice — recommendation: phase the experiment

### What the protocol actually requires

Read the protocol carefully (Griffies 2016 Sect. 2.2 "OMIP/CORE-II
experimental protocol"):

> Sea-ice fields are generally initialized from an existing state taken
> from another simulation, set to the January mean state from that
> simulation.

> Bulk formulae for computing turbulent fluxes for heat and momentum
> must follow Large and Yeager (2009).

And the diagnostic block — the SIMIP companion (Notz et al. 2016) and
the OMIP boundary-flux tables (Tables K1, K2, K3 in this paper)
include `fsitherm` (water flux from sea-ice thermodynamics), `sfdsi`
(downward salt flux from sea ice), `hfsifrazil` (frazil heat flux),
`hfsnthermds` (latent heat from snow melt), `ficeberg`, etc. The
*entire* salt-budget section K3 and the salinity-restoring rationale
in Sect. 2.3 ("the key reason for salinity restoring relates to
high-latitude thermohaline and ocean/sea-ice processes") presume an
interactive ice model.

So formally: **OMIP-1 cannot be submitted without a dynamic-
thermodynamic sea-ice model**. Prescribed-ice is not a protocol-
allowed simplification.

### Reality of legoESM's ice stack

`ice/sea_ice.py` + `dynamics.py` + `rheology.py` are functional. They
have:
- EVP rheology (Hunke & Dukowicz 1997) — production approach.
- Slab thermodynamics with diagnostic free-drift fallback.
- Multi-category transfer (simplified, *not* Lipscomb 2001).
- No snow tracking.

The integration tests under `tests/unit/` exercise dynamics in
isolation and through the coupler stack. **What's missing**:
- A global sea-ice spinup test against NSIDC climatology (extent,
  thickness, drift). I see no such test.
- The Wang et al. (2016a,b) Arctic freshwater diagnostics that OMIP
  expects.
- Coupling stability test under realistic Arctic forcing (cold air,
  strong winds, brine rejection driving open-ocean convection in
  Labrador/Weddell).

This is exactly what `omip_1deg_plan.md` lists as "needs global
validation". I agree. Do not run OMIP with our ice as-is.

### Physical consequences of skipping ice

Skipping ice and substituting prescribed concentration + freeze-T cap
is *not* a free pass on the ocean physics. The four things that go
wrong:

1. **Brine rejection → no DWF in marginal ice zones**. Real Labrador
   Sea convection happens because brine from ice formation makes
   surface water dense enough to overturn. Prescribed concentration
   does not move salt. Without it, the only sinking mechanism left
   is open-ocean cooling, and the model picks the *wrong* DWF site
   (likely north Norwegian Sea or open Arctic) — see
   `realistic_geometry_forcing_literature_review.md` §2 for the
   Wolfe-Cessi-style analysis of this.
2. **Arctic halocline disappears**. The Arctic mixed layer is
   stratified by salinity, not temperature, because ice melt freshens
   the surface. Without an ice freshwater flux, the Arctic restratifies
   thermally and the ocean gets too warm at depth.
3. **Freshwater budget breaks at high latitudes**. OMIP's salinity
   restoring is *gentle* (~50 W/m²/K equivalent piston velocity, ~1.5e-7
   m/s). It's intended to damp drift, not to substitute for ice
   freshwater forcing. Without ice, salinity restoring has to do
   real work it wasn't designed for, and the resulting circulation
   is restoring-controlled rather than physics-controlled.
4. **AMOC is suspect**. Per (1)–(3), AMOC source water is not
   physically generated. The number you get is a number, not a
   diagnosis.

The 60°S–60°N option ("tropical-OMIP") sidesteps all four by closing
the domain at the polar caps. **This is a legitimate published
configuration** — see the regional-OMIP work cited in Tseng et al.
(2016, "North and equatorial Pacific" CORE-II paper) and the more
recent FAFMIP-style basin-only experiments. It cannot diagnose AMOC
*pathway* but it can diagnose AMOC *strength at 26.5°N* if the southern
boundary is south of the Southern Ocean MOC return cell (~45°S is
typical, 30°S is borderline). 60°S works.

### Concrete recommendation

Phase the experiment:

**Phase A — 60°S–60°N tropical-OMIP pilot** (1–2 months wall + dev):
- Closed northern boundary at 60°N, southern at 60°S.
- Sponge layers in the polar 5° (T,S relaxed to WOA monthly clim).
- No ice model. SST under the (closed) cap relaxed to T_freeze.
- JRA55-do forcing inside the domain. 1 cycle, 60 yr.
- Deliverable: AMOC@26.5N, Drake, ITF, MHT, equatorial undercurrent.

**Phase B — Global OMIP** (3 months wall + dev, gated on Phase A):
- Sea-ice validation against NSIDC: extent (Sept Arctic min, Sept
  Antarctic max), thickness (PIOMAS, ICESat), drift speed (NSIDC
  Polar Pathfinder).
- Coupled-stability stress tests: cold-air-outbreak Labrador Sea,
  Weddell open-ocean convection event.
- Then global OMIP, 5 cycles, JRA55-do, with ice.

The phase-A cost is real but the deliverable is publishable on its
own ("legoESM's first global ocean simulation, validated against
JRA55-do for AMOC, Drake, MHT") and de-risks the ice work for phase B.

### Minimum-viable validation set for `sea_ice.py` before full OMIP

Run before any global-with-ice submission:

1. **NSIDC extent test**: 5-yr forced run with JRA55-do; September Arctic
   minimum within 30% of obs (legoESM resolution can't do better);
   September Antarctic max within 30%.
2. **Ice thickness pattern**: 5-yr mean thickness pattern qualitatively
   consistent with PIOMAS — thick in the central Arctic, thin in
   marginal seas, near-zero in summer Antarctic.
3. **Drift speed**: pan-Arctic mean drift within factor 2 of NSIDC PP
   climatology (this is a hard test for EVP — common bug source).
4. **Brine rejection**: surface salinity tendency in the Labrador Sea
   in winter is positive (the protocol that must hold for AMOC).
5. **AD compatibility**: `jax.grad` through one full ocean+ice step
   over a small Arctic patch produces finite gradients. EVP's
   subcycled momentum solver is a real risk here.

Items 1–4 are 3–4 weeks of work each if you're starting from a
working ice model; total ~3 months if they all need iteration. The
schedule estimate in the omip plan ("functional, needs global
validation") is honest about this not being free.

---

## 4. Other significant gaps in `omip_1deg_plan.md` (rank-ordered)

### Gap 1 — Spin-up protocol non-compliance (severity: HIGH)

**What's there**: "10-yr pilot then 60-yr OMIP cycle if pilot succeeds."

**What OMIP-1 actually requires** (Griffies 2016 Sect. 2.2,
"Simulation length"):
> simulations run for no less than five cycles of the 1948–2009 forcing
> have proven useful... For OMIP in CMIP6, we ask for output from
> cycles one through five.

That's **5 × 62 yr = 310 years**, with output from each cycle. 60 yr
= 1 cycle = "preliminary spin-up", not "OMIP submission". The
literature backs this up — Danabasoglu et al. (2014) and Griffies
et al. (2014) explicitly argue 5 cycles is *insufficient* for full
deep equilibration but is the agreed minimum.

**What to add to the plan**:
- Rename "60-yr OMIP cycle" → "OMIP cycle 1 (years 1–62)".
- Budget compute for cycles 1–5 (~310 yr at 1°). At the plan's
  estimated 0.5 s/step × 24×365 steps/yr ≈ 4400 s/yr = 1.2 GPU-h/yr,
  five cycles = ~370 GPU-hr. That's 2 weeks of dedicated GPU. Doable
  but plan accordingly.
- Save restart at end of each cycle — this is the "decadal mean at
  decadal intervals" requirement (Griffies 2016 Sect. 3.3).
- Decide *now* whether to do the JRA55-do 1958–2018 record (61 yr,
  OMIP-2) or the CORE-II 1948–2009 record (62 yr, OMIP-1). They
  are formally different experiments. **Recommendation**:
  JRA55-do/OMIP-2.

**Effort to add**: ~1 day to update the plan + the wall-clock budget
(~370 GPU-hr).

### Gap 2 — Diagnostic protocol coverage (severity: HIGH)

**What's there**: SST, SSS, SSH, BSF, MOC, AMOC@26.5N, Drake transport,
sea-ice extent. ~10 diagnostics.

**What OMIP requires**: ~150 fields across Tables H (scalars), I
(vectors/transports), J (16 strait transports), K (boundary fluxes:
mass K1, salt K2, heat K3, momentum K4), L (heat/salt budgets), M
(vertical SGS parameter coefficients), N (lateral SGS coefficients).
Many of these are mandatory Priority-1 — they are not optional.

The structurally important gaps:

- **Mass transport through 16 named straits** (Table J1: Barents,
  Bering, Caribbean, Davis, Denmark, Drake, English Channel,
  Faroe-Scotland, Florida-Bahamas, Fram, Gibraltar, Iceland-Faroe,
  Indonesian Throughflow, Mozambique, Pacific Equatorial Undercurrent,
  Taiwan-Luzon). The plan mentions only Drake. Each of the others
  is one named transect path through the model native grid; computing
  them online is cheap if designed in. Retrofitting is painful because
  it needs the native-grid path snapped to the section endpoints
  (Sect. C4 zigzag method).
- **Mixed layer depth** (mlotst, mlotstmax, mlotstmin) using the σ-t
  criterion of Levitus 1982 with ΔB_crit = 0.0003 m/s² (App. H24).
  Computed online using local potential density referenced to
  surface. Cheap to add but must be online (can't be reconstructed
  from monthly means accurately).
- **Online MOC accumulation**: the residual-mean overturning
  streamfunction `msftmyz` (depth-space) and `msftmrho` (density-space,
  σ-2000 referenced) by basin (Atlantic-Arctic, Indian-Pacific,
  Global). Density-space MOC requires online binning by σ-2000 — that
  needs designing in. Plan does not mention `msftmrho`.
- **Heat & salt budgets** (Appendix L — `hfx`, `hfy`, plus partition
  into resolved advection, parameterized mesoscale, parameterized
  submesoscale, parameterized diffusion). Without these,
  process-based AMOC analyses (Buckley & Marshall 2016 etc.) are
  not possible. Cheap to add online; expensive to retrofit.
- **Boundary fluxes** (Tables K1–K4): ~30 fields including wfo,
  wfonocorr, wfcorr, hfds, hfsifrazil, hfgeou (geothermal — set to a
  static map per de Lavergne or zero), tauuo, tauvo. Most are
  trivial outputs of the bulk-flux + ice modules; they need wiring.
- **Ideal age tracer** (App. H23): a single passive tracer with `A=0`
  source at surface and ∂t = 1 elsewhere. Cheap to add (one tracer)
  but must run for the full spin-up to be meaningful.
- **Native-grid + spherical-grid output** for tracer fields (App.
  A3.3). The native + 1° regrid is required for Priority-1 tracers.
  Need to wire up an online or offline conservative regridder.

**Effort to add**: 1–2 weeks if designed in from the start of phase 1
(diagnostics module, online accumulators, native grid output, plus a
post-process stage for spherical-grid regridding). 4–6 weeks if
retrofitted to a working pilot. Strongly recommend doing this in
Phase 4 of `omip_1deg_plan.md` *before* the pilot run starts.

### Gap 3 — Bulk formula compliance with Large & Yeager (2009) (severity: MEDIUM-HIGH)

**What's there**: `coupler/bulk_flux.py` (skim shows COARE 3.0) and
`ocean/physics/surface_forcing/bulk_formulas.py` (LY04). The plan
says "COARE 3.0 + LY04".

**What OMIP requires** (Sect. 2.2 "Forcing"):
> Bulk formulae for computing turbulent fluxes for heat and momentum
> must follow Large and Yeager (2009).

Not COARE. **Large & Yeager 2009 specifically** — they hand-tuned
their drag/heat-transfer coefficients to balance against the
JRA55-do/CORE-II forcing fields. Substituting COARE breaks the
intended energy balance and is a known source of inter-model spread
in CORE-II (Tsujino et al. 2020 GMD discusses this).

LY04 ≠ Large & Yeager 2009. Confirm which is implemented. If LY04
is the older (2004) version, it needs updating. If `bulk_formulas.py`
is already LY09, that's fine. **Action**: audit `bulk_formulas.py`
and `bulk_flux.py` for the LY09 coefficient values
(`C_d = 1e-3 * (2.7/U + 0.142 + U/13.09 − 3.14807e-10*U⁶)`,
neutral 10m), and verify which set is the *default* path in the
forced-ocean integration.

The Large-Yeager "corrected" forcing files (Sect. 2.1: "groups should
make use of the 'corrected' forcing data set... these files
incorporate modifications from Large and Yeager (2009) aiming to
address biases in the reanalysis product") are the right input —
the raw JRA55-do or ERA5 fields are *not* protocol-compliant on their
own. The plan needs to specify "corrected" forcing.

**Effort**: 1–2 days to audit the bulk-formula coefficients;
1–2 days to integrate the LY09 correction tables into the
forcing pipeline if not already there.

### Gap 4 — SSS restoring is not in the plan (severity: MEDIUM-HIGH)

**What's there**: The plan does not mention SSS restoring at all.

**What OMIP requires** (Sect. 2.2 + 2.3):
> Surface ocean salinity is damped to a monthly observational-based
> climatology... Details of the damping timescale are not specified
> by the protocol.

Translation: every OMIP submission has SSS restoring with a chosen
piston velocity. Common values:
- GFDL OM4: 50 days over the upper 50 m → piston velocity ~1.16e-5 m/s
  (Adcroft et al. 2019).
- NCAR POP/CESM: 4-yr timescale over 50 m → 4e-7 m/s
  (Danabasoglu et al. 2014, lighter).
- The user's "~50 W/m²/K equivalent" is approximately the NEMO ORCA
  setting (~5e-7 m/s). The corresponding salt restoring is
  `dS/dt = -(S - S_obs)/τ` with τ chosen for the chosen piston
  velocity given a given mixed-layer depth.

**Action**: Add SSS restoring to the plan. Recommendation:
- Choose piston velocity ~5e-7 m/s (matches NEMO/ORCA standard, on
  the lighter end — protocol explicitly says "weak salinity
  restoring is generally preferred").
- Damp to PHC3.0 or WOA monthly SSS climatology.
- Apply globally including under sea ice (the protocol allows this;
  Danabasoglu 2014 App. C reviews choices).
- Diagnostic: report `wfcorr` / `vsfcorr` (the restoring flux as a
  separate diagnostic — Tables K1, K2). Without this, the salt
  budget doesn't close.

**Effort**: ~3 days to implement SSS restoring (a relaxation term in
the surface boundary condition), wire `wfcorr` diagnostic, set up
the climatology load. Most pieces are already in
`ocean/physics/surface_forcing/restoring.py` per the file listing.

### Gap 5 — Initial conditions: WOA18 vs WOA13v2 (severity: LOW)

**What's there**: "WOA18" in the plan.

**What OMIP-1 specifies**: WOA13v2 (Sect. 2.2). WOA18 is newer and
better but a *different* dataset. Using WOA18 means you're not
running OMIP-1 — you're running an OMIP-1-like experiment with
WOA18 init.

**Recommendation**: This is a defensible deviation if documented.
WOA18 is what the post-2018 community uses. **Action**: in the plan
write "Initial T,S from WOA18 — note OMIP-1 specifies WOA13v2; we use
WOA18 because it is the modern equivalent and our results are
intended to be comparable to post-2018 OMIP-style work, not to
pre-2014 CMIP5". Don't pretend it's compliance.

Same for resolution: OMIP supplies WOA on 1° and ¼° spherical grids;
we want to interpolate WOA18 0.25° native onto our model 1° grid
conservatively (not bilinearly — the protocol is explicit about
conservation, App. A3.2).

**Effort**: ~1 day to write the WOA18 → model grid conservative
interpolator (probably 80% reuse of the ETOPO conservative remap
already in `grids/topography.py`).

### Gap 6 — River runoff (severity: MEDIUM)

**What's there**: "Bundled in JRA55-do; we plan to defer." Plan §
"Nice to have".

**What OMIP requires** (Sect. 2.2):
> Surface water fluxes are provided by Large and Yeager (2009) for
> precipitation and Dai and Trenberth (2002) for interannual river
> runoff.

Runoff is mandatory. Not optional. JRA55-do v1.4+ bundles a
modified Dai & Trenberth runoff dataset that is the protocol input.

The "tolerable to defer" question is real, though. At 1° on a model
without explicit river routing:
- If you skip runoff entirely: AMOC is too strong (no Mississippi/St.
  Lawrence freshening of the western N. Atlantic), Bay of Bengal is
  too salty, Amazon plume absent — affects tropical Atlantic SST/SSS.
  Quantitatively: the global runoff is ~37 mSv = ~1.2e6 m³/s. Spread
  over the whole ocean, that's 1.2e6 / 3.6e14 = 3.3e-9 m/s = ~10 cm/yr,
  small. Locally at river mouths (Amazon ~0.21 mSv) it's ~10⁻⁶ m/s,
  large.
- Quick fix: apply runoff as uniform coastal freshwater within ~3
  cells of any coast where the JRA55-do file has nonzero river
  runoff in that latitude band, weighted by the file's lat-binned
  total. This is not Dai & Trenberth's spatial pattern but it
  conserves the global mass flux and avoids the worst Mississippi/
  Amazon biases.

**Recommendation**: Add the quick-fix runoff (uniform-coastal-by-band)
to the pilot. Defer full river-routing infrastructure to phase B.
Document that the runoff *spatial pattern* is approximate.

**Effort**: ~3 days for quick fix, weeks for proper river routing.

### Gap 7 — Forcing cadence: 6-hourly vs 3-hourly (severity: LOW-MEDIUM)

**What's there**: "Time coordinates: 6-hourly, noleap calendar" in
`omip_1deg_plan.md` Phase 2.

**What protocol allows**: CORE-II is 6-hourly. JRA55-do is *available*
at 3-hourly *for winds* (others at 6-hourly). Tsujino et al. (2018,
2020) document that 3-hourly winds are the recommended product
because turbulent flux variance scales nonlinearly with |U_10|, so
6-hourly winds underestimate fluxes in storm tracks by ~5–10%.

At 1° non-eddying resolution with no explicit storm-track resolution
in the *ocean*, the practical impact is small. Going from 6-hourly to
3-hourly costs ~2× I/O and ~1.05× SST bias improvement. **Default to
3-hourly winds, 6-hourly other fields**, which is the JRA55-do v1.4
distribution as published.

**Effort**: zero if the data loader is just "read this Zarr file" —
the file already has the right cadence. Maybe 1 day of pipeline
adjustment.

### Gap 8 — Real-date calendar for interannual forcing (severity: MEDIUM)

**What's there**: "Calendar (noleap only)" listed as blocking-for-
real-dates, small effort.

**Implication**: this *is* blocking for OMIP-1/2. JRA55-do uses
real (Gregorian) dates with leap years 1958–2018. Cycling forcing
with year wraparound + skipping Feb 29 changes the diurnal/seasonal
phase by 1 day per cycle. Over 5 cycles that's 5 days of phase drift —
small for AMOC, larger for tropical seasonal cycle.

The escape hatch: pre-process JRA55-do to noleap (drop Feb 29 from
all leap years) once. The CORE-II forcing files were distributed in
this form. JRA55-do was not but the conversion is a 1-line `xarray`
operation.

**Recommendation**: pre-process JRA55-do to noleap in
`prepare_omip_forcing.py` (Phase 2). The tracking calendar is then
just (year, day-of-year) with day ≤ 365. Document this in the run
README.

**Effort**: 1 day in the prep script.

### Gap 9 — Diagnostics on density-space MOC (severity: MEDIUM)

`msftmrho` (overturning in σ-2000 density space, by basin) is OMIP
Priority 1 (Table I6). It is the *only* diagnostic that distinguishes
"AMOC" from "spurious diapycnal-mixing-driven overturning". Without
it, you can't argue your AMOC is real.

**Action**: design the online σ-2000 binning into the diagnostic
module from day 1. Bin edges: a fixed 80-bin σ-2000 grid covering
24.0–28.5 kg/m³, 0.05 kg/m³ resolution (Griffies 2014 standard).
Online: each timestep, for each (lat, basin), accumulate `v * h * dx`
into the σ-bin selected by the local σ-2000. Monthly mean of this
2D (lat, σ) field = `msftmrho`.

**Effort**: ~3 days to design + implement + test online. Trivial to
add up-front, painful to retrofit.

---

## 5. Updated phase plan (surgical edits to the existing 5 phases)

The five phases in `omip_1deg_plan.md` are sound. Edits, not rewrites:

### Phase 1 — Forced Ocean Driver (unchanged scope, +diagnostics)

Keep as written. Add:
- The driver carries an explicit `OmipDiagnostics` accumulator state
  (online accumulators for σ-2000 MOC bins, basin masks, ideal-age
  tracer, MLD, strait-transport line-integrators).
- Driver decides *now* whether the run is OMIP-2 (JRA55-do, 1958–2018)
  or OMIP-1 (CORE-II, 1948–2009). Recommend OMIP-2.

### Phase 2 — Forcing Data Preparation (+ LY09 corrections, runoff, calendar)

- Use **JRA55-do v1.4+ "corrected" files** specifically — not raw ERA5.
  Adopt their 3-hourly winds, 6-hourly radiation/T/q/P/runoff cadence.
- Add: river runoff field (already in JRA55-do; do not skip).
- Pre-process to noleap calendar (drop Feb 29).
- Verify Large & Yeager (2009) bulk-formula coefficients match what's
  in `coupler/bulk_flux.py` and `bulk_formulas.py` — audit task.

### Phase 3 — Initialization (+ WOA18 conservative remap, basin masks)

- Initial T,S from **WOA18** (document deviation from WOA13v2).
- Conservative regrid (reuse the ETOPO machinery).
- Build **basin masks** (Atlantic-Arctic, Indian-Pacific, Global) at
  init time — needed for online basin-partitioned diagnostics. Use
  the standard NCAR/CICE basin-mask convention.
- Build **strait-section paths** for the 16 OMIP straits at init time
  — pre-compute the (i,j) zig-zag path along native grid lines for
  each named strait per Sect. C4. This is a one-time setup. Static
  arrays, no AD pain, JIT-clean.

### **NEW Phase 3.5 — Polar grid stability hardening** (~3 days)

Adopt before any forced run:
1. Confirm ≥80°N polar cap is locked (u=v=0 inside cap, T,S relaxed
   to climatology with τ ≤ 30 d).
2. Add zonal Fourier filter on tendencies poleward of 78°N. Validate
   with the Phase 4(c) realistic-geometry baseline (should be
   bit-comparable south of 60°N).
3. Document the Arctic as untrustworthy in the diagnostic plan;
   exclude from extent/thickness comparison plots if Phase 3.5b
   (sea ice) is not done yet.

### Phase 4 — Ocean Diagnostics (massive expansion)

This is where the plan most underestimates. Restructure as:

4a. **Online accumulators** (5–7 days):
- σ-2000 MOC binning by basin
- MLD using Levitus σ-t criterion (ΔB_crit = 0.0003 m/s²)
- mlotstmax / mlotstmin per month (max/min over month)
- Ideal age tracer (one passive tracer)
- Heat and salt budget terms (advective + each parameterized
  contribution separately — Tables L1–L3)

4b. **Strait transports** (3 days):
- 16 strait section paths in `OmipDiagnostics` static arrays
- Online accumulation of mass / heat / salt transport per strait
- Monthly mean output

4c. **Boundary-flux diagnostics** (2 days):
- Wire `tauuo, tauvo, hfds, hfsifrazil, hfsnthermds, ficeberg=0,
  fsitherm, sfdsi, wfo, wfonocorr, wfcorr, vsfcorr`
- These are mostly already-computed quantities; just need archival.

4d. **Output infrastructure** (3 days):
- Native-grid + 1° spherical-grid output (conservative regrid stage,
  offline, post-process).
- One netCDF4 file per year for monthly fields, one per cycle for
  decadal-mean fields.
- CMOR-name compliance for the named diagnostics (so post-OMIP
  analysis tools work out of the box).

### Phase 5 — Validation (expanded against named OMIP analysis papers)

Compare against Danabasoglu 2014 (NA mean), Griffies 2014 (sea level),
Downes 2015 (SO water masses), Farneti 2015 (ACC + SO MOC), Wang
2016a,b (Arctic ice + freshwater), Ilicak 2016 (Arctic hydrography).

Specifically required validation:
- **AMOC**: time series at 26.5°N (RAPID), 45°N, 35°S; check both
  strength (CORE-II ensemble: 16–22 Sv) and depth-density structure.
- **Drake**: 130–170 Sv (Meredith 2011 reference value 136.7 ± 6.9 Sv).
- **MHT**: northward heat transport at 26.5°N, peaks ~1.2 PW (RAPID
  + Trenberth-Caron).
- **Sea ice extent and thickness**: NSIDC + PIOMAS.
- **MLD**: de Boyer Montégut 2004 climatology.
- **σ-2000 MOC**: AMOC@26.5N in density space, separate measure of
  AABW vs NADW pathways.

### **NEW Phase 6 — OMIP cycles 2–5** (~10 weeks wall, 1 week dev)

- After cycle 1 validates, run cycles 2–5 from cycle-1 final state.
- Restart at end of each cycle.
- Decadal-mean diagnostics archived per cycle.
- Drift quantification (Sect. 3.3): compute T,S,SSH drift per cycle,
  plot drift-decade for cycles 1–5.

### Sea-ice phasing (cross-cutting — see §3 above)

- Phase A (no ice, 60°S–60°N domain): runs Phase 1–5 above as written.
- Phase B (with ice, global): adds the validation suite from §3
  before starting global submission.

---

## 6. What could derail the pilot — risk register (ranked)

| # | Risk | Severity | Mechanism | Mitigation |
|---|---|---|---|---|
| 1 | **AMOC collapses or runs away in cycle 1** | High | Without sea-ice freshwater + with naive SSS restoring, the salt-advection feedback (Stommel 1961) can flip AMOC mode. Phase 4(c) realistic-geometry already showed MOC = 180 Sv with SST restoring alone. | Use Phase A (60°S–60°N closed) for first numbers. If global with ice, ensure SSS restoring is applied under ice and pin Labrador/GIN sea SSS to avoid flipping. |
| 2 | **Drake transport overshoots (200+ Sv)** | High | Idealised wind in spinup, Drake 1-cell wide at 1° in some places, missing form drag. Realistic JRA55-do τ helps but doesn't fix bottom form drag. | Add bottom form drag as an explicit term (Munk-Palmen 1951 derivation; see Marshall & Radko 2003); document the residual. Ramp wind from 0 → JRA55-do over year 1. |
| 3 | **Spurious deep convection in N. Pacific** | High | Without ice, the Bering/Okhotsk Seas in winter cool below T_freeze and the model convects there → spurious NPDW. | Cap surface T at T_freeze in cells north of 50°N when the prescribed ice climatology has ice present (the "freeze-T cap" hack). Explicitly diagnose Bering/Okhotsk MLD. |
| 4 | **Spin-up too short to equilibrate intermediate water** | Medium-High | 1 cycle = 62 yr is barely enough for AAIW, not enough for NADW lower limb (decadal-centennial), nowhere near AABW (millennial). | Plan for 5 cycles (310 yr) per protocol. Report cycle-1 numbers as preliminary; final numbers from cycle 5. |
| 5 | **Diagnostic retrofitting cost** | Medium-High | Adding `msftmrho`, ideal age, heat-budget terms after the run is done means re-running. | Design phase 4 (online diagnostics) *before* the pilot. Budget 2 weeks of dev for diagnostic infrastructure. |
| 6 | **JRA55-do data volume + I/O bandwidth** | Medium | 60 yr × 6 fields × 3-hourly winds + 6-hourly others, regridded to 1° = ~30 GB. Cycling 5× means same data read 5×. | Pre-stage to local Zarr; use lazy chunked reads; budget I/O per timestep. The current `forcing/external.py` should handle it but verify under the actual pilot setup. |
| 7 | **GM/Redi κ tuning** | Medium | At 1° non-eddying, GM κ = 600–1000 m²/s. Phase 4(c) used 800. CORE-II ensemble shows AMOC sensitivity to κ_GM at the ±20% level. | Stick with Visbeck adaptive (already in tree). Document the chosen baseline κ_GM. |
| 8 | **Bering Strait under-resolved at 1°** | Medium | At 1°, Bering Strait is ~1 cell. Closing it removes the Pacific→Arctic freshwater transport (~0.8 Sv climatologically, large for Arctic). | Either keep open with documented under-resolution, or close and document the freshwater bias. The realistic-geometry plan's `topology_fixes.md` already enumerates fixes — apply consistently. |
| 9 | **Differentiability of full pipeline through ice** | Medium | The pilot probably doesn't need `jax.grad` end-to-end, but the project goal does. EVP subcycling, KPP root-find, and bulk flux iteration can each break AD. | Phase 5 of legoESM's general workflow already includes Taylor tests on the coupled system (`test_diff_coupled_system.py`). Run them on the OMIP config before claiming AD compatibility. |
| 10 | **Calendar drift** | Low-Medium | If we don't pre-process to noleap, 5 cycles of cycling-with-leap-years drifts seasonal phase by 5 days. | Pre-process JRA55-do to noleap in Phase 2. |
| 11 | **WOA18 deviation from protocol** | Low | Initial conditions different from protocol. Model results not strictly comparable to OMIP-1 ensemble. | Document; if needed, also run a WOA13v2 sensitivity test. |
| 12 | **Wright EOS vs TEOS-10** | Low | Protocol Sect. 2.2 says "Conversion to Conservative Temperature and Absolute Salinity should be made for models based on IOC et al. (2010)". We use Wright (potential T, practical S). The OMIP archive accepts both. | Document; convert to CT/SA at the post-process stage if needed for cross-model comparison. |

The top 3 (AMOC stability, Drake overshoot, spurious N. Pacific
convection) are the ones that can sink the pilot. They are all
predictable from the literature and there are concrete mitigations.

---

## 7. Bottom line

The current `omip_1deg_plan.md` is a **good forced-ocean validation
plan** but **not a protocol-compliant OMIP submission plan**. The
gap is mostly diagnostics (large gap, cheap to fix if designed in)
and spin-up length (factor-of-5 gap, expensive). Polar grid is fine
as is for the pilot. Sea ice is the structural blocker for full
OMIP — phase the experiment around it.

The right sequencing:

1. Add the gap-1, gap-2, gap-3, gap-4 fixes to the plan (~2 weeks).
2. Add Phase 3.5 (polar stability hardening) (~3 days).
3. Run Phase A (tropical-OMIP, 60°S–60°N, 1 cycle) as the technical
   shakedown.
4. In parallel, validate ice for Phase B.
5. Run Phase B (full global OMIP-2, 5 cycles) for the publication.

This gets a defensible result on the timescale of months, not the
months-to-years that "tripole + ice + OMIP all at once" implies.
