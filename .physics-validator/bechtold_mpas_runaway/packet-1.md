# Adversarial review packet — Bechtold-on-MPAS thermal runaway (round 1)

You are an **independent adversarial physics reviewer**. Find every bug, sign
error, unit inconsistency, broken-gradient pattern, conservation violation,
experiment-design confound, and mis-attribution in the analysis below. Cite
file:line. Where you agree, say why the candidate concern is/ isn't a bug.
Focus especially on **refuting or confirming my revised root-cause hypothesis**
and on **what single discriminating test decides it**.

Repo: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM (branch mpas-stability-campaign).
This is a JAX-native differentiable ESM. Convection scheme = Bechtold/IFS.

## Observed failure (evidence, all from one day's runs)
MPAS AMIP, SCVT level-5 (~2.2°), L30 sigma, dt=75 s, fresh ERA5 IC, PCMDI SST.
Two runs from the SAME working tree, identical except flags:
- **climeval_pilot2** (`--convection sbm`, GWD mcfarlane, hard-sat cap 5 K,
  1979-pinned radiation deck): HEALTHY at day 5+ (global-mean T flat 251.4 K).
- **probe_bech40** (`--convection bechtold --convective-precip-efficiency 0.8`,
  GWD `mcfarlane+hines`, `--hard-sat-max-heating-k 10`, transient 1979-2014
  ozone/aerosol/volcanic deck): **THERMAL RUNAWAY** — global-mean T
  252.7→287.6 K over days 1→11 (≈3.3 K/day ≈ 368 W/m² sustained column
  heating); T_max 302→377 K; |u|_max peaked 107 m/s day 6 then fell; CWV only
  23→28. Day-10 anatomy: global-mean T(level) INVERTS — level 13/30 = 311.5 K
  vs surface level 29 = 288.4 K (mid-troposphere heat bulge); hottest columns
  372 K, **DRY** (col-max q_v 5-7 g/kg), clustered 39-43°N,171-178°E (N Pacific
  storm track). **ZERO** hard-saturation-drain interception lines in the log
  (SBM runs show ~1000 pts/check).

## Original claims under test
1. Runaway caused by the parameterization pivot, most plausibly **bechtold on
   MPAS** ("convective heating without commensurate moisture sink").
2. A 5-arm single-delta denial experiment (base = climeval_pilot2 verbatim +
   ONE delta each): (a) den_bech = bechtold+pe0.8; (b) den_bechraw = bechtold,
   pe default 0.7; (c) den_gwd = mcfarlane+hines; (d) den_cap10 = hard-sat cap
   10; (e) den_deck = transient ozone+aerosol+volcanic deck.
3. Code trace of bechtold on the standalone MPAS path (energy/moisture
   inconsistency, dropped conv_prog_profile carry, double-count, sign flip).
4. climeval_pilot2's resolved `convective_precip_efficiency: 0.0` is a
   serialization artifact, not a real 0.0 reaching SBM.

## What I found

### NUMERICAL PROBE (the decisive evidence) — bechtold conserves moist enthalpy
Leaf column budget of `bechtold_convection` with probe_bech40's EXACT flags
(pe=0.8, `use_ifs_inplume_precip=True`, all IFS flags on, M_b_max=0.05,
subsidence_solve=implicit_flux, dt=75), spun 30-60 steps with the M_u carry fed
back. Measured R = ∫(c_pd·dT + L_v·dq_v) dp/g [W/m²] (must be 0 if column moist
enthalpy is conserved) and total-water residual ∫(dq_v+dq_c+dq_r)dp/g:

```
sounding            nlev  R_enthalpy   heat        water     CAPE   max dT/dt @ level
tropical Ts=302      20   -0.00 W/m2   96.2 W/m2   0.0 mm/d   -     -
pe0.0 legacy split   20   -0.00 W/m2  207.2 W/m2   0.0 mm/d   -     -
all-IFS-off          20   -0.00 W/m2  292.4 W/m2   0.0 mm/d   -     -
storm-track Ts=288   30   -0.00 W/m2  420.4 W/m2   0.0 mm/d  2644   16.9 K/d @ L26 (902 hPa)
cold Ts=283          30   +0.00 W/m2  246.2 W/m2   0.0 mm/d  2781    9.3 K/d @ L27 (934 hPa)
warm Ts=295          30   -0.00 W/m2  401.2 W/m2   0.0 mm/d  3660   27.2 K/d @ L4  (181 hPa)
```
**R = 0 to machine precision in every case; total water closes to 0 mm/day.**
So the scheme does NOT fabricate column energy and its heating IS paired with a
commensurate vapor sink. **Claim 1's mechanism ("heating without moisture
sink") is REFUTED at the leaf level.** BUT the heating is LARGE (250-420 W/m²
on CAPE>2500 columns) and M_u saturates at the cap (Mu_max=0.0457≈M_b_max=0.05),
CAPE stays 2600-3700 after 60 steps. This is *conservative but excessively
vigorous* convection.

### F1 [HIGH, CONFIRMED] — the bechtold leaf-conservation gate is DEAD
`tests/unit/test_bechtold_column_conservation.py:47` constructs
`BechtoldConfig(cape_sink_heating_ratio=5.0, ...)`. That field was REMOVED from
`BechtoldConfig` (grep: `cape_sink_heating_ratio` appears ONLY in this test,
nowhere in `packages/`). The module errors at **collection**:
`TypeError: BechtoldConfig.__new__() got an unexpected keyword argument
'cape_sink_heating_ratio'`. So the two tests that pin exactly this runaway's
failure mode — `test_production_column_enthalpy_closes` (R≈0, atol 3 W/m²) and
`test_production_column_water_closes` — DO NOT RUN in CI on this tree. The
safety net for a convective heat/moisture mispairing is silently disabled by
config-field drift. (My probe re-implements it → currently PASSES, R=0, so
re-enabling is hygiene, not the bug — but nothing guards a future regression.)

### F2 [MED, CONFIRMED] — precip_efficiency is INERT under use_ifs_inplume_precip=True
With the default `use_ifs_inplume_precip=True`, rain forms from the IFS analytic
in-plume conversion (`bechtold.py:2815-2827`,
`_ifs_convective_precip_conversion` at `bechtold.py:1041-1160`), and the config
doc states "both precip_split_scheme variants are bypassed when on". The
`precip_efficiency` constant split (`bechtold.py:2871-2873`) is only reached in
the `elif` (inplume OFF). So `--convective-precip-efficiency 0.8` DOES NOT reach
rain formation in probe_bech40. Consequence for the denial experiment (claim 2):
**den_bech (pe0.8) and den_bechraw (pe0.7) are byte-identical runs** — both use
IFS in-plume precip, pe ignored. The two arms do not discriminate anything. To
actually test a pe effect you must set `use_ifs_inplume_precip=False`.

### F3 [design, CONFIRMED] — den_bech's "+pe0.8" delta is a no-op both sides
Base = SBM, and `SBMConfig` has no `precip_efficiency` field
(`config.py:316`, SBM resolver `physics_pipeline.py:2928-2933` reads only
sbm_tau_c/sbm_RH_ref/sbm_cape_threshold). So pe is inert for the base AND (F2)
for bechtold-inplume. The genuine single delta of den_bech is sbm→bechtold. The
label "+pe 0.8" is misleading; fine for attribution but not for isolating pe.

### F4 [claim 4, CONFIRMED artifact] — SBM cannot see convective_precip_efficiency
Same as F3: `SBMConfig` lacks the field; the SBM branch never reads
`convective_precip_efficiency`. Whether the resolved json shows 0.0 or None is
physics-irrelevant to SBM. Confirmed harmless artifact. (Origin: the coupled
builder passes it through as-is, `config.py:2510`; the 0.0 is a serialize-time
coercion of the None default, not a value reaching the scheme.)

### F5 [REFUTES a task hypothesis] — conv_prog_profile IS threaded across steps
The task worried the MPAS loop resets the bechtold carry to zeros each step.
It does not. `model_driver.py:6196` seeds `_phys_state = init_physics_state(...)`;
the serial loop threads it `self.state = self.model.step(..., phys_state=_phys_state);
_phys_state = self.model._phys_state` (`model_driver.py:6514-6517`), the MPI loop
via `_mstep` return (`6510-6511`); checkpoint restore at `6224-6357`. The carry
`conv_prog_profile` is the relaxed updraft mass-flux M_u (units kg/m²/s,
`bechtold.py:2697-2702`, tau_M_u_relax=1800 s), zeros only at cold start.
`combined.py:437-533` merges the returned dict into phys_updates. So the carry
is genuinely persistent; **not the runaway cause.**

### F6 [PLAUSIBLE — the leading positive candidate] — recurring M_u-plateau vigor
`bechtold.py:2703-2750` documents a RECURRING "spurious-heating runaway"
(quiescent column heated ~3900 W/m²) from M_u exponential growth pinning the
constant M_b_max clip and ERASING the launch cape_weight gate, progressively
patched (cape_weight²·M_b_max cap at :2732; p_conv_top gate at :2749; the
comment cites day-15→40→65 blowups each only DELAYED). Crucially the cap is
**byte-identical for cape_weight≈1 (genuinely-convecting) columns**, so it does
NOTHING for the storm-track columns that DO carry CAPE. My probe confirms:
CAPE=2644-3660 → M_u pins at 0.0457≈M_b_max=0.05 → 250-420 W/m² of conservative
mid/upper-trop heating, sustained. probe_bech40 (storm-track 39-43°N, mid-trop
bulge) looks like the next instance of this mode.

### F7 [likely FALSE POSITIVE, but the guard is dead] — anvil/rain double book
In-plume path detrains the FULL unconverted plume q_c_u as anvil
(`bechtold.py:2782`, heated +L_v at :2805, sunk by the kernel's −dq_c at
`mass_flux.py:618-619`) AND separately forms rain from precip_frac
(:2824, sunk mass-weighted at :2842-2851, heated +L_v at :2867) — the
"one-pass replay ... O(0.1-0.3 K) approximation" (config doc). Each term pairs
its own +L_v heating and −L_v sink, so column moist enthalpy is conserved (probe
R=0 confirms). Not a conservation bug on the fixtures tested, but the leaf test
that would guard it across soundings/resolutions is DEAD (F1).

### F8 [MINOR, PLAUSIBLE] — hines heating is bounded KE→heat
`hines.py:120-136`: dT_dt = -(u·du_dt+v·dv_dt)/c_pd, heating = dissipated
resolved KE (eps_gwd≥0), capped by Fmax + tendency limiter; `conserves:none`
(launched wave is an unbudgeted source). It amplifies the runaway (converts the
thermal-wind-driven strong winds back to heat) but cannot be the primary
368 W/m² global source. den_gwd tests it.

### F9 [MED, design] — single-delta arms can't catch an INTERACTION
probe_bech40 stacked bechtold + transient deck (incl. volcanic LW aerosol,
`aerosol_lw_od` at `model_driver.py:6479-6485`) + cap10. If NO single arm
reproduces the runaway, suspect an interaction (bechtold's dry hot mid-trop ×
volcanic-LW trapping). Add a combined bechtold×transient-deck arm. Also den_deck
BUNDLES 3 files (ozone+aerosol+volcanic) — cannot isolate which.

### F10 [CONFIRMED] — cap10 is inert for this runaway
The post-step hard-sat drain (`model_driver.py:6539-6558`) fires only on
SUPERSATURATION and requires the q_c tracer present; the runaway columns are DRY
(subsaturated) → zero interceptions (matches log). `--hard-sat-max-heating-k 10`
only limits heating WHEN the drain fires, and it never fired. den_cap10 will
correctly show no effect; cap10 is a red herring.

## My revised root-cause hypothesis (attack this)
The runaway is **NOT a bechtold leaf conservation/sign bug** (probe: R=0 across
soundings and resolutions; water closes). Instead: on the coarse MPAS SCVT-L30
grid, bechtold is moist-enthalpy-conserving but **excessively vigorous on
high-CAPE storm-track columns** — M_u saturates at M_b_max, depositing 250-420
W/m² of mid/upper-trop heating (the mid-trop bulge = detrainment-level heating).
Because the heating is conservative it dries as it heats (hard-sat can't catch
dry air → zero interceptions); the fixed-SST surface re-evaporates, re-supplying
CAPE, and the dried columns' collapsed LW cooling lets the heat accumulate.
Bechtold SHAPES the runaway; the ENERGY comes through surface fluxes + radiation.

- Labeled **CONFIRMED**: bechtold conserves moist enthalpy (not a fabrication/sign
  bug); the F1 conservation gate is dead; F2/F3/F4/F5/F10 as above.
- Labeled **PLAUSIBLE (inferred)**: the positive mechanism (conservative-vigor +
  surface/radiation feedback shaped by bechtold) vs the alternative that the
  transient volcanic-LW deck supplies the energy and bechtold merely shapes it.

**Single most likely root-cause candidate:** convective VIGOR (F6) — M_u pinned
at M_b_max on CAPE>2500 columns with a CAPE closure that doesn't drain CAPE, not
a sign/conservation defect. **Discriminating test:** run **den_bech** (bechtold,
base 1979-pinned deck, base GWD, base cap) — everything else = healthy
climeval_pilot2. If it reproduces the mid-trop-bulge runaway → bechtold vigor
confirmed and the fix is tuning (M_b_max, CAPE-consumption timescale, coarse-grid
entrainment / p_conv_top), NOT a conservation fix. If den_bech is HEALTHY but
den_deck runs away → the transient deck (volcanic LW) is the energy source.

## Your task
Adversarially review ALL of the above. Specifically:
- Is the R=0 leaf-conservation evidence sufficient to refute "convective heating
  without moisture sink", or is there a path (inputs the leaf probe left None:
  omega, shf/lhf, dT_dt_rad, land_frac; or nlev/grid effects) where bechtold
  DOES break the budget on MPAS? Name the exact line if so.
- Is F6 (M_u plateau vigor) really the leading candidate, or am I missing a
  larger direct source (radiation deck sign, a coupling flux sign, the kernel
  subsidence sign at `mass_flux.py:559-620`, the +L_v pairings at
  `bechtold.py:2805/2867`)?
- Find any remaining confound in the denial design beyond F2/F3/F9.
- Do you agree den_bech is the decisive first arm? Propose a better one if not.
