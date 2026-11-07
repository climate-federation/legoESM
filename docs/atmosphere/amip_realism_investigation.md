# AMIP full-physics realism investigation

Status: **root cause identified; a validated realistic config already exists.**
This note consolidates a long diagnostic campaign on why a full-physics,
prescribed-SST (AMIP) run with `scripts/run/run_amip.py` produces an unrealistic
climate, and what is (and is not) the fix.

## TL;DR — the fix is SBM convection (already validated)

The diagnostic campaign below used **`--convection tiedtke`** and hit a
non-precipitating overcast trap. **`config/amip/amip_production.yaml` already
ships the realistic config** — the key difference is **`convection: sbm`**,
validated at C48/L40 (job 25918469): **planetary albedo 0.292** (target 0.29),
**precip 3.2 mm/day** (target 2.8). Its provenance note states SBM "fixes the
structural over-bright (~2×) and too-dry (~4×) biases" — i.e. **exactly the
symptoms this campaign chased under tiedtke**. So the weak, imbalanced
hydrological cycle diagnosed below is a **tiedtke** deficiency; SBM resolves it.

**To restart / reproduce a realistic AMIP:**
- Production: `run_amip.py --config config/amip/amip_production.yaml ...` (C48/L40).
- Cheap iteration: `run_amip.py --config config/amip/amip_realism_c12.yaml
  --forcing-path forcing_amip_woa/sst_sic_amip_1979-2014.nc --days 45 ...`
  (C12/L20, SBM, WOA observed SST, realistic dry IC).

The rest of this note documents the tiedtke-path diagnosis (still useful — it
localises WHY tiedtke fails and rules out the non-convection levers).

## Symptom

A full-physics AMIP run (WOA-derived observed SST, `tiedtke + sundqvist/morrison +
louis + mcfarlane + xu_randall`, RRTMGP radiation) settles into a **cold, overcast,
non-precipitating** state:

| metric | AMIP run | Earth |
|---|---|---|
| planetary albedo | 0.70–0.82 | 0.29 |
| OLR | 143–190 | 240 |
| R_TOA | −66 to −158 W/m² | ~0 |
| precip | 0.3–0.9 mm/day | 2.7 |
| CWV | 62–73 kg/m² | ~25 |
| hfls (evap) | 4–33 W/m² | ~88 |

## Root cause (proven, not guessed)

A **self-sustaining cold/overcast/non-precipitating trap under realistic
radiation**, driven by a **weak, imbalanced hydrological cycle**:

1. The column moistens because evaporation exceeds precipitation (E ≈ 1.1 >
   P ≈ 0.82 mm/day), and both are ~3× too weak (Earth E = P ≈ 2.7).
2. As column water vapour rises, relative humidity approaches saturation and the
   diagnostic cloud **fraction → 1** (overcast) across the column.
3. The `cf × q_c_diagnostic` radiative-condensate floor (needed so coarse
   grid-mean condensate is not optically inert) then makes an overcast column
   **opaque** → planetary albedo 0.7–0.82.
4. Over-reflection starves the surface of shortwave → the column cools → less
   saturation vapour → the cycle locks in.

A realistic clear/warm state **does exist**: initialising dry (`--rh-init 0.25`,
CWV ≈ 30) starts the run at **albedo 0.255, OLR 239, evaporation restored** — but
it re-moistens back to overcast because E > P.

## What is NOT the fix (ruled out by controlled experiments)

- **Surface flux** (`--surface-bulk-scheme coare3`): the trapped column is
  saturated at the surface, so there is no humidity gradient to evaporate into.
- **Microphysics** (`morrison` mixed-phase): a clean 2×2 (micro × radiation)
  showed the trap is **radiation-driven, not micro-driven** — both sundqvist and
  morrison are trapped under RRTMGP; only *gray* radiation (which ignores clouds)
  escapes.
- **Cloud optical depth** (`--q-c-diagnostic`): lowers albedo only 0.82 → 0.70,
  validation-floored at 5e-5; insufficient.
- **Cloud-fraction scheme** (`sundqvist` vs `xu_randall`): both overcast.
- **Cloud-fraction sensitivity** (`--cloud-p-xr` / `--cloud-alpha-xr`): affects
  only moderate RH; as the column moistens toward saturation `cf → 1` regardless.
- **Initial moisture** (`--rh-init`): starts realistic but re-moistens (E > P).

## What IS the fix

**Use SBM convection** (`convection: sbm`) instead of tiedtke. SBM produces the
strong, balanced hydrological cycle the tiedtke path lacks and is validated
realistic in `config/amip/amip_production.yaml` (albedo 0.292, precip 3.2). The
per-knob analysis above explains *why* the tiedtke path fails (weak/imbalanced
P<E → moistening → overcast) and confirms the failure is the **convective
closure**, not the surface / cloud / IC levers — all of which were ruled out.

Open follow-up: confirm SBM also lands realistic at coarse C12/L20 (the cheap
iteration grid) with the WOA observed SST — `config/amip/amip_realism_c12.yaml`
is the config for that check. (C12/L20 SBM smoke, job 8676505: CWV **37.9** — NOT
the 85 overcast trap — so SBM changes the moisture regime at C12 too, but C12/L20
is too coarse for full radiative balance; publication realism needs C48/L40.)

## Refinement: convective_cloud OFF for prescribed-SST AMIP

A fresh **current-code** re-run of `amip_production.yaml` (C48/L40, 10-day, this
repo's HEAD after PRs #685–707; job 8676675) confirms SBM still lands in the
realistic regime — R_TOA **+5.1 W/m²**, precip **2.73 mm/day**, hfls **90.3**
(Earth ~88), CWV **28** (Earth ~25), T_low 283 K — *not* the tiedtke overcast
trap. But its planetary **albedo was 0.391** (OLR 210), higher than the 0.292
quoted for the original validation.

The cause is documented in `amip_production.yaml` itself: it ships
`convective_cloud: true`, and its matched A/B shows conv-cloud ON degrades the
prescribed-SST TOA budget —

| convective_cloud | planetary albedo | OLR (W/m²) | job |
|---|---|---|---|
| **OFF** | 0.295 | 234.7 | 25929870 |
| **ON**  | 0.370 | 213.3 | 25929869 |
| **ON** (fresh, current code) | 0.391 | 210 | 8676675 |

Under **prescribed** SST the convective-cloud OLR-trapping is an inert
**SST-drift compensator** (its benefit is warming a *free* surface in the coupled
base); with SST pinned only its radiative cost — extra albedo — remains. Our fresh
run (0.391 / 210) reproduces that ON-degraded state.

**So the AMIP-optimal choice is `convective_cloud: false`.** The restart config
`amip_realism_c12.yaml` now sets it OFF, and `--convective-cloud` /
`--no-convective-cloud` (argparse `BooleanOptionalAction`) lets a run override a
config-file default either way — previously a `store_true` flag could not turn OFF
what a `--config` YAML turned ON. `amip_production.yaml` keeps it ON for
coupled-parameter-identity with the tuned slab base (its stated purpose); a
standalone AMIP should pass `--no-convective-cloud`.

## Bugs fixed along the way (merged)

Real defects found and fixed during the investigation:

- **#685** — AIMIP-classical path left `microphysics='none'` (tiedtke detrained
  condensate with no precip sink → CWV runaway); force the trained sundqvist.
- **#686** — WOA18 observed-SST forcing builder (replaces the synthetic
  `cos²(lat)` SST).
- **#687** — full-physics AMIP policy (no parameterization slot `'none'`).
- **#695** — the AIMIP injection **clobbered `--surface-bulk-scheme`** (rebuilt
  `SurfaceLayerConfig` at defaults) → coare3 was a silent no-op on every AIMIP run.
- **#697** — `--sundqvist-{qc-crit,rh-crit,auto-rate}` precip tunables.
- **#704** — `--cloud-p-xr` / `--cloud-alpha-xr` cloud-fraction sensitivity knobs.

## Reproduce

```
scripts/run/run_amip.py --dataset hadisst \
  --forcing-path forcing_amip_woa/sst_sic_amip_1979-2014.nc \
  --grid-type cubed_sphere --resolution 12 --nlev 20 --dt 450 --days 45 \
  --radiation rrtmgp --convection tiedtke --microphysics sundqvist \
  --turbulence louis --gravity-wave-drag mcfarlane --clouds xu_randall \
  --surface-bulk-scheme coare3 --gustiness-zi 300 --rh-init 0.25 --cmip-output
```
Inspect `results/.../timeseries.npz`: `CWV` rises, `albedo` runs away 0.25 → 0.8.

---

## Bechtold marine-BL cloud albedo — config-lever exhaustion + χ no-trade partial fix (2026-07)

Context: the user keeps **bechtold** (not SBM) as the AMIP convection default (SBM "too simple"). With bechtold at C48/L40/dt150 the dominant realism failure is a **cloud/albedo catastrophe** (full CMOR scorecard, area-weighted): `rsut` +119, `clt` +23 (90% overcast), `tas` −2.5, `pr` −2.2, `hfls` −39 → **net TOA ≈ −85 W/m²**. The `tas`/`pr` biases are largely downstream of the albedo; in AMIP (prescribed SST) the ocean `hfls` deficit is a *separate* humid-BL problem.

**Root (measured from checkpoints):** bechtold's excess cloud is **marine boundary-layer LIQUID** (LWP ~190–270 g/m², 3–4× obs; ice ≈ 0 — the "IWP-dominated" anvil is EDMF, a different scheme). The BL is moist *despite* low evap → moisture is trapped, keeping the sub-cloud layer saturated → persistent low cloud → high albedo.

**Config levers are EXHAUSTED — the albedo and evap biases are Pareto-coupled** (each simple knob trades one for the other; defaults are near-optimal for the joint objective):
- **cape_threshold sweep {40,70,110,150}** — non-monotonic; baseline 70 is at the BL-humidity minimum. Both raising and lowering wetten the BL and lower evap (convection is the column-drying agent). Exhausted.
- **cloud-sink stack** (`--subgrid-autoconversion --rh-crit --cloud-rh-crit-bl`) — the BL cloud is **supply-limited**: draining it just makes more rain while it re-condenses; `clt`/`rsut` unchanged. Exhausted.
- **PBL mixing** (`--louis-l-mix-max` 100→200) — strongest LWP cut found (−23%) but an **energy↔water trade**: it homogenizes the BL, eroding BL-top cloud (albedo better) while humidifying the surface (evap −20%). Net-negative; reverted the exposure.
- Radiative knobs (`q_c_diagnostic`, `r_eff`) are **prognostic-overridden** by the M2005 PSD when morrison is on — inert.

**The one NO-TRADE lever: χ = cloud inhomogeneity** (Cahalan 1994, `--cloud-inhomogeneity-factor`). It scales only the *radiative* optical depth (`lwp,iwp *= χ` in `cloud_fraction.py`), not the prognostic cloud water or BL moisture — so it cuts albedo with **no hydrological cost**. Validated χ sweep (matched **warm-start** day30–50, χ the only variable):

| χ   | rsut  | tas    | hfls | pr   | net TOA |
|-----|-------|--------|------|------|---------|
| 1.0 | 222.3 | 284.83 | 50.2 | 0.30 | −62     |
| 0.7 | 212.5 | 284.98 | 51.3 | 0.33 | −57     |
| 0.5 | 200.9 | 285.12 | 52.9 | 0.36 | −52     |
| obs | 99    | 287.5  | 88   | 2.9  | +1      |

Monotonic; **every field moves toward obs, none worsens** (~−7 rsut / +0.07 tas / +0.9 hfls per 0.1 of χ). χ=0.5 is defensible (Cahalan marine-Sc inhomogeneity 0.5–0.7). It is **partial** — τ-saturation caps it (rsut still 201 vs 99); the full albedo fix needs a **structural BL-cloud-water reduction** (model development).

**Stability / deployment:** any χ<1 **blows up the cold-start** (non-finite winds at day ~10 — a spin-up shock, not χ itself; baseline χ=1 is clean). It is fully stable from a **warm-start** (spun-up state). So do **NOT** put χ<1 in a cold-start YAML default. Deploy operationally: **2-phase** run (χ=1 spin-up ~30 days, then warm-start/chain at χ=0.5), or a chain-level χ ramp. An in-model time-ramp would need sim-time threaded through `compute_cloud_properties` (no time arg today) — disproportionate for a partial benefit.

Jobs: `conv_bech_{cape40,albstack,louis200,ctrlws,chi07ws,chi05ws}`. See memory `amip_cloud_albedo_lwp_diagnosis` for the full lever-by-lever trail.

## Config-lever exhaustion + deployable best-config (2026-07-12/13)

Continuing the marine-BL albedo work and pivoting to the #2 bias (under-evaporation, hfls ~49 vs ~88), the full config-lever space was mapped. **Conclusion: every config lever is PARTIAL; the residual biases require BL-scheme model development, not a knob.**

**Marine-BL albedo — measured root.** The radiative marine-BL cloud water is the diagnostic FLOOR `cf·q_c_diagnostic` (~11× the prognostic q_c, dominating ~77% of BL cells), not prognostic condensate — which is why convective precip-efficiency, cloud-top entrainment, and microphysics all failed to move rsut (radiation never saw them). The constant `q_c_diagnostic=1 g/kg` over-brightens thin warm Sc.

**Adiabatic in-cloud floor (committed).** Opt-in `--diagnostic-condensate-scheme adiabatic` replaces the flat floor with a capped adiabatic LWC that grows with cloudy depth (thin Sc dim, deep clouds hit the cap). Hardened through 4 codex rounds (smooth deck gate, valid hard reset, phase applied once with ice kept at the constant floor, byte-identical default incl fp32, backend + cross-field guards). **Science verdict: ineffective for THIS model (rsut −0.7)** — the model's marine-BL cloud is ~1400 m / 6 levels deep (sundqvist cf>0 over the too-moist BL), so the adiabatic value caps everywhere (dimming factor ~1.0). The floor is correct (a 1400 m cloud legitimately holds ~1 g/kg); the over-brightness is the cloud VERTICAL EXTENT, not per-level water. It remains a correct tool for realistic thin-Sc regimes.

**Cloud vertical extent.** Raising `cloud_rh_crit_bl` 0.7→0.92 moved rsut by −0.01. RETRACTED 2026-09-05: that knob (and `cloud_sigma_bl`) was never read by the cloud scheme — the null result measured a dead parameter, not the physics. Both fields were removed. The cover closure has ONE `cloud_rh_crit` at every level.

**Under-evaporation.** Convective gustiness (`surface_gustiness_zi`) adds in quadrature `|U|_eff=√(u²+v²+u_gust²)`; at the global-mean wind ~5.6 m/s a deeper zi (300→1000) gives evap +1% only (it dominates only the very-calm deep tropics). The mean wind itself is weak (−15%, coarse-resolution dynamical) and the near-surface air is humid (small sea-air q-gradient = the same humid-BL root). So under-evap is not a gustiness knob.

**Cloud-top entrainment for evap (committed).** `--cloudtop-entrainment-efficiency` at max (1.0) gives a small no-trade DUAL benefit (evap +4% / rsut −1.7) but PLATEAUS.

**Deployable best-config.** Combining the two effective no-trade levers: **χ=0.7 + cloud-top-entrainment=0.5** (warm-start) is stable to day-50 and improves THREE fields simultaneously, no trade: **rsut −10.6 W/m², evap +3%, T_land +0.5 K warmer** (the reduced albedo warms the cold-biased land). The aggressive stack (χ=0.5 + ent=1.0) BLOWS UP (non-finite winds day-40) — the partial fixes do not stack safely at max strength. Deploy: `--precision fp64 --cloud-inhomogeneity-factor 0.7 --cloudtop-entrainment-efficiency 0.5` via warm-start (χ<1 blows cold-start).

**Unified residual.** rsut (+119), evap (−44%), tas (−2.5), and downstream pr/clt/rlut all trace to ONE root: the too-moist/deep marine BL producing optically-thick high-cf cloud, plus coarse-resolution weak winds. Config levers are quadrature/τ-saturation/Pareto-limited because the BL moisture is set by the coupled convection/turbulence, not a knob. The residual needs a BL/Sc scheme model-dev (a proper cloud-top-entrainment/CTEI closure, sub-grid cloud optics to break plane-parallel τ-saturation, or higher resolution). Jobs: `conv_bech_{adiabws,rhbl92,gust1000,ent10,best2}`; memory `amip_cloud_albedo_lwp_diagnosis`, `amip_gustiness_evap_lever`.
