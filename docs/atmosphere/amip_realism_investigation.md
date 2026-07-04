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
