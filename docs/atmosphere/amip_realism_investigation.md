# AMIP full-physics realism investigation

Status: **root cause identified; not yet resolved.** This note consolidates a
long diagnostic campaign on why a full-physics, prescribed-SST (AMIP) run with
`scripts/run/run_amip.py` produces an unrealistic climate, and what is (and is
not) the fix.

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

## What IS the fix (remaining work)

**Strengthen and balance the hydrological cycle** so precipitation removes the
evaporated moisture at moderate RH, before the column saturates:
- raise precipitation efficiency (convective + large-scale) so P ≈ E;
- raise evaporation toward ~2.7 mm/day (hfls 32 vs 88 — likely throttled by low
  surface winds, max_v ≈ 4 vs Earth ~7 m/s).

This is **convection/precipitation scheme development**, not a parameter sweep —
every accessible tuning knob above has been ruled out.

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
