# RCEMIP1 plane CRM (ocean, SST 300 K) — replication runbook

This is the RCEMIP1 protocol (Wing et al. 2018, doi:10.5194/gmd-11-793-2018)
run as a doubly-periodic cloud-resolving model on the plane non-hydrostatic
dycore. It is the configuration that produced the 60-day RCE checkpoint the
microphysics-viability campaign was built on, promoted out of the untracked
scratch scripts it used to live in so that a clean checkout can reproduce it.

## Run it

```bash
scripts/run/run_rcemip1_crm.sh spinup                 # 60 days, gray + Kessler
scripts/run/run_rcemip1_crm.sh production morrison    # 1 day, RRTMGP + clouds
```

## Oracle-faithful configuration (gSAM `CASES/RCEMIP1`)

The wrapper above is the legacy uniform-grid config. To run the case as gSAM
runs it, drive `run_rcemip_plane.py` directly with the oracle's own vertical
grid via `--sam-grd`:

```bash
GRD=<gSAM>/CASES/RCEMIP1/grd
LEGOESM_RCEMIP_PLANE_FP32=1 JAX_PLATFORMS=cuda .venv/bin/python \
  scripts/run/run_rcemip_plane.py \
  --nx 128 --ny 128 --nlev 74 --sam-grd $GRD \
  --dx 3000 --dt 6 --T-sfc 300.0 --insolation rcemip \
  --precision float32 --semi-implicit --radiation-interval 60 \
  --hyperdiff 1.0e7 --radiation rrtmgp --clouds --microphysics morrison \
  --seed-kind band_noise --seed-kmax 8 --theta-noise-amp 0.05 \
  --steps 864000 --print-every 7200 --snapshot-days 1.0 \
  --checkpoint-every 14400 --restart latest --output results/<out>
```

What matches the oracle, and what does not:

| setting | gSAM `prm_rect` / `grd` / `Build` | here |
| --- | --- | --- |
| vertical grid | 74 levels, 37 m first centre → 500 m aloft, 33 km top | identical (`--sam-grd` reads the oracle file) |
| radiation | RRTM, `nrad=30` × dt 12 s = 360 s | RRTMGP, 360 s |
| insolation | `solar_constant=551.58`, `zenith_angle=42.05`, perpetual | identical (`--insolation rcemip`) |
| SST / surface | `OCEAN`, `tabs_s=300`, interactive fluxes | identical |
| Coriolis / large-scale forcing | off / none | identical |
| microphysics | `MICRO_SAM1MOM` (default) or `MICRO_M2005` | `morrison` (SAM M2005 flavor) |
| **dx / domain** | 3000 m, 576×288 = 1728×864 km | 3000 m, 128² = 384×384 km (compute limit) |
| **dt** | 12 s | **6 s** — see below |

**Do not use `--microphysics kessler` for the spin-up.** It is warm-only, so a
60-day spin-up equilibrates with no ice microphysics and no ice-radiative
effect and then hands that biased upper troposphere to every production leg.
Measured: `S_ice` climbs monotonically past 20 because nothing consumes the ice
supersaturation, versus ~1.65 (the IFS homogeneous-freezing allowance) under
morrison.

**dt is 6 s, half the oracle's 12 s.** At dt=12 on this grid the run is
MARGINALLY stable: it went non-finite in 3 of 6 attempts at convection onset
and completed the other 3, with identical seed and flags — fp32 GPU
non-determinism through a chaotic onset decides which. No single operator
reproduces it. dt=6 has been robust in every run. `cfl_guard` warns above
`0.08*dz_min`, which is a "expect trouble" marker, not a sharp limit.

Short smoke on CPU, if you have no GPU:

```bash
DAYS=0.05 NX=32 PLATFORM=cpu CKPT_EVERY=720 scripts/run/run_rcemip1_crm.sh spinup
```

Environment overrides: `DAYS NX NY NLEV DX DT OUT SPINUP_OUT PLATFORM PY CKPT
CKPT_EVERY SEED_AMP SEED_KMAX HYPERDIFF RAD_INTERVAL VALIDATE`.

The spin-up passes `--restart latest`, so it is idempotent: re-running resumes
from the newest checkpoint instead of starting over, and a 60-day run can be
done across several sittings. On an empty output directory it falls back to a
fresh initial condition.

## Why two phases

**Spin-up** — 128 x 128 x 100, dx = 4 km (512 km doubly-periodic domain),
dt = 6 s, RRTMGP radiation with interactive clouds, Kessler microphysics,
float32, run to 60 days with daily checkpoints. Kessler keeps the microphysics
cheap over the long haul, but the radiation stays RRTMGP throughout — see below.

**Production** — restart from that checkpoint with RRTMGP plus interactive
clouds and whichever microphysics scheme you actually want. The six schemes the
viability campaign exercised on this case are `kessler`, `sundqvist`,
`morrison`, `thompson`, `seifert_beheng` and `p3`; `run_rcemip_plane.py` also
accepts `sdm`, `fast_sbm`, `ml_emulator` and `none`, which have not been
assessed here. Then grade the
time mean against the published RCEMIP 300 K ranges with
`scripts/validate/compare_rce_vs_rcemip_sam.py`: column water vapour,
near-surface T and q_v, cold-point T and height, mid-tropospheric w RMS, peak
cloud fraction.

Switching from Kessler to a **two-moment** scheme requires
`--restart-reset-condensate`, which the wrapper applies automatically: copying
bare condensate mass with zero number concentration is thermodynamically
inconsistent and blows the run up. The new scheme regrows condensate in
minutes of simulated time. `kessler` and `sundqvist` inherit the state directly.

## Configuration, and the traps

| Setting | Value | Reason |
| --- | --- | --- |
| grid | 128 x 128 x 100, dx = 4 km, H = 33 km | dx = 4 km is the tractable apples-to-apples CRM resolution (SAM's GATE/LBA are dx = 100 m LES) |
| dt | 6 s | pairs with nlev = 100; see below |
| precision | `float32` + `LEGOESM_RCEMIP_PLANE_FP32=1` | driver default is float64; fp32 roughly halves GPU time |
| **seed** | `--theta-noise-amp 0.05 --seed-kind band_noise --seed-kmax 8` | **mandatory — see below** |
| hyperdiffusion | `1.0e7`, passed explicitly | the value the 60-day reference run actually used and the one verified here. `run_rcemip_plane.py` now defaults to `dx_aware_hyperdiff(dx)` = 1e8*(dx/1000)^4 (2.56e10 at dx = 4 km) instead of the old dx-blind 1.0e6; both convect, so the wrapper pins the proven number rather than silently changing the reference configuration |
| radiation | `rrtmgp --clouds` in both phases, refreshed every 150 steps = 900 s | inside the 300-1800 s RCEMIP norm; gray under-drives, see below |
| microphysics | `kessler` (spin-up) / your choice (production) | |
| SGS, acoustics | driver defaults: Smagorinsky `Cs = 0.19`, `--sgs-vertical`, `--substep-horizontal-acoustic`, `--semi-implicit` | CRM preset, 3-D SGS |

### The initial condition must not be supersaturated

`WING_T_V0` used to be pinned at 295 K. Wing 2018 Eq. (3) prescribes
`T_v0 = T0*(1 + 0.608*q0)` with `T0` the case SST (303.4 K at SST 300 K), and
`q0` is case data too — 12 / 18.65 / 24 g/kg at SST 295 / 300 / 305 K, "adjusted
so that the relative humidity is near 80 % in the lower atmosphere". The pinned
value left the initial column ~8 K too cold and therefore **139 % RH**: the run
began 40 % supersaturated and condensed the excess in its first steps. That
condensation kick — not hydrostatic adjustment — was the ~0.8 m/s `max|w|`
spike previously documented at step 1. Both are fixed
(`wing2018_T_v0`, `wing2018_q_sfc`); the IC now sits at 74-83 % RH across the
three SSTs.

### Supersaturation diagnostics

Every print step reports `Smax=<S_liq>/<S_ice>`: the peak saturation ratio over
warm cells (liquid) and sub-freezing cells (ice). `S_liq` and its per-level
profile are also written into `profile_step_*.npz`. The run prints a verdict at
the end and flags sustained liquid supersaturation above 1.02.

Ice supersaturation is physical and expected: gSAM `MICRO_SAM1MOM/cloud.f90`
(after IFS) allows pristine air below 235 K to reach
`rh_homo = 2.583 - T/207.8` (~1.45 at 235 K, ~1.67 at 190 K) before
homogeneous freezing, withdrawing the allowance where cloud ice already exists.
That ramp is implemented in `thermo.homogeneous_freezing_rh_factor` and applied
to the DEPOSITION target (not the nucleation gate) in morrison/thompson/p3; it
is on by default and disabled per scheme with
`homogeneous_ice_supersaturation=False`.

### The setting that decides whether it convects at all

`run_rcemip_plane.py` defaults to `--theta-noise-amp 0.0` with
`--seed-kind smooth_k1`, which applies **no initial perturbation**. A
horizontally uniform rest state stays horizontally uniform, so the run
completes cleanly, conserves mass to machine precision and exits 0 — while
being a pure radiative-equilibrium column replicated across every grid point,
with `max|w|` of order 1e-4 m/s and not a single convective cell. Nothing in
the run log flags this; the only symptom is the vanishing vertical velocity.
`docs/dev-notes/CRM_faithful_SAM.md` records the same trap at iter-175/176
("launch bug: `--theta-noise-amp` default 0.0 => un-seeded RCE stays laminar")
and as CONV-TRIGGER #83.

`band_noise` rather than `smooth_k1` matters because `smooth_k1` puts all the
seed energy into a single k=1 cosine, which organises one domain-filling
circulation instead of a population of convective cells. `--seed-kmax` must
stay well below `nx/2`; at `nx = 32` the default 8 is fine, at `nx = 16` it is
the Nyquist wavenumber and the grid-scale seed goes non-finite immediately.

### Radiation is RRTMGP in both phases

Gray radiation is much cheaper and is tempting for a 60-day spin-up, but it
**under-drives the circulation**: `docs/dev-notes/CRM_faithful_SAM.md` (#85)
measured w RMS around 0.07 m/s and an inverted w'² profile under gray, versus
healthy mid-tropospheric convection under RRTMGP, and traced the entire
convective-intensity deficit to gray under-driving. A state spun up under gray
is not the state RRTMGP would have equilibrated to, so the production leg
inherits that bias no matter which microphysics it uses. The wrapper therefore
runs RRTMGP with interactive clouds in both phases.

`SPINUP_RAD=gray` is available if you explicitly want to trade fidelity for
wall-clock. The wrapper then stops passing the explicit `--clouds`, which only
means something for a band model — note that `run_rcemip_plane.py` has clouds
enabled by default either way, so this changes the command line, not the cloud
setting; use `--no-clouds` on the driver if you actually want them off.
Radiation is gated to every 150 steps in both phases, which is what keeps
RRTMGP affordable over 60 days.

### Do not mix the vertical grid and the timestep

The seeded initial condition is stable at `nlev = 100` with `dt = 6 s`. The
same seed at `nlev = 30` with `dt = 20 s` goes non-finite within about 25 outer
steps regardless of the hyperdiffusion coefficient. If you change one of
`NLEV`/`DT`, re-check stability over a few hundred steps before committing to a
long run.

## What a healthy run looks like

Convection is not instantaneous. From a verified 32 x 32 x 100 CPU spin-up at
the default settings (RRTMGP + clouds, Kessler, dt = 6 s), `max|w|` sits near
0.17 m/s for the first ~300 steps, grows through 2.1 m/s and 7.4 m/s, peaks at
17.5 m/s at step 600 (t = 1 hour), then settles to a sustained 4.7-8 m/s with
mass drift at 5e-6 or below. **That growth-then-settle signature, not the exact
peak, is the thing to look for.** The same run under `SPINUP_RAD=gray` traces
nearly the same early curve (peak 17.6 m/s) — the two diverge in the sustained
convective intensity over the longer haul, not in whether convection triggers.

You no longer have to notice this yourself. `run_rcemip_plane.py` warns up front
when `--theta-noise-amp` is 0 (or when `seed_kind` is `smooth_k1`), and prints a
verdict at the end:

```
  *** LAMINAR RUN: peak |w| over the second half of the integration was 3.783e-03 m/s (< 0.1 m/s).
  *** The domain never convected — it stayed a horizontally uniform radiative-equilibrium column.
```

The peak is measured over the **second half** of the run, because the rest
state's initial hydrostatic adjustment reaches ~0.8 m/s at step 1 even for a
column that stays perfectly uniform — a whole-run peak would clear any
convective threshold and the check would never fire. It is sampled on print
steps only, so a very coarse `--print-every` yields "no verdict" rather than a
false one. A run that aborts non-finite says so instead of reporting a
healthy-looking peak.

Expected published-range signatures once spun up: CWV around 45-55 mm,
mid-tropospheric w RMS around 0.3-0.7 m/s, cloud fraction around 0.2, a
moist-adiabatic temperature profile, and a cold point near 194-198 K.

## Outputs

```
<OUT>/run.log                        stepwise diagnostics + the validator table
<OUT>/snapshots/profile_step_*.npz   vertical profiles (validator input)
<OUT>/snapshots/sfc_*.npz            surface + 4-level fields
<OUT>/snapshots3d/vol_*.npz          full 3-D condensate/MSE/w/T volumes
<OUT>/checkpoints/ckpt_*.npz         resumable state
```

`--snapshot-every` is what writes `profile_step_*.npz`, and the underlying
driver defaults it off — so a run configured without it produces nothing the
validator can read. The production phase always sets it.

Render volumes with `scripts/plot/plot_rcemip_3d.py` and surface fields with
`scripts/plot/plot_rcemip_snapshots.py`.

## Related

- `docs/dev-notes/CRM_faithful_SAM.md` — faithfulness tracker vs the gSAM oracle
- `docs/science/specs/CRM_implementation.md` — dycore and physics specification
- `scripts/run/run_gate_plane.py`, `scripts/run/run_lba_plane.py` — the other
  two SAM deep-convection cases
