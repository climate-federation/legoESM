# DINO tendency certificate (differentiable-NEMO oracle)

Scoreboard for the legoESM-vs-NEMO DINO (Kamm et al. 2025) single-step tendency
match — the second NEMO oracle after GYRE. Every number is a **correlation
against NEMO's own dumped `*trd_*` trend fields** (never a norm), computed on
the harness `gap_audit/dino_term_compare.py` over NEMO's DINO restart.

## Setup

- **NEMO**: DINO on our local NEMO 5.0.2 build (`cfgs/DINO`, native `iom_nf90`,
  keys `key_qco key_vco_3d`). Trend dump via the ported MY_SRC `trddump` at a
  2000-step (≈62-day) spin-up restart (`RUN_M5SPIN`, finite velocities ~4 cm/s).
- **legoESM**: state + Mercator/topography geometry built directly from NEMO's
  own `mesh_mask` arrays via `fidelity.nemo_state_bridge.bridge_nemo_to_legoesm_topo`
  (PR #1137) — grids match by construction (dx/f to ~1e-7). EOS = `nemo_seos`
  with `eos_depth="geometric"` (required for the S-EOS thermobaric depth term).
- Correlations are **seam-excluded interior** (drop the single redundant
  periodic-wrap u-face + first/last lon columns; DINO is `ln_Iperio`).

## Certified (model-faithful)

| Term | u | v | NEMO scheme (DINO namelist) |
|------|-----|-----|------|
| S-EOS density | max\|Δρ'\| = **1.5e-5** | — | `ln_seos` (Roquet simplified) |
| **hpg** (pressure gradient) | **1.000** | **0.9999** | `ln_hpg_sco` |
| **pvo** (Coriolis) | **1.000** | 0.969* | `ln_dynvor_een` (EEN) |
| **rvo** (relative vorticity) | **0.997** | **0.9996** | EEN |
| **keg** (KE gradient) | **0.9998** | **0.9995** | Hollingsworth (`nn_dynkeg=1`) |
| **ttrd_ldf** (iso-neutral Redi) | k3–k30 **0.68–0.94** | — | `ln_traldf_iso` + `ln_traldf_msc` |

\* pvo-v: the residual is a **probe-diagnostic split artifact**, not a model
gap. The tendency probe splits the Coriolis (ENE Sadourny, 0.969) from the
relative vorticity, but the model runs `vorticity_scheme="al81"` (Arakawa-Lamb
= the full EEN scheme, `ff_f` 4-triad + `e3f`), which combines f+ζ in one
faithful PV flux — the same al81 path that certifies rvo at 0.997. pvo-u is
1.000 with even the simple split.

## Certified numerics reproduced (vs GYRE, which used different selections)

DINO exercises scheme selections GYRE never tested and legoESM matches them:
**S-EOS** (GYRE: EOS-80), **EEN** vorticity (GYRE: ENE), **Hollingsworth** KE
(GYRE: C2), **`ln_hpg_sco`** s-coord PGF (GYRE: `ln_hpg_zco`), and the **MSC**
(`ln_traldf_msc`) explicit-K33 iso term (GYRE: msc=F). The MSC `akz` explicit
vertical diagonal was ported term-for-term from `traldf_iso_a33` (PR #1137).

## Characterized follow-ups (not yet certified)

- **`ttrd_ldf` surface (k1/k2)** — the S²-sensitive K33 diagonal exposes a
  pre-existing **ML-ramp / MLD** slope mismatch: legoESM's native slopes match
  NEMO's dumped `wslpi_stg` at corr 0.99 *below* the mixed layer but ~0.33
  *inside* it. Root = the MLD criterion (NEMO's `MAX(N²,0)` integral from `rn2b`
  vs legoESM's potential-density difference) drives a different ramp anchor.
  A GYRE-shared sub-campaign (the slope code is shared) — deferred.
- **ldf** momentum viscosity — a **near-noise** term at this state
  (`utrd_ldf` rms 1.7e-8, ~140× below hpg/pvo). `nn_ahm_ijk_t=20` gives a
  mesh-scaled `A_h = ½·rn_Uv·max(e1,e2)`, but legoESM's momentum-viscosity
  *tendency* path takes a scalar `A_h` (the spatial-`A_h` operator exists only
  for the KE-dissipation diagnostic). Gated by a spatial-A_h momentum feature;
  deferred (low value on a dynamically negligible term).
- **zad** vertical momentum advection — near-noise here (`utrd_zad` rms
  1.5e-8); NEMO adds the `key_qco` grid-motion velocity (`ww + wsd`) that
  legoESM's z* w lacks. Deferred (small).
- **e3w** in the MSC `akz` — a T-thickness average vs NEMO's analytic `e3w_0`
  (few-percent).

## Provenance

Bridge + MSC: PR #1137 (branch `feat/nemo-dino-topo-bridge`), both with folded
physics-validator reviews. Harness: `~/oracle-builds/nemo5/gap_audit/dino_term_compare.py`
(measurement-only). NEMO consts: rho0=1026, g=9.80665, dt=2700 s.
