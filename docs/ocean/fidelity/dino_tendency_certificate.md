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

## Surface forcing byte-verification (usrdef_sbc CASE(4))

The analytic DINO forcing is a **byte-exact** transcription of NEMO's
`usrdef_sbc` CASE(4). Each field, transcribed from NEMO's Fortran and evaluated
cell-by-cell on the matched grid at the same day-of-year (kt=2000 → day 62.5,
seasonal phases c1=−0.317 @ 21 Jun, c2=−0.749 @ 21 Jul) against legoESM's
forcing functions, using NEMO's restart SST/SSS for the restoring terms:

| field | max\|Δ\| | note |
|-------|--------|------|
| `utau` wind stress | **5.6e-17** | smoothstep (`(3−2s)s²`) on the same 9 lat-nodes, on `gphiu` |
| `T*` (heat target) | **1.8e-14** | seasonal: `T*_s=−0.5−0.5·c2`, `T*_n=5+3·c2` |
| `qns` heat flux | **7.1e-13** | `rn_trp(SST−T*)` ≡ `A_theta(T*−SST)`, `A_theta=40=−rn_trp` |
| `qsr` solar | **5.7e-14** | `max(230·cos(π(φ−23.5·c1)/180),0)`, seasonal declination |
| `S*` (salt target) | **7.1e-15** | non-seasonal Munday cosine − 1.25·gaussian dip |
| `sfx` salt flux | **2.7e-17** | `rn_srp(SSS−S*)` ≡ `A_S(S*−SSS)`, `A_S=3.858e-3=−rn_srp` |

Verified matches: the day-of-year **epoch** (both put kt=0 at Jan 1 — NEMO's
`ztime` drops the `zday_year0` offset), the seasonal-phase formulas (c1/c2,
half-year = 180·24 h), `rn_emp_prop=0` (no E–P; salt restoring only), and the
1.3× westerly `taum` boost for TKE. The only sub-machine difference is the
K-conversion `c_p=3991.86` vs NEMO `rcp=3991.868` (2e-6 relative, negligible);
the fluxes themselves are exact. Harness:
`~/oracle-builds/nemo5/gap_audit/dino_forcing_byteverify.py`.

**Consequence**: forcing is removed as a confound. Any remaining solution
residual (the +0.05 °C thermocline, SST) is dynamics/EOS/numerics, not forcing —
so a controlled comparison for the `zdfevd` threshold refinement (§follow-ups)
is now well-posed.

## Solution check (62-day forward, cell-by-cell on NEMO's own grid)

Tendency-match is necessary but not sufficient — do the *solutions* track? We
run legoESM forward from rest for 62 days **on NEMO's exact grid** (state +
geometry built cell-for-cell from NEMO's `mesh_mask` via the fidelity bridge, so
the comparison is cell-by-cell, not grid-robust interpolation) vs NEMO's
62-day-from-rest `RUN_M5SPIN`:

- **T corr 0.99** cell-by-cell; salinity within ~0.1 PSU (interior).
- **Native NEMO grid** (opt-in, `DINOConfig.nemo_faithful_grid` /
  `nemo_faithful_dino_config` / `run_dino.py --nemo-faithful-grid`): a standalone
  DINO run now builds NEMO's exact DINO R1 mesh — 48×195, equator on a T-point,
  faces [1°, 49°] — reproducing `glamt`/`gphit` to **3e-6°** with **100%
  wet/dry-domain agreement** (bathymetry depth corr 0.92; residual is legoESM's
  analytic bowl vs NEMO's full-step `ln_zps` levels). Default OFF (the 48° vs
  legoESM's 50° basin is not comparable to prior DINO runs). PR #1137.
- **Basin-mean T(z)** now tracks within ~0.1 °C through the thermocline:
  **T@262 m = 9.55 vs NEMO 9.50**, **SST 14.13 vs 14.07** (see the cold-bias
  resolution below).

### Cold-bias root cause — convective-adjustment trigger (RESOLVED)

The first solution check showed legoESM **−0.72 °C colder at 262 m** (8.78 vs
9.50). Characterize-before-patch pinned it code-first — and it was **not** the
suspected iso ML-surface term:

- **Iso / GM / MSC ruled out**: toggling all lateral diffusion fully off moved
  T@262 m by 0.00 °C over 62 days (too slow from rest).
- **Initial condition ruled out**: legoESM's `dino_lat_lon_state` IC reproduces
  NEMO's analytic `usrdef` CASE(4) to **+0.08 °C** (legoESM starts marginally
  *warmer*). The bias is pure dynamics — NEMO holds T@262 m nearly steady
  (9.545→9.50) while legoESM eroded 0.85 °C (9.625→8.78).
- **Culprit = enhanced-diffusion convection**: turning convection off restored
  T@262 m→9.58 and SST→14.12 (both NEMO). NEMO's `zdfevd` (`ln_zdfevd=.true.`,
  `rn_evd=100`, `nn_evdm=1`) is a **hard `rn2<0` switch** on the adiabatic
  (`eosbn2`, local α,β) N². legoESM's default is a differentiable
  **sigmoid** of N² (`smooth_transition`, sharpness 1e6) that **leaks**
  ~O(`K_conv`=100 m²/s) mixing into weakly-**stable** water (N²~+1e-6 s⁻²) that
  NEMO never mixes, over-eroding the thermocline. Variant sweep: hard-step
  restores 9.55–9.57; `n2_mode` (insitu vs adiabatic) is a 0.02 °C tiebreak.
- **Fix (config-only)**: the `nemo_paper` recipe now selects
  `convection_smooth_transition=False` + `convection_n2_mode="adiabatic"`
  (NEMO-faithful hard step on true static stability). The convection *kernel* is
  unchanged; both paths already existed. Other recipes keep the smooth legoESM
  default (byte-identical). PR #1137.
- Caveats: 62 days is short (not the M6 climate); forcing is analytic
  `usrdef`-style, not byte-verified identical to NEMO's.

## Characterized follow-ups (not yet certified)

- **`zdfevd` trigger refinement** — the `nemo_paper` convective switch fires at
  `N² < 0` on a single time level. NEMO's `zdfevd` fires where
  `MIN(rn2, rn2b) <= −1e-12` (two-time-level, with a small *negative* threshold
  to ignore marginally-neutral interfaces). The kernel already supports both
  (`EnhancedDiffusionConfig.n2_threshold`, `two_level_trigger`); `DINOConfig`
  does not yet expose them, so a hard 100 m²/s coefficient can flicker per step
  in marginal columns. Second-order (62-day match is 9.55 vs 9.50); add the two
  DINOConfig fields + threading when sub-0.05 °C oracle fidelity is needed and
  the forcing is byte-verified. (Physics-validator, PR #1137.)
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
