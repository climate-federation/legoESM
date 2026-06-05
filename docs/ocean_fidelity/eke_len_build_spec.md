# EKE mixing-length (`eke_len`) variant — build spec

Branch: `matching_Veros_oracle`. Follows the flux-form (F1–F9) and EKE (E1–E9)
build cadence: one gate per commit, each with an exact gate command + result,
adversarial review per gate, progress logged in `eke_len_build_progress.md`.

## Why (the gap E9 surfaced)

The prognostic-EKE closure (E1–E9) is **oracle-form-exact**: fed Veros's own
`eke`+`eke_len`, legoESM's `eke_kappa_gm = clip(c_k·L·√E, 0, k_max)` reproduces
Veros `K_gm` to machine precision. But it is **not yet switched on** in the ACC
recipe (`GMRediConfig.eke` stays `None`) because the **mixing length differs by
~25×**:

- legoESM `eke_mixing_length(L_rossby, cfg) = max(L_rossby, l_min)` uses the
  Visbeck first-baroclinic length `N̄·H/|f|` clamped to `[L_min, L_max]` →
  saturates near **~200 km** on the ACC.
- Veros uses a **Rhines-limited** length (`veros/core/eke.py:54–67`):

  ```
  C_rossby = Σ_z √(max(0,N²))·dzw·maskW / π          # 1st-baroclinic phase speed c₁ [m/s]
  L_rossby = min( C_rossby/|f|,  √(C_rossby/(2β)) )  # deformation radius (equatorial-limited) [m]
  L_rhines = √( √eke / β )                            # eddy-energy Rhines scale [m]
  eke_len  = max( eke_lmin, min(eke_cross·L_rossby, eke_crhin·L_rhines) )
  K_gm     = min( eke_k_max, eke_c_k·eke_len·√eke )
  ```

  In the developed ACC `√eke` is small so `L_rhines ≈ 8 km (47 km max) ≪
  eke_cross·L_rossby`, and the `min` selects Rhines → `eke_len ≈ 8 km`.

Switching EKE on today would make the prognostic `kappa_GM` ~25× too large.
This variant adds the Rhines-limited length as a **selectable scheme** so EKE can
be adopted apples-to-apples with Veros.

Note legoESM's Visbeck `L = N̄·H/|f|` omits Veros's `1/π` (so it is ~π× larger
than Veros's deformation radius even before clamping) and has no equatorial
`√(C/2β)` limiter. The `"rhines"` scheme therefore computes its **own**
Veros-form `L_rossby` (from `c₁ = ∫N dz/π`), not the clamped Visbeck `L`. The
`"rossby"` scheme keeps the existing behaviour bit-identical.

## ACC oracle target (`veros/setups/acc/acc.py:67–75`)

`enable_eke=True`, `eke_c_k=0.4`, `eke_c_eps=0.5`, `eke_k_max=1e4`,
`eke_cross=2.0`, `eke_crhin=1.0`, `eke_lmin=100`, superbee advection, isopycnal
diffusion. (`EKEConfig` already defaults `c_k`/`c_eps`/`l_min`/`kappa_gm_max` to
these; this build adds `eke_cross`/`eke_crhin`/`mixing_length_scheme`.)

## β (= df/dy) sourcing

Model run: **analytic** `β = 2Ω·cos(φ)/R_earth` from `config.constants.{Omega,
R_earth}` (Veros-pinned: `7.292115e-5`, `6.370e6`) and `grid.cos_lat`. Exact for
the sphere where `f = 2Ω sin φ`; equals Veros's discrete `df/dy` to O(dφ²). The
**oracle gate (L4) feeds Veros's own β** (so the discretization delta never
affects oracle-exactness — same protocol as E9 fed Veros's own `eke`).

## 2-D vs 3-D

legoESM `E` is 2-D (per column) by design (matches the 2-D Visbeck `kappa_GM`);
Veros `eke`/`eke_len` are 3-D. `β`, `L_rossby` are 2-D in both; the only 3-D part
of Veros's `eke_len` is `√eke`. So legoESM's 2-D `eke_len` compares to a
depth-reduced Veros `eke_len`; the per-cell L4 oracle check is pointwise (E9
protocol).

## Design (no duplicate numerics)

New **pure** functions in `lateral_mixing/eke.py`:
- `eke_rhines_length(E, beta, cfg) → √(√(max(E,0)+tiny)/max(β,β_floor))`
- `eke_deformation_radius(int_N_dz, f_abs, beta, cfg) → c₁=int_N_dz/π;
  min(c₁/max(|f|,f_floor), √(c₁/max(2β,β_floor)))`
- `eke_len_composite(L_def, L_rhines, cfg) → max(l_min, min(eke_cross·L_def,
  eke_crhin·L_rhines))`

`EKEConfig` gains `eke_cross: float = 1.0`, `eke_crhin: float = 1.0`,
`mixing_length_scheme: str = "rossby"`. `validate_eke_config` raises on unknown
scheme + non-positive cross/crhin.

`int_N_dz` (= Σ N·dz_half = the column `∫N dz`) is **exposed from the shared
`_eady_growth_and_length`** (it already computes `Σ N·dz_half` as `_col[...,2]`)
— so the `N²`/`N` numerics stay in one place (no re-derivation). `compute_eke_
kappa_gm` gains a `beta` arg and dispatches on `eke_cfg.mixing_length_scheme`;
`compute_eke_step_kappa` computes analytic β from the grid + config constants.

## Gates

| gate | deliverable | hard check |
|---|---|---|
| **L1** | pure `eke_rhines_length`/`eke_deformation_radius`/`eke_len_composite` + config fields + validation | direct unit tests; default `"rossby"` bit-identical (48 EKE/recipe/decomp tests green) |
| **L2** | `_eady_growth_and_length` returns `int_N_dz`; `compute_eke_kappa_gm` β-arg + scheme dispatch; `compute_eke_step_kappa` analytic β | `"rossby"` bit-identical; `"rhines"` yields a strictly smaller `L` on a developed state |
| **L3** | differentiability | `jax.grad` through rhines/deformation/composite finite + nonzero; no NaN at E=0 |
| **L4** | oracle confirmation (extend `compare_eke_kappa_veros.py`) | feed Veros's own `(∫N dz, |f|, β, eke)` → reproduce Veros `L_rossby`/`L_rhines`/`eke_len`/`K_gm` to machine precision |
| **L5** | recipe adoption (`eke=EKEConfig(mixing_length_scheme="rhines", eke_cross=2, eke_crhin=1)`) | developed-state `eke_len` ~8 km (not ~200 km), `kappa_GM` ~Veros magnitude; stale docstrings/ledger updated |
| **L6** | regression lock + measure-first free-run | `eke_len`-active golden at rtol=1e-12; re-run `run_acc_freerun.py` EKE-on, report KE/transport-gap movement (informational; integrator gap also contributes) |

## Doctrine guards

- Dispatch discipline: unknown `mixing_length_scheme` → `ValueError` (no silent fallback).
- No duplicate numerics: `N`/`N²` only in `_eady_growth_and_length`; β reuses `grid.cos_lat`.
- Constants discipline: Ω, R from `config.constants` (Veros-pinned), never literals; π is pure-math.
- Differentiability: √ regularized (`+tiny`), β/|f| floored — finite VJP.
- Default-OFF preservation: `"rossby"` default keeps E8 golden + all existing tests bit-identical.
