      trace-time host float (jnp.sqrt would have broken jit).  Direct f32
      near-threshold gradient test added.
    - Finding 1: a finite positive column below eps was ZEROED, violating the
      conservation claim.  Degenerate columns now split: net-positive keeps the
      plain clip (error bounded by eps_eff, ~1e-14 kg/m2 column water at f32),
      net-negative is zeroed as before (minimum-creation).  The generic axis
      API was also broken for non-trailing axes (weight broadcasts on the last
      dim) — now raises ValueError instead of silently mis-conserving.
      Finite-inputs + positive-weights contract documented.
    - Finding 5 (BLOCKER, documented not code-changed): the p_s dry-mass fixer
      runs after the clamp and shifts diagnosed water by c*B/g.  Pre-existing
      interaction (any q field x any p_s fixer), not introduced here; bounded
      empirically at <~0.2 kg/m2/yr (the clamp-on arm's end-to-end +3.10
      residual, measured WITH the fixer active, matches the +2.9 net-negative
      prediction).  Principled close-out (moisture-aware dry-mass fixer) is a
      follow-up.
    
    Findings 2 (factor>1 impossible for positive weights) and 4 (same-species
    redistribution conserves the latent term) were REFUTED by codex — no change.
    
    Tests 21 -> 27; ruff delta zero (pre-existing debt untouched).
    
    Co-Authored-By: Claude <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_01KDW1s4ErvmddEqGTY6CXQm

codex
Re-review of `901b1d69f`: two blockers remain.

| Prior finding | Status | Evidence |
|---|---|---|
| 1. Tiny-positive loss / bad non-trailing axis | **RESOLVED** | The trailing-axis guard now raises before wrong broadcasting, and degenerate net-positive columns retain the plain clip via factor `1.0`. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:157) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:177) |
| 2. Factor clipping could lose mass | **RESOLVED** (refuted) | With documented positive weights, `after >= before`; a positive column’s ratio is in `[0,1]`. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:137) |
| 3. Float32 VJP Inf | **RESOLVED** | `eps_eff=max(eps,sqrt(tiny))` and strict `after > eps_eff` keep the active quotient’s squared denominator above `tiny`. I verified finite eager and jitted gradients at the first live value for f16, bf16, f32, and f64. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:165) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:177) |
| 4. Same-species redistribution needs a temperature fix | **RESOLVED** (refuted) | The operation redistributes the same water species; with fixed `p_s`, its column latent contribution is unchanged. |
| 5. `p_s` fixer changes diagnosed water after clamp | **STILL-OPEN** | The change only documents the defect. The fixer still computes uncapped `c=(M_old-M_new)/A` and adds it to `p_s` after the clamp, so `ΔW=cB/g` remains unrestricted for a cold start or large transport error. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1022) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1110) MPI has the same uncapped operation. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:599) The cited `<~0.2 kg m⁻² yr⁻¹` is one trajectory’s residual comparison, not a bound on `cB/g`; it cannot justify accepting the defect generally. |
| 6. MPI flag inert | **RESOLVED** for production wiring | MPI now uses the same selector and water/non-water split. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1105) Production passes `model.sigma_coord`, so `dsigma` agrees there. [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:7081) It is not structurally exact: serial reads `self.sigma_coord`, MPI reads the independent factory argument. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1008) [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1112) A mismatched direct caller remains unguarded. |

**NEW-ISSUE:** the degenerate split deliberately creates up to `eps_eff` weighted mass for net-positive mixed-sign columns. Benign for bf16/f32 (`~1.08e-19`) and f64 (`1e-30`), but not f16: `sqrt(tiny_f16)=0.0078125`. For f16 `q=[0.005,-0.001]`, the result is `[0.005,0]`: it creates `0.001` kg/kg, about 10 kg m⁻² at 1000 hPa; the allowed worst case is about 80 kg m⁻². F16 is accepted by the precision parser. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:170) [precision.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/precision.py:50)

Tests: all 27 current unit tests pass. Of `TestCodexFindings`, tiny-positive, axis, f32-gradient, and MPI-source tests fail pre-fix; monotone `1e-25` and ordinary `1e-3` conservation pass pre-fix, so they do not pin a change. The f32 gradient regression catches the old NaN, but its `after=2e-30` is below the new effective threshold and therefore does not exercise the new active quotient branch. [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:268) [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:281)

VERDICT: FIX-FIRST — blocking: finding 5 remains unfixed with no general bound; new FP16 degenerate columns can create material water mass.
tokens used
150,470
Re-review of `901b1d69f`: two blockers remain.

| Prior finding | Status | Evidence |
|---|---|---|
| 1. Tiny-positive loss / bad non-trailing axis | **RESOLVED** | The trailing-axis guard now raises before wrong broadcasting, and degenerate net-positive columns retain the plain clip via factor `1.0`. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:157) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:177) |
| 2. Factor clipping could lose mass | **RESOLVED** (refuted) | With documented positive weights, `after >= before`; a positive column’s ratio is in `[0,1]`. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:137) |
| 3. Float32 VJP Inf | **RESOLVED** | `eps_eff=max(eps,sqrt(tiny))` and strict `after > eps_eff` keep the active quotient’s squared denominator above `tiny`. I verified finite eager and jitted gradients at the first live value for f16, bf16, f32, and f64. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:165) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:177) |
| 4. Same-species redistribution needs a temperature fix | **RESOLVED** (refuted) | The operation redistributes the same water species; with fixed `p_s`, its column latent contribution is unchanged. |
| 5. `p_s` fixer changes diagnosed water after clamp | **STILL-OPEN** | The change only documents the defect. The fixer still computes uncapped `c=(M_old-M_new)/A` and adds it to `p_s` after the clamp, so `ΔW=cB/g` remains unrestricted for a cold start or large transport error. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1022) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1110) MPI has the same uncapped operation. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:599) The cited `<~0.2 kg m⁻² yr⁻¹` is one trajectory’s residual comparison, not a bound on `cB/g`; it cannot justify accepting the defect generally. |
| 6. MPI flag inert | **RESOLVED** for production wiring | MPI now uses the same selector and water/non-water split. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1105) Production passes `model.sigma_coord`, so `dsigma` agrees there. [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:7081) It is not structurally exact: serial reads `self.sigma_coord`, MPI reads the independent factory argument. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1008) [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1112) A mismatched direct caller remains unguarded. |

**NEW-ISSUE:** the degenerate split deliberately creates up to `eps_eff` weighted mass for net-positive mixed-sign columns. Benign for bf16/f32 (`~1.08e-19`) and f64 (`1e-30`), but not f16: `sqrt(tiny_f16)=0.0078125`. For f16 `q=[0.005,-0.001]`, the result is `[0.005,0]`: it creates `0.001` kg/kg, about 10 kg m⁻² at 1000 hPa; the allowed worst case is about 80 kg m⁻². F16 is accepted by the precision parser. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:170) [precision.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/precision.py:50)

Tests: all 27 current unit tests pass. Of `TestCodexFindings`, tiny-positive, axis, f32-gradient, and MPI-source tests fail pre-fix; monotone `1e-25` and ordinary `1e-3` conservation pass pre-fix, so they do not pin a change. The f32 gradient regression catches the old NaN, but its `after=2e-30` is below the new effective threshold and therefore does not exercise the new active quotient branch. [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:268) [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:281)

VERDICT: FIX-FIRST — blocking: finding 5 remains unfixed with no general bound; new FP16 degenerate columns can create material water mass.
