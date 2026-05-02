# Convection — static analysis

Audited modules: `_triggers.py`, `_plume.py`, `mass_flux.py`, `kuo.py`, `dca.py`, `sbm.py`, `zhang_mcfarlane.py`, `kain_fritsch.py`, `emanuel.py`, `tiedtke.py`, `bechtold.py`, `integration.py`, `config.py`.

## A. Confirmed correctness checks (PASS)

1. **Units consistent** at all kernel boundaries — tendency dimension `[K/s]` from `(M/ρ)·∂T/∂z + g/cp`; the latter `g/cp ≈ 0.0098 K/m` vs `M/ρ` [m/s]; product gives K/s. Same for q.
2. **Mass-flux gate** (`stratosphere_mass_flux_gate`, `mass_flux.py:132`): smooth sigmoid, never returns hard zero, defaults at 100 hPa with 15-hPa width — physically motivated, fully differentiable.
3. **Plume integrator** (`_plume.entraining_detraining_plume`): scan body separates *raw* mass flux (used as carry) from *reported* mass flux (suppressed by buoyancy + above-base). The decoupling is a known subtlety (line 488–542); folding suppression into carry would compound multiplicatively across levels — correct decision.
4. **Implicit-Euler relaxation** of `M_u_new` (Tiedtke/Bechtold/ZM): unconditionally stable for any `dt/tau`. Capped at `M_b_max` after relaxation.
5. **CMT closure** (`_plume.cmt_gregory_1997`): correct sign for du/dt = −(1/ρ)dF/dz with `F = −c_u M_u du/dz`. Stratospheric mass flux gating present (line 632–636) — same gate as kernel.
6. **q_c_conv ≥ 0**: every scheme enforces `jnp.maximum(dq_c_conv_dt, 0)` at the emit site OR the kernel structurally returns `≥ 0` (kernel: `delta_0 * M * jnp.clip(q_c_u, 0, None) / rho`).

## B. Potential issues (RANKED)

### B1. **`ν * Cd * |V| * u` surface friction sign convention** [INFO]
Verified consistent across surface_layer.py and vertical_diffusion.py. Tau_x is negative for positive u (drag); applied to the bottom interface as `rhs.at[:, -1].add(dt * tau_x / (ρ·dz))`, which decelerates u. OK.

### B2. **Bechtold downdraft cooling — questionable numeric**: `bechtold.py:283-294` and `tiedtke.py:283-287`
```
dT_dt_dd = -(L_v/c_pd) * (|M_d_base| * below_lcl_norm * 0.05) / rho_safe
```
The literal `0.05` (Tiedtke) / `0.05` (Bechtold) is described as "rough evap rate proxy [kg/kg]" but is undocumented in config and not a scientifically-derived constant. CLAUDE.md's "no hardcoded tunables in physics function bodies" likely applies here. Should be a `BechtoldConfig.downdraft_evap_proxy` field. NOT a numerical bug in isolation — it's a magnitude/dim check: `(|M_d|/ρ) * 0.05 = m/s × kg/kg`, so `(L_v/c_pd) × kg/kg / s = K/s`. Dimensionally OK.

### B3. **Tiedtke's `parcel_dT` / `parcel_dq` perturbation reused twice** [LOW]
`tiedtke.py:120-121` and `bechtold.py:141-142`: surface parcel is perturbed by `+ config.parcel_dT` for the LCL and the moist adiabat; but the moist adiabat used for CAPE is built from `T_base + parcel_dT` (Tiedtke) vs `T_pbl + parcel_dT` (Bechtold). The CAPE definition asymmetry is intentional (Bechtold uses PBL parcel; Tiedtke uses surface). Fine.

### B4. **`mass_flux_convection.diagnose_mass_flux_closure` doesn't gate `M_eq` by stratosphere** [LOW]
Line 279: `M_eq = convective_mask * config.M_scale`. The kernel multiplies the full *profile* by the strato gate. The scalar `M_c` itself is a column-mean, so this is fine — `M_c` is multiplied by `m_profile` (sin shape) before the kernel applies the gate.

### B5. **Emanuel buoyancy-sort variance is a non-physical proxy** [INFO]
`emanuel.py:160-170`: `sort_multiplier = 1 + 4*cu*var(ascending_weight)`. Maximum variance of a sigmoid distribution clipped to `[0,1]` is ¼ (achieved when half the bins are 0 and half are 1). So `4*var ≤ 1`, `sort_multiplier ∈ [1, 1+cu_coefficient]`. The "factor 4" normalization is correct given the variance bound. Documented: "buoyancy-sort detrainment multiplier in [1, 1 + cu]". OK.

### B6. **DCA uses Python `at[].set` inside `jax.lax.scan`** [PERF]
`dca.py:154-157`: `T_work.at[:, k - 1].set(T_adj_below)` etc. inside the scan body — XLA can fold these to scatter ops, but the pattern is unusual. Each iteration writes to two slots progressively. This is intentional (in-place "Newton-style" sweep) but may produce slow JIT compilation for large `nlev`. Documented but not a correctness issue.

### B7. **Surface-last index `T[:, -1]` semantics** [VERIFIED]
Consistently used as the parcel source. `compute_lcl` and the plume integrator are surface-last → reverse → scan → reverse paradigm. Documented inline. No bugs.

### B8. **No symmetry / convergence test for `_plume.entraining_detraining_plume`** [TEST GAP]
The plume integrator has only one indirect test through `test_convection_plume.py`. There's no analytic test for the entrainment limit `ε → 0` (should give a pure moist adiabat) or `ε → ∞` (should follow the environment).

### B9. **Bechtold ``column_MC`` divides by 0.05** [TUNABLE-IN-BODY]
`bechtold.py:201`: `mc_enhancement = column_MC / 0.05`. The `0.05 kg/m²/s` is a "typical strong-convergence value over tropical convective regions (Bechtold 2008 Fig. 2)". Should be a config field. Fine in spirit but violates CLAUDE.md "no hardcoded tunables in body".

### B10. **Smooth crossing index could divide by zero** [LOW]
`_triggers.py:267`: `safe_d = jnp.where(jnp.abs(dprofile) > 1e-30, dprofile, 1e-30)`. Replaces zero `dprofile` with `1e-30`. Differentiable, but `1e-30` is below `f32` denormalized range (~`1.18e-38`). On Apple Metal where `f32` is the default, this could behave oddly. In `f64` context this is fine.

### B11. **DCA `q_new_upper = jnp.minimum(q_upper, q_sat_upper)` is strictly non-differentiable for `q ≤ q_sat`** [POTENTIAL AD ISSUE]
`dca.py:142-143`: hard `min` between current vapor and saturation. Standard JAX `minimum` has a subgradient — gradient flows through whichever is smaller. For oversaturated layers (`q_v > q_sat`), gradient w.r.t. q_v is 0 (only `q_sat` flows). For sub-saturated (the common case), gradient w.r.t. q_v flows. Acceptable but worth noting. The blended assignment `q_adj_upper = q_upper + blend * (q_new_upper - q_upper)` softens this.

## C. Overall correctness assessment

The convection package is in **good shape** — the recent multi-month refactor has cleaned up most of the historical issues (q_v/q_c split in plume, stratosphere gate, AR1 stochastic, implicit-Euler relaxation). The smooth-everywhere AD design is consistent. The most actionable findings are:

* **B2** (Bechtold/Tiedtke `0.05` evap proxy in body) and **B9** (Bechtold `0.05 kg/m²/s` MC normalizer) — both should move into config NamedTuples per CLAUDE.md, but neither is a science bug.
* **B8** — add an analytic plume regression test (limits ε→0 / ε→∞).

No confirmed numerical bugs requiring source changes.
