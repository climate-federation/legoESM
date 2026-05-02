# Adversarial review: legoESM convection package

You are an independent adversarial physics reviewer for an Earth-system model. The codebase under review is a JAX implementation of atmospheric convection schemes that must be:

- Physically correct (units, signs, conservation, monotonicity).
- Differentiable end-to-end (no hard `if` on traced values, no NaN gradients, no unintended dead-gradient regions).
- Numerically stable (avoid divisions by zero, AD-safe limiters).

Your job: read the source files listed below and find every concrete bug, sign error, unit inconsistency, broken-gradient pattern, conservation violation, and AD-safety issue. Cite line numbers from the file paths I will give. If you believe a candidate concern is NOT a bug, explain why concretely. Be specific.

Static analysis already produced (please critique these as hypotheses, not as established facts):

1. **Bechtold/Tiedtke `0.05` evap proxy hardcoded in body** (`bechtold.py:283-294`, `tiedtke.py:283-287`). Style violation; not a numerical bug per-se but should be in config.

2. **Bechtold MC enhancement uses hardcoded `0.05 kg/m²/s` normalizer** (`bechtold.py:201`). Same.

3. **DCA mass conservation via column-rescale** (`dca.py:257-265`). Used to rescale per-level condensation candidate so column integral matches column-net drying. Allegedly preserves column water but distributes per-level non-physically.

4. **Plume integrator `entraining_detraining_plume` carry vs reported `M_u` decoupling** (`_plume.py:484-542`): claims `above_base_weight` and `plume_alive` are reporting filters only, NOT folded into the carry. Question: does the carry's `M_u_raw` correctly reflect entrainment/detrainment dynamics absent these filters?

5. **`compute_lcl` Bolton 1980 formula** (`_plume.py:122-190`). Floor `T - 55 ≥ 1` may produce a non-physical LCL for very cold parcels. Check.

6. **`smooth_lowest_crossing_index` no-crossing fallback** (`_triggers.py:201-327`): blends to the surface index when no crossing found. Question: is the gate sharpness adequate?

7. **Mass-flux kernel uses smoothed pressure gate `stratosphere_mass_flux_gate`** (`mass_flux.py:132-159`): defaults to 100 hPa with 15 hPa width. Sigmoid never returns hard zero. At the model top, a small fraction (factor ~0.013) of the mass flux still lands. Acceptable?

8. **Implicit-Euler relaxation of `M_u_new`** (`tiedtke.py:220`, `bechtold.py:247`, `zhang_mcfarlane.py:142`): `M_u_new = (M_u_old + (dt/τ) M_eq) / (1 + dt/τ)`. Always stable for any dt/τ. Then capped at `M_b_max`. Verify the cap timing — does capping after relaxation mask any other issue?

9. **Per-class `delta_0_eff` rescale in tiedtke/bechtold** (`tiedtke.py:234-260`, `bechtold.py:254-272`): kernel called with nominal `config.delta_deep`, then `dT_dt`/`dq_v_dt` rescaled by `delta_0_eff / config.delta_deep`. Does this preserve sign and magnitude correctly?

10. **Emanuel buoyancy-sort `sort_multiplier ∈ [1, 1+cu]`** (`emanuel.py:160-170`): claim is that `4 * var(asc_weight) ≤ 1`. Verify.

Files to read (full source; please cite line numbers):

- `src/legoesm/atmosphere/physics/convection/_triggers.py` (363 LOC).
- `src/legoesm/atmosphere/physics/convection/_plume.py` (673 LOC).
- `src/legoesm/atmosphere/physics/convection/mass_flux.py` (495 LOC).
- `src/legoesm/atmosphere/physics/convection/sbm.py` (165 LOC).
- `src/legoesm/atmosphere/physics/convection/dca.py` (277 LOC).
- `src/legoesm/atmosphere/physics/convection/kuo.py` (217 LOC).
- `src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py` (220 LOC).
- `src/legoesm/atmosphere/physics/convection/kain_fritsch.py` (241 LOC).
- `src/legoesm/atmosphere/physics/convection/emanuel.py` (258 LOC).
- `src/legoesm/atmosphere/physics/convection/tiedtke.py` (316 LOC).
- `src/legoesm/atmosphere/physics/convection/bechtold.py` (322 LOC).

Convention: column shape `(ncol, nlev)`, surface at `[:, -1]`, top at `[:, 0]`. Pressure increases with index.

Physical constants (`from legoesm import constants`): `R_d=287.04`, `c_pd=1004.64`, `g=9.80616`, `L_v=2.501e6`, `R_v=461.5`, `T_freeze=273.15`, `kappa=R_d/c_pd`, `p_ref=1e5`. `constants.epsilon = R_d/R_v ≈ 0.622`.

For each finding:
- File path : line range.
- One-sentence summary of the concern.
- Brief technical explanation (≤3 sentences).
- Severity: `critical` | `major` | `minor`.
- If the concern is **already-flagged-as-not-a-bug** (the 10 hypotheses above), explicitly say so.

If you find no substantive issues beyond the 10 already flagged, end with: `NO ADDITIONAL SUBSTANTIVE FINDINGS`.
