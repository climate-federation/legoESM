# Bottom-Cell Drag for legoESM Lat-Lon C-Grid Ocean — Physics & Calibration Design

Status: Research / design only. No source files modified. The numerical implementation
is the subject of a parallel dycore-design effort; this document specifies the **physics,
calibration recipe, and best-practice match** to MITgcm / MOM6 / NEMO / POP.

Companion documents (read first):
- `docs/dev-notes/research/bottom_drag_literature.md` — wider literature synthesis on linear vs
  quadratic and ACC drag sensitivity.
- `docs/dev-notes/research/why_westward_drake.md` — diagnosis of the current −405 Sv westward Drake
  transport as a structural failure of depth-mean drag + flat-bottom + GM-on-tracers-only.

Repo touchpoints surveyed before writing:
- `src/legoesm/ocean/dynamics/ocean_tendency_common.py:245` — `implicit_bottom_drag_factor(dt, drag_r, H)` returns the depth-averaged factor `1 − dt·r/H` used by both lat-lon C-grid and MPAS barotropic substeps.
- `src/legoesm/ocean/state.py:398` — `LatLonCGridOceanConfig.bottom_drag_r: float = 0.0`.
- `src/legoesm/ocean/experiments/global_overturning.py:90` — experiment uses `bottom_drag_coeff = 1.1e-3 m/s`.
- Existing companion docs above.

---

## 1. Linear vs quadratic — recommended priority order

**Punchline. Add linear bottom-cell drag first; it directly fixes the Drake-transport
sign problem with minimal code, no new state variables, full AD-friendliness, and a
well-defined linear stability bound. Quadratic and BBL-aware variants come later.**

The diagnostic in `why_westward_drake.md` shows that with the depth-mean form
`(1 − dt·r/H)` applied to `<u>`, `u_bot ≈ −7 cm/s` is left unconstrained because the
column-mean drag is satisfied by *any* `u(z)` whose vertical average meets the
wind/drag balance. The single-point intervention that closes this gap is to apply drag
at the bottom-most wet level only, on the C-grid `u` and `v` faces independently.

Recommended progression:
1. **Linear bottom-cell drag** (this design): `(du/dt)_drag = −r · u_bot` applied in the
   bottom-most wet `u`-cell at each `(i,j)`, identical for `v`. Treat implicitly within
   the baroclinic vertical-viscosity step or as a stand-alone implicit factor
   `u_bot ← u_bot / (1 + dt·r/dz_bot)`.
2. **Quadratic bottom-cell drag with background velocity** (next phase): `(du/dt)_drag
   = −C_d · √(u_bot² + v_bot² + u_bg²) · u_bot`. The `u_bg` floor (MOM6 default
   `DRAG_BG_VEL = 0.1 m/s`) prevents drag vanishing in slow flow; without it,
   stagnation regions develop unphysical near-zero dissipation.
3. **BBL-aware quadratic** (long-term): MOM6 `channel_drag` style, where drag is
   distributed over a BBL of thickness `max(HBBL, dz_bot)` with `HBBL = 10 m` typical;
   in our 200-m-thick bottom layer this collapses to the case-2 bottom-cell-only form,
   so it is only worth the complexity once partial cells / topography are added.

Canonical values (sources: MITgcm Algorithm chapter; NEMO `nambfr`; MOM6 `MOM_input`;
Adcroft et al. 2019, JAMES):
- Linear: MITgcm `bottomDragLinear` "of order 2 × 10⁻⁴ m/s"; NEMO `rn_bfri1 = 4 × 10⁻⁴
  m/s`; MJM13 / Abernathey-Cessi idealized channels `r = 1.1 × 10⁻³ m/s` *applied to
  the bottom layer* (which is the relevant precedent for us).
- Quadratic: MOM6 `CDBOT ≈ 0.003`, `DRAG_BG_VEL = 0.1 m/s`; NEMO `rn_Cd0 = 1 × 10⁻³`;
  POP `c_d ≈ 1.0–1.225 × 10⁻³`.

---

## 2. Where to apply it — C-grid placement and partial cells

**Punchline. Drag acts on the bottom-most wet `u`- and `v`-face cells separately, using
the velocity stored at that face — not interpolated cell-centre `u·v`. Treat thin bottom
cells as a future complication; for now the 200-m-thick bottom level eliminates that
concern.**

In a z-coordinate C-grid (MITgcm `mom_fluxform`, MOM5 `bottom_friction.F90`, MOM6
`MOM_vert_friction.F90`, NEMO `dynbfr.F90`):
- `u`-face drag uses the local `u_bot[i+½, j, k_bot_u]` and the bottom-face thickness
  `dz_bot_u = ½(dz_bot[i] + dz_bot[i+1])` (or its partial-cell equivalent).
- `v`-face drag uses `v_bot[i, j+½, k_bot_v]` and `dz_bot_v` analogously.
- The bottom level index `k_bot_u(i,j)` is set by the *minimum* of the two adjacent
  cell-centred `k_bot` indices (a wet `u`-face needs both bordering tracer cells wet at
  that level). On our flat-bottom global-overturning configuration `k_bot_u = nlev − 1`
  uniformly except at land-mask boundaries, where the existing `u_mask` already zeroes
  the drag contribution.

Partial-cell / thin bottom cell handling. MOM6 `channel_drag` enhances effective drag
when `dz_bot < HBBL` by spreading the drag stress over a fixed BBL thickness rather
than the local cell. At our 5° resolution the abyssal layer is ~200 m thick everywhere
in the Drake band, so `dz_bot` is never the limiting factor and a pure bottom-cell
formulation is adequate. **Recommendation: do not implement thin-bottom-cell logic in
phase 1.** Add it only when introducing real bathymetry where a column may end mid-way
through a partial cell. When the time comes, the cleanest match is MOM6's
`channel_drag` (Hallberg & Adcroft 2009; Adcroft et al. 2019, JAMES, §2.3.3).

GM/Redi do not change this. The Eulerian `u_bot` on the C-grid face is what production
codes use (see Section 4).

---

## 3. Calibration recipe and starting value

**Punchline. Use `r_bot = 1.0 × 10⁻³ m/s` initially, applied to the bottom cell only.
With our `dz_bot ≈ 200 m`, that gives a bottom-cell decay timescale `dz_bot/r ≈ 2.3
days` — fast enough to pin `u_bot ≈ 0` on the seasonal timescale of the test, well
within established MJM13/AC14 idealized-channel practice.**

The relevant timescale is no longer the *barotropic* `H/r ≈ 42 days` (which depends on
total depth), but the *bottom-cell* `dz_bot/r`. Comparable choices in the literature:
- Munday-Hogg-Marshall (2013) and Abernathey-Cessi (2014): `r = 1.1 × 10⁻³ m/s` over
  bottom 100–200 m → `τ ≈ 1–2 days`. By design strong, to act as a sink for the
  resolved-eddy bottom flow.
- MOM6 OM4 quadratic with `C_d = 3 × 10⁻³`, `|u_bot| = 5 cm/s`, `dz_bot = 10 m`: the
  effective linear-equivalent rate is `C_d · |u| = 1.5 × 10⁻⁴ m/s`, and the cell
  timescale is `dz_bot/(C_d|u|) = 67000 s ≈ 0.8 days`. So MOM6 is *more* aggressive in
  the cell than our proposed linear value.

Starting recipe and experimental sequence:
1. **Initial run (calibration):** `r_bot = 1.0 × 10⁻³ m/s`, drop the depth-mean drag.
   Expect: `u_bot < 1 cm/s` in the deep Drake band within ~30 days; Drake transport
   converges to a positive eastward value over ~6 months.
2. **Sensitivity sweep:** repeat with `r_bot ∈ {2.5e-4, 5e-4, 1e-3, 2e-3} m/s`. We
   expect Drake transport to be *weakly sensitive* on this range because at our 5°
   resolution there is no resolved eddy field for drag to control — confirming Marshall
   et al. 2017 result that "transport increases with `r`" but only weakly above some
   threshold.
3. **Cross-check:** compare to companion `bottom_drag_literature.md` recommendation of
   `r = 8e-4 to 1.5e-3 m/s` for our setup. The recommended `1e-3` sits inside that
   band.

If `r_bot = 1e-3 m/s` proves over-damping (e.g., it suppresses any deep ACC core flow
below detectability), drop to `5e-4 m/s`. Going below `1 × 10⁻⁴ m/s` is not advised at
this resolution; the bottom-cell will become only weakly constrained again.

---

## 4. Compatibility with GM/Redi

**Punchline. Drag must use the *Eulerian* bottom-cell velocity, not the residual
(Eulerian + GM bolus). This is what MOM6, MITgcm, NEMO, and POP all do; the GM bolus
is a tracer-equation construct only.**

Standard GM (Gent & McWilliams 1990, JPO; Gent et al. 1995, JPO) introduces an
eddy-induced velocity `u* = ∂z(κ_GM · S)` that advects tracers and *thickness*, but is
*not* added to the prognostic momentum `u`. The momentum equation continues to use the
Eulerian `u`. Therefore the bottom-cell velocity that enters the drag is the prognostic
`u_bot`, exactly the quantity in `state.u[..., -1]`.

This is correct for two reasons:
- The eddy form stress that GM is parameterising is mathematically equivalent
  (Greatbatch & Lamb 1990, JPO; Loose et al. 2023, JAMES) to a vertical viscosity on
  the *Eulerian* momentum, not a modification of the bottom velocity. Adding `u*` to
  `u_bot` and then drag-stressing the sum would double-count the eddy contribution.
- The bolus velocity is divergence-free in the depth integral, so it does not
  contribute to the barotropic mode in any case.

Note however the structural concern raised in `why_westward_drake.md`: GM-on-tracers
without a Greatbatch-Lamb-style vertical momentum mixing leaves the Eulerian momentum
uncoupled to the parameterized eddy form stress. Adding bottom-cell drag is a direct
fix for the *integration-constant* problem in the column momentum balance, but it does
not substitute for GL90. If after bottom-cell drag the Drake transport is still off
(e.g., correct sign but wrong magnitude), a GL90 vertical momentum diffusivity matched
to `κ_GM` is the next intervention (Marshall & Adcroft 2010, *Ocean Mod.*; Loose et al.
2023, JAMES).

---

## 5. Sea-ice and surface-restoring interactions

**Punchline. Drag at the bottom is decoupled from the surface-restoring at the top —
no direct interaction. The only second-order pitfall is that strengthening bottom drag
slightly accelerates Ekman draining of the column, which can mildly increase
upwelling-required surface heat flux from the restoring. Quantify but do not engineer
around.**

Drag enters the bottom-cell momentum equation only. The surface-T restoring acts on
`T[..., 0]` via the `freshwater_closure` and physics modules; the surface wind stress
acts on `u[..., 0]` and `v[..., 0]`. There is no direct coupling. Two modest indirect
effects to monitor:

- **Mass / heat budget:** stronger bottom drag → weaker barotropic flow → modified
  Eulerian-mean meridional overturning → modified surface restoring flux closure to
  maintain `T_ref(y)`. Magnitude in our setup: O(W/m²), negligible for first-pass
  validation.
- **Sea ice (when activated):** sea-ice models impose a top-cell drag from ice on
  ocean. This is *additive* with bottom-cell drag and lives at a different vertical
  level; no coupling, no namelist conflict. Confirm sign convention is consistent with
  the existing `OceanSurfaceForcing.tau_x/tau_y` handling.

---

## 6. Remove or keep the existing depth-mean drag?

**Punchline. Remove the depth-mean barotropic drag entirely once bottom-cell drag is
in. Replace any residual barotropic-mode damping need with the existing
`barotropic_div_damp` and `bebt` semi-implicit PGF, which target gravity-wave
stability directly without adding spurious column-averaged dissipation.**

The current `(1 − dt·r/H)` factor in the barotropic substep is doing two jobs at once:
(a) constraining `<u>` for steady-state momentum balance, and (b) providing a tiny
stability margin against externally-forced barotropic gravity-wave overshoots.
Bottom-cell drag fully covers (a) — and does so at the right level. For (b), the
existing pipeline already has cleaner mechanisms:
- `barotropic_div_damp` directly damps the gravity-wave divergence mode.
- `bebt = 0.2` (semi-implicit barotropic PGF) is the MOM6-default form-stable scheme.
- `barotropic_time_filter = "cosine"` reduces aliasing at the Asselin cutoff.

If a small residual barotropic Rayleigh damping turns out to be needed numerically
during testing (e.g., 50-yr `eta` drift from inertia-gravity-wave residue), reintroduce
it as a *very weak* term `r_baro ≤ 1 × 10⁻⁵ m/s` distinct from `r_bot` and document
clearly that it is for numerical stability, not physics. The current `r = 1.1e-3 m/s`
applied to `<u>` is, as `bottom_drag_literature.md` notes, ~3× more dissipative than
the NEMO default; keeping it on top of bottom-cell drag would over-damp.

**Concrete plan:** in the new config, default `bottom_drag_r` continues to mean the
bottom-cell rate; add a separate `bottom_drag_mode: Literal["depth_mean", "bottom_cell"]
= "bottom_cell"` so the old behaviour can be restored for direct comparison runs.

---

## 7. Predicted impact and validation checklist

**Punchline. With `r_bot = 1 × 10⁻³ m/s` applied at the bottom cell and the depth-mean
drag removed, predicted Drake transport is **+150 to +250 Sv eastward** within the
first year. The diagnostic flip from −405 Sv westward to a positive value is the
single binary go/no-go signal.**

Expected first-order changes (50-yr global-overturning configuration):
- `u_bot` in the abyssal Drake band drops from `−7 cm/s` to magnitude `< 1 cm/s` within
  ~30 days; sign may oscillate near zero.
- `U_baro` in the band converges to `tau_x / (ρ₀ · r_eff) ≈ +2 cm/s` *in the column-
  mean sense*, but because drag is now at the bottom only, `U_baro` is determined by
  the surface Ekman + thermal-wind shear and is no longer pinned to the simple wind/drag
  prediction.
- Drake transport flips sign; `+150 to +250 Sv eastward` is within the family of values
  reported by MJM13 for similar coarse-resolution flat-bottom channels with
  bottom-cell-only drag. Higher than observed (~140 Sv) because we lack form drag, but
  *correct sign* is the qualitative win.
- Surface flow (top 100 m) strengthens slightly because the previous westward bottom
  flow was partially compensating it; expect surface zonal currents in the band to
  increase by ~1–2 cm/s.

Validation checks (in priority order):
1. **Visual u(z) profile in the Drake band:** confirm `u_bot ≈ 0` and the sign of
   `<u>` flips. Plot zonal-mean `u(y, z)` at 50 yr and compare to baseline run.
2. **Drake transport time series:** confirm flip from −405 to positive within 1 yr.
3. **Energy budget:** confirm bottom-drag dissipation `−r · u_bot²` is finite and
   small relative to wind input — the order is set by `u_bot²` which should now be
   small.
4. **Barotropic gravity-wave dispersion test:** run a short `2×fastest barotropic
   wave period` integration without forcing and confirm `eta` standard deviation
   stays bounded. If not, restore a small `r_baro` (Section 6).
5. **GM/Redi baseline regression:** ensure Eady test and 50-yr global-overturning
   tracer fields are unchanged (drag does not enter tracer equations).

---

## 8. Long-term — quadratic bottom drag and BBL-aware schemes

**Punchline. Stage the migration: linear bottom-cell → quadratic bottom-cell with
`u_bg` → MOM6-style BBL-aware quadratic. Each stage is a single localized code change;
defer the BBL scheme until partial cells / real bathymetry are introduced.**

Stage A — quadratic bottom-cell with `u_bg`. Replace `−r · u_bot` with
`−C_d · √(u_bot² + v_bot² + u_bg²) · u_bot` (and analogous for `v_bot`). The `u_bg`
floor is essential: in stagnation regions a pure quadratic drag vanishes as `u² → 0`
and momentum equations become inviscid at the bottom. MOM6 default
`DRAG_BG_VEL = 0.1 m/s` is the right starting value (Adcroft et al. 2019, §2.3.3);
combined with `C_d = 3 × 10⁻³` it gives an effective linear-equivalent rate of
`C_d · u_bg = 3 × 10⁻⁴ m/s` at low speeds, smoothly transitioning to true quadratic
at higher speeds. AD note: the `√(u² + v² + u_bg²)` keeps the term differentiable
everywhere, which is preferable to a piecewise-quadratic max-style implementation.

Stage B — MOM6-style `channel_drag` with BBL thickness. Distribute the bottom drag
stress over a BBL of thickness `max(HBBL, dz_bot)`, where `HBBL = 10 m`. When
`dz_bot < HBBL` (partial cells, thin levels above bathymetry), the drag tendency is
amplified by `dz_bot / HBBL` so total stress equals the no-slip log-layer prediction.
For our z-star coordinate with 20 levels and flat bottom this is not yet relevant; add
when partial cells are introduced. Reference: Hallberg & Adcroft (2009) and Adcroft et
al. (2019, JAMES) §2.3.3; MOM6 source `MOM_set_visc.F90` routine `set_BBL_TKE`. The
key parameter is `BBL_thickness_min` (i.e. `HBBL`).

Stage C — Jackson-style log-layer drag. Optional, only if a log-layer profile within
the BBL is desired. Adds physics but no value at our resolution; defer.

---

## Bottom line — physics recipe in 5 bullets

1. Switch from depth-mean linear drag `(1 − dt·r/H)` on `<u>` to **linear bottom-cell
   drag** acting on `u_bot`, `v_bot` independently on the C-grid `u`/`v` faces. Use
   `r_bot = 1.0 × 10⁻³ m/s` as the starting value (idealized-channel best-practice; in
   the recommended `8e-4 to 1.5e-3 m/s` band of `bottom_drag_literature.md`).
2. Implement implicitly: `u_bot ← u_bot / (1 + dt · r_bot / dz_bot)` (and same for `v`).
   Stable for any `dt`. AD-friendly. With `dz_bot ≈ 200 m` and `r = 1e-3 m/s` this is
   a 2.3-day decay timescale — strong enough to constrain the Drake-band integration
   constant.
3. Use the **Eulerian** `u_bot`, not Eulerian + GM bolus. Standard practice in MOM6,
   MITgcm, NEMO, POP. The GM bolus is a tracer-equation device and has zero depth
   integral.
4. Remove the depth-mean barotropic drag. Rely on `barotropic_div_damp`, `bebt = 0.2`,
   and the cosine time filter for barotropic gravity-wave stability. Add an
   on-by-default `bottom_drag_mode: Literal["depth_mean", "bottom_cell"] =
   "bottom_cell"` flag for back-compat regression.
5. Predicted result: Drake transport flips from −405 Sv to **+150 to +250 Sv
   eastward**; `u_bot < 1 cm/s` in the abyssal channel; surface currents strengthen by
   ~1–2 cm/s. Quadratic-with-`u_bg` and BBL-aware variants are clean follow-ons but
   not needed for the sign-correctness fix.

---

## References (selected, beyond `bottom_drag_literature.md`)

- Adcroft, A., et al. (2019). The GFDL Global Ocean and Sea Ice Model OM4.0. *JAMES*,
  11. doi:10.1029/2019MS001726. [`channel_drag`, `HBBL`, `DRAG_BG_VEL`]
- Gent, P. R., & McWilliams, J. C. (1990). Isopycnal mixing in ocean circulation
  models. *J. Phys. Oceanogr.* 20, 150–155.
- Greatbatch, R. J., & Lamb, K. G. (1990). On parameterizing vertical mixing of
  momentum in non-eddy resolving ocean models. *J. Phys. Oceanogr.* 20, 1634–1637.
- Hallberg, R., & Adcroft, A. (2009). Reconciling estimates of the free surface height
  in Lagrangian vertical-coordinate ocean models with mode-split time stepping. *Ocean
  Modelling* 29, 15–26.
- Loose, N., et al. (2023). Comparing two parameterizations for the restratification
  effect of mesoscale eddies. *JAMES* 15.
- Marshall, D. P., & Adcroft, A. J. (2010). Parameterization of ocean eddies: Potential
  vorticity mixing, energetics and Arnold's first stability theorem. *Ocean Modelling*
  32, 188–204.
- Munday, D. R., Johnson, H. L., & Marshall, D. P. (2013). Eddy saturation of
  equilibrated circumpolar currents. *J. Phys. Oceanogr.* 43, 507–532.
  doi:10.1175/JPO-D-12-095.1.
- MITgcm Algorithm chapter, "Bottom Drag":
  https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html
- NEMO `nambfr` / `zdfbfr` documentation:
  https://www.nemo-ocean.eu/doc/node70.html
- MOM6 `MOM_set_visc.F90` (channel_drag, BBL thickness):
  https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/parameterizations/vertical/MOM_set_visc.F90
- POP Reference Manual:
  https://ncar.github.io/POP/doc/build/html/reference_manual/POPRefManual.html
