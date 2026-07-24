# Adversarial review packet — MPAS horizontal q_v del2 smoothing (round 1)

You are an **independent adversarial physics + numerics reviewer** for LegoESM, a
JAX-native, fully-differentiable Earth System Model. Find EVERY bug, sign error,
unit inconsistency, broken-gradient/retrace pattern, conservation violation,
discrete-max-principle violation, scope/NameError, dtype hazard, validate_strict
corner, CLI round-trip gap, and test-vacuity in this change. Cite file + line
numbers. If you believe a specific claimed invariant holds, say so and explain WHY
each candidate concern is not a bug (with the math). Do not rubber-stamp.

## Why this change exists

The MPAS (Voronoi/SCVT) dynamics lane historically had **NO horizontal moisture
smoothing**, while the finite-volume (cubed-sphere / lat-lon) lanes smooth q_v
every step inside their step factories (`qv_smooth_coeff = hyperdiff/2`). This
asymmetry is the confirmed third suspect behind a cell-scale column-water-vapour
(CWV) recharge/discharge "speckle" in MPAS AMIP runs. The fix adds an **opt-in**
mass-weighted conservative del2 (Laplacian) smoother on q_v, applied post-step,
eager, before the existing hard-saturation drain. Default OFF (`0.0`),
byte-identical to before.

## CLAIMED INVARIANTS (attack every one)

1. **Exact column-water-mass conservation** of the dp-weighted del2:
   `sum_c A_c * dp_c * lap_c = 0` to machine precision, for arbitrary positive
   `dp` and arbitrary `q`.
2. **Discrete max principle / positivity WITHOUT clipping**: under the setup CFL
   guard `nu*dt*g_max <= 0.5` (with `g_c = sum_e dvEdge_e/(A_c*dcEdge_e)`), the
   explicit update `q + nu*dt*lap` is a convex combination of stencil values, so
   `q>=0` is preserved. The 0.5 (vs 1.0) is claimed to give x2 headroom for the
   `dp_edge/dp_cell` weight ratio.
3. **Correct smoothing SIGN**: `+nu*lap` damps extrema (`lap<0` at an interior
   max).
4. **Constant annihilation**: `lap(const) = 0` (both plain and dp-weighted).
5. **Default 0.0 is byte-identical** (no-op path).
6. **Placement before the drain is sound**; **dtype cast** correct; **no
   retrace/JIT hazard** (eager site).
7. **CFL guard formula matches the operator actually applied**.
8. **validate_strict corners**: lane refusal (non-MPAS), bounds `[0,1e8]`, finite.
9. **CLI round-trip** `--mpas-qv-smooth-del2-m2s`.
10. **edgesOnCell padding handled (masked)**; conservation test tolerances
    non-vacuous.

---

## SOURCE 1 — new operator (`packages/core/legoesm/core/operators_voronoi.py`)

The composed primitives (pre-existing, used by the MPAS dycore):

```python
def divergence_cell_3d(u_edge_3d, mesh):
    eoc = mesh.edgesOnCell        # (maxEdges, nCells)
    sign = mesh.edgeSignOnCell    # (maxEdges, nCells)
    mask = (eoc >= 0).astype(u_edge_3d.dtype)
    eoc_safe = jnp.maximum(eoc, 0)
    u_gathered = u_edge_3d[eoc_safe]      # (maxEdges, nCells, nlev)
    dv_gathered = mesh.dvEdge[eoc_safe]   # (maxEdges, nCells)
    flux = (sign[:, :, None] * u_gathered
            * dv_gathered[:, :, None] * mask[:, :, None])
    return jnp.sum(flux, axis=0) / mesh.areaCell[:, None]

def gradient_edge_3d(phi_cell_3d, mesh):
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return (phi_cell_3d[c2] - phi_cell_3d[c1]) / mesh.dcEdge[:, None]
```

The NEW function (the change under review):

```python
def scalar_del2_cell_3d(q_cell_3d, mesh, dp_cell_3d=None):
    """Conservative Laplacian of a cell scalar for all levels (SCVT del2).
    ... (docstring: two-point flux form, conserves sum_c A_c q_c (plain) or
    sum_c A_c dp_c q_c (mass-weighted); monotone under
    nu*dt*max_c[sum_e dvEdge_e*(w_e/w_c)/(A_c*dcEdge_e)] <= 1) ...
    """
    grad = gradient_edge_3d(q_cell_3d, mesh)  # (nEdges, nlev)
    if dp_cell_3d is None:
        return divergence_cell_3d(grad, mesh)
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    dp_edge = 0.5 * (dp_cell_3d[c1] + dp_cell_3d[c2])  # (nEdges, nlev)
    div = divergence_cell_3d(grad * dp_edge, mesh)
    return div / dp_cell_3d
```

Mesh conventions: global SCVT sphere (no boundaries); every edge has exactly two
valid cells; `edgeSignOnCell` is +1/-1 so a given edge's flux enters its two
cells with OPPOSITE sign; `edgesOnCell`/`cellsOnEdge` padding is a NEGATIVE
sentinel (masked by `>= 0`); `dvEdge` (dual/Voronoi edge length), `dcEdge`
(primal/cell-center distance), `areaCell` all strictly positive.

## SOURCE 2 — config field + validate_strict (`.../driver/config.py`)

```python
# in ExperimentConfig(NamedTuple):
mpas_qv_smooth_del2_m2s: float = 0.0   # del2 diffusivity [m^2/s]; 0=off

# in validate_strict(), inside the `if d.discretization not in ('mpas', ...)` block
# that already refuses other MPAS-only land knobs:
    if self.mpas_qv_smooth_del2_m2s != 0.0:
        errors.append(
            "mpas_qv_smooth_del2_m2s is an MPAS-lane knob; "
            f"discretization={d.discretization!r} already smooths q_v "
            "in its step factories (qv_smooth_coeff) and would "
            "silently ignore it.")
# ... later, unconditional bounds:
if not (math.isfinite(self.mpas_qv_smooth_del2_m2s)
        and 0.0 <= self.mpas_qv_smooth_del2_m2s <= 1.0e8):
    errors.append(
        f"mpas_qv_smooth_del2_m2s (horizontal q_v del2 diffusivity "
        f"[m^2/s]) must be finite in [0, 1e8]; got "
        f"{self.mpas_qv_smooth_del2_m2s!r}.")
```

NOTE for the reviewer: the lane-refusal block is entered when `d.discretization`
is NOT one of the MPAS set. Confirm the guard fires for a cd-grid config and does
NOT fire for `discretization='mpas'`. (The exact enclosing condition is the same
one that already refuses `mpas_land_lapse_K_per_km` etc.)

## SOURCE 3 — driver setup guard + application (`.../driver/model_driver.py::_run_mpas`)

Setup (once, before the eager step loop; `DT` = MPAS step timestep in seconds,
`np` = numpy, geometry-only):

```python
_qv_smooth_nu = float(getattr(cfg, "mpas_qv_smooth_del2_m2s", 0.0))
if _qv_smooth_nu > 0.0:
    from legoesm.core.operators_voronoi import scalar_del2_cell_3d
    if self._voronoi_layout is not None:
        raise ValueError("... not wired for distributed Voronoi (MPI) ...")
    if self.state.tracers is None or "q_v" not in self.state.tracers:
        raise ValueError("... needs a q_v tracer (dry configuration) ...")
    _eoc_np = np.asarray(self.grid.edgesOnCell)
    _mask_np = _eoc_np >= 0
    _eoc_safe = np.maximum(_eoc_np, 0)
    _dv_np = np.asarray(self.grid.dvEdge)[_eoc_safe]
    _dc_np = np.asarray(self.grid.dcEdge)[_eoc_safe]
    _g_cell = np.sum(
        np.where(_mask_np, _dv_np / np.maximum(_dc_np, 1e-30), 0.0),
        axis=0) / np.asarray(self.grid.areaCell)
    _g_max = float(np.max(_g_cell))
    _cfl = _qv_smooth_nu * DT * _g_max
    if _cfl > 0.5:
        raise ValueError(f"... nu*dt*g_max = {_cfl:.3f} > 0.5 ... "
                         f"Max stable coeff: {0.5/(DT*_g_max):.3e} m^2/s.")
    logger.info("  MPAS q_v del2 smoothing ON: nu=%.3g m^2/s "
                "(nu*dt*g_max=%.4f of 0.5 monotone bound)", _qv_smooth_nu, _cfl)
```

Application (inside the eager `for step` loop, AFTER dynamics step + top sponge,
BEFORE the hard-saturation drain):

```python
if _qv_smooth_nu > 0.0:
    _trc_sm = self.state.tracers
    _qv_sm = _trc_sm["q_v"].data
    _ph_sm = self.sigma.pressure_at_half(self.state.p_s.data)
    _dp_sm = _ph_sm[..., 1:] - _ph_sm[..., :-1]
    _lap_sm = scalar_del2_cell_3d(_qv_sm, self.grid, dp_cell_3d=_dp_sm)
    _qv_new_sm = _qv_sm + DT * _qv_smooth_nu * _lap_sm.astype(_qv_sm.dtype)
    _new_trc_sm = dict(_trc_sm)
    _new_trc_sm["q_v"] = _trc_sm["q_v"].replace(data=_qv_new_sm)
    self.state = self.state._replace(tracers=_new_trc_sm)
```

Context: `self.sigma.pressure_at_half(p_s) = p_s[...,None] * sigma_half`, so
`_dp_sm = p_s * dsigma > 0` (sigma_half increases top->surface). `self.grid` is
the VoronoiMesh (same object passed to the operator). This is the eager Python
driver loop (mutates `self.state`), NOT a `lax.scan` body. The hard-sat drain
below it is likewise eager.

## SOURCE 4 — CLI (`scripts/run/run_amip.py`)

```python
parser.add_argument("--mpas-qv-smooth-del2-m2s", type=float, default=None,
                    dest="mpas_qv_smooth_del2_m2s", help="MPAS lane only: ...")
# in build_config_from_args:
mpas_qv_smooth_del2_m2s=(args.mpas_qv_smooth_del2_m2s
                         if args.mpas_qv_smooth_del2_m2s is not None
                         else _EXPERIMENT_DEFAULTS.mpas_qv_smooth_del2_m2s),
```

## SOURCE 5 — tests (`tests/unit/test_mpas_qv_smoothing.py`, 15 tests)

Operator: constant annihilation (<1e-18, plain+dp); plain area-integral
conservation (`|sum A*lap| < 1e-12*sum A*|lap|`); dp-weighted water-mass
conservation (`|sum A*dp*lap| < 1e-12*sum A*dp*|lap|`); smoothing sign
(`lap[spike]<0`); explicit-step monotone+positive at `nu=0.5/(dt*g_max)`
(`min(q_new)>=min(q)-1e-15`, `max(q_new)<=max(q)+1e-15`, `min(q_new)>=-1e-15`);
step water-mass conservation (`m1==approx(m0, rel=1e-13)`); checkerboard variance
decay (`var(q_new)<0.9*var(q)` at `nu=0.4/(dt*g_max)`); autodiff (finite,
nonzero grad). validate_strict: MPAS accepts 2e5 and zero; cd-grid refuses
(`match="MPAS-lane"`); bounds parametrized over `[-1, nan, inf, 2e8]`. The test's
`_g_max` replicates the driver formula against the SAME level-2 mesh (162 cells)
the operator runs on, so guard and operator cannot drift.

---

## My static analysis (challenge it)

- **Units**: nu [m^2/s], DT [s], lap [q]/m^2 → `DT*nu*lap` [q], added to q [kg/kg].
  Consistent. g_c [1/m^2], `nu*DT*g_max` dimensionless. dp [Pa] cancels in the
  weighted lap (`div(dp*grad)/dp`). OK.
- **Conservation**: plain `sum_c A_c lap_c = sum_e (sign_{c1}+sign_{c2}) grad_e
  dvEdge_e = 0` (opposite signs telescope). Weighted: `sum_c A_c dp_c lap_c =
  sum_c A_c div(dp_edge*grad)_c = sum_e (±) (dp_edge*grad)_e dvEdge_e = 0` (the
  `dp_c` cancels the `1/dp_c` before area-weighting). Exact, dp held fixed during
  the smoothing sub-step. OK.
- **Sign**: at an interior max, every neighbor `q_n < q_c`; each edge term
  `sign_ce*(q[c2]-q[c1])*positive = (q_n-q_c)*positive < 0`, so `lap<0`,
  `+nu*lap` decreases the max. OK.
- **Monotonicity**: `q_new_c = q_c(1 - nu*dt*S_c) + nu*dt*sum_e w_e q_n`,
  `w_e = dp_edge/(dp_c) * dvEdge/(dcEdge*A_c) > 0`, `S_c = sum_e w_e`. Convex iff
  `nu*dt*S_c <= 1`. `S_c <= (max dp_edge/dp_c) * g_c`, and
  `dp_edge/dp_c = 0.5(1+dp_n/dp_c) <= 2` iff `dp_n/dp_c <= 3`. Guard enforces
  `nu*dt*g_max <= 0.5`, so `nu*dt*S_c <= 0.5*2 = 1` provided `dp_n/dp_c <= 3`
  everywhere. In sigma coords `dp = p_s*dsigma`, so `dp_n/dp_c = p_s_n/p_s_c`;
  at 240 km cell-mean surface pressure this stays < ~2. **Potential soft spot**:
  the x2 headroom silently assumes `dp_n/dp_c <= 3`; extreme cell-to-cell p_s
  contrast (very high-res over Everest-class orography) could pinch it.
- **Differentiability**: pure gather/scatter + arithmetic; no clip/where on q; the
  only `where`/`maximum` are in the geometry mask (static mesh, no gradient
  needed). `jax.grad` test present. OK.
- **Retrace/JIT**: eager site (like the drain); `scalar_del2_cell_3d` is pure JAX
  dispatched eagerly each step — no jit boundary crossed, no retrace. Function-
  local import bound before use (setup runs when nu>0, before the loop). OK.
- **Reuse note (not a bug)**: `_dp_sm` could use `self.sigma.layer_thickness_dp`
  (= `dsigma*p_s`); diffing `pressure_at_half` is identical for pure sigma and
  more general for hybrid coords, and matches the diagnostics' convention.

## Explicit ask

Attack all 10 invariants and my static analysis. Priorities: (a) does the
mass-weighted conservation hold for the ACTUAL applied step (dp from p_s, not a
frozen constant)? (b) is the 0.5 CFL headroom truly sufficient, or can
`dp_edge/dp_c` exceed 2 in a physically-reachable config, breaking positivity?
(c) any scope/NameError/dtype/shape hazard in the driver blocks (p_s shape,
tracers dict copy, `.replace`)? (d) does the lane-refusal guard actually fire for
non-MPAS and not for MPAS? (e) are any conservation-test tolerances vacuous
(e.g. would a WRONG operator also pass)? (f) division by `dp_cell_3d` with no eps
floor — reachable zero? Cite line numbers. If no bug for a given item, say so and
justify.
