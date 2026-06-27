"""Shared machinery for the GEOMETRIC calibration twin (campaign Stage 0).

Both the truth generator (``scripts/run/run_geometric_stage0.py``) and the ETKI
recovery loop (``scripts/run/run_geometric_stage0_etki.py``) build the same 2°
Veros-ACC + GEOMETRIC-closure free run, seed the same AB2 + rigid-lid carries,
step it the same way, and reduce the same observable maps — so that machinery
lives here once (``docs/dev-notes/planning/geometric_calibration_campaign.md`` §3-4).

Observation vector (§3.2): equal-weighted, per-field-normalized spatial maps of
the time-MEAN and temporal-STD of T, S, ψ and the depth-integrated EKE. ψ (not η)
is the rigid-lid interface-equivalent — the faithful ACC recipe runs the rigid-lid
barotropic solver, under which the free surface η ≡ 0. ETKI's ``R`` encodes the
per-field normalization (per-cell ``r_diag = s_field² · N_field``), so each field
contributes ~unit misfit regardless of its grid size; transport is HELD OUT.

fp64 throughout (the differentiability-audit non-negotiable for ocean calibration).
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.0
# ACC channel re-entrant band for the Drake-passage transport (the default
# -65..-45 band is OUTSIDE this idealized channel; matches run_acc_freerun.py).
ACC_DRAKE_LAT_S = -40.0
ACC_DRAKE_LAT_N = -22.0

# The ETKI observation fields (each contributes a time-mean AND a temporal-std map).
OBSERVABLE_FIELDS = ("T", "S", "psi", "eke")


# --------------------------------------------------------------------------- #
# Recipe + model construction
# --------------------------------------------------------------------------- #
def build_geometric_recipe(geom):
    """The 2° ACC free-run recipe with the GEOMETRIC closure at ``geom``."""
    from legoesm.ocean.fidelity.veros_acc_recipe import (
        build_acc_recipe, geometric_eke_config,
    )
    return build_acc_recipe(with_surface_forcing=True,
                            eke_override=geometric_eke_config(geom))


def clone_with_geometric(model, base_cfg, geom):
    """Shallow-clone ``model`` with a (possibly TRACED) ``GeometricConfig``.

    For the adjoint/identifiability path: the model is built ONCE with concrete
    params (so ``validate_geometric_config`` runs eagerly at construction), then
    this swaps in geometric params that may be jax tracers — the GEOMETRIC
    formulas accept traced field values, and no per-step re-validation occurs, so
    ``jax.grad``/``jax.jvp`` flows through them. The host-side rigid-lid island
    cache (mask-derived, param-independent) is shared via the shallow copy, so
    warm it once on the concrete model before transforming.

    ``base_cfg`` should already carry ``gm_redi.adjoint_stabilization=
    "stop_gradient_slopes"`` for horizons beyond ~1 day (frozen-coefficient
    isoneutral linearization — the dc41c0a33 gate).
    """
    import copy

    m2 = copy.copy(model)
    m2.config = base_cfg._replace(
        gm_redi=base_cfg.gm_redi._replace(
            eke=base_cfg.gm_redi.eke._replace(geometric=geom)))
    return m2


def seed_freerun_carries(model, state, cfg, grid):
    """Seed AB2 + rigid-lid carries so the jitted scan keeps a constant pytree.

    Mirrors ``run_acc_freerun._run_legoesm`` (the faithful ACC composition is
    ``outer_integrator="ab2"`` + ``barotropic_solver="rigid_lid"``, both of which
    carry prior-step state that must exist as arrays/Fields from step 0).
    """
    import jax.numpy as jnp
    from legoesm.core.field import Field

    if cfg.outer_integrator == "ab2" and state.T_incr_prev is None:
        def _z(d):
            return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                         dims=d.dims, units=d.units)
        state = state._replace(
            T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
            u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))

    if cfg.barotropic.barotropic_solver == "rigid_lid" and state.psi is None:
        rl = model._ensure_rigid_lid_data(state)
        _zV = jnp.zeros((grid.n_lat + 1, grid.n_lon + 1), dtype=state.u.data.dtype)
        _zI = jnp.zeros((rl.nisle,), dtype=state.u.data.dtype)
        state = state._replace(psi=_zV, dpsi=_zV, dpsi_prev=_zV,
                               dpsin=_zI, dpsin_prev=_zI)
    return state


def make_block_stepper(model, surface_forcing, dt):
    """Return ``(block_fn, steps_per_block)`` — a jitted 1-model-day scan block."""
    import jax

    steps_per_block = max(1, int(round(_SECONDS_PER_DAY / dt)))

    def _body(st, _):
        return model.step(st, dt, surface_forcing=surface_forcing), None

    @partial(jax.jit, static_argnames=("n",))
    def _block(st, n):
        st, _ = jax.lax.scan(_body, st, None, length=n)
        return st

    return _block, steps_per_block


# --------------------------------------------------------------------------- #
# Exact-IC restart (the Perezhogin no-equilibration member protocol)
# --------------------------------------------------------------------------- #
def restore_state(template_state, snapshot, *, model, cfg, grid):
    """Overwrite ``template_state`` with the saved truth ``snapshot`` arrays.

    The exact-IC member starts from the truth's spun-up attractor (T, S, u, v,
    eke, tke AND the AB2/rigid-lid carries), then integrates with the MEMBER's
    own GEOMETRIC parameters. The carries are seeded first so every slot exists
    (None → array would otherwise have nothing to overwrite), then every key
    present in the snapshot is replaced.
    """
    import jax.numpy as jnp
    from legoesm.core.field import Field

    state = seed_freerun_carries(model, template_state, cfg, grid)
    files = snapshot.files if hasattr(snapshot, "files") else snapshot.keys()
    for key in files:
        if key not in state._fields:
            continue
        obj = getattr(state, key)
        arr = np.asarray(snapshot[key])
        if obj is None:
            continue
        if hasattr(obj, "data"):  # Field leaf
            new = Field(data=jnp.asarray(arr, dtype=obj.data.dtype),
                        name=obj.name, dims=obj.dims, units=obj.units)
            state = state._replace(**{key: new})
        elif isinstance(obj, jnp.ndarray):  # plain-array carry
            state = state._replace(
                **{key: jnp.asarray(arr, dtype=obj.dtype)})
    return state


# --------------------------------------------------------------------------- #
# Observable maps + held-out metrics
# --------------------------------------------------------------------------- #
def sample_observable_maps(state) -> dict[str, np.ndarray]:
    """T, S, ψ, depth-integrated EKE pulled to host (fp64).

    ψ replaces η: the rigid-lid solver leaves η ≡ 0 (degenerate); ψ is the
    barotropic interface-equivalent and the field most sensitive to GEOMETRIC's
    ``kappa_u``. For the geometric closure ``state.eke`` IS the depth-integrated
    2-D budget variable.
    """
    return {
        "T": np.asarray(state.T.data, dtype=np.float64),
        "S": np.asarray(state.S.data, dtype=np.float64),
        "psi": np.asarray(state.psi, dtype=np.float64),
        "eke": np.asarray(state.eke.data, dtype=np.float64),
    }


class Accumulator:
    """Streaming sum / sum-of-squares → time-mean + temporal-std maps (fp64)."""

    def __init__(self, fields=OBSERVABLE_FIELDS):
        self._fields = tuple(fields)
        self._sum: dict[str, np.ndarray] = {}
        self._sumsq: dict[str, np.ndarray] = {}
        self._n = 0

    def add(self, sample: dict[str, np.ndarray]):
        for k in self._fields:
            x = np.asarray(sample[k], dtype=np.float64)
            if k not in self._sum:
                self._sum[k] = np.zeros_like(x)
                self._sumsq[k] = np.zeros_like(x)
            self._sum[k] += x
            self._sumsq[k] += x * x
        self._n += 1

    @property
    def n(self) -> int:
        return self._n

    def finalize(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for k in self._fields:
            mean = self._sum[k] / self._n
            var = self._sumsq[k] / self._n - mean * mean
            out[f"mean_{k}"] = mean
            out[f"std_{k}"] = np.sqrt(np.maximum(var, 0.0))
        return out


def held_out_metrics(state, z_coord, grid, *, rho_0, g_val):
    """ACC transport [Sv] + total KE [J] via the SHARED diagnostics (held-out
    early-stopping monitor; NOT part of the ETKI observation vector). ``rho_0``/
    ``g_val`` are the recipe's Veros constants so the KE monitor matches the model.
    """
    from legoesm.ocean.diagnostics_streamfunction import barotropic_streamfunction
    from legoesm.ocean.budgets import compute_energy_budget
    from legoesm.ocean.vertical import compute_layer_thickness

    u = np.asarray(state.u.data)
    mask = np.asarray(state.land_mask.data)
    h = np.asarray(compute_layer_thickness(state.eta.data, state.H_bathy.data,
                                           z_coord))
    lat = np.degrees(np.asarray(grid.lat))
    psi = np.asarray(barotropic_streamfunction(u, h, mask, grid))   # (n_lat,n_lon) [Sv]
    band = (lat >= ACC_DRAKE_LAT_S) & (lat <= ACC_DRAKE_LAT_N)
    acc_T = (float(psi[band, :].max() - psi[band, :].min())
             if band.any() else float("nan"))
    eb = compute_energy_budget(state, z_coord, grid_type="latlon", grid=grid,
                               rho_0=rho_0, g_val=g_val)
    return acc_T, float(eb.KE)


def advance_and_observe(block_fn, steps_per_block, state, n_days, *, phase="",
                        accumulator=None, sample_every_days=None,
                        held=None, z_coord=None, grid=None, rho_0=None,
                        g_val=None, spin_offset_days=0.0, log_every_days=30,
                        eke_blowup_cap=None, verbose=True):
    """Integrate ``n_days`` in 1-day blocks (blow-up-checked); optional sampling.

    When ``accumulator`` is given, every ``sample_every_days`` the observable maps
    are accumulated; if ``held`` (a dict with lists) + diagnostics args are given,
    the held-out transport/KE are recorded too.

    Blow-up handling: u AND eke are checked for non-finiteness every block, and —
    when ``eke_blowup_cap`` is set (the ETKI member path) — a finite-but-saturated
    eke (``max|eke| > cap``) ALSO raises. The geometric EKE budget is runaway-prone
    (kappa_gm ∝ ∫E dz), and a finite-huge member would pass a u-only / NaN-only
    guard and then skew the ensemble statistics (adversarial-review M1); converting
    it to a RuntimeError lets the caller drop it (member → NaN vector → ETKI drops).
    """
    import jax
    import jax.numpy as jnp

    t0 = time.time()
    last_sample = -1e30
    n_int = int(round(n_days))
    for d in range(1, n_int + 1):
        state = block_fn(state, steps_per_block)
        jax.block_until_ready(state.u.data)
        if not (bool(jnp.all(jnp.isfinite(state.u.data)))
                and bool(jnp.all(jnp.isfinite(state.eke.data)))):
            raise RuntimeError(
                f"GEOMETRIC ACC blew up (non-finite u/eke) during {phase} at day {d}")
        if eke_blowup_cap is not None:
            emax = float(jnp.max(jnp.abs(state.eke.data)))
            if emax > eke_blowup_cap:
                raise RuntimeError(
                    f"GEOMETRIC eke saturated ({emax:.3e} > cap {eke_blowup_cap:.3e}) "
                    f"during {phase} at day {d}")
        if (accumulator is not None
                and d - last_sample >= (sample_every_days or 1) - 1e-9):
            accumulator.add(sample_observable_maps(state))
            if held is not None:
                aT, KE = held_out_metrics(state, z_coord, grid,
                                          rho_0=rho_0, g_val=g_val)
                held["day"].append(float(d + spin_offset_days))
                held["acc_transport_Sv"].append(aT)
                held["total_KE_J"].append(KE)
            last_sample = d
        if verbose and (d % log_every_days == 0 or d == n_int):
            umax = float(jnp.max(jnp.abs(state.u.data)))
            emax = float(np.nanmax(np.asarray(state.eke.data)))
            print(f"  [{phase}] day {d:6d}/{n_int} | max|u|={umax:.4f} m/s | "
                  f"max EKE={emax:.3e} | {time.time()-t0:6.0f}s", flush=True)
    return state


# --------------------------------------------------------------------------- #
# Observation-vector assembly (the ETKI y / g / R)
# --------------------------------------------------------------------------- #
def _field_wet_mask(key: str, mean_map: np.ndarray, land_mask: np.ndarray):
    """Boolean wet mask broadcast to a given observable map's shape.

    T/S are 3-D (lat,lon,level) → broadcast the 2-D land mask over levels; eke is
    2-D (lat,lon) → the land mask directly; ψ lives on the (lat+1,lon+1) vorticity
    grid → use all cells (a vertex is "wet" if any adjacent cell is, which the
    streamfunction solve already encodes as zeros on the dry hull).
    """
    lm = np.asarray(land_mask) > 0.5
    if key == "psi":
        return np.ones(mean_map.shape, dtype=bool)
    if mean_map.ndim == 3:
        return np.broadcast_to(lm[:, :, None], mean_map.shape)
    return lm


def build_observation_target(truth_obs, land_mask, *, rel_floor=1e-3,
                             abs_floor=1e-30):
    """Assemble the ETKI target ``y`` + per-cell ``r_diag`` from the truth maps.

    Returns ``(y, r_diag, packer)`` where ``packer(member_obs) -> g`` flattens a
    member's observable dict in the IDENTICAL cell order. Each of the 8 maps
    (mean+std of T,S,ψ,eke) is scaled by its own truth spatial RMS ``s`` over wet
    cells and weighted so it contributes ~unit misfit (``r_diag = s²·N`` per cell);
    that is exactly the §3.2 equal-weighted, per-field-normalized design with R=I
    on the normalized maps, expressed through ETKI's diagonal R.

    The scale floor is RELATIVE (``rel_floor·|mean(map)|``), not a tiny absolute
    ``eps``: a near-spatially-UNIFORM field has ``s → 0`` and would otherwise get a
    runaway ``r_inv = 1/s²`` and hijack the whole misfit (adversarial-review M2). A
    field tripping the floor is flagged — it carries ~no spatial information and
    should probably be dropped from the observation vector.
    """
    keys = [f"{stat}_{f}" for f in OBSERVABLE_FIELDS for stat in ("mean", "std")]
    masks, scales, sizes = {}, {}, {}
    for k in keys:
        f = k.split("_", 1)[1]
        m = _field_wet_mask(f, truth_obs[k], land_mask)
        masks[k] = m
        vals = np.asarray(truth_obs[k])[m]
        # Robust spatial scale: RMS about the map mean over wet cells.
        s = float(np.sqrt(np.mean((vals - vals.mean()) ** 2)))
        floor = max(rel_floor * abs(float(vals.mean())), abs_floor)
        if s < floor:
            import warnings
            warnings.warn(
                f"observable '{k}' is near-spatially-uniform (RMS {s:.3e} < floor "
                f"{floor:.3e}); it carries ~no spatial signal — consider dropping it",
                stacklevel=2)
        scales[k] = max(s, floor)
        sizes[k] = int(m.sum())

    y_parts, r_parts = [], []
    for k in keys:
        vals = np.asarray(truth_obs[k])[masks[k]]
        y_parts.append(vals)
        r_parts.append(np.full(sizes[k], scales[k] ** 2 * max(sizes[k], 1)))
    y = np.concatenate(y_parts)
    r_diag = np.concatenate(r_parts)

    def packer(member_obs) -> np.ndarray:
        return np.concatenate([np.asarray(member_obs[k])[masks[k]] for k in keys])

    return y, r_diag, packer


def relative_param_error(theta_unconstrained, true_geom, specs):
    """L1 relative error of constrained ensemble-mean params vs the truth.

    ``theta_unconstrained`` is (n_e, n_p) in logit space; returns a dict
    {param_name: |mean(constrained) - true| / true} for the diagnostics table.
    """
    import jax.numpy as jnp
    from legoesm.training.trainable_ocean_params import constrain

    theta = jnp.asarray(theta_unconstrained)
    mean_raw = jnp.mean(theta, axis=0)
    out = {}
    for i, sp in enumerate(specs):
        val = float(constrain(mean_raw[i], sp))
        true = float(getattr(true_geom, sp.constraint.name))
        out[sp.constraint.name] = (val, true, abs(val - true) / abs(true))
    return out
