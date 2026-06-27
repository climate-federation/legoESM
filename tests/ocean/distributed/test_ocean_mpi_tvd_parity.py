"""Gathered-field MPI-vs-serial parity for the lat-lon C-grid ocean step.

Codex P2 review prescription (bench_ocean_mpi_scaling review, MAJOR 1):
the np=2 global conservation gate CANNOT detect partition-cut
corruption whose flux divergence still telescopes — exactly the failure
mode of the pre-fix ``tvd_to_v_points`` (second-neighbour limiter
ratios built from rank-local rows, cut faces filled one face row off).
The decisive check is gathered-FIELD parity:

    global state -> serial step (reference)
    global state -> scatter -> band MPI step -> gather
    compare T, S, eta, u, v   (f64, rtol = atol = 1e-10)

with ``tracer_advection="tvd"`` and NONZERO MERIDIONAL TRANSPORT at the
partition cut (seeded v + an eta gravity-wave perturbation), so the
meridional TVD limiter path is genuinely exercised across the cut.

Run (np=1 is the degenerate single-rank-band case, np=2 puts the cut at
the equator of the symmetric test grid):

    mpirun -np 1 python -m pytest tests/ocean/distributed/test_ocean_mpi_tvd_parity.py -v
    mpirun -np 2 python -m pytest tests/ocean/distributed/test_ocean_mpi_tvd_parity.py -v

Deadlock discipline (the pattern that bit P1): the serial reference is
computed on EVERY rank BEFORE ``initialize_distributed_latlon`` arms
the MPI halo backend.  A rank-0-only serial step traced after arming
embeds sendrecv/allreduce collectives that no other rank matches.
"""

from __future__ import annotations

import jax

# f64 BEFORE any array is built: the 1e-10 parity tolerance is
# meaningless in float32.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

# Guard pattern of this directory (test_ocean_mpi_conservation.py):
# skip the module when the MPI stack is absent or the size is invalid.
mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

# Small parity grid: cheap, symmetric (np=2 cut lands exactly on the
# equator), with land caps poleward of the default 80 deg threshold so
# masks are exercised.
N_LAT, N_LON, N_LEV = 16, 32, 4
DT = 600.0
# Two steps: step 1 develops in-step transport from the IC; step 2
# advects with fully nonzero v at the cut (codex minimum is one step —
# two is strictly stronger and still <1 s at this size).
N_STEPS = 2
RTOL = 1e-10
ATOL = 1e-10
PARITY_FIELDS = ("T", "S", "eta", "u", "v")

# pad_halo at width 2 (the TVD cell pad) needs >=2 lat rows per rank.
_MIN_ROWS_PER_RANK = 2
_n_procs = MPI.COMM_WORLD.Get_size()
if _n_procs > 1 and (N_LAT // _n_procs) < _MIN_ROWS_PER_RANK:
    pytest.skip(
        f"MPI size {_n_procs} gives {N_LAT // _n_procs} lat rows/rank on "
        f"the n_lat={N_LAT} parity grid (need >={_MIN_ROWS_PER_RANK} for "
        f"the halo=2 TVD cell pad). Use np in 1..{N_LAT // 2}.",
        allow_module_level=True,
    )

from legoesm.core.flux_limiters import van_leer_limiter
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.grids.operators_latlon_cgrid import (
    compute_vertex_mask,
    divergence_cgrid,
    gradient_curl_to_v,
    is_tripolar,
    pad_ns_scalar,
    pad_ns_zero,
)
from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    tvd_to_v_points,
    upwind_to_v_points,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _reset_halo_backend():
    """Restore the local backend after each test (mirrors the autouse
    fixture in test_ocean_mpi_conservation.py)."""
    yield
    set_halo_backend("local")


# ===========================================================================
# Helpers
# ===========================================================================


def _perturbed_global_state(grid, z_coord):
    """Deterministic, mask-aware perturbation of the global rest state.

    Built identically on every rank (no comm), BEFORE the MPI backend is
    armed, so band slices stay globally consistent.  Seeds:

    * eta: gravity-wave exciter (same pattern as the bench);
    * T:   horizontal structure so tracer advection moves real signal;
    * v:   NONZERO MERIDIONAL TRANSPORT, maximal at mid-band — for np=2
      the partition cut sits at the equator where sin(pi*i/n_faces)=1,
      so the TVD second-neighbour path is exercised exactly at the cut.
    """
    state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    mask = state.land_mask.data
    eta_pert = state.eta.data + 0.05 * mask * (
        jnp.sin(3.0 * grid.lon2d) * jnp.cos(2.0 * grid.lat2d)
    )
    T_pert = state.T.data + 0.5 * mask[..., jnp.newaxis] * (
        jnp.cos(grid.lon2d) * jnp.sin(2.0 * grid.lat2d)
    )[..., jnp.newaxis]
    v_data = state.v.data                      # (n_lat+1, n_lon, nlev)
    n_faces, n_lon = v_data.shape[0], v_data.shape[1]
    i_face = jnp.arange(n_faces, dtype=v_data.dtype)[:, None, None]
    j_cell = jnp.arange(n_lon, dtype=v_data.dtype)[None, :, None]
    v_pert = v_data + 0.05 * state.v_mask.data[..., jnp.newaxis] * (
        jnp.sin(jnp.pi * i_face / (n_faces - 1))
        * jnp.cos(2.0 * jnp.pi * j_cell / n_lon)
    )
    return state._replace(
        eta=state.eta.replace(data=eta_pert),
        T=state.T.replace(data=T_pert),
        v=state.v.replace(data=v_pert),
    )


def _tvd_to_v_points_legacy(f, mass_flux_v, grid=None,
                            limiter_fn=van_leer_limiter):
    """Pre-fix serial implementation, copied VERBATIM (non-fold paths).

    Reference for the bit-identity check: the cell-pad-first rewrite
    must reproduce this exactly on the local backend, including the
    edge clamps (``f_south2 := f[:1]``, ``f_north2 := f[-1:]``) and the
    zero pole faces from ``pad_ns_zero`` / ``pad_ns_scalar``.
    """
    eps = 1e-30
    f_south = f[:-1]
    f_north = f[1:]
    f_south2 = jnp.concatenate([f[:1], f[:-2]], axis=0)
    f_north2 = jnp.concatenate([f[2:], f[-1:]], axis=0)
    delta_pos = f_north - f_south
    r_pos = (f_south - f_south2) / jnp.where(
        jnp.abs(delta_pos) > eps, delta_pos, eps)
    delta_neg = f_south - f_north
    r_neg = (f_north2 - f_north) / jnp.where(
        jnp.abs(delta_neg) > eps, delta_neg, eps)
    f_pos = f_south + 0.5 * limiter_fn(r_pos) * delta_pos
    f_neg = f_north + 0.5 * limiter_fn(r_neg) * delta_neg
    f_tvd = jnp.where(mass_flux_v[1:-1] > 0, f_pos, f_neg)
    if grid is not None:
        return pad_ns_scalar(f_tvd, grid)
    return pad_ns_zero(f_tvd)


# ===========================================================================
# Serial bit-identity (codex requirement (a)) — runs at any np, local
# backend only (no MPI arming).
# ===========================================================================


class TestTVDToVPointsSerialBitIdentity:
    """Cell-pad-first ``tvd_to_v_points`` is BIT-identical to the legacy
    serial implementation on the local backend (poles at both ends)."""

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("with_grid", [False, True])
    def test_bit_identity(self, ndim, with_grid):
        rng = np.random.default_rng(20260610)
        n_lat, n_lon, nlev = 7, 8, 3  # odd n_lat: asymmetric edge cases
        shape_f = (n_lat, n_lon) if ndim == 2 else (n_lat, n_lon, nlev)
        shape_v = (n_lat + 1,) + shape_f[1:]
        f = jnp.asarray(rng.standard_normal(shape_f))
        # Sign-mixed mass flux (exercises both upwind branches), with
        # exact zeros sprinkled in (the strict ``> 0`` branch edge).
        mf = rng.standard_normal(shape_v)
        mf[::3] = 0.0
        mass_flux_v = jnp.asarray(mf)
        grid = (
            ensure_geometry(create_latlon_grid(n_lat=n_lat, n_lon=n_lon))
            if with_grid else None
        )
        new = tvd_to_v_points(f, mass_flux_v, grid=grid)
        legacy = _tvd_to_v_points_legacy(f, mass_flux_v, grid=grid)
        np.testing.assert_array_equal(
            np.asarray(new), np.asarray(legacy),
            err_msg=(
                f"tvd_to_v_points (cell-pad-first) is not bit-identical "
                f"to the legacy serial path (ndim={ndim}, "
                f"with_grid={with_grid})"
            ),
        )


# ===========================================================================
# Cut-row operator fixes: legacy serial copies (verbatim pre-fix
# implementations; the backend-dispatched pads reduce to ``jnp.pad`` on
# the local backend, which is what these copies inline).
# ===========================================================================


def _divergence_cgrid_legacy(u, v, grid, *, u_mask=None, v_mask=None):
    """Pre-fix serial ``divergence_cgrid``, copied VERBATIM (regular /
    Mercator path; the tripolar branch is untouched by the fix)."""
    u_eff = u
    v_eff = v
    if u_mask is not None:
        um = u_mask[..., jnp.newaxis] if u.ndim == 3 and u_mask.ndim == 2 else u_mask
        u_eff = u * um
    if v_mask is not None:
        vm = v_mask[..., jnp.newaxis] if v.ndim == 3 and v_mask.ndim == 2 else v_mask
        v_eff = v * vm

    face_dy = grid.dy * 0.5  # (n_lat,)
    if u.ndim == 2:
        u_east = u_eff[:, 1:]
        u_west = u_eff[:, :-1]
        face_dy = face_dy[:, jnp.newaxis]
    else:
        u_east = u_eff[:, 1:, :]
        u_west = u_eff[:, :-1, :]
        face_dy = face_dy[:, jnp.newaxis, jnp.newaxis]
    net_zonal = (u_east - u_west) * face_dy

    lat = grid.lat
    lat_interior = 0.5 * (lat[:-1] + lat[1:])
    cos_lat_v_interior = jnp.cos(lat_interior)
    # Local-backend pad_with_pole_bc_lat == zero pad at both ends.
    cos_lat_v = jnp.pad(cos_lat_v_interior, (1, 1))
    face_dx = grid.radius * cos_lat_v * grid.dlon  # (n_lat+1,)

    if v.ndim == 2:
        v_north = v_eff[1:]
        v_south = v_eff[:-1]
        net_merid = (v_north * face_dx[1:, jnp.newaxis]
                     - v_south * face_dx[:-1, jnp.newaxis])
    else:
        v_north = v_eff[1:, :, :]
        v_south = v_eff[:-1, :, :]
        net_merid = (v_north * face_dx[1:, jnp.newaxis, jnp.newaxis]
                     - v_south * face_dx[:-1, jnp.newaxis, jnp.newaxis])

    area = grid.area
    if u.ndim == 3:
        area = area[..., jnp.newaxis]
    return (net_zonal + net_merid) / area


def _gradient_curl_to_v_legacy(zeta, grid):
    """Pre-fix serial ``gradient_curl_to_v``, copied VERBATIM (both the
    regular and the tripolar/fold branch: interior rows then
    ``pad_ns_scalar`` refill)."""
    if is_tripolar(grid):
        dx_v_int = grid.dx_v[1:-1]  # (n_lat-1, n_lon) — full 2D
    else:
        R = grid.radius
        dlon = grid.dlon
        lat = grid.lat
        lat_interior = 0.5 * (lat[:-1] + lat[1:])
        cos_lat_v_int = jnp.cos(lat_interior)
        dx_v_int = R * cos_lat_v_int * dlon
    dzeta = zeta[:, 1:] - zeta[:, :-1]
    dzeta_int = dzeta[1:-1]
    if dx_v_int.ndim == 2:
        if zeta.ndim == 3:
            grad_int = dzeta_int / dx_v_int[:, :, jnp.newaxis]
        else:
            grad_int = dzeta_int / dx_v_int
    else:
        if zeta.ndim == 2:
            grad_int = dzeta_int / dx_v_int[:, jnp.newaxis]
        else:
            grad_int = dzeta_int / dx_v_int[:, jnp.newaxis, jnp.newaxis]
    return pad_ns_scalar(grad_int, grid)


def _upwind_to_v_points_legacy(f, mass_flux_v, grid=None):
    """Pre-fix serial ``upwind_to_v_points``, copied VERBATIM."""
    f_south = f[:-1]
    f_north = f[1:]
    mf_interior = mass_flux_v[1:-1]
    f_upwind = jnp.where(mf_interior > 0, f_south, f_north)
    if grid is not None:
        return pad_ns_scalar(f_upwind, grid)
    return pad_ns_zero(f_upwind)


def _compute_vertex_mask_legacy(land_mask, grid=None):
    """Pre-fix serial ``compute_vertex_mask``, copied VERBATIM."""
    m = land_mask
    m_sw = jnp.roll(m, 1, axis=1)
    interior = m[:-1] * m[1:] * m_sw[:-1] * m_sw[1:]
    interior_full = jnp.concatenate([interior, interior[:, 0:1]], axis=1)
    fold = getattr(grid, "fold", None) if grid is not None else None
    if fold is not None and fold.is_active and fold.fold_j >= 0:
        return pad_ns_scalar(interior_full, grid)
    return pad_ns_zero(interior_full)


class TestCutRowOperatorsSerialBitIdentity:
    """The cut-row fixes (cell-pad-first metric in ``divergence_cgrid``,
    all-row ``gradient_curl_to_v``, cell-pad-first ``upwind_to_v_points``
    and ``compute_vertex_mask``) are BIT-identical to the legacy serial
    implementations on the local backend (poles at both ends).

    ``grid_kind="tripole"`` (synthetic tripole: regular metrics, ACTIVE
    fold) exercises the fold branch of the legacy ``pad_ns_scalar``
    refill vs the explicit fold-row rebuild in the fixed operators.
    ``divergence_cgrid``'s tripolar branch (``face_dx = grid.dx_v``) is
    untouched by the fix, so its bit-identity case stays regular-only.
    """

    N_LAT, N_LON, NLEV = 7, 8, 3  # odd n_lat: asymmetric edge cases

    def _grid(self, kind):
        if kind == "tripole":
            return create_synthetic_tripole(self.N_LAT, self.N_LON)
        grid = create_latlon_grid(n_lat=self.N_LAT, n_lon=self.N_LON)
        return ensure_geometry(grid) if kind == "geometry" else grid

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("with_masks", [False, True])
    @pytest.mark.parametrize("grid_kind", ["latlon", "geometry"])
    def test_divergence_cgrid(self, ndim, with_masks, grid_kind):
        rng = np.random.default_rng(20260610)
        n_lat, n_lon, nlev = self.N_LAT, self.N_LON, self.NLEV
        shape_u = (n_lat, n_lon + 1) if ndim == 2 else (n_lat, n_lon + 1, nlev)
        shape_v = (n_lat + 1, n_lon) if ndim == 2 else (n_lat + 1, n_lon, nlev)
        u = jnp.asarray(rng.standard_normal(shape_u))
        v = jnp.asarray(rng.standard_normal(shape_v))
        masks = {}
        if with_masks:
            masks = dict(
                u_mask=jnp.asarray(
                    rng.integers(0, 2, (n_lat, n_lon + 1)).astype(np.float64)),
                v_mask=jnp.asarray(
                    rng.integers(0, 2, (n_lat + 1, n_lon)).astype(np.float64)),
            )
        grid = self._grid(grid_kind)
        new = divergence_cgrid(u, v, grid, **masks)
        legacy = _divergence_cgrid_legacy(u, v, grid, **masks)
        np.testing.assert_array_equal(
            np.asarray(new), np.asarray(legacy),
            err_msg=(
                f"divergence_cgrid (cell-pad-first v-face metric) is not "
                f"bit-identical to the legacy serial path (ndim={ndim}, "
                f"with_masks={with_masks}, grid_kind={grid_kind})"
            ),
        )

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("grid_kind", ["latlon", "geometry", "tripole"])
    def test_gradient_curl_to_v(self, ndim, grid_kind):
        rng = np.random.default_rng(20260610)
        n_lat, n_lon, nlev = self.N_LAT, self.N_LON, self.NLEV
        shape_z = ((n_lat + 1, n_lon + 1) if ndim == 2
                   else (n_lat + 1, n_lon + 1, nlev))
        zeta = jnp.asarray(rng.standard_normal(shape_z))
        grid = self._grid(grid_kind)
        new = gradient_curl_to_v(zeta, grid)
        legacy = _gradient_curl_to_v_legacy(zeta, grid)
        np.testing.assert_array_equal(
            np.asarray(new), np.asarray(legacy),
            err_msg=(
                f"gradient_curl_to_v (all-row + cell-pad-first metric) is "
                f"not bit-identical to the legacy serial path (ndim={ndim}, "
                f"grid_kind={grid_kind})"
            ),
        )

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("grid_kind", [None, "geometry", "tripole"])
    def test_upwind_to_v_points(self, ndim, grid_kind):
        rng = np.random.default_rng(20260610)
        n_lat, n_lon, nlev = self.N_LAT, self.N_LON, self.NLEV
        shape_f = (n_lat, n_lon) if ndim == 2 else (n_lat, n_lon, nlev)
        shape_v = (n_lat + 1,) + shape_f[1:]
        f = jnp.asarray(rng.standard_normal(shape_f))
        # Sign-mixed flux with exact zeros (the strict ``> 0`` edge).
        mf = rng.standard_normal(shape_v)
        mf[::3] = 0.0
        mass_flux_v = jnp.asarray(mf)
        grid = self._grid(grid_kind) if grid_kind else None
        new = upwind_to_v_points(f, mass_flux_v, grid=grid)
        legacy = _upwind_to_v_points_legacy(f, mass_flux_v, grid=grid)
        np.testing.assert_array_equal(
            np.asarray(new), np.asarray(legacy),
            err_msg=(
                f"upwind_to_v_points (cell-pad-first) is not bit-identical "
                f"to the legacy serial path (ndim={ndim}, "
                f"grid_kind={grid_kind})"
            ),
        )

    @pytest.mark.parametrize("grid_kind", [None, "latlon", "geometry", "tripole"])
    def test_compute_vertex_mask(self, grid_kind):
        rng = np.random.default_rng(20260610)
        mask = jnp.asarray(
            rng.integers(0, 2, (self.N_LAT, self.N_LON)).astype(np.float64))
        grid = self._grid(grid_kind) if grid_kind else None
        new = compute_vertex_mask(mask, grid=grid)
        legacy = _compute_vertex_mask_legacy(mask, grid=grid)
        np.testing.assert_array_equal(
            np.asarray(new), np.asarray(legacy),
            err_msg=(
                f"compute_vertex_mask (cell-pad-first) is not bit-identical "
                f"to the legacy serial path (grid_kind={grid_kind})"
            ),
        )


# ===========================================================================
# Gathered-field step parity (codex-prescribed regression for the
# partition-cut corruption) — np=1 degenerate + np>=2 real cuts.
# ===========================================================================


class TestOceanMPIStepParity:

    def test_step_parity_gathered_vs_serial(self):
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()

        grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
        z_coord = create_ocean_z_star(n_levels=N_LEV)
        # tvd is the production default; explicit_substep is the only
        # MPI-safe barotropic solver (fixed collective schedule).
        config = LatLonCGridOceanConfig.from_flat(
            tracer_advection="tvd",
            barotropic_solver="explicit_substep",
        )
        state_global = _perturbed_global_state(grid, z_coord)

        # --- Serial reference: EVERY rank, local backend, BEFORE the
        # MPI halo backend is armed (rank0-only-after-arming deadlocks;
        # see module docstring). ---
        serial_model = LatLonCGridOceanModel(grid, z_coord, config)
        ref = state_global
        for _ in range(N_STEPS):
            ref = serial_model.step(ref, DT)
        ref_np = {
            name: np.asarray(getattr(ref, name).data)
            for name in PARITY_FIELDS
        }

        # --- Arm the band layout + MPI halo backend, then build the
        # model ON the band geometry (never the global grid) — same
        # construction order as scripts/bench/bench_ocean_mpi_scaling.py.
        from legoesm.parallel.distributed import initialize_distributed_latlon
        from legoesm.parallel.latlon_mpi import (
            gather_state_latlon_cgrid_ocean,
            scatter_state_latlon_cgrid_ocean,
            slice_cgrid_geometry_to_band,
            slice_zcoord_to_band,
        )

        layout = initialize_distributed_latlon(
            global_n_lat=N_LAT, global_n_lon=N_LON,
        )
        band_geom = slice_cgrid_geometry_to_band(ensure_geometry(grid), layout)
        z_band = slice_zcoord_to_band(z_coord, layout)
        band_model = LatLonCGridOceanModel(band_geom, z_band, config)

        local = scatter_state_latlon_cgrid_ocean(state_global, layout)
        for _ in range(N_STEPS):
            local = band_model.step(local, DT)
        jax.block_until_ready(jax.tree.leaves(local))

        gathered = gather_state_latlon_cgrid_ocean(local, layout)

        # Collective gather above is the last MPI call: non-root ranks
        # may return now (their pytest process passes; any rank-0
        # assert failure fails the mpirun exit code).
        if rank != 0:
            assert gathered is None
            return

        assert gathered is not None
        for name in PARITY_FIELDS:
            got = np.asarray(getattr(gathered, name).data)
            np.testing.assert_allclose(
                got, ref_np[name], rtol=RTOL, atol=ATOL,
                err_msg=(
                    f"np={n_ranks} gathered '{name}' diverged from the "
                    f"serial reference after {N_STEPS} step(s) "
                    f"(tracer_advection='tvd'): partition-cut "
                    f"corruption that global conservation cannot see."
                ),
            )
