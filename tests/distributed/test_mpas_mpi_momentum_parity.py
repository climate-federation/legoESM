"""MPAS cell-partition MPI: edge-momentum parity against the serial step.

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_mpi_momentum_parity.py

Why this exists
---------------
2026-09-24: the same code, the same deck and the same cold start gave a
day-1 ocean boundary-layer wind of 8.17 m/s on ONE GPU and 6.32 m/s on
FOUR (-22.6 %), momentum only, smooth and global (78 % of cells, neighbour
correlation 0.975), with T / q / p_s agreeing to 0.02 K / 1 % / 0.5 Pa.
``test_mpas_mpi_amip`` did not see it because it runs with
``turbulence="none"`` and a loose 1 m/s absolute bound.

This test integrates ONE step per lane and compares the owned-edge wind
(and the owned-cell T / p_s as a control) between the serial global model
and the MPI model at the same global indices, separately for

* ``turbulence="none"``  -> dynamics-only step (physics adds no momentum),
* ``turbulence="louis"`` -> the physics momentum path (surface drag + PBL
  mixing on the Perot cell reconstruction, projected back to edges),

so a momentum-only defect is attributed to the lane that carries it.
The residual budget is the mass-fixer allreduce reduction order, i.e.
machine epsilon scaled — anything above 1e-6 relative RMS is a defect.
"""
from __future__ import annotations

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402

RES, NLEV, DT = 3, 20, 300.0  # level 3 = 642 cells
REL_RMS_TOL = 1e-6
# A wind-bearing initial state: the analytical IC is at rest, so an edge
# wind of 1e-2 m/s makes a relative measure meaningless.  ERA5 on this
# mesh is the production cold start; skip when the snapshot is absent.
ERA5_IC = os.environ.get(
    "LEGOESM_ERA5_IC", "/scratch/b/b309178/era5_ic_1979-01-01.zarr")


# Dynamics rows of the production 36-level deck that a default DycoreConfig
# does not carry (each is a candidate lane difference on its own).
DECK_DYN = dict(
    a_h_scale=0.375, mpas_vert_advection_scheme="van_leer",
    mpas_sponge_del2_top_layers=3, mpas_sponge_del2_top_factor=8.0,
    mpas_qv_smooth_del4_m4s=3.6421e14, mpas_conservative_tracer_clamp=True,
)
VARIANTS = {
    "none": dict(turbulence="none"),
    "louis": dict(turbulence="louis"),
    "deck": dict(turbulence="louis", **DECK_DYN),
    # Production-deck physics, one component at a time (amip_sundqvist_l36).
    "clubb": dict(turbulence="clubb", clubb_prognostic=True,
                  surface_bulk_scheme="coare3", surface_gustiness_zi=300.0),
    "gwd": dict(turbulence="none", gravity_wave_drag="mcfarlane+e3sm_cam",
                e3sm_cam_source="background", e3sm_cam_pgwv=32,
                e3sm_cam_effgw=1.0, e3sm_cam_taubgnd=0.7e-3, e3sm_cam_c0=30.0,
                e3sm_cam_launch_p=50000.0, e3sm_cam_latitude_taper=False),
    "bechtold": dict(turbulence="none", convection="bechtold",
                     bechtold_M_b_max=0.05),
}


def _build(distributed: bool, variant: str, n_steps: int):
    kw = dict(VARIANTS[variant])
    dyn = {k: kw.pop(k) for k in list(kw) if k in DycoreConfig._fields}
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=RES, nlev=NLEV,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(discretization="mpas", dt=DT, **dyn),
        output=OutputConfig(output_dir="", diag_days=0),
        days=n_steps * DT / 86400.0, dataset="analytical", radiation="gray",
        precision="fp64",
        distributed=distributed, ic="era5", ic_path=ERA5_IC,
        **{"convection": "none", **kw},
    )
    d = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
    d.setup()
    return d


def _rel_rms(a, b):
    num = np.sqrt(np.mean((a - b) ** 2))
    den = np.sqrt(np.mean(b ** 2)) + 1e-30
    return float(num / den)


def _ring_distance_to_halo(part, mesh):
    """Owned-cell graph distance (rings) to the nearest halo cell."""
    n_local = part.n_local_cells
    n_owned = part.n_owned_cells
    coc = np.asarray(mesh.cellsOnCell)          # (maxEdges, n_local), -1 padded
    dist = np.full(n_local, 99, dtype=int)
    dist[n_owned:] = 0
    frontier = np.arange(n_owned, n_local)
    ring = 0
    while frontier.size:
        ring += 1
        nb = coc[:, frontier].ravel()
        nb = nb[nb >= 0]
        nb = np.unique(nb[dist[nb] > ring])
        dist[nb] = ring
        frontier = nb
    return dist


NSTEPS = tuple(int(x) for x in os.environ.get("LEGOESM_PARITY_STEPS", "1").split(","))


@pytest.mark.skipif(not os.path.isdir(ERA5_IC), reason="ERA5 IC snapshot absent")
@pytest.mark.parametrize("n_steps", NSTEPS)
@pytest.mark.parametrize("turbulence", list(VARIANTS))
def test_mpas_mpi_one_step_momentum_matches_serial(turbulence, n_steps):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ref = _build(False, turbulence, n_steps)
    assert int(ref.config.days * 86400.0 / DT) == n_steps, "config must give n_steps"
    u0_ref = np.asarray(ref.state.u.data)
    T0_ref = np.asarray(ref.state.T.data)
    ic_ref_state = ref.state
    ref.run()
    u_ref = np.asarray(ref.state.u.data)
    T_ref = np.asarray(ref.state.T.data)
    ps_ref = np.asarray(ref.state.p_s.data)

    # Cross-rank consistency of the SERIAL reference: every rank builds its
    # own; they must agree to the bit or the comparison below is meaningless.
    print(f"[{turbulence} x{n_steps}] rank{rank} serial ref sums: "
          f"u0={float(np.sum(u0_ref)):.12e} T0={float(np.sum(T0_ref)):.12e} "
          f"u={float(np.sum(u_ref)):.12e} T={float(np.sum(T_ref)):.12e} "
          f"ps={float(np.sum(ps_ref)):.12e}")

    d = _build(True, turbulence, n_steps)
    assert d._voronoi_layout is not None
    _lm = d._voronoi_layout.local_mesh
    _gm = ref.grid
    print(f"[{turbulence} x{n_steps}] rank{rank} dx_min local={float(jnp.min(_lm.dcEdge)):.3f} "
          f"global={float(jnp.min(_gm.dcEdge)):.3f}; min areaCell local={float(jnp.min(_lm.areaCell)):.6e} "
          f"global={float(jnp.min(_gm.areaCell)):.6e}")
    for _k in ("nu_del2", "nu_del4", "nu_div4", "nu_del2_sponge", "div_damp"):
        _a = getattr(d.model.config, _k, None); _b = getattr(ref.model.config, _k, None)
        if _a is not None or _b is not None:
            print(f"[{turbulence} x{n_steps}] rank{rank} config.{_k}: mpi={_a} serial={_b}")
    part = d._voronoi_layout.partition
    n_owned = part.n_owned_cells
    owned_cells = np.asarray(part.local_cells[:n_owned])
    owned_edges = np.asarray(part.local_edges[:part.n_owned_edges])

    # Instrument check: the two lanes must START from the same state, or a
    # step-parity residual is an IC residual in disguise.  The ERA5 IC built
    # on the rank-local mesh is NOT bit-identical to the global one in T
    # (measured 2026-09-24: up to 0.05 K; u is exact) -- a separate lane
    # difference, reported here and neutralised by scattering the serial
    # IC into the MPI state so what follows tests the STEP alone.
    u0_mpi = np.asarray(d.state.u.data[:part.n_owned_edges])
    T0_mpi = np.asarray(d.state.T.data[:n_owned])
    ic_du = float(np.max(np.abs(u0_mpi - u0_ref[owned_edges])))
    ic_dT = np.abs(T0_mpi - T0_ref[owned_cells])
    n_bad = int(np.sum(np.max(ic_dT, axis=1) > 1e-9))
    if rank == 0:
        print(f"\n[{turbulence} x{n_steps}] IC lane mismatch: max|du0|={ic_du:.2e} "
              f"max|dT0|={float(ic_dT.max()):.2e} K in {n_bad}/{n_owned} owned cells")
    assert np.sqrt(np.mean(u0_ref ** 2)) > 1.0, "IC carries no wind"

    from legoesm.parallel.voronoi_partition import scatter_to_local

    def _sc(field_local, global_data, entity):
        return field_local.replace(data=jnp.asarray(
            scatter_to_local(global_data, part, entity), dtype=field_local.data.dtype))

    st = d.state
    tr = None
    if st.tracers is not None:
        tr = {k: _sc(st.tracers[k], np.asarray(ic_ref_state.tracers[k].data), "cell")
              for k in st.tracers}
    d.state = st._replace(
        u=_sc(st.u, u0_ref, "edge"),
        T=_sc(st.T, T0_ref, "cell"),
        p_s=_sc(st.p_s, np.asarray(ic_ref_state.p_s.data), "cell"),
        phis=_sc(st.phis, np.asarray(ic_ref_state.phis.data), "cell"),
        tracers=tr,
    )
    assert np.max(np.abs(np.asarray(d.state.T.data[:n_owned]) - T0_ref[owned_cells])) == 0.0

    d.run()

    u_mpi = np.asarray(d.state.u.data[:part.n_owned_edges])
    T_mpi = np.asarray(d.state.T.data[:n_owned])
    ps_mpi = np.asarray(d.state.p_s.data[:n_owned])

    assert np.all(np.isfinite(u_mpi))
    r_u = _rel_rms(u_mpi, u_ref[owned_edges])
    r_T = _rel_rms(T_mpi, T_ref[owned_cells])
    r_ps = _rel_rms(ps_mpi, ps_ref[owned_cells])
    # lowest level separately: the measured defect is surface-weighted
    r_u_sfc = _rel_rms(u_mpi[:, -1], u_ref[owned_edges, -1])
    du_max = float(np.max(np.abs(u_mpi - u_ref[owned_edges])))
    u_rms = float(np.sqrt(np.mean(u_ref[owned_edges] ** 2)))

    # Localisation: where along the owned edges does |du| sit?  Ring 1 =
    # an edge with a flanking cell adjacent to the halo; ring >= 3 = interior.
    dist = _ring_distance_to_halo(part, d._voronoi_layout.local_mesh)
    coe = np.asarray(d._voronoi_layout.local_mesh.cellsOnEdge)[:, :part.n_owned_edges]
    edge_ring = np.minimum(dist[np.maximum(coe[0], 0)], dist[np.maximum(coe[1], 0)])
    du_col = np.max(np.abs(u_mpi - u_ref[owned_edges]), axis=1)
    loc = {r: (float(du_col[edge_ring == r].max()) if np.any(edge_ring == r) else 0.0,
               float(np.sqrt(np.mean(du_col[edge_ring == r] ** 2))) if np.any(edge_ring == r) else 0.0,
               int(np.sum(edge_ring == r)))
           for r in (1, 2, 3, 4)}
    loc["int"] = (float(du_col[edge_ring >= 5].max()) if np.any(edge_ring >= 5) else 0.0,
                  float(np.sqrt(np.mean(du_col[edge_ring >= 5] ** 2))) if np.any(edge_ring >= 5) else 0.0,
                  int(np.sum(edge_ring >= 5)))
    mean_bias = float(np.mean(np.abs(u_mpi) - np.abs(u_ref[owned_edges])))

    print(f"\n[{turbulence} x{n_steps}] rank{rank} owned: rel-rms u={r_u:.3e} "
          f"(sfc {r_u_sfc:.3e}, max|du|={du_max:.3e} m/s, "
          f"u rms {u_rms:.2f} m/s, mean(|u_mpi|-|u_ref|)={mean_bias:+.2e}) "
          f"T={r_T:.3e} p_s={r_ps:.3e}")
    print(f"[{turbulence} x{n_steps}] rank{rank} |du| by ring (max, rms, n): "
          + ", ".join(f"{k}:({v[0]:.1e},{v[1]:.1e},{v[2]})" for k, v in loc.items()))

    assert r_T < REL_RMS_TOL, f"T parity broken: {r_T:.3e}"
    assert r_ps < REL_RMS_TOL, f"p_s parity broken: {r_ps:.3e}"
    assert r_u < REL_RMS_TOL, (
        f"[{turbulence}] owned-edge u rel-rms {r_u:.3e} "
        f"(sfc {r_u_sfc:.3e}, max {du_max:.3e} m/s) > {REL_RMS_TOL}")
