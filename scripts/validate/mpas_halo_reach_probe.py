"""Sharded step at effective ring count E vs the SINGLE-DEVICE model.step
(the truth), plus the collective census per step."""
import os, numpy as np, jax
from legoesm.parallel import sharded_dynamics as sd
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig
from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
from legoesm.parallel.mesh import create_voronoi_device_mesh, replicate_pytree
from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
WIDE = os.environ.get("REACH_WIDE", "0"); os.environ["LEGOESM_MPAS_WIDE_HALO"] = WIDE
mesh = create_voronoi_mesh(subdivision_level=4); mesh = reorder_voronoi_for_sharding(mesh, 4)
sigma = create_sigma_coordinate(8)
cfg = MPASPrimitiveEquationConfig(nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True, pv_scheme="energy", time_integrator="ssp_rk3")
dt, n_steps = 600.0, 3
model1 = MPASPrimitiveEquationModel(mesh, sigma, cfg)
s = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
for _ in range(n_steps): s = model1.step(s, dt)
truth = {k: np.asarray(getattr(s, k).data) for k in ("u", "T", "p_s")}
orig = sd._close_halo_under_cellsOnEdge
import ast
for depth, passes in ast.literal_eval(os.environ.get("REACH_GRID", "((1, 1), (2, 1), (3, 1))")):
    sd.SPMD_HALO_DEPTH = depth
    sd._close_halo_under_cellsOnEdge = lambda o, h, c, n_passes, _p=passes: orig(o, h, c, _p)
    dev = create_voronoi_device_mesh(nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices, n_devices=4)
    model = MPASPrimitiveEquationModel(replicate_pytree(mesh, dev), sigma, cfg)
    s0 = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    step = sd.make_voronoi_sharded_step(model, dev, halo_strategy="ppermute")
    txt = jax.jit(lambda st: step(st, dt)).lower(s0).compile().as_text()
    ncp = txt.count("collective-permute-start(") + txt.count(" collective-permute(")
    nar = txt.count("all-reduce-start(") + txt.count(" all-reduce(") + txt.count("all-gather")
    s1 = s0
    for _ in range(n_steps): s1 = step(s1, dt)
    got = {k: np.asarray(getattr(s1, k).data) for k in ("u", "T", "p_s")}
    errs = {k: float(np.max(np.abs(got[k]-truth[k]))/max(float(np.max(np.abs(truth[k]))),1e-30)) for k in got}
    print(f"wide={WIDE} eff_depth={step._halo_depth_effective} E={step._halo_depth_effective+passes} rings (depth={depth},passes={passes}): collective-permutes/step={ncp} allreduce/gather={nar} rel err vs single-device {errs}")
