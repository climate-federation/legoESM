"""Production-path gate: the JRA55 block scan with the lat-band SPMD step AND
the prognostic slab sea-ice tile (``--enable-latlon-spmd --jra55-sea-ice``).

Until 2026-09 ``run_omip`` refused this combination in three places (the arg
guard, both block-scan builders, the run loop).  The slab tile is elementwise,
so laid out on the ocean's lat bands (``shard_cell_pytree_latlon``) one block
of the PRODUCTION ``_build_jra55_block_fn`` under ``spmd_step`` must match the
serial block for both the ocean state and the ice tile.  Runs in a SUBPROCESS
so the 4-virtual-device XLA flag always takes effect (the in-process variant
silently skips once another test initialised the single-device backend).
The synthetic JRA55 cache / tiny lat-lon setup come from the day-23 dispatch
harness (``tests/unit/test_run_omip_jra55_dispatch.py``).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()


def _load_harness():
    day23 = _THIS.parent / "test_run_omip_jra55_dispatch.py"
    spec = importlib.util.spec_from_file_location("_day23_spmd_ice", day23)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_day23_spmd_ice"] = mod
    spec.loader.exec_module(mod)
    return mod


def _worker(tmp_dir: str) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    assert jax.device_count() >= 4, jax.device_count()
    _day23 = _load_harness()
    run_omip = _day23.run_omip
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_cell_pytree_latlon,
        gather_state_latlon,
        make_sharded_ocean_step,
        shard_cell_pytree_latlon,
        shard_forcing_stack_latlon,
        shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    n_lat, n_lon, dt, n_block = 8, 16, 600.0, 4
    cache = _day23._make_synthetic_cache(Path(tmp_dir), n_lat=n_lat,
                                         n_lon=n_lon, n_records=64)
    grid, z_coord, _, model, _ = _day23._make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon)
    T_woa, S_woa = _day23._make_woa_like_targets(grid, nlev=4)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _day23._argparse_namespace(jra55_cache=str(cache))
    args.jra55_sea_ice = True
    js = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
    assert js["enable_sea_ice"] and js["ice_config"].dynamics == "none"
    # seed a non-trivial ice cover so the tile's evolution is load-bearing
    # (the tiny harness grid spans only +-1.4 deg latitude, so seed by
    # longitude: the western half carries 0.6 ice cover 1 m thick, which the
    # ~290 K synthetic air then melts)
    ice0 = js["ice_state_init"]
    conc = np.zeros((n_lat, n_lon))
    conc[:, : n_lon // 2] = 0.6
    _dt = ice0.concentration.data.dtype
    ice0 = ice0._replace(
        concentration=ice0.concentration.replace(
            data=jnp.asarray(conc, dtype=_dt)),
        h_ice=ice0.h_ice.replace(data=jnp.asarray(1.0 * (conc > 0), dtype=_dt)))
    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(
        0, n_block, dt, js)

    # --- serial reference, built from PERTURBED reference fields ---
    # The SPMD builders below keep the UNPERTURBED closure and see the
    # perturbed fields only through the ``refs`` argument, so parity holds
    # only if the scan really consumes refs (a builder ignoring refs fails).
    _REF_KEYS = (("sponge_gamma_2d", "sponge_gamma"),
                 ("sponge_T_ref_3d", "sponge_T_ref"),
                 ("sponge_S_ref_3d", "sponge_S_ref"),
                 ("sss_target_2d", "sss_target"),
                 ("_ocean_mask_2d", "ocean_mask"))
    js_ref = dict(js)
    for key, _ in _REF_KEYS:
        if js.get(key) is None or key == "_ocean_mask_2d":
            continue
        js_ref[key] = (js[key] * 1.5 if key == "sponge_gamma_2d"
                       else js[key] + 0.25)
    assert any(js_ref[k] is not js.get(k) for k, _ in _REF_KEYS), \
        "harness must exercise at least one reference field"
    block_fn = run_omip._build_jra55_block_fn(model, js_ref, dt)
    ref_state, ref_ice = block_fn(state, atm_stack, runoff_stack,
                                  jnp.int32(0), ice0)

    # --- production SPMD path ---
    model.prime_step_caches(state)
    dev = create_latlon_mesh(n_devices=4)
    spmd_step = make_sharded_ocean_step(model, dev.mesh)
    block_spmd = run_omip._build_jra55_block_fn(model, js, dt,
                                                spmd_step=spmd_step)
    ss = shard_state_latlon(state, dev.mesh)
    ice_s = shard_cell_pytree_latlon(ice0, dev.mesh)
    atm_s = shard_forcing_stack_latlon(atm_stack, dev.mesh)
    run_s = shard_forcing_stack_latlon(runoff_stack, dev.mesh)
    # perturbed reference fields as band-sharded ARGUMENTS (route-B layout)
    from legoesm.ocean.dynamics.sharded_ocean_step import shard_forcing_latlon
    refs = {name: shard_forcing_latlon(jnp.asarray(js_ref[key]), dev.mesh)
            for key, name in _REF_KEYS if js_ref.get(key) is not None}
    out_state, out_ice = block_spmd(ss, atm_s, run_s, jnp.int32(0), ice_s,
                                    aux=getattr(spmd_step, "aux", None),
                                    refs=refs)
    out_state = gather_state_latlon(out_state, dev.mesh, to_host=True)
    out_ice = gather_cell_pytree_latlon(out_ice, dev.mesh, to_host=True)

    for nm in ("T", "S", "eta", "u", "v"):
        np.testing.assert_allclose(
            np.asarray(getattr(out_state, nm).data),
            np.asarray(getattr(ref_state, nm).data),
            atol=2e-4, rtol=1e-3, err_msg=f"SPMD+ice ocean {nm}")
    for nm in ref_ice._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(out_ice, nm).data),
            np.asarray(getattr(ref_ice, nm).data),
            atol=1e-9, rtol=1e-9, err_msg=f"SPMD+ice ice {nm}")
    moved = {nm: float(np.abs(np.asarray(getattr(ref_ice, nm).data)
                              - np.asarray(getattr(ice0, nm).data)).max())
             for nm in ref_ice._fields}
    print("ice field max |change| over the block:", moved)
    assert max(moved.values()) > 0, f"ice did not evolve: {moved}"

    # --- the GPU-interp block builder (the runner DEFAULT, --gpu-interp) ---
    raw_stack, runoff_records, meta = run_omip._preload_jra55_raw_records(
        0, n_block, dt, js)
    get_ref = run_omip._build_jra55_block_fn_interp(model, js_ref, dt)
    ref2_state, ref2_ice = get_ref(n_block)(
        state, raw_stack, runoff_records, meta["record_days"],
        jnp.float64(meta["block_start_day"]),
        jnp.float64(meta["block_start_day_forcing"]), ice0)
    get_spmd = run_omip._build_jra55_block_fn_interp(model, js, dt,
                                                    spmd_step=spmd_step)
    raw_s = shard_forcing_stack_latlon(raw_stack, dev.mesh)
    rr_s = shard_forcing_stack_latlon(runoff_records, dev.mesh)
    out2_state, out2_ice = get_spmd(n_block)(
        ss, raw_s, rr_s, meta["record_days"],
        jnp.float64(meta["block_start_day"]),
        jnp.float64(meta["block_start_day_forcing"]), ice_s,
        aux=getattr(spmd_step, "aux", None), refs=refs)
    out2_state = gather_state_latlon(out2_state, dev.mesh, to_host=True)
    out2_ice = gather_cell_pytree_latlon(out2_ice, dev.mesh, to_host=True)
    for nm in ("T", "S", "eta", "u", "v"):
        np.testing.assert_allclose(
            np.asarray(getattr(out2_state, nm).data),
            np.asarray(getattr(ref2_state, nm).data),
            atol=2e-4, rtol=1e-3, err_msg=f"SPMD+ice (gpu-interp) ocean {nm}")
    for nm in ref2_ice._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(out2_ice, nm).data),
            np.asarray(getattr(ref2_ice, nm).data),
            atol=1e-9, rtol=1e-9, err_msg=f"SPMD+ice (gpu-interp) ice {nm}")
    print("SPMD_ICE_PARITY_OK")


@pytest.mark.timeout(900)
def test_block_scan_spmd_with_slab_ice_matches_serial(tmp_path):
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    code = (f"import sys; sys.path.insert(0, {str(_THIS.parent)!r}); "
            f"import test_run_omip_spmd_sea_ice as t; t._worker({str(tmp_path)!r})")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=840)
    assert r.returncode == 0, r.stdout[-4000:] + "\n" + r.stderr[-6000:]
    assert "SPMD_ICE_PARITY_OK" in r.stdout


def test_run_loop_refuses_spmd_ice_without_gather_hook():
    """The loop's ice-under-SPMD precondition is now 'the ice gather hook is
    wired', not a blanket refusal: without it the restart writer would
    np.asarray a sharded tile."""
    _day23 = _load_harness()
    run_omip = _day23.run_omip
    js = {"enable_sea_ice": True, "_use_single_step": False}
    with pytest.raises(ValueError, match="spmd_gather_ice"):
        run_omip._run_omip_loop(
            None, None, "latlon", None, None, 600.0, 1, 1,
            jra55_state=js, spmd_step=object(), spmd_gather=None,
            spmd_gather_ice=None)


def test_host_numpy_regrid_matches_device_regrid():
    """The multi-process JRA55 regrid is done in NumPy on the host (byte-identical
    across processes by construction); it must equal the device regrid_scalar
    used by the single-process path to round-off."""
    import jax.numpy as jnp
    import numpy as np
    from legoesm.grids.regridding import (
        compute_latlon_to_voronoi_weights, regrid_scalar)
    src_lat = np.deg2rad(np.linspace(-89, 89, 18))
    src_lon = np.deg2rad(np.arange(0, 360, 10.0))
    tgt_lat = np.deg2rad(np.linspace(-60, 60, 7)[:, None] * np.ones((1, 9)))
    tgt_lon = np.deg2rad(np.linspace(5, 355, 9)[None, :] * np.ones((7, 1)))
    rw = compute_latlon_to_voronoi_weights(src_lat, src_lon, tgt_lat.ravel(),
                                           tgt_lon.ravel())
    rw = rw._replace(target_shape=tgt_lat.shape)
    _day23 = _load_harness()
    run_omip = _day23.run_omip
    recs = np.random.default_rng(1).standard_normal((3, 18, 36))
    ref = np.stack([np.asarray(regrid_scalar(jnp.asarray(r), rw)) for r in recs])
    out = np.asarray(run_omip._regrid_records_host(recs, rw))
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-12)
    # trailing (level) dim is preserved like regrid_scalar
    recs3 = np.random.default_rng(2).standard_normal((2, 18, 36, 3))
    ref3 = np.stack([np.asarray(regrid_scalar(jnp.asarray(r), rw)) for r in recs3])
    out3 = np.asarray(run_omip._regrid_records_host(recs3, rw))
    assert out3.shape == ref3.shape == (2,) + tuple(rw.target_shape) + (3,)
    np.testing.assert_allclose(out3, ref3, rtol=1e-12)
