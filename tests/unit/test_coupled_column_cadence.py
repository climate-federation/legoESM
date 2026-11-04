"""Coupling cadence on the column loop (composable P5b), on a tiny coupled
MPAS run (icosahedral level 2, dt 600 s, slab aquaplanet):

* the coupler is called every coupling_dt with the window MEANS of the
  per-step net surface SW/LW and precipitation (closure: mean x window ==
  sum of the steps the atmosphere produced);
* with the coupler's ocean SST replaced by the prescribed field, the coupled
  atmosphere equals the standalone atmosphere bitwise over three windows,
  although every exchange re-samples SST/SIC;
* a restart in the middle of a window reproduces the continuous run bitwise
  (atmosphere state and every coupled checkpoint array);
* a coupled checkpoint paired with the wrong atmosphere step is refused."""
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

DT = 600.0


def _atm_cfg(days, **kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    out = dict(output_dir="", diag_days=0, checkpoint_days=0)
    out.update(kw.pop("output", {}))
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=8,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(**out),
        days=days, dataset="analytical", radiation="gray",
        rad_update_steps=1, convection="none", turbulence="none",
        microphysics="kessler", precision="fp64", **kw)


def _coupled(days, coupling_dt, outdir, **kw):
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    cc = PRESETS["aquaplanet"]()._replace(coupling_dt=coupling_dt)
    d = CoupledESMDriver(_atm_cfg(days, **kw), cc, output_dir=outdir)
    d.setup()
    return d


def _record_steps(atm, rain=False):
    """Per-step (sw_net, lw_net, precip, frozen precip, forcing T_sfc) of
    each dycore step.  ``rain``: overwrite the published surface precip with
    a positive per-step field and its frozen part (this dry start rains
    nothing in two hours), so both channels are tested on live values."""
    from legoesm.driver.model_driver import land_snowfall_from_sfc_diag
    rec = []
    model, orig = atm.model, atm.model.step
    rng = np.random.default_rng(0)
    from legoesm.core.state import (
        MPAS_SFC_DIAG_BASE_KEYS, MPAS_SFC_DIAG_EXTRA_KEYS)
    frozen_slot = (len(MPAS_SFC_DIAG_BASE_KEYS)
                   + MPAS_SFC_DIAG_EXTRA_KEYS.index("precip_frozen"))

    def step(*a, **k):
        out = orig(*a, **k)
        sd = model._sfc_diag
        if rain:
            r = jnp.asarray(1e-4 * rng.random(np.shape(sd[0].data)))
            sd = list(sd)
            sd[2] = types.SimpleNamespace(data=r)
            sd[frozen_slot] = types.SimpleNamespace(
                data=r * jnp.asarray(rng.random(np.shape(r))))
            sd = model._sfc_diag = tuple(sd)
        pr = np.asarray(sd[2].data).reshape(-1)
        rec.append((np.asarray(sd[0].data).reshape(-1),
                    np.asarray(sd[1].data).reshape(-1), pr,
                    np.asarray(land_snowfall_from_sfc_diag(
                        sd, jnp.asarray(pr))).reshape(-1),
                    np.asarray(k["forcing"]["T_sfc"]).reshape(-1)))
        return out
    model.step = step
    return rec


def _record_exchanges(d):
    calls = []
    hook = d._segment_hook

    def wrapped(drv, day, dt_segment):
        aux = drv._carry_aux
        calls.append((day, dt_segment, {
            k: np.asarray(aux[k]).reshape(-1) for k in
            ("seg_sw_net_sfc", "seg_lw_net_sfc", "seg_precip",
             "seg_precip_frozen") if k in aux}))
        return hook(drv, day, dt_segment)
    d._segment_hook = wrapped
    return calls


def test_exchange_every_window_with_window_means(tmp_path):
    """Four 1800 s windows in two hours: four calls, dt 1800 s, at the
    window ends; each export is the mean of exactly that window's steps
    (closure: mean x window == the steps' own integral, every channel)."""
    d = _coupled(4 * 1800.0 / 86400.0, 1800.0, str(tmp_path))
    steps = _record_steps(d._atm, rain=True)
    calls = _record_exchanges(d)
    assert d.run() == "COMPLETED"
    k = 3
    assert len(steps) == 4 * k
    assert [c[1] for c in calls] == [1800.0] * 4
    np.testing.assert_allclose([c[0] for c in calls],
                               [(w + 1) * 1800.0 / 86400.0 for w in range(4)],
                               rtol=0, atol=1e-12)
    assert all(np.all(s[2] > 0) and np.any(s[0] != 0) for s in steps)
    assert any(np.any(s[3] > 0) for s in steps), "no snowfall: vacuous"
    for w, (_, _, ex) in enumerate(calls):
        win = steps[w * k:(w + 1) * k]
        for i, key in enumerate(("seg_sw_net_sfc", "seg_lw_net_sfc",
                                 "seg_precip", "seg_precip_frozen")):
            total = sum(s[i] * DT for s in win)
            np.testing.assert_allclose(ex[key] * (k * DT), total,
                                       rtol=1e-13, atol=0, err_msg=key)
        # a window MEAN, not the closing step's value (the old export)
        assert not np.array_equal(ex["seg_precip"], win[-1][2])


def test_sst_refreshed_after_each_exchange(tmp_path):
    """The coupler's SST moves by 0.25 K at every exchange: the surface
    temperature the atmosphere steps with is constant inside a window and
    shifts by exactly that amount from one window to the next."""
    d = _coupled(4 * 1800.0 / 86400.0, 1800.0, str(tmp_path))
    steps = _record_steps(d._atm)
    n_ex = [0]
    hook = d._segment_hook

    def counted(*a):
        n_ex[0] += 1
        return hook(*a)
    d._segment_hook = counted
    presc = d._original_get_sst_sic

    def sst(day):
        t, ice = presc(day)
        return jnp.asarray(t) + 0.25 * n_ex[0], ice
    d._atm.get_sst_sic = sst
    assert d.run() == "COMPLETED" and n_ex[0] == 4
    _, sic = presc(0.0)
    ocean = np.asarray(sic).reshape(-1) == 0.0
    assert ocean.any()
    ts = [s[4][ocean] for s in steps]
    for w in range(4):
        for t in ts[w * 3 + 1:(w + 1) * 3]:
            np.testing.assert_array_equal(t, ts[w * 3])
        if w:
            np.testing.assert_allclose(ts[w * 3] - ts[(w - 1) * 3], 0.25,
                                       rtol=0, atol=1e-12)


def test_prescribed_sst_stub_matches_standalone_atmosphere(tmp_path):
    """Coupler ocean replaced by the prescribed SST: the coupled atmosphere
    is bitwise the standalone one over three windows, through the two
    post-exchange SST refreshes."""
    from legoesm.driver.model_driver import ModelDriver
    days = 3 * 1800.0 / 86400.0
    amip = ModelDriver(_atm_cfg(days), output_dir=str(tmp_path / "amip"))
    amip.setup()
    d = _coupled(days, 1800.0, str(tmp_path / "cpl"))
    d._atm.get_sst_sic = d._original_get_sst_sic
    assert amip.run() == "COMPLETED" and d.run() == "COMPLETED"
    for name in ("u", "T", "p_s"):
        np.testing.assert_array_equal(
            np.asarray(getattr(amip.state, name).data),
            np.asarray(getattr(d._atm.state, name).data), err_msg=name)
    for k in (amip.state.tracers or {}):
        np.testing.assert_array_equal(
            np.asarray(amip.state.tracers[k].data),
            np.asarray(d._atm.state.tracers[k].data), err_msg=k)


def _final(d):
    s = d._atm.state
    out = {f"atm_{n}": np.asarray(getattr(s, n).data) for n in ("u", "T", "p_s")}
    out.update({f"trc_{k}": np.asarray(v.data)
                for k, v in (s.tracers or {}).items()})
    out.update({k: np.asarray(v) for k, v in d.coupled_checkpoint_arrays().items()})
    return out


@pytest.mark.parametrize("ck", [4, 6])
def test_restart_is_bitwise(tmp_path, ck):
    """1800 s windows (3 steps); checkpoint after an exchange with one step
    pending (4), and exactly on a window close (6); the resumed run equals
    the continuous one after 12 steps, atmosphere and every coupled
    checkpoint array."""
    cont = _coupled(12 * DT / 86400.0, 1800.0, str(tmp_path / "cont"))
    assert cont.run() == "COMPLETED"
    ref = _final(cont)

    first = _coupled(ck * DT / 86400.0, 1800.0, str(tmp_path / "a"),
                     output=dict(checkpoint_days=(ck * DT + 1.0) / 86400.0))
    assert first.run() == "COMPLETED"
    ckpt = sorted(Path(tmp_path / "a").glob("checkpoint_day_*.npz"))
    assert len(ckpt) == 1, ckpt
    with np.load(ckpt[0]) as z:
        assert int(z["cpl_win_n"]) == ck % 3 and int(z["cpl_win_k"]) == 3

    # this loop's days = what the job advances
    res = _coupled((12 - ck) * DT / 86400.0, 1800.0, str(tmp_path / "b"))
    step, day = res._atm.load_checkpoint(str(ckpt[0]))
    assert step == ck
    res.load_coupled_checkpoint(day - res.atm_config.start_day,
                                checkpoint_dir=str(tmp_path / "a"),
                                required=True, atm_step=step)
    assert res.run(start_step=step, start_day=day) == "COMPLETED"
    got = _final(res)
    assert sorted(got) == sorted(ref)
    for k in ref:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)


def test_mispaired_coupled_checkpoint_refused(tmp_path):
    d = _coupled(DT / 86400.0, 1800.0, str(tmp_path))
    d.save_checkpoint(step=7, day=0.0)
    with pytest.raises(ValueError, match="mismatched atmosphere/ocean pair"):
        d.load_coupled_checkpoint(0.0, checkpoint_dir=str(tmp_path),
                                  required=True, atm_step=8)
    d.load_coupled_checkpoint(0.0, checkpoint_dir=str(tmp_path),
                              required=True, atm_step=7)


def test_coupling_dt_not_multiple_of_1800_refused(tmp_path):
    d = _coupled(DT / 86400.0, 1200.0, str(tmp_path))
    with pytest.raises(ValueError, match="multiple of 1800 s"):
        d.run()
