"""Restart from a checkpoint older than the CAM6 droplet-activation carries.

Own file (one pytest process): the driver runs here exhaust LLVM in one
process alongside test_physics_state_carry.py.
"""
import numpy as np

from tests.unit.test_physics_state_carry import _mpas_driver_cfg


def test_mpas_checkpoint_without_activation_fields_seeds_them(tmp_path, allow_version_drift):
    """User decision 2026-10-24, 3a: a checkpoint written before the CAM6
    droplet-activation carries (CLUBB w'^2 / Kh, CLDO) restarts with them
    seeded at zero instead of tripping the completeness gate."""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _mpas_driver_cfg()
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run() == "COMPLETED"
    ckpt = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))[-1]
    new = ("clubb_wp2_iface", "clubb_kh_iface", "cloud_fraction_old")
    old = tmp_path / "pre_activation.npz"
    with np.load(ckpt) as d:
        assert all(f"physstate_{f}" in d.files for f in new)
        np.savez(old, **{k: d[k] for k in d.files
                         if k not in {f"physstate_{f}" for f in new}})
    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    driver_b.load_checkpoint(old)
    assert driver_b.run() == "COMPLETED"
