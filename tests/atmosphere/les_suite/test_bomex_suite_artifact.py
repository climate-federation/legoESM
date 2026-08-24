"""Unit test for run_bomex_les._emit_suite_artifact — the moist LES → LESReferenceArtifact
converter (assembles the tuner's self-describing moist artifact from les_record prof_NNN.npz
+ the run's forcing). Uses synthetic prof files so it needs no GPU / moist LES run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_MOD = Path(__file__).resolve().parents[3] / "scripts" / "run" / "run_bomex_les.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("run_bomex_les", _MOD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_bomex_les"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_emit_suite_artifact_builds_valid_moist_artifact(tmp_path):
    mod = _load_converter()
    nz, nt = 40, 3
    z = np.linspace(0.0, 3000.0, nz)
    prof_dir = tmp_path / "profiles"
    prof_dir.mkdir()
    rng = np.random.default_rng(0)
    for k in range(nt):
        np.savez(
            prof_dir / f"prof_{k:03d}.npz", t_hours=0.5 * k, z=z,
            theta=298.0 + 0.004 * z, u=-8.0 + 1e-3 * z, v=np.zeros(nz),
            wtheta=0.01 * np.exp(-z / 500.0), qt=0.017 - 4e-6 * z,
            wqt=5e-5 * np.exp(-z / 500.0) + 1e-9 * rng.normal(size=nz))

    args = SimpleNamespace(
        output=tmp_path, case_label="bomex", sgs_model="lasd")
    forc = dict(ug=-8.0 + 1e-3 * z, vg=np.zeros(nz), w_ls=-0.01 * np.ones(nz),
                tls=np.full(nz, -1e-5), qls=np.full(nz, -1e-8),
                th_flux=0.00806, qv_flux=5.27e-5)
    out = tmp_path / "bomex__lasd.npz"
    mod._emit_suite_artifact(args, forc, out)

    # load_artifact runs the bridge's validation (moist artifact must carry wqt_resolved etc.)
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    a = load_artifact(out)
    assert a.is_moist and a.case_name == "bomex" and a.sgs == "lasd"
    assert a.theta.shape == (nt, nz) and a.qt.shape == (nt, nz)
    assert a.wqt_resolved.shape == (nt, nz) and a.wtheta_resolved.shape == (nt, nz)
    assert np.isclose(a.f_c, mod._FCOR) and np.isclose(float(a.w_theta_s[0]), 0.00806)
    assert np.isclose(float(a.w_qv_s[0]), 5.27e-5)
    assert a.u_geo is not None and float(a.subsidence_w[0]) < 0.0  # subsidence is downward


def test_record_theta_l_derives_thetal_not_theta():
    # The spectral moist LES prognoses ACTUAL θ; the artifact must record θ_l = θ −
    # (L_v/c_pd)·q_c/Π derived from θ + cloud q_c (slot 1), else the score compares LES θ to
    # SCM θ_l. Verify _record_theta_l applies exactly that reduction (q_c only, not rain).
    import numpy as np
    from types import SimpleNamespace
    from legoesm import constants
    mod = _load_converter()
    ny, nx, nz = 2, 2, 6
    theta = np.full((ny, nx, nz), 300.0)
    tr = np.zeros((ny, nx, nz, 4))
    tr[..., 1] = 1.0e-3          # q_c (cloud liquid)
    tr[..., 2] = 5.0e-4          # q_r (rain) — must NOT enter θ_l
    exner = np.linspace(1.0, 0.9, nz)
    st = SimpleNamespace(theta=theta, tracers=tr)
    ref = SimpleNamespace(exner_c=exner)
    thl = mod._record_theta_l(st, ref)
    expect = 300.0 - (constants.L_v / constants.c_pd) * 1.0e-3 / exner
    assert np.allclose(thl[0, 0], expect, rtol=1e-12)   # q_c only, rain excluded
    assert np.all(thl < theta)                          # cloud ⇒ θ_l < θ (correct sign)


def test_end_to_end_recording_puts_thetal_in_artifact(tmp_path):
    # Close the codex LOW: drive the REAL path _record_theta_l → les_record.record_frame →
    # prof_NNN.npz → _emit_suite_artifact, from a synthetic 3D state carrying cloud q_c, and
    # confirm the artifact's `theta` is θ_l (BELOW raw θ where q_c>0) — not the raw prognostic
    # θ. This is the coverage the hand-built-profile tests can't give.
    import importlib.util as ilu
    import numpy as np
    from types import SimpleNamespace
    from legoesm import constants
    mod = _load_converter()
    # load les_record the same way the driver does (sibling script)
    lr_path = Path(__file__).resolve().parents[3] / "scripts" / "run" / "les_record.py"
    spec = ilu.spec_from_file_location("les_record", lr_path)
    lr = ilu.module_from_spec(spec); sys.modules["les_record"] = lr
    spec.loader.exec_module(lr)

    ny, nx, nz = 4, 4, 12
    z = np.linspace(20.0, 2000.0, nz)
    theta = np.broadcast_to(298.0 + 0.004 * z, (ny, nx, nz)).astype(float).copy()
    exner = np.linspace(1.0, 0.9, nz)
    tr = np.zeros((ny, nx, nz, 4))
    tr[..., 0] = 0.016 - 5e-6 * z        # q_v
    cloudy = (z > 600.0) & (z < 1200.0)
    tr[..., 1] = np.where(cloudy, 8e-4, 0.0)[None, None, :]   # q_c cloud layer
    u3 = np.full((ny, nx, nz), -8.0)
    v3 = np.zeros((ny, nx, nz))
    wc3 = np.full((ny, nx, nz), 0.1)      # cell-centred w (record_frame expects centres)
    st = SimpleNamespace(theta=theta, tracers=tr)
    ref = SimpleNamespace(exner_c=exner, rho_c=np.linspace(1.1, 0.9, nz))

    out = tmp_path
    h_idx, h_z = lr.select_heights(z, 2000.0)
    for k in range(3):
        lr.record_frame(
            out, k, 0.5 * k, "bomex_cu", z, u3, v3, wc3,
            mod._record_theta_l(st, ref), 6400.0, 6400.0, h_idx, h_z, 0.1,
            qc3=tr[..., 1], rho_z=ref.rho_c, qr3=tr[..., 2], qv3=tr[..., 0])

    args = SimpleNamespace(output=out, case_label="bomex_cu", sgs_model="smagorinsky",
                           dynamic=True)
    forc = dict(ug=np.full(nz, -8.0), vg=np.zeros(nz), w_ls=-0.006 * z,
                tls=np.full(nz, -2e-5), qls=np.full(nz, -1e-8),
                th_flux=0.008, qv_flux=5.2e-5)
    art_path = tmp_path / "bomex_cu__lasd.npz"
    mod._emit_suite_artifact(args, forc, art_path)

    from legoesm.atmosphere.les_suite.bridge import load_artifact
    a = load_artifact(art_path)
    theta_l_prof = a.theta[-1]              # recorded θ (should be θ_l)
    raw_theta = theta.mean((0, 1))          # raw prognostic θ
    # in the cloud layer the recorded value must be BELOW raw θ by the latent decrement
    drop = raw_theta - theta_l_prof
    assert np.max(drop) > 0.1               # θ_l is strictly below θ in cloud (not raw θ)
    assert np.allclose(drop[~cloudy], 0.0, atol=1e-6)   # cloud-free ⇒ θ_l == θ
