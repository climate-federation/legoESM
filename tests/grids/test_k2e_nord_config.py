"""k2e_nord configuration fidelity (2026-07-27 root cause).

The authoritative Zenodo model runs k2e_nord=2 (fv_arrays.F90:150 +
global_grid_data.F90:57 defaults, no namelist override); the historic
4 came from the luanfs mirror monolith and its corner-adjacent
extrapolation lobes are the measured vertex amplifier.  These tests
pin the {2,4} guard surface, the nord-keyed table cache, and the
lobe-structure discriminator that separates the two orders."""

import numpy as np
import pytest


def test_generator_accepts_2_and_4_rejects_others():
    from legoesm.grids.fv3_native_halos import compute_fv3_native_k2e

    t2 = compute_fv3_native_k2e(12, remap_ng=3, k2e_nord=2)
    t4 = compute_fv3_native_k2e(12, remap_ng=3, k2e_nord=4)
    assert t2["k2e_nord"] == 2 and t4["k2e_nord"] == 4
    assert t2["A_coef"].shape[1] == 2 and t4["A_coef"].shape[1] == 4
    for bad in (3, 6, 0):
        with pytest.raises(NotImplementedError):
            compute_fv3_native_k2e(12, remap_ng=3, k2e_nord=bad)


def test_duogrid_data_accepts_2_and_4_rejects_others():
    from legoesm.grids.fv3_native_halos import (
        create_fv3_native_duogrid_data,
    )

    create_fv3_native_duogrid_data(12, ng=3, k2e_nord=2)
    create_fv3_native_duogrid_data(12, ng=3, k2e_nord=4)
    with pytest.raises(NotImplementedError):
        create_fv3_native_duogrid_data(12, ng=3, k2e_nord=3)


def test_ed_route_guard_admits_2():
    from legoesm.grids.cubed_sphere import create_fv3_native_cubed_sphere

    g2 = create_fv3_native_cubed_sphere(12, use_duogrid=True, k2e_nord=2)
    assert g2 is not None
    with pytest.raises(NotImplementedError):
        create_fv3_native_cubed_sphere(12, use_duogrid=True, k2e_nord=3)


def test_ring_remap_nord_discriminates():
    """The two orders must produce DIFFERENT corner-adjacent ring
    responses (synthetic-violation teeth for every consumer that
    threads nord), and the nord-2 response must have NO source weight
    outside its 2-point window (the monotone-window property that
    kills the amplifier)."""
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
        k2e_remap_halo_rings,
    )

    n, ng = 12, 3
    lo = 1 - ng
    resp = {}
    for nord in (2, 4):
        f6 = [np.zeros((n + 2 * ng, n + 2 * ng)) for _ in range(6)]
        # vertex-adjacent bumps on tile 3 (both, like the C48 probe:
        # tile-1's north halo receives from the (1, n) bump)
        f6[2][1 - lo, 1 - lo] = 1.0
        f6[2][1 - lo, n - lo] = 1.0
        for t in range(1, 7):
            exchange_agrid_scalar_halos(f6, t, n, ng)
        k2e_remap_halo_rings(f6, "A", n, ng, k2e_nord=nord)
        resp[nord] = np.array([f6[0][i - lo, (n + 1) - lo]
                               for i in (1, 2, 3)])
    # the response must be live (a dead probe proves nothing) and the
    # two orders must disagree — cache keyed by nord, no leakage
    assert np.abs(resp[2]).max() > 1e-3, resp
    assert np.abs(resp[4]).max() > 1e-3, resp
    assert not np.allclose(resp[2], resp[4], atol=1e-12), resp


def test_stepper_ctx_threads_nord():
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    ctx2 = build_six_face_duo_context(12, 3, use_ext_bundle=True,
                                      oracle_conventions=True,
                                      omega=0.0)          # default 2
    ctx4 = build_six_face_duo_context(12, 3, use_ext_bundle=True,
                                      oracle_conventions=True,
                                      omega=0.0, k2e_nord=4)
    assert ctx2["ectx"]["k2e_nord"] == 2
    assert ctx4["ectx"]["k2e_nord"] == 4


def test_nord2_tables_match_unforced_auth_fixture():
    """Our nord-2 tables vs the packed UNFORCED-generator fixture
    (fv3_duogrid_oracle_n2.npz; driver FATALs unless the auth tree's
    own default is 2).  Fail-loud if the fixture is absent — a missing
    certificate must never read as a pass (codex r6 P1)."""
    from pathlib import Path

    from legoesm.grids.fv3_native_halos import compute_fv3_native_k2e

    fx = (Path(__file__).parent / "fixtures"
          / "fv3_duogrid_oracle_n2.npz")
    assert fx.exists(), (
        "nord-2 fixture missing — regenerate with "
        "scripts/validate/fv3_native/gen_duogrid_oracle_n2.sh")
    z = np.load(fx)
    for res in (12, 24):          # C48 covered by the sbatch gate
        assert int(z[f"c{res}_k2e_nord"]) == 2
        ours = compute_fv3_native_k2e(res, remap_ng=3, k2e_nord=2)
        for fam in ("A", "B", "CX", "CY", "DX", "DY"):
            rows = {tuple(z[f"c{res}_{fam}_ij"][k]):
                    (int(z[f"c{res}_{fam}_loc"][k]),
                     z[f"c{res}_{fam}_coef"][k])
                    for k in range(len(z[f"c{res}_{fam}_ij"]))}
            for k, (fi, fj) in enumerate(ours[f"{fam}_ij"]):
                lv, cw = rows[(int(fi), int(fj))]
                assert int(ours[f"{fam}_loc"][k]) == lv, (res, fam, k)
                assert np.abs(ours[f"{fam}_coef"][k, :len(cw)]
                              - cw).max() < 1e-12, (res, fam, k)
