"""The one del4 law the scaling harnesses share (scripts/bench/hyperdiff.py)."""
import importlib.util
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "hyperdiff.py"
_spec = importlib.util.spec_from_file_location("bench_hyperdiff", _P)
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)


def test_icosahedral_anchor_and_dx4_law():
    f = lambda n: hd.hyperdiff_coeff(n, "icosahedral")  # noqa: E731
    assert f(3) == f(4) == 1.0e16          # smoke meshes keep their value
    for n in range(5, 11):                 # dx halves per level -> 16x
        assert f(n) == pytest.approx(f(n - 1) / 16.0)


@pytest.mark.parametrize("grid,ref_n,ref", [("spectral", 42, 2.5e16),
                                            ("latlon", 64, 5e16),
                                            ("cubed-sphere", 48, 5e16)])
def test_other_grids_keep_their_anchor_and_dx4_law(grid, ref_n, ref):
    assert hd.hyperdiff_coeff(ref_n, grid) == pytest.approx(ref)
    assert hd.hyperdiff_coeff(2 * ref_n, grid) == pytest.approx(ref / 16.0)


def test_unknown_grid_raises():
    with pytest.raises(ValueError):
        hd.hyperdiff_coeff(48, "cubed_sphere")


def test_both_harnesses_use_the_shared_helper():
    bench = _P.parent
    for script in ("run_levante_gpu_scaling.py", "bench_mpas_spmd_scaling.py"):
        src = (bench / script).read_text()
        assert "from hyperdiff import hyperdiff_coeff" in src, script
        assert "def _hyperdiff_coeff" not in src and "nu_del4=1e16" not in src, script
