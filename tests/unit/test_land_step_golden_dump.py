import numpy as np

from scripts.validate.land_step_golden_dump import compare, dump


def test_dump_is_reproducible_and_compare_detects_a_change(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    dump(str(a)); dump(str(b))
    assert compare(str(a), str(b)) == 0
    d = dict(np.load(a))
    k = sorted(d)[0]
    d[k] = d[k] + 1e-12
    np.savez(b, **d)
    assert compare(str(a), str(b)) == 1
