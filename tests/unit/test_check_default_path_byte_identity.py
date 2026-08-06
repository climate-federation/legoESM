"""Direct test for ``scripts/validate/check_default_path_byte_identity.py``.

Every new ``.py`` gets a direct test (CLAUDE.md), and a comparator that decides
whether a byte-identity claim holds must itself be proven non-vacuous: each
control is fired with synthetic checkpoint directories.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.check_default_path_byte_identity import main


def _write(tmp_path, name, day=1, **arrays):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    np.savez(d / f"checkpoint_day_{day:04d}.npz", **arrays)
    return str(d)


def _good(seed=0, bump=0.0):
    """A realistically-shaped checkpoint: enough arrays/elements to clear the
    comparator's own vacuity floors, with a few constant fields as a real one
    has."""
    rng = np.random.default_rng(seed)
    a = {
        "T": 250.0 + rng.normal(size=(500, 30)) + bump,
        "u": rng.normal(size=(700, 30)),
        "p_s": 1e5 + rng.normal(size=(500,)),
        "phis": np.zeros(500),
    }
    for i in range(6):
        a[f"trc_{i}"] = np.abs(rng.normal(size=(500, 30)))
    for i in range(3):
        a[f"const_{i}"] = np.zeros((500, 30))
    return a


def test_passes_when_identical_and_the_differ_run_differs(tmp_path):
    a = _good()
    base = _write(tmp_path, "base", **a)
    same = _write(tmp_path, "same", **a)
    differ = _write(tmp_path, "differ", **_good(bump=1e-6))
    assert main(["--base", base, "--same", same, "--differ", differ]) == 0


def test_fails_when_the_default_path_moved(tmp_path):
    base = _write(tmp_path, "base", **_good())
    same = _write(tmp_path, "same", **_good(bump=1e-12))
    with pytest.raises(SystemExit, match="DEFAULT PATH PERTURBED"):
        main(["--base", base, "--same", same])


def test_fails_when_the_differ_control_does_not_differ(tmp_path):
    a = _good()
    base = _write(tmp_path, "base", **a)
    same = _write(tmp_path, "same", **a)
    differ = _write(tmp_path, "differ", **a)
    with pytest.raises(SystemExit, match="NON-VACUITY FAILED"):
        main(["--base", base, "--same", same, "--differ", differ])


def test_fails_on_all_constant_arrays(tmp_path):
    """The exact false pass a prior agent on this campaign got: every compared
    array was an all-zero diagnostic, so equality proved nothing."""
    z = {f"d{i}": np.zeros((200, 30)) for i in range(12)}
    z.update(T=np.zeros((200, 30)), u=np.zeros((300, 30)), p_s=np.zeros(200))
    base = _write(tmp_path, "base", **z)
    same = _write(tmp_path, "same", **z)
    with pytest.raises(SystemExit, match="VACUOUS"):
        main(["--base", base, "--same", same])


def test_fails_on_a_too_small_comparison(tmp_path):
    tiny = {"T": np.ones((2, 2)), "u": np.ones((2, 2)), "p_s": np.ones(2)}
    base = _write(tmp_path, "base", **tiny)
    same = _write(tmp_path, "same", **tiny)
    with pytest.raises(SystemExit, match="VACUOUS"):
        main(["--base", base, "--same", same])


def test_fails_when_a_required_prognostic_is_missing(tmp_path):
    a = _good()
    a.pop("u")
    base = _write(tmp_path, "base", **a)
    same = _write(tmp_path, "same", **a)
    with pytest.raises(SystemExit, match="required field 'u' missing"):
        main(["--base", base, "--same", same])


# --- codex round 3: the controls the first version did not fire -------------


def test_signed_zero_and_dtype_are_not_bit_identical(tmp_path):
    """``np.array_equal`` accepts +0.0 vs -0.0 and cross-dtype equality; a
    BYTE-identity claim must not."""
    a = _good()
    a["T"] = np.where(np.abs(a["T"]) > 1e9, 0.0, a["T"])  # keep shape/dtype
    a["phis"] = np.zeros(500)
    base = _write(tmp_path, "base", **a)
    b = dict(a, phis=np.zeros(500) * -1.0)  # -0.0 everywhere
    assert np.array_equal(a["phis"], b["phis"])  # value-equal, byte-different
    same = _write(tmp_path, "same", **b)
    with pytest.raises(SystemExit, match="DEFAULT PATH PERTURBED"):
        main(["--base", base, "--same", same])

    # dtype control with IDENTICAL VALUES on both sides (codex round 4): build
    # the float64 reference FROM a float32 source so the only difference is the
    # storage dtype, and assert value-equality before comparing.
    u32 = a["u"].astype(np.float32)
    a64 = dict(a, u=u32.astype(np.float64))
    assert np.array_equal(a64["u"], u32)
    base64 = _write(tmp_path, "base64", **a64)
    same32 = _write(tmp_path, "same32", **dict(a, u=u32))
    with pytest.raises(SystemExit, match="DEFAULT PATH PERTURBED"):
        main(["--base", base64, "--same", same32])


def test_fails_on_a_constant_required_field(tmp_path):
    """A constant prognostic compares equal whatever the model did — even when
    the GLOBAL non-constant threshold is satisfied by other arrays."""
    a = _good()
    a["p_s"] = np.full(500, 1e5)
    base = _write(tmp_path, "base", **a)
    same = _write(tmp_path, "same", **a)
    with pytest.raises(SystemExit, match="'p_s' is constant"):
        main(["--base", base, "--same", same])


def test_fails_on_a_non_finite_required_field(tmp_path):
    a = _good()
    a["T"][0, 0] = np.nan
    base = _write(tmp_path, "base", **a)
    same = _write(tmp_path, "same", **a)
    with pytest.raises(SystemExit, match="'T' is non-finite"):
        main(["--base", base, "--same", same])


def test_compares_non_numeric_payloads_too(tmp_path):
    """A real MPAS checkpoint carries string metadata (``tracer_names``,
    ``physstate_meta_*``); a changed scheme name there is a changed run, so the
    comparison must not be numeric-only (codex round 4)."""
    a = _good()
    a["tracer_names"] = np.array(["q_v", "q_c"])
    base = _write(tmp_path, "base", **a)
    same = _write(tmp_path, "same", **dict(a, tracer_names=np.array(["q_v", "q_i"])))
    with pytest.raises(SystemExit, match="DEFAULT PATH PERTURBED"):
        main(["--base", base, "--same", same])
    dropped = _write(tmp_path, "dropped",
                     **{k: v for k, v in a.items() if k != "tracer_names"})
    with pytest.raises(SystemExit, match="key sets differ base vs --same"):
        main(["--base", base, "--same", dropped])


def test_fails_when_the_differ_run_has_a_different_schema(tmp_path):
    a = _good()
    base = _write(tmp_path, "base", **a)
    same = _write(tmp_path, "same", **a)
    d = dict(a)
    d.pop("trc_0")
    differ = _write(tmp_path, "differ", **d)
    with pytest.raises(SystemExit, match="key sets differ base vs --differ"):
        main(["--base", base, "--same", same, "--differ", differ])
