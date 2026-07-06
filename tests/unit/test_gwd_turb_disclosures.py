"""Faithfulness / labeling locks for the GWD + YSU disclosed-variant items.

Pure source-text checks (no JAX import — login-node safe) that lock the
docstring/label fixes for the physics-oracle disclosure findings:

  * gwd/hines is a single-bulk-amplitude WKB scheme, NOT the full Hines
    (1997) Doppler-spread spectrum — the module headline must say so.
  * turb/ysu models entrainment as a down-gradient Gaussian K-bump, NOT
    YSU's prescribed gradient-independent entrainment flux.
  * gwd/prognostic_spectral is experimental / opt-in and its momentum
    deposition is missing the ``sign(c - U_proj)`` factor (F-GWD-1), so its
    ``eps_gwd`` is NOT positive-definite — the false "positive-definite"
    wording must be dropped/qualified in both the scheme module and the
    shared ``GWDOutput`` container.

Each assertion FAILS against the pre-fix source (the misleading text was
present) and passes once the disclosure is corrected.
"""

from __future__ import annotations

from tests import _ratchet_audit as ra

_PKG = "packages/atmosphere/legoesm/atmosphere/physics"
_HINES = f"{_PKG}/gravity_wave_drag/hines.py"
_YSU = f"{_PKG}/turbulence/ysu.py"
_PROG = f"{_PKG}/gravity_wave_drag/prognostic_spectral.py"
_OUTPUT = f"{_PKG}/gravity_wave_drag/output.py"


def _read(rel: str) -> str:
    return (ra.repo_root() / rel).read_text(encoding="utf-8")


def test_hines_headline_discloses_single_bulk_amplitude():
    """hines.py must crisply state it is a single-bulk-amplitude WKB scheme,
    not the full Hines (1997) Doppler-spread spectrum."""
    src = _read(_HINES)
    assert "single-bulk-amplitude WKB" in src, (
        "hines.py headline must crisply state it is a single-bulk-amplitude "
        "WKB scheme, not the full Hines Doppler-spread spectrum."
    )
    assert ("NOT** the full Hines" in src) or ("NOT the full Hines" in src), (
        "hines.py must explicitly say it is NOT the full Hines (1997) "
        "Doppler-spread spectrum."
    )


def test_ysu_labels_gaussian_k_entrainment_as_variant():
    """ysu.py must label its Gaussian-K entrainment as an intentional
    down-gradient differentiable variant of published YSU."""
    src = _read(_YSU)
    assert "gradient-independent" in src, (
        "ysu.py must contrast against published YSU's gradient-independent "
        "entrainment flux."
    )
    assert "down-gradient" in src, (
        "ysu.py must label its entrainment as a down-gradient K bump."
    )
    assert "differentiable" in src.lower(), (
        "ysu.py must state the variant is chosen for differentiability."
    )


def test_prognostic_spectral_drops_false_positive_definite_claim():
    """prognostic_spectral.py must not claim eps_gwd is positive-definite,
    must mark itself experimental, and must disclose the missing
    sign(c - U_proj) deposition factor (F-GWD-1)."""
    src = _read(_PROG)
    # Pre-fix inline comment claimed the column dissipation is positive-definite,
    # contradicting the F-GWD-1 FIXME (missing sign factor -> eps_gwd can be < 0).
    assert "positive-definite: KE lost by the mean flow" not in src, (
        "prognostic_spectral.py still claims eps_gwd is positive-definite — "
        "that contradicts the F-GWD-1 FIXME (missing sign(c-U) means eps_gwd "
        "can be negative)."
    )
    assert "EXPERIMENTAL" in src, (
        "prognostic_spectral.py docstring must mark the scheme experimental."
    )
    assert "sign(c - U_proj)" in src, (
        "prognostic_spectral.py docstring must disclose the missing "
        "sign(c - U_proj) deposition factor (F-GWD-1)."
    )
    assert "NOT guaranteed positive" in src, (
        "prognostic_spectral.py must state eps_gwd is not guaranteed positive "
        "for this scheme."
    )


def test_gwd_output_qualifies_eps_gwd_positivity():
    """The shared GWDOutput.eps_gwd doc must no longer make an unconditional
    positive-definite claim — it is false for prognostic_spectral."""
    src = _read(_OUTPUT)
    assert "Positive-definite: eps_gwd" not in src, (
        "GWDOutput.eps_gwd doc still makes an unconditional positive-definite "
        "claim; it is false for the experimental prognostic_spectral scheme."
    )
    assert "NOT guaranteed positive" in src, (
        "GWDOutput.eps_gwd doc must qualify that eps_gwd is not guaranteed "
        "positive for prognostic_spectral (F-GWD-1)."
    )
