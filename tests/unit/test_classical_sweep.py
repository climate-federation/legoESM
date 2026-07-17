"""Classical physics-combo sweep config tests (JAX-free, login-node safe).

The sweep under ``config/aimip/wbcompare/classical_sweep/<combo>/`` trains the
SAME run_aimip ``classical`` variant with a DIFFERENT convection/turbulence/gwd/
microphysics/cloud combination per combo. run_aimip merges shallow, in order:

    base  <-  suite.cfg_overrides  <-  variant_<variant>.yaml

with the overlay resolved from the SUITE FILE'S OWN PARENT DIR
(``run_aimip.py`` L879-880; the WB2 eval bridge reproduces this in
``merged_cfg_from_suite``). Because the overlay is merged LAST, its schemes win
over ``cfg_overrides`` — so each combo needs its OWN ``variant_classical.yaml``
carrying that combo's schemes, else the shared wbcompare overlay (winner combo)
would be applied to ALL combos and the sweep would be a no-op.

The KEY test (``test_merge_yields_intended_combo_schemes``) reproduces that
shallow merge with a tiny stub — exactly like
``tests/unit/test_run_aimip_wb2_eval.py::test_merged_cfg_from_suite_matches_run_aimip_merge``
— and asserts each combo's merged schemes equal that combo's INTENDED schemes,
proving the winner combo is NOT silently applied to every combo.

No jax, no network: every YAML is loaded by ``yaml.safe_load`` and the merge is
a pure-Python shallow ``dict.update``.
"""
import glob
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _ROOT / "config" / "aimip" / "wbcompare" / "classical_sweep"

# Proven-valid scheme allowlists (the earlier stage-1 sweep set; a typo'd scheme
# would hit run_aimip's factory raise at runtime — this allowlist is the
# login-node guard against that).
_CONVECTION = {"tiedtke", "sbm", "edmf", "bechtold"}
_TURBULENCE = {"louis", "ysu", "tke", "smagorinsky"}
_GWD = {"mcfarlane", "hines"}
_MICROPHYSICS = {"sundqvist", "kessler", "morrison", "thompson"}
_CLOUD = {"xu_randall", "sundqvist"}

# The INTENDED schemes per combo (the sweep's ground truth). The merge test
# asserts the on-disk config resolves to EXACTLY these.
_INTENDED = {
    "winner":   dict(aimip_convection="edmf", aimip_turbulence="louis",
                     aimip_gwd="mcfarlane", aimip_microphysics="sundqvist",
                     aimip_cloud="xu_randall"),
    "tiedtke":  dict(aimip_convection="tiedtke", aimip_turbulence="louis",
                     aimip_gwd="mcfarlane", aimip_microphysics="sundqvist",
                     aimip_cloud="xu_randall"),
    "bechtold": dict(aimip_convection="bechtold", aimip_turbulence="louis",
                     aimip_gwd="mcfarlane", aimip_microphysics="kessler",
                     aimip_cloud="xu_randall"),
    "sbm_ysu":  dict(aimip_convection="sbm", aimip_turbulence="ysu",
                     aimip_gwd="hines", aimip_microphysics="morrison",
                     aimip_cloud="sundqvist"),
    "edmf_tke": dict(aimip_convection="edmf", aimip_turbulence="tke",
                     aimip_gwd="mcfarlane", aimip_microphysics="thompson",
                     aimip_cloud="xu_randall"),
}
_SCHEME_KEYS = (
    "aimip_convection", "aimip_turbulence", "aimip_gwd",
    "aimip_microphysics", "aimip_cloud",
)


def _load(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _combo_dirs():
    return sorted(
        d for d in _SWEEP_DIR.iterdir()
        if d.is_dir() and (d / "suite.yaml").exists()
    )


def _shallow_merge(base: dict, overlay: dict) -> dict:
    """Reproduce run_aimip._merge (shallow overlay wins)."""
    out = dict(base)
    out.update(overlay)
    return out


def _merged_cfg(combo_dir: Path) -> dict:
    """Reproduce run_aimip's suite merge for the ``classical`` variant:
    base <- cfg_overrides <- variant_classical.yaml (overlay from suite's parent
    dir). Mirrors run_aimip.py L840-882 with a pure-Python shallow merge stub.
    """
    suite = _load(combo_dir / "suite.yaml")
    base = _load(_ROOT / suite["base"])
    cfg = base
    if suite.get("cfg_overrides"):
        cfg = _shallow_merge(cfg, suite["cfg_overrides"])
    # overlay resolved from the SUITE's parent dir, exactly like the loader.
    overlay = _load(combo_dir / "variant_classical.yaml")
    cfg = _shallow_merge(cfg, overlay)
    cfg["aimip_variant"] = "classical"
    return cfg


# ------------------------------------------------------------------ discovery --
def test_sweep_dir_exists_with_expected_combos():
    dirs = _combo_dirs()
    names = {d.name for d in dirs}
    assert names == set(_INTENDED), (
        f"combo dirs on disk {names} != intended {set(_INTENDED)}")
    assert len(dirs) == 5


# -------------------------------------------------------------------- parsing --
@pytest.mark.parametrize("combo", sorted(_INTENDED))
def test_suite_and_overlay_parse(combo):
    """(1) each combo suite + its variant_classical overlay parse cleanly."""
    cdir = _SWEEP_DIR / combo
    suite = _load(cdir / "suite.yaml")
    overlay = _load(cdir / "variant_classical.yaml")
    assert suite["base"] == "config/aimip/aimip_era5.yaml"
    assert (_ROOT / suite["base"]).exists()
    assert suite["variants"] == ["classical"]
    assert overlay["aimip_variant"] == "classical"


def test_all_sweep_yaml_files_parse():
    files = sorted(glob.glob(str(_SWEEP_DIR / "*" / "*.yaml")))
    assert len(files) == 10, f"expected 10 yaml files (5 combos x 2), got {len(files)}"
    for f in files:
        assert _load(Path(f)), f"empty/invalid YAML: {f}"


# ------------------------------------ KEY TEST: merge yields the combo schemes --
@pytest.mark.parametrize("combo", sorted(_INTENDED))
def test_merge_yields_intended_combo_schemes(combo):
    """(2) EACH combo's classical schemes come out of the base<-cfg_overrides<-
    overlay merge EQUAL to that combo's intended schemes.

    This is the guard that the sweep really VARIES the physics: if every combo
    reused the shared wbcompare variant_classical.yaml (winner schemes), the
    non-winner combos would resolve to the winner's schemes and this fails.
    """
    cfg = _merged_cfg(_SWEEP_DIR / combo)
    for key, want in _INTENDED[combo].items():
        assert cfg[key] == want, (
            f"{combo}: merged {key}={cfg[key]!r} != intended {want!r} — the "
            f"overlay did not win the merge (winner combo silently applied?)")


def test_non_winner_combos_differ_from_winner():
    """A stronger form of the guard: at least one scheme in each non-winner
    combo's MERGED config differs from the winner's merged config, so the
    winner combo is provably NOT applied to all."""
    winner = _merged_cfg(_SWEEP_DIR / "winner")
    for combo in _INTENDED:
        if combo == "winner":
            continue
        merged = _merged_cfg(_SWEEP_DIR / combo)
        assert any(merged[k] != winner[k] for k in _SCHEME_KEYS), (
            f"{combo} merged schemes identical to winner — sweep is a no-op")


# ------------------------------------------------------------------ allowlist --
@pytest.mark.parametrize("combo", sorted(_INTENDED))
def test_schemes_in_proven_allowlist(combo):
    """(3) every scheme is in the proven allowlist (a typo'd scheme would hit
    run_aimip's factory raise at runtime; this is the login-node guard)."""
    cfg = _merged_cfg(_SWEEP_DIR / combo)
    assert cfg["aimip_convection"] in _CONVECTION
    assert cfg["aimip_turbulence"] in _TURBULENCE
    assert cfg["aimip_gwd"] in _GWD
    assert cfg["aimip_microphysics"] in _MICROPHYSICS
    assert cfg["aimip_cloud"] in _CLOUD


# ------------------------------------------------------ real multi-combo sweep --
def test_at_least_four_distinct_convection_schemes():
    """(4) >=4 distinct convection schemes across combos — proves a real
    multi-combo sweep, not winner-only."""
    convs = {_merged_cfg(_SWEEP_DIR / c)["aimip_convection"] for c in _INTENDED}
    assert len(convs) >= 4, f"only {len(convs)} distinct convection schemes: {convs}"


# -------------------------------------------------------------- classical + rrtmgp --
@pytest.mark.parametrize("combo", sorted(_INTENDED))
def test_radiation_rrtmgp_and_variant_classical(combo):
    """(5) all combos use aimip_radiation=rrtmgp and aimip_variant=classical."""
    cfg = _merged_cfg(_SWEEP_DIR / combo)
    assert cfg["aimip_radiation"] == "rrtmgp"
    assert cfg["aimip_variant"] == "classical"
    # the suite lists exactly the classical variant, and it is a valid run_aimip
    # variant name (combos vary SCHEMES, not the variant name).
    suite = _load(_SWEEP_DIR / combo / "suite.yaml")
    assert suite["variants"] == ["classical"]


# ------------------------------------------------------------ distinct outputs --
def test_output_dirs_all_distinct():
    """(6) every combo writes to a distinct output_dir (no scorecard clobber)."""
    outs = [_load(_SWEEP_DIR / c / "suite.yaml")["output_dir"] for c in _INTENDED]
    assert len(set(outs)) == len(outs), f"duplicate output_dirs: {outs}"
    for c, o in zip(_INTENDED, outs):
        assert o == f"results/aimip_wbcompare/classical_sweep/{c}"


# ------------------------------------------------------------- eval-year + epochs --
@pytest.mark.parametrize("combo", sorted(_INTENDED))
def test_cfg_overrides_epochs_and_eval_year(combo):
    """Every combo trains 12 epochs and evals on 2020 (held fixed across the
    sweep so combos are compared under an identical protocol)."""
    over = _load(_SWEEP_DIR / combo / "suite.yaml")["cfg_overrides"]
    assert over["aimip_n_epochs"] == 12
    assert over["eval_years"] == [2020]
    # the ACE2 loss block is present and identical to the wbcompare suite's.
    assert over["loss"]["residual_normalize"] is True
    assert over["loss"]["multi_step_hours"] == [6, 12]
