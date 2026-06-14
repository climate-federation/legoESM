"""Tests for the ocean ``setup:`` selector and the idealized-case templates.

Issue #388 Ask#2: the idealized ocean experiments
(``legoesm.ocean.experiments.AVAILABLE_EXPERIMENTS``) used to be reachable only
from Python. A template can now name one via a top-level ``setup:`` block, which
``OceanExperimentConfig`` validates (against the live registry + the
experiment's ``grid_support`` + the matrix grid set) and routes by EXACT match
to the existing matrix runner (``run_ocean_test_matrix.py --only =<name>``) —
keeping the *setup* axis distinct from the ``ocean:`` physics *recipe*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legoesm.ocean.config import OceanExperimentConfig
from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OCEAN_TEMPLATES = _REPO_ROOT / "config" / "templates" / "ocean"

# Registry experiments deliberately NOT (yet) exposed as templates, with the
# reason — keeps coverage honest (no silent gap).  Shrink as templates are
# added; the *_NO_EXACT_CASE entries can only be wired once the matrix runner
# gains a single canonical case for them.
_NO_EXACT_MATRIX_CASE = {
    "rest_state",     # matrix has only rest_state_{stratified,uniform}_{no,with}_land
    "baroclinic",     # matrix exposes baroclinic_gyre*, not a bare `baroclinic` case
    "regional_gyre",  # matrix exposes single_gyre / *_regional, not `regional_gyre`
}
_PROCEDURAL_DEFERRED = {
    "dino", "neverworld2_lite", "isomip_plus",  # Levy grid / ice-shelf / grid-restricted
}
_EXACT_CASE_TODO: set[str] = set()
# exact-match experiments not yet templated — now EMPTY: every experiment with a
# single exact matrix case ships a template. Only the procedural-deferred and
# no-exact-matrix-case experiments above remain unwired (with documented
# reasons). Re-populate only if a NEW exact-match experiment is registered.


def _ocean_template_paths() -> list[Path]:
    return sorted(_OCEAN_TEMPLATES.glob("*.yaml"))


def _matrix_case_pairs() -> set[tuple[str, str]]:
    """The live (case, grid_type) catalog from the matrix runner."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "matrix"))
    import run_ocean_test_matrix as M  # noqa: PLC0415
    return {(c.case, c.grid_type) for c in M._build_test_matrix()}


# --------------------------------------------------------------------- templates
class TestOceanSetupTemplates:
    def test_templates_exist(self):
        assert _ocean_template_paths(), f"no setup templates under {_OCEAN_TEMPLATES}"

    @pytest.mark.parametrize("path", _ocean_template_paths(),
                             ids=lambda p: p.name)
    def test_template_validates_and_runs(self, path):
        cfg = OceanExperimentConfig.from_yaml(str(path))
        assert cfg.get("model.type") == "ocean_only"
        name = cfg.get("setup.name")
        grid = cfg.get("setup.grid")
        assert name in AVAILABLE_EXPERIMENTS, name
        cfg.validate_strict()
        cmd = cfg.run_command(str(path))
        assert "run_ocean_test_matrix.py" in cmd
        assert f"--only ={name}" in cmd          # EXACT match, not substring
        assert f"--grid {grid}" in cmd
        assert "run_omip_core2.py" not in cmd


# ----------------------------------------------------------------- selector unit
class TestSetupSelectorValidation:
    def _cfg(self, setup, **extra):
        d = {"model": {"type": "ocean_only"}, "setup": setup}
        d.update(extra)
        return OceanExperimentConfig.from_dict(d)

    def test_valid_setup_passes(self):
        self._cfg({"name": "lock_exchange", "grid": "latlon"}).validate_strict()

    def test_unknown_experiment_name_raises(self):
        with pytest.raises(ValueError, match="not a known ocean experiment"):
            self._cfg({"name": "no_such_case", "grid": "latlon"}).validate_strict()

    def test_grid_not_in_matrix_set_raises(self):
        # gridless experiment (no grid_support) still rejects a non-matrix grid
        with pytest.raises(ValueError, match="not a valid ocean matrix grid"):
            self._cfg({"name": "barotropic_wave", "grid": "nonsense"}).validate_strict()

    def test_unsupported_grid_for_experiment_raises(self):
        # acc_channel supports only *_channel grids (grid_support gate)
        with pytest.raises(ValueError, match="not supported by experiment"):
            self._cfg({"name": "acc_channel", "grid": "latlon"}).validate_strict()

    def test_unknown_setup_field_raises(self):
        with pytest.raises(ValueError, match="unknown setup field"):
            self._cfg({"name": "lock_exchange", "grid": "latlon",
                       "gird": "typo"}).validate_strict()

    def test_non_mapping_setup_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            self._cfg("lock_exchange").validate_strict()

    def test_missing_grid_raises(self):
        with pytest.raises(ValueError, match="setup.grid must be"):
            self._cfg({"name": "lock_exchange"}).validate_strict()

    def test_ocean_overrides_with_setup_raise(self):
        with pytest.raises(ValueError, match="not applied to a named"):
            self._cfg({"name": "lock_exchange", "grid": "latlon"},
                      ocean={"A_h": 1234.0}).validate_strict()

    def test_bad_run_control_raises(self):
        with pytest.raises(ValueError, match="setup.levels must be"):
            self._cfg({"name": "lock_exchange", "grid": "latlon",
                       "levels": 0}).validate_strict()
        with pytest.raises(ValueError, match="setup.dt_seconds must be"):
            self._cfg({"name": "lock_exchange", "grid": "latlon",
                       "dt_seconds": -1.0}).validate_strict()

    def test_bool_levels_rejected(self):
        # ``levels: true`` must NOT coerce to 1
        with pytest.raises(ValueError, match="setup.levels must be"):
            self._cfg({"name": "lock_exchange", "grid": "latlon",
                       "levels": True}).validate_strict()

    def test_nonfinite_dt_rejected(self):
        with pytest.raises(ValueError, match="setup.dt_seconds must be"):
            self._cfg({"name": "lock_exchange", "grid": "latlon",
                       "dt_seconds": float("inf")}).validate_strict()

    def test_quick_must_be_bool(self):
        with pytest.raises(ValueError, match="setup.quick must be a bool"):
            self._cfg({"name": "lock_exchange", "grid": "latlon",
                       "quick": "yes"}).validate_strict()


# --------------------------------------------------------------- run_command map
class TestSetupRunCommand:
    def _cfg(self, setup):
        return OceanExperimentConfig.from_dict(
            {"model": {"type": "ocean_only"}, "setup": setup,
             "output": {"path": "out/x/"}})

    def test_minimal_command_exact_match(self):
        cmd = self._cfg({"name": "lock_exchange", "grid": "latlon"}).run_command()
        assert "--only =lock_exchange" in cmd
        assert "--grid latlon" in cmd
        assert "--output out/x/" in cmd
        # nothing pinned => no run-control flags leaked from OMIP defaults
        for flag in ("--levels", "--dt ", "--days", "--quick", "--resolution"):
            assert flag not in cmd, flag

    def test_run_controls_mapped(self):
        cmd = self._cfg({
            "name": "lock_exchange", "grid": "latlon",
            "levels": 20, "dt_seconds": 30.0, "duration_days": 0.5,
            "resolution": "72x144", "quick": True,
        }).run_command()
        assert "--levels 20" in cmd
        assert "--dt 30.0" in cmd
        assert "--days 0.5" in cmd
        assert "--resolution 72x144" in cmd
        assert "--quick" in cmd

    def test_signature_tracks_setup_grid(self):
        a = self._cfg({"name": "lock_exchange", "grid": "latlon"}).signature()
        b = self._cfg({"name": "lock_exchange", "grid": "latlon_regional"}).signature()
        assert a != b, "signature must change with setup.grid"

    def test_signature_ignores_omip_only_overrides(self):
        # a `time:` override is a NO-OP on the matrix path -> signature must not
        # change, so init_experiment flags it (instead of silently accepting).
        base = self._cfg({"name": "lock_exchange", "grid": "latlon"})
        over = OceanExperimentConfig.from_dict(
            {"model": {"type": "ocean_only"},
             "setup": {"name": "lock_exchange", "grid": "latlon"},
             "output": {"path": "out/x/"},
             "time": {"duration_days": 999}})
        assert base.signature() == over.signature()


# -------------------------------------------------------------------- coverage
class TestSetupTemplateCoverage:
    def test_shipped_templates_select_a_real_matrix_case(self):
        """Every shipped (setup.name, setup.grid) must be a concrete case in the
        live matrix catalog — the runnability guarantee (closes the gap that a
        registry name need not have an exact matrix case)."""
        pairs = _matrix_case_pairs()
        for path in _ocean_template_paths():
            cfg = OceanExperimentConfig.from_yaml(str(path))
            name, grid = cfg.get("setup.name"), cfg.get("setup.grid")
            assert (name, grid) in pairs, (
                f"{path.name}: ({name!r}, {grid!r}) is not a matrix case "
                f"(--only ={name} --grid {grid} would select nothing)")

    def test_coverage_accounting_is_complete(self):
        """No registry experiment is silently unaccounted-for: every name is
        either shipped, an explicit exact-match TODO, has no exact matrix case,
        or is a deferred procedural case."""
        shipped = {
            OceanExperimentConfig.from_yaml(str(p)).get("setup.name")
            for p in _ocean_template_paths()
        }
        buckets = [shipped, _EXACT_CASE_TODO, _NO_EXACT_MATRIX_CASE,
                   _PROCEDURAL_DEFERRED]
        accounted = set().union(*buckets)
        missing = set(AVAILABLE_EXPERIMENTS) - accounted
        assert not missing, f"unaccounted ocean experiments: {sorted(missing)}"
        # The four buckets must PARTITION the registry — pairwise disjoint
        # (sum of sizes == size of the union), so a name can never be both
        # shipped and TODO/deferred, now or after a future edit.
        assert sum(len(b) for b in buckets) == len(accounted), (
            "coverage buckets overlap: "
            f"{[sorted(a & b) for i, a in enumerate(buckets) for b in buckets[i + 1:] if a & b]}")
        # and nothing extraneous beyond the registry
        assert accounted <= set(AVAILABLE_EXPERIMENTS), (
            f"unknown names in coverage buckets: "
            f"{sorted(accounted - set(AVAILABLE_EXPERIMENTS))}")

    def test_validate_passing_but_not_instantiated_is_runner_guarded(self):
        """A (name, grid) can pass the package-level validate_strict (grid in
        the experiment's grid_support + the matrix grid set) yet have no
        concrete matrix case — e.g. lock_exchange/mpas. validate_strict cannot
        see the scripts/ matrix catalog (layering), so the matrix runner is the
        safety net: ``--only =name`` selecting zero cases raises (not a silent
        no-op). This test pins that boundary so the contract is explicit."""
        cfg = OceanExperimentConfig.from_dict(
            {"model": {"type": "ocean_only"},
             "setup": {"name": "lock_exchange", "grid": "mpas"}})
        cfg.validate_strict()  # passes: mpas is in lock_exchange grid_support
        assert ("lock_exchange", "mpas") not in _matrix_case_pairs()
        # the runner guard (run_ocean_test_matrix.main) raises SystemExit on an
        # empty exact-match selection — verified by inspection of the
        # `if str(args.only).startswith("="): raise SystemExit` guard.
