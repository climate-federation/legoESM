"""Spec-first guard: every physics scheme module declares a machine-checked
``__physics_contract__``.

This is the "Domain Architect dictates logic" half of the harness (see
``docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md``). Before the body of
a physics parameterization is written, the domain expert pins its *contract* —
the units of every input/output, the sign convention, what it conserves, whether
it is differentiable, the literature reference, and an idealized acceptance
criterion. The AI fills in the body; reviewers and the ``physics-validator`` agent
check the body against the *declared* contract instead of re-inferring intent. The
contract is a module-level dict literal::

    __physics_contract__ = {
        "summary": "...",                       # one line
        "inputs":  {"u": "m/s", "T": "K", ...},  # name -> unit
        "outputs": {"du_dt": "m/s^2", ...},
        "sign_convention": "...",
        "conserves": ["energy"],                # subset of _VALID_CONSERVES; "none" alone
        "differentiable": True,
        "reference": "Author (year) / DOI",
        "idealized_test": "rest state -> zero tendency; ...",
    }

**Every** ``.py`` under ``atmosphere/physics`` and ``ocean/physics`` is partitioned
into exactly one of three pinned sets, so nothing can dodge the requirement:
  * ``EXCLUDED`` — helpers/plumbing that are not single-tendency schemes
    (``__init__``/``config``/``output``/``integration``/state/thermo utilities and
    the RRTMGP radiative-transfer *engine* data/optics/lookup internals).
  * ``CONTRACT_TODO`` — scheme modules not yet annotated (SHRINK-ONLY: annotate a
    module then delete it here).
  * annotated — everything else; each MUST carry a valid ``__physics_contract__``.
A new physics file appears in none of the pinned sets, so it is treated as
annotated and FAILS unless it ships a contract (a new scheme) or is added to
``EXCLUDED`` (a new helper) — forcing an explicit classification.

The contract is validated by AST literal-eval (no heavy imports), so it must be a
pure literal. A tripwire, not a proof: the contract records intent;
conservation/equivariance/analytic tests verify the body meets it. Self-tests
prove non-vacuity.
"""

from __future__ import annotations

import ast

import pytest

from tests import _ratchet_audit as ra

_VALID_CONSERVES = frozenset(
    {"mass", "energy", "moisture", "momentum", "tracer", "salt", "none"}
)
_REQUIRED_KEYS = frozenset(
    {
        "summary",
        "inputs",
        "outputs",
        "sign_convention",
        "conserves",
        "differentiable",
        "reference",
        "idealized_test",
    }
)


def _is_physics_py(rel: str) -> bool:
    return "/physics/" in rel and (
        rel.startswith("packages/atmosphere/") or rel.startswith("packages/ocean/")
    )


# Helpers/plumbing — NOT single-tendency schemes: package ``__init__``; the
# ``config``/``output``/``integration`` trio per category; shared ``_shared``/
# ``combined``/``physics_state``/``tendencies``/``thermodynamics``/``mpas_physics``
# utilities; and the RRTMGP radiative-transfer *engine* (gas/cloud optics, lookup
# tables, data loaders, interpolation, kernel ops, constants, rte_utils). The
# RRTMGP *solvers* (two_stream/monochromatic_two_stream/rrtmgp.py) are schemes
# and live in CONTRACT_TODO.
EXCLUDED: frozenset[str] = frozenset(
    {
        "packages/atmosphere/legoesm/atmosphere/physics/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/_shared.py",
        "packages/atmosphere/legoesm/atmosphere/physics/clouds/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/clouds/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/combined.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/output.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/output.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/grid.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/output.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/particles.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/box_model.py",
        "packages/atmosphere/legoesm/atmosphere/physics/physics_state.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/output.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/config/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/config/radiative_transfer.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/constants.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/interpolation.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/kernel_ops.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/atmospheric_state.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/cloud_optics.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/constants.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/data_loader_base.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/gas_optics.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_cloud_optics.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_base.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_longwave.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_shortwave.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_volume_mixing_ratio.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/optics.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/optics_base.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/optics_utils.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/rte_utils.py",
        # mc3d: 3D Monte-Carlo ray tracer is a SPATIAL SOLVER (like the rte/
        # two-stream tree above), not a single-tendency parameterization. Its
        # energy-conservation invariant is gated analytically in
        # tests/unit/test_mc3d_raytracer.py, not via __physics_contract__.
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/emulator.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/knull_grid.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/mie.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/parallel.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/photon_walk.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/plane_adapter.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/qrng.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/raytracer_lw.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/raytracer_sw.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/sampling.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/tally.py",
        "packages/atmosphere/legoesm/atmosphere/physics/thermodynamics.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/__init__.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/output.py",
        # MPI plumbing: slices a deployed per-column override to a rank's tile,
        # not a single-tendency physics scheme.
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py",
        "packages/ocean/legoesm/ocean/physics/__init__.py",
        "packages/ocean/legoesm/ocean/physics/bottom_drag/__init__.py",
        "packages/ocean/legoesm/ocean/physics/bottom_drag/config.py",
        "packages/ocean/legoesm/ocean/physics/bottom_drag/integration.py",
        "packages/ocean/legoesm/ocean/physics/bottom_drag/output.py",
        "packages/ocean/legoesm/ocean/physics/combined.py",
        "packages/ocean/legoesm/ocean/physics/convection/__init__.py",
        "packages/ocean/legoesm/ocean/physics/convection/config.py",
        "packages/ocean/legoesm/ocean/physics/convection/integration.py",
        "packages/ocean/legoesm/ocean/physics/convection/output.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/__init__.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/integration.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/output.py",
        "packages/ocean/legoesm/ocean/physics/mpas_physics.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/__init__.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/_shared.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/config.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/integration.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/output.py",
        "packages/ocean/legoesm/ocean/physics/tendencies.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/__init__.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/integration.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/output.py",
    }
)

# Scheme modules not yet carrying a contract (iter 2026-06-09). SHRINK-ONLY.
CONTRACT_TODO: frozenset[str] = frozenset(
    {
        "packages/atmosphere/legoesm/atmosphere/physics/clouds/cloud_fraction.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/_emanuel_mixing.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/_plume.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/_triggers.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/_zm_dilute.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/bechtold.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/dca.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/emanuel.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/kain_fritsch.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/kuo.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/mass_flux.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/sbm.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/tiedtke.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/e3sm_cam.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/hines.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py",
        "packages/atmosphere/legoesm/atmosphere/physics/learned_column.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/_thompson_snow.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/kessler.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/ml_emulator.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/p3.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/seifert_beheng.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/sundqvist.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/thompson.py",
        "packages/atmosphere/legoesm/atmosphere/physics/ml_parameterization.py",
        "packages/atmosphere/legoesm/atmosphere/physics/neural_physics.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/gray.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/ozone_ml.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/ozone_mls.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/monochromatic_two_stream.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/solar.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb_lite.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/edmf.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/holtslag_boville.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/lasd_core.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/louis.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/mynn25.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/pbl_height.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/smagorinsky.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/tke.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/vertical_diffusion.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/vreman.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/ysu.py",
        "packages/ocean/legoesm/ocean/physics/convection/enhanced_diffusion.py",
        "packages/ocean/legoesm/ocean/physics/convection/plume.py",
        "packages/ocean/legoesm/ocean/physics/ice_shelf.py",
        "packages/ocean/legoesm/ocean/physics/ice_shelf_basal_melt.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/backscatter.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/biharmonic.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/eke.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi.py",
        # MLE shared core: config + Fox-Kemper formulas/helpers (coefficient,
        # vertical structure, MLE-MLD + buoyancy), NOT a single-tendency scheme.
        # The scheme (mle_latlon_cgrid.py) carries the __physics_contract__.
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/harmonic.py",
        "packages/ocean/legoesm/ocean/physics/mixing.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/bulk_formulas.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/external.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/prescribed.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/restoring.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/wind_profiles.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/constant.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/kpp.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/richardson.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/tidal.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py",
    }
)


def extract_contract(src: str):
    """Module-level ``__physics_contract__`` literal (``Assign`` or ``AnnAssign``);
    ``None`` if absent, sentinel ``"__UNPARSEABLE__"`` if present but not a literal."""
    tree = ast.parse(src)
    for node in tree.body:  # module level only
        target_match = (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__physics_contract__" for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__physics_contract__"
            and node.value is not None
        )
        if target_match:
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                return "__UNPARSEABLE__"
    return None


def validate_contract(contract) -> list[str]:
    """Schema errors for a contract dict (empty list == valid)."""
    if contract == "__UNPARSEABLE__":
        return ["__physics_contract__ is not a pure literal (AST literal-eval failed)"]
    if not isinstance(contract, dict):
        return ["__physics_contract__ must be a dict"]
    errors: list[str] = []
    keys = set(contract)
    if keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - keys)
        extra = sorted(keys - _REQUIRED_KEYS)
        if missing:
            errors.append(f"missing keys: {missing}")
        if extra:
            errors.append(f"unexpected keys: {extra}")
    for key in ("summary", "sign_convention", "reference", "idealized_test"):
        if key in contract and (not isinstance(contract[key], str) or not contract[key].strip()):
            errors.append(f"{key!r} must be a non-empty string")
    for key in ("inputs", "outputs"):
        if key in contract:
            val = contract[key]
            if not isinstance(val, dict) or not val:
                errors.append(f"{key!r} must be a non-empty dict of name->unit")
            else:
                for nm, unit in val.items():
                    if not isinstance(nm, str) or not nm.strip():
                        errors.append(f"{key} has a non-string variable name {nm!r}")
                    if not isinstance(unit, str) or not unit.strip():
                        errors.append(f"{key}[{nm!r}] unit must be a non-empty string")
    if "conserves" in contract:
        cons = contract["conserves"]
        if not isinstance(cons, list) or not cons:
            errors.append("'conserves' must be a non-empty list")
        else:
            bad = [c for c in cons if c not in _VALID_CONSERVES]
            if bad:
                errors.append(f"'conserves' has invalid entries {bad}; allowed {sorted(_VALID_CONSERVES)}")
            if "none" in cons and len(cons) > 1:
                errors.append("'conserves' = 'none' must be the sole entry (not mixed with real quantities)")
    if "differentiable" in contract and not isinstance(contract["differentiable"], bool):
        errors.append("'differentiable' must be a bool")
    return errors


_ALL_PHYSICS = sorted(ra.rel(f) for f in ra.discover_py_files() if _is_physics_py(ra.rel(f)))
_SCHEME_FILES = sorted(set(_ALL_PHYSICS) - EXCLUDED)  # CONTRACT_TODO ∪ annotated


def test_discovery_sane() -> None:
    ra.assert_discovery_sane(ra.discover_py_files())
    assert len(_ALL_PHYSICS) >= 150, (
        f"only {len(_ALL_PHYSICS)} physics .py discovered (expected ~156) — "
        f"discovery scope likely broken."
    )


def test_partition_is_disjoint_and_pinned_to_real_files() -> None:
    overlap = sorted(EXCLUDED & CONTRACT_TODO)
    assert not overlap, f"EXCLUDED & CONTRACT_TODO overlap: {overlap}"
    allset = set(_ALL_PHYSICS)
    stale_ex = sorted(EXCLUDED - allset)
    stale_todo = sorted(CONTRACT_TODO - allset)
    assert not stale_ex, f"EXCLUDED names files not discovered (rot/scope regression): {stale_ex}"
    assert not stale_todo, f"CONTRACT_TODO names files not discovered: {stale_todo}"


@pytest.mark.parametrize("rel", _SCHEME_FILES)
def test_physics_scheme_has_valid_contract(rel: str) -> None:
    """Every non-EXCLUDED physics file (a scheme) must carry a valid contract OR
    still be in CONTRACT_TODO. A new scheme file is in neither pinned set and so
    must ship a contract."""
    contract = extract_contract((ra.repo_root() / rel).read_text())
    if contract is None:
        assert rel in CONTRACT_TODO, (
            f"{rel} is a physics scheme module with no ``__physics_contract__``. "
            f"Add one (see this file's docstring); a pre-existing module must be in "
            f"CONTRACT_TODO; a new helper must be added to EXCLUDED with a reason."
        )
        return
    errors = validate_contract(contract)
    assert not errors, f"{rel} has an invalid __physics_contract__:\n  " + "\n  ".join(errors)
    assert rel not in CONTRACT_TODO, (
        f"{rel} now has a valid contract — remove it from CONTRACT_TODO (shrink-only)."
    )


def test_contract_todo_entries_still_lack_contracts() -> None:
    annotated = [
        rel for rel in CONTRACT_TODO
        if (ra.repo_root() / rel).is_file()
        and extract_contract((ra.repo_root() / rel).read_text()) is not None
    ]
    assert not annotated, (
        "CONTRACT_TODO entries now have contracts — remove them (shrink-only): "
        + ", ".join(sorted(annotated))
    )


def test_excluded_files_exist() -> None:
    missing = sorted(rel for rel in EXCLUDED if not (ra.repo_root() / rel).is_file())
    assert not missing, f"EXCLUDED names non-existent files: {missing}"


# ---------------------------------------------------------------------------
# Non-vacuity self-tests
# ---------------------------------------------------------------------------
_GOOD = {
    "summary": "s",
    "inputs": {"u": "m/s"},
    "outputs": {"du_dt": "m/s^2"},
    "sign_convention": "drag opposes u",
    "conserves": ["energy"],
    "differentiable": True,
    "reference": "Author (2024)",
    "idealized_test": "u=0 -> 0",
}


def test_validator_accepts_good_contract() -> None:
    assert validate_contract(_GOOD) == []


def test_validator_flags_missing_extra_and_bad_values() -> None:
    assert validate_contract({})  # missing everything
    assert validate_contract({**_GOOD, "extra": 1})  # extra key rejected
    assert validate_contract({**_GOOD, "conserves": ["wrong"]})
    assert validate_contract({**_GOOD, "conserves": ["none", "energy"]})  # none not exclusive
    assert validate_contract({**_GOOD, "differentiable": "yes"})
    assert validate_contract({**_GOOD, "inputs": {}})
    assert validate_contract({**_GOOD, "reference": "  "})


def test_extractor_handles_assign_annassign_and_nonliteral() -> None:
    assert extract_contract("__physics_contract__ = {'a': 1}\n") == {"a": 1}
    assert extract_contract("__physics_contract__: dict = {'a': 1}\n") == {"a": 1}
    assert extract_contract("x = 1\n") is None
    assert extract_contract("__physics_contract__ = dict(a=1)\n") == "__UNPARSEABLE__"
