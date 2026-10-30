"""Direct tests for scripts/run/run_fesom_core2.py helpers (CLI + snapshot
writer); the model integration itself is exercised by the SLURM arm."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_fesom_core2", _ROOT / "scripts/run/run_fesom_core2.py")
m = importlib.util.module_from_spec(_SPEC)
sys.modules["run_fesom_core2"] = m
_SPEC.loader.exec_module(m)


def test_arg_parser_round_trip():
    p = m.build_arg_parser()
    a = p.parse_args(["--mesh-dir", "M", "--ic-dir", "I", "--output", "O",
                      "--days", "30", "--dt", "1800", "--year", "1958"])
    assert (a.mesh_dir, a.ic_dir, a.output) == ("M", "I", "O")
    assert a.days == 30.0 and a.dt == 1800.0 and a.year == 1958


def test_snapshot_writer_conventions(tmp_path):
    """Writer must emit the comparator's node-cloud conventions: (n, nreal)
    tracers (padded bottom slot sliced off), degrees in [-180, 180],
    land_mask 1.0 == OCEAN, positive-down depths.  The state contract is
    NODE-first (nod2D, nl) with nl = nreal + 1 padded slots; the writer
    REFUSES level-first input rather than heuristically transposing (codex
    FESOM-arm hardening) -- the old 'tolerate level-first' expectation was
    stale against that guard."""
    n, nreal = 7, 3
    nl = nreal + 1                                       # +1 padded bottom slot
    mesh = types.SimpleNamespace(
        nod2D=n,
        geo_coord_nod2D=np.stack(
            [np.deg2rad(np.linspace(10.0, 350.0, n)),    # lon rad, wraps >180
             np.deg2rad(np.linspace(-60.0, 60.0, n))], axis=1),
        node_layer_mask=np.ones((n, nl), dtype=bool),
        Z=-np.array([5.0, 15.0, 30.0]),                  # nreal midpoints
        depth=-np.full(n, 100.0),
    )
    state = types.SimpleNamespace(
        T=np.arange(n * nl, dtype=np.float64).reshape(n, nl),  # node-first
        S=np.full((n, nl), 35.0),
        eta_n=np.zeros(n), a_ice=np.zeros(n), m_ice=np.zeros(n),
    )
    p = m.write_snapshot(tmp_path, "day0001", state, mesh)
    z = np.load(p)
    assert z["T"].shape == (n, nreal)                    # padded slot sliced
    assert z["T"][0, 1] == state.T[0, 1]                 # passthrough layout
    assert z["lon_T"].min() >= -180.0 and z["lon_T"].max() <= 180.0
    assert z["land_mask"].min() == 1.0                   # 1.0 == OCEAN
    assert (z["z_center_ref"] > 0).all() and (z["H_bathy"] > 0).all()


def test_snapshot_writer_refuses_level_first(tmp_path):
    """The guard that replaced the transpose heuristic must actually fire."""
    import pytest
    n, nl = 7, 4
    mesh = types.SimpleNamespace(
        nod2D=n,
        geo_coord_nod2D=np.zeros((n, 2)),
        node_layer_mask=np.ones((n, nl), dtype=bool),
        Z=-np.array([5.0, 15.0, 30.0]),
        depth=-np.full(n, 100.0),
    )
    state = types.SimpleNamespace(
        T=np.zeros((nl, n)), S=np.zeros((nl, n)),        # level-first: refuse
        eta_n=np.zeros(n), a_ice=np.zeros(n), m_ice=np.zeros(n),
    )
    with pytest.raises(SystemExit):
        m.write_snapshot(tmp_path, "day0001", state, mesh)


def test_forcing_flag_round_trip_and_default():
    p = m.build_arg_parser()
    base = ["--mesh-dir", "M", "--ic-dir", "I", "--output", "O"]
    assert p.parse_args(base).forcing == "jra55"
    a = p.parse_args(base + ["--forcing", "core2_nyf", "--nyf-zarr", "/x/nyf.zarr"])
    assert a.forcing == "core2_nyf" and a.nyf_zarr == "/x/nyf.zarr"


def test_unknown_forcing_rejected():
    import pytest
    p = m.build_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--mesh-dir", "M", "--ic-dir", "I", "--output", "O",
                      "--forcing", "era5"])


def test_tke_surface_bc_flag():
    import pytest
    p = m.build_arg_parser()
    base = ["--mesh-dir", "M", "--ic-dir", "I", "--output", "O"]
    assert p.parse_args(base).tke_surface_bc == "neumann"
    assert p.parse_args(base + ["--tke-surface-bc", "dirichlet"]
                        ).tke_surface_bc == "dirichlet"
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--tke-surface-bc", "robin"])


def test_nyf_zarr_defaults_to_the_shared_core2_cache():
    """FESOM must resolve the SAME CORE-II cache the tripole and MPAS drivers
    do when --nyf-zarr is omitted.

    Before this, --nyf-zarr was required and every FESOM arm carried a
    hand-written path.  One of those paths pointing at the raw-wind cache
    while the structured grids had moved to the bias-corrected one would make
    a three-grid comparison differ in the forcing -- a confound presented as a
    grid difference, which is exactly the claim these arms exist to make.
    """
    from legoesm.ocean.forcing import core2_nyf_cache_dir

    p = m.build_arg_parser()
    a = p.parse_args(["--mesh-dir", "M", "--ic-dir", "I", "--output", "O",
                      "--forcing", "core2_nyf"])
    assert a.nyf_zarr is None            # the driver resolves it, not argparse
    resolved = core2_nyf_cache_dir() / "nyf.zarr"
    assert resolved.name == "nyf.zarr"
    assert resolved.parent.name in ("core2_nyf_mod", "core2_nyf")


def test_explicit_nyf_zarr_still_wins():
    """An explicit path must override the shared default, so an older arm can
    still be reproduced against the raw-wind cache."""
    p = m.build_arg_parser()
    a = p.parse_args(["--mesh-dir", "M", "--ic-dir", "I", "--output", "O",
                      "--forcing", "core2_nyf", "--nyf-zarr", "/some/other.zarr"])
    assert a.nyf_zarr == "/some/other.zarr"

def test_the_ported_turbulence_card_must_be_taken_whole():
    """The two settings are one card in the model being ported.

    The reference turns both on together and the flags' own help calls either
    one alone a half port. Both directions are checked: an earlier version of
    this guard refused only the Dirichlet-without-anchor split and silently
    admitted the anchor-without-Dirichlet one, which is exactly as unported.
    The permitted combinations are asserted to RETURN, so a guard that
    rejected everything could not pass this.
    """
    import types

    import pytest

    def _args(bc, anchor, allow=False):
        return types.SimpleNamespace(
            tke_surface_bc=bc, tke_mxl0_anchor=anchor,
            allow_half_ported_tke=allow)

    # whole card, either way round: no exception, and nothing returned
    assert m.validate_tke_pair(_args("dirichlet", "on")) is None
    assert m.validate_tke_pair(_args("neumann", "off")) is None

    # both halves refused
    for bc, anchor in (("dirichlet", "off"), ("neumann", "on")):
        with pytest.raises(SystemExit, match="half port"):
            m.validate_tke_pair(_args(bc, anchor))
        # ... unless taken deliberately
        assert m.validate_tke_pair(_args(bc, anchor, allow=True)) is None


def test_the_half_port_guard_runs_before_the_driver_does_any_work():
    """The guard has to be reached from the entry point, not just exist.

    A guard nothing calls is not a guard, and this one has to fire before the
    mesh is read: a run that spends its allocation and then refuses is no
    better than one that does not refuse.
    """
    import inspect

    src = inspect.getsource(m.main)
    body = src.split("\n")
    called_at = next(i for i, line in enumerate(body)
                     if "validate_tke_pair(" in line)
    mesh_at = next((i for i, line in enumerate(body) if "load_mesh" in line),
                   len(body))
    assert called_at < mesh_at



def _manifest_args(**kw):
    a = m.build_arg_parser().parse_args(
        ["--mesh-dir", "M", "--ic-dir", "I", "--output", "O"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_manifest_goes_through_the_canonical_writer(tmp_path):
    """#2102: the canonical writer validates the record, keeps the old file's
    config keys, and hashes the fesom_jax install path."""
    import json

    import pytest

    fesom_jax = pytest.importorskip("fesom_jax")  # the helper reads its path

    from legoesm.driver.restart import validate_run_manifest

    path = m.write_fesom_core2_manifest(tmp_path, _manifest_args(), 30)
    man = json.loads(path.read_text())
    validate_run_manifest(man)
    assert man["reproducibility"]["runner_tag"] == "run_fesom_core2"
    assert man["config"]["config_kind"] == "ocean"
    rc = man["config"]["resolved_config"]
    assert rc["runner"] == "run_fesom_core2"
    assert rc["fesom_jax"] == str(Path(fesom_jax.__file__).parent)
    assert (rc["days"], rc["dt_s"], rc["year"], rc["tke_surface_bc"]) == (
        30, 1800.0, 1958, "neumann")


def test_manifest_hash_includes_the_fesom_jax_path(tmp_path, monkeypatch):
    import json

    hashes = []
    for i, root in enumerate(("/a/fesom_jax", "/b/fesom_jax")):
        monkeypatch.setitem(sys.modules, "fesom_jax",
                            types.SimpleNamespace(__file__=f"{root}/__init__.py"))
        path = m.write_fesom_core2_manifest(tmp_path / str(i),
                                            _manifest_args(), 30)
        hashes.append(json.loads(path.read_text())["config"]["config_hash"])
    assert hashes[0] != hashes[1]


def test_main_records_the_final_state_digest(tmp_path, monkeypatch):
    """main() writes the manifest and digests the final state at run end;
    fesom_jax's mesh/IC/forcing/step are stubbed (no data needed)."""
    import json
    from typing import NamedTuple

    import jax.numpy as jnp
    import pytest

    from legoesm.driver.restart import pytree_state_digest

    pytest.importorskip("fesom_jax")
    import fesom_jax.integrate  # noqa: F401
    import fesom_jax.mesh
    import fesom_jax.phc_ic
    import fesom_jax.ssh
    import fesom_jax.step
    import fesom_jax.tke
    from fesom_jax import surface_forcing

    class _State(NamedTuple):
        T: object

    n = 4
    mesh = types.SimpleNamespace(
        nod2D=n, elem2D=2, nl=3, ulevels_nod2D=np.ones(n, int),
        node_layer_mask=np.ones((n, 2)))
    monkeypatch.setattr(fesom_jax.mesh, "load_mesh", lambda d: mesh)
    monkeypatch.setattr(fesom_jax.phc_ic, "cold_start_state",
                        lambda mesh, d: _State(T=jnp.ones((n, 2))))
    monkeypatch.setattr(fesom_jax.ssh, "build_ssh_operator",
                        lambda mesh, dt: None)
    # The installed fesom_jax may predate the TKE flags main() passes.
    monkeypatch.setattr(fesom_jax.tke, "TkeConfig", lambda **kw: kw)
    # Each step adds 1, so the digest must be of the state AFTER the loop.
    monkeypatch.setattr(fesom_jax.step, "step",
                        lambda state, *a, **kw: state._replace(T=state.T + 1.0))
    forcing = types.SimpleNamespace(
        static=None, stack=lambda dates: jnp.zeros((len(dates), 1)))
    monkeypatch.setattr(surface_forcing, "build_surface_forcing",
                        lambda *a, **kw: forcing)
    monkeypatch.setattr(surface_forcing, "dates_for_steps",
                        lambda year, dt, n_steps: list(range(n_steps)))
    monkeypatch.setattr(m, "write_snapshot", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_fesom_core2.py", "--mesh-dir", "M", "--ic-dir", "I",
        "--days", "1", "--dt", "43200", "--output", str(tmp_path)])
    assert m.main() == 0
    man = json.loads((tmp_path / "run_manifest.json").read_text())
    # --days 1 at dt=43200 s is 2 steps: T goes 1 -> 3.
    assert man["result"]["state_digest"] == pytree_state_digest(
        _State(T=jnp.full((n, 2), 3.0)))
