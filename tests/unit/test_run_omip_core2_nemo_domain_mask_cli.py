"""--nemo-domain-mask: the wet domain follows the oracle's domain_cfg.

The eORCA1.2 mesh keeps the Great Lakes, the Caspian Sea and Lake Victoria
wet (207 cells); the ORCA1 domain_cfg NEMO actually runs has them as land.
These tests pin the plumbing: the flag exists, it is off unless asked, the
path override lands in its dest, every NEMO-mesh builder accepts the keyword,
and the domain_cfg path takes part in the restart fingerprint as a PATH (so a
symlinked spelling does not false-abort a chained leg).
"""
from __future__ import annotations

import inspect
import pathlib

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_omip_core2_ndm", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flag_round_trips_and_is_off_by_default():
    p = _module()._build_arg_parser()
    a = p.parse_args([])
    assert a.nemo_domain_mask is False
    assert a.nemo_domain_cfg is None
    b = p.parse_args(["--nemo-domain-mask", "--nemo-domain-cfg", "/x/domain_cfg.nc"])
    assert b.nemo_domain_mask is True
    assert b.nemo_domain_cfg == "/x/domain_cfg.nc"


def test_every_nemo_mesh_builder_accepts_the_keyword():
    mod = _module()
    for name in ("build_tripole", "build_latlon_bathy", "build_cubed_sphere",
                 "build_mpas_ocean"):
        params = inspect.signature(getattr(mod, name)).parameters
        assert "nemo_domain_cfg" in params, name
        assert params["nemo_domain_cfg"].default is None, name


def test_the_default_domain_cfg_is_the_orca1_input_file():
    mod = _module()
    assert mod._NEMO_DOMAIN_CFG.endswith("input_fields/domain_cfg.nc")


def test_domain_cfg_path_is_fingerprinted_as_a_path():
    src = RUNNER.read_text()
    i = src.index("_RESTART_FP_PATH_KEYS = frozenset({")
    block = src[i:src.index("})", i)]
    assert '"nemo_domain_cfg"' in block


def test_the_flag_is_forwarded_to_every_builder_and_into_the_mesh_read():
    """Codex: signature inspection cannot see forwarding. Removing any
    ``nemo_domain_cfg=`` argument -- at the four dispatch sites in main() or at
    the four mesh reads inside the builders -- fails this."""
    src = RUNNER.read_text()
    # No closing paren: the tripole read also carries strip_north_rows=.
    assert src.count("read_mesh_mask_bathy(mesh_path, nemo_domain_cfg=nemo_domain_cfg") == 4
    assert src.count("nemo_domain_cfg=_nemo_domain_cfg,") == 4
    assert "_nemo_domain_cfg = (args.nemo_domain_cfg or _NEMO_DOMAIN_CFG)" in src
    assert "if args.nemo_domain_mask else None" in src


def test_nemo_init_tint_flag_round_trips_and_is_forwarded():
    p = _module()._build_arg_parser()
    assert p.parse_args([]).nemo_init_tint is False
    assert p.parse_args(["--nemo-init-tint"]).nemo_init_tint is True
    src = RUNNER.read_text()
    assert src.count("nemo_tint=bool(args.nemo_init_tint)") == 1
    assert src.count("nemo_tint=bool(nemo_init_tint)") == 1
    assert src.count("nemo_init_tint=args.nemo_init_tint,") == 1


def test_nemo_init_tint_is_reachable_on_the_fesom_lane():
    """Wired is not reachable: the FESOM lane refuses every user-set dest
    absent from _FESOM_WIRED_DESTS, and --nemo-init-tint was forwarded into
    build_fesom_ocean (8e7d98a55) without being listed, so the flag raised
    SystemExit on that lane while its forwarding test passed (2026-09-15)."""
    assert "nemo_init_tint" in _module()._FESOM_WIRED_DESTS
