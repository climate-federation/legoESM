"""YAML experiment adapter for standalone idealized sea-ice cases (#388).

Sea-ice has no OMIP-style standalone *recipe* path (unlike the ocean); its
templates are purely ``setup:`` selectors that NAME an idealized case in
``scripts/matrix/run_sea_ice_test_matrix.py``.  So this adapter is **setup-only**:
a template MUST carry a ``setup:`` block, and it routes to the matrix runner via
``--test <case> --grid <grid>`` (the sea-ice ``--test`` filter is already an
EXACT ``tc.case == args.test`` match, so no ``=`` prefix is used).

It implements the uniform experiment-adapter protocol consumed by
``init_experiment`` / ``validate_templates`` through
``legoesm.experiment_registry`` (``from_yaml``, ``get``/``set``, ``to_yaml``,
``get_meta``, ``signature``, ``validate_strict``, ``run_command``), reusing the
shared component-agnostic mechanics in :mod:`legoesm.core.setup_selector`.
"""
from __future__ import annotations

import copy
from typing import Any

import yaml

_DEFAULT_SEAICE_CONFIG: dict = {
    "model": {"name": "legoESM-sea-ice", "type": "sea_ice_only"},
    "mode": "sea_ice",
    "output": {"path": "output/sea_ice/"},
}

# Top-level keys consumed on the setup path (everything else would be silently
# ignored by the matrix run, so a customised one is rejected).
_CONSUMED_TOP = {"setup", "model", "experiment", "mode", "output"}


def _ice_matrix_spec():
    from legoesm.core.setup_selector import MatrixRunnerSpec
    return MatrixRunnerSpec(
        runner_path="scripts/matrix/run_sea_ice_test_matrix.py",
        valid_grids=("cubed_sphere", "column"),
        case_flag="--test", exact_prefix="",   # --test is already exact
        levels_flag=None, dt_flag=None, days_flag=None, resolution_flag=None,
        output_flag="--output", quick_flag="--quick",
    )


class SeaIceExperimentConfig:
    """Setup-only YAML adapter for idealized sea-ice matrix cases (#388)."""

    def __init__(self, data: dict | None = None):
        self._data = data or copy.deepcopy(_DEFAULT_SEAICE_CONFIG)

    @classmethod
    def from_yaml(cls, path: str) -> "SeaIceExperimentConfig":
        with open(path, "r") as f:
            user = yaml.safe_load(f)
        from legoesm.core.setup_selector import deep_merge
        cfg = copy.deepcopy(_DEFAULT_SEAICE_CONFIG)
        if user:
            deep_merge(cfg, user)
        return cls(cfg)

    @classmethod
    def from_dict(cls, d: dict) -> "SeaIceExperimentConfig":
        from legoesm.core.setup_selector import deep_merge
        cfg = copy.deepcopy(_DEFAULT_SEAICE_CONFIG)
        deep_merge(cfg, d)
        return cls(cfg)

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data
        for k in key.split("."):
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    def get_meta(self) -> dict:
        return self.get("experiment") or {}

    def validate_strict(self) -> None:
        """Validate the setup-only sea-ice template (raises on invalid)."""
        from legoesm.core.setup_selector import validate_setup

        setup = self.get("setup")
        if setup is None:
            raise ValueError(
                "sea-ice templates are setup-only: a `setup:` block "
                "(name + grid) is required (there is no standalone sea-ice "
                "recipe path)."
            )
        validate_setup(setup, _ice_matrix_spec())
        # Only setup: + output.path are consumed; any other top-level section
        # would be silently ignored by the matrix run -> reject it.
        ignored = []
        for k, v in self._data.items():
            if k not in _CONSUMED_TOP:
                ignored.append(k)
            elif k == "output":
                if not isinstance(v, dict):
                    raise ValueError(
                        f"output: must be a mapping (with `path`), got "
                        f"{type(v).__name__}")
                extra = {kk for kk in v if kk != "path"}
                if extra:
                    ignored.append("output (only output.path is used)")
        if ignored:
            raise ValueError(
                f"top-level section(s) {sorted(ignored)} are ignored by a "
                "sea-ice `setup:` template (per-run controls go in the "
                "`setup:` block: quick). Remove them."
            )

    def signature(self) -> str:
        try:
            from legoesm.core.setup_selector import setup_signature
            return setup_signature(_ice_matrix_spec(), self.get("setup"),
                                   output_path=self.get("output.path"))
        except Exception as exc:  # noqa: BLE001
            return f"<unresolvable: {type(exc).__name__}: {exc}>"

    def run_command(self, config_path: str = "config.yaml") -> str:
        from legoesm.core.setup_selector import build_matrix_command
        return build_matrix_command(_ice_matrix_spec(), self.get("setup"),
                                    output_path=self.get("output.path"))

    def __repr__(self) -> str:
        return f"SeaIceExperimentConfig({self._data})"
