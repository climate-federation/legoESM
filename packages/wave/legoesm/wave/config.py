"""Configuration for the coupled wave component.

Every knob of the ported NeuralSWM model that a run could legitimately
change is a field here.  Nothing is read from a process-global registry:
the source implementation used ``gin``, whose ``clear_config()`` makes
two models in one process share state, so the port replaces it with an
explicit NamedTuple tree.

The field names are lower-case versions of the WW3-style switches in
the source (``FACHFA`` -> ``fachfa``), following the repository rule
that NamedTuple fields are snake_case.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

__param_spec__ = {
    "WavePhysicsConfig": {
        "scheme_key": "wave.physics",
        "excluded": {},
        "params": {
            "fachfa": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": "source-term activity switch (WW3 FACHFA)", "shape": None,
            },
            "facvx": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": "x-advection activity switch (WW3 FACVX)", "shape": None,
            },
            "facvy": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": "y-advection activity switch (WW3 FACVY)", "shape": None,
            },
            "facth": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": "direction-refraction activity switch (WW3 FACTH)", "shape": None,
            },
            "fdg": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": "spherical deep-water direction-gradient switch (WW3 FDG)",
                "shape": None,
            },
        },
    },
    "WaveGSEConfig": {
        "scheme_key": "wave.gse",
        "excluded": {},
        "params": {
            "alpha_s": {
                "units": "1", "bounds": (0.0, 5.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": (
                    "GSE stream-direction averaging strength (Tolman 2002); "
                    "the global config uses 1.0"),
                "shape": None,
            },
            "alpha_n": {
                "units": "1", "bounds": (0.0, 5.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": (
                    "GSE normal-direction averaging strength; the global config uses 1.0"),
                "shape": None,
            },
            "eps": {
                "units": "1", "bounds": (1e-12, 1e-2), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": "GSE division floor", "shape": None,
            },
            "max_corner_extent": {
                "units": "1", "bounds": (1e-6, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": "GSE corner-weight cap (validate requires > 0)", "shape": None,
            },
        },
    },
    "WaveNetworkConfig": {
        "scheme_key": "wave.network",
        "excluded": {},
        "params": {
            "layernorm_eps": {
                "units": "1", "bounds": (1e-8, 1e-2), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": "nnx.LayerNorm epsilon used by the released weights", "shape": None,
            },
            "sample_ratio": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "numerics",
                "reference": (
                    "fraction of columns whose gradient is kept (sampling variants)"),
                "shape": None,
            },
        },
    },
}
"""Machine-readable parameter spec (see ``tests/test_param_specs.py``).

Every field is tier 0: the released weights are a fixed artefact, so none
of these are calibration targets.  A future training PR would promote the
network widths and any re-opened source coefficients.
"""


class WaveGridConfig(NamedTuple):
    """Regular lat-lon spectral grid.

    Longitude and latitude are the cell-centre coordinates in degrees, in
    storage order (the source stores ascending longitude and descending
    latitude for the ERA5 demo domain).  ``freq_hz`` is the frequency
    grid in Hz and ``theta_rad`` the direction grid in radians, using the
    source convention ``theta = (180 - phi_data) mod 360`` where
    ``phi_data`` is the ECMWF direction the wave propagates towards.
    """

    lon_deg: tuple[float, ...]
    lat_deg: tuple[float, ...]
    freq_hz: tuple[float, ...]
    theta_rad: tuple[float, ...]
    coordinate_type: Literal["Spherical", "Cartesian"] = "Spherical"
    boundary_mode: Literal["physical", "absorbing"] = "physical"
    x_stencil: Literal["circular", "replicate"] = "circular"
    y_stencil: Literal["circular", "replicate"] = "replicate"


class WaveTimeConfig(NamedTuple):
    """Split time-step sizes in seconds.

    A zero size disables that component, except GSE: ``enabled=True`` with
    ``dt_gse_s=0`` is rejected; set ``WaveGSEConfig.enabled=False`` to run
    without GSE.  The integrator substeps at ``gcd`` of the active sizes for
    ``dt_global_s`` seconds.  The released NeuralSWM decks use a 3600 s
    global step; ``dt_th``/``dt_k`` are 3600 s in the NNTend/GSE deck and
    0 in the lighter default deck.  A coupled production run must select
    its split explicitly instead of relying on these library defaults.
    """

    dt_global_s: int = 3600
    dt_x_s: int = 3600
    dt_y_s: int = 3600
    dt_th_s: int = 3600
    dt_k_s: int = 3600
    dt_stot_s: int = 3600
    # NeuralSWM's architecture metadata also carries 1800; the released GSE
    # deck sets 3600, which this default follows.
    dt_gse_s: int = 3600


class WavePhysicsConfig(NamedTuple):
    """Numerical switches of the operator-splitting kernel."""

    fachfa: float = 1.0
    facvx: float = 1.0
    facvy: float = 1.0
    facth: float = 1.0
    fdg: float = 1.0
    yfirst: bool = True
    checkpoint_level: str = "none"


class WaveNetworkConfig(NamedTuple):
    """Learned-network shape and sampling settings."""

    use_transport_networks: bool = True
    parameterization: Literal["mlp", "none"] = "mlp"
    mlp_input_channels: int = 2880
    mlp_hidden_channels_0: int = 128
    mlp_hidden_channels_1: int = 256
    mlp_hidden_channels_2: int = 256
    transport_input_channels: int = 90
    transport_hidden_channels: int = 128
    transport_output_vectors: int = 60
    layernorm_eps: float = 1e-5
    sample_ratio: float = 1.0


class WaveGSEConfig(NamedTuple):
    """GSE alleviation (Tolman-style nine-point conservative averaging).

    Global runs at dt = 3600 s need this: the garden-sprinkler effect from
    the discrete spectral grid otherwise accumulates.  It is enabled only
    when BOTH ``enabled`` is True and ``WaveTimeConfig.dt_gse_s > 0`` (the
    cadence lives in ONE place, the time config).  The default matches the
    released ``model_mlpParam_c5_sdnEmb2_NNTend_gse`` deck: enabled at
    ``dt_gse_s = 3600`` with alpha 1.0 and periodic x.
    """

    enabled: bool = True
    alpha_s: float = 1.0
    alpha_n: float = 1.0
    renormalize_land_blocked: bool = True
    preserve_spectral_totals: bool = True
    periodic_x: bool = True
    periodic_y: bool = False
    eps: float = 1e-8
    max_corner_extent: float = 1.0
    substeps: int = 1


class WaveConfig(NamedTuple):
    """Complete configuration of the wave component."""

    grid: WaveGridConfig
    time: WaveTimeConfig = WaveTimeConfig()
    physics: WavePhysicsConfig = WavePhysicsConfig()
    network: WaveNetworkConfig = WaveNetworkConfig()
    gse: WaveGSEConfig = WaveGSEConfig()

    @classmethod
    def from_json(cls, path: str) -> "WaveConfig":
        """Load a run deck.

        The JSON mirrors this NamedTuple::

            {"grid": {"lon_deg": [...], "lat_deg": [...],
                      "freq_hz": [...], "theta_rad": [...]},
             "time": {...}, "physics": {...}, "network": {...}}

        Every scientific choice a run makes must live in this file; the
        only defaults are the source model's production values in the
        nested configs.  Unknown keys raise rather than being ignored.
        """
        import json
        from pathlib import Path

        raw = json.loads(Path(path).read_text())
        grid = WaveGridConfig(**raw["grid"])
        time = WaveTimeConfig(**raw.get("time", {}))
        physics = WavePhysicsConfig(**raw.get("physics", {}))
        network = WaveNetworkConfig(**raw.get("network", {}))
        gse = WaveGSEConfig(**raw.get("gse", {}))
        config = cls(grid=grid, time=time, physics=physics, network=network, gse=gse)
        config.validate()
        return config

    def validate(self) -> None:
        """Fail early on an unknown selection or an inconsistent grid."""
        if self.network.parameterization not in ("mlp", "none"):
            raise ValueError(
                f"unknown wave parameterization {self.network.parameterization!r}; "
                "expected 'mlp' or 'none'")
        if self.physics.checkpoint_level not in ("none", "per_step", "per_phase"):
            raise ValueError(
                f"unknown wave checkpoint_level {self.physics.checkpoint_level!r}; "
                "expected 'none', 'per_step' or 'per_phase'")
        if self.grid.coordinate_type not in ("Spherical", "Cartesian"):
            raise ValueError("wave coordinate_type must be Spherical or Cartesian")
        if self.grid.boundary_mode not in ("physical", "absorbing"):
            raise ValueError("wave boundary_mode must be physical or absorbing")
        if (self.grid.x_stencil not in ("circular", "replicate")
                or self.grid.y_stencil not in ("circular", "replicate")):
            raise ValueError("wave spatial stencils must be circular or replicate")
        n_dir = len(self.grid.theta_rad)
        n_freq = len(self.grid.freq_hz)
        if n_dir < 2 or n_freq < 2:
            raise ValueError("wave spectral grid needs at least two directions and frequencies")
        if len(self.grid.lon_deg) < 2 or len(self.grid.lat_deg) < 2:
            raise ValueError("wave spatial grid needs at least two longitudes and latitudes")
        freq = np.asarray(self.grid.freq_hz, dtype=float)
        if np.any(freq <= 0):
            raise ValueError("wave freq_hz must be strictly positive")
        if np.any(np.diff(freq) <= 0):
            raise ValueError("wave freq_hz must be strictly increasing")
        theta = np.asarray(self.grid.theta_rad, dtype=float)
        spacing = np.diff(theta)
        if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=0.0):
            raise ValueError("wave theta_rad must be uniformly spaced")
        if not np.isclose(abs(spacing[0]) * n_dir, 2.0 * np.pi, rtol=1e-6):
            raise ValueError("wave theta_rad must cover the full circle")
        if (self.network.parameterization == "none"
                and self.network.use_transport_networks):
            raise ValueError(
                "parameterization 'none' requires use_transport_networks=False; "
                "otherwise the step applies randomly initialised transport "
                "networks and is not a physics-only run")
        if self.network.parameterization == "mlp":
            expected = n_dir * n_freq
            if self.network.mlp_input_channels != 4 * expected:
                raise ValueError(
                    "mlp_input_channels must be 4 * n_directions * n_frequencies "
                    f"({4 * expected}); got {self.network.mlp_input_channels}")
        if self.network.use_transport_networks:
            if self.network.transport_output_vectors != 2 * n_freq:
                raise ValueError(
                    "transport_output_vectors must be 2 * n_frequencies "
                    f"({2 * n_freq}); got {self.network.transport_output_vectors}")
            # The transport features are three (Nlon*Nlat, Nfreq) blocks:
            # k*dDepth/dx, k*dDepth/dy and DsDd.
            if self.network.transport_input_channels != 3 * n_freq:
                raise ValueError(
                    "transport_input_channels must be 3 * n_frequencies "
                    f"({3 * n_freq}); got {self.network.transport_input_channels}")
        if (not 0 < self.gse.max_corner_extent <= 1
                or isinstance(self.gse.substeps, bool)
                or not isinstance(self.gse.substeps, int)
                or self.gse.substeps < 1):
            raise ValueError(
                "GSE requires 0 < max_corner_extent <= 1 and positive integer substeps")
        if self.gse.alpha_s < 0 or self.gse.alpha_n < 0:
            raise ValueError("GSE spread coefficients must be nonnegative")
        if self.gse.enabled and self.time.dt_gse_s <= 0:
            raise ValueError(
                "wave GSE is enabled but WaveTimeConfig.dt_gse_s is not positive; "
                "set dt_gse_s (the released GSE deck uses 3600)")
        for name in ("dt_global_s", "dt_x_s", "dt_y_s", "dt_th_s", "dt_k_s",
                     "dt_stot_s", "dt_gse_s"):
            value = getattr(self.time, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"WaveTimeConfig.{name} must be a non-negative int; got {value!r}")
            if self.time.dt_global_s <= 0 or (value and self.time.dt_global_s % value):
                raise ValueError("wave component time steps must divide a positive global interval")
