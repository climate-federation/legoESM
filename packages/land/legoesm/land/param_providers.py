"""Land parameter providers: constant, PFT, and neural.

Each provider is an ``eqx.Module`` that produces a ``LandSurfaceParams``
NamedTuple of per-grid-cell arrays.  Trainable leaves (PFT table values,
neural-network weights) are visible to ``eqx.filter_value_and_grad``.

Usage::

    provider = ConstantParamProvider.from_config(ncol, land_config)
    land_params = provider()          # -> LandSurfaceParams

    provider = PFTParamProvider(...)
    land_params = provider()          # weighted average

    provider = NeuralParamProvider(...)
    land_params = provider(features)  # NN forward pass
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from legoesm.land.surface_params import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    N_PARAMS,
    LandSurfaceParams,
    array_to_params,
    bounds_arrays,
    clm5_pft_table,
    default_land_surface_params,
    N_PFT_CLM5,
)


# =====================================================================
# 1. Constant / Prescribed
# =====================================================================

class ConstantParamProvider(eqx.Module):
    """Return fixed (possibly per-pixel) parameter arrays.

    For backward compatibility, ``from_config`` broadcasts the existing
    config scalars to ``(ncol,)`` arrays — identical behaviour to
    ``land_params=None``.
    """
    params: LandSurfaceParams

    @staticmethod
    def from_config(ncol: int, land_config) -> "ConstantParamProvider":
        """Broadcast config scalars to (ncol,) arrays."""
        return ConstantParamProvider(
            params=default_land_surface_params(ncol, land_config),
        )

    @staticmethod
    def from_arrays(param_dict: dict[str, jax.Array]) -> "ConstantParamProvider":
        """Build from a dict of pre-computed per-pixel arrays."""
        return ConstantParamProvider(
            params=LandSurfaceParams(**param_dict),
        )

    def __call__(self) -> LandSurfaceParams:
        return self.params


# =====================================================================
# 2. PFT-weighted lookup
# =====================================================================

class PFTParamProvider(eqx.Module):
    """CLM5-style plant-functional-type weighted parameter lookup.

    ``raw_table`` is an unconstrained ``(n_pft, n_params)`` array mapped
    to physical bounds via sigmoid.  ``pft_fractions`` ``(ncol, n_pft)``
    contains per-cell PFT weights (from MODIS or prescribed).  Gradients
    flow through ``raw_table`` but not through ``pft_fractions``.
    """
    raw_table: jax.Array                           # (n_pft, n_params) trainable
    pft_fractions: jax.Array                       # (ncol, n_pft)
    param_bounds: list[tuple[float, float]] = eqx.field(static=True)
    param_names: tuple[str, ...] = eqx.field(static=True)

    @staticmethod
    def from_defaults(
        pft_fractions: jax.Array,
        param_bounds: list[tuple[float, float]] | None = None,
        param_names: tuple[str, ...] | None = None,
    ) -> "PFTParamProvider":
        """Initialize with CLM5 default table values.

        The raw table is set to the inverse-sigmoid of the CLM5 defaults
        so that ``sigmoid(raw) ≈ physical values`` before any training.
        """
        if param_bounds is None:
            param_bounds = [PARAM_BOUNDS[n] for n in PARAM_NAMES]
        if param_names is None:
            param_names = PARAM_NAMES

        table = clm5_pft_table()  # (17, 12)
        lo, hi = bounds_arrays()
        # Inverse sigmoid: raw = logit((table - lo) / (hi - lo))
        eps = 1e-6
        normalized = jnp.clip((table - lo) / (hi - lo), eps, 1.0 - eps)
        raw = jnp.log(normalized / (1.0 - normalized))

        return PFTParamProvider(
            raw_table=raw,
            pft_fractions=pft_fractions,
            param_bounds=param_bounds,
            param_names=param_names,
        )

    def _constrained_table(self) -> jax.Array:
        """Apply sigmoid bounds: raw -> (n_pft, n_params) physical values."""
        lo, hi = bounds_arrays()
        return lo + (hi - lo) * jax.nn.sigmoid(self.raw_table)

    def __call__(self) -> LandSurfaceParams:
        table = self._constrained_table()  # (n_pft, n_params)

        # Normalize fractions — convexity guarantee (Finding 5).
        # stop_gradient prevents accidental differentiation through data.
        fracs = jax.lax.stop_gradient(self.pft_fractions)
        frac_sum = jnp.sum(fracs, axis=-1, keepdims=True)
        fracs = fracs / jnp.maximum(frac_sum, 1e-10)

        cell_params = fracs @ table  # (ncol, n_params)
        return array_to_params(cell_params, self.param_names)


# =====================================================================
# 3. Neural parameter provider
# =====================================================================

class NeuralParamProvider(eqx.Module):
    """Small FCNN mapping static features to bounded land parameters.

    Architecture follows Fang et al. (2024): 3 hidden layers with GELU
    activation, sigmoid output bounds.  Global NN weights — spatial
    variation comes entirely from input features.

    Features are passed to ``__call__`` (not stored in the module) so
    they are not part of the trainable pytree.
    """
    layers: list                                   # eqx.nn.Linear layers
    param_bounds: list[tuple[float, float]] = eqx.field(static=True)
    param_names: tuple[str, ...] = eqx.field(static=True)
    n_input: int = eqx.field(static=True)
    n_output: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        key: jax.Array,
        n_input: int = 20,
        hidden_dim: int = 48,
        n_hidden: int = 3,
        param_bounds: list[tuple[float, float]] | None = None,
        param_names: tuple[str, ...] | None = None,
    ):
        if param_bounds is None:
            param_bounds = [PARAM_BOUNDS[n] for n in PARAM_NAMES]
        if param_names is None:
            param_names = PARAM_NAMES

        n_output = len(param_names)
        keys = jax.random.split(key, n_hidden + 1)
        dims = [n_input] + [hidden_dim] * n_hidden + [n_output]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]
        self.param_bounds = param_bounds
        self.param_names = param_names
        self.n_input = n_input
        self.n_output = n_output

    def _forward_single(self, x: jax.Array) -> jax.Array:
        """Single grid cell: (n_input,) -> (n_output,) bounded params."""
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        raw = self.layers[-1](x)
        lo, hi = bounds_arrays()
        return lo + (hi - lo) * jax.nn.sigmoid(raw)

    def __call__(self, features: jax.Array) -> LandSurfaceParams:
        """Forward pass over all grid cells.

        Parameters
        ----------
        features : jax.Array
            Shape ``(ncol, n_input)`` — static features per cell.

        Returns
        -------
        LandSurfaceParams
            Per-cell parameter arrays with shape ``(ncol,)``.
        """
        cell_params = jax.vmap(self._forward_single)(features)  # (ncol, n_output)
        return array_to_params(cell_params, self.param_names)


# =====================================================================
# Feature construction helper
# =====================================================================

def build_land_features(
    lat: jax.Array,
    lon: jax.Array,
    *,
    soil_type: jax.Array | None = None,
    n_soil_types: int = 12,
    elevation: jax.Array | None = None,
    mean_precip: jax.Array | None = None,
    mean_temp: jax.Array | None = None,
    forest_age: jax.Array | None = None,
) -> jax.Array:
    """Build static feature matrix for ``NeuralParamProvider``.

    Parameters
    ----------
    lat, lon : jax.Array
        Latitude and longitude in **radians**, shape ``(ncol,)``.
    soil_type : jax.Array, optional
        Integer soil type indices in ``[0, n_soil_types)``, shape ``(ncol,)``.
        One-hot encoded.  If None, a uniform vector ``1/n_soil_types`` is used.
    n_soil_types : int
        Number of USDA soil categories for one-hot encoding (default 12).
    elevation : jax.Array, optional
        Surface elevation [m], shape ``(ncol,)``.  Normalized by 5000 m.
    mean_precip : jax.Array, optional
        Mean annual precipitation [kg/m2/s], ``(ncol,)``.  Normalized by 5e-5.
    mean_temp : jax.Array, optional
        Mean annual temperature [K], ``(ncol,)``.  Normalized to ``(T-273)/40``.
    forest_age : jax.Array, optional
        Mean ecosystem age [years], ``(ncol,)``.  Normalized by 200.

    Returns
    -------
    jax.Array
        Shape ``(ncol, n_input)`` where ``n_input = 4 + n_soil_types + 3``.
    """
    ncol = lat.shape[0]
    parts: list[jax.Array] = []

    # Cyclic encoding of lat / lon
    parts.append(jnp.sin(lat)[:, None])
    parts.append(jnp.cos(lat)[:, None])
    parts.append(jnp.sin(lon)[:, None])
    parts.append(jnp.cos(lon)[:, None])

    # Soil type (one-hot or uniform)
    if soil_type is not None:
        soil_oh = jax.nn.one_hot(soil_type, n_soil_types)  # (ncol, n_soil_types)
    else:
        soil_oh = jnp.full((ncol, n_soil_types), 1.0 / n_soil_types)
    parts.append(soil_oh)

    # Climate / age features (normalized)
    if elevation is not None:
        parts.append((elevation / 5000.0)[:, None])
    else:
        parts.append(jnp.zeros((ncol, 1)))

    if mean_precip is not None:
        parts.append((mean_precip / 5e-5)[:, None])
    else:
        parts.append(jnp.zeros((ncol, 1)))

    if mean_temp is not None:
        parts.append(((mean_temp - 273.0) / 40.0)[:, None])
    else:
        parts.append(jnp.zeros((ncol, 1)))

    if forest_age is not None:
        parts.append((forest_age / 200.0)[:, None])
    else:
        parts.append(jnp.zeros((ncol, 1)))

    return jnp.concatenate(parts, axis=-1)


# =====================================================================
# Factory
# =====================================================================

def create_land_param_provider(
    mode: str,
    ncol: int,
    land_config,
    *,
    pft_fractions: jax.Array | None = None,
    prescribed_arrays: dict[str, jax.Array] | None = None,
    key: jax.Array | None = None,
    n_input: int = 20,
    hidden_dim: int = 48,
    n_hidden: int = 3,
):
    """Create a land parameter provider.

    Parameters
    ----------
    mode : str
        One of ``"constant"``, ``"prescribed"``, ``"pft"``, ``"neural"``.
    ncol : int
        Number of grid columns.
    land_config : LandConfig or MultiLayerLandConfig
        Used by constant mode to broadcast scalar defaults.
    pft_fractions : jax.Array, optional
        Shape ``(ncol, n_pft)``.  Required for ``mode="pft"``.
    prescribed_arrays : dict, optional
        Per-pixel arrays.  Required for ``mode="prescribed"``.
    key : jax.Array, optional
        PRNG key.  Required for ``mode="neural"``.
    n_input, hidden_dim, n_hidden : int
        Neural network architecture parameters.

    Returns
    -------
    ConstantParamProvider | PFTParamProvider | NeuralParamProvider
    """
    if mode == "constant":
        return ConstantParamProvider.from_config(ncol, land_config)

    if mode == "prescribed":
        if prescribed_arrays is None:
            raise ValueError("prescribed_arrays required for mode='prescribed'")
        return ConstantParamProvider.from_arrays(prescribed_arrays)

    if mode == "pft":
        if pft_fractions is None:
            raise ValueError("pft_fractions required for mode='pft'")
        return PFTParamProvider.from_defaults(pft_fractions)

    if mode == "neural":
        if key is None:
            raise ValueError("key required for mode='neural'")
        return NeuralParamProvider(
            key=key, n_input=n_input,
            hidden_dim=hidden_dim, n_hidden=n_hidden,
        )

    raise ValueError(f"Unknown land_param_mode: {mode!r}")
