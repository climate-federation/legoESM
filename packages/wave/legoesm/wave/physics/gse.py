"""Conservative GSE alleviation (Tolman-style nine-point averaging).

Port of WADEN 46aadc7 ``src/waden/gse_alleviation.py``.  The operator is
configured by :class:`legoesm.wave.config.WaveGSEConfig`; the integrator
runs it only when the config enables it and ``dt_gse_s > 0``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.wave.config import WaveGSEConfig
from legoesm.wave.grid import WaveEnv, wave_cell_area

__physics_contract__ = {
    "summary": (
        "Tolman GSE alleviation: a conservative nine-point spatial average "
        "whose weights follow the group-velocity stream and normal directions. "
        "Area-weighted spectral mass is scattered; blocked transfers stay local "
        "or valid outgoing weights are renormalized, according to configuration."
    ),
    "inputs": {
        "spectra": "(B,Nlon,Nlat,Ndir,Nfreq)",
        "env": "WaveEnv with Cg, theta and grid metrics",
        "mask": "True where sea",
        "dt": "averaging time step [s]",
    },
    "outputs": {"spectra": "averaged spectrum, same shape"},
    "sign_convention": "Weights are non-negative and sum to one per column.",
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": "WADEN 46aadc7 src/waden/gse_alleviation.py; Tolman (2002).",
    "idealized_test": "tests/wave/unit/test_gse.py::test_weights_sum_to_one",
}


# Fallback geometric frequency ratio when the grid has fewer than two bins;
# matches the released weights' frequency grid (WADEN gse_alleviation.py).
_DEFAULT_FREQ_RATIO = 1.1


def _axis_source_target_slices(size: int, offset: int, periodic: bool) -> tuple[slice, slice]:
    if periodic:
        return slice(None), slice(None)
    return (
        slice(max(0, -offset), size - max(0, offset)),
        slice(max(0, offset), size - max(0, -offset)),
    )


def _target_valid_mask_for_source(
    valid_mask: jax.Array,
    offset_x: int,
    offset_y: int,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> jax.Array:
    _, nx, ny = valid_mask.shape
    src_x, tgt_x = _axis_source_target_slices(nx, offset_x, periodic_x)
    src_y, tgt_y = _axis_source_target_slices(ny, offset_y, periodic_y)
    target_lookup = valid_mask
    if periodic_x and offset_x != 0:
        target_lookup = jnp.roll(target_lookup, shift=-offset_x, axis=1)
    if periodic_y and offset_y != 0:
        target_lookup = jnp.roll(target_lookup, shift=-offset_y, axis=2)
    return jnp.zeros_like(valid_mask, dtype=bool).at[:, src_x, src_y].set(
        target_lookup[:, tgt_x, tgt_y]
    )


def _add_constant_offset(
    weights: jax.Array,
    offset_x: int,
    offset_y: int,
    value: jax.Array,
) -> jax.Array:
    return weights.at[..., offset_x + 1, offset_y + 1].add(value)


class GSESpatialAveraging(NamedTuple):
    """Operator of the nine-point GSE averaging.

    The knobs live in :class:`WaveGSEConfig`; this wrapper carries the one
    config object so the parameters are declared in exactly one place.
    """

    config: WaveGSEConfig

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def __call__(
        self,
        spectra: jax.Array,
        env: WaveEnv,
        mask: jax.Array,
        dt: float,
    ) -> jax.Array:
        if not self.config.enabled or dt <= 0:
            return spectra

        valid_mask = self._normalize_mask(mask)
        Cg = self._expand_to_spectra(jnp.asarray(env.group_velocity, dtype=spectra.dtype), spectra)

        weights = self.compute_average_weights(Cg, env, dt=dt / self.config.substeps)
        effective_weights = self.prepare_valid_scatter_weights(weights, valid_mask)
        area = jnp.asarray(wave_cell_area(env), dtype=spectra.dtype)

        def body(_, current):
            mass = self.scatter_by_weights(current * area, effective_weights, valid_mask)
            return mass / area

        return jax.lax.fori_loop(0, self.config.substeps, body, spectra)


    def _normalize_mask(self, mask: jax.Array) -> jax.Array:
        mask = jnp.asarray(mask)
        if mask.ndim == 5:
            mask = mask[..., 0, 0]
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        return mask.astype(bool)

    def _expand_to_spectra(self, value: jax.Array, spectra: jax.Array) -> jax.Array:
        return jnp.broadcast_to(jnp.asarray(value, dtype=spectra.dtype), spectra.shape)

    def _frequency_ratio(self, env: WaveEnv, ref: jax.Array) -> jax.Array:
        freq = jnp.asarray(env.freq, dtype=ref.dtype).reshape(-1)
        if freq.size < 2:
            return jnp.asarray(_DEFAULT_FREQ_RATIO, dtype=ref.dtype)
        ratios = freq[1:] / jnp.clip(freq[:-1], min=self.config.eps)
        ratios = jnp.where(
            jnp.isfinite(ratios) & (ratios > self.config.eps),
            ratios, jnp.ones_like(ratios))
        return jnp.median(ratios)

    def _physical_grid_steps(self, env: WaveEnv, ref: jax.Array) -> tuple[jax.Array, jax.Array]:
        dx = jnp.asarray(env.dlon, dtype=ref.dtype)
        dy = jnp.asarray(env.dlat, dtype=ref.dtype)
        dx = dx * jnp.asarray(env.x_metric, dtype=ref.dtype)
        dy = dy * jnp.asarray(env.y_metric, dtype=ref.dtype)
        return jnp.clip(dx, min=self.config.eps), jnp.clip(dy, min=self.config.eps)

    def _corners(
        self,
        Cg: jax.Array,
        env: WaveEnv,
        dt: float,
    ) -> tuple:
        Cg, theta = jnp.broadcast_arrays(Cg, jnp.asarray(env.theta, dtype=Cg.dtype))
        dtheta = jnp.asarray(env.dth, dtype=Cg.dtype)
        dx, dy = self._physical_grid_steps(env, Cg)
        freq_ratio = self._frequency_ratio(env, Cg)

        sin_t = jnp.sin(theta)
        cos_t = jnp.cos(theta)
        stream_scale = (
            0.5
            * self.config.alpha_s
            * (freq_ratio - 1.0 / jnp.clip(freq_ratio, min=self.config.eps))
            * Cg
            * jnp.asarray(dt, dtype=Cg.dtype)
        )
        normal_scale = self.config.alpha_n * dtheta * Cg * jnp.asarray(dt, dtype=Cg.dtype)

        sx = stream_scale * sin_t / dx
        sy = stream_scale * cos_t / dy
        nx = -normal_scale * cos_t / dx
        ny = normal_scale * sin_t / dy

        corners = (
            (sx + nx, sy + ny),
            (-sx + nx, -sy + ny),
            (-sx - nx, -sy - ny),
            (sx - nx, sy - ny),
        )
        return corners


    def required_substeps(self, Cg: jax.Array, env: WaveEnv, dt: float) -> jax.Array:
        """Return the static subdivision count to choose before compiling."""
        extent = jnp.max(jnp.stack([jnp.maximum(jnp.abs(x), jnp.abs(y))
                                    for x, y in self._corners(Cg, env, dt)]))
        return jnp.maximum(1, jnp.ceil(extent / self.config.max_corner_extent)).astype(jnp.int32)


    def validate_displacement(self, Cg: jax.Array, env: WaveEnv, dt: float) -> None:
        """Check a concrete environment outside JIT; never silently clip offsets.

        dt is the interval of one GSE call, before this object's internal
        subdivision. With step/rollout use config.dt_gse/config.substeps;
        do not divide by self.config.substeps again here. __call__ is JIT-compatible
        and assumes this preflight was performed for the supplied environment.
        """
        required = int(self.required_substeps(Cg, env, dt))
        if self.config.substeps < required:
            raise ValueError(
                f"GSE footprint requires at least {required} substeps, "
                f"got {self.config.substeps}")


    def compute_average_weights(self, Cg: jax.Array, env: WaveEnv, dt: float) -> jax.Array:
        """Return nonnegative normalized weights within the validated footprint."""
        Cg = jnp.broadcast_arrays(Cg, jnp.asarray(env.theta, dtype=Cg.dtype))[0]
        weights = jnp.zeros((*Cg.shape, 3, 3), dtype=Cg.dtype)
        weights = _add_constant_offset(weights, 0, 0, jnp.full(Cg.shape, 1.0 / 3.0, dtype=Cg.dtype))

        for rx, ry in self._corners(Cg, env, dt):
            abs_x = jnp.abs(rx)
            abs_y = jnp.abs(ry)
            max_abs = jnp.maximum(abs_x, abs_y)
            min_abs = jnp.minimum(abs_x, abs_y)

            center_w = (1.0 - max_abs) / 6.0
            axis_w = (max_abs - min_abs) / 6.0
            diag_w = min_abs / 6.0
            weights = _add_constant_offset(weights, 0, 0, center_w)

            sign_x = jnp.sign(rx)
            sign_y = jnp.sign(ry)
            x_major = abs_x >= abs_y
            y_major = ~x_major

            for ox in (-1, 1):
                mask_x = (x_major & (sign_x == ox)).astype(Cg.dtype)
                weights = _add_constant_offset(weights, ox, 0, axis_w * mask_x)
            for oy in (-1, 1):
                mask_y = (y_major & (sign_y == oy)).astype(Cg.dtype)
                weights = _add_constant_offset(weights, 0, oy, axis_w * mask_y)
            for ox in (-1, 1):
                for oy in (-1, 1):
                    mask_xy = ((sign_x == ox) & (sign_y == oy)).astype(Cg.dtype)
                    weights = _add_constant_offset(weights, ox, oy, diag_w * mask_xy)

        return weights


    def prepare_valid_scatter_weights(
        self,
        weights: jax.Array,
        valid_mask: jax.Array,
    ) -> jax.Array:
        effective = jnp.zeros_like(weights)
        availability = jnp.zeros_like(weights[..., 0, 0])

        for ix, ox in enumerate((-1, 0, 1)):
            for iy, oy in enumerate((-1, 0, 1)):
                target_mask = _target_valid_mask_for_source(
                    valid_mask,
                    ox,
                    oy,
                    periodic_x=self.config.periodic_x,
                    periodic_y=self.config.periodic_y,
                )[..., None, None]
                weighted = weights[..., ix, iy] * target_mask.astype(weights.dtype)
                effective = effective.at[..., ix, iy].set(weighted)
                availability = availability + weighted

        if self.config.renormalize_land_blocked:
            normalizer = jnp.clip(availability, min=self.config.eps)
            effective = jnp.where(
                availability[..., None, None] > self.config.eps,
                effective / normalizer[..., None, None],
                jnp.zeros_like(effective),
            )
        elif self.config.preserve_spectral_totals:
            # Blocked transfers remain in the source cell, rather than being
            # lost and restored by a global rescaling of unrelated cells.
            effective = effective.at[..., 1, 1].add(jnp.maximum(0, 1 - availability))
        return effective


    def scatter_by_weights(
        self,
        spectra: jax.Array,
        effective_weights: jax.Array,
        valid_mask: jax.Array,
    ) -> jax.Array:
        sea_source = valid_mask[..., None, None]
        source = jnp.where(sea_source, spectra, jnp.zeros_like(spectra))
        _, nx, ny, _, _ = spectra.shape
        out = jnp.zeros_like(spectra)

        for ix, ox in enumerate((-1, 0, 1)):
            for iy, oy in enumerate((-1, 0, 1)):
                contribution = source * effective_weights[..., ix, iy]
                if self.config.periodic_x and ox != 0:
                    contribution = jnp.roll(contribution, shift=ox, axis=1)
                if self.config.periodic_y and oy != 0:
                    contribution = jnp.roll(contribution, shift=oy, axis=2)
                src_x, tgt_x = _axis_source_target_slices(nx, ox, self.config.periodic_x)
                src_y, tgt_y = _axis_source_target_slices(ny, oy, self.config.periodic_y)
                out = out.at[:, tgt_x, tgt_y, :, :].add(contribution[:, src_x, src_y, :, :])

        return jnp.where(sea_source, out, jnp.zeros_like(out))



__all__ = ["GSESpatialAveraging"]
