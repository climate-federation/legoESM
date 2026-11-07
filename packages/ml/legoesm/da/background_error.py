"""Background error covariance (B matrix) for 4D-Var.

B is never formed explicitly. Instead, we provide B^{1/2} multiply
(sqrt_multiply) and B^{-1} multiply (inv_multiply) in control space.
All operations are differentiable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Diagonal B
# ---------------------------------------------------------------------------

class DiagonalB(NamedTuple):
    """Diagonal background error covariance.

    B = diag(sigma^2).
    B^{1/2} x = sigma * x.
    B^{-1} x = x / sigma^2.
    """
    sigma: jax.Array  # Standard deviations, shape (control_size,)

    def sqrt_multiply(self, x: jax.Array) -> jax.Array:
        return self.sigma * x

    def inv_multiply(self, x: jax.Array) -> jax.Array:
        return x / (self.sigma ** 2)


# ---------------------------------------------------------------------------
# Diffusion-based B (Weaver & Courtier 2001)
# ---------------------------------------------------------------------------

class DiffusionB:
    """Implicit diffusion-based background error covariance.

    B = Sigma @ C @ Sigma where:
    - Sigma = diagonal standard deviation matrix
    - C = correlation matrix via implicit diffusion

    The diffusion is applied on column-flattened fields:
    C^{1/2} ≈ (I - kappa * dt_diff * nabla^2)^{-n/2}

    Parameters
    ----------
    grid : GridProtocol
        Model grid.
    sigma : jax.Array
        Standard deviations per control element.
    horizontal_length_scale : float
        Correlation length [m].
    vertical_length_scale : float or None
        Vertical correlation [levels].
    n_diffusion_iter : int
        Number of diffusion iterations.
    spec : ControlVectorSpec
        Control vector specification.
    """

    def __init__(
        self,
        grid,
        sigma: jax.Array,
        horizontal_length_scale: float,
        vertical_length_scale: float | None = None,
        n_diffusion_iter: int = 10,
        spec=None,
    ):
        self.grid = grid
        self.sigma = sigma
        self.horizontal_length_scale = horizontal_length_scale
        self.vertical_length_scale = vertical_length_scale
        self.n_iter = n_diffusion_iter
        self.spec = spec

        # Smoothing factor per iteration: fraction of relaxation toward mean.
        # alpha in (0, 1) ensures damping. Derived from L / R_earth ratio.
        dx_mean = grid.grid_radius * jnp.sqrt(4.0 * jnp.pi / grid.grid_n_columns)
        self.kappa = jnp.clip(
            (horizontal_length_scale / dx_mean) ** 2 / (2.0 * n_diffusion_iter),
            0.0, 0.5,
        )

    def _apply_per_field(self, field_flat: jax.Array, block_op) -> jax.Array:
        """Apply ``block_op`` to EACH control field independently.

        ``block_op`` maps a single ``(ncol, nlev)`` field block to a block of
        the same shape (the smoother or its inverse).  When ``self.spec`` is set
        we slice ``field_flat`` by the control-vector entries so a multi-field
        control (u/v/T/q) is smoothed field-by-field — reshaping the WHOLE flat
        vector as ``(ncol, total//ncol)`` instead smears unrelated fields
        together (wrong J_b gradient) and crashes when ``total % ncol != 0``.
        With no spec we keep the legacy single-field reshape.

        The diffusion smoother is block-diagonal across fields, so applying the
        SAME block operator to each slice independently preserves the exact
        transpose-consistent inverse identity ``inv_multiply(B v) == v``.
        """
        ncol = self.grid.grid_n_columns

        def _block(flat_slice):
            # Each control field is stored as field.ravel(); its spatial dims
            # flatten to ncol, leaving size // ncol vertical levels.
            if flat_slice.size % ncol != 0:
                raise ValueError(
                    f"control field of size {flat_slice.size} is not a multiple "
                    f"of grid_n_columns={ncol}; cannot reshape to (ncol, nlev)"
                )
            nlev = flat_slice.size // ncol
            return block_op(flat_slice.reshape(ncol, nlev)).ravel()

        if field_flat.ndim == 1 and self.spec is not None:
            # Validate length BEFORE slicing: jax.lax.dynamic_slice CLAMPS an
            # out-of-range offset, so a wrong-sized control would be silently
            # corrupted (a short vector duplicates trailing values; a long one
            # drops the tail) — mirror the control_to_state guard.
            if field_flat.shape[0] != self.spec.total_size:
                raise ValueError(
                    f"control vector has size {field_flat.shape[0]}; expected "
                    f"{self.spec.total_size} (sum of spec entry sizes). "
                    f"dynamic_slice would silently clamp a wrong size."
                )
            # Slice by control-vector entries and smooth each field on its own.
            parts = []
            for entry in self.spec.entries:
                flat_slice = jax.lax.dynamic_slice(
                    field_flat, (entry.offset,), (entry.size,)
                )
                parts.append(_block(flat_slice))
            return jnp.concatenate(parts)

        if field_flat.ndim == 1:
            # Legacy single-field path (no spec): treat the whole vector as one
            # (ncol, nlev) field.
            if field_flat.size > ncol:
                nlev = field_flat.size // ncol
                field_2d = field_flat.reshape(ncol, nlev)
            else:
                field_2d = field_flat.reshape(ncol, 1)
            return block_op(field_2d).ravel()

        # Already-2D input: a single (ncol, nlev) field block.
        return block_op(field_flat)

    def _smooth_field(self, field_flat: jax.Array, n_iter: int) -> jax.Array:
        """Apply n_iter iterations of implicit diffusion smoothing.

        Uses iterative Jacobi smoothing as a proxy for (I - kappa*nabla^2)^{-1}.
        Applied per control field on the flattened column representation.
        """
        # Area weights for normalization
        area = self.grid.to_columns(self.grid.grid_area)
        area_norm = area / jnp.sum(area)

        # Smoothing factor per iteration (already clipped to [0, 0.5])
        alpha = self.kappa

        def smooth_step(x, _):
            # Relax toward area-weighted global mean
            mean_x = jnp.sum(x * area_norm[:, None], axis=0, keepdims=True)
            x_new = x + alpha * (mean_x - x)
            return x_new, None

        def block_op(field_2d):
            result, _ = jax.lax.scan(smooth_step, field_2d, None, length=n_iter)
            return result

        return self._apply_per_field(field_flat, block_op)

    def sqrt_multiply(self, x: jax.Array) -> jax.Array:
        """Apply B^{1/2} to control vector x. Differentiable."""
        # B^{1/2} = Sigma @ C^{1/2}
        smoothed = self._smooth_field(x, self.n_iter // 2 + 1)
        return self.sigma * smoothed

    def inv_multiply(self, x: jax.Array) -> jax.Array:
        """Apply the EXACT inverse of ``B = sqrt_multiply ∘ sqrt_multiplyᵀ``.

        ``sqrt_multiply`` is ``B^{1/2} = Σ·Sᵐ`` with ``m = n_iter//2 + 1`` and
        ``S`` the area-mean Jacobi smoother, so

            B = Σ Sᵐ (Sᵀ)ᵐ Σ   ⇒   B^{-1} = Σ^{-1} (Sᵀ)^{-m} S^{-m} Σ^{-1}.

        Each smoother step ``S = (1-α)I + α·1·aᵀ`` (``a`` = area weights,
        ``aᵀ1 = 1``) is a rank-1 update of ``(1-α)I``, so its inverse is the
        closed-form Sherman-Morrison step ``S^{-1}z = (z - α·aᵀz)/(1-α)`` (and
        the transpose ``(Sᵀ)^{-1}y = (y - α·a·1ᵀy)/(1-α)``) — NOT the previous
        first-order ``I - αΔ`` approximation, which also used the wrong step
        count (``n_iter`` instead of ``m``) and so was not the inverse of
        ``sqrt_multiply`` (‖B^{-1}B − I‖ was O(1)). With these exact steps and
        matching count, ``inv_multiply(B v) == v`` to machine precision, so the
        change-of-variable ``x = x_b + B^{1/2} v`` and ``J_b = ½ dxᵀ B^{-1} dx``
        are mutually consistent.
        """
        m = self.n_iter // 2 + 1
        x_scaled = x / self.sigma          # Σ^{-1}

        area = self.grid.to_columns(self.grid.grid_area)
        area_norm = area / jnp.sum(area)
        alpha = self.kappa
        inv_damp = 1.0 / (1.0 - alpha)     # α ≤ 0.5 ⇒ inv_damp ∈ [1, 2]

        # S^{-1}: z -> (z - α·(area-weighted mean of z)) / (1-α)
        def inv_step(z, _):
            mean_z = jnp.sum(z * area_norm[:, None], axis=0, keepdims=True)
            return inv_damp * (z - alpha * mean_z), None

        # (Sᵀ)^{-1}: y -> (y - α·area_norm·(column sum of y)) / (1-α)
        def invT_step(y, _):
            col_sum = jnp.sum(y, axis=0, keepdims=True)
            return inv_damp * (y - alpha * area_norm[:, None] * col_sum), None

        def block_op(field_2d):
            field_2d, _ = jax.lax.scan(inv_step, field_2d, None, length=m)   # S^{-m}
            field_2d, _ = jax.lax.scan(invT_step, field_2d, None, length=m)  # (Sᵀ)^{-m}
            return field_2d

        # Apply the inverse smoother per control field (mirrors _smooth_field's
        # spec-aware slicing) so a multi-field control is inverted field-by-field
        # and the exact transpose-consistent identity inv_multiply(B v) == v holds.
        result = self._apply_per_field(x_scaled, block_op)
        return result / self.sigma         # Σ^{-1}


# ---------------------------------------------------------------------------
# Hybrid B (static + ensemble)
# ---------------------------------------------------------------------------

def _gaspari_cohn(r: jax.Array, c: float) -> jax.Array:
    """Gaspari-Cohn 5th-order compactly-supported correlation function.

    Parameters
    ----------
    r : jax.Array
        Distances.
    c : float
        Half-width (correlation vanishes at 2c).

    Returns
    -------
    jax.Array
        Correlation values in [0, 1].
    """
    z = jnp.abs(r) / c
    z = jnp.minimum(z, 2.0)

    # Branch 1: 0 <= z <= 1
    c1 = -0.25 * z**5 + 0.5 * z**4 + 5.0/8.0 * z**3 - 5.0/3.0 * z**2 + 1.0

    # Branch 2: 1 < z <= 2
    c2 = (1.0/12.0 * z**5 - 0.5 * z**4 + 5.0/8.0 * z**3
          + 5.0/3.0 * z**2 - 5.0 * z + 4.0 - 2.0/(3.0 * jnp.maximum(z, 1e-10)))

    return jnp.where(z <= 1.0, c1, jnp.where(z <= 2.0, c2, 0.0))


class HybridB:
    """Hybrid B = beta_s * B_static + beta_e * B_ensemble.

    Parameters
    ----------
    static_B : DiagonalB or DiffusionB
        Static background error covariance.
    ensemble_perts : jax.Array
        Ensemble perturbations, shape (n_members, control_size).
    beta_static : float
        Weight for static component.
    localization_length : float
        Gaspari-Cohn localization half-width [m]. RESERVED — see the note.
    grid : GridProtocol, optional
        Grid for localization distance computation. RESERVED — see the note.

    Notes
    -----
    Gaspari-Cohn localization is NOT yet applied: ``sqrt_multiply`` uses the RAW
    (un-localized) ensemble covariance, and ``localization_length`` / ``grid``
    are accepted but unused (reserved for a future localized implementation;
    ``_gaspari_cohn`` exists and is unit-tested but is not yet wired in here).
    ``inv_multiply`` is an explicit static-only approximation (see its
    docstring). Use ``HybridB`` as an inner-loop preconditioner only — do not
    rely on it for a spatially-localized hybrid B or an exact inverse.
    """

    def __init__(self, static_B, ensemble_perts: jax.Array,
                 beta_static: float = 0.5,
                 localization_length: float = 1000e3,
                 grid=None):
        self.static_B = static_B
        self.ensemble_perts = ensemble_perts  # (n_members, control_size)
        self.beta_s = beta_static
        self.beta_e = 1.0 - beta_static
        self.localization_length = localization_length
        self.grid = grid
        self.n_members = ensemble_perts.shape[0]
        # sqrt_multiply scales the ensemble term by 1/sqrt(n_members - 1); a
        # single-member ensemble makes that 0/0 = NaN (and a 0-member ensemble
        # has no perturbations at all). Require >= 2 members at construction so
        # the divide-by-zero can never reach the AD tape.
        if self.n_members < 2:
            raise ValueError(
                f"HybridB requires an ensemble with >= 2 members, got "
                f"{self.n_members}; the ensemble covariance scaling "
                f"1/sqrt(n_members - 1) is undefined otherwise."
            )

    def sqrt_multiply(self, x: jax.Array) -> jax.Array:
        """Apply B^{1/2} x ≈ sqrt(beta_s)·B_static^{1/2} x + sqrt(beta_e)·ensemble.

        NOTE: the ensemble term uses the RAW (un-localized) ensemble covariance —
        no Gaspari-Cohn localization is applied (see the class Notes). For a
        small/preconditioning ensemble weight this is acceptable.
        """
        static_part = jnp.sqrt(self.beta_s) * self.static_B.sqrt_multiply(x)
        # Ensemble part: project x onto ensemble directions
        # B_e^{1/2} x ≈ (1/sqrt(n-1)) * X_pert @ (X_pert^T @ x) / (n-1)
        # Simplified: scale by ensemble spread
        ens_mean = jnp.mean(self.ensemble_perts, axis=0)
        ens_centered = self.ensemble_perts - ens_mean[None, :]
        coeffs = ens_centered @ x / jnp.sqrt(self.n_members - 1.0)
        ens_part = jnp.sqrt(self.beta_e) * jnp.mean(
            ens_centered * coeffs[:, None], axis=0
        ) * jnp.sqrt(self.n_members - 1.0)
        return static_part + ens_part

    def inv_multiply(self, x: jax.Array) -> jax.Array:
        """Apply approximate B^{-1} x using static component only.

        The full hybrid inverse (B_static + B_ensemble)^{-1} requires
        a Sherman-Morrison-Woodbury solve per evaluation, which is
        expensive.  This approximation uses B_static^{-1} / beta_s
        instead.  It is adequate when:
        - beta_static is close to 1 (ensemble weight is small), or
        - the ensemble subspace is nearly orthogonal to x.

        It degrades when beta_ensemble is large and x projects strongly
        onto the ensemble subspace.  For preconditioning the inner loop
        this is usually acceptable.
        """
        return self.static_B.inv_multiply(x) / self.beta_s
