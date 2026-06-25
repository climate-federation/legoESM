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

    def _smooth_field(self, field_flat: jax.Array, n_iter: int) -> jax.Array:
        """Apply n_iter iterations of implicit diffusion smoothing.

        Uses iterative Jacobi smoothing as a proxy for (I - kappa*nabla^2)^{-1}.
        Applied on the flattened column representation.
        """
        ncol = self.grid.grid_n_columns

        # Handle multi-level fields
        if field_flat.ndim == 1 and field_flat.size > ncol:
            nlev = field_flat.size // ncol
            field_2d = field_flat.reshape(ncol, nlev)
        elif field_flat.ndim == 1:
            field_2d = field_flat.reshape(ncol, 1)
        else:
            field_2d = field_flat

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

        result, _ = jax.lax.scan(smooth_step, field_2d, None, length=n_iter)

        if field_flat.ndim == 1:
            return result.ravel()
        return result

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

        ncol = self.grid.grid_n_columns
        if x_scaled.ndim == 1 and x_scaled.size > ncol:
            nlev = x_scaled.size // ncol
            field_2d = x_scaled.reshape(ncol, nlev)
        elif x_scaled.ndim == 1:
            field_2d = x_scaled.reshape(ncol, 1)
        else:
            field_2d = x_scaled

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

        field_2d, _ = jax.lax.scan(inv_step, field_2d, None, length=m)   # S^{-m}
        field_2d, _ = jax.lax.scan(invT_step, field_2d, None, length=m)  # (Sᵀ)^{-m}

        result = field_2d.ravel() if x_scaled.ndim == 1 else field_2d
        return result / self.sigma         # Σ^{-1}


# ---------------------------------------------------------------------------
# Spectral B (for GaussianGrid)
# ---------------------------------------------------------------------------

class SpectralB:
    """Spectral background error for GaussianGrid.

    B is diagonal in spectral space with prescribed power spectrum:
    sigma^2(n) = sigma_0^2 * (n(n+1) / n_0(n_0+1))^{-alpha}

    Parameters
    ----------
    grid : GaussianGrid
        Must have ls, ms, n_sh attributes.
    sigma_0 : float
        Base standard deviation.
    n_0 : int
        Decorrelation wavenumber.
    alpha : float
        Spectral slope.
    spec : ControlVectorSpec, optional
        Control vector specification.
    """

    def __init__(self, grid, sigma_0: float, n_0: int = 10,
                 alpha: float = 2.0, spec=None):
        self.grid = grid
        self.sigma_0 = sigma_0
        self.n_0 = n_0
        self.alpha = alpha
        self.spec = spec

        # Build spectral variance profile
        ls = grid.ls  # Total wavenumber for each spectral index
        n0_factor = n_0 * (n_0 + 1.0)
        n_factor = ls * (ls + 1.0)
        # Avoid division by zero at n=0
        ratio = jnp.where(ls > 0, n_factor / n0_factor, 1.0)
        self.variance = sigma_0 ** 2 * jnp.power(ratio, -alpha)
        self.variance = self.variance.at[0].set(sigma_0 ** 2)
        self.std = jnp.sqrt(self.variance)

    def sqrt_multiply(self, x: jax.Array) -> jax.Array:
        """Apply B^{1/2} to control vector x."""
        return self.std * x

    def inv_multiply(self, x: jax.Array) -> jax.Array:
        """Apply B^{-1} to control vector x."""
        return x / self.variance


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
    static_B : DiagonalB, DiffusionB, or SpectralB
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
