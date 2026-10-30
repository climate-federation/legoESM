"""#1819: can the GEN_BE background-error covariance be inverted on an MPAS mesh?

``GenBETransform`` smooths each channel with ``n`` explicit diffusion steps,
``U_h = (I + s L)^n`` with ``s = L_c^2 / (2 n)`` (``L_c`` the channel length
scale, ``L`` the finite-volume cell Laplacian).  It multiplies the eigenmode of
the most negative eigenvalue ``lam_min`` of ``L`` by ``(1 + s lam_min)^n``.
While the step is not clipped (``s < 0.49 / max_i sum_j w_ij``, true at the
scales printed here), ``s n = L_c^2 / 2`` is fixed, so this tends to
``exp(-L_c^2 |lam_min| / 2)`` as ``n`` grows: the loss belongs to the
correlation model at that length scale and mesh spacing, not to ``n`` or the
solver, and no inverse can recover a mode attenuated below float64 resolution.

This probe prints, per length scale, the attenuation of the worst-resolved
(grid-scale) mode and the closed form.  An attenuation below ~1e-8 means
``B^{-1}`` cannot be applied to a useful accuracy in float64.

Run on an interactive node::

    python scripts/validate/genbe_mpas_inverse_conditioning_1819.py \
        --resolution 6 --lloyd-iterations 0 --n-diffusion-iter 400
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_POWER_ITERATIONS = 300


def genbe_laplacian(grid, len_scale_m: float = 5.0e5, n_iter: int = 400):
    """The cell Laplacian and diffusion step a real ``GenBETransform`` builds on
    ``grid`` (one channel, length scale ``len_scale_m``).

    Returns ``(lap, step, transform)`` with ``lap(x)`` the operator applied by
    ``_diffusion_smooth``, ``step`` its per-iteration coefficient (clip
    included) and the transform itself.
    """
    from legoesm.da.control_vector import ControlEntry, ControlVectorSpec
    from legoesm.da.gen_be import GenBEParams, GenBETransform

    n = grid.grid_n_columns
    entries, off = [], 0
    for name, shape in (("u", (n, 1)), ("T", (n, 1)), ("p_s", (n,))):
        size = int(np.prod(shape))
        entries.append(ControlEntry(name, off, size, shape, "identity"))
        off += size
    params = GenBEParams(
        vert_eig_vec=jnp.stack([jnp.eye(1)] * 3), vert_eig_val=jnp.ones((3, 1)),
        std_ps=jnp.asarray(1.0), reg_coeff=jnp.zeros((4, 1)),
        len_scale=jnp.full((4,), len_scale_m), tracer_names=(), n_levels=1,
        wind_transform="identity",
    )
    t = GenBETransform(params, ControlVectorSpec(tuple(entries), off), grid, n_diffusion_iter=n_iter)
    weights, neighbors = t._laplacian_weights, t._neighbors
    if weights is None:
        raise ValueError("grid lacks MPAS cell/edge geometry; GenBETransform does not diffuse on it")

    def lap(x):
        return jnp.sum(weights * (x[neighbors] - x[None, :]), axis=0)

    return lap, float(t._diffusion_step_m2[1]), t


def lambda_min(lap, n: int) -> float:
    """Most negative eigenvalue of the MPAS cell Laplacian.

    ``areaCell * L`` is a symmetric graph Laplacian, so ``L`` has a real
    spectrum and power iteration on ``L`` converges to its largest-magnitude
    (most negative) eigenvalue.
    """
    x = jax.random.normal(jax.random.PRNGKey(3), (n,))
    for _ in range(_POWER_ITERATIONS):
        x = lap(x)
        x = x / jnp.linalg.norm(x)
    lam = float(jnp.dot(x, lap(x)))
    if not lam < 0.0:
        raise RuntimeError(f"power iteration returned lambda={lam}; expected the most negative eigenvalue")
    return lam


def attenuation(lam: float, step: float, len_scale_m: float, n_iter: int) -> tuple[float, float]:
    """(discrete ``(1 + step lam)^n``, closed form ``exp(-L^2 |lam| / 2)``)."""
    return (1.0 + step * lam) ** n_iter, float(np.exp(-(len_scale_m**2) * abs(lam) / 2.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resolution", type=int, default=3)
    ap.add_argument("--lloyd-iterations", type=int, default=0)
    ap.add_argument("--n-diffusion-iter", type=int, default=400)
    ap.add_argument("--len-scales-km", type=str, default="100,200,300,500,1000")
    args = ap.parse_args()

    from legoesm.grids.factory import create_grid

    grid = create_grid("mpas", args.resolution, lloyd_iterations=args.lloyd_iterations)
    n = grid.grid_n_columns
    lap, _, _ = genbe_laplacian(grid, n_iter=args.n_diffusion_iter)
    lam = lambda_min(lap, n)
    print(f"resolution={args.resolution} lloyd={args.lloyd_iterations} nCells={n} "
          f"n_iter={args.n_diffusion_iter} lambda_min={lam:.4e} 1/m^2")
    for km in (float(v) for v in args.len_scales_km.split(",")):
        _, step, _ = genbe_laplacian(grid, len_scale_m=km * 1e3, n_iter=args.n_diffusion_iter)
        disc, closed = attenuation(lam, step, km * 1e3, args.n_diffusion_iter)
        print(f"  L={km:6.0f} km  step*|lambda_min|={step * abs(lam):.3e}  "
              f"attenuation={disc:.3e}  closed-form={closed:.3e}")


if __name__ == "__main__":
    main()
