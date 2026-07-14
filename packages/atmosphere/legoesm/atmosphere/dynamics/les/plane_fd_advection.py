"""Shared flux-form finite-difference advection for plane LES on the
horizontally-collocated, w-staggered ``(ny, nx, nz)`` layout.

This is the common FD advection kernel for the plane LES dycores: it reuses the
repo's public reconstruction kernels (:func:`legoesm.core.weno.weno5_z`,
:func:`legoesm.core.flux_limiters.van_leer_face_values`) and adds only the
layout-specific flux assembly — no re-derivation of the reconstruction numerics.
The pseudo-incompressible plane dycore consumes it; the spectral and compressible
plane dycores can adopt it in follow-up (they currently inline their own assembly
around the same shared kernels).

Conventions (match ``spectral_les_plane`` / ``pseudo_incompressible_poisson``)
-----------------------------------------------------------------------------
* Physical layout ``(ny, nx, nz)``: y axis 0, x axis 1, z axis 2.
* Scalars live at cell CENTRES ``(ny, nx, nz)``. The advecting velocities ``u``, ``v``
  may be COLLOCATED at centres (``vel_at_faces=False``) or on the x/y FACES
  (``vel_at_faces=True``, the Arakawa C-grid case used by the pseudo-incompressible core).
* ``w`` lives at z-FACES ``(ny, nx, nz+1)`` with rigid walls ``w[..,0]=w[..,nz]=0``.
* Horizontal (x, y) is PERIODIC (``jnp.roll`` neighbours). The vertical has rigid
  walls: NO advective flux through the ground/lid (flux ``= 0`` at the z walls).
* Uniform ``Δx, Δy, Δz``.

The tendency returned is the ADVECTIVE form ``−u·∇φ`` written conservatively as
``−∇·(uφ) + φ ∇·u`` (the ``φ ∇·u`` term cancels the compressible part so a constant
field is preserved and the operator reduces to pure advection when ``∇·u=0`` — the
post-projection state). This mirrors LEX's ``advection_scalar`` flow-divergence
correction (``advection.py``).

Pure-pytree, JIT/``jax.grad``/vmap-safe.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.flux_limiters import van_leer_face_values
from legoesm.core.weno import weno5_z, weno7_z, weno9_z

_AY, _AX, _AZ = 0, 1, 2
# "central" = 2nd-order central (non-dissipative); the others are upwind-biased.
# Central is for LES MOMENTUM, where WENO5's inherent upwind k⁶ diffusion
# over-smooths the resolved eddies (cf. the spectral core's energy-conserving
# advection). It carries no numerical dissipation, so it relies on the SGS (and
# optionally hyperdiff/shapiro) for 2Δ control — do NOT use it for scalars that
# need monotonicity (θ/moisture).
_SCHEMES = ("upwind", "van_leer", "weno5", "weno7", "weno9", "central")


def _upwind_face(fm1, f0, vel_pos):
    """1st-order upwind face value given the sign of the advecting face velocity."""
    return jnp.where(vel_pos, fm1, f0)


def _face_values_x(phi, scheme):
    """Left/right reconstructions of ``phi`` at the i+½ x-faces (periodic)."""
    r = lambda s: jnp.roll(phi, s, axis=_AX)        # r(+1)=phi_{i-1}
    if scheme == "upwind":
        return r(0), r(-1)                           # left=phi_i, right=phi_{i+1}
    if scheme == "van_leer":
        # one call returns (phi_pos, phi_neg) = (left-biased, right-biased) at i+½
        return van_leer_face_values(r(1), r(0), r(-1), r(-2))
    if scheme == "weno7":   # [f_{i-3..i+4}] — less upwind dissipation than weno5
        return weno7_z([r(3), r(2), r(1), r(0), r(-1), r(-2), r(-3), r(-4)])
    if scheme == "weno9":   # [f_{i-4..i+5}]
        return weno9_z([r(4), r(3), r(2), r(1), r(0), r(-1), r(-2), r(-3), r(-4), r(-5)])
    # weno5: stencil [f_{i-2..i+3}] for face i+½
    stencil = [r(2), r(1), r(0), r(-1), r(-2), r(-3)]
    return weno5_z(stencil)


def _face_values_y(phi, scheme):
    r = lambda s: jnp.roll(phi, s, axis=_AY)
    if scheme == "upwind":
        return r(0), r(-1)
    if scheme == "van_leer":
        return van_leer_face_values(r(1), r(0), r(-1), r(-2))
    if scheme == "weno7":
        return weno7_z([r(3), r(2), r(1), r(0), r(-1), r(-2), r(-3), r(-4)])
    if scheme == "weno9":
        return weno9_z([r(4), r(3), r(2), r(1), r(0), r(-1), r(-2), r(-3), r(-4), r(-5)])
    stencil = [r(2), r(1), r(0), r(-1), r(-2), r(-3)]
    return weno5_z(stencil)


def _upwind_flux_h(phi, vel, axis, dx, scheme, vel_at_faces=False):
    """Conservative horizontal flux divergence ``∂(vel·phi)/∂axis`` (periodic).

    ``phi`` is cell-centred. ``vel`` is the advecting velocity: COLLOCATED at centres
    (``vel_at_faces=False``, averaged to the i+½ faces) or already on the i+½ faces
    (``vel_at_faces=True``, the C-grid case — ``vel[...,i]`` is the velocity at face
    i+½). The upwind reconstruction side is chosen by the sign of the face velocity.
    """
    vface = vel if vel_at_faces else 0.5 * (vel + jnp.roll(vel, -1, axis=axis))
    if scheme == "central":
        # 2nd-order central i+½ interpolation — NON-dissipative (no upwind pick).
        phi_face = 0.5 * (phi + jnp.roll(phi, -1, axis=axis))
    else:
        fv = _face_values_x if axis == _AX else _face_values_y
        f_left, f_right = fv(phi, scheme)
        phi_face = jnp.where(vface >= 0.0, f_left, f_right)
    flux = vface * phi_face                                 # flux at i+½
    # divergence: (F_{i+½} − F_{i−½})/Δ
    return (flux - jnp.roll(flux, 1, axis=axis)) / dx


def _flux_div_z(phi, w, dz, scheme):
    """Conservative vertical flux divergence ``∂(w·phi)/∂z`` with rigid walls.

    ``phi`` centres ``(…,nz)``; ``w`` faces ``(…,nz+1)`` with ``w[..,0]=w[..,nz]=0``.
    Interior faces k+½ (k=0..nz-2) carry the flux; wall faces carry zero (no flux
    through ground/lid). Upwind side from the sign of the interior face ``w``.
    """
    wf = w[..., 1:-1]                                       # interior faces (…,nz-1)
    if scheme == "central":
        # 2nd-order central at interior faces — NON-dissipative (no upwind pick).
        phi_face = 0.5 * (phi[..., :-1] + phi[..., 1:])
    else:
        # reconstruct phi at interior faces from the two adjacent centres (upwind-biased)
        if scheme == "weno5":
            # 6-cell vertical stencil [k-2..k+3] at interior faces; clamp walls by edge-repeat.
            pp = jnp.pad(phi, [(0, 0), (0, 0), (2, 3)], mode="edge")
            st = [pp[..., j:j + (phi.shape[-1] - 1)] for j in range(6)]
            f_left, f_right = weno5_z(st)
        elif scheme == "weno7":               # 8-cell [k-3..k+4]
            pp = jnp.pad(phi, [(0, 0), (0, 0), (3, 4)], mode="edge")
            st = [pp[..., j:j + (phi.shape[-1] - 1)] for j in range(8)]
            f_left, f_right = weno7_z(st)
        elif scheme == "weno9":               # 10-cell [k-4..k+5]
            pp = jnp.pad(phi, [(0, 0), (0, 0), (4, 5)], mode="edge")
            st = [pp[..., j:j + (phi.shape[-1] - 1)] for j in range(10)]
            f_left, f_right = weno9_z(st)
        elif scheme == "van_leer":
            # face k+½ stencil [φ_{k-1},φ_k,φ_{k+1},φ_{k+2}]; pad(2,2) ⇒ pp[2+k]=φ_k.
            pp = jnp.pad(phi, [(0, 0), (0, 0), (2, 2)], mode="edge")
            n = phi.shape[-1] - 1                           # number of interior faces
            f_left, f_right = van_leer_face_values(
                pp[..., 1:n + 1], pp[..., 2:n + 2], pp[..., 3:n + 3], pp[..., 4:n + 4])
        else:  # upwind
            f_left, f_right = phi[..., :-1], phi[..., 1:]
        phi_face = jnp.where(wf >= 0.0, f_left, f_right)
    flux_int = wf * phi_face                                # (…,nz-1)
    flux = jnp.pad(flux_int, [(0, 0), (0, 0), (1, 1)])     # 0 at walls → (…,nz+1)
    return (flux[..., 1:] - flux[..., :-1]) / dz           # (…,nz)


def divergence_centre(u, v, w, dx, dy, dz, vel_at_faces=False):
    """Cell-centred divergence ``∂u/∂x+∂v/∂y+∂w/∂z``.

    ``vel_at_faces=False``: collocated u,v (centred 2Δ horizontal). ``vel_at_faces=True``:
    C-grid — ``u[...,i]`` at x-face i+½, ``v`` at y-face j+½ ⇒ COMPACT difference
    ``(u_{i+½}−u_{i−½})/Δx`` (consistent with the compact Poisson Laplacian). ``w`` is
    always on z-faces.
    """
    if vel_at_faces:
        du = (u - jnp.roll(u, 1, axis=_AX)) / dx
        dv = (v - jnp.roll(v, 1, axis=_AY)) / dy
    else:
        du = (jnp.roll(u, -1, axis=_AX) - jnp.roll(u, 1, axis=_AX)) / (2.0 * dx)
        dv = (jnp.roll(v, -1, axis=_AY) - jnp.roll(v, 1, axis=_AY)) / (2.0 * dy)
    dw = (w[..., 1:] - w[..., :-1]) / dz
    return du + dv + dw


def advect_scalar(phi, u, v, w, dx, dy, dz, scheme="weno5", vel_at_faces=False):
    """Advective tendency ``−u·∇φ`` in flow-divergence-corrected flux form.

    ``φ`` centres ``(ny,nx,nz)``; ``w`` faces ``(ny,nx,nz+1)``. ``u, v`` are centred
    (``vel_at_faces=False``) or on the x/y faces (``vel_at_faces=True``, C-grid). With
    ``∇·u=0`` (post-projection) this is exact conservative advection; the ``φ ∇·u``
    correction keeps a constant field constant when the divergence is not machine-zero.
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}.")
    flux_div = (_upwind_flux_h(phi, u, _AX, dx, scheme, vel_at_faces)
                + _upwind_flux_h(phi, v, _AY, dy, scheme, vel_at_faces)
                + _flux_div_z(phi, w, dz, scheme))
    div_u = divergence_centre(u, v, w, dx, dy, dz, vel_at_faces)
    return -flux_div + phi * div_u


def advect_momentum(u, v, w, dx, dy, dz, scheme="weno5", vel_at_faces=False):
    """Advective tendencies ``(−u·∇u, −u·∇v, −u·∇w)`` for the velocities.

    Each velocity component is advected component-wise as a scalar by ``(u, v, w)``
    (``vel_at_faces`` selects the C-grid face-velocity treatment). ``w`` (faces) is
    advected via its face→centre value, then the tendency is mapped back to faces; the
    rigid walls (``w=0``) are reimposed by the caller. Component-wise scalar advection
    is the standard plane-LES treatment for smooth flows (Durran); a fully
    momentum-conservative C-grid flux form is a documented upgrade. (The divergence-free
    constraint is enforced exactly by the pressure projection regardless of the
    advection form.)
    """
    au = advect_scalar(u, u, v, w, dx, dy, dz, scheme, vel_at_faces)
    av = advect_scalar(v, u, v, w, dx, dy, dz, scheme, vel_at_faces)
    aw_c = advect_scalar(_face_to_centre_z(w), u, v, w, dx, dy, dz, scheme, vel_at_faces)
    aw = _centre_to_face_z(aw_c)
    return au, av, aw


def _centre_to_face_z(fc):
    """Average a centred ``(…,nz)`` field to z-faces ``(…,nz+1)`` (wall=edge)."""
    inner = 0.5 * (fc[..., :-1] + fc[..., 1:])
    return jnp.concatenate([fc[..., :1], inner, fc[..., -1:]], axis=_AZ)


def _face_to_centre_z(ff):
    """Average a face ``(…,nz+1)`` field to centres ``(…,nz)``."""
    return 0.5 * (ff[..., :-1] + ff[..., 1:])
