"""Convection output container.

ConvectionOutput is the common interface produced by every convection
backend (SBM, DCA, Kuo, mass-flux, EDMF). Convection no longer carries
its own surface-precipitation diagnostic; instead it emits a 3D
convective source for cloud water (``dq_c_conv_dt``). Downstream the
orchestrator routes that source into the cloud-water budget so that
microphysics processes the convective condensate through its
autoconversion / sedimentation / evaporation chain. Total surface
precipitation is the sole responsibility of microphysics
(``MicrophysicsOutput.precipitation``).

This routing was introduced because the previous scalar
``precipitation`` field forced every convection scheme to assume that
all detrained condensate falls instantly to the surface, bypassing
melting, evaporation in dry layers, and proper terminal-velocity
sedimentation — a restrictive simplification.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class ConvectionOutput(NamedTuple):
    """Output from a convection scheme (backend-agnostic).

    All required fields are at full levels with shape ``(ncol, nlev)``
    except column-mean diagnostics which have shape ``(ncol,)``.

    Fields
    ------
    dT_dt : jax.Array
        Temperature tendency [K/s], shape (ncol, nlev).
    dq_v_dt : jax.Array
        Water vapor specific humidity tendency [kg/kg/s], shape
        (ncol, nlev).
    dq_c_conv_dt : jax.Array
        Convective source term for cloud water mixing ratio [kg/kg/s],
        shape (ncol, nlev). Replaces the legacy scalar surface
        ``precipitation``: convective condensate now joins the
        cloud-water bucket and is processed by microphysics, which
        owns the resulting surface precipitation diagnostic. By
        construction this field is non-negative.
    cape : jax.Array
        CAPE diagnostic [J/kg], shape (ncol,).
    convective_mask : jax.Array
        Smooth 0-1 convective indicator, shape (ncol,).
    du_dt_conv : jax.Array or None
        Optional convective momentum tendency for zonal wind [m/s²],
        shape (ncol, nlev).  ``None`` for schemes without convective
        momentum transport (the existing five schemes plus the
        Kain-Fritsch and Emanuel schemes added in PRs 2 and 3).
        Populated by Zhang-McFarlane (PR 1), Tiedtke (PR 4), and
        Bechtold/IFS (PR 5) via the Gregory et al. 1997 closure.  When
        ``None``, the integration bridge zero-fills the dynamical-core
        wind tendencies.
    dv_dt_conv : jax.Array or None
        Optional convective momentum tendency for meridional wind
        [m/s²], shape (ncol, nlev).  Same conventions as ``du_dt_conv``.
    """
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    dq_c_conv_dt: jax.Array
    cape: jax.Array
    convective_mask: jax.Array
    du_dt_conv: jax.Array | None = None
    dv_dt_conv: jax.Array | None = None
