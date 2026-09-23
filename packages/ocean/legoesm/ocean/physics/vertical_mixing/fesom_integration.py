"""legoESM (NEMO-card) TKE closure on FESOM node columns.

The three OMIP grids must run ONE vertical-mixing closure.  FESOM's own
schemes (fesom_jax PP / KPP / cvmix-TKE) gave the Southern Ocean a 22 m
mixed layer against NEMO's 41 m and a +1.45 C summer SST at day 30 while
the tripole and MPAS (legoESM TKE, ORCA1 ``&namzdf_tke`` card) sat at
+0.2.  This bridge is the FESOM twin of
:func:`legoesm.ocean.physics.vertical_mixing.mpas_integration.make_tke_profiles_mpas`:
the SAME grid-agnostic :func:`tke_vertical_mixing` on the node columns, the
result handed to ``fesom_jax.step(vertical_mixing=(Kv, Av))`` which bypasses
the internal scheme.  Nothing of the closure is re-derived here.

Layouts (fesom_jax carries ``nl = nlev + 1`` vertical slots):

* legoESM half-levels ``(nod2D, nlev-1)`` = interfaces between layers
  ``k`` and ``k+1``; fesom ``Kv[:, k]`` is the interface ABOVE layer ``k``
  (``tracer_diff.py:39-40``), so ``Kv[:, 1:nlev] = K_H``, ``Kv[:, 0]`` and
  ``Kv[:, nlev:]`` are 0 (no flux through surface / below the bottom).
* ``Av`` lives on elements ``(elem2D, nl)``: the node ``K_M`` averaged over
  the element's three vertices (``mesh.elem_nodes``).
* Prognostic TKE rides in the existing fesom ``State.tke`` ``(nod2D, nl)``
  interface slot with the same indexing as ``Kv``.

Deliberately NOT ported (ledger, measured before any claim): the bottom
partial cell (fesom's ``hnode`` last wet layer is thinner than ``dz_ref``;
the closure sees the uniform z* stretch only), and fesom's ``mo_convect``
convective adjustment (the NEMO card runs ``convection='none'`` with the
closure handling static instability, as on the other two grids).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
    bn2_ladder_kwargs,
)
from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing


def fesom_zgeom(mesh) -> SimpleNamespace:
    """Uniform reference ladder of a fesom mesh in legoESM ``z_coord`` terms.

    ``mesh.zbar`` is the NEGATIVE-down interface ladder ``(nl,)``, ``mesh.Z``
    the mid-levels ``(nl-1,)``.  Depth-positive ladders (``t_depth_ref``,
    ``w_depth_ref``) are what ``eos.nemo_bn2_live_geometry`` reads.
    """
    zbar = np.asarray(mesh.zbar, dtype=np.float64)
    zc = np.asarray(mesh.Z, dtype=np.float64)
    nlev = zc.size
    dz_ref = -np.diff(zbar)                                  # (nlev,) > 0
    return SimpleNamespace(
        n_levels=nlev,
        dz_ref=jnp.asarray(dz_ref),
        z_full_ref=jnp.asarray(zc),                          # negative-down
        z_half_ref=jnp.asarray(zbar[:nlev + 1]),             # (nlev+1,)
        dz_half_ref=jnp.asarray(0.5 * (dz_ref[:-1] + dz_ref[1:])),
        t_depth_ref=jnp.asarray(-zc),
        w_depth_ref=jnp.asarray(-zbar[:nlev + 1]),
        linear_free_surface=False,
    )


def make_tke_profiles_fesom(config: VerticalMixingConfig, eos_fn=None,
                            constants_config: ConstantsConfig = ConstantsConfig(),
                            iwm_fields=None) -> Callable:
    """Build ``profiles_fn(state, mesh, zgeom, surface_forcing, dt_tke)``
    returning ``(Kv_nod, Av_elem, tke_new_nl)`` for ``fesom_jax.step``.

    Requires ``config.scheme == "tke"`` and ``config.tke.prognostic`` (the
    ORCA1 card's ``--tke-prognostic``; the diagnostic sub-iterated mode is
    not offered on FESOM — one closure mode across the three grids).
    Options the MPAS bridge rejects are rejected here for the same reasons.
    """
    if config.scheme != "tke" or config.tke is None:
        raise ValueError(
            f"make_tke_profiles_fesom needs VerticalMixingConfig(scheme='tke'), "
            f"got scheme={config.scheme!r}")
    cfg = config.tke
    if not bool(getattr(cfg, "prognostic", False)):
        raise NotImplementedError(
            "FESOM runs the legoESM TKE closure PROGNOSTICALLY only "
            "(--tke-prognostic, the ORCA1 card): the diagnostic sub-iterated "
            "mode is not offered on this grid.")
    for name, bad in (("bottom_tke_bc", True), ("veros_dz_slots", True),
                      ("source_eke_diss", True)):
        if bool(getattr(cfg, name, False)) == bad:
            raise NotImplementedError(
                f"vertical_mixing.tke.{name}={bad} is not wired on the FESOM "
                "TKE bridge (same limits as the MPAS bridge).")
    if getattr(cfg, "n2_mode", "insitu") not in ("insitu", "nemo_bn2"):
        raise NotImplementedError(
            f"vertical_mixing.tke.n2_mode={cfg.n2_mode!r} is not wired on FESOM "
            "(insitu / nemo_bn2 only, as on MPAS).")
    if getattr(cfg, "buoyancy_timing", "pre_mixing") != "pre_mixing":
        raise NotImplementedError(
            "vertical_mixing.tke.buoyancy_timing != 'pre_mixing' is not wired "
            "on FESOM.")
    if getattr(cfg, "advection_scheme", "none") != "none":
        raise NotImplementedError(
            "vertical_mixing.tke.advection_scheme != 'none' is not wired on "
            "FESOM (no node-cloud TKE advection operator).")
    _iwm_cfg = (config.iwm if (getattr(config, "iwm", None) is not None
                               and config.iwm.enabled) else None)
    _eice = int(getattr(cfg, "eice", 0))
    if _eice not in (0, 1, 3):
        raise ValueError(f"Unknown TKEConfig.eice={_eice!r}; expected 0, 1 or 3.")

    from legoesm.ocean.eos import compute_ocean_rho

    def profiles_fn(state, mesh, zgeom, surface_forcing=None, dt_tke=None):
        if dt_tke is None:
            raise ValueError("profiles_fn needs dt_tke (the model timestep).")
        nlev = int(zgeom.n_levels)
        nl = nlev + 1
        inner = state.inner
        # Live layer thickness: fesom's z* stretches only the layers above
        # a set depth (fesom_jax/ale.py), so the centre-to-centre spacing
        # comes from the LIVE hnode column, not from a uniform stretch
        # (codex 9600324 MAJOR).  J (surface stretch) only feeds the
        # reference-ladder pieces of the closure.
        hnode = jnp.asarray(inner.hnode, dtype=jnp.float64)[:, :nlev]
        layer_mask = jnp.asarray(mesh.node_layer_mask)[:, :nlev]
        h_live = jnp.where(layer_mask, hnode, zgeom.dz_ref[None, :])
        dz_half = 0.5 * (h_live[:, :-1] + h_live[:, 1:])            # (nod, nlev-1)
        J = jnp.where(layer_mask[:, 0], hnode[:, 0] / zgeom.dz_ref[0], 1.0)
        uv = jnp.asarray(state.uv_node, dtype=jnp.float64)[:, :nlev]
        u_node = jnp.where(layer_mask, uv[..., 0], 0.0)
        v_node = jnp.where(layer_mask, uv[..., 1], 0.0)
        T = jnp.where(layer_mask, state.T.data, 0.0)
        S = jnp.where(layer_mask, state.S.data, 35.0)
        rho = compute_ocean_rho(
            SimpleNamespace(T=SimpleNamespace(data=T), S=SimpleNamespace(data=S)),
            zgeom, J, eos_fn=eos_fn, eos_depth="geometric",
            rho0=constants_config.rho_0, g=constants_config.g)
        tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing is not None else None
        tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing is not None else None
        ice_frac = (getattr(surface_forcing, "ice_concentration", None)
                    if (_eice != 0 and surface_forcing is not None) else None)
        # Forced without an ice field: fail fast (the MPAS-bridge contract).
        # UNFORCED (surface_forcing None, the B1 smoke path): no fluxes, no
        # ice, no wave sources to attenuate — nothing to gate (codex 9600324).
        if _eice != 0 and ice_frac is None and surface_forcing is not None:
            raise ValueError(
                f"TKEConfig.eice={_eice} requires surface_forcing.ice_concentration "
                "on FESOM (attach it as the host loop does for MPAS) or set eice=0.")
        if ice_frac is not None:
            ice_frac = ice_frac if _eice == 1 else jnp.minimum(4.0 * ice_frac, 1.0)
        # Interface mask for legoESM half-levels: interface k (between layers
        # k and k+1) is active when layer k+1 is wet.
        half_mask = layer_mask[:, 1:]
        tke_prev = jnp.asarray(inner.tke, dtype=jnp.float64)[:, 1:nlev]  # (nod, nlev-1)
        tke_seed = jnp.where(half_mask, jnp.maximum(tke_prev, cfg.tke_background),
                             cfg.tke_background)
        eta_state = SimpleNamespace(eta=SimpleNamespace(data=state.eta.data),
                                    H_bathy=SimpleNamespace(data=state.H_bathy.data))
        out = tke_vertical_mixing(
            u_node, v_node, T, S, rho, dz_half,
            tke_old=tke_seed,
            tau_x_surface=tau_x, tau_y_surface=tau_y,
            dt=float(dt_tke), cfg=cfg,
            rho_0=constants_config.rho_0, g=constants_config.g,
            n_iterations=1,
            z_interface=zgeom.z_half_ref[1:-1],
            lat_deg=jnp.degrees(jnp.asarray(mesh.geo_coord_nod2D[:, 1])),
            ice_frac=ice_frac,
            dz_ref=zgeom.dz_ref,
            jacobian=J,
            **bn2_ladder_kwargs(cfg, zgeom, eta_state),
        )
        K_M = out.K_M
        K_H = out.K_H
        if _iwm_cfg is not None:
            # NEMO zdfiwm: internal-wave-driven mixing, ADDITIVE on top of the
            # closure (zdfphy order: closure first, zdf_iwm adds onto avt/avm)
            # — the SAME single-owner kernel the lat-lon and MPAS lanes call,
            # fed this grid's own live geometry and N2 (no re-derivation).
            from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
            from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
                compute_iwm_diffusivity, uniform_iwm_forcing,
            )
            depth_cell = jnp.cumsum(h_live, axis=-1) - 0.5 * h_live   # gdept
            H_col = jnp.sum(h_live, axis=-1)                          # ht
            N2_iwm = compute_N2(rho, dz_half, constants_config.rho_0,
                                g=constants_config.g)
            _fields = (iwm_fields if iwm_fields is not None
                       else uniform_iwm_forcing(_iwm_cfg, H_col.shape,
                                                dtype=K_H.dtype))
            K_iwm, _ = compute_iwm_diffusivity(
                _fields, depth_cell, dz_half, H_col, N2_iwm,
                cfg=_iwm_cfg, rho_0=constants_config.rho_0)
            K_H = K_H + K_iwm
            K_M = K_M + K_iwm
        K_M = jnp.where(half_mask, K_M, 0.0)
        K_H = jnp.where(half_mask, K_H, 0.0)
        tke_h = jnp.where(half_mask, out.tke_new, 0.0)
        nod2D = K_H.shape[0]
        zeros1 = jnp.zeros((nod2D, 1), dtype=K_H.dtype)
        pad_tail = jnp.zeros((nod2D, nl - nlev), dtype=K_H.dtype)  # interfaces >= nlev
        Kv_nod = jnp.concatenate([zeros1, K_H, pad_tail], axis=1)      # (nod, nl)
        Km_nod = jnp.concatenate([zeros1, K_M, pad_tail], axis=1)
        tke_nl = jnp.concatenate([zeros1, tke_h, pad_tail], axis=1)
        elem_nodes = jnp.asarray(mesh.elem_nodes)                        # (elem, 3)
        Av_elem = jnp.mean(Km_nod[elem_nodes], axis=1)                   # (elem, nl)
        return Kv_nod, Av_elem, tke_nl

    return profiles_fn
