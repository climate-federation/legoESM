#!/usr/bin/env python3
"""Bit gate for ORCA2's WHOLE lateral momentum viscosity operator.

Round 9 landed the READ coefficient and round 10 transcribed one downstream
difference (a second masking) without landing it.  This gate scores the WHOLE
operator: NEMO's compiled ``dynldf_lev_lap`` vorticity-divergence loops
(``dynldf_lev.f90:121-141``, the ``np_typ_rot`` arm that ``dynldf.f90:85``
dispatches and ``stprk3_stg.f90:493`` calls at Runge-Kutta stage three) are
recomputed here from the RECORD's own inputs -- its mesh metrics, masks and
reference thicknesses, its ``ahmt``/``ahmf``, and its kt=1 step-entry
velocities and sea surface -- and compared with legoESM's PRODUCTION operator,
reached through the production momentum-tendency factory, on the same state.

No new NEMO run: every oracle number is recomputed from recorded inputs
(round 8's technique).

CONTROLLED COMPARISON, stated rather than implied.  NEMO's stage-three call
reads the velocities and the flux stretch at ``Kbb`` and the RHS face-thickness
divisor at ``Kmm``.  legoESM's momentum tendency carries ONE sea surface, so
this gate drives BOTH sides with the SAME recorded step-entry sea surface.
The Kbb-vs-Kmm split is therefore NOT scored here; it is a separate statement
and is reported as such.

WHY NOT kt=1.  ORCA2 starts from REST: the recorded kt=1 step-entry velocity
is identically zero on all 799,200 cells, so at kt=1 this operator is exactly
zero on BOTH sides and every comparison and every ablation is a
zero-against-zero.  The gate therefore defaults to the earliest step entry the
record carries with a non-zero velocity (kt=2), and REFUSES outright if the
requested step's velocity is identically zero.

Rows:
  * the production operator vs the compiled loops, every scoreable wet face;
  * ablation A -- the oracle with legoESM's extra zero/one vertex mask applied
    to ``ahmf`` (round 9's second masking, in isolation);
  * ablation B -- the oracle with legoESM's min-rule face/vertex thicknesses
    in place of the record's own ``e3u_0``/``e3v_0``/``e3f_0``;
  * instrument validation -- on a uniform thickness, unit stretch and all-wet
    masks the transcription and legoESM's operator must agree to 1e-12
    relative, so the harness is checked on a case whose answer is known before
    its disagreement is quoted;
  * a one-representable-value plant on the production output must refuse.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

NX_G, NY_G, NZ = 180, 148, 30
_PP = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
CITATIONS = {
    "operator_dispatch": f"{_PP}/dynldf.f90:85",
    "stage_call": f"{_PP}/stprk3_stg.f90:493",
    "zwf": f"{_PP}/dynldf_lev.f90:123",
    "zwt": f"{_PP}/dynldf_lev.f90:127-129",
    "rhs_u": f"{_PP}/dynldf_lev.f90:133-136",
    "rhs_v": f"{_PP}/dynldf_lev.f90:137-140",
    "r3t": f"{_PP}/domqco.f90:257",
    "r3uv": f"{_PP}/domqco.f90:266-269",
    "r3f": f"{_PP}/domqco.f90:281-285",
    "h0": f"{_PP}/domain.f90:197-205",
    "r1_h0": f"{_PP}/domain.f90:212-215",
    "ssfmask": f"{_PP}/dommsk.f90:247-250",
    "fe3mask": f"{_PP}/dommsk.f90:258",
    "shlat": f"{_PP}/dommsk.f90:269-277",
    "e3f_read": f"{_PP}/domzgr.f90:188",
    "coefficient": f"{_PP}/ldfdyn.f90:348-353",
}


class GateError(RuntimeError):
    """A mechanically binding round-11 condition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


# ---------------------------------------------------------------- record I/O

def _stitch(root: Path, filename: str, names: tuple[str, ...],
            ) -> dict[str, np.ndarray]:
    import netCDF4  # noqa: N813

    out: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for rank in (0, 1):
        path = root / filename.format(rank=rank)
        require(path.is_file(), f"missing recorded stream: {path}")
        with netCDF4.Dataset(path, "r") as ds:
            ds.set_auto_maskandscale(False)
            for name in names:
                require(name in ds.variables, f"{path.name}: no {name}")
                block = np.asarray(ds.variables[name][0], dtype=np.float64)
                out[name].append(block)
    stitched: dict[str, np.ndarray] = {}
    for name, blocks in out.items():
        joined = np.concatenate(blocks, axis=blocks[0].ndim - 1)
        if joined.ndim == 3:            # (z, y, x) -> (y, x, z)
            joined = np.moveaxis(joined, 0, -1)[..., :NZ]
        stitched[name] = np.ascontiguousarray(joined)
    return stitched


def read_entry_frame(root: Path, kt: int) -> dict[str, np.ndarray]:
    """kt step-entry u, v and sea surface on the owned global domain."""
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    blocks: list[dict[str, np.ndarray]] = []
    for rank in (0, 1):
        name = (f"oracle_step_entry_kt{kt:08d}.bin" if rank == 0
                else f"oracle_step_entry_rank{rank:04d}_kt{kt:08d}.bin")
        blocks.append(ladder.read_state_frame(root / name, kt=kt, stage=None))
    return {key: np.concatenate([b[key] for b in blocks], axis=1)
            for key in ("T", "S", "u", "v", "ssh")}


# -------------------------------------------------- the compiled transcription

def _xp(a):                       # a[..., j, i+1, ...] : east, periodic
    return np.roll(a, -1, axis=1)


def _xm(a):                       # a[..., j, i-1, ...] : west, periodic
    return np.roll(a, 1, axis=1)


def _yp(a):                       # a[j+1, i] : north, outside -> NaN
    pad = np.full_like(a[:1], np.nan)
    return np.concatenate([a[1:], pad], axis=0)


def _ym(a):                       # a[j-1, i] : south, outside -> NaN
    pad = np.full_like(a[:1], np.nan)
    return np.concatenate([pad, a[:-1]], axis=0)


def _safe_reciprocal(a):
    positive = a > 0.0
    return np.where(positive, 1.0 / np.where(positive, a, 1.0), 0.0)


def nemo_dynldf_lev_lap_rot(mesh, ahmt, ahmf, u, v, ssh, *,
                            ahmf_extra_mask=None,
                            thickness_override=None,
                            e3t_override=None,
                            metric_override=None,
                            curl_zonal_edge_from_cell=False):
    """NEMO ``dynldf_lev_lap``'s ``np_typ_rot`` arm, statement by statement.

    ``ahmf_extra_mask`` (ablation A) multiplies the stored ``ahmf`` by a
    further zero/one vertex mask -- the thing legoESM's operator does and the
    compiled line ``dynldf_lev.f90:123`` explicitly does NOT ("ahmf already *
    by fmask").  ``thickness_override`` (ablation B) supplies the LIVE
    ``e3u``/``e3v``/``e3f`` directly, in place of the record's own reference
    thicknesses stretched by NEMO's ``r3``.

    ROUND 12 -- the three substitutions that name the residual those two leave.

    ``e3t_override`` replaces the OUTER DIVISOR of the divergence term
    (``dynldf_lev.f90:127``): NEMO divides by its own live ``e3t``, and round
    11's thickness ablation overrode only the FACE and VERTEX thicknesses, so
    this divisor was never substituted.

    ``metric_override`` is a mapping from NEMO metric name (``e1t_e2t``,
    ``e1f_e2f``, ``e1u``, ``e2u``, ``e1v``, ``e2v``) to legoESM's OWN stored
    array for the same role, already on NEMO's index convention.

    ``curl_zonal_edge_from_cell`` weights the circulation's two zonal edges by
    the CELL zonal length at the eastern neighbour instead of ``e1u``, which
    is what legoESM's shared ``curl_vertex_cgrid`` does
    (``operators_latlon_cgrid.py:1252-1270`` reads ``grid.dx_T``, the T-point
    metric, where ``dynldf_lev.f90:125`` reads ``e1u``).
    """
    e1t, e2t = mesh["e1t"], mesh["e2t"]
    e1u, e2u = mesh["e1u"], mesh["e2u"]
    e1v, e2v = mesh["e1v"], mesh["e2v"]
    e1f, e2f = mesh["e1f"], mesh["e2f"]
    tmask, umask, vmask = mesh["tmask"], mesh["umask"], mesh["vmask"]
    e3t0, e3u0, e3v0, e3f0 = (mesh["e3t_0"], mesh["e3u_0"],
                              mesh["e3v_0"], mesh["e3f_0"])

    # Round 12, substitution A-metric.  The reference geometry (the reference
    # thicknesses, the column sums and their reciprocals) stays NEMO's -- only
    # the metrics the four compiled statements read are swapped, so this row
    # isolates the stored-metric values and nothing else.
    e1t_e2t = e1t * e2t
    e1f_e2f = e1f * e2f
    if metric_override is not None:
        e1t_e2t = metric_override.get("e1t_e2t", e1t_e2t)
        e1f_e2f = metric_override.get("e1f_e2f", e1f_e2f)
        e1u = metric_override.get("e1u", e1u)
        e2u = metric_override.get("e2u", e2u)
        e1v = metric_override.get("e1v", e1v)
        e2v = metric_override.get("e2v", e2v)

    r1_e1e2t = _safe_reciprocal(e1t_e2t)
    r1_e1e2u = _safe_reciprocal(mesh["e1u"] * mesh["e2u"])
    r1_e1e2v = _safe_reciprocal(mesh["e1v"] * mesh["e2v"])
    r1_e1e2f = _safe_reciprocal(e1f_e2f)
    r1_e1u = _safe_reciprocal(e1u)
    r1_e2u = _safe_reciprocal(e2u)
    r1_e1v = _safe_reciprocal(e1v)
    r1_e2v = _safe_reciprocal(e2v)

    # domain.f90:197-205 reference column thicknesses; :212-215 reciprocals.
    # dommsk.f90:258 freezes fe3mask from the FREE-SLIP fmask (:247-250 takes
    # ssfmask from the same array), BEFORE rn_shlat rewrites fmask at :269-277
    # -- so the free-slip four-T-cell product is what enters the stretch, not
    # the 0/0.5/1/2 field the mesh mask file stores.
    fs_fmask = tmask * _xp(tmask) * _yp(tmask) * _xp(_yp(tmask))
    ht0 = np.sum(e3t0 * tmask, axis=-1)
    hu0 = np.sum(e3u0 * umask, axis=-1)
    hv0 = np.sum(e3v0 * vmask, axis=-1)
    hf0 = np.sum(e3f0 * vmask * _xp(vmask), axis=-1)
    ssmask = np.max(tmask, axis=-1)
    ssumask = np.max(umask, axis=-1)
    ssvmask = np.max(vmask, axis=-1)
    # NOT nanmax: the northern row's four-T-cell product is NaN by
    # construction (its jj+1 neighbour is outside the owned block), and
    # that NaN must PROPAGATE so the row is excluded from scoring rather
    # than silently given a value.
    ssfmask = np.max(fs_fmask, axis=-1)
    r1_ht0 = ssmask / (ht0 + 1.0 - ssmask)
    r1_hu0 = ssumask / (hu0 + 1.0 - ssumask)
    r1_hv0 = ssvmask / (hv0 + 1.0 - ssvmask)
    r1_hf0 = ssfmask / (hf0 + 1.0 - ssfmask)

    # domqco.f90:257,266-269,281-285
    at = (e1t * e2t) * ssh
    r3t = ssh * r1_ht0
    r3u = 0.5 * (at + _xp(at)) * r1_hu0 * r1_e1e2u
    r3v = 0.5 * (at + _yp(at)) * r1_hv0 * r1_e1e2v
    r3f = 0.25 * ((at + _xp(at)) + (_yp(at) + _xp(_yp(at)))) * r1_hf0 * r1_e1e2f

    e3t_L = e3t0 * (1.0 + r3t[..., None] * tmask)
    if e3t_override is not None:
        e3t_L = e3t_override
    e3u_L = e3u0 * (1.0 + r3u[..., None] * umask)
    e3v_L = e3v0 * (1.0 + r3v[..., None] * vmask)
    e3f_L = e3f0 * (1.0 + r3f[..., None] * fs_fmask)
    if thickness_override is not None:
        e3u_L, e3v_L, e3f_L = thickness_override

    ahmf_used = ahmf if ahmf_extra_mask is None else ahmf * ahmf_extra_mask

    # dynldf_lev.f90:123 -- zwf(ji-1,jj-1), written on the F index it lands on.
    e2v_v = e2v[..., None] * v
    # dynldf_lev.f90:125 weights the two zonal edges of the circulation loop
    # by e1u.  legoESM's shared curl reads the CELL metric of the EASTERN
    # neighbour for the same edge (operators_latlon_cgrid.py:1262-1270).
    zonal_edge = _xp(e1t) if curl_zonal_edge_from_cell else e1u
    e1u_u = zonal_edge[..., None] * u
    circulation = (_xp(e2v_v) - e2v_v) - (_yp(e1u_u) - e1u_u)
    zwf = ahmf_used * e3f_L * r1_e1e2f[..., None] * circulation

    # dynldf_lev.f90:127-129 -- zwt(ji,jj)
    fu = (e2u[..., None] * e3u_L) * u
    fv = (e1v[..., None] * e3v_L) * v
    flux_sum = (fu - _xm(fu)) + (fv - _ym(fv))
    zwt = (ahmt * r1_e1e2t[..., None]
           / np.where(e3t_L > 0.0, e3t_L, 1.0)) * flux_sum

    # dynldf_lev.f90:133-136 and :137-140
    e3u_div = np.where(e3u_L > 0.0, e3u_L, 1.0)
    e3v_div = np.where(e3v_L > 0.0, e3v_L, 1.0)
    du = umask * (-(zwf - _ym(zwf)) * r1_e2u[..., None] / e3u_div
                  + (_xp(zwt) - zwt) * r1_e1u[..., None])
    dv = vmask * ((zwf - _xm(zwf)) * r1_e1v[..., None] / e3v_div
                  + (_yp(zwt) - zwt) * r1_e2v[..., None])
    return du, dv


# ------------------------------------------------------------------- scoring

def score(candidate, oracle, weight):
    """Score on the cells where the oracle is defined and the face is wet."""
    valid = np.isfinite(oracle) & np.isfinite(candidate) & (weight > 0.0)
    n = int(valid.sum())
    require(n > 0, "no scoreable cells")
    a = candidate[valid]
    b = oracle[valid]
    unequal_mask = (np.ascontiguousarray(a).view(np.uint64)
                    != np.ascontiguousarray(b).view(np.uint64))
    unequal = int(unequal_mask.sum())
    delta = np.abs(a - b)
    scale = np.maximum(np.abs(b), 1e-300)
    return {
        "scored_cells": n,
        "unequal": unequal,
        "bit_identical": unequal == 0,
        "max_abs": float(delta.max()) if n else 0.0,
        "max_rel": float((delta / scale).max()) if n else 0.0,
        "l2_ratio": float(np.sqrt((delta ** 2).sum()
                                  / max((b ** 2).sum(), 1e-300))),
    }


def _plant_one_value(a):
    out = np.array(a, dtype=np.float64, copy=True)
    flat = out.reshape(-1)
    idx = int(np.argmax(np.abs(np.nan_to_num(flat))))
    flat[idx] = np.nextafter(flat[idx], np.inf)
    return out


# ---------------------------------------------------------- the candidate side

def _nemo_u_to_legoesm(u_nemo):
    """NEMO ``u(jj, ji)`` onto legoESM's ``(n_lat, n_lon + 1)`` u-face array.

    ``card_fields`` scores ``state.u.data[:, 1:91]`` against the rank-0 record,
    so legoESM ``u[j, i + 1] == NEMO u[jj=j, ji=i]``; column 0 is the periodic
    image of the last face.
    """
    return np.concatenate([u_nemo[:, -1:], u_nemo], axis=1)


def _nemo_v_to_legoesm(v_nemo):
    """NEMO ``v(jj, ji)`` onto legoESM's ``(n_lat + 1, n_lon)`` v-face array.

    legoESM ``v[j + 1, i] == NEMO v[jj=j, ji=i]``; row 0 is the closed southern
    wall and carries no velocity.
    """
    return np.concatenate([np.zeros_like(v_nemo[:1]), v_nemo], axis=0)


def production_ldf_tendency(deck_root: Path, root: Path, kt: int):
    """legoESM's PRODUCTION lateral-viscosity tendency for the ORCA2 card.

    Reached through ``LatLonCGridOceanModel.tendencies_with_diagnostics`` --
    the production momentum-tendency path, not a re-implementation of the
    operator -- on the record's own kt=1 step-entry state with Decision 52's
    sea-surface bridge.
    """
    import jax.numpy as jnp

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    _, card = ladder.card_fields(deck_root)
    config = card.recipe.model_config
    require(getattr(config, "lateral_viscosity_operator", None)
            == "nemo_div_curl",
            "ORCA2 card no longer selects the div-curl viscosity operator")
    require(getattr(config, "lateral_viscosity_e3_weighting", None)
            == "nemo_e3",
            "ORCA2 card no longer selects the e3-weighted div-curl")
    require(getattr(config, "lateral_viscosity_coefficient_source", None)
            == "nemo_ahm_3d_file",
            "ORCA2 card no longer reads the file coefficient")

    entry = read_entry_frame(root, kt)
    require(float(np.abs(entry["u"]).max()) > 0.0,
            f"the kt={kt} recorded entry velocity is identically zero; this "
            "operator would be scored zero against zero")
    state = card.recipe.initial_state
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(entry["T"], dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(entry["S"], dtype=jnp.float64)),
        u=state.u.replace(data=jnp.asarray(
            _nemo_u_to_legoesm(entry["u"]), dtype=jnp.float64)),
        v=state.v.replace(data=jnp.asarray(
            _nemo_v_to_legoesm(entry["v"]), dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))
    surface_fields = ladder.assemble_surface_fields(root, kt)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, kt)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, config,
        iwm_forcing=card.recipe.iwm_forcing,
    )
    _, diag = model.tendencies_with_diagnostics(
        state, surface_forcing=surface, dt=card.dt_s)
    # legoESM u[j, i+1] == NEMO u[jj=j, ji=i]; legoESM v[j+1, i] == NEMO v.
    du = np.asarray(diag.Ah_lap_u.data, dtype=np.float64)[:, 1:, :NZ]
    dv = np.asarray(diag.Ah_lap_v.data, dtype=np.float64)[1:, :, :NZ]
    require(du.shape == (NY_G, NX_G, NZ), f"candidate du shape {du.shape}")
    require(dv.shape == (NY_G, NX_G, NZ), f"candidate dv shape {dv.shape}")
    return du, dv, card, entry


def legoesm_min_rule_thicknesses(card, entry, mesh):
    """legoESM's OWN face/vertex thicknesses, on NEMO's index convention.

    The production operator builds these from the cell-centre live thickness
    by the min rule (``min_cell_to_uface`` / ``min_cell_to_vface`` /
    ``min_cell_to_vertex``); NEMO reads ``e3u_0``/``e3v_0``/``e3f_0`` from its
    own domain file and stretches each by its own ``r3``.  Ablation B feeds
    these to the compiled transcription to isolate that difference.
    """
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface, min_cell_to_vertex, min_cell_to_vface,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    grid = card.recipe.grid
    state = card.recipe.initial_state
    h_k = compute_layer_thickness(
        jnp.asarray(entry["ssh"], dtype=jnp.float64),
        state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    h_u = np.asarray(min_cell_to_uface(h_k), dtype=np.float64)[:, 1:, :NZ]
    h_v = np.asarray(min_cell_to_vface(h_k, grid), dtype=np.float64)[1:, :, :NZ]
    # vertex[j, i] == F[j-1, (i-1) % n_lon]  (round 9's proven index map),
    # so NEMO's F(jj, ji) is the vertex at (jj+1, ji+1).
    h_f = np.asarray(min_cell_to_vertex(h_k, grid),
                     dtype=np.float64)[1:, 1:, :NZ]
    h_t = np.asarray(h_k, dtype=np.float64)[:, :, :NZ]
    require(h_u.shape == (NY_G, NX_G, NZ), f"h_u shape {h_u.shape}")
    require(h_v.shape == (NY_G, NX_G, NZ), f"h_v shape {h_v.shape}")
    require(h_f.shape == (NY_G, NX_G, NZ), f"h_f shape {h_f.shape}")
    require(h_t.shape == (NY_G, NX_G, NZ), f"h_t shape {h_t.shape}")
    return h_u, h_v, h_f, h_t


def legoesm_metrics_on_nemo_index(card):
    """legoESM's OWN stored horizontal metrics, on NEMO's index convention.

    The card's grid stores each metric on legoESM's face layout; the map is
    the one round 9 proved and round 11 reuses -- legoESM ``u[j, i+1]`` is
    NEMO ``u(ji=i, jj=j)``, legoESM ``v[j+1, i]`` is NEMO ``v(ji=i, jj=j)``,
    legoESM ``vertex[j, i]`` is NEMO ``F(ji=i-1, jj=j-1)``, and legoESM cell
    ``(j, i)`` is NEMO ``T(ji=i, jj=j)``.  Each entry names the role the
    compiled statements read it in, NOT merely a similarly-shaped array.
    """
    grid = card.recipe.grid
    out = {
        "e1t_e2t": np.asarray(grid.area, dtype=np.float64),
        "e1f_e2f": np.asarray(grid.area_q, dtype=np.float64)[1:, 1:],
        "e1u": np.asarray(grid.dx_u, dtype=np.float64)[:, 1:],
        "e2u": np.asarray(grid.dy_u, dtype=np.float64)[:, 1:],
        "e1v": np.asarray(grid.dx_v, dtype=np.float64)[1:, :],
        "e2v": np.asarray(grid.dy_v, dtype=np.float64)[1:, :],
    }
    for name, value in out.items():
        require(value.shape == (NY_G, NX_G),
                f"legoESM metric {name} has shape {value.shape}")
    return out


def run_gate(deck_root: Path, root: Path, json_out: Path | None,
             plant: bool = False, kt: int = 2) -> dict[str, object]:
    stamp = worktree_stamp()
    mesh = _stitch(root, "mesh_mask_{rank:04d}.nc",
                   ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
                    "e3t_0", "e3u_0", "e3v_0", "e3f_0",
                    "tmask", "umask", "vmask"))
    coeff = _stitch(root, "output.init_{rank:04d}.nc", ("ahmt", "ahmf"))
    ahmt, ahmf = coeff["ahmt"], coeff["ahmf"]
    require(ahmt.shape == (NY_G, NX_G, NZ), f"ahmt shape {ahmt.shape}")

    du_prod, dv_prod, card, entry = production_ldf_tendency(
        deck_root, root, kt)
    if plant:
        du_prod = _plant_one_value(du_prod)

    du_ref, dv_ref = nemo_dynldf_lev_lap_rot(
        mesh, ahmt, ahmf, entry["u"], entry["v"], entry["ssh"])

    # ONE cell set for every row: wet, and defined by the transcription.  The
    # transcription leaves the rows whose latitude neighbours fall outside the
    # owned block undefined, and an ablation can make a cell defined that the
    # reference row could not score -- so without this the closure row would
    # be scored on 1,319 more v cells than the row it is compared against.
    weight_u = mesh["umask"] * np.isfinite(du_ref)
    weight_v = mesh["vmask"] * np.isfinite(dv_ref)
    rows: dict[str, object] = {
        "scored_cell_set": (
            "wet faces where the compiled transcription is defined; the SAME "
            "set for every row below"),
        "u_momentum": score(du_prod, du_ref, weight_u),
        "v_momentum": score(dv_prod, dv_ref, weight_v),
    }

    # --- ablation A: legoESM's extra zero/one vertex mask on ahmf ----------
    # The PRODUCTION mask, built the way ocean_pe_latlon_cgrid.py:3417-3428
    # builds it: a per-level staircase vertex mask from ``is_active``, times
    # the surface vertex mask.  Using the 2-D surface mask instead would
    # understate this ablation.
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_vertex_mask,
    )
    grid = card.recipe.grid
    cell_mask = jnp.asarray(card.recipe.initial_state.land_mask.data)
    surface_vtx = compute_vertex_mask(cell_mask, grid=grid)
    active = getattr(card.recipe.z_coord, "is_active", None)
    require(active is not None, "the ORCA2 card carries no is_active mask")
    cell3 = jnp.asarray(active).astype(jnp.float64) * cell_mask[..., jnp.newaxis]
    vtx3 = jax.vmap(lambda m2: compute_vertex_mask(m2, grid=grid),
                    in_axes=-1, out_axes=-1)(cell3) * surface_vtx[..., jnp.newaxis]
    nemo_vertex_mask = np.asarray(vtx3, dtype=np.float64)[1:, 1:, :NZ]
    du_maskA, dv_maskA = nemo_dynldf_lev_lap_rot(
        mesh, ahmt, ahmf, entry["u"], entry["v"], entry["ssh"],
        ahmf_extra_mask=nemo_vertex_mask)
    rows["ablation_extra_vertex_mask_u"] = score(
        du_maskA, du_ref, weight_u)
    rows["ablation_extra_vertex_mask_v"] = score(
        dv_maskA, dv_ref, weight_v)
    rows["coefficient_cells_zeroed_by_the_extra_mask"] = int(
        ((ahmf != 0.0) & (nemo_vertex_mask == 0.0)).sum())

    # --- ablation B: legoESM's min-rule thicknesses ------------------------
    h_u, h_v, h_f, h_t = legoesm_min_rule_thicknesses(card, entry, mesh)
    du_thick, dv_thick = nemo_dynldf_lev_lap_rot(
        mesh, ahmt, ahmf, entry["u"], entry["v"], entry["ssh"],
        thickness_override=(h_u, h_v, h_f))
    rows["ablation_min_rule_thickness_u"] = score(
        du_thick, du_ref, weight_u)
    rows["ablation_min_rule_thickness_v"] = score(
        dv_thick, dv_ref, weight_v)
    rows["thickness_operand_disagreement"] = {
        "note": ("legoESM's LIVE min-rule face/vertex thickness against the "
                 "record's own REFERENCE e3u_0/e3v_0/e3f_0; the stretch is "
                 "about 1e-5, so a difference here is a rule difference, not "
                 "a free-surface one"),
        "e3u_max_abs": float(np.max(np.abs(h_u - mesh["e3u_0"]))),
        "e3v_max_abs": float(np.max(np.abs(h_v - mesh["e3v_0"]))),
        "e3f_max_abs": float(np.max(np.abs(h_f - mesh["e3f_0"]))),
        "e3u_unequal": int(np.count_nonzero(h_u != mesh["e3u_0"])),
        "e3v_unequal": int(np.count_nonzero(h_v != mesh["e3v_0"])),
        "e3f_unequal": int(np.count_nonzero(h_f != mesh["e3f_0"])),
        "cells": int(mesh["e3u_0"].size),
    }

    # --- CLOSURE: both ablations together ----------------------------------
    # This is the instrument's own validation.  If applying EXACTLY the two
    # statements legoESM differs by (the second vertex mask and its min-rule
    # live thicknesses) to the compiled transcription reproduces the
    # production output, then the transcription, the index map and the
    # attribution are all certified together; a residual is a THIRD,
    # unattributed difference and is reported as such rather than ignored.
    du_both, dv_both = nemo_dynldf_lev_lap_rot(
        mesh, ahmt, ahmf, entry["u"], entry["v"], entry["ssh"],
        ahmf_extra_mask=nemo_vertex_mask,
        thickness_override=(h_u, h_v, h_f))
    rows["closure_both_ablations_u"] = score(du_prod, du_both, weight_u)
    rows["closure_both_ablations_v"] = score(dv_prod, dv_both, weight_v)

    # --- ROUND 12: the THIRD difference, named -----------------------------
    # Each row below is the closure ABOVE plus exactly ONE further legoESM
    # behaviour.  Scored against the production operator on the SAME cell set,
    # so a row that falls below the closure has explained part of what the
    # closure left, and a row that does not move is VACUOUS and is reported as
    # vacuous rather than as agreement.
    common = dict(ahmf_extra_mask=nemo_vertex_mask,
                  thickness_override=(h_u, h_v, h_f))
    variants = {
        "third_e3t_divisor": dict(common, e3t_override=h_t),
        "third_curl_zonal_edge": dict(common, curl_zonal_edge_from_cell=True),
        "third_stored_metrics": dict(
            common, metric_override=legoesm_metrics_on_nemo_index(card)),
    }
    for name, kwargs in variants.items():
        du_x, dv_x = nemo_dynldf_lev_lap_rot(
            mesh, ahmt, ahmf, entry["u"], entry["v"], entry["ssh"], **kwargs)
        rows[f"{name}_u"] = score(du_prod, du_x, weight_u)
        rows[f"{name}_v"] = score(dv_prod, dv_x, weight_v)
        # A substitution that moves NOTHING is vacuous, and says so.
        rows[f"{name}_moves_the_oracle"] = bool(
            not np.array_equal(np.nan_to_num(du_x), np.nan_to_num(du_both))
            or not np.array_equal(np.nan_to_num(dv_x),
                                  np.nan_to_num(dv_both)))

    # The operands themselves, so "vacuous" is a MEASURED equality and not an
    # inference from two tendencies happening to agree.
    _r3t_ref = entry["ssh"] * (np.max(mesh["tmask"], axis=-1)
                               / (np.sum(mesh["e3t_0"] * mesh["tmask"], axis=-1)
                                  + 1.0 - np.max(mesh["tmask"], axis=-1)))
    _e3t_nemo = mesh["e3t_0"] * (1.0 + _r3t_ref[..., None] * mesh["tmask"])
    rows["e3t_operand_disagreement"] = {
        "note": ("legoESM's live cell thickness against NEMO's own "
                 "e3t_0*(1+r3t(Kbb)*tmask), the divisor at "
                 "dynldf_lev.f90:127"),
        "cells": int(_e3t_nemo.size),
        "unequal": int(np.count_nonzero(h_t != _e3t_nemo)),
        "max_abs": float(np.max(np.abs(h_t - _e3t_nemo))),
    }
    _lego_metrics = legoesm_metrics_on_nemo_index(card)
    _nemo_metrics = {
        "e1t_e2t": mesh["e1t"] * mesh["e2t"],
        "e1f_e2f": mesh["e1f"] * mesh["e2f"],
        "e1u": mesh["e1u"], "e2u": mesh["e2u"],
        "e1v": mesh["e1v"], "e2v": mesh["e2v"],
    }
    rows["metric_operand_disagreement"] = {
        name: {
            "cells": int(_nemo_metrics[name].size),
            "unequal": int(np.count_nonzero(
                _lego_metrics[name] != _nemo_metrics[name])),
            "max_rel": float(np.max(
                np.abs(_lego_metrics[name] - _nemo_metrics[name])
                / np.maximum(np.abs(_nemo_metrics[name]), 1e-300))),
        }
        for name in _nemo_metrics
    }

    # Where the one differing metric differs, since "unequal almost
    # everywhere" and "wrong almost everywhere" are different statements.
    _rel_f = (np.abs(_lego_metrics["e1f_e2f"] - _nemo_metrics["e1f_e2f"])
              / np.maximum(np.abs(_nemo_metrics["e1f_e2f"]), 1e-300))
    _wet_f = np.max(mesh["tmask"], axis=-1) > 0.0
    _arg = np.unravel_index(int(np.argmax(_rel_f)), _rel_f.shape)
    rows["vertex_area_relative_difference"] = {
        "note": ("legoESM's stored vertex area against the record's own "
                 "e1f*e2f; the vorticity bracket divides by it"),
        "cells": int(_rel_f.size),
        "above_1e-6": int(np.count_nonzero(_rel_f > 1e-6)),
        "above_1e-3": int(np.count_nonzero(_rel_f > 1e-3)),
        "above_1e-1": int(np.count_nonzero(_rel_f > 1e-1)),
        "above_1e-3_on_a_wet_column": int(
            np.count_nonzero((_rel_f > 1e-3) & _wet_f)),
        "median": float(np.median(_rel_f)),
        "argmax_row_col": [int(_arg[0]), int(_arg[1])],
        "argmax_is_on_a_wet_column": bool(_wet_f[_arg]),
    }

    du_all, dv_all = nemo_dynldf_lev_lap_rot(
        mesh, ahmt, ahmf, entry["u"], entry["v"], entry["ssh"],
        ahmf_extra_mask=nemo_vertex_mask,
        thickness_override=(h_u, h_v, h_f),
        e3t_override=h_t,
        curl_zonal_edge_from_cell=True,
        metric_override=legoesm_metrics_on_nemo_index(card))
    rows["closure_all_five_u"] = score(du_prod, du_all, weight_u)
    rows["closure_all_five_v"] = score(dv_prod, dv_all, weight_v)

    # The fourth candidate round 11 named -- the slope-foot factor the
    # production path applies to the operator's output
    # (ocean_pe_latlon_cgrid.py:3221-3223) -- is settled by reading the value
    # it multiplies by, not by a substitution: at alpha = 0 both factors are
    # the scalar 1.0 and no array exists to substitute.
    _alpha = float(getattr(card.recipe.model_config, "slope_foot_alpha", 0.0))
    rows["slope_foot"] = {
        "alpha": _alpha,
        "factor_is_the_scalar_one": _alpha <= 0.0,
        "verdict": ("VACUOUS -- the factor is the scalar 1.0, so the "
                    "production operator's output is returned unmultiplied"
                    if _alpha <= 0.0 else
                    "LIVE -- a 3-D factor multiplies the operator's output"),
    }
    # NEMO's stage-three Kbb/Kmm split is EXCLUDED a priori, not measured:
    # this gate drives both sides with ONE recorded sea surface, so no such
    # difference can exist inside it (preregistration, statement A).
    rows["kbb_kmm_split"] = (
        "EXCLUDED BY CONSTRUCTION -- one sea surface drives both sides")

    result = {
        "gate": "nemo_testcase_l4_orca2_round11_dynldf_operator_gate",
        "kt": kt,
        "entry_velocity_max_abs": float(np.abs(entry["u"]).max()),
        "label": f"given NEMO's entry (kt={kt} recorded state)",
        "citations": CITATIONS,
        "controlled_comparison": (
            "both sides driven by the record's kt=1 step-entry sea surface; "
            "NEMO's stage-three Kbb/Kmm split is a separate, unscored "
            "statement"),
        "provenance": stamp,
        "rows": rows,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--kt", type=int, default=2,
                        help="which recorded step entry to score on; kt=1 is "
                             "rest and would be scored zero against zero")
    parser.add_argument("--plant", action="store_true",
                        help="move the production output by one representable "
                             "value; the gate must refuse")
    args = parser.parse_args()
    try:
        result = run_gate(args.deck_root, args.record_root, args.json_out,
                          plant=args.plant, kt=args.kt)
    except (GateError, OSError, ValueError, struct.error) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    rows = result["rows"]
    at_bar = (rows["u_momentum"]["bit_identical"]
              and rows["v_momentum"]["bit_identical"])
    if args.plant:
        if at_bar:
            print("REFUSE: the plant did NOT fire -- the gate cannot bind",
                  file=sys.stderr)
            return 3
        print("PLANT FIRED: the gate refuses a one-representable-value move",
              file=sys.stderr)
        return 1
    if not at_bar:
        print("REFUSE: the production lateral-viscosity operator is not the "
              "compiled dynldf_lev_lap statement", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
