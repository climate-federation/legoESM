#!/usr/bin/env python3
"""Committed validation, layout-fixture, log-check, and admission gate for SI3 Rung O1 (`si3_r4`).

Covers (`DECISION 159`, `docs/ice/fidelity/si3_r4_o1_preregister.md`):
- Binary snapshot reading and multi-rank MPI subdomain stitching (`O1SI3R01`).
- Compiled `gfortran` 4-rank layout fixture verification (`--verify-layout-fixture`)
  with distinct rank-dependent and field-dependent payloads and negative-control plants.
- Whitespace-tolerant `ocean.output` log verification (`--verify-log`) with plants.
- Full ORCA2 `(184, 152)` `jpl = 5`, `nlay_i = 10`, `nlay_s = 5` state, HFN category
  bounds, `si3_var_glo2eqv`, `si3_var_agg`, `si3_lbc_lnk_orca2`, `si3_o1_step`, and
  NetCDF restart (`read_si3_netcdf_restart` / `write_si3_netcdf_restart`) verification
  at `0` ULP exact bar (`max|Delta| <= 1e-15` trajectory bar) with plants P1-P7.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.si3_orca2_state import (
    SI3AggregatedVars,
    SI3AllPratherMoments,
    SI3CategoryBounds,
    SI3EquivalentVars,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    SI3State,
    read_si3_netcdf_restart,
    si3_apply_state_halos_orca2,
    si3_hfn_category_bounds,
    si3_lbc_lnk_orca2,
    si3_o1_step,
    si3_rst_to_netcdf_dict,
    si3_update_snwice_mass,
    si3_var_agg,
    si3_var_glo2eqv,
    write_si3_netcdf_restart,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round326_hmi_gate import (
    objects_equal,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import (
    BUILD_CONFIG_FILES,
    GateError as SameProgramError,
    same_program_observers,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def read_o1_binary_snapshot(path: str | Path) -> dict[str, Any]:
    """Read one rank-local `o1_si3_state_*.bin` file written by `o1_si3_state_writer.F90`."""
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) < 228:
        raise ValueError(f"Truncated O1 binary header in {p}: {len(raw)} bytes < 228")

    magic = raw[:8].decode("ascii", errors="replace")
    if magic != "O1SI3R01":
        raise ValueError(f"Invalid O1 binary magic {magic!r} in {p}, expected 'O1SI3R01'")
    version = int(np.frombuffer(raw, dtype="<i4", count=1, offset=8)[0])
    if version != 1:
        raise ValueError(f"Unsupported O1 binary version {version} in {p}")

    i_meta = np.frombuffer(raw, dtype="<i4", count=26, offset=12)
    r_meta = np.frombuffer(raw, dtype="<f8", count=3, offset=116)
    jpi = int(i_meta[3])
    jpj = int(i_meta[4])
    jpl = int(i_meta[5])
    nlay_i = int(i_meta[6])
    nlay_s = int(i_meta[7])

    offset = 140
    hi_max = np.frombuffer(raw, dtype="<f8", count=jpl + 1, offset=offset).copy()
    offset += (jpl + 1) * 8
    hi_mean = np.frombuffer(raw, dtype="<f8", count=jpl, offset=offset).copy()
    offset += jpl * 8

    meta = {
        "kt": int(i_meta[0]),
        "mpprank": int(i_meta[1]),
        "narea": int(i_meta[2]),
        "jpi": jpi,
        "jpj": jpj,
        "jpl": jpl,
        "nlay_i": nlay_i,
        "nlay_s": nlay_s,
        "nn_hls": int(i_meta[8]),
        "Ni0glo": int(i_meta[9]),
        "Nj0glo": int(i_meta[10]),
        "nimpp": int(i_meta[11]),
        "njmpp": int(i_meta[12]),
        "Nis0": int(i_meta[13]),
        "Nie0": int(i_meta[14]),
        "Njs0": int(i_meta[15]),
        "Nje0": int(i_meta[16]),
        "nn_fsbc": int(i_meta[17]),
        "nn_icesal": int(i_meta[18]),
        "nn_snwfra": int(i_meta[19]),
        "ln_pnd": int(i_meta[20]),
        "ln_pnd_LEV": int(i_meta[21]),
        "ln_pnd_TOPO": int(i_meta[22]),
        "ln_pnd_lids": int(i_meta[23]),
        "ln_cpl": int(i_meta[24]),
        "lrst_ice": int(i_meta[25]),
        "rn_Dt": float(r_meta[0]),
        "rDt_ice": float(r_meta[1]),
        "rn_himean": float(r_meta[2]),
    }

    fields: dict[str, np.ndarray] = {}
    for fname, _foff, ndim, nlay, _is_a2d in _FIXTURE_FIELD_OFFSETS:
        if ndim == 2:
            shape = (jpi, jpj)
        elif ndim == 3:
            shape = (jpi, jpj, jpl)
        elif ndim == 4:
            shape = (jpi, jpj, nlay, jpl)
        else:
            raise ValueError(f"Unsupported ndim={ndim}")
        count = int(np.prod(shape))
        nbytes = count * 8
        if offset + nbytes > len(raw):
            raise ValueError(
                f"Truncated O1 binary payload at field {fname} in {p}: need {offset + nbytes}, got {len(raw)}"
            )
        arr = (
            np.frombuffer(raw, dtype="<f8", count=count, offset=offset)
            .reshape(shape, order="F")
            .copy()
        )
        fields[fname] = arr
        offset += nbytes

    if offset != len(raw):
        raise ValueError(f"Trailing bytes in {p}: consumed {offset}, file size {len(raw)}")

    return {
        "magic": magic,
        "version": version,
        "meta": meta,
        "hi_max": hi_max,
        "hi_mean": hi_mean,
        "fields": fields,
    }


def stitch_o1_binary_snapshots(snaps: list[dict[str, Any]]) -> dict[str, Any]:
    """Stitch rank-local O1 binary snapshots onto the global `(Ni0glo + 2*nn_hls, Nj0glo + 2*nn_hls)` grid."""
    if not snaps:
        raise ValueError("Cannot stitch empty snapshot list")
    if len(snaps) == 1:
        return copy.deepcopy(snaps[0])

    meta0 = copy.deepcopy(snaps[0]["meta"])
    nn_hls = int(meta0["nn_hls"])
    ni0glo = int(meta0["Ni0glo"])
    nj0glo = int(meta0["Nj0glo"])
    g_jpi = ni0glo + 2 * nn_hls
    g_jpj = nj0glo + 2 * nn_hls
    jpl = int(meta0["jpl"])

    stitched_fields: dict[str, np.ndarray] = {}
    for fname, _foff, ndim, nlay, _is_a2d in _FIXTURE_FIELD_OFFSETS:
        if ndim == 2:
            shape = (g_jpi, g_jpj)
        elif ndim == 3:
            shape = (g_jpi, g_jpj, jpl)
        else:
            shape = (g_jpi, g_jpj, nlay, jpl)
        stitched_fields[fname] = np.zeros(shape, dtype=np.float64)

    for snap in snaps:
        m = snap["meta"]
        nis0, nie0 = int(m["Nis0"]), int(m["Nie0"])
        njs0, nje0 = int(m["Njs0"]), int(m["Nje0"])
        gi0 = int(m["nimpp"]) + nis0 - 2
        gi1 = int(m["nimpp"]) + nie0 - 1
        gj0 = int(m["njmpp"]) + njs0 - 2
        gj1 = int(m["njmpp"]) + nje0 - 1
        if gi0 < 0 or gj0 < 0 or gi1 > g_jpi or gj1 > g_jpj:
            raise ValueError(
                f"Subdomain slice [{gi0}:{gi1}, {gj0}:{gj1}] out of global bounds ({g_jpi}, {g_jpj})"
            )
        for fname in stitched_fields:
            stitched_fields[fname][gi0:gi1, gj0:gj1] = snap["fields"][fname][
                nis0 - 1 : nie0, njs0 - 1 : nje0
            ]

    meta0["jpi"] = g_jpi
    meta0["jpj"] = g_jpj
    meta0["nimpp"] = 1
    meta0["njmpp"] = 1
    meta0["Nis0"] = 1 + nn_hls
    meta0["Nie0"] = g_jpi - nn_hls
    meta0["Njs0"] = 1 + nn_hls
    meta0["Nje0"] = g_jpj - nn_hls
    return {
        "magic": snaps[0]["magic"],
        "version": snaps[0]["version"],
        "meta": meta0,
        "hi_max": snaps[0]["hi_max"].copy(),
        "hi_mean": snaps[0]["hi_mean"].copy(),
        "fields": stitched_fields,
    }


def o1_snapshot_to_si3_state(
    snap: dict[str, Any],
    config: SI3ORCA2StateConfig | None = None,
) -> tuple[SI3State, SI3ORCA2StateConfig]:
    """Convert a stitched O1 binary snapshot dict into `SI3State` and `SI3ORCA2StateConfig`."""
    m = snap["meta"]
    f = snap["fields"]
    cfg = config or SI3ORCA2StateConfig(
        jpl=int(m["jpl"]),
        nlay_i=int(m["nlay_i"]),
        nlay_s=int(m["nlay_s"]),
        nn_icesal=int(m["nn_icesal"]),
        rn_himean=float(m["rn_himean"]),
        ln_pnd=bool(m["ln_pnd"]),
        ln_pnd_LEV=bool(m["ln_pnd_LEV"]),
        ln_pnd_TOPO=bool(m["ln_pnd_TOPO"]),
        ln_pnd_lids=bool(m["ln_pnd_lids"]),
        nn_snwfra=int(m["nn_snwfra"]),
        nn_fsbc=int(m["nn_fsbc"]),
        rDt_ice=float(m["rDt_ice"]),
        nn_hls=int(m["nn_hls"]),
        ln_cpl=bool(m["ln_cpl"]),
    )

    def _fam(suffix: str) -> SI3PratherMoments:
        return SI3PratherMoments(
            sx=jnp.asarray(f[f"sx{suffix}"], dtype=jnp.float64),
            sy=jnp.asarray(f[f"sy{suffix}"], dtype=jnp.float64),
            sxx=jnp.asarray(f[f"sxx{suffix}"], dtype=jnp.float64),
            syy=jnp.asarray(f[f"syy{suffix}"], dtype=jnp.float64),
            sxy=jnp.asarray(f[f"sxy{suffix}"], dtype=jnp.float64),
        )

    prather = SI3AllPratherMoments(
        ice=_fam("ice"),
        sn=_fam("sn"),
        a=_fam("a"),
        age=_fam("age"),
        sal=_fam("sal"),
        ap=_fam("ap"),
        vp=_fam("vp"),
        vl=_fam("vl"),
        c0=_fam("c0"),
        e=_fam("e"),
        si=_fam("si"),
    )
    bounds = SI3CategoryBounds(
        hi_max=jnp.asarray(snap["hi_max"], dtype=jnp.float64),
        hi_mean=jnp.asarray(snap["hi_mean"], dtype=jnp.float64),
    )
    eqv = SI3EquivalentVars(
        h_i=jnp.asarray(f["h_i"], dtype=jnp.float64),
        h_s=jnp.asarray(f["h_s"], dtype=jnp.float64),
        o_i=jnp.asarray(f["o_i"], dtype=jnp.float64),
        h_ip=jnp.asarray(f["h_ip"], dtype=jnp.float64),
        h_il=jnp.asarray(f["h_il"], dtype=jnp.float64),
        a_ip_frac=jnp.asarray(f["a_ip_frac"], dtype=jnp.float64),
        a_ip_eff=jnp.asarray(f["a_ip_eff"], dtype=jnp.float64),
        s_i=jnp.asarray(f["s_i"], dtype=jnp.float64),
        sz_i=jnp.asarray(f["sz_i"], dtype=jnp.float64),
        t_i=jnp.asarray(f["t_i"], dtype=jnp.float64),
        t_s=jnp.asarray(f["t_s"], dtype=jnp.float64),
        a_i=jnp.asarray(f["a_i"], dtype=jnp.float64),
        v_s=jnp.asarray(f["v_s"], dtype=jnp.float64),
    )
    agg = SI3AggregatedVars(
        vt_i=jnp.asarray(f["vt_i"], dtype=jnp.float64),
        vt_s=jnp.asarray(f["vt_s"], dtype=jnp.float64),
        at_i=jnp.asarray(f["at_i"], dtype=jnp.float64),
        at_ip=jnp.asarray(f["at_ip"], dtype=jnp.float64),
        vt_ip=jnp.asarray(f["vt_ip"], dtype=jnp.float64),
        vt_il=jnp.asarray(f["vt_il"], dtype=jnp.float64),
        ato_i=jnp.asarray(f["ato_i"], dtype=jnp.float64),
        et_s=jnp.asarray(f["et_s"], dtype=jnp.float64),
        et_i=jnp.asarray(f["et_i"], dtype=jnp.float64),
        at_ip_eff=jnp.asarray(f["at_ip_eff"], dtype=jnp.float64),
        tm_su=jnp.asarray(f["tm_su"], dtype=jnp.float64),
        st_i=jnp.asarray(f["st_i"], dtype=jnp.float64),
        hm_i=jnp.asarray(f["hm_i"], dtype=jnp.float64),
        hm_s=jnp.asarray(f["hm_s"], dtype=jnp.float64),
        tm_si=jnp.asarray(f["tm_si"], dtype=jnp.float64),
        om_i=jnp.asarray(f["om_i"], dtype=jnp.float64),
        sm_i=jnp.asarray(f["sm_i"], dtype=jnp.float64),
        tm_i=jnp.asarray(f["tm_i"], dtype=jnp.float64),
        tm_s=jnp.asarray(f["tm_s"], dtype=jnp.float64),
        hm_ip=jnp.asarray(f["hm_ip"], dtype=jnp.float64),
        hm_il=jnp.asarray(f["hm_il"], dtype=jnp.float64),
    )
    state = SI3State(
        v_i=jnp.asarray(f["v_i"], dtype=jnp.float64),
        v_s=jnp.asarray(f["v_s"], dtype=jnp.float64),
        sv_i=jnp.asarray(f["sv_i"], dtype=jnp.float64),
        a_i=jnp.asarray(f["a_i"], dtype=jnp.float64),
        t_su=jnp.asarray(f["t_su"], dtype=jnp.float64),
        u_ice=jnp.asarray(f["u_ice"], dtype=jnp.float64),
        v_ice=jnp.asarray(f["v_ice"], dtype=jnp.float64),
        oa_i=jnp.asarray(f["oa_i"], dtype=jnp.float64),
        a_ip=jnp.asarray(f["a_ip"], dtype=jnp.float64),
        v_ip=jnp.asarray(f["v_ip"], dtype=jnp.float64),
        v_il=jnp.asarray(f["v_il"], dtype=jnp.float64),
        e_s=jnp.asarray(f["e_s"], dtype=jnp.float64),
        e_i=jnp.asarray(f["e_i"], dtype=jnp.float64),
        szv_i=jnp.asarray(f["szv_i"], dtype=jnp.float64),
        cnd_ice=jnp.asarray(f["cnd_ice"], dtype=jnp.float64),
        t1_ice=jnp.asarray(f["t1_ice"], dtype=jnp.float64),
        t_si=jnp.asarray(f["t_si"], dtype=jnp.float64),
        snwice_mass=jnp.asarray(f["snwice_mass"], dtype=jnp.float64),
        snwice_mass_b=jnp.asarray(f["snwice_mass_b"], dtype=jnp.float64),
        snwice_fmass=jnp.asarray(f["snwice_fmass"], dtype=jnp.float64),
        prather=prather,
        bounds=bounds,
        equivalent=eqv,
        aggregated=agg,
    )
    return state, cfg


def max_ulp_diff(
    a: np.ndarray | jax.Array,
    b: np.ndarray | jax.Array,
    mask: np.ndarray | jax.Array | None = None,
    *,
    strict_signed_zero: bool = False,
) -> int:
    """Compute maximum IEEE-754 float64 ULP distance between two arrays (optionally wet-masked)."""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if a_np.shape != b_np.shape:
        raise ValueError(f"Shape mismatch: {a_np.shape} vs {b_np.shape}")
    if mask is not None:
        m_np = np.asarray(mask, dtype=np.float64) > 0.0
        while m_np.ndim < a_np.ndim:
            m_np = m_np[..., None]
        wet = np.broadcast_to(m_np, a_np.shape)
        if int(np.count_nonzero(wet)) == 0:
            raise ValueError("STATUS PLANT-FIRED: wet mask has zero wet cells")
        a_np = a_np[wet]
        b_np = b_np[wet]
    if not (np.all(np.isfinite(a_np)) and np.all(np.isfinite(b_np))):
        return 2**62

    if strict_signed_zero:
        sign_diff = (a_np == 0.0) & (b_np == 0.0) & (np.signbit(a_np) != np.signbit(b_np))
    else:
        sign_diff = np.zeros(a_np.shape, dtype=bool)

    def _to_ordered_int64(x: np.ndarray) -> np.ndarray:
        x_clean = np.where(x == 0.0, 0.0, x)
        bits = x_clean.view(np.int64)
        return np.where(bits < 0, np.int64(-9223372036854775808) - bits, bits)

    ia = _to_ordered_int64(a_np)
    ib = _to_ordered_int64(b_np)
    diff = np.abs(ia - ib)
    if np.any(sign_diff):
        diff = np.maximum(diff, sign_diff.astype(np.int64))
    return int(np.max(diff)) if diff.size > 0 else 0


def max_normalized_diff(
    a: np.ndarray | jax.Array,
    b: np.ndarray | jax.Array,
    mask: np.ndarray | jax.Array | None = None,
) -> float:
    """Compute max(|a - b| / max(|b|, 1.0)) over wet cells."""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if a_np.shape != b_np.shape:
        raise ValueError(f"Shape mismatch: {a_np.shape} vs {b_np.shape}")
    if mask is not None:
        m_np = np.asarray(mask, dtype=np.float64) > 0.0
        while m_np.ndim < a_np.ndim:
            m_np = m_np[..., None]
        wet = np.broadcast_to(m_np, a_np.shape)
        if int(np.count_nonzero(wet)) == 0:
            raise ValueError("STATUS PLANT-FIRED: wet mask has zero wet cells")
        a_np = a_np[wet]
        b_np = b_np[wet]
    if not (np.all(np.isfinite(a_np)) and np.all(np.isfinite(b_np))):
        return float("inf")
    denom = np.maximum(np.abs(b_np), 1.0)
    return float(np.max(np.abs(a_np - b_np) / denom)) if a_np.size > 0 else 0.0


def first_wet_ij(
    mask_2d: np.ndarray | jax.Array,
    *,
    interior_only: bool = True,
    nn_hls: int = 2,
) -> tuple[int, int]:
    """Return the first wet cell (i, j) and verify wet_index.size > 0."""
    m = np.asarray(mask_2d, dtype=np.float64) > 0.0
    if interior_only and nn_hls > 0:
        interior = np.zeros_like(m, dtype=bool)
        interior[nn_hls:-nn_hls, nn_hls:-nn_hls] = True
        m = m & interior
    wet_idx = np.argwhere(m)
    if wet_idx.shape[0] == 0:
        raise ValueError("STATUS PLANT-FIRED: wet_index.size == 0 (no wet cells found)")
    wi, wj = int(wet_idx[0, 0]), int(wet_idx[0, 1])
    if float(np.asarray(mask_2d)[wi, wj]) <= 0.0:
        raise ValueError(f"STATUS PLANT-FIRED: cell ({wi}, {wj}) is not wet")
    return wi, wj


def first_dry_ij(
    mask_2d: np.ndarray | jax.Array,
    *,
    interior_only: bool = True,
    nn_hls: int = 2,
) -> tuple[int, int]:
    """Return the first dry cell (i, j) where mask == 0."""
    m = np.asarray(mask_2d, dtype=np.float64) == 0.0
    if interior_only and nn_hls > 0:
        interior = np.zeros_like(m, dtype=bool)
        interior[nn_hls:-nn_hls, nn_hls:-nn_hls] = True
        m = m & interior
    dry_idx = np.argwhere(m)
    if dry_idx.shape[0] == 0:
        dry_idx = np.argwhere(np.asarray(mask_2d, dtype=np.float64) == 0.0)
    if dry_idx.shape[0] == 0:
        raise ValueError("STATUS PLANT-FIRED: dry_index.size == 0 (no dry cells found)")
    return int(dry_idx[0, 0]), int(dry_idx[0, 1])


def restart_payload_view(
    expected: np.ndarray | jax.Array,
    payload: np.ndarray | jax.Array,
    mask: np.ndarray | jax.Array | None,
    *,
    nn_hls: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Align a full ORCA2 field/mask with NEMO's halo-free restart payload."""
    exp = np.asarray(expected)
    raw = np.asarray(payload)
    scored_mask = None if mask is None else np.asarray(mask)
    if raw.ndim >= 2 and exp.shape[:2] == (
        raw.shape[0] + 2 * nn_hls,
        raw.shape[1] + 2 * nn_hls,
    ):
        exp = exp[nn_hls:-nn_hls, nn_hls:-nn_hls, ...]
        if scored_mask is not None:
            scored_mask = scored_mask[nn_hls:-nn_hls, nn_hls:-nn_hls, ...]
    elif raw.ndim >= 2 and exp.shape[:2] != raw.shape[:2]:
        raise ValueError(
            f"STATUS PLANT-FIRED: restart payload shape {raw.shape} is not "
            f"full {exp.shape} or halo-free by nn_hls={nn_hls}"
        )
    return exp, scored_mask


def wet_mask_for_field(
    fname: str,
    tmask: np.ndarray | jax.Array,
    umask: np.ndarray | jax.Array | None = None,
    vmask: np.ndarray | jax.Array | None = None,
    *,
    nn_hls: int = 2,
    score_defined_halos: bool | None = None,
) -> np.ndarray:
    """Return wet cells on which NEMO defines the named field.

    ``iom_get(..., jpdom_auto)`` applies ``lbc_lnk(..., ldfull=.TRUE.)`` to
    restart-read prognostics and Prather moments, so their cyclic-seam and
    north-fold halo cells are defined and scored.  Callers also score the
    equivalent and aggregate fields that NEMO computes over
    ``DO_2D(nn_hls, nn_hls, nn_hls, nn_hls)``; reduced ``A2D(0)`` diagnostics
    retain undefined halos and are scored only on their computed interior.
    """
    if fname == "u_ice" and umask is not None:
        mask = np.asarray(umask, dtype=np.float64).copy()
    elif fname == "v_ice" and vmask is not None:
        mask = np.asarray(vmask, dtype=np.float64).copy()
    else:
        mask = np.asarray(tmask, dtype=np.float64).copy()
    if score_defined_halos is None:
        score_defined_halos = fname in {
            "v_i",
            "v_s",
            "sv_i",
            "a_i",
            "t_su",
            "u_ice",
            "v_ice",
            "oa_i",
            "a_ip",
            "v_ip",
            "v_il",
            "e_s",
            "e_i",
            "szv_i",
            "snwice_mass",
            "snwice_mass_b",
        }
    if nn_hls and not score_defined_halos:
        mask[:nn_hls, :] = 0.0
        mask[-nn_hls:, :] = 0.0
        mask[:, :nn_hls] = 0.0
        mask[:, -nn_hls:] = 0.0
    return mask


def verify_dry_signed_zero_fixture(
    arr: np.ndarray | jax.Array,
    mask_2d: np.ndarray | jax.Array,
    *,
    nn_hls: int = 2,
) -> dict[str, Any]:
    """Prove that a dry `-0.0` (and dry nonzero perturbation) is ignored by wet-masked scoring while a wet `-0.0` / 1-ULP perturbation at the first wet cell fires."""
    base = np.asarray(arr, dtype=np.float64).copy()
    mask = wet_mask_for_field("fixture", mask_2d, nn_hls=nn_hls, score_defined_halos=True)
    wi, wj = first_wet_ij(mask, interior_only=True, nn_hls=nn_hls)
    di, dj = first_dry_ij(mask, interior_only=True, nn_hls=nn_hls)

    # 1. Dry -0.0 and dry nonzero perturbation must give 0 ULP on wet-masked comparison.
    dry_perturbed = base.copy()
    dry_perturbed[di, dj] = -0.0
    ulp_dry_negzero = max_ulp_diff(base, dry_perturbed, mask=mask, strict_signed_zero=True)
    dry_perturbed[di, dj] = -999.0
    ulp_dry_val = max_ulp_diff(base, dry_perturbed, mask=mask, strict_signed_zero=True)
    norm_dry_val = max_normalized_diff(base, dry_perturbed, mask=mask)
    if ulp_dry_negzero != 0 or ulp_dry_val != 0 or norm_dry_val != 0.0:
        raise ValueError(
            f"STATUS PLANT-FIRED: dry cell perturbation leaked into wet-masked score "
            f"(ulp_negzero={ulp_dry_negzero}, ulp_val={ulp_dry_val}, norm={norm_dry_val})"
        )

    # The southern closed-boundary halo is dry and therefore does not score.
    halo_perturbed = base.copy()
    halo_idx = (0, 0) + (0,) * (base.ndim - 2)
    halo_perturbed[halo_idx] = -777.0
    ulp_halo_val = max_ulp_diff(base, halo_perturbed, mask=mask)
    if ulp_halo_val != 0:
        raise ValueError("STATUS PLANT-FIRED: dry southern halo leaked into wet score")

    # 2. Wet -0.0 vs +0.0 at first wet cell must fire under strict_signed_zero=True.
    wet_zero_a = base.copy()
    wet_zero_b = base.copy()
    wet_zero_a[wi, wj] = +0.0
    wet_zero_b[wi, wj] = -0.0
    ulp_wet_negzero = max_ulp_diff(wet_zero_a, wet_zero_b, mask=mask, strict_signed_zero=True)
    if ulp_wet_negzero == 0:
        raise ValueError("STATUS PLANT-FIRED: wet -0.0 vs +0.0 at first wet cell did not fire")

    # 3. Wet +1 ULP at first wet cell must fire.
    wet_1ulp = base.copy()
    idx = (wi, wj) + (0,) * (base.ndim - 2)
    wet_1ulp[idx] = np.nextafter(wet_1ulp[idx], np.inf)
    ulp_wet_1ulp = max_ulp_diff(base, wet_1ulp, mask=mask)
    if ulp_wet_1ulp != 1:
        raise ValueError(
            f"STATUS PLANT-FIRED: expected 1 ULP at first wet cell {idx}, got {ulp_wet_1ulp}"
        )

    return {
        "first_wet_ij": [wi, wj],
        "first_dry_ij": [di, dj],
        "ulp_dry_negzero": ulp_dry_negzero,
        "ulp_dry_val": ulp_dry_val,
        "ulp_halo_val": ulp_halo_val,
        "ulp_wet_negzero": ulp_wet_negzero,
        "ulp_wet_1ulp": ulp_wet_1ulp,
    }


WRITER_UNIT = "o1_si3_state_writer"
OBSERVER_UNITS = ("icedyn_adv_pra", "icestp")


def same_program(
    new_bld: Path,
    ref_bld: Path,
    pra_patch: Path,
    stp_patch: Path,
    *,
    new_cfg: str = "ORCA2_SAS_ICE_O1_REC",
    ref_cfg: str = "ORCA2_SAS_ICE_O1_REF",
) -> dict[str, Any]:
    """Adapt the shared ORCA2 multi-observer same-program proof for O1."""
    try:
        report = same_program_observers(
            new_bld,
            ref_bld,
            dict(zip(OBSERVER_UNITS, (pra_patch, stp_patch), strict=True)),
            (WRITER_UNIT,),
            new_cfg=new_cfg,
            ref_cfg=ref_cfg,
        )
    except SameProgramError as exc:
        raise ValueError(f"STATUS PLANT-FIRED: {exc}") from exc
    return {
        **report,
        "status": "PASS_SAME_PROGRAM",
        "reference_bld": str(ref_bld),
        "new_bld": str(new_bld),
    }


def same_source_and_flags(
    new_bld: Path,
    ref_bld: Path,
    pra_patch: Path,
    stp_patch: Path,
    *,
    new_cfg: str = "ORCA2_SAS_ICE_O1_REC",
    ref_cfg: str = "ORCA2_SAS_ICE_O1_REF",
) -> dict[str, Any]:
    """Exercise the planned source/flag comparison on the preflight fixture."""
    try:
        checked = same_program_observers(
            new_bld,
            ref_bld,
            dict(zip(OBSERVER_UNITS, (pra_patch, stp_patch), strict=True)),
            (WRITER_UNIT,),
            new_cfg=new_cfg,
            ref_cfg=ref_cfg,
        )
    except SameProgramError as exc:
        raise ValueError(f"STATUS PLANT-FIRED: {exc}") from exc
    return {
        "status": "PASS_SAME_SOURCE_AND_FLAGS",
        "identical_preprocessed_units": checked["identical_ppsrc"],
        "observer_units": checked["observer_units"],
        "writer_units": [WRITER_UNIT],
        "build_config_files": list(BUILD_CONFIG_FILES),
    }


def _expected_fixture_field(
    jpi: int,
    jpj: int,
    nis0: int,
    nie0: int,
    njs0: int,
    nje0: int,
    offset: float,
    ndim: int,
    nlay: int = 0,
    jpl: int = 5,
    is_a2d: bool = False,
) -> np.ndarray:
    """Reproduce the exact rank/field pattern written by `o1_si3_layout_fixture.F90`."""
    ji = np.arange(1, jpi + 1, dtype=np.float64)[:, None]
    jj = np.arange(1, jpj + 1, dtype=np.float64)[None, :]
    base_2d = offset + 100.0 * ji + jj
    if ndim == 2:
        out = base_2d.copy()
        if is_a2d:
            mask = np.zeros((jpi, jpj), dtype=bool)
            mask[nis0 - 1 : nie0, njs0 - 1 : nje0] = True
            out = np.where(mask, out, 0.0)
        return out
    if ndim == 3:
        jl = np.arange(1, jpl + 1, dtype=np.float64)[None, None, :]
        out = base_2d[:, :, None] + 0.1 * jl
        if is_a2d:
            mask = np.zeros((jpi, jpj, 1), dtype=bool)
            mask[nis0 - 1 : nie0, njs0 - 1 : nje0, 0] = True
            out = np.where(mask, out, 0.0)
        return out
    if ndim == 4:
        jk = np.arange(1, nlay + 1, dtype=np.float64)[None, None, :, None]
        jl = np.arange(1, jpl + 1, dtype=np.float64)[None, None, None, :]
        return base_2d[:, :, None, None] + 0.01 * jk + 0.1 * jl
    raise ValueError(f"Unsupported ndim={ndim}")


_FIXTURE_FIELD_OFFSETS: tuple[tuple[str, float, int, int, bool], ...] = (
    ("tmask", 1.1, 2, 0, False),
    ("umask", 2.1, 2, 0, False),
    ("vmask", 3.1, 2, 0, False),
    ("sst_m", 4.0, 2, 0, False),
    ("sss_m", 5.0, 2, 0, False),
    ("v_i", 10.0, 3, 0, False),
    ("v_s", 11.0, 3, 0, False),
    ("sv_i", 12.0, 3, 0, False),
    ("a_i", 13.0, 3, 0, False),
    ("t_su", 14.0, 3, 0, False),
    ("oa_i", 17.0, 3, 0, False),
    ("a_ip", 18.0, 3, 0, False),
    ("v_ip", 19.0, 3, 0, False),
    ("v_il", 20.0, 3, 0, False),
    ("cnd_ice", 24.0, 3, 0, True),
    ("t1_ice", 25.0, 3, 0, True),
    ("t_si", 25.5, 3, 0, True),
    ("u_ice", 15.0, 2, 0, False),
    ("v_ice", 16.0, 2, 0, False),
    ("snwice_mass", 26.0, 2, 0, False),
    ("snwice_mass_b", 27.0, 2, 0, False),
    ("snwice_fmass", 28.0, 2, 0, False),
    ("e_s", 21.0, 4, 5, False),
    ("e_i", 22.0, 4, 10, False),
    ("szv_i", 23.0, 4, 10, False),
    ("sxice", 30.0, 3, 0, False),
    ("syice", 31.0, 3, 0, False),
    ("sxxice", 32.0, 3, 0, False),
    ("syyice", 33.0, 3, 0, False),
    ("sxyice", 34.0, 3, 0, False),
    ("sxsn", 35.0, 3, 0, False),
    ("sysn", 36.0, 3, 0, False),
    ("sxxsn", 37.0, 3, 0, False),
    ("syysn", 38.0, 3, 0, False),
    ("sxysn", 39.0, 3, 0, False),
    ("sxa", 40.0, 3, 0, False),
    ("sya", 41.0, 3, 0, False),
    ("sxxa", 42.0, 3, 0, False),
    ("syya", 43.0, 3, 0, False),
    ("sxya", 44.0, 3, 0, False),
    ("sxage", 45.0, 3, 0, False),
    ("syage", 46.0, 3, 0, False),
    ("sxxage", 47.0, 3, 0, False),
    ("syyage", 48.0, 3, 0, False),
    ("sxyage", 49.0, 3, 0, False),
    ("sxsal", 60.0, 3, 0, False),
    ("sysal", 61.0, 3, 0, False),
    ("sxxsal", 62.0, 3, 0, False),
    ("syysal", 63.0, 3, 0, False),
    ("sxysal", 64.0, 3, 0, False),
    ("sxap", 70.0, 3, 0, False),
    ("syap", 71.0, 3, 0, False),
    ("sxxap", 72.0, 3, 0, False),
    ("syyap", 73.0, 3, 0, False),
    ("sxyap", 74.0, 3, 0, False),
    ("sxvp", 75.0, 3, 0, False),
    ("syvp", 76.0, 3, 0, False),
    ("sxxvp", 77.0, 3, 0, False),
    ("syyvp", 78.0, 3, 0, False),
    ("sxyvp", 79.0, 3, 0, False),
    ("sxvl", 80.0, 3, 0, False),
    ("syvl", 81.0, 3, 0, False),
    ("sxxvl", 82.0, 3, 0, False),
    ("syyvl", 83.0, 3, 0, False),
    ("sxyvl", 84.0, 3, 0, False),
    ("sxc0", 50.0, 4, 5, False),
    ("syc0", 51.0, 4, 5, False),
    ("sxxc0", 52.0, 4, 5, False),
    ("syyc0", 53.0, 4, 5, False),
    ("sxyc0", 54.0, 4, 5, False),
    ("sxe", 55.0, 4, 10, False),
    ("sye", 56.0, 4, 10, False),
    ("sxxe", 57.0, 4, 10, False),
    ("syye", 58.0, 4, 10, False),
    ("sxye", 59.0, 4, 10, False),
    ("sxsi", 65.0, 4, 10, False),
    ("sysi", 66.0, 4, 10, False),
    ("sxxsi", 67.0, 4, 10, False),
    ("syysi", 68.0, 4, 10, False),
    ("sxysi", 69.0, 4, 10, False),
    ("h_i", 90.0, 3, 0, False),
    ("h_s", 91.0, 3, 0, False),
    ("o_i", 92.0, 3, 0, False),
    ("h_ip", 93.0, 3, 0, False),
    ("h_il", 94.0, 3, 0, False),
    ("a_ip_frac", 95.0, 3, 0, True),
    ("a_ip_eff", 96.0, 3, 0, True),
    ("s_i", 97.0, 3, 0, False),
    ("sz_i", 98.0, 4, 10, False),
    ("t_i", 99.0, 4, 10, False),
    ("t_s", 100.0, 4, 5, False),
    ("vt_i", 110.0, 2, 0, False),
    ("vt_s", 111.0, 2, 0, False),
    ("at_i", 112.0, 2, 0, False),
    ("at_ip", 113.0, 2, 0, False),
    ("vt_ip", 114.0, 2, 0, False),
    ("vt_il", 115.0, 2, 0, False),
    ("ato_i", 116.0, 2, 0, False),
    ("et_s", 117.0, 2, 0, True),
    ("et_i", 118.0, 2, 0, True),
    ("at_ip_eff", 119.0, 2, 0, True),
    ("tm_su", 120.0, 2, 0, True),
    ("st_i", 121.0, 2, 0, True),
    ("hm_i", 122.0, 2, 0, True),
    ("hm_s", 123.0, 2, 0, True),
    ("tm_si", 124.0, 2, 0, True),
    ("om_i", 125.0, 2, 0, True),
    ("sm_i", 126.0, 2, 0, True),
    ("tm_i", 127.0, 2, 0, True),
    ("tm_s", 128.0, 2, 0, True),
    ("hm_ip", 129.0, 2, 0, True),
    ("hm_il", 130.0, 2, 0, True),
)


def verify_layout_fixture(
    records_dir: str | Path,
    plant: str | None = None,
) -> dict[str, Any]:
    """Verify the 4-rank compiled `gfortran` layout fixture and its negative controls."""
    rdir = Path(records_dir)
    expected_bytes = 868548
    entry_paths = [rdir / f"o1_si3_state_entry_rank{r:04d}.bin" for r in range(4)]
    step1_paths = [rdir / f"o1_si3_state_step0001_rank{r:04d}.bin" for r in range(4)]

    if plant == "missing-frame":
        step1_paths = step1_paths[:-1]

    if len(entry_paths) != 4 or len(step1_paths) != 4:
        raise ValueError("STATUS PLANT-FIRED: missing rank file in layout fixture")

    for p in entry_paths + step1_paths:
        if not p.is_file():
            raise ValueError(f"STATUS PLANT-FIRED: missing expected fixture file {p}")
        if p.stat().st_size != expected_bytes:
            raise ValueError(
                f"STATUS PLANT-FIRED: file size {p.stat().st_size} != {expected_bytes}"
            )

    if plant == "truncation":
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "truncated.bin"
            bad_file.write_bytes(entry_paths[0].read_bytes()[:-16])
            try:
                read_o1_binary_snapshot(bad_file)
            except ValueError as exc:
                raise ValueError(f"STATUS PLANT-FIRED: {exc}") from exc
            raise RuntimeError("Truncation plant did not fire")

    if plant == "bad-magic":
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "bad_magic.bin"
            raw = bytearray(entry_paths[0].read_bytes())
            raw[:8] = b"BADMAGIC"
            bad_file.write_bytes(bytes(raw))
            try:
                read_o1_binary_snapshot(bad_file)
            except ValueError as exc:
                raise ValueError(f"STATUS PLANT-FIRED: {exc}") from exc
            raise RuntimeError("Bad-magic plant did not fire")

    snaps_entry = [read_o1_binary_snapshot(p) for p in entry_paths]
    snaps_step1 = [read_o1_binary_snapshot(p) for p in step1_paths]

    if plant == "rank-collapse":
        snaps_entry[1]["fields"]["v_i"] = snaps_entry[0]["fields"]["v_i"].copy()
    elif plant == "wrong-offset":
        snaps_entry[2]["meta"]["nimpp"] += 1
    elif plant == "field-swap":
        v_i_tmp = snaps_entry[0]["fields"]["v_i"].copy()
        snaps_entry[0]["fields"]["v_i"] = snaps_entry[0]["fields"]["v_s"].copy()
        snaps_entry[0]["fields"]["v_s"] = v_i_tmp

    # Verify rank uniqueness across all 4 ranks
    for r in range(4):
        for s in range(r + 1, 4):
            if np.allclose(snaps_entry[r]["fields"]["v_i"], snaps_entry[s]["fields"]["v_i"]):
                raise ValueError(f"STATUS PLANT-FIRED: ranks {r} and {s} have identical v_i data")

    # Verify every single field in each rank matches exact formula to 0 ULP
    bounds = si3_hfn_category_bounds(jpl=5, rn_himean=2.0, opt_hips=0, rn_himax=99.0)
    for irank, snap in enumerate(snaps_entry):
        meta = snap["meta"]
        rx = irank % 2
        ry = irank // 2
        if meta["mpprank"] != irank or meta["narea"] != irank + 1:
            raise ValueError("STATUS PLANT-FIRED: rank/narea metadata mismatch")
        if meta["nimpp"] != 1 + rx * 6 or meta["njmpp"] != 1 + ry * 5:
            raise ValueError("STATUS PLANT-FIRED: subdomain nimpp/njmpp offset mismatch")
        if max_ulp_diff(snap["hi_max"], np.asarray(bounds.hi_max)) != 0:
            raise ValueError("STATUS PLANT-FIRED: hi_max bounds mismatch in fixture")
        if max_ulp_diff(snap["hi_mean"], np.asarray(bounds.hi_mean)) != 0:
            raise ValueError("STATUS PLANT-FIRED: hi_mean bounds mismatch in fixture")

        rbase = 10000.0 * float(irank + 1)
        for fname, foff, ndim, nlay, is_a2d in _FIXTURE_FIELD_OFFSETS:
            expected_arr = _expected_fixture_field(
                jpi=meta["jpi"],
                jpj=meta["jpj"],
                nis0=meta["Nis0"],
                nie0=meta["Nie0"],
                njs0=meta["Njs0"],
                nje0=meta["Nje0"],
                offset=rbase + foff,
                ndim=ndim,
                nlay=nlay,
                jpl=meta["jpl"],
                is_a2d=is_a2d,
            )
            ulp = max_ulp_diff(snap["fields"][fname], expected_arr)
            if ulp != 0:
                raise ValueError(f"STATUS PLANT-FIRED: rank {irank} field {fname} ULP={ulp}")

    stitched_entry = stitch_o1_binary_snapshots(snaps_entry)
    stitched_step1 = stitch_o1_binary_snapshots(snaps_step1)
    if stitched_entry["meta"]["jpi"] != 16 or stitched_entry["meta"]["jpj"] != 14:
        raise ValueError("STATUS PLANT-FIRED: stitched global grid shape mismatch")
    if stitched_step1["meta"]["lrst_ice"] != 1 or stitched_entry["meta"]["lrst_ice"] != 0:
        raise ValueError("STATUS PLANT-FIRED: lrst_ice metadata mismatch")

    # Verify stitched subdomain interiors match each rank's interior to 0 ULP
    for irank, snap in enumerate(snaps_entry):
        meta = snap["meta"]
        nis0, nie0 = meta["Nis0"], meta["Nie0"]
        njs0, nje0 = meta["Njs0"], meta["Nje0"]
        gi0 = meta["nimpp"] + nis0 - 2
        gi1 = meta["nimpp"] + nie0 - 1
        gj0 = meta["njmpp"] + njs0 - 2
        gj1 = meta["njmpp"] + nje0 - 1
        for fname, *_ in _FIXTURE_FIELD_OFFSETS:
            sub_local = snap["fields"][fname][nis0 - 1 : nie0, njs0 - 1 : nje0]
            sub_global = stitched_entry["fields"][fname][gi0:gi1, gj0:gj1]
            if max_ulp_diff(sub_local, sub_global) != 0:
                raise ValueError(
                    f"STATUS PLANT-FIRED: stitched interior mismatch on rank {irank} field {fname}"
                )

    return {
        "status": "PASS",
        "ranks_verified": 4,
        "fields_per_rank": len(_FIXTURE_FIELD_OFFSETS),
        "bytes_per_rank": expected_bytes,
        "stitched_shape": [
            int(stitched_entry["meta"]["jpi"]),
            int(stitched_entry["meta"]["jpj"]),
        ],
        "max_ulp": 0,
    }


def verify_ocean_output_log(
    log_path: str | Path,
    expect_icedyn: bool | None = None,
    expect_icethd: bool | None = None,
    expect_rstart: bool | None = None,
    plant: str | None = None,
) -> dict[str, Any]:
    """Whitespace-tolerant NEMO `ocean.output` verifier (`LOG-CHECK NOTE`)."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    if plant == "error-token":
        text = text + "\n E R R O R : planted fatal error in ocean.output\n"
    elif plant == "missing-aaa":
        text = text.replace("AAAAAAAA", "BBBBBBBB")
    elif plant == "wrong-jpl":
        text = re.sub(r"(jpl\s*=\s*)\d+", r"\g<1>3", text)
    elif plant == "wrong-flag":
        text = re.sub(r"(ln_cat_hfn\s*=\s*)[TF]", r"\g<1>F", text)

    error_lines = text.splitlines()
    error_index = next(
        (i for i, line in enumerate(error_lines) if re.search(r"E\s+R\s+R\s+O\s+R", line)),
        None,
    )
    if error_index is not None:
        quoted = "\n".join(
            f"ocean.output:{i + 1}: {error_lines[i].rstrip()}"
            for i in range(error_index, min(error_index + 7, len(error_lines)))
            if error_lines[i].strip()
        )
        raise ValueError(
            "STATUS PLANT-FIRED: ocean.output contains E R R O R; NEMO error lines:\n"
            + quoted
        )

    aaa_count = len(re.findall(r"^\s*AAAAAAAA\s*$", text, flags=re.M))
    if aaa_count < 2:
        raise ValueError(
            f"STATUS PLANT-FIRED: expected >= 2 AAAAAAAA banner markers, found {aaa_count}"
        )

    def _extract_int(name: str) -> int:
        m = re.search(rf"\b{re.escape(name)}\s*=\s*([+-]?\d+)", text)
        if not m:
            raise ValueError(f"STATUS PLANT-FIRED: missing integer {name} in log")
        return int(m.group(1))

    def _extract_float(name: str) -> float:
        m = re.search(
            rf"\b{re.escape(name)}\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?)",
            text,
        )
        if not m:
            raise ValueError(f"STATUS PLANT-FIRED: missing float {name} in log")
        return float(m.group(1).replace("D", "e").replace("d", "e"))

    def _extract_bool(name: str) -> bool:
        m = re.search(
            rf"\b{re.escape(name)}\s*=\s*(\.?true\.?|\.?false\.?|T|F)\b",
            text,
            flags=re.I,
        )
        if not m:
            raise ValueError(f"STATUS PLANT-FIRED: missing logical {name} in log")
        token = m.group(1).strip(".").upper()
        return token in ("T", "TRUE")

    parsed = {
        "Ni0glo": _extract_int("Ni0glo"),
        "Nj0glo": _extract_int("Nj0glo"),
        "nn_hls": _extract_int("nn_hls"),
        "jpl": _extract_int("jpl"),
        "nlay_i": _extract_int("nlay_i"),
        "nlay_s": _extract_int("nlay_s"),
        "ln_virtual_itd": _extract_bool("ln_virtual_itd"),
        "ln_icedyn": _extract_bool("ln_icedyn"),
        "ln_icethd": _extract_bool("ln_icethd"),
        "ln_cat_hfn": _extract_bool("ln_cat_hfn"),
        "rn_himean": _extract_float("rn_himean"),
        "aaa_markers": aaa_count,
    }
    if (
        parsed["Ni0glo"] != 180
        or parsed["Nj0glo"] != 148
        or parsed["nn_hls"] != 2
        or parsed["jpl"] != 5
        or parsed["nlay_i"] != 10
        or parsed["nlay_s"] != 5
        or parsed["ln_virtual_itd"] is not False
        or parsed["ln_cat_hfn"] is not True
        or abs(parsed["rn_himean"] - 2.0) > 1e-12
    ):
        raise ValueError(f"STATUS PLANT-FIRED: unexpected ORCA2 SI3 parameters {parsed}")

    if expect_icedyn is not None and parsed["ln_icedyn"] != expect_icedyn:
        raise ValueError(
            f"STATUS PLANT-FIRED: ln_icedyn={parsed['ln_icedyn']}, expected {expect_icedyn}"
        )
    if expect_icethd is not None and parsed["ln_icethd"] != expect_icethd:
        raise ValueError(
            f"STATUS PLANT-FIRED: ln_icethd={parsed['ln_icethd']}, expected {expect_icethd}"
        )
    if expect_rstart is not None:
        parsed["ln_rstart"] = _extract_bool("ln_rstart")
        if parsed["ln_rstart"] != expect_rstart:
            raise ValueError(
                f"STATUS PLANT-FIRED: ln_rstart={parsed['ln_rstart']}, expected {expect_rstart}"
            )
    return {"status": "PASS", "parsed": parsed}


def verify_o1_deck_fixture(path: str | Path, plant: str | None = None) -> dict[str, Any]:
    """Verify only log lines derivable from the committed O1 namelist deck."""
    text = Path(path).read_text(encoding="utf-8")
    if "DERIVED O1 DECK FIXTURE; NOT NEMO RUN EVIDENCE" not in text:
        raise ValueError("STATUS PLANT-FIRED: derived-fixture disclaimer missing")
    if plant == "error-token":
        text += "\n E R R O R\n"
    elif plant == "missing-line":
        text = re.sub(r"^.*nlay_s.*$", "", text, flags=re.M)
    elif plant == "wrong-jpl":
        text = re.sub(r"(jpl\s*=\s*)5", r"\g<1>3", text)
    elif plant == "wrong-flag":
        text = re.sub(r"(ln_cat_hfn\s*=\s*)T", r"\g<1>F", text)
    if re.search(r"E\s+R\s+R\s+O\s+R", text):
        raise ValueError("STATUS PLANT-FIRED: fixture contains E R R O R")

    expected = {
        "jpl": "5",
        "nlay_i": "10",
        "nlay_s": "5",
        "ln_virtual_itd": "F",
        "ln_icedyn": "F",
        "ln_icethd": "F",
        "ln_cat_hfn": "T",
        "rn_himean": "2.0000000000000000",
    }
    for name, value in expected.items():
        match = re.search(rf"\b{re.escape(name)}\s*=\s*([^\s!]+)", text)
        if not match or match.group(1) != value:
            raise ValueError(
                f"STATUS PLANT-FIRED: derived fixture {name}="
                f"{None if not match else match.group(1)!r}, expected {value!r}"
            )
    return {"status": "PASS_DERIVED_DECK_FIXTURE", "parsed": expected}


def build_synthetic_orca2_o1_state(
    config: SI3ORCA2StateConfig | None = None,
) -> tuple[SI3State, jax.Array, jax.Array, jax.Array]:
    """Construct a deterministic non-trivial full ORCA2 `(184, 152)` SI3 state for gate self-tests."""
    set_policy(PrecisionPolicy.fp64())
    cfg = config or SI3ORCA2StateConfig.for_orca2_sas_ice()
    jpi, jpj, jpl, nlay_i, nlay_s = 184, 152, cfg.jpl, cfg.nlay_i, cfg.nlay_s

    rng = np.random.default_rng(20261024)
    tmask_np = np.ones((jpi, jpj), dtype=np.float64)
    tmask_np[:, :2] = 0.0
    tmask_np[40:55, 60:75] = 0.0
    umask_np = tmask_np.copy()
    vmask_np = tmask_np.copy()

    tmask = jnp.asarray(tmask_np)
    umask = jnp.asarray(umask_np)
    vmask = jnp.asarray(vmask_np)

    def _halo_t(arr: np.ndarray, psgn: float = 1.0) -> jax.Array:
        return si3_lbc_lnk_orca2(
            jnp.asarray(arr, dtype=jnp.float64),
            cd_nat="T",
            psgn=psgn,
            nn_hls=cfg.nn_hls,
            ldfull=True,
        )

    a_i_raw = rng.uniform(0.02, 0.17, size=(jpi, jpj, jpl)) * tmask_np[:, :, None]
    a_i = _halo_t(a_i_raw, 1.0)
    h_i_cat = np.array([0.25, 0.80, 1.60, 2.80, 4.50], dtype=np.float64)[None, None, :]
    v_i = _halo_t(np.asarray(a_i) * h_i_cat * rng.uniform(0.9, 1.1, size=(jpi, jpj, jpl)), 1.0)
    v_s = _halo_t(np.asarray(a_i) * 0.15 * rng.uniform(0.8, 1.2, size=(jpi, jpj, jpl)), 1.0)
    s_bulk = rng.uniform(4.0, 6.5, size=(jpi, jpj, jpl))
    sv_i = _halo_t(np.asarray(v_i) * s_bulk, 1.0)
    t_su = _halo_t(rng.uniform(258.0, 271.5, size=(jpi, jpj, jpl)), 1.0)
    oa_i = _halo_t(np.asarray(a_i) * rng.uniform(10.0, 120.0, size=(jpi, jpj, jpl)), 1.0)
    a_ip = _halo_t(np.asarray(a_i) * rng.uniform(0.05, 0.25, size=(jpi, jpj, jpl)), 1.0)
    v_ip = _halo_t(np.asarray(a_ip) * rng.uniform(0.02, 0.08, size=(jpi, jpj, jpl)), 1.0)
    v_il = _halo_t(np.asarray(a_ip) * rng.uniform(0.002, 0.02, size=(jpi, jpj, jpl)), 1.0)

    e_s = _halo_t(
        np.asarray(v_s)[:, :, None, :]
        * (1.0 / float(nlay_s))
        * rng.uniform(1.05e8, 1.15e8, size=(jpi, jpj, nlay_s, jpl)),
        1.0,
    )
    e_i = _halo_t(
        np.asarray(v_i)[:, :, None, :]
        * (1.0 / float(nlay_i))
        * rng.uniform(2.8e8, 3.2e8, size=(jpi, jpj, nlay_i, jpl)),
        1.0,
    )
    szv_i = _halo_t(
        np.asarray(v_i)[:, :, None, :]
        * (1.0 / float(nlay_i))
        * rng.uniform(3.5, 6.5, size=(jpi, jpj, nlay_i, jpl)),
        1.0,
    )
    u_ice = si3_lbc_lnk_orca2(
        jnp.asarray(rng.uniform(-0.15, 0.15, size=(jpi, jpj)) * umask_np),
        cd_nat="U",
        psgn=-1.0,
        nn_hls=cfg.nn_hls,
        ldfull=True,
    )
    v_ice = si3_lbc_lnk_orca2(
        jnp.asarray(rng.uniform(-0.15, 0.15, size=(jpi, jpj)) * vmask_np),
        cd_nat="V",
        psgn=-1.0,
        nn_hls=cfg.nn_hls,
        ldfull=True,
    )

    def _rand_moments(shape: tuple[int, ...], scale: float) -> SI3PratherMoments:
        return SI3PratherMoments(
            sx=_halo_t(rng.normal(0.0, scale, size=shape), -1.0),
            sy=_halo_t(rng.normal(0.0, scale, size=shape), -1.0),
            sxx=_halo_t(rng.normal(0.0, scale, size=shape), 1.0),
            syy=_halo_t(rng.normal(0.0, scale, size=shape), 1.0),
            sxy=_halo_t(rng.normal(0.0, scale, size=shape), 1.0),
        )

    s3 = (jpi, jpj, jpl)
    s4_s = (jpi, jpj, nlay_s, jpl)
    s4_i = (jpi, jpj, nlay_i, jpl)
    prather = SI3AllPratherMoments(
        ice=_rand_moments(s3, 1e-2),
        sn=_rand_moments(s3, 1e-3),
        a=_rand_moments(s3, 1e-3),
        age=_rand_moments(s3, 1e-1),
        sal=(_rand_moments(s3, 1e-2) if cfg.nn_icesal != 4 else SI3PratherMoments.zeros(s3)),
        ap=_rand_moments(s3, 1e-4),
        vp=_rand_moments(s3, 1e-5),
        vl=_rand_moments(s3, 1e-6),
        c0=_rand_moments(s4_s, 1e5),
        e=_rand_moments(s4_i, 1e6),
        si=(_rand_moments(s4_i, 1e-2) if cfg.nn_icesal == 4 else SI3PratherMoments.zeros(s4_i)),
    )

    cnd_ice = jnp.zeros(s3, dtype=jnp.float64)
    t1_ice = jnp.full(s3, cfg.ice_constants.T0, dtype=jnp.float64)
    interior_3d = np.zeros((jpi, jpj, 1), dtype=bool)
    interior_3d[cfg.nn_hls : -cfg.nn_hls, cfg.nn_hls : -cfg.nn_hls, 0] = True
    t_si = jnp.where(
        jnp.asarray(interior_3d),
        jnp.full(s3, cfg.ice_constants.T0, dtype=jnp.float64),
        0.0,
    )
    bounds = si3_hfn_category_bounds(
        jpl=jpl, rn_himean=cfg.rn_himean, opt_hips=cfg.opt_hips, rn_himax=cfg.rn_himax
    )
    eqv = si3_var_glo2eqv(
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        a_i=a_i,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        kn=2,
        hi_max_top=bounds.hi_max[-1],
        config=cfg,
    )
    agg = si3_var_agg(
        v_i=v_i,
        v_s=eqv.v_s,
        sv_i=sv_i,
        a_i=eqv.a_i,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        t_su=t_su,
        t_si=t_si,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        a_ip_eff=eqv.a_ip_eff,
        t_i=eqv.t_i,
        t_s=eqv.t_s,
        kn=2,
        config=cfg,
    )
    snwice_mass, snwice_mass_b, snwice_fmass = si3_update_snwice_mass(
        vt_i=agg.vt_i,
        vt_s=agg.vt_s,
        vt_ip=agg.vt_ip,
        vt_il=agg.vt_il,
        snwice_mass_prev=jnp.zeros((jpi, jpj), dtype=jnp.float64),
        tmask=tmask,
        config=cfg,
        is_init=True,
    )
    state = SI3State(
        v_i=v_i,
        v_s=eqv.v_s,
        sv_i=sv_i,
        a_i=eqv.a_i,
        t_su=t_su,
        u_ice=u_ice,
        v_ice=v_ice,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        cnd_ice=cnd_ice,
        t1_ice=t1_ice,
        t_si=t_si,
        snwice_mass=snwice_mass,
        snwice_mass_b=snwice_mass_b,
        snwice_fmass=snwice_fmass,
        prather=prather,
        bounds=bounds,
        equivalent=eqv,
        aggregated=agg,
    )
    return state, tmask, umask, vmask


_PROG_FIELDS: tuple[str, ...] = (
    "v_i",
    "v_s",
    "sv_i",
    "a_i",
    "t_su",
    "u_ice",
    "v_ice",
    "oa_i",
    "a_ip",
    "v_ip",
    "v_il",
    "e_s",
    "e_i",
    "szv_i",
    "t1_ice",
    "t_si",
    "snwice_mass",
    "snwice_mass_b",
)

_EQUIVALENT_HALO_FIELDS = frozenset(
    {"h_i", "h_s", "o_i", "h_ip", "h_il", "s_i", "sz_i", "t_i", "t_s", "a_i", "v_s"}
)
_AGGREGATED_HALO_FIELDS = frozenset({"vt_i", "vt_s", "at_i", "at_ip", "vt_ip", "vt_il", "ato_i"})


def _compare_si3_states_wet_and_full(
    expected: SI3State,
    actual: SI3State,
    tmask: np.ndarray | jax.Array,
    umask: np.ndarray | jax.Array,
    vmask: np.ndarray | jax.Array,
    *,
    check_fmass: bool = False,
    check_prather: bool = True,
    score_t1_ice: bool = False,
    nn_hls: int = 2,
) -> dict[str, Any]:
    """Compare every NEMO-defined wet cell, including seam/fold halos."""
    prog_names = _PROG_FIELDS + (("snwice_fmass",) if check_fmass else ())
    if not score_t1_ice:
        # icerst.F90:329 defines t1_ice only for ln_cpl; O1 has ln_cpl=.false.
        prog_names = tuple(name for name in prog_names if name != "t1_ice")
    max_ulp_wet = 0
    max_ulp_full = 0
    max_norm_wet = 0.0

    for fname in prog_names:
        m = wet_mask_for_field(fname, tmask, umask, vmask, nn_hls=nn_hls)
        a_arr = getattr(expected, fname)
        b_arr = getattr(actual, fname)
        ulp_w = max_ulp_diff(a_arr, b_arr, mask=m)
        ulp_f = max_ulp_diff(a_arr, b_arr)
        norm_w = max_normalized_diff(a_arr, b_arr, mask=m)
        max_ulp_wet = max(max_ulp_wet, ulp_w)
        max_ulp_full = max(max_ulp_full, ulp_f)
        max_norm_wet = max(max_norm_wet, norm_w)
        if ulp_w != 0 or norm_w > 1e-15:
            raise ValueError(
                f"STATUS PLANT-FIRED: prognostic field {fname} wet_ULP={ulp_w}, "
                f"full_ULP={ulp_f}, wet_norm={norm_w:.3e}"
            )

    if check_prather:
        for group_name in SI3AllPratherMoments._fields:
            grp1 = getattr(expected.prather, group_name)
            grp2 = getattr(actual.prather, group_name)
            for mname in SI3PratherMoments._fields:
                a_arr = getattr(grp1, mname)
                b_arr = getattr(grp2, mname)
                wet_t = wet_mask_for_field(mname, tmask, nn_hls=nn_hls, score_defined_halos=True)
                ulp_w = max_ulp_diff(a_arr, b_arr, mask=wet_t)
                ulp_f = max_ulp_diff(a_arr, b_arr)
                norm_w = max_normalized_diff(a_arr, b_arr, mask=wet_t)
                max_ulp_wet = max(max_ulp_wet, ulp_w)
                max_ulp_full = max(max_ulp_full, ulp_f)
                max_norm_wet = max(max_norm_wet, norm_w)
                if ulp_w != 0 or norm_w > 1e-15:
                    raise ValueError(
                        f"STATUS PLANT-FIRED: Prather moment {group_name}.{mname} "
                        f"wet_ULP={ulp_w}, full_ULP={ulp_f}, wet_norm={norm_w:.3e}"
                    )

    for eq_name in expected.equivalent._fields:
        a_arr = getattr(expected.equivalent, eq_name)
        b_arr = getattr(actual.equivalent, eq_name)
        wet_t = wet_mask_for_field(
            eq_name,
            tmask,
            nn_hls=nn_hls,
            score_defined_halos=eq_name in _EQUIVALENT_HALO_FIELDS,
        )
        ulp_w = max_ulp_diff(a_arr, b_arr, mask=wet_t)
        ulp_f = max_ulp_diff(a_arr, b_arr)
        norm_w = max_normalized_diff(a_arr, b_arr, mask=wet_t)
        max_ulp_wet = max(max_ulp_wet, ulp_w)
        max_ulp_full = max(max_ulp_full, ulp_f)
        max_norm_wet = max(max_norm_wet, norm_w)
        if ulp_w != 0 or norm_w > 1e-15:
            raise ValueError(
                f"STATUS PLANT-FIRED: equivalent field {eq_name} wet_ULP={ulp_w}, "
                f"full_ULP={ulp_f}, wet_norm={norm_w:.3e}"
            )

    for agg_name in expected.aggregated._fields:
        a_arr = getattr(expected.aggregated, agg_name)
        b_arr = getattr(actual.aggregated, agg_name)
        wet_t = wet_mask_for_field(
            agg_name,
            tmask,
            nn_hls=nn_hls,
            score_defined_halos=agg_name in _AGGREGATED_HALO_FIELDS,
        )
        ulp_w = max_ulp_diff(a_arr, b_arr, mask=wet_t)
        ulp_f = max_ulp_diff(a_arr, b_arr)
        norm_w = max_normalized_diff(a_arr, b_arr, mask=wet_t)
        max_ulp_wet = max(max_ulp_wet, ulp_w)
        max_ulp_full = max(max_ulp_full, ulp_f)
        max_norm_wet = max(max_norm_wet, norm_w)
        if ulp_w != 0 or norm_w > 1e-15:
            raise ValueError(
                f"STATUS PLANT-FIRED: aggregated field {agg_name} wet_ULP={ulp_w}, "
                f"full_ULP={ulp_f}, wet_norm={norm_w:.3e}"
            )

    return {
        "max_ulp_wet": max_ulp_wet,
        "max_ulp_full": max_ulp_full,
        "max_norm_wet": max_norm_wet,
    }


def run_self_test(plant: str | None = None) -> dict[str, Any]:
    """Run the full O1 gate self-test and negative-control plants."""
    from legoesm.ice.config import build_si3_orca2_o1_config
    from legoesm.ice.sea_ice import step_sea_ice
    from legoesm.ice.si3_orca2_state import SI3ORCA2Forcing

    set_policy(PrecisionPolicy.fp64())
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state0, tmask, umask, vmask = build_synthetic_orca2_o1_state(cfg)
    wi, wj = first_wet_ij(tmask, interior_only=True, nn_hls=cfg.nn_hls)
    wi_u, wj_u = first_wet_ij(umask, interior_only=True, nn_hls=cfg.nn_hls)
    wi_v, wj_v = first_wet_ij(vmask, interior_only=True, nn_hls=cfg.nn_hls)

    state1 = si3_o1_step(state0, tmask=tmask, config=cfg)

    # Production dispatch must refuse until the NEMO ice_update_flx response
    # contract exists; direct si3_o1_step remains the O1 state operator.
    sea_ice_cfg = build_si3_orca2_o1_config(cfg)
    z2 = jnp.zeros_like(tmask)
    try:
        step_sea_ice(
            state0,
            SI3ORCA2Forcing(tmask=tmask),
            z2,
            z2,
            z2,
            sea_ice_cfg,
            0.1,
            cfg.rDt_ice,
        )
    except NotImplementedError as exc:
        if "ice_update_flx" not in str(exc):
            raise
    else:
        raise ValueError("STATUS PLANT-FIRED: production SI3 dispatcher silently returned fluxes")

    dry_fixture = verify_dry_signed_zero_fixture(state1.v_i, tmask, nn_hls=cfg.nn_hls)

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Live O1 NetCDF restart schema (ln_icedyn=False -> write_prather=False, ln_rstart=False)
        rst_o1_path = Path(tmp) / "ORCA2_SAS_00000001_restart_ice.nc"
        write_si3_netcdf_restart(rst_o1_path, state1, kt=1, config=cfg)
        state_o1_reloaded_init = read_si3_netcdf_restart(
            rst_o1_path, tmask=tmask, config=cfg, ln_rstart=False
        )
        state_o1_reloaded = si3_o1_step(state_o1_reloaded_init, tmask=tmask, config=cfg)
        _compare_si3_states_wet_and_full(
            state1, state_o1_reloaded, tmask, umask, vmask, check_prather=False
        )

        # 2. Full Prather moment restart schema (write_prather=True, ln_rstart=True)
        rst_path = Path(tmp) / "ORCA2_SAS_00000001_restart_ice_pra.nc"
        write_si3_netcdf_restart(rst_path, state1, kt=1, config=cfg, write_prather=True)
        state_reloaded_init = read_si3_netcdf_restart(
            rst_path, tmask=tmask, config=cfg, ln_rstart=True
        )
        state_reloaded = si3_o1_step(state_reloaded_init, tmask=tmask, config=cfg)

    if plant == "1ulp-state":
        v_i_np = np.asarray(state_reloaded.v_i).copy()
        v_i_np[wi, wj, 0] = np.nextafter(v_i_np[wi, wj, 0], np.inf)
        state_reloaded = state_reloaded._replace(v_i=jnp.asarray(v_i_np))
    elif plant == "1ulp-moment":
        sx_np = np.asarray(state_reloaded.prather.ice.sx).copy()
        sx_np[wi, wj, 0] = np.nextafter(sx_np[wi, wj, 0], np.inf)
        state_reloaded = state_reloaded._replace(
            prather=state_reloaded.prather._replace(
                ice=state_reloaded.prather.ice._replace(sx=jnp.asarray(sx_np))
            )
        )
    elif plant == "north-fold-sign":
        u_bad = si3_lbc_lnk_orca2(
            state1.u_ice, cd_nat="U", psgn=1.0, nn_hls=cfg.nn_hls, ldfull=True
        )
        state_reloaded = state_reloaded._replace(u_ice=u_bad)
    elif plant == "agg-formula":
        hm_bad = np.asarray(state_reloaded.aggregated.hm_i).copy()
        hm_bad[wi, wj] += 1e-6
        state_reloaded = state_reloaded._replace(
            aggregated=state_reloaded.aggregated._replace(hm_i=jnp.asarray(hm_bad))
        )

    cmp_metrics = _compare_si3_states_wet_and_full(
        state1, state_reloaded, tmask, umask, vmask, check_prather=True
    )

    # Differentiability gate (Plant P6: stop_gradient)
    def _loss(v_i_in: jax.Array) -> jax.Array:
        v_in = jax.lax.stop_gradient(v_i_in) if plant == "stop-gradient" else v_i_in
        s_in = state0._replace(v_i=v_in)
        s_out = si3_o1_step(s_in, tmask=tmask, config=cfg)
        return jnp.sum(
            s_out.aggregated.vt_i
            + s_out.aggregated.hm_i
            + s_out.snwice_mass
            + s_out.equivalent.t_i[:, :, 0, 0]
        )

    loss_val, grad_v_i = jax.jit(jax.value_and_grad(_loss))(state0.v_i)
    grad_norm = float(jnp.linalg.norm(grad_v_i))
    if not np.isfinite(float(loss_val)) or not np.isfinite(grad_norm) or grad_norm == 0.0:
        raise ValueError(
            f"STATUS PLANT-FIRED: AD gate failed (loss={loss_val}, grad_norm={grad_norm})"
        )

    return {
        "status": "PASS",
        "grid_shape": [184, 152],
        "jpl": cfg.jpl,
        "nlay_i": cfg.nlay_i,
        "nlay_s": cfg.nlay_s,
        "first_wet_t_ij": [wi, wj],
        "first_wet_u_ij": [wi_u, wj_u],
        "first_wet_v_ij": [wi_v, wj_v],
        "dry_signed_zero_fixture": dry_fixture,
        "prognostic_fields_checked": len(_PROG_FIELDS),
        "prather_arrays_checked": 55,
        "equivalent_fields_checked": len(state1.equivalent._fields),
        "aggregated_fields_checked": len(state1.aggregated._fields),
        "max_ulp": cmp_metrics["max_ulp_wet"],
        "max_ulp_full": cmp_metrics["max_ulp_full"],
        "max_norm_wet": cmp_metrics["max_norm_wet"],
        "ad_grad_norm": grad_norm,
    }


def _verify_sha256_manifest(
    manifest_path: Path,
    *,
    label: str,
    plant_mismatch: bool = False,
) -> int:
    """Verify a `<sha256>  <path>` manifest file."""
    if not manifest_path.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing manifest {manifest_path}")
    lines = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"STATUS PLANT-FIRED: empty manifest {manifest_path}")
    checked = 0
    for line_number, line in enumerate(lines):
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise ValueError(
                f"STATUS PLANT-FIRED: malformed {label} entry {line!r} in {manifest_path}"
            )
        expected_sha, target_str = parts[0], parts[1].strip()
        if plant_mismatch and line_number == 0:
            expected_sha = "0" * 64
        target_path = Path(target_str)
        if not target_path.is_file():
            raise ValueError(
                f"STATUS PLANT-FIRED: missing file {target_path} referenced by {manifest_path}"
            )
        actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"STATUS PLANT-FIRED: {label} SHA mismatch for {target_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        checked += 1
    return checked


def _verify_run_completion_and_logs(
    run_dir: Path,
    run_label: str,
    *,
    plant_missing_complete: bool = False,
    plant_missing_stop0: bool = False,
    expect_icedyn: bool = False,
    expect_icethd: bool = False,
) -> dict[str, Any]:
    """Verify `RUN_COMPLETE`, `ocean.output`, and `run.user.stdout.log` in `run_dir`."""
    if not run_dir.is_dir():
        raise ValueError(f"STATUS PLANT-FIRED: missing run directory {run_dir}")
    if plant_missing_complete or not (run_dir / "RUN_COMPLETE").is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing RUN_COMPLETE in {run_label} ({run_dir})")
    ocean_out = run_dir / "ocean.output"
    if not ocean_out.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing ocean.output in {run_label} ({run_dir})")
    log_report = verify_ocean_output_log(
        ocean_out,
        expect_icedyn=expect_icedyn,
        expect_icethd=expect_icethd,
        expect_rstart=run_label == "rst_1step",
    )
    stdout_log = run_dir / "run.user.stdout.log"
    if not stdout_log.is_file():
        raise ValueError(
            f"STATUS PLANT-FIRED: missing run.user.stdout.log in {run_label} ({run_dir})"
        )
    stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace")
    if plant_missing_stop0:
        stdout_text = re.sub(r"STOP\s+0\b", "STOP 1", stdout_text)
    if not re.search(r"STOP\s+0\b", stdout_text):
        raise ValueError(f"STATUS PLANT-FIRED: missing STOP 0 in {run_label}/run.user.stdout.log")
    if re.search(r"E\s+R\s+R\s+O\s+R|\bABORT\b", stdout_text):
        raise ValueError(
            f"STATUS PLANT-FIRED: fatal error token in {run_label}/run.user.stdout.log"
        )
    return log_report


def admit_evidence_directory(
    evidence_dir: str | Path,
    plant: str | None = None,
    *,
    expect_commit: str | None = None,
) -> dict[str, Any]:
    """Admit a completed NEMO 5.0.2 `ORCA2_SAS_ICE` O1 acquisition directory."""
    set_policy(PrecisionPolicy.fp64())
    ev = Path(evidence_dir)
    manifest_dir = ev / "manifest"
    ref_dir = ev / "ref_2step"
    rec_dir = ev / "rec_2step"
    rst_dir = ev / "rst_1step"

    # 1. Producer commit and provenance manifests (Item 10)
    commit_file = manifest_dir / "producer_commit.txt"
    if not commit_file.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing {commit_file}")
    recorded_commit = commit_file.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", recorded_commit):
        raise ValueError(f"STATUS PLANT-FIRED: malformed producer commit {recorded_commit!r}")
    if plant == "stamp":
        recorded_commit = "0" * 40
    commit_object = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{recorded_commit}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if commit_object.returncode != 0:
        raise ValueError(
            f"STATUS PLANT-FIRED: producer commit {recorded_commit} is not a commit object"
        )
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            recorded_commit,
            "HEAD",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise ValueError(
            f"STATUS PLANT-FIRED: producer commit {recorded_commit} is not an ancestor of HEAD"
        )
    if expect_commit is not None and recorded_commit != expect_commit.strip():
        raise ValueError(
            f"STATUS PLANT-FIRED: producer commit {recorded_commit} != expected {expect_commit.strip()}"
        )

    same_prog_path = manifest_dir / "same_program.json"
    if not same_prog_path.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing {same_prog_path}")
    same_prog_data = json.loads(same_prog_path.read_text(encoding="utf-8"))
    if plant == "same-program":
        same_prog_data["status"] = "FAIL"
    if same_prog_data.get("status") != "PASS_SAME_PROGRAM":
        raise ValueError(
            f"STATUS PLANT-FIRED: same_program.json status={same_prog_data.get('status')!r}"
        )

    binaries_checked = _verify_sha256_manifest(
        manifest_dir / "binaries.sha256",
        label="binaries.sha256",
        plant_mismatch=plant == "binary-sha",
    )
    binary_targets = {
        Path(line.strip().split(None, 1)[1]).resolve()
        for line in (manifest_dir / "binaries.sha256").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    try:
        expected_binary_targets = {
            (Path(same_prog_data["reference_bld"]) / "bin/nemo.exe").resolve(),
            (Path(same_prog_data["new_bld"]) / "bin/nemo.exe").resolve(),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "STATUS PLANT-FIRED: same_program.json lacks reference/new build paths"
        ) from exc
    if binary_targets != expected_binary_targets or binaries_checked != 2:
        raise ValueError(
            "STATUS PLANT-FIRED: binaries.sha256 is not the two same-program nemo.exe files"
        )
    inputs_checked = _verify_sha256_manifest(
        manifest_dir / "inputs.sha256",
        label="inputs.sha256",
        plant_mismatch=plant == "input-sha",
    )

    # The gate is provenance, but is intentionally outside inputs.sha256 so a
    # repaired admission gate can re-admit retained producer evidence.
    gate_manifest = manifest_dir / "gate.sha256"
    gate_changed_since_run = None
    if gate_manifest.is_file():
        gate_parts = gate_manifest.read_text(encoding="utf-8").strip().split(None, 1)
        if len(gate_parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", gate_parts[0]):
            raise ValueError("STATUS PLANT-FIRED: malformed gate.sha256")
        current_gate = Path(gate_parts[1])
        gate_changed_since_run = (
            not current_gate.is_file()
            or hashlib.sha256(current_gate.read_bytes()).hexdigest() != gate_parts[0]
        )

    # 2. Completion markers and log checks on all three runs (ref_2step, rec_2step, rst_1step)
    for rdir, rlabel in (
        (ref_dir, "ref_2step"),
        (rec_dir, "rec_2step"),
        (rst_dir, "rst_1step"),
    ):
        run_stamp = rdir / "RUN_PRODUCER_COMMIT"
        if (
            not run_stamp.is_file()
            or run_stamp.read_text(encoding="utf-8").strip() != recorded_commit
        ):
            raise ValueError(f"STATUS PLANT-FIRED: producer stamp mismatch in {rlabel}")
        _verify_run_completion_and_logs(
            rdir,
            rlabel,
            plant_missing_complete=plant == "missing-complete" and rlabel == "rst_1step",
            plant_missing_stop0=plant == "missing-stop0" and rlabel == "rst_1step",
        )

    # 3. Reference vs instrumented restart byte-identity (steps 1 and 2)
    for step in (1, 2):
        rst_name = f"ORCA2_SAS_{step:08d}_restart_ice.nc"
        ref_rst = ref_dir / rst_name
        rec_rst = rec_dir / rst_name
        if not ref_rst.is_file() or not rec_rst.is_file():
            raise ValueError(f"STATUS PLANT-FIRED: missing restart file {rst_name}")
        ref_payload = ref_rst.read_bytes()
        rec_payload = rec_rst.read_bytes()
        if plant == "ref-rec-diff" and step == 1:
            rec_payload += b"PLANTED_REF_REC_DIFFERENCE"
        if ref_payload != rec_payload:
            raise ValueError(
                f"STATUS PLANT-FIRED: instrumented restart {rst_name} differs from reference"
            )

    # 4. Score rec_2step (entry, step 1, step 2) over wet cells (tmask/umask/vmask)
    rec_records = rec_dir / "o1_si3_records"
    entry_files = sorted(rec_records.glob("o1_si3_state_entry_rank*.bin"))
    step1_files = sorted(rec_records.glob("o1_si3_state_step0001_rank*.bin"))
    step2_files = sorted(rec_records.glob("o1_si3_state_step0002_rank*.bin"))
    if not entry_files or not step1_files or not step2_files:
        raise ValueError("STATUS PLANT-FIRED: missing binary state records in rec_2step")

    snap_entry = stitch_o1_binary_snapshots([read_o1_binary_snapshot(p) for p in entry_files])
    snap_step1 = stitch_o1_binary_snapshots([read_o1_binary_snapshot(p) for p in step1_files])
    snap_step2 = stitch_o1_binary_snapshots([read_o1_binary_snapshot(p) for p in step2_files])

    state_entry, cfg = o1_snapshot_to_si3_state(snap_entry)
    tmask = jnp.asarray(snap_entry["fields"]["tmask"], dtype=jnp.float64)
    umask = jnp.asarray(snap_entry["fields"]["umask"], dtype=jnp.float64)
    vmask = jnp.asarray(snap_entry["fields"]["vmask"], dtype=jnp.float64)
    sss_entry = jnp.asarray(snap_entry["fields"]["sss_m"], dtype=jnp.float64)
    sss_step1_np = np.asarray(snap_step1["fields"]["sss_m"], dtype=np.float64)
    sss_step2_np = np.asarray(snap_step2["fields"]["sss_m"], dtype=np.float64)
    sss_step2 = jnp.asarray(sss_step2_np, dtype=jnp.float64)

    wi, wj = first_wet_ij(tmask, interior_only=True, nn_hls=cfg.nn_hls)
    dry_fixture = verify_dry_signed_zero_fixture(state_entry.v_i, tmask, nn_hls=cfg.nn_hls)

    o1_step_jit = jax.jit(
        lambda step_state, step_tmask, step_sss: si3_o1_step(
            step_state, tmask=step_tmask, sss_m=step_sss, config=cfg
        )
    )
    state_kt1 = o1_step_jit(state_entry, tmask, sss_entry)
    no_ice_wet = (np.asarray(tmask) != 0.0) & np.any(
        np.asarray(state_kt1.v_i) == 0.0, axis=2
    )
    sss_plant_i, sss_plant_j = first_wet_ij(
        no_ice_wet, interior_only=True, nn_hls=cfg.nn_hls
    )
    if plant == "kt2-sss-perturb":
        # icevar.f90:376-389 consumes sss_m only in an ice-free category for
        # nn_icesal=4.  Put the kt=2 value one ULP below the MIN branch point;
        # this must change the scored s_i/sz_i result from rn_simin.
        sss_step2_plant = sss_step2_np.copy()
        branch_point = cfg.rn_simin / cfg.rn_sinew
        sss_step2_plant[sss_plant_i, sss_plant_j] = np.nextafter(branch_point, -np.inf)
        sss_step2 = jnp.asarray(sss_step2_plant, dtype=jnp.float64)
    state_kt2 = o1_step_jit(state_kt1, tmask, sss_step2)

    if plant == "1ulp-state":
        v_np = np.asarray(state_kt1.v_i).copy()
        v_np[wi, wj, 0] = np.nextafter(v_np[wi, wj, 0], np.inf)
        state_kt1 = state_kt1._replace(v_i=jnp.asarray(v_np))

    nemo_kt1, _ = o1_snapshot_to_si3_state(snap_step1, config=cfg)
    nemo_kt2, _ = o1_snapshot_to_si3_state(snap_step2, config=cfg)

    cmp_kt1 = _compare_si3_states_wet_and_full(
        nemo_kt1, state_kt1, tmask, umask, vmask, check_fmass=True, check_prather=True
    )
    cmp_kt2 = _compare_si3_states_wet_and_full(
        nemo_kt2, state_kt2, tmask, umask, vmask, check_fmass=True, check_prather=True
    )

    # 5. Score rst_1step (Item 5 & Item 6): NetCDF restart read + 1-step trajectory vs rec_2step step 2
    rst_records = rst_dir / "o1_si3_records"
    rst_entry_files = sorted(rst_records.glob("o1_si3_state_entry_rank*.bin"))
    rst_step2_pattern = (
        "o1_si3_state_step0001_rank*.bin"
        if plant == "rst-step-name"
        else "o1_si3_state_step0002_rank*.bin"
    )
    rst_step2_files = sorted(rst_records.glob(rst_step2_pattern))
    rst_nc = rst_dir / "output_restart/ORCA2_SAS_00000002_restart_ice_o1_output.nc"
    rst_ocean_nc = rst_dir / "output_restart/ORCA2_SAS_00000002_restart_o1_output.nc"
    rst_input = rst_dir / "input_restart/ORCA2_SAS_00000001_restart_ice.nc"
    rst_ocean_input = rst_dir / "input_restart/ORCA2_SAS_00000001_restart.nc"
    producer_restart = rec_dir / "ORCA2_SAS_00000001_restart_ice.nc"
    producer_ocean_restart = rec_dir / "ORCA2_SAS_00000001_restart.nc"
    producer_ocean_restart_checked = (
        rec_dir / "PLANTED_MISSING_SAS_OCEAN_RESTART.nc"
        if plant == "ocean-restart-missing"
        else producer_ocean_restart
    )
    rst_nc_checked = (
        rst_nc.with_name("PLANTED_MISSING_restart_ice.nc") if plant == "rst-missing" else rst_nc
    )
    if (
        not rst_entry_files
        or not rst_step2_files
        or not rst_nc_checked.is_file()
        or not rst_ocean_nc.is_file()
        or not rst_input.is_file()
        or not rst_ocean_input.is_file()
        or not producer_restart.is_file()
        or not producer_ocean_restart_checked.is_file()
    ):
        raise ValueError(
            "STATUS PLANT-FIRED: rst_1step requires its entry/step0002 records, "
            "the rec_2step ice and SAS ocean restart inputs, and both step-2 outputs"
        )
    if (
        rst_input.resolve() == rst_nc.resolve()
        or rst_input.read_bytes() != producer_restart.read_bytes()
    ):
        raise ValueError(
            "STATUS PLANT-FIRED: restart input/output isolation or copied-input hash failed"
        )
    if (
        rst_ocean_input.resolve() == rst_ocean_nc.resolve()
        or rst_ocean_input.read_bytes() != producer_ocean_restart_checked.read_bytes()
    ):
        raise ValueError(
            "STATUS PLANT-FIRED: ocean restart input/output isolation or copied-input hash failed"
        )

    rst_snap_entry = stitch_o1_binary_snapshots(
        [read_o1_binary_snapshot(p) for p in rst_entry_files]
    )
    rst_snap_step2 = stitch_o1_binary_snapshots(
        [read_o1_binary_snapshot(p) for p in rst_step2_files]
    )
    if int(rst_snap_entry["meta"]["kt"]) != 0 or int(rst_snap_step2["meta"]["kt"]) != 2:
        raise ValueError(
            "STATUS PLANT-FIRED: rst_1step snapshot metadata must be entry kt=0 and step kt=2"
        )
    nemo_rst_entry, _ = o1_snapshot_to_si3_state(rst_snap_entry, config=cfg)
    nemo_rst_step2, _ = o1_snapshot_to_si3_state(rst_snap_step2, config=cfg)
    rst_sss_entry = jnp.asarray(rst_snap_entry["fields"]["sss_m"], dtype=jnp.float64)
    rst_sss_step2 = jnp.asarray(rst_snap_step2["fields"]["sss_m"], dtype=jnp.float64)

    lego_rst_entry = read_si3_netcdf_restart(
        rec_dir / "ORCA2_SAS_00000001_restart_ice.nc",
        tmask=tmask,
        sss_m=rst_sss_entry,
        config=cfg,
        ln_rstart=True,
    )
    lego_rst_step2 = o1_step_jit(lego_rst_entry, tmask, rst_sss_step2)

    if plant == "rst-1ulp":
        v_rst_np = np.asarray(lego_rst_step2.v_i).copy()
        v_rst_np[wi, wj, 0] = np.nextafter(v_rst_np[wi, wj, 0], np.inf)
        lego_rst_step2 = lego_rst_step2._replace(v_i=jnp.asarray(v_rst_np))

    cmp_rst_entry = _compare_si3_states_wet_and_full(
        nemo_rst_entry,
        lego_rst_entry,
        tmask,
        umask,
        vmask,
        check_fmass=True,
        check_prather=True,
    )
    cmp_rst_step2 = _compare_si3_states_wet_and_full(
        nemo_rst_step2,
        lego_rst_step2,
        tmask,
        umask,
        vmask,
        check_fmass=True,
        check_prather=True,
    )
    cmp_rst_vs_kt2 = _compare_si3_states_wet_and_full(
        nemo_kt2,
        lego_rst_step2,
        tmask,
        umask,
        vmask,
        check_fmass=True,
        check_prather=True,
    )

    # Verify serialized NetCDF restart variables match NEMO's ORCA2_SAS_00000001_restart_ice.nc at 0 ULP
    expected_rst_dict = si3_rst_to_netcdf_dict(state_kt1, kt=1, config=cfg)
    with netCDF4.Dataset(rec_dir / "ORCA2_SAS_00000001_restart_ice.nc", "r") as ds:
        for vname, exp_arr in expected_rst_dict.items():
            if vname not in ds.variables:
                raise ValueError(
                    f"STATUS PLANT-FIRED: missing variable {vname} in NEMO restart file"
                )
            raw = np.asarray(ds.variables[vname][...], dtype=np.float64)
            dims = tuple(ds.variables[vname].dimensions)
            if "time_counter" in dims and dims[0] == "time_counter":
                raw = raw[0]
                dims = dims[1:]
            if raw.ndim == 2 and dims == ("y", "x"):
                raw = raw.T
            elif raw.ndim == 3 and dims == ("numcat", "y", "x"):
                raw = np.transpose(raw, (2, 1, 0))
            m_full = (
                wet_mask_for_field(vname, tmask, umask, vmask, nn_hls=cfg.nn_hls)
                if raw.ndim >= 2
                else None
            )
            exp_arr, m = restart_payload_view(
                exp_arr, raw, m_full, nn_hls=cfg.nn_hls
            )
            ulp_nc = max_ulp_diff(exp_arr, raw, mask=m)
            if ulp_nc != 0:
                raise ValueError(
                    f"STATUS PLANT-FIRED: NetCDF restart variable {vname} ULP={ulp_nc} != 0"
                )

    return {
        "status": "PASS",
        "evidence_dir": str(ev),
        "producer_commit": recorded_commit,
        "binaries_checked": binaries_checked,
        "inputs_checked": inputs_checked,
        "gate_changed_since_run": gate_changed_since_run,
        "execution_backend": jax.default_backend(),
        "execution_jit": True,
        "execution_x64": bool(jax.config.jax_enable_x64),
        "first_wet_t_ij": [wi, wj],
        "sss_m_kt1_sha256": hashlib.sha256(sss_step1_np.tobytes()).hexdigest(),
        "sss_m_kt2_sha256": hashlib.sha256(sss_step2_np.tobytes()).hexdigest(),
        "sss_m_kt1_kt2_different_elements": int(
            np.count_nonzero(sss_step1_np.view(np.uint64) != sss_step2_np.view(np.uint64))
        ),
        "sss_m_kt1_kt2_max_abs": float(np.max(np.abs(sss_step1_np - sss_step2_np))),
        "sss_m_kt2_plant_ij": [sss_plant_i, sss_plant_j],
        "sss_m_kt2_plant_branch_point": cfg.rn_simin / cfg.rn_sinew,
        "dry_signed_zero_fixture": dry_fixture,
        "max_ulp_kt1": cmp_kt1["max_ulp_wet"],
        "max_ulp_kt2": cmp_kt2["max_ulp_wet"],
        "max_ulp_rst_entry": cmp_rst_entry["max_ulp_wet"],
        "max_ulp_rst_step2": cmp_rst_step2["max_ulp_wet"],
        "max_ulp_rst_vs_kt2": cmp_rst_vs_kt2["max_ulp_wet"],
        "max_norm_kt1": cmp_kt1["max_norm_wet"],
        "max_norm_kt2": cmp_kt2["max_norm_wet"],
        "max_norm_rst_step2": cmp_rst_step2["max_norm_wet"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SI3 Rung O1 gate and verifier.")
    parser.add_argument("--verify-layout-fixture", type=Path, default=None)
    parser.add_argument("--verify-log", type=Path, default=None)
    parser.add_argument("--verify-deck-fixture", type=Path, default=None)
    parser.add_argument("--expect-icedyn", choices=("T", "F"), default=None)
    parser.add_argument("--expect-icethd", choices=("T", "F"), default=None)
    parser.add_argument("--expect-rstart", choices=("T", "F"), default=None)
    parser.add_argument("--same-program", type=Path, default=None)
    parser.add_argument("--same-source-flags", type=Path, default=None)
    parser.add_argument("--reference-bld", type=Path, default=None)
    parser.add_argument("--pra-patch", type=Path, default=None)
    parser.add_argument("--stp-patch", type=Path, default=None)
    parser.add_argument("--new-cfg", type=str, default="ORCA2_SAS_ICE_O1_REC")
    parser.add_argument("--reference-cfg", type=str, default="ORCA2_SAS_ICE_O1_REF")
    parser.add_argument("--admit-evidence", type=Path, default=None)
    parser.add_argument("--expect-commit", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant", type=str, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if args.verify_layout_fixture is not None:
            report = verify_layout_fixture(args.verify_layout_fixture, plant=args.plant)
        elif args.verify_deck_fixture is not None:
            report = verify_o1_deck_fixture(args.verify_deck_fixture, plant=args.plant)
        elif args.verify_log is not None:
            exp_dyn = None if args.expect_icedyn is None else (args.expect_icedyn == "T")
            exp_thd = None if args.expect_icethd is None else (args.expect_icethd == "T")
            exp_rst = None if args.expect_rstart is None else (args.expect_rstart == "T")
            report = verify_ocean_output_log(
                args.verify_log,
                expect_icedyn=exp_dyn,
                expect_icethd=exp_thd,
                expect_rstart=exp_rst,
                plant=args.plant,
            )
        elif args.same_program is not None or args.same_source_flags is not None:
            if args.reference_bld is None or args.pra_patch is None or args.stp_patch is None:
                parser.error(
                    "same-source/program checks require --reference-bld, --pra-patch, and --stp-patch"
                )
            checker = same_program if args.same_program is not None else same_source_and_flags
            report = checker(
                args.same_program if args.same_program is not None else args.same_source_flags,
                args.reference_bld,
                args.pra_patch,
                args.stp_patch,
                new_cfg=args.new_cfg,
                ref_cfg=args.reference_cfg,
            )
        elif args.admit_evidence is not None:
            report = admit_evidence_directory(
                args.admit_evidence,
                plant=args.plant,
                expect_commit=args.expect_commit,
            )
        elif args.self_test:
            report = run_self_test(plant=args.plant)
        else:
            parser.error(
                "Specify one of --verify-layout-fixture, --verify-log, --same-program, --admit-evidence, or --self-test"
            )
    except Exception as exc:
        print(f"GATE_FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
