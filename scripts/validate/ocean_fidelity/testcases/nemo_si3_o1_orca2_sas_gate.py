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
import json
from pathlib import Path
import re
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
    si3_update_snwice_mass,
    si3_var_agg,
    si3_var_glo2eqv,
    write_si3_netcdf_restart,
)


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
        arr = np.frombuffer(raw, dtype="<f8", count=count, offset=offset).reshape(
            shape, order="F"
        ).copy()
        fields[fname] = arr
        offset += nbytes

    if offset != len(raw):
        raise ValueError(
            f"Trailing bytes in {p}: consumed {offset}, file size {len(raw)}"
        )

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


def max_ulp_diff(a: np.ndarray | jax.Array, b: np.ndarray | jax.Array) -> int:
    """Compute maximum IEEE-754 float64 ULP distance between two arrays."""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if a_np.shape != b_np.shape:
        raise ValueError(f"Shape mismatch: {a_np.shape} vs {b_np.shape}")
    if not (np.all(np.isfinite(a_np)) and np.all(np.isfinite(b_np))):
        return 2**62

    def _to_ordered_int64(x: np.ndarray) -> np.ndarray:
        # Canonicalize -0.0 to +0.0 before bit reinterpretation
        x_clean = np.where(x == 0.0, 0.0, x)
        bits = x_clean.view(np.int64)
        return np.where(bits < 0, np.int64(-9223372036854775808) - bits, bits)

    ia = _to_ordered_int64(a_np)
    ib = _to_ordered_int64(b_np)
    diff = np.abs(ia - ib)
    return int(np.max(diff)) if diff.size > 0 else 0


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
            if np.allclose(
                snaps_entry[r]["fields"]["v_i"], snaps_entry[s]["fields"]["v_i"]
            ):
                raise ValueError(
                    f"STATUS PLANT-FIRED: ranks {r} and {s} have identical v_i data"
                )

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
                raise ValueError(
                    f"STATUS PLANT-FIRED: rank {irank} field {fname} ULP={ulp}"
                )

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

    if re.search(r"E\s+R\s+R\s+O\s+R", text):
        raise ValueError("STATUS PLANT-FIRED: ocean.output contains E R R O R")

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
    return {"status": "PASS", "parsed": parsed}


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
        sal=(
            _rand_moments(s3, 1e-2)
            if cfg.nn_icesal != 4
            else SI3PratherMoments.zeros(s3)
        ),
        ap=_rand_moments(s3, 1e-4),
        vp=_rand_moments(s3, 1e-5),
        vl=_rand_moments(s3, 1e-6),
        c0=_rand_moments(s4_s, 1e5),
        e=_rand_moments(s4_i, 1e6),
        si=(
            _rand_moments(s4_i, 1e-2)
            if cfg.nn_icesal == 4
            else SI3PratherMoments.zeros(s4_i)
        ),
    )

    cnd_ice = jnp.zeros(s3, dtype=jnp.float64)
    t1_ice = jnp.full(s3, cfg.ice_constants.T0, dtype=jnp.float64)
    t_si = jnp.zeros(s3, dtype=jnp.float64)
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


def run_self_test(plant: str | None = None) -> dict[str, Any]:
    """Run the full O1 gate self-test and negative-control plants."""
    set_policy(PrecisionPolicy.fp64())
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state0, tmask, _umask, _vmask = build_synthetic_orca2_o1_state(cfg)
    state1 = si3_o1_step(state0, tmask=tmask, config=cfg)

    with tempfile.TemporaryDirectory() as tmp:
        rst_path = Path(tmp) / "ORCA2_SAS_00000001_restart_ice.nc"
        write_si3_netcdf_restart(rst_path, state1, kt=1, config=cfg)
        state_reloaded_init = read_si3_netcdf_restart(
            rst_path, tmask=tmask, config=cfg
        )
        state_reloaded = si3_o1_step(state_reloaded_init, tmask=tmask, config=cfg)

    if plant == "1ulp-state":
        v_i_np = np.asarray(state_reloaded.v_i).copy()
        v_i_np[50, 50, 0] = np.nextafter(v_i_np[50, 50, 0], np.inf)
        state_reloaded = state_reloaded._replace(v_i=jnp.asarray(v_i_np))
    elif plant == "1ulp-moment":
        sx_np = np.asarray(state_reloaded.prather.ice.sx).copy()
        sx_np[50, 50, 0] = np.nextafter(sx_np[50, 50, 0], np.inf)
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
        hm_bad = np.asarray(state_reloaded.aggregated.hm_i).copy() + 1e-6
        state_reloaded = state_reloaded._replace(
            aggregated=state_reloaded.aggregated._replace(hm_i=jnp.asarray(hm_bad))
        )

    # Check 0-ULP round-trip on all prognostic, Prather, equivalent, and aggregated fields
    prog_fields = (
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
    )
    for fname in prog_fields:
        ulp = max_ulp_diff(getattr(state1, fname), getattr(state_reloaded, fname))
        if ulp != 0:
            raise ValueError(
                f"STATUS PLANT-FIRED: prognostic field {fname} ULP={ulp} != 0"
            )

    for group_name in SI3AllPratherMoments._fields:
        grp1 = getattr(state1.prather, group_name)
        grp2 = getattr(state_reloaded.prather, group_name)
        for mname in SI3PratherMoments._fields:
            ulp = max_ulp_diff(getattr(grp1, mname), getattr(grp2, mname))
            if ulp != 0:
                raise ValueError(
                    f"STATUS PLANT-FIRED: Prather moment {group_name}.{mname} ULP={ulp} != 0"
                )

    for eq_name in state1.equivalent._fields:
        ulp = max_ulp_diff(
            getattr(state1.equivalent, eq_name),
            getattr(state_reloaded.equivalent, eq_name),
        )
        if ulp != 0:
            raise ValueError(
                f"STATUS PLANT-FIRED: equivalent field {eq_name} ULP={ulp} != 0"
            )

    for agg_name in state1.aggregated._fields:
        ulp = max_ulp_diff(
            getattr(state1.aggregated, agg_name),
            getattr(state_reloaded.aggregated, agg_name),
        )
        if ulp != 0:
            raise ValueError(
                f"STATUS PLANT-FIRED: aggregated field {agg_name} ULP={ulp} != 0"
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
        "prognostic_fields_checked": len(prog_fields),
        "prather_arrays_checked": 55,
        "equivalent_fields_checked": len(state1.equivalent._fields),
        "aggregated_fields_checked": len(state1.aggregated._fields),
        "max_ulp": 0,
        "ad_grad_norm": grad_norm,
    }


def admit_evidence_directory(
    evidence_dir: str | Path,
    plant: str | None = None,
) -> dict[str, Any]:
    """Admit a completed NEMO 5.0.2 `ORCA2_SAS_ICE` O1 acquisition directory."""
    set_policy(PrecisionPolicy.fp64())
    ev = Path(evidence_dir)
    ref_dir = ev / "ref_2step"
    rec_dir = ev / "rec_2step"
    rst_dir = ev / "rst_1step"

    for step in (1, 2):
        rst_name = f"ORCA2_SAS_{step:08d}_restart_ice.nc"
        ref_rst = ref_dir / rst_name
        rec_rst = rec_dir / rst_name
        if not ref_rst.is_file() or not rec_rst.is_file():
            raise ValueError(f"STATUS PLANT-FIRED: missing restart file {rst_name}")
        if ref_rst.read_bytes() != rec_rst.read_bytes():
            raise ValueError(
                f"STATUS PLANT-FIRED: instrumented restart {rst_name} differs from reference"
            )

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

    state_kt1 = si3_o1_step(state_entry, tmask=tmask, config=cfg)
    state_kt2 = si3_o1_step(state_kt1, tmask=tmask, config=cfg)

    if plant == "1ulp-state":
        v_np = np.asarray(state_kt1.v_i).copy()
        v_np[90, 140, 0] = np.nextafter(v_np[90, 140, 0], np.inf)
        state_kt1 = state_kt1._replace(v_i=jnp.asarray(v_np))

    nemo_kt1, _ = o1_snapshot_to_si3_state(snap_step1, config=cfg)
    nemo_kt2, _ = o1_snapshot_to_si3_state(snap_step2, config=cfg)

    max_ulp_kt1 = max(
        max_ulp_diff(getattr(state_kt1, f), getattr(nemo_kt1, f))
        for f in ("v_i", "v_s", "sv_i", "a_i", "t_su", "u_ice", "v_ice", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i", "snwice_mass", "snwice_mass_b", "snwice_fmass")
    )
    max_ulp_kt2 = max(
        max_ulp_diff(getattr(state_kt2, f), getattr(nemo_kt2, f))
        for f in ("v_i", "v_s", "sv_i", "a_i", "t_su", "u_ice", "v_ice", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i", "snwice_mass", "snwice_mass_b", "snwice_fmass")
    )
    if max_ulp_kt1 != 0 or max_ulp_kt2 != 0:
        raise ValueError(
            f"STATUS PLANT-FIRED: trajectory ULP mismatch kt1={max_ulp_kt1}, kt2={max_ulp_kt2}"
        )

    return {
        "status": "PASS",
        "evidence_dir": str(ev),
        "max_ulp_kt1": max_ulp_kt1,
        "max_ulp_kt2": max_ulp_kt2,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SI3 Rung O1 gate and verifier.")
    parser.add_argument("--verify-layout-fixture", type=Path, default=None)
    parser.add_argument("--verify-log", type=Path, default=None)
    parser.add_argument("--expect-icedyn", choices=("T", "F"), default=None)
    parser.add_argument("--expect-icethd", choices=("T", "F"), default=None)
    parser.add_argument("--admit-evidence", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant", type=str, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if args.verify_layout_fixture is not None:
            report = verify_layout_fixture(args.verify_layout_fixture, plant=args.plant)
        elif args.verify_log is not None:
            exp_dyn = None if args.expect_icedyn is None else (args.expect_icedyn == "T")
            exp_thd = None if args.expect_icethd is None else (args.expect_icethd == "T")
            report = verify_ocean_output_log(
                args.verify_log,
                expect_icedyn=exp_dyn,
                expect_icethd=exp_thd,
                plant=args.plant,
            )
        elif args.admit_evidence is not None:
            report = admit_evidence_directory(args.admit_evidence, plant=args.plant)
        elif args.self_test:
            report = run_self_test(plant=args.plant)
        else:
            parser.error(
                "Specify one of --verify-layout-fixture, --verify-log, --admit-evidence, or --self-test"
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
