#!/usr/bin/env python3
"""Round-46 GYRE kt=2, stage-by-stage momentum fidelity gate.

The acquisition phase uses ``--mode validate``.  It parses all six named,
ranked streams through physical EOF, replays compiled WZV/KEG/ZAD arithmetic,
checks the widened round-40/41 kt=1 twins, and verifies the producer SHA.  The
operator resumes with ``--mode all`` for NEMO-given-input and kt=2 trajectory
scores.  A non-executed operator is represented by a header presence flag,
never by an all-zero pseudo-tendency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    STAGE_WW_ROOT,
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_entry,
    read_stage_ww,
    read_transport,
    require,
    score,
)
from nemo_testcase_l2_gyre_round40_stage3_operators import read_stage3_terms
from nemo_testcase_l2_gyre_round41_dynadv_split import (
    _keg_replay,
    _model_terms,
    _zad_replay,
    read_split,
)
from nemo_testcase_l2_gyre_round21_admission import _compare_self_describing

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage")
ROUND41 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round41/oracle_dynadv_split")
ADVMEAN_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round75/oracle_advmean_kt2")
MEMORY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round48/oracle_bt_memory")
BTSTEP_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round81/oracle_btstep_kt2")
STAGE_CLOSURE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round94/oracle_stage_closure")
MAGIC = "NEMO_L2_R46STG1"
DIMS = (36, 26, 31)
OWNED_DIMS = (32, 22, 31)
OWNED_3D_FIELDS = {"tke_en", "tke_avt_k", "tke_dissl"}
HEADER_FIELDS = (
    "version",
    "kt",
    "stage",
    "Kbb",
    "Kmm",
    "Krhs",
    "Kaa",
    "jpi",
    "jpj",
    "jpk",
    "jpkm1",
    "ntsi",
    "ntei",
    "ntsj",
    "ntej",
    "bits",
)
STAGES = tuple((kt, stage) for kt in (1, 2) for stage in (1, 2, 3))
PRESENCE = {
    1: {"hpg": 1, "vor": 1, "keg": 1, "zad": 1, "ldf": 1, "zdf": 0},
    2: {"hpg": 1, "vor": 1, "keg": 1, "zad": 1, "ldf": 0, "zdf": 0},
    3: {"hpg": 1, "vor": 1, "keg": 1, "zad": 1, "ldf": 1, "zdf": 1},
}
REQUIRED = {
    "u_Kbb",
    "v_Kbb",
    "u_Kmm",
    "v_Kmm",
    "u_Kaa_in",
    "v_Kaa_in",
    "T_Kbb",
    "S_Kbb",
    "ssh_Kbb",
    "T_Kmm",
    "S_Kmm",
    "ssh_Kmm",
    "T_Kaa_in",
    "S_Kaa_in",
    "ssh_Kaa",
    "rhd_in",
    "r1_Dt",
    "tke_en",
    "tke_avm_k",
    "tke_avt_k",
    "tke_dissl",
    "r3t_Kbb",
    "r3u_Kbb",
    "r3v_Kbb",
    "r3t_Kmm",
    "r3u_Kmm",
    "r3v_Kmm",
    "r3t_Kaa",
    "r3u_Kaa",
    "r3v_Kaa",
    "e3t_Kbb",
    "e3u_Kbb",
    "e3v_Kbb",
    "e3w_Kbb",
    "e3t_Kmm",
    "e3u_Kmm",
    "e3v_Kmm",
    "e3w_Kmm",
    "e3t_Kaa",
    "e3u_Kaa",
    "e3v_Kaa",
    "e3w_Kaa",
    "e3t_0",
    "e3u_0",
    "e3v_0",
    "e3w_0",
    "umask",
    "vmask",
    "tmask",
    "wmask",
    "e1e2t",
    "e1e2u",
    "e1e2v",
    "r1_e1e2u",
    "r1_e1e2v",
    "r1_e1u",
    "r1_e2v",
    "r1_e1e2t",
    "e2u",
    "e1v",
    "uu_b_Kbb",
    "vv_b_Kbb",
    "uu_b_Kmm",
    "vv_b_Kmm",
    "uu_b_Kaa",
    "vv_b_Kaa",
    "rhs_entry_u",
    "rhs_entry_v",
    "ww",
    "wsd_effective",
    "after_hpg_u",
    "after_hpg_v",
    "after_vor_u",
    "after_vor_v",
    "after_keg_u",
    "after_keg_v",
    "after_zad_u",
    "after_zad_v",
    "after_adv_u",
    "after_adv_v",
    "post_baro_u",
    "post_baro_v",
    "has_hpg",
    "has_vor",
    "has_keg",
    "has_zad",
    "has_ldf",
    "has_zdf",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _xy(raw: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return raw.reshape((nx, ny), order="F").T.copy()


def _xyz(raw: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return raw.reshape((nx, ny, nz), order="F").transpose(1, 0, 2).copy()


def _owned3(value, nlev: int = 30) -> np.ndarray:
    return np.asarray(value)[2:-2, 2:-2, :nlev]


def _owned2(value) -> np.ndarray:
    return np.asarray(value)[2:-2, 2:-2]


def read_stage(path: Path, *, plant: str | None = None) -> dict:
    """Fail-closed named/ranked reader; duplicate, extra and short fields fail."""
    with path.open("rb") as f:
        raw_magic = f.read(16)
        require(len(raw_magic) == 16, f"{path}: short magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = f.read(64)
        require(len(raw_header) == 64, f"{path}: short header")
        values = list(struct.unpack("=16i", raw_header))
        if plant == "header":
            values[2] = 9
        header = dict(zip(HEADER_FIELDS, values, strict=True))
        require(magic == MAGIC, f"{path}: bad magic {magic!r}")
        require(
            header["version"] == 1 and header["bits"] == 64, f"{path}: wrong version/dtype {header}"
        )
        require(
            (header["jpi"], header["jpj"], header["jpk"], header["jpkm1"]) == (*DIMS, 30),
            f"{path}: wrong domain {header}",
        )
        require(
            (header["ntsi"], header["ntei"], header["ntsj"], header["ntej"]) == (3, 34, 3, 24),
            f"{path}: wrong owned bounds {header}",
        )
        require((header["kt"], header["stage"]) in STAGES, f"{path}: wrong kt/stage {header}")
        arrays: dict[str, np.ndarray | float] = {}
        while True:
            raw_name = f.read(16)
            if not raw_name:
                break
            require(len(raw_name) == 16, f"{path}: truncated field name")
            name = raw_name.decode("ascii").rstrip()
            require(name not in arrays, f"{path}: duplicate field {name!r}")
            shape = f.read(16)
            require(len(shape) == 16, f"{path}: short shape for {name}")
            rank, n1, n2, n3 = struct.unpack("=4i", shape)
            require(rank in (0, 2, 3), f"{path}: invalid rank for {name}")
            count = 1 if rank == 0 else n1 * n2 * (n3 if rank == 3 else 1)
            raw = f.read(8 * count)
            if plant == "truncation" and name == "u_Kbb":
                raw = raw[:-8]
            require(len(raw) == 8 * count, f"{path}: short payload for {name}")
            a = np.frombuffer(raw, dtype=np.float64)
            require(np.isfinite(a).all(), f"{path}: non-finite {name}")
            if rank == 0:
                arrays[name] = float(a[0])
            elif rank == 2:
                require((n1, n2, n3) == (*DIMS[:2], 1), f"{path}: bad 2-D extents for {name}")
                arrays[name] = _xy(a, n1, n2)
            else:
                expected = OWNED_DIMS if name in OWNED_3D_FIELDS else DIMS
                require((n1, n2, n3) == expected,
                        f"{path}: bad 3-D extents for {name}: "
                        f"{(n1, n2, n3)} != {expected}")
                arrays[name] = _xyz(a, n1, n2, n3)
    missing = REQUIRED - arrays.keys()
    # Stage-specific fields are additive to the common contract.
    if header["stage"] == 3:
        missing |= {
            "after_ldf_u",
            "after_ldf_v",
            "pre_zdf_rhs_u",
            "pre_zdf_rhs_v",
            "post_zdf_u",
            "post_zdf_v",
        } - arrays.keys()
    else:
        missing |= {"post_update_u", "post_update_v"} - arrays.keys()
    require(not missing, f"{path}: missing fields {sorted(missing)}")
    allowed = set(REQUIRED) | {
        "after_ldf_u",
        "after_ldf_v",
        "pre_zdf_rhs_u",
        "pre_zdf_rhs_v",
        "post_zdf_u",
        "post_zdf_v",
        "post_update_u",
        "post_update_v",
        "pre_baro_u",
        "pre_baro_v",
    }
    require(
        not (arrays.keys() - allowed), f"{path}: unknown fields {sorted(arrays.keys() - allowed)}"
    )
    for op, expected in PRESENCE[header["stage"]].items():
        require(
            arrays[f"has_{op}"] == float(expected), f"{path}: false operator-presence flag for {op}"
        )
    require(
        np.count_nonzero(arrays["wsd_effective"]) == 0, f"{path}: GYRE wsd arm is not resolved zero"
    )
    return {"header": header, "arrays": arrays}


def _wzv_replay(a: dict) -> np.ndarray:
    """Compiled divhor.f90:123-154 + sshwzv.f90:293-299 replay."""
    u, v = a["u_Kmm"], a["v_Kmm"]
    out = np.zeros(DIMS[::-1], dtype=np.float64).transpose(1, 0, 2)
    # out is (ny,nx,nz); pww bottom was initialized to zero by NEMO.
    out = np.zeros((DIMS[1], DIMS[0], DIMS[2]), dtype=np.float64)
    ze3div = np.zeros_like(out)
    for k in range(DIMS[2] - 1):
        for j in range(1, 25):
            for i in range(1, 35):
                e2u, e2uw = a["e2u"][j, i], a["e2u"][j, i - 1]
                e1v, e1vs = a["e1v"][j, i], a["e1v"][j - 1, i]
                zu = np.float64(e2u * a["e3u_Kmm"][j, i, k])
                zu = np.float64(zu * u[j, i, k])
                zuw = np.float64(e2uw * a["e3u_Kmm"][j, i - 1, k])
                zuw = np.float64(zuw * u[j, i - 1, k])
                zv = np.float64(e1v * a["e3v_Kmm"][j, i, k])
                zv = np.float64(zv * v[j, i, k])
                zvs = np.float64(e1vs * a["e3v_Kmm"][j - 1, i, k])
                zvs = np.float64(zvs * v[j - 1, i, k])
                hdiv = np.float64(np.float64(zu - zuw) + np.float64(zv - zvs))
                hdiv = np.float64(hdiv * a["r1_e1e2t"][j, i])
                hdiv = np.float64(hdiv / a["e3t_Kmm"][j, i, k])
                ze3div[j, i, k] = np.float64(hdiv * a["e3t_Kmm"][j, i, k])
    for k in range(DIMS[2] - 2, -1, -1):
        for j in range(1, 25):
            for i in range(1, 35):
                stretch = np.float64(a["r3t_Kaa"][j, i] - a["r3t_Kbb"][j, i])
                stretch = np.float64(a["r1_Dt"] * a["e3t_0"][j, i, k] * stretch)
                total = np.float64(ze3div[j, i, k] + stretch)
                out[j, i, k] = np.float64(out[j, i, k + 1] - total) * a["tmask"][j, i, k]
    return out


def _split_view(a: dict) -> dict:
    return {
        "uu_Kmm": a["u_Kmm"],
        "vv_Kmm": a["v_Kmm"],
        "before_keg_u": a["after_vor_u"],
        "before_keg_v": a["after_vor_v"],
        "after_keg_u": a["after_keg_u"],
        "after_keg_v": a["after_keg_v"],
        "after_zad_u": a["after_zad_u"],
        "after_zad_v": a["after_zad_v"],
        **{
            k: a[k]
            for k in (
                "ww",
                "e3t_Kmm",
                "e3u_Kmm",
                "e3v_Kmm",
                "e3w_Kmm",
                "e3t_0",
                "e3u_0",
                "e3v_0",
                "e3w_0",
                "e1e2t",
                "e1e2u",
                "e1e2v",
                "r1_e1u",
                "r1_e2v",
                "r1_e1e2u",
                "r1_e1e2v",
                "tmask",
                "umask",
                "vmask",
                "wmask",
            )
        },
    }


def _calibrate(records: dict, plant: str | None) -> dict:
    rows = {}
    for key, record in records.items():
        a = record["arrays"]
        wzv_inputs = a
        if key == (2, 1):
            # stp2d.f90:157-162 updates r3t(Kaa)=ssh(Kaa)*r1_ht_0 after
            # r46_begin recorded the named bundle and immediately before WZV.
            # The same bundle carries the developed Kbb identity
            # r3t(Kbb)=ssh(Kbb)*r1_ht_0, so recover NEMO's stored reciprocal
            # from those two operands and replay the missing statement.  Every
            # wet WZV cell has nonzero developed ssh; dry cells are masked out
            # by sshwzv.f90:297-298.
            wet = a["tmask"][..., 0] > 0.5
            ext2 = np.zeros_like(wet)
            ext2[1:25, 1:35] = True
            require(np.all(a["ssh_Kbb"][wet & ext2] != 0.0),
                    "kt2 stage1 cannot recover stored r1_ht_0 from zero ssh_Kbb")
            r1_ht_0 = np.divide(
                a["r3t_Kbb"], a["ssh_Kbb"],
                out=np.zeros_like(a["r3t_Kbb"]),
                where=a["ssh_Kbb"] != 0.0,
            )
            r3t_kaa = a["ssh_Kaa"] * r1_ht_0
            rows["kt2.s1.r3t_Kaa_replay"] = int(np.count_nonzero(
                (a["ssh_Kbb"] * r1_ht_0)[wet & ext2]
                != a["r3t_Kbb"][wet & ext2]
            ))
            wzv_inputs = {**a, "r3t_Kaa": r3t_kaa}
        ww = _wzv_replay(wzv_inputs)
        keg_u, keg_v = _keg_replay(_split_view(a))
        zad_u, zad_v = _zad_replay(_split_view(a))
        if plant == "calibration" and key == (2, 1):
            keg_u[3, 3, 0] = np.nextafter(keg_u[3, 3, 0], np.inf)
        ext = np.s_[1:25, 1:35, :]
        rows[f"kt{key[0]}.s{key[1]}.ww"] = int(np.count_nonzero(ww[ext] != a["ww"][ext]))
        rows[f"kt{key[0]}.s{key[1]}.keg_u"] = int(np.count_nonzero(keg_u != a["after_keg_u"]))
        rows[f"kt{key[0]}.s{key[1]}.keg_v"] = int(np.count_nonzero(keg_v != a["after_keg_v"]))
        rows[f"kt{key[0]}.s{key[1]}.zad_u"] = int(np.count_nonzero(zad_u != a["after_zad_u"]))
        rows[f"kt{key[0]}.s{key[1]}.zad_v"] = int(np.count_nonzero(zad_v != a["after_zad_v"]))
    require(all(v == 0 for v in rows.values()), f"source calibration failed: {rows}")
    return rows


def _given_inputs(
    records: dict,
    plant: str | None,
    *,
    kt: int = 2,
    stages: tuple[int, ...] = (1, 2, 3),
) -> tuple[list[dict], dict]:
    """Run legoESM operator routes on complete NEMO stage bundles."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    ocean = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    rows = []
    for stage in stages:
        a = records[(kt, stage)]["arrays"]
        un, vn = _owned3(a["u_Kmm"]), _owned3(a["v_Kmm"])
        ju = np.concatenate([un[:, -1:, :], un], axis=1)
        jv = np.concatenate([np.zeros_like(vn[:1]), vn], axis=0)
        ub, vb = _owned3(a["u_Kbb"]), _owned3(a["v_Kbb"])
        ju_b = np.concatenate([ub[:, -1:, :], ub], axis=1)
        jv_b = np.concatenate([np.zeros_like(vb[:1]), vb], axis=0)
        st = card.recipe.initial_state._replace(
            u=card.recipe.initial_state.u.replace(data=jnp.asarray(ju)),
            v=card.recipe.initial_state.v.replace(data=jnp.asarray(jv)),
            T=card.recipe.initial_state.T.replace(data=jnp.asarray(_owned3(a["T_Kmm"]))),
            S=card.recipe.initial_state.S.replace(data=jnp.asarray(_owned3(a["S_Kmm"]))),
            eta=card.recipe.initial_state.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kmm"]))),
        )
        _, surface = _surface_forcings(card, st, kt)
        hu0, hv0 = _owned3(a["e3u_Kmm"]), _owned3(a["e3v_Kmm"])
        hu = np.concatenate([hu0[:, -1:, :], hu0], axis=1)
        hv = np.concatenate([np.ones_like(hv0[:1]), hv0], axis=0)

        def operators(state):
            return ocean.tendencies(
                state,
                surface,
                dt=card.dt_s,
                momentum_only=True,
                skip_lateral_viscosity=(stage == 2),
                # Compiled dyn_ldf reads Kbb at every stage, whereas the rest
                # of the RHS reads this stage's Kmm state.
                ldf_state=(state.T.data, state.S.data,
                           jnp.asarray(ju_b), jnp.asarray(jv_b)),
                momentum_flux_face_thickness=(jnp.asarray(hu), jnp.asarray(hv)),
                zad_continuity_dt=np.float64(1.0 / a["r1_Dt"]),
                nemo_operator_association=True,
                return_nemo_operator_components=True,
            )

        _, diagnostics, components = jax.jit(operators)(st)
        vor_before_u = a["after_ldf_u"] if stage == 1 else a["after_hpg_u"]
        vor_before_v = a["after_ldf_v"] if stage == 1 else a["after_hpg_v"]
        references = {
            "hpg": (a["after_hpg_u"], a["after_hpg_v"]),
            "vorticity": (a["after_vor_u"] - vor_before_u, a["after_vor_v"] - vor_before_v),
            "advection": (a["after_adv_u"] - a["after_vor_u"], a["after_adv_v"] - a["after_vor_v"]),
        }
        for op, pair in references.items():
            for face, ref_full in zip(("u", "v"), pair, strict=True):
                value = np.asarray(components[f"{op}_{face}"].data)
                got = value[:, 1:, :] if face == "u" else value[1:, :, :]
                ref = _owned3(ref_full)
                mask = _owned3(a[f"{face}mask"]) > 0.5
                rows.append(
                    {
                        "name": f"GYRE-zco.kt{kt}.s{stage}.{op}.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                        "reference_max_abs": float(np.max(np.abs(ref[mask]))),
                        "model_max_abs": float(np.max(np.abs(got[mask]))),
                        "execution": "production-jit model component from NEMO stage inputs",
                    }
                )
        # LDF is exposed by the ordinary momentum diagnostics and exists only
        # in stages 1 and 3.  ZDF is an update/solve, scored in trajectory.
        if stage in (1, 3):
            prior = a["after_hpg_u"] if stage == 1 else a["after_adv_u"]
            prior_v = a["after_hpg_v"] if stage == 1 else a["after_adv_v"]
            for face, prior_full in (("u", prior), ("v", prior_v)):
                value = sum(
                    np.asarray(getattr(diagnostics, f"{name}_{face}").data)
                    for name in ("Ah_lap", "Bh_bilap", "Cs_smag", "Cl_leith")
                )
                got = value[:, 1:, :] if face == "u" else value[1:, :, :]
                ref = _owned3(a[f"after_ldf_{face}"] - prior_full)
                mask = _owned3(a[f"{face}mask"]) > 0.5
                rows.append(
                    {
                        "name": f"GYRE-zco.kt{kt}.s{stage}.ldf.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                        "reference_max_abs": float(np.max(np.abs(ref[mask]))),
                        "model_max_abs": float(np.max(np.abs(got[mask]))),
                        "execution": "production-jit model diagnostic from NEMO stage inputs",
                    }
                )
        # Full-accumulator score through the production JIT path. Stages 2/3
        # have the same source order as the shared model route. Stage 1 is the
        # compiled stp2d exception (HPG -> LDF -> VOR -> KEG -> ZAD), so rebuild
        # only its accumulator association from the already-computed production
        # arrays inside a second JIT with an explicit barrier at every routine
        # boundary. This is neither an oracle replay nor a tendency subtraction.
        if stage == 1:
            def stage1_accumulators(parts):
                hpg_u = parts["after_hpg_u"].data
                hpg_v = parts["after_hpg_v"].data
                ldf_u = jax.lax.optimization_barrier(hpg_u + parts["ldf_u"].data)
                ldf_v = jax.lax.optimization_barrier(hpg_v + parts["ldf_v"].data)
                vor_u = jax.lax.optimization_barrier(ldf_u + parts["vorticity_u"].data)
                vor_v = jax.lax.optimization_barrier(ldf_v + parts["vorticity_v"].data)
                keg_u = jax.lax.optimization_barrier(vor_u + parts["keg_u"].data)
                keg_v = jax.lax.optimization_barrier(vor_v + parts["keg_v"].data)
                adv_u = jax.lax.optimization_barrier(keg_u + parts["zad_u"].data)
                adv_v = jax.lax.optimization_barrier(keg_v + parts["zad_v"].data)
                return {"hpg_u": hpg_u, "hpg_v": hpg_v,
                        "ldf_u": ldf_u, "ldf_v": ldf_v,
                        "vor_u": vor_u, "vor_v": vor_v,
                        "adv_u": adv_u, "adv_v": adv_v}
            accumulators = jax.jit(stage1_accumulators)(components)
            boundaries = ("hpg", "ldf", "vor", "adv")
            execution = "production-jit components in compiled stage-1 accumulator order"
        else:
            accumulators = {
                "hpg_u": components["after_hpg_u"].data,
                "hpg_v": components["after_hpg_v"].data,
                "vor_u": components["after_vor_u"].data,
                "vor_v": components["after_vor_v"].data,
                "adv_u": components["after_adv_u"].data,
                "adv_v": components["after_adv_v"].data,
                "ldf_u": components["after_ldf_u"].data,
                "ldf_v": components["after_ldf_v"].data,
            }
            boundaries = ("hpg", "vor", "adv", "ldf") if stage == 3 else (
                "hpg", "vor", "adv")
            execution = "production-jit accumulator at model routine barrier"
        for op in boundaries:
            for face in ("u", "v"):
                value = np.asarray(accumulators[f"{op}_{face}"])
                got = value[:, 1:, :] if face == "u" else value[1:, :, :]
                ref = _owned3(a[f"after_{op}_{face}"])
                mask = _owned3(a[f"{face}mask"]) > 0.5
                if plant == "given" and (stage, op, face) == (1, "hpg", "u"):
                    got = got.copy()
                    got[tuple(np.argwhere(mask)[0])] += 1.0
                if (plant == "stage-rhs-ulp"
                        and (stage, op, face) == (1, "hpg", "u")):
                    got = got.copy()
                    index = tuple(np.argwhere(mask)[0])
                    got[index] = np.nextafter(got[index], np.float64(np.inf))
                rows.append({
                    "name": f"GYRE-zco.kt{kt}.s{stage}.post_{op}_accumulator.{face}",
                    "n": int(mask.sum()),
                    "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                    "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                    "execution": execution,
                })
        # Separate accumulator-level KEG and explicit ZAD scores retain the
        # exact NEMO ww injection; the combined advection component above uses
        # the model's own WZV route and therefore diagnoses that seam too.
        model = _model_terms(_split_view(a))
        for op in ("keg", "zad"):
            for face in ("u", "v"):
                ref = a[f"after_{op}_{face}"][2:-2, 2:-2, :30]
                got = np.asarray(model[f"after_{op}_{face}"])
                mask = a[f"{face}mask"][2:-2, 2:-2, :30] > 0.5
                if plant == "given" and (stage, op, face) == (1, "zad", "u"):
                    got = got.copy()
                    got[tuple(np.argwhere(mask)[0])] += 1.0
                rows.append(
                    {
                        "name": f"GYRE-zco.kt{kt}.s{stage}.{op}_accumulator.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                    }
                )
    if plant == "given":
        require(any(r["max_abs"] > 0.5 for r in rows), "given-input plant did not land")
    if plant == "stage-rhs-ulp":
        planted = next(
            row for row in rows
            if row["name"] == "GYRE-zco.kt1.s1.post_hpg_accumulator.u")
        require(planted["n_unequal"] == 1,
                "one-ULP exact-accumulator plant did not flip exactly one cell")
    first = {}
    # Historical audit only: rounds 49/50 changed VOR/LDF after this round-48
    # prediction was registered.  Retain the discriminating measurement and
    # label the old prediction; never make a current gate fail on a retracted
    # owner expectation.
    expected = {1: "ldf", 2: "vor", 3: "vor"} if kt == 2 else {}
    for stage in stages:
        order = ("hpg", "ldf", "vor", "adv") if stage == 1 else (
            ("hpg", "vor", "adv", "ldf") if stage == 3 else ("hpg", "vor", "adv")
        )
        measured = None
        counts = None
        for op in order:
            pair = [
                row for row in rows
                if row["name"] in {
                    f"GYRE-zco.kt{kt}.s{stage}.post_{op}_accumulator.u",
                    f"GYRE-zco.kt{kt}.s{stage}.post_{op}_accumulator.v",
                }
            ]
            require(len(pair) == 2, f"missing model-path boundary kt{kt} stage {stage} {op}")
            by_face = {row["name"].rsplit(".", 1)[-1]: row for row in pair}
            if any(row["n_unequal"] for row in pair):
                measured = op
                counts = {face: by_face[face]["n_unequal"] for face in ("u", "v")}
                break
        first[f"stage{stage}"] = {
            "operator": measured,
            "n_unequal": counts,
            "preregistered_operator": expected.get(stage),
            "prediction": (
                "CONFIRMED" if measured == expected.get(stage) else
                "REFUTED" if stage in expected else
                "ROUND96_MEASUREMENT"
            ),
            "interpretation": (
                "POSTHOC_AFTER_ROUND49_ROUND50" if kt == 2 else
                "ROUND96_PREREGISTERED_COMPILED_ORDER"
            ),
        }
    return rows, first


def _trajectory(records: dict, plant: str | None) -> list[dict]:
    """Advance the exact kt=1 card state, then score kt=2 stage states."""
    # The widened record closes every named 3-D/TKE field, but not NEMO's
    # persistent dynspg_ts AB3/AM4 substep history.  Seeding that history from
    # an independent legoESM kt=1 step moved every stage globally even after
    # all recorded entry fields were injected bit-exactly.  Refuse to print
    # those contaminated numbers as a stage verdict; a future acquisition must
    # add the six ubb_e/ub_e/vbb_e/vb_e/sshbb_e/sshb_e endpoint arrays.
    if plant == "trajectory":
        require(False, "trajectory plant: missing-history refusal fired")
    return [{
        "name": "GYRE-zco.kt1_end_to_kt2_stage_trajectory",
        "status": "UNMEASURED",
        "reason": "NEMO kt=1-end cross-window barotropic history was not recorded",
        "required_fields": [
            "ubb_e", "ub_e", "vbb_e", "vb_e", "sshbb_e", "sshb_e"
        ],
    }]

    # Retained below as the preregistered experimental arm; unreachable until
    # the required record exists and the refusal above is replaced by a parser.
    import jax
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "trajectory requires x64")
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    a = records[(2, 1)]["arrays"]
    # Independently advancing kt=1 preserves every hidden prognostic and TKE
    # history slot.  Reconstructing only named fields from the initial state
    # silently seeded different viscosity and produced a false all-cell miss.
    st = card.recipe.initial_state
    freshwater1, surface1 = _surface_forcings(card, st, 1)
    st = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg
    ).step(st, dt=card.dt_s, freshwater=freshwater1, surface_forcing=surface1)
    # Replace every recorded kt=1 endpoint operand while retaining the
    # otherwise-unrecorded history slots from the independently advanced
    # legoESM step.  This is the consumed-field bridge: no NEMO routine is
    # called, and the entry rows below prove the fields used by kt=2 are exact.
    u0, v0 = _owned3(a["u_Kbb"]), _owned3(a["v_Kbb"])
    ub0, vb0 = _owned2(a["uu_b_Kbb"]), _owned2(a["vv_b_Kbb"])

    def field(data, name, dims):
        return Field(data=jnp.asarray(data), name=name, dims=dims)

    st = st._replace(
        u=st.u.replace(data=jnp.asarray(np.concatenate([u0[:, -1:, :], u0], axis=1))),
        v=st.v.replace(data=jnp.asarray(np.concatenate([np.zeros_like(v0[:1]), v0], axis=0))),
        T=st.T.replace(data=jnp.asarray(_owned3(a["T_Kbb"]))),
        S=st.S.replace(data=jnp.asarray(_owned3(a["S_Kbb"]))),
        eta=st.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kbb"]))),
        uu_b=st.uu_b.replace(data=jnp.asarray(np.concatenate([ub0[:, -1:], ub0], axis=1))),
        vv_b=st.vv_b.replace(data=jnp.asarray(np.concatenate([np.zeros_like(vb0[:1]), vb0], axis=0))),
        tke=field(a["tke_en"][..., 1:30], "tke", ("lat", "lon", "level")),
        tke_avm=field(_owned3(a["tke_avm_k"], 31)[..., 1:30], "tke_avm", ("lat", "lon", "level")),
        tke_avt=field(a["tke_avt_k"][..., 1:30], "tke_avt", ("lat", "lon", "level")),
        tke_dissl=field(a["tke_dissl"][..., 1:30], "tke_dissl", ("lat", "lon", "level")),
        tke_avm_surface=field(_owned3(a["tke_avm_k"], 31)[..., 0], "tke_avm_surface", ("lat", "lon")),
    )
    freshwater, surface = _surface_forcings(card, st, 2)
    masks = expected_masks(card)
    rows = []
    entry = lego_fields(st)
    for field, ref in (
        ("u", _owned3(a["u_Kbb"])),
        ("v", _owned3(a["v_Kbb"])),
        ("T", _owned3(a["T_Kbb"])),
        ("S", _owned3(a["S_Kbb"])),
        ("ssh", _owned2(a["ssh_Kbb"])),
    ):
        mask = masks[field]
        rows.append({
            "name": f"GYRE-zco.kt1_end_bridge.{field}",
            "n": int(mask.sum()),
            "n_unequal": int(np.count_nonzero(entry[field][mask] != ref[mask])),
            "max_abs": float(np.max(np.abs(entry[field][mask] - ref[mask]))),
        })
    for stage in (1, 2, 3):
        hooks = _NEMOWSRK3TestHooks(expose_momentum_stage=stage)
        got = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg, _nemo_ws_test_hooks=hooks
        ).step(st, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
        fields = lego_fields(got)
        refa = records[(2, stage)]["arrays"]
        for face in ("u", "v"):
            ref = _owned3(refa[f"post_baro_{face}"])
            candidate = fields[face]
            mask = masks[face]
            if plant == "trajectory" and (stage, face) == (1, "u"):
                candidate = candidate.copy()
                candidate[tuple(np.argwhere(mask)[0])] += 1.0
            rows.append(
                {
                    "name": f"GYRE-zco.kt2.s{stage}.post_baro.{face}",
                    "n": int(mask.sum()),
                    "n_unequal": int(np.count_nonzero(candidate[mask] != ref[mask])),
                    "max_abs": float(np.max(np.abs(candidate[mask] - ref[mask]))),
                }
            )
    if plant == "trajectory":
        require(any(r["max_abs"] > 0.5 for r in rows), "trajectory plant did not land")
    return rows


def _u_full(value):
    value = np.asarray(value)
    return np.concatenate([value[:, -1:, ...], value], axis=1)


def _v_full(value):
    value = np.asarray(value)
    return np.concatenate([np.zeros_like(value[:1]), value], axis=0)


def _classification(row: dict) -> dict:
    row["classification"] = "BIT" if row["exact"] else row["status"]
    return row


def _unmeasured(kt: int, stage: int | str, field: str, reason: str) -> dict:
    return {
        "name": f"GYRE-zco.kt{kt}.s{stage}.{field}",
        "kt": kt,
        "stage": stage,
        "field": field,
        "classification": "UNMEASURED_WITH_SPEC",
        "reason": reason,
    }


def _raw_history_override(memory_root: Path):
    """Read the admitted kt=1 endpoint histories without importing round 51."""
    from nemo_testcase_l2_gyre_round48_bt_memory_gate import read_record

    arrays = read_record(memory_root / "oracle_bt_memory_kt00000001_end.bin")["arrays"]
    return (
        _u_full(_owned2(arrays["ub_e"])),
        _u_full(_owned2(arrays["ubb_e"])),
        _v_full(_owned2(arrays["vb_e"])),
        _v_full(_owned2(arrays["vbb_e"])),
        _owned2(arrays["sshb_e"]),
        _owned2(arrays["sshbb_e"]),
    )


def _bridge_kt2_state(card, cfg, records):
    """Build the kt=2 Kbb state, retaining non-recorded pytrees from kt=1."""
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    state = card.recipe.initial_state
    freshwater, surface = _surface_forcings(card, state, 1)
    state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg
    ).step(state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
    a = records[(2, 1)]["arrays"]

    def field(data, name, dims):
        return Field(data=jnp.asarray(data), name=name, dims=dims)

    state = state._replace(
        u=state.u.replace(data=jnp.asarray(_u_full(_owned3(a["u_Kbb"])))),
        v=state.v.replace(data=jnp.asarray(_v_full(_owned3(a["v_Kbb"])))),
        T=state.T.replace(data=jnp.asarray(_owned3(a["T_Kbb"]))),
        S=state.S.replace(data=jnp.asarray(_owned3(a["S_Kbb"]))),
        eta=state.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kbb"]))),
        uu_b=state.uu_b.replace(data=jnp.asarray(_u_full(_owned2(a["uu_b_Kbb"])) )),
        vv_b=state.vv_b.replace(data=jnp.asarray(_v_full(_owned2(a["vv_b_Kbb"])) )),
        tke=field(a["tke_en"][..., 1:30], "tke", ("lat", "lon", "level")),
        tke_avm=field(_owned3(a["tke_avm_k"], 31)[..., 1:30],
                      "tke_avm", ("lat", "lon", "level")),
        tke_avt=field(a["tke_avt_k"][..., 1:30],
                      "tke_avt", ("lat", "lon", "level")),
        tke_dissl=field(a["tke_dissl"][..., 1:30],
                        "tke_dissl", ("lat", "lon", "level")),
        tke_avm_surface=field(_owned3(a["tke_avm_k"], 31)[..., 0],
                              "tke_avm_surface", ("lat", "lon")),
    )
    return state


def _bridge_stage_context(state, arrays, *, plant: bool = False):
    """Install the pre-stage closure bundle recorded after ``zdf_phy``."""
    import jax.numpy as jnp
    from legoesm.core.field import Field

    tke = np.array(arrays["tke_en"][..., 1:30], copy=True)
    if plant:
        mask = _owned3(arrays["wmask"], 31)[..., 1:30] > 0.5
        index = tuple(np.argwhere(mask & (tke != 0.0))[0])
        tke[index] = np.nextafter(tke[index], np.float64(np.inf))

    def field(data, name, dims):
        return Field(data=jnp.asarray(data), name=name, dims=dims)

    return state._replace(
        tke=field(tke, "tke", ("lat", "lon", "level")),
        tke_avm=field(_owned3(arrays["tke_avm_k"], 31)[..., 1:30],
                      "tke_avm", ("lat", "lon", "level")),
        tke_avt=field(arrays["tke_avt_k"][..., 1:30],
                      "tke_avt", ("lat", "lon", "level")),
        tke_dissl=field(arrays["tke_dissl"][..., 1:30],
                        "tke_dissl", ("lat", "lon", "level")),
        tke_avm_surface=field(_owned3(arrays["tke_avm_k"], 31)[..., 0],
                              "tke_avm_surface", ("lat", "lon")),
    )


def _stage_reference(records, next_entries, kt: int, stage: int) -> dict:
    a = records[(kt, stage)]["arrays"]
    if stage < 3:
        nxt = records[(kt, stage + 1)]["arrays"]
        return {
            "u": _owned3(a["post_baro_u"]),
            "v": _owned3(a["post_baro_v"]),
            "T": _owned3(nxt["T_Kmm"]),
            "S": _owned3(nxt["S_Kmm"]),
            "ssh": _owned2(nxt["ssh_Kmm"]),
        }
    entry = next_entries[kt + 1]
    return {
        "u": _owned3(a["post_baro_u"]),
        "v": _owned3(a["post_baro_v"]),
        "T": entry["T"][..., :30],
        "S": entry["S"][..., :30],
        "ssh": entry["ssh"],
    }


def _entry_override(records, kt: int, stage: int, *, plant=False):
    import jax.numpy as jnp

    a = records[(kt, stage)]["arrays"]
    T = _owned3(a["T_Kmm"]).copy()
    if plant:
        index = tuple(np.argwhere(_owned3(a["tmask"]) > 0.5)[0])
        T[index] = np.nextafter(T[index], np.float64(np.inf))
    return (
        stage,
        jnp.asarray(_u_full(_owned3(a["u_Kmm"]))),
        jnp.asarray(_v_full(_owned3(a["v_Kmm"]))),
        jnp.asarray(T),
        jnp.asarray(_owned3(a["S_Kmm"])),
        jnp.asarray(_owned2(a["ssh_Kmm"])),
    )


def _barotropic_override(records, advmean_root: Path, kt: int):
    import jax.numpy as jnp
    from nemo_testcase_l2_gyre_round14_advmean import read_advmean
    from nemo_testcase_l2_gyre_phase3_gate import read_bt

    avg = read_advmean(
        advmean_root / f"oracle_bt_advmean_operands_kt{kt:08d}.bin",
        expected_kt=kt,
    )
    bt = read_bt(
        advmean_root / f"oracle_bt_frames_kt{kt:08d}.bin", kt)
    next_entry = read_entry(
        advmean_root / f"oracle_step_entry_kt{kt + 1:08d}.bin")
    return (
        jnp.asarray(next_entry["ssh"]),
        jnp.asarray(_u_full(bt["uu_b"])),
        jnp.asarray(_v_full(bt["vv_b"])),
        jnp.asarray(_u_full(avg["post_lbc_u"])),
        jnp.asarray(_v_full(avg["post_lbc_v"])),
    )


def _final_history_from_btstep(arrays: dict) -> tuple[np.ndarray, ...]:
    """Apply NEMO's final ``bb<-b, b<-n`` history rotation to a record."""
    last = -1
    return (
        arrays["u_entry"][last], arrays["u_b"][last],
        arrays["v_entry"][last], arrays["v_b"][last],
        arrays["eta_entry"][last], arrays["eta_b"][last],
    )


def _history_reference(memory_root: Path, btstep_root: Path, kt: int):
    """Return the six absolute histories at the completed external boundary."""
    if kt == 1:
        from nemo_testcase_l2_gyre_round48_bt_memory_gate import read_record

        arrays = read_record(
            memory_root / "oracle_bt_memory_kt00000001_end.bin")["arrays"]
        return tuple(arrays[name] for name in (
            "ub_e", "ubb_e", "vb_e", "vbb_e", "sshb_e", "sshbb_e"))
    from nemo_testcase_l2_gyre_round81_btstep_gate import read_record

    arrays = read_record(
        btstep_root / "oracle_bt_step_operands_kt00000002.bin", expected_kt=2)
    # dynspg_ts rotates bb<-b, b<-n, n<-a at the end of each substep.
    return _final_history_from_btstep(arrays)


def _closure_rows(context, tke_entry, arrays, masks, kt: int, stage: int,
                  mode: str, boundary: str) -> list[dict]:
    """Score the closure fields computed once and consumed by every stage."""
    candidate_fields = {
        "tke_en": tke_entry,
        "tke_avm_k": context.tke_avm,
        "tke_avt_k": context.tke_avt,
        "tke_dissl": context.tke_dissl,
        "tke_avm_surface": context.tke_avm_surface,
    }
    references = {
        "tke_en": np.asarray(arrays["tke_en"])[..., 1:30],
        "tke_avm_k": _owned3(arrays["tke_avm_k"], 31)[..., 1:30],
        "tke_avt_k": np.asarray(arrays["tke_avt_k"])[..., 1:30],
        "tke_dissl": np.asarray(arrays["tke_dissl"])[..., 1:30],
        "tke_avm_surface": _owned3(arrays["tke_avm_k"], 31)[..., 0],
    }
    wet_w = _owned3(arrays["wmask"], 31)[..., 1:30] > 0.5
    field_masks = {name: wet_w for name in candidate_fields}
    field_masks["tke_avm_surface"] = masks["ssh"]
    rows = []
    for field, candidate_field in candidate_fields.items():
        if candidate_field is None:
            row = _unmeasured(
                kt, stage, field,
                "model stage-entry context has no explicit carried field")
            row.update({"entry_mode": mode, "boundary": boundary})
            rows.append(row)
            continue
        candidate = np.asarray(
            candidate_field if field == "tke_en" else candidate_field.data)
        row = _classification(score(
            f"GYRE-zco.kt{kt}.s{stage}.{boundary}.{field}",
            references[field], candidate, field_masks[field]))
        row.update({"kt": kt, "stage": stage, "field": field,
                    "entry_mode": mode, "boundary": boundary})
        rows.append(row)
    return rows


def _output_rows(trace, records, next_entries, transports, masks, area_t,
                 context, kt: int, mode: str,
                 direct_stage_ww: dict | None = None) -> list[dict]:
    rows = []
    for stage, output in enumerate(trace.stage_outputs, start=1):
        u, v, T, S, eta = (np.asarray(value) for value in output)
        candidates = {"u": u[:, 1:, :], "v": v[1:, :, :],
                      "T": T, "S": S, "ssh": eta}
        refs = _stage_reference(records, next_entries, kt, stage)
        for field in ("T", "S", "u", "v", "ssh"):
            row = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.output.{field}",
                refs[field], candidates[field], masks[field]))
            row.update({"kt": kt, "stage": stage, "field": field,
                        "entry_mode": mode})
            rows.append(row)

        a = records[(kt, stage)]["arrays"]
        geom = trace.stage_geometry[stage - 1]
        ww = np.asarray(geom[2])
        ww_nlev = ww.shape[-1]
        ww_reference = _owned3(a["ww"], ww_nlev)
        ww_boundary = "pre_external_velocity_form"
        if direct_stage_ww is not None and (kt, stage) in direct_stage_ww:
            ww_reference = np.asarray(
                direct_stage_ww[(kt, stage)]["ww"])[..., :ww_nlev]
            ww_boundary = "post_tra_adv_trp_transport_form"
        geom_rows = (
            ("e3t_Kmm", _owned3(a["e3t_Kmm"]), np.asarray(geom[3]),
             _owned3(a["tmask"]) > 0.5),
            ("e3u_Kmm", _owned3(a["e3u_Kmm"]), np.asarray(geom[4])[:, 1:, :],
             _owned3(a["umask"]) > 0.5),
            ("e3v_Kmm", _owned3(a["e3v_Kmm"]), np.asarray(geom[5])[1:, :, :],
             _owned3(a["vmask"]) > 0.5),
            ("ww", ww_reference, ww,
             _owned3(a["wmask"], ww_nlev) > 0.5),
        )
        for field, ref, candidate, mask in geom_rows:
            row = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.handoff.{field}",
                ref, candidate, mask))
            row.update({"kt": kt, "stage": stage, "field": field,
                        "entry_mode": mode})
            if field == "ww":
                row["reference_boundary"] = ww_boundary
            rows.append(row)

        rows.extend(_closure_rows(
            context, trace.tke_entry, a, masks, kt, stage, mode, "output"))

        transport = transports.get((kt, stage))
        if transport is None:
            for field in ("zFu", "zFv", "zFw"):
                rows.append(_unmeasured(
                    kt, stage, field,
                    "no admitted direct transport payload for this kt/stage"))
            continue
        for field, candidate, mask in (
            ("zFu", np.asarray(geom[7])[:, 1:, :], _owned3(a["umask"]) > 0.5),
            ("zFv", np.asarray(geom[8])[1:, :, :], _owned3(a["vmask"]) > 0.5),
            ("zFw", np.asarray(geom[2]) * area_t[..., None],
             _owned3(a["wmask"], np.asarray(geom[2]).shape[-1]) > 0.5),
        ):
            nlev = candidate.shape[-1]
            row = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.handoff.{field}",
                np.asarray(transport[field])[..., :nlev], candidate,
                mask[..., :nlev]))
            row.update({"kt": kt, "stage": stage, "field": field,
                        "entry_mode": mode})
            rows.append(row)
    return rows


def _entry_rows(trace, records, masks, context, kt: int, stage: int,
                mode: str, *, plant_rhs: bool = False) -> list[dict]:
    """Prove the exact Kmm state that actually entered one compiled stage."""
    a = records[(kt, stage)]["arrays"]
    u, v, temperature, salinity, eta = (
        np.asarray(value) for value in trace.stage_states[stage - 1])
    candidates = {
        "u": u[:, 1:, :], "v": v[1:, :, :],
        "T": temperature, "S": salinity, "ssh": eta,
    }
    references = {
        "u": _owned3(a["u_Kmm"]), "v": _owned3(a["v_Kmm"]),
        "T": _owned3(a["T_Kmm"]), "S": _owned3(a["S_Kmm"]),
        "ssh": _owned2(a["ssh_Kmm"]),
    }
    rows = []
    for field in ("T", "S", "u", "v", "ssh"):
        row = _classification(score(
            f"GYRE-zco.kt{kt}.s{stage}.entry.{field}",
            references[field], candidates[field], masks[field]))
        row.update({"kt": kt, "stage": stage, "field": field,
                    "entry_mode": mode})
        rows.append(row)
    rhs_boundary = "pre_zdf_rhs" if stage == 3 else "after_adv"
    rhs_u, rhs_v = (np.asarray(value)
                    for value in trace.stage_rhs[stage - 1])
    rhs_candidates = {"u": rhs_u[:, 1:, :], "v": rhs_v[1:, :, :]}
    for face in ("u", "v"):
        reference = _owned3(a[f"{rhs_boundary}_{face}"]).copy()
        candidate = rhs_candidates[face]
        mask = masks[face]
        planted_at = None
        if plant_rhs and face == "u":
            equal = (
                np.ascontiguousarray(candidate).view(np.uint64)
                == np.ascontiguousarray(reference).view(np.uint64)
            ) & mask
            indices = np.argwhere(equal & np.isfinite(reference))
            require(indices.size > 0,
                    "stage-RHS plant found no exact finite wet cell")
            planted_at = tuple(int(value) for value in indices[0])
            reference[planted_at] = np.nextafter(
                reference[planted_at], np.float64(np.inf))
        row = _classification(score(
            f"GYRE-zco.kt{kt}.s{stage}.entry.momentum_rhs_{face}",
            reference, candidate, mask))
        row.update({
            "kt": kt,
            "stage": stage,
            "field": f"momentum_rhs_{face}",
            "entry_mode": mode,
            "boundary": "entry",
            "nemo_boundary": rhs_boundary,
            "plant_index": planted_at,
        })
        rows.append(row)
    if kt == 1 and stage == 1:
        rhs_walk_names = (
            ("operator_accumulator", "compiled stp2d HPG/LDF/VOR/KEG/ZAD"),
            ("transport_reconcile", "model-only stage transport reassociation"),
            ("zad_reassociation", "model-only stage ZAD reassociation"),
            ("final", "RK assignment input"),
        )
        reference_by_face = {
            face: _owned3(a[f"after_adv_{face}"])
            for face in ("u", "v")
        }
        for boundary_index, (boundary, statement) in enumerate(rhs_walk_names):
            for face, candidate_full in zip(
                    ("u", "v"), trace.stage1_rhs_walk[boundary_index],
                    strict=True):
                candidate_full = np.asarray(candidate_full)
                candidate = (candidate_full[:, 1:, :] if face == "u"
                             else candidate_full[1:, :, :])
                row = _classification(score(
                    f"GYRE-zco.kt1.s1.rhs_walk.{boundary}.{face}",
                    reference_by_face[face], candidate, masks[face]))
                row.update({
                    "kt": 1,
                    "stage": 1,
                    "field": f"stage1_rhs_{boundary}_{face}",
                    "entry_mode": mode,
                    "boundary": boundary,
                    "statement": statement,
                    "reference_boundary": "after_adv",
                })
                rows.append(row)
    qco = tuple(np.asarray(value) for value in trace.stage_qco[stage - 1])
    qco_candidates = {
        "r3t_Kmm": qco[0],
        "r3u_Kmm": qco[1][:, 1:, 0],
        "r3v_Kmm": qco[2][1:, :, 0],
    }
    for field, candidate in qco_candidates.items():
        face = field[2]
        mask = masks["ssh" if face == "t" else face][..., 0] if face != "t" else masks["ssh"]
        row = _classification(score(
            f"GYRE-zco.kt{kt}.s{stage}.entry.{field}",
            _owned2(a[field]), candidate, mask))
        row.update({"kt": kt, "stage": stage, "field": field,
                    "entry_mode": mode, "boundary": "entry"})
        rows.append(row)
    rows.extend(_closure_rows(
        context, trace.tke_entry, a, masks, kt, stage, mode, "entry"))
    return rows


_STAGE1_W_ORDER = (
    "transport_u", "transport_v", "zonal_difference",
    "meridional_difference", "numerator", "scaled", "hdiv", "e3div",
    "r3_delta", "stretch", "bracket", "incoming_carry",
    "outgoing_carry", "ww",
)


def _stage1_transport_w_trace(
    transport_u, transport_v, reciprocal_area, thickness_t, e3t_0,
    r3_kbb, r3_kaa, r1_dt, tmask, *, source_round: bool,
):
    """Trace the existing shared transport-form W arithmetic."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round

    materialize = nemo_source_round if source_round else jax.lax.optimization_barrier
    transport_u = jnp.asarray(transport_u)[..., :30]
    transport_v = jnp.asarray(transport_v)[..., :30]
    tmask = jnp.asarray(tmask)[..., :30]
    west = jnp.roll(transport_u, 1, axis=1)
    south = jnp.concatenate(
        [jnp.zeros_like(transport_v[:1]), transport_v[:-1]], axis=0)
    zonal = materialize(transport_u - west)
    meridional = materialize(transport_v - south)
    numerator = materialize(zonal + meridional)
    scaled = materialize(numerator * reciprocal_area[..., None]) * tmask
    safe_thickness = jnp.where(tmask > 0.5, thickness_t[..., :30], 1.0)
    hdiv = materialize(scaled / safe_thickness) * tmask
    e3div = materialize(hdiv * thickness_t[..., :30]) * tmask
    r3_delta = materialize(r3_kaa - r3_kbb)
    stretch = materialize(
        materialize(r1_dt * e3t_0[..., :30]) * r3_delta[..., None])
    incoming = []
    outgoing = [None] * 30
    brackets = [None] * 30
    carry = jnp.zeros_like(r3_kbb)
    for level in range(29, -1, -1):
        incoming.append(carry)
        bracket = materialize(e3div[..., level] + stretch[..., level])
        carry = materialize(
            carry - materialize(bracket * tmask[..., level]))
        brackets[level] = bracket
        outgoing[level] = carry
    incoming = list(reversed(incoming))
    ww = jnp.stack(outgoing + [jnp.zeros_like(carry)], axis=-1)
    return {
        "transport_u": transport_u,
        "transport_v": transport_v,
        "zonal_difference": zonal,
        "meridional_difference": meridional,
        "numerator": numerator,
        "scaled": scaled,
        "hdiv": hdiv,
        "e3div": e3div,
        "r3_delta": r3_delta,
        "stretch": stretch,
        "bracket": jnp.stack(brackets, axis=-1),
        "incoming_carry": jnp.stack(incoming, axis=-1),
        "outgoing_carry": jnp.stack(outgoing, axis=-1),
        "ww": ww,
    }


def _stage1_transport_w_scalar_reference(arrays, transport_u, transport_v):
    """Scalar replay of compiled divhor + WZV, calibrated to direct W."""
    shape = (22, 32, 30)
    tmask = _owned3(arrays["tmask"]) > 0.5
    thickness = _owned3(arrays["e3t_Kmm"])
    e3t_0 = _owned3(arrays["e3t_0"])
    reciprocal_area = _owned2(arrays["r1_e1e2t"])
    r3_kbb = _owned2(arrays["r3t_Kbb"])
    r3_kaa = _owned2(arrays["r3t_Kaa"])
    hdiv = np.zeros(shape, dtype=np.float64)
    e3div = np.zeros(shape, dtype=np.float64)
    zonal = np.zeros(shape, dtype=np.float64)
    meridional = np.zeros(shape, dtype=np.float64)
    numerator = np.zeros(shape, dtype=np.float64)
    scaled = np.zeros(shape, dtype=np.float64)
    stretch = np.zeros(shape, dtype=np.float64)
    bracket = np.zeros(shape, dtype=np.float64)
    incoming = np.zeros(shape, dtype=np.float64)
    outgoing = np.zeros(shape, dtype=np.float64)
    for level in range(30):
        for j in range(22):
            for i in range(32):
                west = transport_u[j, i - 1, level]
                south = (np.float64(0.0) if j == 0
                         else transport_v[j - 1, i, level])
                zu = np.float64(transport_u[j, i, level] - west)
                zv = np.float64(transport_v[j, i, level] - south)
                total = np.float64(zu + zv)
                area_scaled = np.float64(total * reciprocal_area[j, i])
                value = np.float64(area_scaled / thickness[j, i, level])
                zonal[j, i, level] = zu
                meridional[j, i, level] = zv
                numerator[j, i, level] = total
                scaled[j, i, level] = area_scaled * tmask[j, i, level]
                hdiv[j, i, level] = value * tmask[j, i, level]
                e3div[j, i, level] = np.float64(
                    value * thickness[j, i, level]) * tmask[j, i, level]
    carry = np.zeros((22, 32), dtype=np.float64)
    r3_delta = np.asarray(r3_kaa - r3_kbb, dtype=np.float64)
    for level in range(29, -1, -1):
        for j in range(22):
            for i in range(32):
                incoming[j, i, level] = carry[j, i]
                term = np.float64(
                    np.float64(arrays["r1_Dt"] * e3t_0[j, i, level])
                    * r3_delta[j, i])
                stretch[j, i, level] = term
                total = np.float64(e3div[j, i, level] + term)
                bracket[j, i, level] = total
                carry[j, i] = np.float64(
                    carry[j, i] - np.float64(total * tmask[j, i, level]))
                outgoing[j, i, level] = carry[j, i]
    ww = np.concatenate(
        [outgoing, np.zeros((22, 32, 1), dtype=np.float64)], axis=-1)
    return {
        "transport_u": np.asarray(transport_u)[..., :30],
        "transport_v": np.asarray(transport_v)[..., :30],
        "zonal_difference": zonal,
        "meridional_difference": meridional,
        "numerator": numerator,
        "scaled": scaled,
        "hdiv": hdiv,
        "e3div": e3div,
        "r3_delta": r3_delta,
        "stretch": stretch,
        "bracket": bracket,
        "incoming_carry": incoming,
        "outgoing_carry": outgoing,
        "ww": ww,
    }


def _stage1_w_walk(records, transports, direct_stage_ww, plant: str | None):
    """Walk kt=1 stage-1 transport W from admitted NEMO operands."""
    import jax
    import jax.numpy as jnp

    arrays = records[(1, 1)]["arrays"]
    transport = transports[(1, 1)]
    transport_u = np.asarray(transport["zFu"])
    transport_v = np.asarray(transport["zFv"])
    reference = _stage1_transport_w_scalar_reference(
        arrays, transport_u, transport_v)
    direct_ww = np.asarray(direct_stage_ww[(1, 1)]["ww"])
    wet_w = _owned3(arrays["wmask"], 31) > 0.5
    require(np.array_equal(reference["ww"][wet_w], direct_ww[wet_w]),
            "scalar transport-W replay does not reproduce direct NEMO W")
    operands = (
        jnp.asarray(transport_u), jnp.asarray(transport_v),
        jnp.asarray(_owned2(arrays["r1_e1e2t"])),
        jnp.asarray(_owned3(arrays["e3t_Kmm"])),
        jnp.asarray(_owned3(arrays["e3t_0"])),
        jnp.asarray(_owned2(arrays["r3t_Kbb"])),
        jnp.asarray(_owned2(arrays["r3t_Kaa"])),
        jnp.asarray(arrays["r1_Dt"]),
        jnp.asarray(_owned3(arrays["tmask"])),
    )
    ordinary = jax.device_get(jax.jit(
        lambda: _stage1_transport_w_trace(*operands, source_round=False))())
    source_rounded = jax.device_get(jax.jit(
        lambda: _stage1_transport_w_trace(*operands, source_round=True))())
    wet_t = _owned3(arrays["tmask"]) > 0.5
    wet_2d = wet_t[..., 0]

    def active(name):
        if name == "ww":
            return wet_w
        if name == "r3_delta":
            return wet_2d
        return wet_t

    ordinary_rows = []
    candidate_rows = []
    for name in _STAGE1_W_ORDER:
        oracle = np.asarray(reference[name]).copy()
        planted_at = None
        if plant == "stage-w-transport-ulp" and name == "transport_u":
            planted_at = tuple(int(value) for value in np.argwhere(wet_t)[0])
            oracle[planted_at] = np.nextafter(
                oracle[planted_at], np.float64(np.inf))
        if plant == "stage-w-carry-ulp" and name == "outgoing_carry":
            planted_at = tuple(int(value) for value in np.argwhere(wet_t)[0])
            oracle[planted_at] = np.nextafter(
                oracle[planted_at], np.float64(np.inf))
        for destination, values, label in (
            (ordinary_rows, ordinary, "production_association"),
            (candidate_rows, source_rounded, "source_rounded_candidate"),
        ):
            row = _classification(score(
                f"GYRE-zco.kt1.s1.w_walk.{label}.{name}",
                oracle, np.asarray(values[name]), active(name)))
            row.update({"boundary": name, "association": label,
                        "plant_index": planted_at})
            destination.append(row)
    first = next((row for row in ordinary_rows
                  if row["classification"] != "BIT"), None)
    candidate_first = next((row for row in candidate_rows
                            if row["classification"] != "BIT"), None)
    if plant:
        target = "transport_u" if plant == "stage-w-transport-ulp" else "outgoing_carry"
        row = next(value for value in ordinary_rows if value["boundary"] == target)
        require(row["n_unequal"] == 1,
                f"{plant} did not flip exactly one named row cell")
    return {
        "compiled_order": list(_STAGE1_W_ORDER),
        "direct_record_clock_seconds": direct_stage_ww[(1, 1)]["rDt_s"],
        "pre_external_zad_operand_w": _classification(score(
            "GYRE-zco.kt1.s1.w_walk.pre_external_zad_operand_w",
            _owned3(arrays["ww"], 31), direct_ww, wet_w)),
        "production_rows": ordinary_rows,
        "first_nonbit": (None if first is None else {
            key: first[key] for key in (
                "boundary", "n_unequal", "absolute_max", "classification")}),
        "source_rounded_candidate_rows": candidate_rows,
        "source_rounded_first_nonbit": (
            None if candidate_first is None else {
                key: candidate_first[key] for key in (
                    "boundary", "n_unequal", "absolute_max", "classification")}),
        "scalar_replay_direct_w_bit_exact": True,
        "plant": plant,
    }


def _external_rows(outputs, histories, records, advmean_root: Path,
                   memory_root: Path, btstep_root: Path, masks, kt: int,
                   mode: str) -> list[dict]:
    """Score the split-explicit handoff consumed by the RK3 stage ladder."""
    eta, uu_b, vv_b, hu_avg, hv_avg = (
        np.asarray(value) for value in outputs)
    reference = _barotropic_override(records, advmean_root, kt)
    ref_eta, ref_u, ref_v, ref_hu, ref_hv = (
        np.asarray(value) for value in reference)
    rows = []
    fields = (
        ("ssh", ref_eta, eta, masks["ssh"]),
        ("uu_b", ref_u[:, 1:], uu_b[:, 1:], masks["u"][..., 0]),
        ("vv_b", ref_v[1:, :], vv_b[1:, :], masks["v"][..., 0]),
        ("Hu_avg", ref_hu[:, 1:], hu_avg[:, 1:], masks["u"][..., 0]),
        ("Hv_avg", ref_hv[1:, :], hv_avg[1:, :], masks["v"][..., 0]),
    )
    for field, oracle, candidate, mask in fields:
        row = _classification(score(
            f"GYRE-zco.kt{kt}.external.output.{field}",
            oracle, candidate, mask))
        row.update({"kt": kt, "stage": "external", "field": field,
                    "entry_mode": mode})
        rows.append(row)
    history_ref = _history_reference(memory_root, btstep_root, kt)
    history_fields = (
        ("ub_e", history_ref[0], histories[0][:, 1:], masks["u"][..., 0]),
        ("ubb_e", history_ref[1], histories[1][:, 1:], masks["u"][..., 0]),
        ("vb_e", history_ref[2], histories[2][1:, :], masks["v"][..., 0]),
        ("vbb_e", history_ref[3], histories[3][1:, :], masks["v"][..., 0]),
        ("sshb_e", history_ref[4], histories[4], masks["ssh"]),
        ("sshbb_e", history_ref[5], histories[5], masks["ssh"]),
    )
    for field, oracle, candidate, mask in history_fields:
        if kt == 1:
            oracle = _owned2(oracle)
        row = _classification(score(
            f"GYRE-zco.kt{kt}.external.output.{field}",
            oracle, candidate, mask))
        row.update({"kt": kt, "stage": "external", "field": field,
                    "entry_mode": mode})
        rows.append(row)
    return rows


def _stage_twin(records: dict, stage_root: Path, advmean_root: Path,
                memory_root: Path, btstep_root: Path,
                stage_closure_root: Path,
                plant: str | None) -> dict:
    """Decision-41 stage tables from recorded entries and the shared stage."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "stage twin requires x64")
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    masks = expected_masks(card)
    next_entries = {
        kt: read_entry(stage_root / f"oracle_step_entry_kt{kt:08d}.bin")
        for kt in (2, 3)
    }
    from nemo_testcase_l2_gyre_round71_fct_stage2_gate import (
        read_record as read_tracer_stage)

    transports = {}
    for kt in (1, 2):
        for stage in (1, 2):
            path = advmean_root / (
                f"oracle_rktracer_operands_kt{kt:08d}_s{stage}.bin")
            record = read_tracer_stage(path, stage, expected_kt=kt)["fields"]
            transports[(kt, stage)] = {
                field: np.ascontiguousarray(record[field].swapaxes(0, 1))
                for field in ("zFu", "zFv", "zFw")
            }
    for kt, transport_root in ((1, advmean_root), (2, stage_closure_root)):
        path = transport_root / f"oracle_tracer_transport_kt{kt:08d}_s3.bin"
        header = records[(kt, 3)]["header"]
        transports[(kt, 3)] = read_transport(
            path, 3, expected_kt=kt,
            expected_slots=tuple(
                header[name] for name in ("Kbb", "Kmm", "Kaa", "Krhs")))

    direct_stage_ww = {(1, 1): read_stage_ww(
        STAGE_WW_ROOT / "oracle_rkstage_ww_kt00000001_s1.bin", 1)}
    stage1_w_walk = _stage1_w_walk(
        records, transports, direct_stage_ww, plant)
    if plant in {"stage-w-transport-ulp", "stage-w-carry-ulp"}:
        return {
            "format": "nemo-testcase-l2-gyre-stage-twin-v4",
            "given_nemo_entry": [], "chained": [],
            "stage_entry_identity": [], "first_owned_nonbit": None,
            "stage1_w_walk": stage1_w_walk,
        }

    if plant == "stage-rhs-ulp":
        planted_rows, _ = _given_inputs(
            records, "stage-rhs-ulp", kt=1, stages=(1,))
        planted = next(
            row for row in planted_rows
            if row["name"] == "GYRE-zco.kt1.s1.post_hpg_accumulator.u")
        return {
            "format": "nemo-testcase-l2-gyre-stage-twin-v3",
            "given_nemo_entry": [], "chained": [],
            "stage_entry_identity": [], "first_owned_nonbit": None,
            "stage_rhs_ulp_plant_flipped_row": planted["n_unequal"] == 1,
            "stage_rhs_ulp_plant_target": planted["name"],
        }

    given = []
    given_entries = []
    states = {1: _bridge_stage_context(
                  card.recipe.initial_state, records[(1, 1)]["arrays"],
                  plant=plant == "stage-context-ulp"),
              2: _bridge_kt2_state(card, cfg, records)}
    raw_history = tuple(jnp.asarray(value) for value in _raw_history_override(memory_root))
    kt_values = ((1,) if plant in {
        "stage-entry-ulp", "stage-context-ulp"
    } else (1, 2))
    stage_values = ((2,) if plant == "stage-entry-ulp" else
                    (1,) if plant in {
                        "stage-context-ulp"
                    } else (1, 2, 3))
    for kt in kt_values:
        state = states[kt]
        freshwater, surface = _surface_forcings(card, state, kt)
        for stage in stage_values:
            entry_override = None if stage == 1 else _entry_override(
                records, kt, stage,
                plant=(plant == "stage-entry-ulp" and kt == 1 and stage == 2))
            hooks = _NEMOWSRK3TestHooks(
                expose_live_stage_operands=True,
                barotropic_raw_history_override=(raw_history if kt == 2 else None),
                stage_barotropic_output_override=_barotropic_override(
                    records, advmean_root, kt),
                stage_entry_override=entry_override,
            )
            trace = jax.device_get(LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, cfg,
                _nemo_ws_test_hooks=hooks).step(
                    state, dt=card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
            entry_rows = _entry_rows(
                trace, records, masks, state, kt, stage, "NEMO_RECORDED",
                plant_rhs=False)
            given_entries.extend(entry_rows)
            if plant == "stage-entry-ulp" and (kt, stage) == (1, 2):
                planted = next(row for row in entry_rows
                               if kt == 1 and stage == 2
                               and row["field"] == "T")
                require(planted["n_unequal"] == 1,
                        "one-ULP stage-entry plant did not flip exactly one entry row cell")
                return {
                    "format": "nemo-testcase-l2-gyre-stage-twin-v1",
                    "given_nemo_entry": [], "chained": [],
                    "stage_entry_identity": entry_rows,
                    "first_owned_nonbit": None,
                    "stage_entry_ulp_plant_flipped_row": True,
                    "stage_entry_ulp_plant_changed_output": False,
                }
            if plant == "stage-context-ulp" and (kt, stage) == (1, 1):
                planted = next(row for row in entry_rows
                               if row["field"] == "tke_en")
                require(planted["n_unequal"] == 1,
                        "one-ULP stage-context plant did not flip exactly one row cell")
                return {
                    "format": "nemo-testcase-l2-gyre-stage-twin-v2",
                    "given_nemo_entry": [], "chained": [],
                    "stage_entry_identity": entry_rows,
                    "first_owned_nonbit": None,
                    "stage_context_ulp_plant_flipped_row": True,
                }
            rows = _output_rows(
                trace, records, next_entries, transports, masks,
                np.asarray(card.recipe.grid.area_T), state, kt,
                "NEMO_RECORDED", direct_stage_ww)
            given.extend(row for row in rows if row.get("stage") == stage)

    for kt in (1, 2):
        state = states[kt]
        freshwater, surface = _surface_forcings(card, state, kt)
        hooks = _NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_raw_history_override=(raw_history if kt == 2 else None),
        )
        external = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=hooks).step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
        external_outputs = (
            external.state_after_barotropic.eta.data,
            external.state_after_barotropic.uu_b.data,
            external.state_after_barotropic.vv_b.data,
            external.transport_average[0], external.transport_average[1],
        )
        given.extend(_external_rows(
            external_outputs, external.state_after_barotropic.bt_hist,
            records, advmean_root, memory_root, btstep_root, masks, kt,
            "NEMO_RECORDED"))

    chained = []
    state = card.recipe.initial_state
    for kt in (1, 2):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True))
        # Mirror the public step shim before observing the stage-entry carry.
        # A cold start seeds avm/avt/surface-avm/dissl; scoring the caller's
        # pre-shim object would falsely report those consumed fields absent.
        state = model._seed_tke_preclosure_carry(state)
        freshwater, surface = _surface_forcings(card, state, kt)
        trace = jax.device_get(model.step(
                    state, dt=card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
        chained.extend(_external_rows(
            (trace.barotropic_targets[4], trace.barotropic_targets[0],
             trace.barotropic_targets[1], trace.barotropic_targets[2],
             trace.barotropic_targets[3]),
            trace.state_after.bt_hist, records, advmean_root, memory_root,
            btstep_root, masks, kt, "LEGO_CHAINED"))
        chained.extend(_output_rows(
            trace, records, next_entries, transports, masks,
            np.asarray(card.recipe.grid.area_T), state, kt,
            "LEGO_CHAINED", direct_stage_ww))
        state = trace.state_after

    measured = [row for row in given
                if row.get("classification") != "UNMEASURED_WITH_SPEC"]
    first = next((row for row in measured
                  if row.get("classification") != "BIT"), None)
    stage1_operator_rows, stage1_operator_first = _given_inputs(
        records, None, kt=1, stages=(1,))
    stage1_rhs_rows = [
        row for row in given_entries
        if row.get("field", "").startswith("stage1_rhs_")
    ]
    first_rhs = next(
        (row for row in stage1_rhs_rows
         if row.get("classification") != "BIT"), None)
    return {
        "format": "nemo-testcase-l2-gyre-stage-twin-v4",
        "given_nemo_entry": given,
        "chained": chained,
        "stage_entry_identity": given_entries,
        "stage1_operator_walk": {
            "compiled_order": ("hpg", "ldf", "vor", "adv"),
            "rows": stage1_operator_rows,
            "first_nonbit": stage1_operator_first["stage1"],
            "post_operator_rhs_walk": stage1_rhs_rows,
            "first_nonbit_model_statement": (
                None if first_rhs is None else {
                    key: first_rhs[key] for key in (
                        "boundary", "statement", "field", "n_unequal",
                        "absolute_max", "classification")
                }
            ),
        },
        "stage1_w_walk": stage1_w_walk,
        "first_owned_nonbit": None if first is None else {
            key: first[key] for key in (
                "kt", "stage", "field", "n_unequal", "absolute_max",
                "classification")},
        "stage_entry_ulp_plant_flipped_row": None,
        "stage_entry_ulp_plant_changed_output": None,
    }


def _exact_payload_pairs(left: dict, right: dict, pairs: dict[str, str], *,
                         right_is_full: bool) -> dict:
    """Bit-test the duplicate kt=2 payloads written by independent streams."""
    rows = {}
    for left_name, right_name in pairs.items():
        a = np.asarray(left[left_name])
        b = np.asarray(right[right_name])
        if right_is_full:
            b = _owned2(b) if a.ndim == 2 else _owned3(b, a.shape[-1])
        rows[f"{left_name}={right_name}"] = bool(
            a.shape == b.shape
            and np.array_equal(
                np.ascontiguousarray(a).view(np.uint64),
                np.ascontiguousarray(b).view(np.uint64),
            )
        )
    return {"pair_count": len(rows), "all_bit_identical": all(rows.values()),
            "pairs": rows}


def run(
    root: Path,
    *,
    expect_commit: str,
    mode: str,
    plant: str | None,
    round40_kt1: Path,
    round41_kt1: Path,
    advmean_root: Path = ADVMEAN_ROOT,
    memory_root: Path = MEMORY_ROOT,
    btstep_root: Path = BTSTEP_ROOT,
    stage_closure_root: Path = STAGE_CLOSURE_ROOT,
) -> dict:
    stamp = worktree_stamp()
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(
        len(expected) == 40 and stamp["commit"].lower() == expected,
        f"commit stamp mismatch: {stamp['commit']} != {expected}",
    )
    records = {}
    hashes = {}
    for kt, stage in STAGES:
        path = root / f"oracle_momstage_kt{kt:08d}_s{stage}.bin"
        p = plant if (kt, stage) == (2, 1) and plant in {"header", "truncation"} else None
        records[(kt, stage)] = read_stage(path, plant=p)
        hashes[path.name] = sha256(path)
    calibration = _calibrate(records, plant)
    new40 = root / "oracle_rkstage3_terms_kt00000001.bin"
    new41 = root / "oracle_dynadv_split_kt00000001_s3.bin"
    legacy_records = {
        "round40_kt1": read_stage3_terms(new40, expect_kt=1)["header"],
        "round40_kt2": read_stage3_terms(
            root / "oracle_rkstage3_terms_kt00000002.bin", expect_kt=2
        )["header"],
        "round41_kt1": read_split(new41, expect_kt=1)["header"],
        "round41_kt2": read_split(root / "oracle_dynadv_split_kt00000002_s3.bin", expect_kt=2)[
            "header"
        ],
    }
    twin_reports = {
        "round40": _compare_self_describing(
            round40_kt1, new40, [True], max_listed=64),
        "round41": _compare_self_describing(
            round41_kt1, new41, [True], max_listed=64),
    }
    twin = {key: bool(value["consumed_equal"])
            for key, value in twin_reports.items()}
    raw_twin = {
        "round40": new40.read_bytes() == round40_kt1.read_bytes(),
        "round41": new41.read_bytes() == round41_kt1.read_bytes(),
    }
    if plant == "twin":
        twin["round40"] = False
    require(all(twin.values()), f"consumed kt1 legacy twin moved: {twin}")

    stage3 = records[(2, 3)]["arrays"]
    kt2_payload_identity = {
        "rkstage3_terms": _exact_payload_pairs(
            read_stage3_terms(root / "oracle_rkstage3_terms_kt00000002.bin",
                              expect_kt=2)["arrays"],
            stage3,
            {
                "before_u": "rhs_entry_u", "before_v": "rhs_entry_v",
                "after_hpg_u": "after_hpg_u", "after_hpg_v": "after_hpg_v",
                "after_vor_u": "after_vor_u", "after_vor_v": "after_vor_v",
                "after_adv_u": "after_adv_u", "after_adv_v": "after_adv_v",
                "uu_Kmm": "u_Kmm", "vv_Kmm": "v_Kmm", "ww": "ww",
                "r3t_Kmm": "r3t_Kmm", "r3u_Kmm": "r3u_Kmm",
                "r3v_Kmm": "r3v_Kmm", "e3u_Kmm": "e3u_Kmm",
                "e3v_Kmm": "e3v_Kmm", "e3t_Kmm": "e3t_Kmm",
                "e3w_Kmm": "e3w_Kmm", "e3u_0": "e3u_0",
                "e3v_0": "e3v_0", "e3t_0": "e3t_0",
                "umask": "umask", "vmask": "vmask", "tmask": "tmask",
                "wmask": "wmask", "e1e2t": "e1e2t",
                "r1_e1u": "r1_e1u", "r1_e2v": "r1_e2v",
            },
            right_is_full=True,
        ),
        "dynadv_split": _exact_payload_pairs(
            read_split(root / "oracle_dynadv_split_kt00000002_s3.bin",
                       expect_kt=2)["arrays"],
            stage3,
            {
                "before_keg_u": "after_vor_u", "before_keg_v": "after_vor_v",
                "after_keg_u": "after_keg_u", "after_keg_v": "after_keg_v",
                "after_zad_u": "after_zad_u", "after_zad_v": "after_zad_v",
                "uu_Kmm": "u_Kmm", "vv_Kmm": "v_Kmm", "ww": "ww",
                "wsd_effective": "wsd_effective", "e3t_Kmm": "e3t_Kmm",
                "e3u_Kmm": "e3u_Kmm", "e3v_Kmm": "e3v_Kmm",
                "e3w_Kmm": "e3w_Kmm", "e3t_0": "e3t_0",
                "e3u_0": "e3u_0", "e3v_0": "e3v_0", "e3w_0": "e3w_0",
                "e1e2t": "e1e2t", "e1e2u": "e1e2u", "e1e2v": "e1e2v",
                "r1_e1u": "r1_e1u", "r1_e2v": "r1_e2v",
                "r1_e1e2u": "r1_e1e2u", "r1_e1e2v": "r1_e1e2v",
                "tmask": "tmask", "umask": "umask", "vmask": "vmask",
                "wmask": "wmask",
            },
            right_is_full=False,
        ),
    }
    require(all(row["all_bit_identical"] for row in kt2_payload_identity.values()),
            f"kt2 duplicate payload moved: {kt2_payload_identity}")
    report = {
        "format": "nemo-testcase-l2-gyre-round46-kt2-stage-v1",
        "worktree": stamp,
        "mode": mode,
        "record_sha256": hashes,
        "calibration_cells_unequal": calibration,
        "legacy_records_parsed": {key: True for key in legacy_records},
        "legacy_kt1_raw_byte_identity": raw_twin,
        "legacy_kt1_consumed_identity": twin,
        "legacy_kt1_admission": twin_reports,
        "kt2_duplicate_payload_identity": kt2_payload_identity,
        "plant": plant,
        "given_inputs": [],
        "model_path_first_nonbit": {},
        "trajectory": [],
        "stage_twin": None,
        "status": "PASS",
    }
    if mode in {"given-inputs", "all"}:
        report["given_inputs"], report["model_path_first_nonbit"] = _given_inputs(
            records, plant
        )
    if mode in {"trajectory", "all"}:
        report["trajectory"] = _trajectory(records, plant)
        if any(row.get("status") != "PASS" for row in report["trajectory"]):
            report["status"] = "UNMEASURED"
    if mode == "stage-twin":
        report["stage_twin"] = _stage_twin(
            records, root, advmean_root, memory_root, btstep_root,
            stage_closure_root, plant)
        missing = [
            row for table in ("given_nemo_entry", "chained")
            for row in report["stage_twin"][table]
            if row.get("classification") == "UNMEASURED_WITH_SPEC"
        ]
        if missing:
            report["status"] = "UNMEASURED"
            report["stage_twin"]["missing_required_rows"] = [
                row["name"] for row in missing]
            # Decision 41 permits an ownership label only after the complete
            # stage contract closes.  Retain the ordering diagnostic without
            # allowing a partial table to print an owned-stage claim.
            report["stage_twin"]["first_measured_nonbit"] = (
                report["stage_twin"]["first_owned_nonbit"])
            report["stage_twin"]["first_owned_nonbit"] = None
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--expect-commit", required=True)
    p.add_argument(
        "--mode",
        choices=("validate", "given-inputs", "trajectory", "all", "stage-twin"),
        default="validate",
    )
    p.add_argument("--round40-kt1", type=Path, required=True)
    p.add_argument(
        "--round41-kt1", type=Path, default=ROUND41 / "oracle_dynadv_split_kt00000001_s3.bin"
    )
    p.add_argument("--advmean-root", type=Path, default=ADVMEAN_ROOT)
    p.add_argument("--memory-root", type=Path, default=MEMORY_ROOT)
    p.add_argument("--btstep-root", type=Path, default=BTSTEP_ROOT)
    p.add_argument(
        "--stage-closure-root", type=Path, default=STAGE_CLOSURE_ROOT)
    p.add_argument(
        "--plant",
        choices=("header", "truncation", "calibration", "given", "trajectory",
                 "twin", "stage-entry-ulp", "stage-context-ulp",
                 "stage-rhs-ulp", "stage-w-transport-ulp",
                 "stage-w-carry-ulp", "stamp"),
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    report = run(
        args.root,
        expect_commit=args.expect_commit,
        mode=args.mode,
        plant=args.plant,
        round40_kt1=args.round40_kt1,
        round41_kt1=args.round41_kt1,
        advmean_root=args.advmean_root,
        memory_root=args.memory_root,
        btstep_root=args.btstep_root,
        stage_closure_root=args.stage_closure_root,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print("STATUS", report["status"])
    return 1 if args.plant or report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
