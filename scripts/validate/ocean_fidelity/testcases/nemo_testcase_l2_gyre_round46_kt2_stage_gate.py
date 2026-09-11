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
    _surface_forcings,
    expected_masks,
    lego_fields,
    require,
)
from nemo_testcase_l2_gyre_round40_stage3_operators import read_stage3_terms
from nemo_testcase_l2_gyre_round41_dynadv_split import (
    _keg_replay,
    _model_terms,
    _zad_replay,
    read_split,
)

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage")
ROUND41 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round41/oracle_dynadv_split")
MAGIC = "NEMO_L2_R46STG1"
DIMS = (36, 26, 31)
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
                require((n1, n2, n3) == DIMS, f"{path}: bad 3-D extents for {name}")
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
        ww = _wzv_replay(a)
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


def _given_inputs(records: dict, plant: str | None) -> list[dict]:
    """Run legoESM operator routes on each complete NEMO kt=2 stage bundle."""
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
    for stage in (1, 2, 3):
        a = records[(2, stage)]["arrays"]
        un, vn = _owned3(a["u_Kmm"]), _owned3(a["v_Kmm"])
        ju = np.concatenate([un[:, -1:, :], un], axis=1)
        jv = np.concatenate([np.zeros_like(vn[:1]), vn], axis=0)
        st = card.recipe.initial_state._replace(
            u=card.recipe.initial_state.u.replace(data=jnp.asarray(ju)),
            v=card.recipe.initial_state.v.replace(data=jnp.asarray(jv)),
            T=card.recipe.initial_state.T.replace(data=jnp.asarray(_owned3(a["T_Kmm"]))),
            S=card.recipe.initial_state.S.replace(data=jnp.asarray(_owned3(a["S_Kmm"]))),
            eta=card.recipe.initial_state.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kmm"]))),
        )
        _, surface = _surface_forcings(card, st, 2)
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
                ldf_state=(state.T.data, state.S.data, state.u.data, state.v.data),
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
                        "name": f"GYRE-zco.kt2.s{stage}.{op}.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
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
                        "name": f"GYRE-zco.kt2.s{stage}.ldf.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                        "execution": "production-jit model diagnostic from NEMO stage inputs",
                    }
                )
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
                        "name": f"GYRE-zco.kt2.s{stage}.{op}_accumulator.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                    }
                )
    if plant == "given":
        require(any(r["max_abs"] > 0.5 for r in rows), "given-input plant did not land")
    return rows


def _trajectory(records: dict, plant: str | None) -> list[dict]:
    """Start with NEMO's full kt=1 end bundle and score kt=2 stage states."""
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
    u0 = _owned3(a["u_Kbb"])
    v0 = _owned3(a["v_Kbb"])
    u = np.concatenate([u0[:, -1:, :], u0], axis=1)
    v = np.concatenate([np.zeros_like(v0[:1]), v0], axis=0)
    ub0, vb0 = _owned2(a["uu_b_Kbb"]), _owned2(a["vv_b_Kbb"])
    ub = np.concatenate([ub0[:, -1:], ub0], axis=1)
    vb = np.concatenate([np.zeros_like(vb0[:1]), vb0], axis=0)
    st = card.recipe.initial_state

    def field(data, name, dims):
        return Field(data=jnp.asarray(data), name=name, dims=dims)

    st = st._replace(
        u=st.u.replace(data=jnp.asarray(u)),
        v=st.v.replace(data=jnp.asarray(v)),
        T=st.T.replace(data=jnp.asarray(_owned3(a["T_Kbb"]))),
        S=st.S.replace(data=jnp.asarray(_owned3(a["S_Kbb"]))),
        eta=st.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kbb"]))),
        uu_b=st.uu_b.replace(data=jnp.asarray(ub)),
        vv_b=st.vv_b.replace(data=jnp.asarray(vb)),
        tke=field(_owned3(a["tke_en"], 31)[..., 1:30], "tke", ("lat", "lon", "level")),
        tke_avm=field(_owned3(a["tke_avm_k"], 31)[..., 1:30], "tke_avm", ("lat", "lon", "level")),
        tke_avt=field(_owned3(a["tke_avt_k"], 31)[..., 1:30], "tke_avt", ("lat", "lon", "level")),
        tke_dissl=field(
            _owned3(a["tke_dissl"], 31)[..., 1:30], "tke_dissl", ("lat", "lon", "level")
        ),
        tke_avm_surface=field(
            _owned3(a["tke_avm_k"], 31)[..., 0], "tke_avm_surface", ("lat", "lon")
        ),
    )
    freshwater, surface = _surface_forcings(card, st, 2)
    masks = expected_masks(card)
    rows = []
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


def run(
    root: Path,
    *,
    expect_commit: str,
    mode: str,
    plant: str | None,
    round40_kt1: Path,
    round41_kt1: Path,
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
    twin = {
        "round40": new40.read_bytes() == round40_kt1.read_bytes(),
        "round41": new41.read_bytes() == round41_kt1.read_bytes(),
    }
    if plant == "twin":
        twin["round40"] = False
    require(all(twin.values()), f"deterministic kt1 legacy twin moved: {twin}")
    report = {
        "format": "nemo-testcase-l2-gyre-round46-kt2-stage-v1",
        "worktree": stamp,
        "mode": mode,
        "record_sha256": hashes,
        "calibration_cells_unequal": calibration,
        "legacy_records_parsed": {key: True for key in legacy_records},
        "legacy_kt1_byte_identity": twin,
        "plant": plant,
        "given_inputs": [],
        "trajectory": [],
        "status": "PASS",
    }
    if mode in {"given-inputs", "all"}:
        report["given_inputs"] = _given_inputs(records, plant)
    if mode in {"trajectory", "all"}:
        report["trajectory"] = _trajectory(records, plant)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--expect-commit", required=True)
    p.add_argument(
        "--mode", choices=("validate", "given-inputs", "trajectory", "all"), default="validate"
    )
    p.add_argument("--round40-kt1", type=Path, required=True)
    p.add_argument(
        "--round41-kt1", type=Path, default=ROUND41 / "oracle_dynadv_split_kt00000001_s3.bin"
    )
    p.add_argument(
        "--plant",
        choices=("header", "truncation", "calibration", "given", "trajectory", "twin", "stamp"),
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
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print("STATUS", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
