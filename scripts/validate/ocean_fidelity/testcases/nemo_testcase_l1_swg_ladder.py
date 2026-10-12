#!/usr/bin/env python3
"""SWG bit-exact ladder: ten-step stages and the admitted thirty-day rung."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_phase3_trajectory_gate import BAR, require, score  # noqa: E402

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round0_probe/kt10")
DEFAULT_STAGE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round1/oracle_swg_r1/run")
DEFAULT_MONTH_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round6/oracle_swg_r6_30d")
DEFAULT_SHIPPED_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round0_probe/year10")
MONTH_RECORD_KTS = tuple(range(1, 698, 24))
MONTH_RESTART_KTS = tuple(range(24, 721, 24))
TEN_STEP_ADMISSION_SHA256 = (
    "cdf8afc272c962635564110111b10386b79d8afdc2373fdf0df3c5860dbd3937")
MONTH_ADMISSION_SHA256 = (
    "94db6dfea297fe828c55dc6c26fa78702cbbf9df399aa8f6793d252f33efae8e")
TEN_STEP_RESTART_SHA256 = dict(zip(range(1, 11), (
    "bdadad749f29e868a1910cac7e0d98458cd0f40300d60d92c9bda0e992d8fd37",
    "122be763d83d2ec800038b84ac4db5d86b6a66dfd3970a3eae9a902141a69346",
    "d4341ef17879fe61045ac99e47f535b8002a02ac75c9b856837f5ab2ceac7634",
    "093c6eb7475e8874018c16af644247c29c03e4c7d1d7c844b603efa776cfff72",
    "008e49e294db2a32f14fbee87f00beb7e6be47f0f7877d1b06aea62dc211cfb4",
    "7f067d0f720b70b687c66d59f51ee6bd62450105e4380acf77a024ef754e8921",
    "c49ca7d7f9d6df89862e3a984d630c312bae405742bac8fe5bb88104164c6c25",
    "27f4fe2ea827fcf0ff28546f558b2628eb3c8604e8c1f3744bbaed26edb07419",
    "a7f51a31a7bf95f004c9bda18e12668428b14e28d2910aedebc58f73b64dc8c0",
    "bf6b1bbbc0ccd70bcedc6daf80458b2ee9f0f1566d5ef1bdcdf19aa0bc21fb8e",
), strict=True))
MONTH_RESTART_SHA256 = dict(zip(MONTH_RESTART_KTS, (
    "4ff9482e27e6e95783f1b1d9d2a6de1b8d2cffc8066dbfd32b5598d191526a89",
    "0755328132ee122110b9d249f6e4cfc860bafaa342750aae19521c4cad3aa9ea",
    "bbc00865241a25057e810a1d8e5832798dc6986cb9b0b88f1d6c6845f78be208",
    "3987f42de0379166c4fd2af0c5968f98a6867e1664fca76aaea0fe6d19c68527",
    "2a88db60696c3c5c79516846296bb22272eff4ddaee558a935f327bf5d43c2f6",
    "6b62e347425619d477ea182d54fc1f73ce8c85ff8ecec649362c7fc5be95d511",
    "1898d8126ef54df6abdf734da8038b541cb33fef1d1bafb6e20839ba9ad4ccba",
    "28bc61309a18e2b68edb13be29648db8cc9593c4b6eaf65b89b88a24d9f9e366",
    "c3171be6dfab5519cded6db40efc051f23160ddd3cbf5c63052801bbf5862651",
    "5a46dc1ca5c9335a759a1f41028dc77054468cf8d788ae6d641fa2988d7afe35",
    "ab90eb34f4b974fc9d54cddc92af757fd39d30457aacfd07700f7abac7a2b507",
    "6f0ed77151b44ed7e66f01c0e429f774487991af088bf9e518bf03e8a9fa3997",
    "b9d26d33a22150da0debf5d6d58cdd125da6a68284c5db9bf1d6feaacf623530",
    "d5e89f4b2ac81ec4dc7762c42d23a8509d2ec1b0ab388d203a4ed67ce0204419",
    "88bd2952602d4ad1991d168721a8c2ae3c90f240b8ba1ac044273205a79bf078",
    "c97eafaa044e34681432e1e6f13ad30834de0c5e8a2233f112e4180b7a531cdd",
    "e636772fd5faf8d7ed06ce7edc73d830bc852bee822e6bb4d5ef93a5e521d3d9",
    "cdbd6e508cd42c9e88b48f0d7f27a47242d779ea9b0214874dde36361c5e0e16",
    "e546c861884cf91988de93a49c9ff9ad0963b62f517e948193bffca56f7227dd",
    "45472792038f21637a57312181abeb4b566da28274184e713017d535a177837a",
    "e63b6e69ca7292fba96d96d0de76921733255b9d337eaaf05329ec49c61e8c73",
    "d64d901ba8bbd743b538e8f19d27e54bcf8d09f91cd1f73f7bfd5915b86e71c3",
    "971b9db699f8406b79bc9ff363b6f7c2002b480683a5a0f4b9aa7d9a88c456b1",
    "e07e570b3cfa9add7a2bd1e1f397348309560875733cdd065422ecc4b95d6440",
    "bd3a264169fd0ceac9052b3753d07e5b7d0d84184122667552f620daa6546812",
    "2ee72e51f5ec197c782a766d54cb38a2eda479a7c8b1539f751646d162d34d7b",
    "b5ea123a4e3dbe87c5bed3b84e7efb284217c6c7da224827862ca93cfef3079d",
    "23191678cc881d8e711456953c23499554dd624e1fd7389e9ba397dbc7757ebb",
    "0db212ecee876aa2fb93791184e1962c2c484d5bd47b59cc127ecfb77b4c23bc",
    "350c3644181bd4b75cabc3ebbab64c66eb05d3fbe107ca709561a08f4f465deb",
), strict=True))
MONTH_OUTPUT_SHA256 = {
    "instrumented/SWG_1d_00010101_00010130_grid_T_0000.nc":
        "f30b644d44304d607d0f1e139a573d7d4b748c88f67e3139161bfac73ef759d4",
    "instrumented/SWG_1d_00010101_00010130_grid_U_0000.nc":
        "0353fa985a2e309966cd32801597cf859c0de5b9355154a58e511847bc3225a8",
    "instrumented/SWG_1d_00010101_00010130_grid_V_0000.nc":
        "0d1594b9c68edb5029addc09e9a8d544fbd777abdc582d396a18fd04dccef8d7",
    "instrumented/SWG_1d_00010101_00010130_grid_W_0000.nc":
        "f1e84e5af46a5d1064e002b7e30390e0dafa37c8d4883175191bfea24fc03a19",
    "reference/SWG_1d_00010101_00010130_grid_T_0000.nc":
        "355ef4c3e877a1796b511e49848e350a1f5ce125897721265bd7d64315d74b7a",
    "reference/SWG_1d_00010101_00010130_grid_U_0000.nc":
        "26d8c73a09f9f414c5184407803c778d73e26b3e2002111d3264b9e18efd5672",
    "reference/SWG_1d_00010101_00010130_grid_V_0000.nc":
        "612666a199c7396134e3f1878755bcefc99fbaabbc4993ddda83fb10140e7e15",
    "reference/SWG_1d_00010101_00010130_grid_W_0000.nc":
        "5d700936098ed919cc726a9348d7766a400ac3ad2b93ddcb870075262beb976b",
}
SHIPPED_SHA256 = {
    "manifest.sha256":
        "22b46f1496c51da574203b0bbb67887351c22325806a184c30d643429dbc15b0",
    "SWG_00172800_restart.nc":
        "eb800fee0dfe0723fc7c33c1338142fbeaac1a1a9f6899a38e71e7850894abd8",
    "SWG_20d_00010101_00201230_grid_T_0000.nc":
        "42263cf6e6a63d770112eb90bea73bd52eb64546d862c5635d77cf706022d948",
    "SWG_20d_00010101_00201230_grid_U_0000.nc":
        "5b2d38e00234027e89a60bc1d3045ec9389c8da58dad40c61ffc7bc7e5d6e0f0",
    "SWG_20d_00010101_00201230_grid_V_0000.nc":
        "ee9e8dc4ed39574cbce4df5627d59fc42efc72ffc7da565830f844f141fccc0e",
    "SWG_20d_00010101_00201230_grid_W_0000.nc":
        "25e3a3f4972571e908fc98984602f6f6f9d606616b2222cbcbd64eff5b979dcb",
    "mesh_mask.nc":
        "4e03dbf1211dbb299d2f6411106dd7a4ff491f8c25101491eb215845058d819b",
    "output.init.nc":
        "19e12d4fb9aaed96a24488d026a3738ad4b2dbfd766ba0771c7f2d6bda8b4a19",
    "ocean.output":
        "af7c4272dece97042aab67beaa81ca4d702f9c516703e24586c4468ace6cbd26",
}
SHIPPED_MANIFEST = {
    "nemo": "a5b312b8b8fccadd67152d446811c7d2bc9d0e1cd55dce0d3148c8b5c6817eb9",
    "namelist_cfg":
        "43c81e7ad7b92bf491aa74f56636800a5a0b3d2fda7c61ba03f4ff6942a1893b",
    "namelist_ref":
        "b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd",
}
SHIPPED_RESTART_SHA256 = {172800: SHIPPED_SHA256["SWG_00172800_restart.nc"]}
RESTART_FIELDS = (
    "sshn", "un", "vn", "uu_n", "vv_n", "utau_b", "vtau_b",
    "emp_b", "qns_b", "sfx_b",
)
# Compiled sbc_oce.f90:116-117 defines the before stresses at U/V, and
# sbcmod.f90:567-568 reads the restart records with cd_type='U'/'V'.
U_FIELDS = frozenset(("un", "uu_n", "utau_b"))
V_FIELDS = frozenset(("vn", "vv_n", "vtau_b"))
STAGE_U_FIELDS = frozenset((
    "u_k", "r3u_k", "u_before", "utau_before", "utau_now"))
STAGE_V_FIELDS = frozenset((
    "v_k", "r3v_k", "v_before", "vtau_before", "vtau_now"))
STAGE_F_FIELDS = frozenset(("r3f",))


def _stage_checker():
    path = HERE / "nemo_testcase_l1_swg/check_records.py"
    spec = importlib.util.spec_from_file_location("swg_check_records", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(path: Path, expected: str) -> None:
    require(path.is_file(), f"missing content-pinned record {path}")
    require(_sha256(path) == expected, f"content SHA-256 mismatch: {path}")


def _pinned_stage_hashes(root: Path) -> dict[str, str]:
    """Load the immutable admission that owns one known stage-record root."""
    if root.resolve() == DEFAULT_STAGE_ROOT.resolve():
        path = root.parent / "swg_round1_admission.json"
        expected = TEN_STEP_ADMISSION_SHA256
    elif root.resolve() == (DEFAULT_MONTH_ROOT / "instrumented").resolve():
        path = root.parent / "swg_round6_30d_admission.json"
        expected = MONTH_ADMISSION_SHA256
    else:
        raise RuntimeError(f"missing admission for unregistered stage-record root {root}")
    _require_sha256(path, expected)
    return json.loads(path.read_text())["records"]


def _restart_hashes(root: Path) -> dict[int, str]:
    if root.resolve() == DEFAULT_ROOT.resolve():
        return TEN_STEP_RESTART_SHA256
    if root.resolve() == (DEFAULT_MONTH_ROOT / "instrumented").resolve():
        return MONTH_RESTART_SHA256
    if root.resolve() == DEFAULT_SHIPPED_ROOT.resolve():
        return SHIPPED_RESTART_SHA256
    raise RuntimeError(f"unregistered restart-record root {root}")


def read_restarts(
        root: Path,
        kts: tuple[int, ...] = tuple(range(1, 11)),
) -> dict[int, dict[str, np.ndarray]]:
    """Read requested restart AFTER states, refusing omissions or casts."""
    import netCDF4
    files = sorted(root.glob("SWG_????????_restart.nc"))
    expected = [root / f"SWG_{kt:08d}_restart.nc" for kt in kts]
    require(files == expected, f"restart sequence is {[p.name for p in files]}")
    pins = _restart_hashes(root)
    out = {}
    for kt, path in zip(kts, files, strict=True):
        require(kt in pins, f"no content pin for restart kt={kt}")
        _require_sha256(path, pins[kt])
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            file_kt = int(np.asarray(ds["kt"][:]).item())
            require(file_kt == kt, f"{path.name}: kt={file_kt}")
            require(float(np.asarray(ds["rdt"][:]).item()) == 3600.0,
                    f"{path.name}: rdt is not 3600 s")
            row = {}
            for name in RESTART_FIELDS:
                require(name in ds.variables, f"{path.name}: missing {name}")
                var = ds[name]
                require(var.dtype == np.dtype("float64"),
                        f"{path.name}.{name}: dtype {var.dtype}")
                a = np.asarray(var[0], dtype=np.float64)
                if name in ("un", "vn"):
                    require(a.shape[0] == 2, f"{path.name}.{name}: no dummy record")
                    a = a[0]                 # executed level; crop dummy bottom
                require(a.shape == (84, 84) and np.all(np.isfinite(a)),
                        f"{path.name}.{name}: bad shape/non-finite")
                row[name] = a
            out[kt] = row
    return out


def read_stage_entries(
        root: Path,
        record_kts: tuple[int, ...] = tuple(range(1, 11)),
        stages: tuple[int, ...] = (1, 2, 3),
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Read all acquired stage entries through the admission parser."""
    cr = _stage_checker()
    pins = _pinned_stage_hashes(root)
    out = {}
    for kt in record_kts:
        for stage in stages:
            path = root / f"oracle_swgstage_kt{kt:08d}_s{stage}.bin"
            require(path.is_file(), f"missing {path}")
            require(path.name in pins, f"no content pin for {path.name}")
            _require_sha256(path, pins[path.name])
            head, groups = cr.parse(path)
            require((head[1], head[2]) == (kt, stage),
                    f"{path.name}: header {(head[1], head[2])}")
            is0, js0, ie0, je0 = head[cr.BOUNDS_AT:cr.BOUNDS_AT + 4]
            out[kt, stage] = {
                name: np.asarray(a[js0 - 1:je0, is0 - 1:ie0])
                for name, a in groups.items()
            }
    return out


def _card_masks(card) -> dict[str, np.ndarray]:
    """Return native NEMO T/U/V/F masks from shared card operands."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks

    tmask = np.asarray(card.recipe.land_mask, dtype=bool)
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    fmask = np.asarray(
        card.recipe.z_coord.nemo_een_barotropic.fe3mask[..., 0], dtype=bool)
    return {
        "T": tmask,
        "U": np.asarray(u_full[:, 1:], dtype=bool),
        "V": np.asarray(v_full[1:, :], dtype=bool),
        "F": fmask,
    }


def _field_mask(field: str, masks: dict[str, np.ndarray]) -> np.ndarray:
    return masks[
        "U" if field in U_FIELDS | STAGE_U_FIELDS
        else "V" if field in V_FIELDS | STAGE_V_FIELDS
        else "F" if field in STAGE_F_FIELDS else "T"]


def _row(name, ref, cand, mask):
    ref, cand = np.asarray(ref), np.asarray(cand)
    use = np.asarray(mask, dtype=bool)
    result = score(name, ref, cand, use)
    delta = cand - ref
    bad = np.argwhere((ref != cand) & use)
    result.update({
        "bit_identical": bool(bad.size == 0),
        "n_unequal": int(bad.shape[0]),
        "first_unequal_cell": bad[0].tolist() if bad.size else None,
        "max_abs": float(np.max(np.abs(delta[use]))),
    })
    # This campaign's requested endpoint is bit-for-bit.  Retain the TSUNAMI
    # 1e-15 numerical score, but never let a non-bit row claim AT-BAR.
    if bad.size and result["status"] == "AT-BAR":
        result["numerical_status"] = "AT-BAR"
        result["status"] = "DEBT"
        result["reason"] = "bit-for-bit SWG ladder: non-identical value"
    return result


def _gate_rows(out: dict) -> dict:
    rows = [row for step in out.get("per_kt", ()) for row in step["rows"]]
    controls = list(out.get("controls", ()))
    gated = rows + controls
    out["all_at_bar"] = bool(gated) and all(
        row["status"] == "AT-BAR" for row in gated)
    out["all_bit_identical"] = bool(gated) and all(
        row["bit_identical"] for row in gated)
    out["first_kt_over_bar"] = next(
        (step["kt"] for step in out.get("per_kt", ())
         if any(row["status"] == "DEBT" for row in step["rows"])), None)
    out["first_day_over_bar"] = next(
        (step["day"] for step in out.get("per_kt", ()) if "day" in step
         and any(row["status"] == "DEBT" for row in step["rows"])), None)
    return out


def _exit_code(out: dict) -> int:
    return 0 if out.get("all_at_bar", True) else 1


def _stage3_input_audit(entries, masks) -> dict:
    """Audit the recorded kt=2 stage-3 wind and whole-step inputs."""
    stage3 = entries[2, 3]
    step_entry = entries[2, 1]

    def compare(left, right, mask):
        use = np.asarray(mask, dtype=bool)
        delta = np.asarray(left) - np.asarray(right)
        unequal = (np.asarray(left) != np.asarray(right)) & use
        return {
            "bit_identical": bool(not np.any(unequal)),
            "n_unequal": int(np.count_nonzero(unequal)),
            "max_abs": float(np.max(np.abs(delta[use]))),
        }

    return {
        "wind_before_vs_now": {
            "u": compare(stage3["utau_before"], stage3["utau_now"], masks["U"]),
            "v": compare(stage3["vtau_before"], stage3["vtau_now"], masks["V"]),
        },
        "whole_step_entry_vs_stage3_entry": {
            "u_k": compare(step_entry["u_k"], stage3["u_k"], masks["U"]),
            "v_k": compare(step_entry["v_k"], stage3["v_k"], masks["V"]),
            "r3f": compare(step_entry["r3f"], stage3["r3f"], masks["F"]),
        },
    }


def self_check(root: Path, *, plant: bool = False, allow_dirty: bool = False) -> dict:
    """Calibrate restart decoding only; this is not a fidelity measurement."""
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
    masks = _card_masks(build_swg_card())
    records = read_restarts(root)
    per_kt = []
    for kt, fields in records.items():
        rows = []
        for name, ref in fields.items():
            cand = ref.copy()
            if plant and kt == 1 and name == "sshn":
                cand[42, 42] += 1.0e-3
            rows.append(_row(f"self_check.kt{kt}.{name}", ref, cand,
                             _field_mask(name, masks)))
        per_kt.append({"kt": kt, "rows": rows})
    return _gate_rows({
        "label": "RESTART-READER-SELF-CHECK (not fidelity)",
        "legoesm_git_sha": sha, "bar": BAR, "plant": plant,
        "per_kt": per_kt,
    })


def _swg_production_case(label: str):
    """Build the one independently forced production case used by all rungs."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_NAMELIST, build_swg_card, swg_horizontal_coordinates,
        validate_nemo_testcase_card_for_execution)
    from legoesm.ocean.state import OceanSurfaceForcing

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", f"{label} must run on CPU")
    card = build_swg_card()
    validate_nemo_testcase_card_for_execution(card)
    masks = _card_masks(card)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config)
    src = swg_horizontal_coordinates()
    # Compiled usrdef_sbc.f90:99-106, with the shipped theta=0 deck.  This is
    # generated from the card, never read from an oracle stage/restart record.
    tau_u = (-np.float64(SWG_NAMELIST.rn_tau_pa)
             * np.cos(np.float64(np.pi) * src["gphit"]
                      / np.float64(SWG_NAMELIST.rn_domsiz_m)))
    tau_v = np.zeros_like(tau_u)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64))
    return card, masks, model, forcing


def _month_admission(root: Path) -> Path:
    """Reproduce the saved passive-record admission before scoring the month."""
    admission_path = root / "swg_round6_30d_admission.json"
    instrumented = root / "instrumented"
    reference = root / "reference"
    require(admission_path.is_file(), f"missing month admission {admission_path}")
    _require_sha256(admission_path, MONTH_ADMISSION_SHA256)
    saved = json.loads(admission_path.read_text())
    live = _stage_checker().check(
        instrumented, reference, record_kts=MONTH_RECORD_KTS, stages=(1,),
        restart_kts=MONTH_RESTART_KTS)
    require(live == saved, "live month admission differs from its saved verdict")
    require(saved["legoesm_git_sha"]
            == "011ba6c48c2243467097af35a66a041cdee3f565",
            "month acquisition commit is not the admitted round-6 tip")
    for side in (instrumented, reference):
        for kt, expected in MONTH_RESTART_SHA256.items():
            _require_sha256(side / f"SWG_{kt:08d}_restart.nc", expected)
    for relative, expected in MONTH_OUTPUT_SHA256.items():
        _require_sha256(root / relative, expected)
    return instrumented


def _shipped_admission(root: Path) -> dict[str, Path]:
    """Content-pin the operator's uninstrumented shipped reference run."""
    for relative, expected in SHIPPED_SHA256.items():
        _require_sha256(root / relative, expected)
    manifest = {}
    for line in (root / "manifest.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        manifest[name] = digest
    require(manifest == SHIPPED_MANIFEST,
            f"shipped manifest contents differ: {manifest}")
    for name, expected in SHIPPED_MANIFEST.items():
        _require_sha256(root / name, expected)
    return {
        kind: root / next(name for name in SHIPPED_SHA256
                          if f"grid_{kind}_" in name)
        for kind in ("T", "U", "V", "W")
    }


def _read_shipped_outputs(root: Path, block_steps: int, dt_s: float):
    """Read the three 32-bit averaged fields after immutable admission."""
    import netCDF4

    paths = _shipped_admission(root)
    fields = {"T": "sossheig", "U": "vozocrtx", "V": "vomecrty"}
    out = {}
    expected_time = ((np.arange(360, dtype=np.float64) + 0.5)
                     * np.float64(block_steps) * np.float64(dt_s))
    for kind, variable in fields.items():
        with netCDF4.Dataset(paths[kind]) as ds:
            ds.set_auto_mask(False)
            require(variable in ds.variables, f"{paths[kind].name}: missing {variable}")
            var = ds[variable]
            require(var.dtype == np.dtype("float32"),
                    f"{paths[kind].name}.{variable}: dtype {var.dtype}")
            values = np.asarray(var[:])
            if kind in ("U", "V"):
                require(values.shape == (360, 2, 84, 84),
                        f"{variable}: shape {values.shape}")
                require(np.all(values[:, 1] == 0.0),
                        f"{variable}: dummy bottom record is nonzero")
                values = values[:, 0]
            require(values.shape == (360, 84, 84)
                    and np.all(np.isfinite(values)),
                    f"{variable}: bad shape/non-finite stored value")
            times = np.asarray(ds["time_counter"][:], dtype=np.float64)
            require(np.array_equal(times, expected_time),
                    f"{paths[kind].name}: output clock differs")
            out[kind] = values
    return out


def _ioipsl_moycum(previous: np.ndarray | None, sample: np.ndarray,
                   previous_count: int) -> np.ndarray:
    """IOIPSL ``moycum`` average with its exact source association."""
    value = np.asarray(sample)
    require(value.dtype == np.dtype("float64"), "moycum sample is not fp64")
    if previous_count == 0:
        require(previous is None, "moycum first sample has a prior accumulator")
        return value.copy()
    require(previous is not None and previous.dtype == np.dtype("float64"),
            "moycum accumulator is not fp64")
    return ((previous * np.float64(previous_count) + value)
            / np.float64(previous_count + 1))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _shared_shipped_statistics(ssh, u_native, v_native, card):
    """Apply only repository-shared diagnostics to one masked output series."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import (
        domain_kinetic_energy, eddy_kinetic_energy,
        isotropic_energy_spectrum, velocity_to_cell_centre)
    from legoesm.ocean.diagnostics_streamfunction import barotropic_streamfunction
    from legoesm.ocean.fidelity.nemo_testcase_recipe import swg_horizontal_coordinates

    masks = _card_masks(card)
    ssh = np.where(masks["T"][None], np.asarray(ssh, dtype=np.float64), 0.0)
    u_native = np.where(
        masks["U"][None], np.asarray(u_native, dtype=np.float64), 0.0)
    v_native = np.where(
        masks["V"][None], np.asarray(v_native, dtype=np.float64), 0.0)
    u = np.concatenate([np.zeros_like(u_native[:, :, :1]), u_native], axis=2)
    v = np.concatenate([np.zeros_like(v_native[:, :1, :]), v_native], axis=1)
    u_h, v_h = jax.vmap(velocity_to_cell_centre)(jnp.asarray(u), jnp.asarray(v))
    base_h = np.asarray(card.recipe.z_coord.h_partial, dtype=np.float64)
    h = np.where(masks["T"][None, ..., None],
                 base_h[None] + ssh[..., None], 0.0)
    area = jnp.asarray(card.recipe.grid.area, dtype=jnp.float64)
    ke = np.asarray(jax.vmap(
        lambda ui, vi, hi: domain_kinetic_energy(ui, vi, area, hi))(
            u_h[..., None], v_h[..., None], jnp.asarray(h)))
    eke = np.asarray(jax.vmap(
        lambda ui, vi, hi: eddy_kinetic_energy(ui, vi, area, hi))(
            u_h[..., None], v_h[..., None], jnp.asarray(h)))
    mean_ssh = np.mean(ssh, axis=0, dtype=np.float64)
    mean_u = np.mean(u, axis=0, dtype=np.float64)
    mean_v = np.mean(v, axis=0, dtype=np.float64)
    mean_h = np.where(masks["T"][..., None],
                      base_h + mean_ssh[..., None], 0.0)
    psi = np.asarray(barotropic_streamfunction(
        mean_u[..., None], mean_h, masks["T"], card.recipe.grid))
    mean_u_h, mean_v_h = velocity_to_cell_centre(
        jnp.asarray(mean_u), jnp.asarray(mean_v))
    wet = np.argwhere(masks["T"])
    j0, i0 = wet.min(axis=0); j1, i1 = wet.max(axis=0) + 1
    require((int(j1 - j0), int(i1 - i0)) == (80, 80),
            "SWG wet spectral basin is not 80 by 80")
    spacing = np.float64(swg_horizontal_coordinates()["spacing_m"])
    wavenumber, spectrum = isotropic_energy_spectrum(
        mean_u_h[j0:j1, i0:i1], mean_v_h[j0:j1, i0:i1], spacing)
    arrays = {
        "mean_ssh": mean_ssh,
        "transport_streamfunction_sv": psi,
        "kinetic_energy_series": ke,
        "eddy_kinetic_energy_series": eke,
        "spectrum_wavenumber_m-1": np.asarray(wavenumber),
        "kinetic_energy_spectrum": np.asarray(spectrum),
    }
    wet_ssh = mean_ssh[masks["T"]]
    summary = {
        "mean_ssh_m": {
            "wet_mean": float(np.mean(wet_ssh, dtype=np.float64)),
            "wet_min": float(np.min(wet_ssh)),
            "wet_max": float(np.max(wet_ssh)),
            "sha256": _array_sha256(mean_ssh),
        },
        "transport_streamfunction_sv": {
            "min": float(np.min(psi)), "max": float(np.max(psi)),
            "sha256": _array_sha256(psi),
        },
        "kinetic_energy_m5_s-2": {
            "time_mean": float(np.mean(ke, dtype=np.float64)),
            "min": float(np.min(ke)), "max": float(np.max(ke)),
            "sha256": _array_sha256(ke),
        },
        "eddy_kinetic_energy_m5_s-2": {
            "time_mean": float(np.mean(eke, dtype=np.float64)),
            "min": float(np.min(eke)), "max": float(np.max(eke)),
            "sha256": _array_sha256(eke),
        },
        "basin_ke_spectrum": {
            "bins": int(np.asarray(spectrum).size),
            "sha256": _array_sha256(np.asarray(spectrum)),
            "wavenumber_sha256": _array_sha256(np.asarray(wavenumber)),
        },
    }
    return arrays, summary


def _stage_candidate(state, forcing, card) -> dict[str, np.ndarray]:
    """Reconstruct the production entry represented by a public whole-step state."""
    from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state

    now = (forcing.tau_i_native, forcing.tau_j_native)
    before = (
        now[0] if state.tau_x_prev is None else state.tau_x_prev,
        now[1] if state.tau_y_prev is None else state.tau_y_prev,
    )
    entry = _entry_from_state(
        state, before, now, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config)
    return {name: np.asarray(value) for name, value in entry._asdict().items()}


def _restart_candidate(state) -> dict[str, np.ndarray]:
    """Map the public SWG state to the ten admitted restart arrays."""
    eta = np.asarray(state.eta.data)
    require(state.tau_x_prev is not None and state.tau_y_prev is not None,
            "restart boundary lacks the before stress")
    zero = np.zeros_like(eta)
    return {
        "sshn": eta,
        "un": np.asarray(state.u.data)[:, 1:, 0],
        "vn": np.asarray(state.v.data)[1:, :, 0],
        "uu_n": np.asarray(state.uu_b.data)[:, 1:],
        "vv_n": np.asarray(state.vv_b.data)[1:, :],
        "utau_b": np.asarray(state.tau_x_prev),
        "vtau_b": np.asarray(state.tau_y_prev),
        "emp_b": zero,
        "qns_b": zero,
        "sfx_b": zero,
    }


def month(root: Path, *, allow_dirty: bool = False) -> dict:
    """Run 720 independent production steps and score both daily record kinds."""
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    instrumented = _month_admission(root)
    entries = read_stage_entries(
        instrumented, record_kts=MONTH_RECORD_KTS, stages=(1,))
    restarts = read_restarts(instrumented, kts=MONTH_RESTART_KTS)
    card, masks, model, forcing = _swg_production_case("month ladder")
    state = card.recipe.initial_state
    per_day = []
    for day, (entry_kt, restart_kt) in enumerate(
            zip(MONTH_RECORD_KTS, MONTH_RESTART_KTS, strict=True), 1):
        candidate_entry = _stage_candidate(state, forcing, card)
        require(all(value.dtype == np.dtype("float64")
                    for value in candidate_entry.values()),
                "month stage candidate is not fp64")
        rows = [
            _row(f"month.day{day:02d}.entry.kt{entry_kt}.{name}",
                 entries[entry_kt, 1][name], value,
                 _field_mask(name, masks))
            for name, value in candidate_entry.items()
        ]
        for _ in range(24):
            state = model.step(state, card.dt_s, surface_forcing=forcing)
        candidate_restart = _restart_candidate(state)
        require(all(value.dtype == np.dtype("float64")
                    for value in candidate_restart.values()),
                "month restart candidate is not fp64")
        rows.extend(
            _row(f"month.day{day:02d}.restart.kt{restart_kt}.{name}",
                 restarts[restart_kt][name], value,
                 _field_mask(name, masks))
            for name, value in candidate_restart.items()
        )
        per_day.append({"day": day, "kt": restart_kt, "rows": rows})
    return _gate_rows({
        "label": "INDEPENDENT-THIRTY-DAY",
        "legoesm_git_sha": sha,
        "oracle_acquisition_git_sha": (
            "011ba6c48c2243467097af35a66a041cdee3f565"),
        "bar": BAR,
        "record_counts": {"stage_entries": len(entries),
                          "restarts": len(restarts)},
        "per_kt": per_day,
    })


def month_control(root: Path, *, allow_dirty: bool = False) -> dict:
    """Prove one ULP at an exact daily U input changes production output."""
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    instrumented = _month_admission(root)
    restarts = read_restarts(instrumented, kts=MONTH_RESTART_KTS)
    card, masks, model, forcing = _swg_production_case("month control")
    state = card.recipe.initial_state
    for _ in range(24):
        state = model.step(state, card.dt_s, surface_forcing=forcing)
    clean = _restart_candidate(state)
    clean_rows = [
        _row(f"month_control.day01.clean.{name}", restarts[24][name], value,
             _field_mask(name, masks))
        for name, value in clean.items()
    ]
    require(all(row["bit_identical"] for row in clean_rows),
            "month control input is not the admitted day-one boundary")

    native_u = np.asarray(state.u.data)[:, 1:, 0]
    wet_nonzero = np.argwhere(masks["U"] & (native_u != 0.0))
    require(wet_nonzero.size > 0,
            "admitted day-one U field has no nonzero wet value")
    at = tuple(wet_nonzero[np.argmax(
        np.abs(native_u[tuple(wet_nonzero.T)]))])
    original = np.float64(native_u[at])
    planted = np.nextafter(original, np.float64(np.inf))
    require(original != 0.0 and planted != original,
            "month nextafter plant is not one binding ULP")
    planted_u = state.u.data.at[at[0], at[1] + 1, 0].set(
        jnp.asarray(planted, dtype=state.u.data.dtype))
    moved_state = state._replace(u=state.u.replace(data=planted_u))
    baseline = model.step(state, card.dt_s, surface_forcing=forcing)
    moved = model.step(moved_state, card.dt_s, surface_forcing=forcing)
    output_rows = [
        _row("month_control.output.ssh", baseline.eta.data, moved.eta.data,
             masks["T"]),
        _row("month_control.output.u", baseline.u.data[:, 1:, 0],
             moved.u.data[:, 1:, 0], masks["U"]),
        _row("month_control.output.v", baseline.v.data[1:, :, 0],
             moved.v.data[1:, :, 0], masks["V"]),
    ]
    changed = next((row for row in output_rows if not row["bit_identical"]), None)
    require(changed is not None,
            "one-ULP daily input plant did not change a production output")
    changed.update({
        "input_field": "u",
        "input_cell": [int(at[0]), int(at[1])],
        "input_before": float(original),
        "input_after": float(planted),
        "input_ulp_delta": 1,
        "input_selection": "largest_absolute_nonzero_wet_u",
        "input_matches_admitted_boundary": True,
    })
    return _gate_rows({
        "label": "INDEPENDENT-THIRTY-DAY-CONTROL",
        "legoesm_git_sha": sha,
        "bar": BAR,
        "plant": "ulp",
        "controls": [changed],
        "per_kt": [{"day": 1, "kt": 24, "rows": clean_rows}],
    })


def shipped(root: Path, *, statistics_npz: Path | None = None,
            progress: bool = False, allow_dirty: bool = False) -> dict:
    """Run and score all 172800 shipped production steps from rest."""
    import resource
    import time
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    card, masks, model, forcing = _swg_production_case("shipped ladder")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_NAMELIST
    # The shared model does not own output cadence; this scorer consumes the
    # REQUIRED resolved deck choice rather than supplying a local default.
    block_steps = int(SWG_NAMELIST.nn_write)
    require(block_steps == 480, f"shipped output block is {block_steps} steps")
    require(card.n_steps == 172800 and card.n_steps % block_steps == 0,
            f"shipped step count is {card.n_steps}")
    n_outputs = card.n_steps // block_steps
    reference = _read_shipped_outputs(root, block_steps, card.dt_s)
    final_restart = read_restarts(root, kts=(card.n_steps,))[card.n_steps]

    # Compile all three production stages before starting the wall clock, then
    # discard the warm-up state so the certified trajectory still starts from
    # the card's independent rest state.
    warm = model.step(
        card.recipe.initial_state, card.dt_s, surface_forcing=forcing)
    warm.eta.data.block_until_ready()
    state = card.recipe.initial_state
    candidate = {
        kind: np.empty((n_outputs, 84, 84), dtype=np.float32)
        for kind in ("T", "U", "V")
    }
    per_output = []
    rss_start_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.perf_counter()
    averages = {kind: None for kind in candidate}
    previous_count = 0
    for kt in range(1, card.n_steps + 1):
        state = model.step(state, card.dt_s, surface_forcing=forcing)
        samples = {
            "T": np.asarray(state.eta.data),
            "U": np.asarray(state.u.data)[:, 1:, 0],
            "V": np.asarray(state.v.data)[1:, :, 0],
        }
        averages = {
            kind: _ioipsl_moycum(averages[kind], value, previous_count)
            for kind, value in samples.items()
        }
        previous_count += 1
        if kt % block_steps:
            continue
        output_index = kt // block_steps
        rows = []
        for kind in ("T", "U", "V"):
            stored = averages[kind].astype(np.float32)
            candidate[kind][output_index - 1] = stored
            rows.append(_row(
                f"shipped.output{output_index:04d}.kt{kt}.{kind}",
                reference[kind][output_index - 1], stored, masks[kind]))
        per_output.append({
            "kind": "output", "output_index": output_index,
            "kt": kt, "model_day": kt * card.dt_s / 86400.0,
            "rows": rows,
        })
        averages = {kind: None for kind in candidate}
        previous_count = 0
        if progress and output_index % 18 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"PROGRESS model_year={output_index // 18}/20 "
                f"steps={kt}/{card.n_steps} wall_s={elapsed:.1f}",
                file=sys.stderr, flush=True)

    state.eta.data.block_until_ready()
    wall_s = time.perf_counter() - started
    restart_candidate = _restart_candidate(state)
    restart_rows = [
        _row(f"shipped.restart.kt{card.n_steps}.{name}",
             final_restart[name], value, _field_mask(name, masks))
        for name, value in restart_candidate.items()
    ]
    per_output.append({
        "kind": "restart", "kt": card.n_steps, "rows": restart_rows})
    out = _gate_rows({
        "label": "INDEPENDENT-SHIPPED-MULTIYEAR",
        "legoesm_git_sha": sha, "bar": BAR,
        "record_counts": {
            "output_times": n_outputs,
            "output_rows": n_outputs * 3,
            "restart_rows": len(restart_rows),
        },
        "oracle_content_sha256": SHIPPED_SHA256,
        "per_kt": per_output,
    })
    output_steps = [step for step in per_output if step["kind"] == "output"]
    out["first_output_over_bar"] = next((
        {"output_index": step["output_index"], "kt": step["kt"],
         "model_day": step["model_day"],
         "field": next(row["name"].rsplit(".", 1)[-1]
                       for row in step["rows"] if row["status"] == "DEBT")}
        for step in output_steps
        if any(row["status"] == "DEBT" for row in step["rows"])), None)
    out["first_restart_over_bar"] = next((
        row["name"].rsplit(".", 1)[-1]
        for row in restart_rows if row["status"] == "DEBT"), None)
    out["growth_curve"] = [{
        "output_index": step["output_index"], "kt": step["kt"],
        "max_abs": {
            row["name"].rsplit(".", 1)[-1]: row["max_abs"]
            for row in step["rows"]},
    } for step in output_steps]
    peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out["performance"] = {
        "wall_seconds_excluding_jit_warmup": wall_s,
        "seconds_per_360_day_model_year": wall_s / 20.0,
        "model_steps_per_second": card.n_steps / wall_s,
        "rss_high_water_mib": peak_kib / 1024.0,
        "rss_high_water_at_timing_start_mib": rss_start_kib / 1024.0,
    }
    if out["all_bit_identical"]:
        nemo_arrays, nemo_summary = _shared_shipped_statistics(
            reference["T"], reference["U"], reference["V"], card)
        lego_arrays, lego_summary = _shared_shipped_statistics(
            candidate["T"], candidate["U"], candidate["V"], card)
        require(nemo_summary == lego_summary,
                "identical stored series produced different statistics summaries")
        require(nemo_arrays.keys() == lego_arrays.keys()
                and all(np.array_equal(nemo_arrays[name], lego_arrays[name])
                        for name in nemo_arrays),
                "identical stored series produced different statistics arrays")
        out["statistics"] = {
            "inputs": "separately computed from masked NEMO and legoESM series",
            "bit_identical": True,
            "nemo": nemo_summary,
            "legoesm": lego_summary,
        }
        if statistics_npz is not None:
            statistics_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(statistics_npz, **lego_arrays)
            out["statistics"]["artifact"] = {
                "path": str(statistics_npz.resolve()),
                "sha256": _sha256(statistics_npz),
            }
    else:
        out["statistics"] = {
            "status": "WITHHELD",
            "reason": "the shipped trajectory is not bit-identical",
        }
    return out


def independent(root: Path, *, allow_dirty: bool = False) -> dict:
    """Run the independently initialized, production ``step()`` ladder."""
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    card, masks, model, forcing = _swg_production_case("ladder")
    records = read_restarts(root)
    state = card.recipe.initial_state
    per_kt = []
    for kt in range(1, 11):
        state = model.step(state, card.dt_s, surface_forcing=forcing)
        cand = {
            "sshn": np.asarray(state.eta.data),
            "un": np.asarray(state.u.data)[:, 1:, 0],
            "vn": np.asarray(state.v.data)[1:, :, 0],
            "uu_n": np.asarray(state.uu_b.data)[:, 1:],
            "vv_n": np.asarray(state.vv_b.data)[1:, :],
        }
        rows = [_row(f"independent.kt{kt}.{n}", records[kt][n], cand[n],
                     _field_mask(n, masks))
                for n in cand]
        per_kt.append({"kt": kt, "rows": rows})
    return _gate_rows({"label": "INDEPENDENT", "legoesm_git_sha": sha,
                       "bar": BAR,
                       "per_kt": per_kt})


def carried_state_audit(stage_root: Path, *, allow_dirty: bool = False) -> dict:
    """Compare the fifth production step and rebuilt sixth-step boundary."""
    from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    card, masks, model, forcing = _swg_production_case("carry audit")
    entries = read_stage_entries(stage_root)
    target = entries[6, 1]

    initial = card.recipe.initial_state
    state = initial
    state4 = None
    for kt in range(1, 6):
        if kt == 5:
            state4 = state
        state = model.step(state, card.dt_s, surface_forcing=forcing)
    require(state4 is not None, "failed to retain the fourth-step state")

    now = (forcing.tau_i_native, forcing.tau_j_native)
    before4 = (
        now[0] if state4.tau_x_prev is None else state4.tau_x_prev,
        now[1] if state4.tau_y_prev is None else state4.tau_y_prev,
    )
    boundary5 = _entry_from_state(
        state4, before4, now, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config)
    s1 = model.replay_swe_rk3_stage(boundary5, card.dt_s, 1, boundary5)
    s2 = model.replay_swe_rk3_stage(s1, card.dt_s, 2, boundary5)
    s3 = model.replay_swe_rk3_stage(s2, card.dt_s, 3, boundary5)

    before6 = (state.tau_x_prev, state.tau_y_prev)
    rebuilt6 = _entry_from_state(
        state, before6, now, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config)
    stage_rows = [
        _row(f"carry.stage3_to_kt6.{name}", target[name], value,
             _field_mask(name, masks))
        for name, value in s3._asdict().items()
    ]
    rebuilt_rows = [
        _row(f"carry.rebuilt_kt6.{name}", target[name], value,
             _field_mask(name, masks))
        for name, value in rebuilt6._asdict().items()
    ]
    # The carried boundary is now fully specified.  Execute the next ordinary
    # production step through its three materialized, separately JIT-compiled
    # stage calls and compare its public result with the admitted next entry.
    # The explicit stage chain above is the source-order control for the same
    # transition.
    state6 = model.step(state, card.dt_s, surface_forcing=forcing)
    target7 = entries[7, 1]
    whole_step6_rows = [
        _row("carry.whole_step6.ssh", target7["ssh_k"], state6.eta.data,
             masks["T"]),
        _row("carry.whole_step6.u", target7["u_k"], state6.u.data[:, 1:, 0],
             masks["U"]),
        _row("carry.whole_step6.v", target7["v_k"], state6.v.data[1:, :, 0],
             masks["V"]),
    ]

    public_rows = [
        _row("carry.public.ssh", target["ssh_k"], state.eta.data, masks["T"]),
        _row("carry.public.u", target["u_k"], state.u.data[:, 1:, 0], masks["U"]),
        _row("carry.public.v", target["v_k"], state.v.data[1:, :, 0], masks["V"]),
        _row("carry.public.uu_b", np.zeros_like(target["u_k"]),
             state.uu_b.data[:, 1:], masks["U"]),
        _row("carry.public.vv_b", np.zeros_like(target["v_k"]),
             state.vv_b.data[1:, :], masks["V"]),
    ]
    unchanged_fields = ("T", "S", "H_bathy", "land_mask", "u_mask",
                        "v_mask", "w", "uu_b", "vv_b")
    unchanged = {}
    for name in unchanged_fields:
        after = np.asarray(getattr(getattr(state, name), "data"))
        start = np.asarray(getattr(getattr(initial, name), "data"))
        unchanged[name] = {
            "bit_identical": bool(np.array_equal(after, start)),
            "n_unequal": int(np.count_nonzero(after != start)),
        }
    require(all(row["bit_identical"] for row in public_rows),
            "step-five public state differs before the rebuilt boundary")
    require(all(item["bit_identical"] for item in unchanged.values()),
            "an inert/static SWG state leaf changed")
    optional_none = [
        name for name, value in state._asdict().items()
        if value is None
    ]
    accounted = set(unchanged_fields) | {
        "u", "v", "eta", "tau_x_prev", "tau_y_prev"
    } | set(optional_none)
    require(accounted == set(state._fields),
            f"unaccounted SWG state leaves: {sorted(set(state._fields)-accounted)}")
    redundant = {
        "u_west": {
            "shape": list(np.asarray(state.u.data[:, :1, :]).shape),
            "n_nonzero": int(np.count_nonzero(np.asarray(state.u.data[:, :1, :]))),
        },
        "v_south": {
            "shape": list(np.asarray(state.v.data[:1, :, :]).shape),
            "n_nonzero": int(np.count_nonzero(np.asarray(state.v.data[:1, :, :]))),
        },
    }
    require(redundant["u_west"]["n_nonzero"] == 0
            and redundant["v_south"]["n_nonzero"] == 0,
            "walled redundant face storage is nonzero")
    return _gate_rows({
        "label": "CARRIED-STATE-AUDIT", "legoesm_git_sha": sha,
        "bar": BAR,
        "state_disposition": {
            "scored_public": [row["name"] for row in public_rows],
            "unchanged": unchanged,
            "optional_none": optional_none,
            "redundant_faces": redundant,
        },
        "per_kt": [{"kt": 6,
                    "rows": stage_rows + rebuilt_rows + whole_step6_rows}],
    })


def swe_rk3_stages(stage_root: Path, *, mode: str = "given",
                   plant: str | None = None,
                   allow_dirty: bool = False) -> dict:
    """Drive the JIT production stages on non-degenerate given/chained rows."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.dynamics.swe_rk3 import SWERK3StageEntry

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "ladder must run on CPU")
    card = build_swg_card()
    vor = card.recipe.model_config.vorticity_scheme
    require(vor.scheme == "ens_total",
            "SWG card does not select ens_total")
    require(vor.ln_dynvor_msk is True,
            "SWG card does not explicitly select ln_dynvor_msk=true")
    require(mode in ("given", "chained"),
            f"unknown stage replay mode {mode!r}")
    entries = read_stage_entries(stage_root)
    masks = _card_masks(card)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    def item(fields):
        return SWERK3StageEntry(**{
            name: jax.numpy.asarray(fields[name], dtype=jax.numpy.float64)
            for name in SWERK3StageEntry._fields
        })

    def replay(fields, before_fields, stage):
        result = model.replay_swe_rk3_stage(
            item(fields), card.dt_s, stage, item(before_fields))
        return {name: np.asarray(value) for name, value in result._asdict().items()}

    identity = all(np.array_equal(entries[1, 1][name], entries[1, 2][name])
                   for name in SWERK3StageEntry._fields)
    require(identity, "kt=1 stage-1 boundary is no longer an identity")
    nondiscriminating = {
        "kt": 1, "stage": 1, "status": "NON-DISCRIMINATING",
        "reason": "oracle entry and target are identical on all 13 fields",
    }

    per_kt = []
    if mode == "given":
        for kt in range(2, 11):
            before = entries[kt, 1]
            for stage in (1, 2, 3):
                if stage == 3 and kt == 10:
                    continue
                target = entries[kt, stage + 1] if stage < 3 else entries[kt + 1, 1]
                candidate = replay(entries[kt, stage], before, stage)
                require(all(value.dtype == np.dtype("float64")
                            for value in candidate.values()),
                        "production stage output is not fp64")
                rows = [
                    _row(f"given.kt{kt}.s{stage}.{name}", target[name], value,
                         _field_mask(name, masks))
                    for name, value in candidate.items()
                ]
                per_kt.append({"kt": kt, "stage": stage, "rows": rows})
    else:
        current = {name: value.copy() for name, value in entries[2, 1].items()}
        for kt in range(2, 11):
            before = {name: value.copy() for name, value in current.items()}
            for stage in (1, 2, 3):
                if stage == 3 and kt == 10:
                    continue
                candidate = replay(current, before, stage)
                target = entries[kt, stage + 1] if stage < 3 else entries[kt + 1, 1]
                rows = [
                    _row(f"chained.kt{kt}.s{stage}.{name}", target[name], value,
                         _field_mask(name, masks))
                    for name, value in candidate.items()
                ]
                per_kt.append({"kt": kt, "stage": stage, "rows": rows})
                current = candidate
    controls = []
    if plant is not None:
        require(plant == "ulp",
                f"unknown stage-entry plant {plant!r}")
        # One genuine ULP of an admitted, nonzero, wet value.  The whole-step
        # before entry stays byte-identical, so exactly one current-stage input
        # differs before the JIT production stage consumes it.
        current = {name: value.copy() for name, value in entries[2, 1].items()}
        before = {name: value.copy() for name, value in entries[2, 1].items()}
        wet_nonzero = np.argwhere(masks["U"] & (current["u_k"] != 0.0))
        require(wet_nonzero.size > 0,
                "admitted kt=2 stage-1 U field has no nonzero wet value")
        at = tuple(wet_nonzero[np.argmin(
            np.abs(current["u_k"][tuple(wet_nonzero.T)]))])
        original = np.float64(current["u_k"][at])
        planted_value = np.nextafter(original, np.float64(np.inf))
        require(original != 0.0 and planted_value != original,
                "recorded-entry nextafter plant is not one binding ULP")
        baseline = replay(current, before, 1)
        current["u_k"][at] = planted_value
        moved = replay(current, before, 1)
        control = _row(
            "production_recorded_nonzero_u_one_ulp", baseline["ssh_k"],
            moved["ssh_k"], masks["T"])
        require(not control["bit_identical"],
                "recorded-entry one-ULP plant did not change production SSH")
        control.update({
            "input_field": "u_k",
            "input_cell": [int(x) for x in at],
            "input_before": float(original),
            "input_after": float(planted_value),
            "input_ulp_delta": 1,
        })
        controls.append(control)
    return _gate_rows({
        "label": f"SWE-RK3-STAGES-{mode.upper()}", "legoesm_git_sha": sha,
        "bar": BAR, "plant": plant, "production_stage_executed": True,
        "stage3_input_audit": _stage3_input_audit(entries, masks),
        "nondiscriminating": [nondiscriminating], "controls": controls,
        "per_kt": per_kt,
    })


def ens_stage1(stage_root: Path, *, plant: str | None = None,
               allow_dirty: bool = False) -> dict:
    """Compatibility name for the complete given-entry production stage arm."""
    return swe_rk3_stages(
        stage_root, mode="given", plant=plant, allow_dirty=allow_dirty)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=(
        "self-check", "independent", "stage-entry-read", "ens-stage1",
        "stages-given", "stages-chained", "carried-state", "month",
        "shipped"),
                   required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    p.add_argument("--month-root", type=Path, default=DEFAULT_MONTH_ROOT)
    p.add_argument("--shipped-root", type=Path, default=DEFAULT_SHIPPED_ROOT)
    p.add_argument("--statistics-npz", type=Path)
    p.add_argument("--plant", nargs="?", const="ulp", choices=("ulp",))
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    try:
        if a.arm == "self-check":
            out = self_check(a.root, plant=a.plant is not None,
                             allow_dirty=a.allow_dirty)
        elif a.arm == "independent":
            out = independent(a.root, allow_dirty=a.allow_dirty)
        elif a.arm == "carried-state":
            out = carried_state_audit(
                a.stage_root, allow_dirty=a.allow_dirty)
        elif a.arm == "month":
            out = (month_control(a.month_root, allow_dirty=a.allow_dirty)
                   if a.plant else
                   month(a.month_root, allow_dirty=a.allow_dirty))
        elif a.arm == "shipped":
            require(a.plant is None, "shipped arm has no post-computation plant")
            out = shipped(
                a.shipped_root, statistics_npz=a.statistics_npz,
                progress=True, allow_dirty=a.allow_dirty)
        elif a.arm == "stage-entry-read":
            entries = read_stage_entries(a.stage_root)
            out = {"label": "STAGE-ENTRY-READ", "records": len(entries),
                   "keys": [list(k) for k in entries]}
        elif a.arm == "ens-stage1":
            out = ens_stage1(a.stage_root, plant=a.plant,
                             allow_dirty=a.allow_dirty)
        else:
            out = swe_rk3_stages(
                a.stage_root,
                mode="given" if a.arm == "stages-given" else "chained",
                plant=a.plant, allow_dirty=a.allow_dirty)
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    if a.json:
        a.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(out["label"])
    if "all_at_bar" in out:
        print(f"all_at_bar={out['all_at_bar']} bit_identical={out['all_bit_identical']}")
        return _exit_code(out)
    print(f"records={out.get('records', len(out.get('per_kt', [])))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
