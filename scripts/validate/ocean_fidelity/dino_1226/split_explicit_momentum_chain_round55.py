#!/usr/bin/env python3
"""Replay the current production tracer-entry ladder against held row 8."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import numpy as np

import kamm_twin_90d as twin
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gm_module
import zdf_chain_sweep as sweep
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.latlon import ensure_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import upwind_to_u_points
from legoesm.ocean.vertical import compute_layer_thickness


ROUND54_SHA = "a1b76177330b83d7bb21c7f35c9e606d10d36a4b98d46b3b570f53588188eaa3"
# 2026-08-30 round-58 population correction: supersedes the invalid
# e3u!=0 round-56 admission hash; same production arm on exact NEMO umask.
ROUND56_SHA = "c114565363360439e155deec96882828e553887ba49b1bfbfd940c50c6998e29"
ROUND59_SHA = "85ea27cce4804d98f281940fe472e798d9fa64c741c23bb55e3fca40ee9ca677"
ROUND60_SHA = "b3ef5c0534ff1348dbdb581686aa602cc1d9eca9ef61336ca0b4130217e54e2d"
ROUND61_SHA = "b8f7a376a0a13fd384cb4195cf27db8ff6f82c71e576bb8d449451903d3bd07c"
ROUND62_SHA = "290caa5bb3c3187d5ea13fe62276f299bd47953b556615dfab3f4443d1597a86"
ROUND63_SHA = "3a1a25ca328761b1bcbeb87953751a3a15b1ac00852b2ff62fd4223d107d24e0"
ROUND64_SHA = "8858d60a51b07181e290fead087e4bab69c0d15e271bbdecb7d458ba12d4e4c7"
ROUND65_SHA = "6b4a80ddaf020916770dd0eb0005a6ce9dd69ffe79a6604a0e2125cfbadf4c60"
ROUND66_SHA = "673d9dc978ac4c05a0c8a6995f7c9e1870d9502afedff9b765045df23014785f"
ROUND67_SHA = "2c1c1819f08cc07d9aa90fd43624285fe276558bbb5bba84aa97b3b83b345d95"
ROUND68_SHA = "8e9a2ee7ee1969da86ee2ef24198047d6ea83ee14842753b528271453efaf9a0"
ROUND69_SHA = "bd109e2869333a31b8b6a8410d3ec10f24acae94353ed1f35d60900ae5b62eb2"
ROUND70_SHA = "d831bbcb89e06c8795a085103041293becedb8ac79c438f1302b44296b3314bc"
ROUND71_SHA = "6500acfa930c0342430fd1e57cfb1da023b0978e8fda3561e6133ffe12368821"
ROUND72_SHA = "1abf7718e8dc1fc4f75d23295ebaaf46c368ffa07d8e56467577f1abb7201230"
ROUND73_SHA = "c7445a0e26e4b74999bbe89f79c043ad4e9d9754f357d0e94ae136471cc9961d"
ROUND74_SHA = "56db4716cba582654fbd7bb55178a699b55678a1afdea9d8d8fe3cc670eea6fb"
ROUND75_SHA = "750c40875300ddda48287d84089c8931eaaecc71e8aab6ce7f4e18a0c806edf4"
ROUND76_SHA = "45e4f8afda737b41e457668fe1ab7cc28ded09d3f7be06fabdd15e9804936a76"
ROUND77_SHA = "dcc0cff4c63b30024794ad25b023b65223fe87137e5052b6d7dc478066973d14"
ROUND78_SHA = "30f63d47e11d632e214490c2d6fc8756170a03f12f86635ba6a8b0a8b8d6e7f3"
ROUND79_SHA = "dca39985e54bd95f20ee9b1bfb4ab3bc39013953c95aa8fb74965eb6618f6f61"
ROUND80_SHA = "e0803ac9f844ec77c843885247b6ad0809b33f02894b88376690ec870f9b8493"
ROUND81_SHA = "247acae491d6568febb16d81a7a96e5b2c22a4246cf6b2e613ce640d38eaee71"
ROUND82_SHA = "4177c99481bdbc6996477dad848d1e645ada3a26a08fb9aa1e9dd9e4b20f9e7a"
ROUND83_SHA = "f759087db5680d8ee13f8053000812e7b0a89471f8efc28196f2ead0006f8bb3"
ROUND84_SHA = "6483fc67d59bea3b5bb81e546a31af47eec85aeea48573f42e11c2e575740451"
ROUND85_SHA = "10a5695a804c0e96ec36455ed160169b18897c43b57614c6a979349fcf53ec6c"
ROUND86_SHA = "93e39c3ee0fbf973464aa4ea583a404668e4be13dfe044153a28c0e3022c260e"
ROUND87_SHA = "df56a496d6e7ebc14d9e8214cdc6742010d3b8b4286c0d9b2704526593ed6caf"
ROUND88_SHA = "9107fa743aa3deb6b4f7e2180711f956599f099b908e8eb87cf36bd1bd76bd66"
ROUND89_SHA = "555520203984430e988c075646e7feb674cc522de33f5cadfc4052ccdc560a76"
ROUND90_SHA = "32a79283ab3ed45fbf909ffc03ed8a8afc5ccb9b82e99a5f5de6f22f9ae1dc05"
ROUND93_WSLP_SHA = "3ff570e4359ffd30b4314fd49a8af678d3091191d095a5347e359ee5951ce75a"
RAW_ARTIFACT_SHA = "ec4885a1e7c059872f1b575c5f93f00c0e6538b65613eede71f082fac24885ea"
HELD_SHA = {
    "DINO_00005760_restart.nc": "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "fct_entry_dump_un.bin": "89d5c37cb77e03ee0f6ab50ba0814461a7459a13c31fb221329f4cf9b5ec829d",
    "fct_entry_dump_e2u.bin": "d55f36aa52f6686c4d6a152c81ee7b8360ca415342810b68dc232baefb631052",
    "fct_entry_dump_e3u.bin": "d071b046cf5e841d2475b753b3c834ad02e69128e73a59c7dd15576a47d7b4f8",
    "fct_entry_dump_e2e3u.bin": "75fdac4eeaa0791f304b604fad626109cea73d98e3f79dbb455d12881b386a59",
    "fct_entry_dump_pu_euler.bin": "b0bcfbc94bbb0a203e475774b83e93fce8884f949f3364d79716704abbd68bde",
    "fct_entry_dump_pu_bolus.bin": "69dfe222dbbf460f9ae86f657b2b9ca114e4203ac8beb34d00d76f0e8f3f58d5",
    "fct_entry_dump_pu_total.bin": "4006ee7f7c8c140f1e4b3dfcdec5a4a3573b322cd86edfff245a206e05c5d270",
    "fct_dump_zwx_up.bin": "ecb7caf9cb610c274cc206877fc484ffe6a278cf558db73d6cf10aac184b715d",
    "spg_dump_zu_frc.bin": "13138faa46149968c009f0d6c20956b8d2457a904667686a1f6cb1a93d7bcca6",
    "spg_dump_zv_frc.bin": "81d01381ad5164b1a0cc3fb5f22590f471b0e24bb9b68084290e0cb9e2ef4c0e",
    "spg_dump_un_adv_final.bin": "4d8e7a6445ba465c8229954b443c467862381409805163f2045f5854017917ab",
    "spg_dump_vn_adv_final.bin": "ed1e28aa27e1c49d237e07a07707220251b3a1845a599d21b06df18b8425d088",
}
BOLUS_HELD_SHA = {
    "eiv_dump_u.bin": "a0e7b0f0a84cb87bd5e059c7161d261016f2528ba127df666a37966395c0fc00",
    "eiv_dump_psi_uw.bin": "feb5ba7a1e4882cb43c71db049fb78ecd2e2301c9573777bf5a0e1738518101b",
    "eiv_dump_wslpi.bin": "e3073a8501e8046732e301d58781dcaf35a6a84d9071309353b3de0bad08fb64",
    "eiv_dump_aeiu.bin": "8146cf02d33e8013bf240623948b42bd8b2cda8ee15c11847393ad298a5a60e8",
}
KAPPA_HELD_SHA = {
    "eiv_dump_zn.bin": "8d4f9557f62803af99f5ff03eb2d980bbe161b03ff37093f05a03d89e64e985b",
    "eiv_dump_zah.bin": "f40b23c4ab54544b8663699c2011e7709239afebd235e930824225bd3167810c",
    "eiv_dump_zhw.bin": "60b23de68ef7eff2219725a3b1f0abf732b2dc78879c7097cf58cca0e6a8260f",
    "eiv_dump_zRo.bin": "b2465f3e63f1ea4332a8fda2bb4785b45577f8cc939cf5957d6497600a9e0b24",
    "eiv_dump_zaeiw.bin": "adc570d4d7e385ebe54f7cef0e249e391c6b5eb193acbd4315feb50cf035db44",
}
KAPPA_GEOMETRY_SHA = {
    "eiv_dump_e3w.bin": "fa204dd2ea7f02af0c5fddda5c61255e0db916b429729311444d21a737c12a3b",
    "eiv_dump_rn2b.bin": "b419cf3ef5a6bba9100e37ead3e481b7e1d0da4a11709c5dcd55d43fec1770c5",
}
REDI_HELD_SHA = {
    "stp_dump_22_before_traldf_tem.bin": "6f6594593bf3089907de8c2aaf9706c08647a139080f72c0eb46f11d38ce2bc8",
    "stp_dump_22_before_traldf_sal.bin": "d8985e7056391862c64d3f4c3da92cd3d437f606ea0d66c58ca7d5ea4edb9a50",
    "stp_dump_23_after_traldf_tem.bin": "ff109b54b8f04c9dfcb09643b940b29f603fceff48d1896a510ba9af74a96f2f",
    "stp_dump_23_after_traldf_sal.bin": "7e751ab80519d2097e9332e120b7c0423d744ec98c3a9260c82c47a4064811fb",
}
REDI_SKEW_INPUT_SHA = {
    "ldftra_dump_ahtu.bin": "098b95a3548ee8d3c66694ca43c3b720840e2c193895b0df222e8e186b2580da",
    "ldftra_dump_ahtv.bin": "769cb4b4e87bbc4bee9567cd933216b8931086c9b236b9ec2acf35e38fb7006b",
    "eiv_dump_wslpi.bin": "e3073a8501e8046732e301d58781dcaf35a6a84d9071309353b3de0bad08fb64",
    "eiv_dump_wslpj.bin": "2d24bcace748f7d550b34ae51dc1736a3e6943967a2ffd068618b0e92a723acc",
}
FOCUS = [(11, 1), (12, 1), (13, 1), (13, 23)]
POINTWISE_BAR = 1.0e-15
ACCUMULATION_BAR = 1.0e-12


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True).strip()


def _load(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    expected = 35 * 203 * 56
    if raw.size != expected:
        raise SystemExit(
            f"{path}: full-halo writer contract is (35,203,56), got {raw.size}")
    return np.moveaxis(raw.reshape(35, 203, 56)[:, 2:-2, 2:-2], 0, -1)


def _load2(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != 203 * 56:
        raise SystemExit(f"{path}: full-halo 2-D writer contract changed")
    return raw.reshape(203, 56)[2:-2, 2:-2]


def _load_levels(path: Path, nlev: int) -> np.ndarray:
    """Load a full-halo NEMO level-major stream and strip its two-cell halo."""
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != nlev * 203 * 56:
        raise SystemExit(
            f"{path}: expected ({nlev},203,56), got {raw.size} values")
    return np.moveaxis(raw.reshape(nlev, 203, 56)[:, 2:-2, 2:-2], 0, -1)


def _controls(oracle: np.ndarray, wet: np.ndarray, bar: float) -> dict:
    identity = sweep.metrics(oracle, oracle, wet, FOCUS, bar)
    point = np.array(oracle, copy=True)
    idx = tuple(int(x) for x in np.argwhere(wet)[0])
    scale = identity["reference_rms"]
    point[idx] += max(4.0 * bar * scale, 4.0 * abs(float(np.spacing(point[idx]))))
    return {
        "identity": bool(identity["pass"]),
        "wet_point": not sweep.metrics(point, oracle, wet, FOCUS, bar)["pass"],
        "meridional_roll": not sweep.metrics(
            np.roll(oracle, 1, axis=0), oracle, wet, FOCUS, bar)["pass"],
        "zonal_roll": not sweep.metrics(
            np.roll(oracle, 1, axis=1), oracle, wet, FOCUS, bar)["pass"],
        "sign": not sweep.metrics(
            -oracle, oracle, wet, FOCUS, bar)["pass"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-dir", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round54", type=Path, required=True)
    parser.add_argument("--round56", type=Path)
    parser.add_argument("--round59", type=Path)
    parser.add_argument("--round60", type=Path)
    parser.add_argument("--round61", type=Path)
    parser.add_argument("--round62", type=Path)
    parser.add_argument("--round63", type=Path)
    parser.add_argument("--round64", type=Path)
    parser.add_argument("--round65", type=Path)
    parser.add_argument("--round66", type=Path)
    parser.add_argument("--round67", type=Path)
    parser.add_argument("--round68", type=Path)
    parser.add_argument("--round69", type=Path)
    parser.add_argument("--round70", type=Path)
    parser.add_argument("--round71", type=Path)
    parser.add_argument("--round72", type=Path)
    parser.add_argument("--round73", type=Path)
    parser.add_argument("--round74", type=Path)
    parser.add_argument("--round75", type=Path)
    parser.add_argument("--round76", type=Path)
    parser.add_argument("--round77", type=Path)
    parser.add_argument("--round78", type=Path)
    parser.add_argument("--round79", type=Path)
    parser.add_argument("--round80", type=Path)
    parser.add_argument("--round81", type=Path)
    parser.add_argument("--round82", type=Path)
    parser.add_argument("--round83", type=Path)
    parser.add_argument("--round84", type=Path)
    parser.add_argument("--round85", type=Path)
    parser.add_argument("--round86", type=Path)
    parser.add_argument("--round87", type=Path)
    parser.add_argument("--round88", type=Path)
    parser.add_argument("--round89", type=Path)
    parser.add_argument("--round90", type=Path)
    parser.add_argument("--round93-wslp", type=Path)
    parser.add_argument("--hold-slow-forcing", action="store_true")
    parser.add_argument("--oracle-transport", action="store_true")
    parser.add_argument("--capture-cycle", action="store_true")
    parser.add_argument("--direct-cycle-entry", action="store_true")
    parser.add_argument("--live-thickness-entry", action="store_true")
    parser.add_argument("--capture-bolus-operands", action="store_true")
    parser.add_argument("--capture-kappa-operands", action="store_true")
    parser.add_argument("--literal-kappa-reduction", action="store_true")
    parser.add_argument("--capture-kappa-geometry", action="store_true")
    parser.add_argument("--coupled-kappa-carry", action="store_true")
    parser.add_argument("--surface-kmm-carry", action="store_true")
    parser.add_argument("--exact-surface-kmm-carry", action="store_true")
    parser.add_argument("--post-chain-factorial", action="store_true")
    parser.add_argument("--rossby-factorial", action="store_true")
    parser.add_argument("--zn-sqrt-factorial", action="store_true")
    parser.add_argument("--exact-sqrt-production", action="store_true")
    parser.add_argument("--capture-redi-tail", action="store_true")
    parser.add_argument("--redi-e3w-factorial", action="store_true")
    parser.add_argument("--redi-flux-ladder", action="store_true")
    parser.add_argument("--redi-zfu-operand-ladder", action="store_true")
    parser.add_argument("--redi-zfu-postfix", action="store_true")
    parser.add_argument("--redi-zfu-kmm-postfix", action="store_true")
    parser.add_argument("--redi-zfu-slope-kmm-postfix", action="store_true")
    parser.add_argument("--redi-zfu-kmm-operator-postfix", action="store_true")
    parser.add_argument("--redi-zfu-bolus-stage-split-postfix", action="store_true")
    parser.add_argument("--redi-zfu-ahtu-postfix", action="store_true")
    parser.add_argument("--redi-zfw-association-factorial", action="store_true")
    parser.add_argument("--redi-zfw-component-score", action="store_true")
    parser.add_argument("--redi-zfw-skew-factorial", action="store_true")
    parser.add_argument("--redi-zfw-skew-postfix", action="store_true")
    parser.add_argument("--redi-zfw-skew-operand-factorial", action="store_true")
    parser.add_argument("--redi-zfw-wslp-stage-factorial", action="store_true")
    parser.add_argument("--redi-zfw-a33-floor-factorial", action="store_true")
    parser.add_argument("--redi-run-dir", type=Path)
    parser.add_argument("--redi-flux-run-dir", type=Path)
    parser.add_argument("--redi-flux-bracket", type=Path)
    parser.add_argument("--redi-flux-bracket-sha")
    parser.add_argument("--redi-zfw-component-dir", type=Path)
    parser.add_argument("--redi-zfw-component-bracket", type=Path)
    parser.add_argument("--redi-zfw-component-bracket-sha")
    parser.add_argument("--raw-artifact", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round54.resolve()) != ROUND54_SHA:
        raise SystemExit("official round-54 receipt changed")
    prior = json.loads(args.round54.read_text())
    if (prior.get("session_id") != session or prior.get("disposition")
            != "MOMENTUM_TAIL_AT_BAR_UPSTREAM_TRANSPORT_EXACT"):
        raise SystemExit("round 54 does not release the tracer tail")
    prior56 = None
    if args.hold_slow_forcing:
        if args.round56 is None or _sha(args.round56.resolve()) != ROUND56_SHA:
            raise SystemExit("official round-56 red receipt required")
        prior56 = json.loads(args.round56.read_text())
        if (prior56.get("session_id") != session or prior56.get("disposition")
                != "TRACER_ENTRY_DIVERGED_8.3"):
            raise SystemExit("round 56 does not admit the forcing substitution")
    elif args.round56 is not None:
        raise SystemExit("--round56 is only valid with --hold-slow-forcing")
    prior59 = prior60 = None
    if args.oracle_transport:
        if not args.hold_slow_forcing:
            raise SystemExit("--oracle-transport requires --hold-slow-forcing")
        if (args.round59 is None
                or _sha(args.round59.resolve()) != ROUND59_SHA):
            raise SystemExit("official round-59 held receipt required")
        if (args.round60 is None
                or _sha(args.round60.resolve()) != ROUND60_SHA):
            raise SystemExit("official round-60 receipt required")
        prior59 = json.loads(args.round59.read_text())
        prior60 = json.loads(args.round60.read_text())
        if (prior59.get("disposition") != "TRACER_ENTRY_DIVERGED_8.3"
                or prior59["rows"][0]["metrics"]["n_diverged_columns"] != 104):
            raise SystemExit("round 59 does not admit the conditional replay")
        if (prior60.get("disposition")
                != "ROW8_3_LITERAL_ACCUMULATOR_AT_BAR_UPSTREAM_OPERANDS_OPEN"):
            raise SystemExit("round 60 does not release the oracle transport arm")
    elif args.round59 is not None or args.round60 is not None:
        raise SystemExit("--round59/--round60 require --oracle-transport")
    prior61 = None
    if args.capture_cycle:
        if not args.oracle_transport:
            raise SystemExit("--capture-cycle requires --oracle-transport")
        if (args.round61 is None
                or _sha(args.round61.resolve()) != ROUND61_SHA):
            raise SystemExit("official round-61 null receipt required")
        prior61 = json.loads(args.round61.read_text())
        if (prior61.get("disposition") != "TRACER_ENTRY_DIVERGED_8.3"
                or prior61["rows"][0]["metrics"]["n_diverged_columns"] != 104):
            raise SystemExit("round 61 does not admit the Kmm-cycle capture")
    elif args.round61 is not None:
        raise SystemExit("--round61 requires --capture-cycle")
    prior62 = None
    if args.direct_cycle_entry:
        if not args.capture_cycle:
            raise SystemExit("--direct-cycle-entry requires --capture-cycle")
        if (args.round62 is None
                or _sha(args.round62.resolve()) != ROUND62_SHA):
            raise SystemExit("official round-62 capture receipt required")
        prior62 = json.loads(args.round62.read_text())
        capture62 = prior62.get("cycle_capture_metrics", {})
        if (prior62.get("disposition") != "ROW8_3_CYCLE_CAPTURE_UNRESOLVED"
                or not capture62.get("production_Hu_avg", {}).get("pass")
                or not capture62.get("consumed_Hu_avg", {}).get("pass")
                or not capture62.get("cycle_corrected_u", {}).get("pass")):
            raise SystemExit("round 62 does not release the direct-cycle score")
    elif args.round62 is not None:
        raise SystemExit("--round62 requires --direct-cycle-entry")
    prior63 = None
    if args.live_thickness_entry:
        if not args.direct_cycle_entry:
            raise SystemExit(
                "--live-thickness-entry requires --direct-cycle-entry")
        if (args.round63 is None
                or _sha(args.round63.resolve()) != ROUND63_SHA):
            raise SystemExit("official round-63 thickness receipt required")
        prior63 = json.loads(args.round63.read_text())
        if (prior63.get("disposition") != "TRACER_ENTRY_DIVERGED_8.5"
                or prior63["rows"][0]["status"] != "AT_BAR"
                or prior63["rows"][1]["status"] != "AT_BAR"
                or prior63["rows"][2]["metrics"]["n_diverged_columns"]
                != 9758):
            raise SystemExit("round 63 does not release the live-thickness score")
    elif args.round63 is not None:
        raise SystemExit("--round63 requires --live-thickness-entry")
    prior64 = None
    if args.capture_bolus_operands:
        if not args.live_thickness_entry:
            raise SystemExit(
                "--capture-bolus-operands requires --live-thickness-entry")
        if (args.round64 is None
                or _sha(args.round64.resolve()) != ROUND64_SHA):
            raise SystemExit("official round-64 row-8.8 receipt required")
        prior64 = json.loads(args.round64.read_text())
        if (prior64.get("disposition") != "TRACER_ENTRY_DIVERGED_8.8"
                or any(prior64["rows"][i]["status"] != "AT_BAR"
                       for i in range(5))
                or prior64["rows"][5]["metrics"]["n_diverged_columns"]
                != 8938):
            raise SystemExit("round 64 does not release the bolus operand peel")
    elif args.round64 is not None:
        raise SystemExit("--round64 requires --capture-bolus-operands")
    prior65 = None
    if args.capture_kappa_operands:
        if not args.capture_bolus_operands:
            raise SystemExit(
                "--capture-kappa-operands requires --capture-bolus-operands")
        if (args.round65 is None
                or _sha(args.round65.resolve()) != ROUND65_SHA):
            raise SystemExit("official round-65 aeiu receipt required")
        prior65 = json.loads(args.round65.read_text())
        if (prior65.get("disposition") != "ROW8_8_LOCALIZED_TO_AEIU_OPERAND"
                or prior65["bolus_operand_metrics"]["aeiu_face"]["pass"]
                or not prior65["bolus_operand_metrics"]
                    ["wslpi_kp1_face_sum"]["pass"]):
            raise SystemExit("round 65 does not release the kappa ladder")
    elif args.round65 is not None:
        raise SystemExit("--round65 requires --capture-kappa-operands")
    prior66 = None
    if args.literal_kappa_reduction:
        if not args.capture_kappa_operands:
            raise SystemExit(
                "--literal-kappa-reduction requires --capture-kappa-operands")
        if (args.round66 is None
                or _sha(args.round66.resolve()) != ROUND66_SHA):
            raise SystemExit("official round-66 reduction debt receipt required")
        prior66 = json.loads(args.round66.read_text())
        if (prior66.get("disposition")
                != "ROW8_8_GM_COEFFICIENT_DIVERGED_ZN"
                or prior66["kappa_operand_metrics"]["zn"]["pass"]):
            raise SystemExit("round 66 does not release the literal reduction")
    elif args.round66 is not None:
        raise SystemExit("--round66 requires --literal-kappa-reduction")
    prior67 = None
    if args.capture_kappa_geometry:
        if not args.literal_kappa_reduction:
            raise SystemExit(
                "--capture-kappa-geometry requires --literal-kappa-reduction")
        if (args.round67 is None
                or _sha(args.round67.resolve()) != ROUND67_SHA):
            raise SystemExit("official round-67 null receipt required")
        prior67 = json.loads(args.round67.read_text())
        if prior67.get("disposition") != "ROW8_8_GM_COEFFICIENT_DIVERGED_ZN":
            raise SystemExit("round 67 does not release the geometry peel")
    elif args.round67 is not None:
        raise SystemExit("--round67 requires --capture-kappa-geometry")
    prior68 = None
    if args.coupled_kappa_carry:
        if not args.capture_kappa_geometry:
            raise SystemExit(
                "--coupled-kappa-carry requires --capture-kappa-geometry")
        if (args.round68 is None
                or _sha(args.round68.resolve()) != ROUND68_SHA):
            raise SystemExit("official round-68 coupled-operand receipt required")
        prior68 = json.loads(args.round68.read_text())
        if (prior68.get("disposition") != "ROW8_8_LOCALIZED_TO_RN2B"
                or prior68["kappa_geometry_metrics"]["e3w_Kmm"]["pass"]
                or prior68["kappa_geometry_metrics"]["rn2b"]["pass"]):
            raise SystemExit("round 68 does not release the coupled carry")
    elif args.round68 is not None:
        raise SystemExit("--round68 requires --coupled-kappa-carry")
    prior69 = None
    if args.surface_kmm_carry:
        if not args.coupled_kappa_carry:
            raise SystemExit("--surface-kmm-carry requires --coupled-kappa-carry")
        if (args.round69 is None
                or _sha(args.round69.resolve()) != ROUND69_SHA):
            raise SystemExit("official round-69 surface residual required")
        prior69 = json.loads(args.round69.read_text())
        if (prior69.get("disposition") != "ROW8_8_LOCALIZED_TO_KMM_E3W"
                or not prior69["kappa_geometry_metrics"]["rn2b"]["pass"]
                or prior69["kappa_geometry_metrics"]["e3w_Kmm"]["pass"]):
            raise SystemExit("round 69 does not release the surface carry")
    elif args.round69 is not None:
        raise SystemExit("--round69 requires --surface-kmm-carry")
    prior70 = None
    if args.exact_surface_kmm_carry:
        if not args.surface_kmm_carry:
            raise SystemExit(
                "--exact-surface-kmm-carry requires --surface-kmm-carry")
        if (args.round70 is None
                or _sha(args.round70.resolve()) != ROUND70_SHA):
            raise SystemExit("official round-70 planted violation required")
        prior70 = json.loads(args.round70.read_text())
        if (prior70.get("disposition") != "ROW8_8_LOCALIZED_TO_KMM_E3W"
                or prior70["kappa_geometry_metrics"]["e3w_Kmm"]["pass"]
                or prior70["rows"][5]["metrics"]["n_diverged_columns"]
                != 9257
                or prior70["rows"][5]["metrics"]["max_column_error"]
                < 8.0e-4):
            raise SystemExit(
                "round 70 does not admit the exact-surface Kmm carry")
    elif args.round70 is not None:
        raise SystemExit("--round70 requires --exact-surface-kmm-carry")
    prior71 = None
    if args.post_chain_factorial:
        if not args.exact_surface_kmm_carry:
            raise SystemExit(
                "--post-chain-factorial requires --exact-surface-kmm-carry")
        if (args.round71 is None
                or _sha(args.round71.resolve()) != ROUND71_SHA):
            raise SystemExit("official round-71 exact-geometry receipt required")
        prior71 = json.loads(args.round71.read_text())
        if (prior71.get("disposition") != "ROW8_8_KAPPA_GEOMETRY_AT_BAR"
                or not all(prior71["kappa_geometry_metrics"][name]["pass"]
                           for name in ("e3w_Kmm", "rn2b"))
                or prior71["kappa_operand_metrics"]["zaeiw"]["pass"]
                or prior71["rows"][5]["metrics"]["n_diverged_columns"]
                != 1071):
            raise SystemExit("round 71 does not release the post-chain peel")
    elif args.round71 is not None:
        raise SystemExit("--round71 requires --post-chain-factorial")
    prior72 = None
    if args.rossby_factorial:
        if not args.post_chain_factorial:
            raise SystemExit("--rossby-factorial requires --post-chain-factorial")
        if (args.round72 is None
                or _sha(args.round72.resolve()) != ROUND72_SHA):
            raise SystemExit("official round-72 zRo receipt required")
        prior72 = json.loads(args.round72.read_text())
        if (prior72.get("disposition")
                != "ROW8_8_LOCALIZED_TO_ZRO_PRECURSOR"
                or not all(prior72["post_chain_metrics"]
                           ["literal_oracle_zRo"][name]["pass"]
                           for name in ("zaeiw", "aeiu", "row8_8"))
                or prior72["post_chain_metrics"]["literal_own"]
                           ["row8_8"]["pass"]):
            raise SystemExit("round 72 does not release the Rossby peel")
    elif args.round72 is not None:
        raise SystemExit("--round72 requires --rossby-factorial")
    prior73 = None
    if args.zn_sqrt_factorial:
        if not args.rossby_factorial:
            raise SystemExit("--zn-sqrt-factorial requires --rossby-factorial")
        if (args.round73 is None
                or _sha(args.round73.resolve()) != ROUND73_SHA):
            raise SystemExit("official round-73 zn receipt required")
        prior73 = json.loads(args.round73.read_text())
        if (prior73.get("disposition")
                != "ROW8_8_LOCALIZED_TO_ZN_PRECURSOR"
                or not prior73["rossby_metrics"]["literal_oracle_zn"]
                              ["row8_8"]["pass"]
                or prior73["rossby_metrics"]["literal_stored_f"]
                              ["row8_8"]["pass"]):
            raise SystemExit("round 73 does not release the sqrt peel")
    elif args.round73 is not None:
        raise SystemExit("--round73 requires --zn-sqrt-factorial")
    prior74 = None
    if args.exact_sqrt_production:
        if not args.zn_sqrt_factorial:
            raise SystemExit(
                "--exact-sqrt-production requires --zn-sqrt-factorial")
        if (args.round74 is None
                or _sha(args.round74.resolve()) != ROUND74_SHA):
            raise SystemExit("official round-74 sqrt owner receipt required")
        prior74 = json.loads(args.round74.read_text())
        if (prior74.get("disposition")
                != "ROW8_8_LOCALIZED_TO_ZN_SQRT_FORWARD_FLOOR"
                or not all(prior74["zn_sqrt_metrics"]["exact_forward"]
                           [name]["pass"] for name in
                           ("zn", "zRo", "zaeiw", "aeiu", "row8_8"))
                or prior74["zn_sqrt_metrics"]["guarded_floor"]
                          ["row8_8"]["pass"]):
            raise SystemExit("round 74 does not release production certification")
    elif args.round74 is not None:
        raise SystemExit("--round74 requires --exact-sqrt-production")
    prior75 = None
    if args.capture_redi_tail:
        if not args.exact_sqrt_production:
            raise SystemExit(
                "--capture-redi-tail requires --exact-sqrt-production")
        if (args.round75 is None
                or _sha(args.round75.resolve()) != ROUND75_SHA):
            raise SystemExit("official round-75 tracer-entry receipt required")
        prior75 = json.loads(args.round75.read_text())
        if (prior75.get("disposition")
                != "TRACER_ENTRY_ROW8_AT_BAR_EXACT_GM_SQRT"
                or prior75.get("first_diverged_subrow") is not None
                or any(row["status"] != "AT_BAR"
                       for row in prior75["rows"])):
            raise SystemExit("round 75 does not release the Redi tail")
    elif args.round75 is not None:
        raise SystemExit("--round75 requires --capture-redi-tail")
    prior76 = None
    if args.redi_e3w_factorial:
        if (not args.capture_redi_tail or not args.capture_kappa_geometry
                or not args.exact_surface_kmm_carry):
            raise SystemExit(
                "--redi-e3w-factorial requires the Redi capture and exact "
                "surface-Kmm geometry stack")
        if (args.round76 is None
                or _sha(args.round76.resolve()) != ROUND76_SHA):
            raise SystemExit("official round-76 Redi receipt required")
        prior76 = json.loads(args.round76.read_text())
        if (prior76.get("session_id") != session
                or prior76.get("disposition")
                != "TRACER_TAIL_DIVERGED_REDI_T"
                or prior76["redi_metrics"]["temperature"]["pass"]):
            raise SystemExit("round 76 does not admit the MSC thickness peel")
    elif args.round76 is not None:
        raise SystemExit("--round76 requires --redi-e3w-factorial")
    prior77 = None
    if args.redi_flux_ladder:
        if not args.redi_e3w_factorial:
            raise SystemExit(
                "--redi-flux-ladder requires --redi-e3w-factorial")
        if (args.round77 is None
                or _sha(args.round77.resolve()) != ROUND77_SHA):
            raise SystemExit("official round-77 MSC majority receipt required")
        prior77 = json.loads(args.round77.read_text())
        if (prior77.get("session_id") != session
                or prior77.get("disposition") != "REDI_MSC_E3W_MAJORITY"):
            raise SystemExit("round 77 does not admit the Redi flux ladder")
        if (args.redi_flux_run_dir is None
                or args.redi_flux_bracket is None
                or not args.redi_flux_bracket_sha):
            raise SystemExit(
                "--redi-flux-ladder requires run dir, bracket, and bracket SHA")
        if _sha(args.redi_flux_bracket.resolve()) != args.redi_flux_bracket_sha:
            raise SystemExit("Redi flux bracket SHA changed")
        flux_bracket = json.loads(args.redi_flux_bracket.read_text())
        expected_flux = {
            f"redi_dump_{flux}_{tracer}.bin"
            for flux in ("zfu", "zfv", "zfw")
            for tracer in ("tem", "sal")
        }
        if (flux_bracket.get("schema") != "dino-redi-flux-bracket-v1"
                or not flux_bracket.get("shared_exact")
                or set(flux_bracket.get("new_streams", ())) != expected_flux
                or not all(flux_bracket.get("controls", {}).values())):
            raise SystemExit("Redi flux bracket does not admit measurement")
        flux_dir = args.redi_flux_run_dir.resolve()
        for name in expected_flux:
            path = flux_dir / name
            item = flux_bracket["new_stream_manifest"][name]
            if (not path.is_file() or path.stat().st_size != 35 * 203 * 56 * 8
                    or _sha(path) != item["sha256"]):
                raise SystemExit(f"held Redi flux stream changed: {name}")
    else:
        if (args.round77 is not None or args.redi_flux_run_dir is not None
                or args.redi_flux_bracket is not None
                or args.redi_flux_bracket_sha is not None):
            raise SystemExit("round-78 flux inputs require --redi-flux-ladder")
    prior78 = None
    if args.redi_zfu_operand_ladder:
        if not args.redi_flux_ladder:
            raise SystemExit(
                "--redi-zfu-operand-ladder requires --redi-flux-ladder")
        if args.round78 is None or _sha(args.round78.resolve()) != ROUND78_SHA:
            raise SystemExit("official corrected round-78 receipt required")
        prior78 = json.loads(args.round78.read_text())
        if (prior78.get("session_id") != session
                or prior78.get("disposition") != "REDI_DIVERGED_ZFU_T"
                or prior78.get("first_diverged_subrow") != "78.T.1"
                or prior78.get("first_diverged_flux_operand") != "zfu_tem"
                or prior78["redi_flux_metrics"]["zfu_tem"]
                          ["n_diverged_columns"] != 9758):
            raise SystemExit("round 78 does not admit the zfu operand peel")
    elif args.round78 is not None:
        raise SystemExit(
            "--round78 requires --redi-zfu-operand-ladder")
    prior79 = None
    if args.redi_zfu_postfix:
        if not args.redi_zfu_operand_ladder:
            raise SystemExit(
                "--redi-zfu-postfix requires --redi-zfu-operand-ladder")
        if args.round79 is None or _sha(args.round79.resolve()) != ROUND79_SHA:
            raise SystemExit("official round-79 operand receipt required")
        prior79 = json.loads(args.round79.read_text())
        if (prior79.get("session_id") != session
                or prior79.get("disposition") !=
                "REDI_ZFU_T_OPERANDS_BOUNDED_ASSOCIATION_OPEN"):
            raise SystemExit("round 79 does not admit the production replay")
    elif args.round79 is not None:
        raise SystemExit("--round79 requires --redi-zfu-postfix")
    prior80 = None
    if args.redi_zfu_kmm_postfix:
        if not args.redi_zfu_postfix:
            raise SystemExit(
                "--redi-zfu-kmm-postfix requires --redi-zfu-postfix")
        if args.round80 is None or _sha(args.round80.resolve()) != ROUND80_SHA:
            raise SystemExit("official round-80 postfix receipt required")
        prior80 = json.loads(args.round80.read_text())
        if (prior80.get("session_id") != session
                or prior80.get("disposition") !=
                "REDI_ZFU_T_POSTFIX_REGRESSION"):
            raise SystemExit("round 80 does not admit the Kmm correction")
    elif args.round80 is not None:
        raise SystemExit("--round80 requires --redi-zfu-kmm-postfix")
    prior81 = None
    if args.redi_zfu_slope_kmm_postfix:
        if not args.redi_zfu_kmm_postfix:
            raise SystemExit(
                "--redi-zfu-slope-kmm-postfix requires the Kmm face arm")
        if args.round81 is None or _sha(args.round81.resolve()) != ROUND81_SHA:
            raise SystemExit("official corrected round-81 receipt required")
        prior81 = json.loads(args.round81.read_text())
        if prior81.get("disposition") != (
                "REDI_ZFU_T_KMM_FACE_FIXED_RESIDUAL_FINAL_USLP"):
            raise SystemExit("round 81 does not admit the slope Kmm carry")
    elif args.round81 is not None:
        raise SystemExit("--round81 requires --redi-zfu-slope-kmm-postfix")
    if args.redi_zfu_kmm_operator_postfix:
        if not args.redi_zfu_slope_kmm_postfix:
            raise SystemExit("Kmm operator arm requires slope Kmm arm")
        if args.round82 is None or _sha(args.round82.resolve()) != ROUND82_SHA:
            raise SystemExit("bound invalid round-82 receipt required")
    elif args.round82 is not None:
        raise SystemExit("--round82 requires --redi-zfu-kmm-operator-postfix")
    if args.redi_zfu_bolus_stage_split_postfix:
        if not args.redi_zfu_kmm_operator_postfix:
            raise SystemExit("bolus stage split requires Kmm operator arm")
        if args.round83 is None or _sha(args.round83.resolve()) != ROUND83_SHA:
            raise SystemExit("bound invalid round-83 receipt required")
        prior83 = json.loads(args.round83.read_text())
        if (prior83.get("disposition") != "INVALID"
                or prior83["rows"][5]["metrics"]["n_diverged_columns"]
                != 5988):
            raise SystemExit("round 83 does not admit the bolus stage split")
    elif args.round83 is not None:
        raise SystemExit(
            "--round83 requires --redi-zfu-bolus-stage-split-postfix")
    if args.redi_zfu_ahtu_postfix:
        if not args.redi_zfu_bolus_stage_split_postfix:
            raise SystemExit("ahtu postfix requires bolus stage split arm")
        if args.round84 is None or _sha(args.round84.resolve()) != ROUND84_SHA:
            raise SystemExit("bound official round-84 receipt required")
        prior84 = json.loads(args.round84.read_text())
        if prior84.get("disposition") != (
                "REDI_ZFU_T_BOLUS_STAGE_SPLIT_FIXED_AHTU_RESIDUAL"):
            raise SystemExit("round 84 does not admit the ahtu postfix")
    elif args.round84 is not None:
        raise SystemExit("--round84 requires --redi-zfu-ahtu-postfix")
    if args.redi_zfw_association_factorial:
        if not args.redi_zfu_ahtu_postfix:
            raise SystemExit("zfw factorial requires the ahtu postfix")
        if args.round85 is None or _sha(args.round85.resolve()) != ROUND85_SHA:
            raise SystemExit("bound official round-85 receipt required")
        prior85 = json.loads(args.round85.read_text())
        if (prior85.get("disposition") != "REDI_ZFU_T_AHTU_AT_BAR"
                or prior85["redi_flux_metrics"]["zfv_tem"]["pass"] is not True
                or prior85["redi_flux_metrics"]["zfw_tem"]["pass"] is not False):
            raise SystemExit("round 85 does not admit the zfw factorial")
    elif args.round85 is not None:
        raise SystemExit("--round85 requires --redi-zfw-association-factorial")
    component_dir = None
    if args.redi_zfw_component_score:
        if not args.redi_zfw_association_factorial:
            raise SystemExit("component score requires zfw factorial")
        if args.round86 is None or _sha(args.round86.resolve()) != ROUND86_SHA:
            raise SystemExit("bound official round-86 receipt required")
        if (args.redi_zfw_component_dir is None
                or args.redi_zfw_component_bracket is None
                or not args.redi_zfw_component_bracket_sha):
            raise SystemExit("component score requires dir, bracket, and SHA")
        if (_sha(args.redi_zfw_component_bracket.resolve())
                != args.redi_zfw_component_bracket_sha):
            raise SystemExit("zfw component bracket changed")
        component_dir = args.redi_zfw_component_dir.resolve()
        expected_components = {
            f"redi_dump_zfw_{term}_{tracer}.bin"
            for term in ("skew", "a33") for tracer in ("tem", "sal")}
        bracket = json.loads(args.redi_zfw_component_bracket.read_text())
        if (bracket.get("schema") != "dino-redi-zfw-component-bracket-v1"
                or set(bracket.get("new_streams", ())) != expected_components
                or not all(bracket.get("controls", {}).values())):
            raise SystemExit("zfw component bracket does not admit scoring")
        for name in expected_components:
            path = component_dir / name
            if (not path.is_file() or path.stat().st_size != 35 * 203 * 56 * 8
                    or _sha(path) != bracket["new_stream_manifest"][name]["sha256"]):
                raise SystemExit(f"held zfw component changed: {name}")
    elif (args.round86 is not None or args.redi_zfw_component_dir is not None
          or args.redi_zfw_component_bracket is not None
          or args.redi_zfw_component_bracket_sha is not None):
        raise SystemExit("round-87 component inputs require component score")
    if args.redi_zfw_skew_factorial:
        if not args.redi_zfw_component_score:
            raise SystemExit("skew factorial requires component score")
        if args.round87 is None or _sha(args.round87.resolve()) != ROUND87_SHA:
            raise SystemExit("bound official round-87 receipt required")
        prior87 = json.loads(args.round87.read_text())
        if (prior87.get("disposition") != "REDI_ZFW_T_DIVERGED_A31_A32"
                or prior87["redi_zfw_component_metrics"]["skew"]["pass"]
                or prior87["redi_zfw_component_metrics"]["a33"]["pass"]):
            raise SystemExit("round 87 does not admit the ordered skew peel")
        for name, expected in REDI_SKEW_INPUT_SHA.items():
            if _sha(args.redi_flux_run_dir.resolve() / name) != expected:
                raise SystemExit(f"held Redi skew input changed: {name}")
    elif args.round87 is not None:
        raise SystemExit("--round87 requires --redi-zfw-skew-factorial")
    if args.redi_zfw_skew_postfix:
        if not args.redi_zfw_component_score:
            raise SystemExit("skew postfix requires component score")
        if args.round88 is None or _sha(args.round88.resolve()) != ROUND88_SHA:
            raise SystemExit("bound official round-88 receipt required")
        prior88 = json.loads(args.round88.read_text())
        if (prior88.get("disposition") !=
                "REDI_ZFW_T_SKEW_LOCALIZED_TO_COEFFICIENT_ASSEMBLY"
                or not prior88["redi_zfw_skew_factorial"]["arms"][
                    "C1G1T0"]["pass"]):
            raise SystemExit("round 88 does not admit the production postfix")
    elif args.round88 is not None:
        raise SystemExit("--round88 requires --redi-zfw-skew-postfix")
    if args.redi_zfw_skew_operand_factorial:
        if not args.redi_zfw_skew_postfix:
            raise SystemExit("skew operand factorial requires skew postfix")
        if args.round89 is None or _sha(args.round89.resolve()) != ROUND89_SHA:
            raise SystemExit("bound official round-89 receipt required")
        prior89 = json.loads(args.round89.read_text())
        if (prior89.get("disposition") !=
                "REDI_ZFW_T_SKEW_POSTFIX_REGRESSION"
                or prior89["redi_zfw_component_metrics"]["skew"]["pass"]):
            raise SystemExit("round 89 does not admit the four-operand peel")
        for name, expected in REDI_SKEW_INPUT_SHA.items():
            if _sha(args.redi_flux_run_dir.resolve() / name) != expected:
                raise SystemExit(f"held Redi skew input changed: {name}")
    elif args.round89 is not None:
        raise SystemExit(
            "--round89 requires --redi-zfw-skew-operand-factorial")
    if args.redi_zfw_wslp_stage_factorial:
        if not args.redi_zfw_skew_operand_factorial:
            raise SystemExit("W-slope stage factorial requires operand factorial")
        if args.round90 is None or _sha(args.round90.resolve()) != ROUND90_SHA:
            raise SystemExit("bound official round-90 receipt required")
        prior90 = json.loads(args.round90.read_text())
        if prior90.get("disposition") != "REDI_ZFW_T_SKEW_OWNED_WSLPI_WSLPJ":
            raise SystemExit("round 90 does not admit the W-slope stage peel")
    elif args.round90 is not None:
        raise SystemExit(
            "--round90 requires --redi-zfw-wslp-stage-factorial")
    prior93_wslp = None
    if args.redi_zfw_a33_floor_factorial:
        if not args.redi_zfw_wslp_stage_factorial:
            raise SystemExit("A33 floor factorial requires W-slope factorial")
        if (args.round93_wslp is None
                or _sha(args.round93_wslp.resolve()) != ROUND93_WSLP_SHA):
            raise SystemExit("bound round-93 W-slope receipt required")
        prior93_wslp = json.loads(args.round93_wslp.read_text())
        if (prior93_wslp.get("disposition")
                != "WSLOPE_ASSOCIATION_AT_BAR_R1S0"
                or not all(prior93_wslp["arms"]["R1S0"][name]["pass"]
                           for name in ("wslpi", "wslpj"))):
            raise SystemExit("round-93 W-slope receipt does not release A33")
    elif args.round93_wslp is not None:
        raise SystemExit(
            "--round93-wslp requires --redi-zfw-a33-floor-factorial")
    if _sha(args.raw_artifact.resolve()) != RAW_ARTIFACT_SHA:
        raise SystemExit("admitted held row-8 artifact changed")
    held = args.held_dir.resolve()
    for name, expected in HELD_SHA.items():
        if not (held / name).is_file() or _sha(held / name) != expected:
            raise SystemExit(f"held row-8 input changed: {name}")
    if args.redi_zfu_operand_ladder:
        for name, expected in {
                "ldftra_dump_ahtu.bin":
                    "098b95a3548ee8d3c66694ca43c3b720840e2c193895b0df222e8e186b2580da",
                "eiv_dump_uslp.bin":
                    "ecb3ca3c50c97ef67ad45e96f828a5264c18f94b767a0f9d619f010535d9d0f9",
        }.items():
            path = flux_dir / name
            if not path.is_file() or _sha(path) != expected:
                raise SystemExit(f"held Redi zfu operand changed: {name}")
    if args.capture_bolus_operands:
        for name, expected in BOLUS_HELD_SHA.items():
            if not (held / name).is_file() or _sha(held / name) != expected:
                raise SystemExit(f"held GM operand changed: {name}")
    if args.capture_kappa_operands:
        for name, expected in KAPPA_HELD_SHA.items():
            if not (held / name).is_file() or _sha(held / name) != expected:
                raise SystemExit(f"held GM coefficient operand changed: {name}")
    if args.capture_kappa_geometry:
        for name, expected in KAPPA_GEOMETRY_SHA.items():
            if not (held / name).is_file() or _sha(held / name) != expected:
                raise SystemExit(f"held GM geometry operand changed: {name}")
    if args.capture_redi_tail:
        if args.redi_run_dir is None:
            raise SystemExit("--capture-redi-tail requires --redi-run-dir")
        redi_dir = args.redi_run_dir.resolve()
        for name, expected in REDI_HELD_SHA.items():
            if not (redi_dir / name).is_file() or _sha(redi_dir / name) != expected:
                raise SystemExit(f"held Redi bracket changed: {name}")
    elif args.redi_run_dir is not None:
        raise SystemExit("--redi-run-dir requires --capture-redi-tail")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    _, _, cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(held),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    state = model._seed_tke_preclosure_carry(state)
    captured = []
    original = model_module.add_bolus_to_advecting_flux
    original_solver = model_module.barotropic_substeps_latlon_cgrid
    original_cycle = model_module.nemo_qco_kmm_velocity_cycle
    # The MLF tracer transport's live-thickness entry is now the SHARED
    # NEMO qco builder (vertical.nemo_qco_live_face_geometry_cgrid, the one
    # the WS-RK3 stage transport calls too), and it returns legoESM's
    # REDUNDANT west/south faces, so the capture is sliced to native below.
    original_thickness = model_module.nemo_qco_live_face_geometry_cgrid
    solver_calls = 0
    transport_substitutions = 0
    solver_transport_captures = []
    cycle_captures = []
    thickness_captures = []
    bolus_captures = []
    kappa_captures = []
    redi_captures = []
    held_u_native = np.fromfile(held / "spg_dump_zu_frc.bin", dtype="<f8")
    held_v_native = np.fromfile(held / "spg_dump_zv_frc.bin", dtype="<f8")
    if held_u_native.size != 199 * 52 or held_v_native.size != 199 * 52:
        raise SystemExit("held slow forcing must be cited-interior (199,52)")
    held_u = np.concatenate(
        [held_u_native.reshape(199, 52)[:, -1:], held_u_native.reshape(199, 52)],
        axis=1)
    held_v = np.concatenate(
        [np.zeros((1, 52)), held_v_native.reshape(199, 52)], axis=0)
    oracle_u_native = np.fromfile(
        held / "spg_dump_un_adv_final.bin", dtype="<f8")
    oracle_v_native = np.fromfile(
        held / "spg_dump_vn_adv_final.bin", dtype="<f8")
    if oracle_u_native.size != 203 * 56 or oracle_v_native.size != 203 * 56:
        raise SystemExit("final transport writers must be full-halo (203,56)")
    oracle_u_native = oracle_u_native.reshape(203, 56)[2:-2, 2:-2]
    oracle_v_native = oracle_v_native.reshape(203, 56)[2:-2, 2:-2]
    oracle_u = np.concatenate([
        oracle_u_native[:, -1:], oracle_u_native], axis=1)
    oracle_v = np.concatenate([np.zeros((1, 52)), oracle_v_native], axis=0)

    def observe(bolus, mass_flux_u, mass_flux_v, u_mask, v_mask, grid, z_coord):
        result = original(
            bolus, mass_flux_u, mass_flux_v, u_mask, v_mask, grid, z_coord)
        captured.append((mass_flux_u, result[0], u_mask))
        return result

    def held_solver(state_arg, dt_s, n_substeps, grid, z_coord, config, **kwargs):
        nonlocal solver_calls, transport_substitutions
        kwargs = dict(kwargs)
        is_target = kwargs.get("eta_init") is not None
        if is_target:
            solver_calls += 1
            kwargs["F_slow_u"] = jnp.asarray(
                held_u, dtype=kwargs["F_slow_u"].dtype)
            kwargs["F_slow_v"] = jnp.asarray(
                held_v, dtype=kwargs["F_slow_v"].dtype)
        result = original_solver(
            state_arg, dt_s, n_substeps, grid, z_coord, config, **kwargs)
        if is_target and args.capture_cycle:
            solver_transport_captures.append(tuple(
                np.asarray(value) for value in result[1]))
        if is_target and args.oracle_transport:
            transport_substitutions += 1
            solved_state, _ = result
            result = (solved_state, (
                jnp.asarray(oracle_u, dtype=state_arg.eta.data.dtype),
                jnp.asarray(oracle_v, dtype=state_arg.eta.data.dtype)))
        return result

    def observed_cycle(*cycle_args, **cycle_kwargs):
        result = original_cycle(*cycle_args, **cycle_kwargs)
        cycle_captures.append({
            "un_adv": np.asarray(cycle_args[3]),
            "vn_adv": np.asarray(cycle_args[4]),
            "corrected_u": np.asarray(result[0]),
            "corrected_v": np.asarray(result[1]),
        })
        return result

    def observed_thickness(*thickness_args, **thickness_kwargs):
        result = original_thickness(*thickness_args, **thickness_kwargs)
        thickness_captures.append(tuple(np.asarray(value) for value in result))
        return result

    original_bolus = gm_module.nemo_eiv_bolus_transport
    original_kappa = gm_module.compute_treguier_kappa_gm_nemo_native
    original_redi = gm_module.nemo_iso_lap_tracer_tendency_latlon_cgrid

    def observed_bolus(*bolus_args, **bolus_kwargs):
        result = original_bolus(*bolus_args, **bolus_kwargs)
        bolus_captures.append((
            tuple(np.asarray(value) if hasattr(value, "shape") else value
                  for value in bolus_args),
            dict(bolus_kwargs),
            tuple(np.asarray(value) for value in result),
        ))
        return result

    def observed_kappa(*kappa_args, **kappa_kwargs):
        result = original_kappa(*kappa_args, **kappa_kwargs)
        diagnostic_kwargs = dict(kappa_kwargs)
        diagnostic_kwargs["return_diagnostics"] = True
        replay, diagnostics = original_kappa(
            *kappa_args, **diagnostic_kwargs)
        raw_operands = {}
        if args.capture_kappa_geometry:
            if not all(name in kappa_kwargs for name in
                       ("rho_0", "g", "active_3d", "jacobian", "slope_n2")):
                raise SystemExit("production kappa call omitted registered operands")
            raw_e3w, raw_wmask, raw_pn2 = gm_module._nemo_wpoint_e3w_wmask_n2(
                kappa_args[0], kappa_args[1], kappa_args[2], kappa_args[6],
                kappa_args[10], kappa_kwargs["rho_0"], kappa_kwargs["g"],
                kappa_kwargs["active_3d"],
                slope_n2=kappa_kwargs["slope_n2"],
                jacobian=kappa_kwargs["jacobian"])
            if args.coupled_kappa_carry:
                if (kappa_kwargs.get("pn2_override") is None
                        or kappa_kwargs.get("e3w_override") is None):
                    raise SystemExit("production Treguier call dropped Kmm carry")
                nlev = kappa_args[0].shape[-1]
                raw_pn2 = np.asarray(kappa_kwargs["pn2_override"])
                if raw_pn2.shape[-1] == nlev - 1:
                    raw_pn2 = np.concatenate(
                        [np.zeros_like(raw_pn2[..., :1]), raw_pn2], axis=-1)
                raw_e3w_override = np.asarray(kappa_kwargs["e3w_override"])
                if raw_e3w_override.shape[-1] == nlev - 1:
                    surface = np.broadcast_to(
                        np.asarray(raw_e3w)[..., :1],
                        raw_e3w_override.shape[:-1] + (1,))
                    raw_e3w = np.concatenate(
                        [surface, raw_e3w_override], axis=-1)
                else:
                    raw_e3w = raw_e3w_override
            raw_operands = {
                "e3w": np.asarray(raw_e3w),
                "wmask": np.asarray(raw_wmask),
                "pn2": np.asarray(raw_pn2),
            }
        if args.post_chain_factorial:
            raw_operands.update({
                "f_coriolis": np.asarray(kappa_args[8]),
                "surface_mask": np.asarray(kappa_args[5]),
                "omega": float(kappa_kwargs["omega"]),
                "aei0": float(kappa_args[9].aei0),
                "lat_t": np.asarray(kappa_args[7].lat2d),
            })
        kappa_captures.append((
            np.asarray(result), np.asarray(replay),
            {name: np.asarray(value) for name, value in diagnostics.items()},
            raw_operands))
        return result

    def observed_redi(*redi_args, **redi_kwargs):
        result = original_redi(*redi_args, **redi_kwargs)
        output = (result[0] if isinstance(result, tuple)
                  and redi_kwargs.get("return_bolus") else result)
        redi_captures.append({
            "args": redi_args,
            "kwargs": dict(redi_kwargs),
            "q": np.asarray(redi_args[0]),
            "output": np.asarray(output),
        })
        return result

    model_module.add_bolus_to_advecting_flux = observe
    if args.hold_slow_forcing:
        model_module.barotropic_substeps_latlon_cgrid = held_solver
    if args.capture_cycle:
        model_module.nemo_qco_kmm_velocity_cycle = observed_cycle
    if args.live_thickness_entry:
        model_module.nemo_qco_live_face_geometry_cgrid = observed_thickness
    if args.capture_bolus_operands:
        gm_module.nemo_eiv_bolus_transport = observed_bolus
    if args.capture_kappa_operands:
        gm_module.compute_treguier_kappa_gm_nemo_native = observed_kappa
    if args.capture_redi_tail:
        gm_module.nemo_iso_lap_tracer_tendency_latlon_cgrid = observed_redi
    try:
        with jax.disable_jit():
            model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.add_bolus_to_advecting_flux = original
        model_module.barotropic_substeps_latlon_cgrid = original_solver
        model_module.nemo_qco_kmm_velocity_cycle = original_cycle
        model_module.nemo_qco_live_face_geometry_cgrid = original_thickness
        gm_module.nemo_eiv_bolus_transport = original_bolus
        gm_module.compute_treguier_kappa_gm_nemo_native = original_kappa
        gm_module.nemo_iso_lap_tracer_tendency_latlon_cgrid = original_redi
    restored = (model_module.add_bolus_to_advecting_flux is original
                and model_module.barotropic_substeps_latlon_cgrid is original_solver
                and model_module.nemo_qco_kmm_velocity_cycle is original_cycle
                and model_module.nemo_qco_live_face_geometry_cgrid
                is original_thickness
                and gm_module.nemo_eiv_bolus_transport is original_bolus)
    restored = (restored
                and gm_module.compute_treguier_kappa_gm_nemo_native
                is original_kappa)
    restored = (restored
                and gm_module.nemo_iso_lap_tracer_tendency_latlon_cgrid
                is original_redi)
    if len(captured) != 1:
        raise SystemExit(f"expected one single-pass tracer handoff, got {len(captured)}")

    base_u, total_u, u_mask = (np.asarray(x) for x in captured[0])
    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, model.z_coord,
        min_water_column_m=cfg.min_water_column_m)
    h_u = np.asarray(min_cell_to_uface(h_k))
    u_corrected = np.divide(
        base_u, h_u, out=np.zeros_like(base_u),
        where=(np.asarray(u_mask, dtype=bool) & (h_u != 0.0)))
    e2u = np.asarray(model.grid.dy_u)[:, :, None]
    proxy_un = u_corrected[:, 1:, :35]
    values = {
        "un": proxy_un,
        "e2u": np.broadcast_to(e2u[:, 1:, :], u_corrected[:, 1:, :35].shape),
        "e3u": h_u[:, 1:, :35],
        "e2e3u": (e2u * h_u)[:, 1:, :35],
        "pu_euler": (e2u * base_u)[:, 1:, :35],
        "pu_bolus": (e2u * (total_u - base_u))[:, 1:, :35],
        "pu_total": (e2u * total_u)[:, 1:, :35],
    }
    donor = np.asarray(upwind_to_u_points(state.T_before.data, total_u))
    values["upstream"] = (e2u * total_u * donor)[:, 1:, :35]
    oracle = {
        "un": _load(held / "fct_entry_dump_un.bin"),
        "e2u": _load(held / "fct_entry_dump_e2u.bin"),
        "e3u": _load(held / "fct_entry_dump_e3u.bin"),
        "e2e3u": _load(held / "fct_entry_dump_e2e3u.bin"),
        "pu_euler": _load(held / "fct_entry_dump_pu_euler.bin"),
        "pu_bolus": _load(held / "fct_entry_dump_pu_bolus.bin"),
        "pu_total": _load(held / "fct_entry_dump_pu_total.bin"),
        "upstream": _load(held / "fct_dump_zwx_up.bin"),
    }
    # Registry population is NEMO's real 3-D umask, not nonzero geometric
    # thickness.  The latter admits 590 dry surface faces because e3u remains
    # defined under land and produced the invalid 10,348-column round-55--57
    # population.  The captured production mask is the bridge of that umask.
    wet = np.asarray(u_mask, dtype=bool)[:, 1:, :35]
    if int(np.any(wet, axis=-1).sum()) != 9758 or int(wet.sum()) != 336338:
        raise SystemExit("registered U population changed from 9758/336338")
    capture_metrics = {}
    if args.capture_cycle:
        if len(solver_transport_captures) != 1 or not cycle_captures:
            raise SystemExit(
                "expected one solver transport and at least one Kmm-cycle capture")
        wet2 = np.any(wet, axis=-1)
        wet2_3d = wet2[..., None]
        oracle_transport_3d = oracle_u_native[..., None]
        production_hu = solver_transport_captures[0][0][:, 1:]
        consumed_hu = cycle_captures[0]["un_adv"][:, 1:]
        capture_metrics = {
            "production_Hu_avg": sweep.metrics(
                production_hu[..., None], oracle_transport_3d,
                wet2_3d, FOCUS, POINTWISE_BAR),
            "consumed_Hu_avg": sweep.metrics(
                consumed_hu[..., None], oracle_transport_3d,
                wet2_3d, FOCUS, POINTWISE_BAR),
            "cycle_corrected_u": sweep.metrics(
                cycle_captures[0]["corrected_u"][:, 1:, :35],
                oracle["un"], wet, FOCUS, POINTWISE_BAR),
        }
        if args.direct_cycle_entry:
            values["un"] = cycle_captures[0]["corrected_u"][:, 1:, :35]
    if args.live_thickness_entry:
        if len(thickness_captures) != 1:
            raise SystemExit(
                "expected exactly one tracer live-thickness capture")
        live_u_raw = thickness_captures[0][0][:, 1:, :35]
        values["e3u"] = live_u_raw
        values["e2e3u"] = e2u[:, 1:, :] * live_u_raw
    bolus_operand_metrics = {}
    if args.capture_bolus_operands:
        if not bolus_captures:
            raise SystemExit("expected at least one GM bolus producer capture")
        bargs, bkwargs, bresult = bolus_captures[0]
        if not all(np.array_equal(capture[2][0], bresult[0])
                   for capture in bolus_captures[1:]):
            raise SystemExit("multiple GM bolus calls produced different U transports")
        kappa, slope_kp1, _, e2u_native, _, u_mask_native, _, act, act_below = bargs[:9]
        out_shape = tuple(bargs[9])
        kappa = np.asarray(kappa)
        if kappa.ndim == 2:
            kappa = np.broadcast_to(kappa[..., None], out_shape)
        elif kappa.ndim != 3:
            kappa = np.broadcast_to(kappa, out_shape)
        if not bkwargs.get("kappa_face_average", False):
            raise SystemExit("round 65 requires production face-averaged kappa")
        aeiu = 0.5 * (kappa + np.roll(kappa, -1, axis=1))
        slope_sum = slope_kp1 + np.roll(slope_kp1, -1, axis=1)
        aeiu_sum = aeiu + np.roll(aeiu, -1, axis=2)
        wumask = (u_mask_native[:, 1:, None] * act
                  * np.roll(act, -1, axis=1) * act_below
                  * np.roll(act_below, -1, axis=1))
        current_psi = -(e2u_native[..., None] * (0.5 * slope_sum)
                        * (0.5 * aeiu_sum) * wumask)
        literal_psi = (((-0.25 * e2u_native[..., None]) * slope_sum)
                       * aeiu_sum) * wumask
        oracle_psi = _load(held / "eiv_dump_psi_uw.bin")
        oracle_u_eiv = -_load(held / "eiv_dump_u.bin")
        oracle_wslpi = _load(held / "eiv_dump_wslpi.bin")
        oracle_aeiu = _load(held / "eiv_dump_aeiu.bin")
        zero_level = np.zeros_like(oracle_wslpi[..., :1])
        oracle_slope_kp1 = np.concatenate(
            [oracle_wslpi[..., 1:], zero_level], axis=2)
        oracle_aeiu_kp1 = np.concatenate(
            [oracle_aeiu[..., 1:], np.zeros_like(oracle_aeiu[..., :1])], axis=2)
        oracle_slope_sum = oracle_slope_kp1 + np.roll(
            oracle_slope_kp1, -1, axis=1)
        oracle_aeiu_sum = oracle_aeiu + oracle_aeiu_kp1
        own_oracle_slope = (((-0.25 * e2u_native[..., None])
                             * oracle_slope_sum) * aeiu_sum[..., :35]) \
            * wumask[..., :35]
        own_oracle_aeiu = (((-0.25 * e2u_native[..., None])
                            * slope_sum[..., :35]) * oracle_aeiu_sum) \
            * wumask[..., :35]
        oracle_both = (((-0.25 * e2u_native[..., None])
                        * oracle_slope_sum) * oracle_aeiu_sum) \
            * wumask[..., :35]
        wet_psi = np.asarray(wumask[..., :35], dtype=bool)
        if int(np.any(wet_psi, axis=-1).sum()) != 9758:
            raise SystemExit("GM psi registered U-column population changed")
        def bm(value, oracle_value=oracle_psi):
            return sweep.metrics(
                np.asarray(value)[..., :35], oracle_value, wet_psi,
                FOCUS, ACCUMULATION_BAR)
        bolus_operand_metrics = {
            "captured_u_increment": sweep.metrics(
                bresult[0][..., :35], oracle_u_eiv, wet_psi,
                FOCUS, ACCUMULATION_BAR),
            "aeiu_face": bm(aeiu, oracle_aeiu),
            "wslpi_kp1_face_sum": bm(slope_sum, oracle_slope_sum),
            "psi_current_normalized": bm(current_psi),
            "psi_literal_same_operands": bm(literal_psi),
            "factorial_own_slope_own_aeiu": bm(literal_psi),
            "factorial_oracle_slope_own_aeiu": bm(own_oracle_slope),
            "factorial_own_slope_oracle_aeiu": bm(own_oracle_aeiu),
            "factorial_oracle_slope_oracle_aeiu": bm(oracle_both),
        }
    kappa_operand_metrics = {}
    kappa_geometry_metrics = {}
    post_chain_metrics = {}
    rossby_metrics = {}
    zn_sqrt_metrics = {}
    redi_metrics = {}
    redi_e3w_metrics = {}
    redi_flux_metrics = {}
    redi_zfu_operand_ladder = {}
    redi_zfw_association_factorial = {}
    redi_zfw_component_metrics = {}
    redi_zfw_skew_factorial = {}
    redi_zfw_skew_operand_factorial = {}
    redi_zfw_wslp_stage_factorial = {}
    redi_zfw_a33_floor_factorial = {}
    redi_flux_rows = []
    first_flux = None
    first_flux_subrow = None
    if args.capture_kappa_operands:
        if len(kappa_captures) != 1:
            raise SystemExit(
                f"expected one Treguier coefficient call, got {len(kappa_captures)}")
        kappa_result, kappa_replay, diagnostics, raw_kappa = kappa_captures[0]
        if not np.array_equal(kappa_result, kappa_replay):
            raise SystemExit("diagnostic replay changed the production kappa")
        oracle2 = {
            name: _load2(held / f"eiv_dump_{name}.bin")
            for name in ("zn", "zah", "zhw", "zRo", "zaeiw")
        }
        wet2 = np.any(wet, axis=-1)
        wet2_3d = wet2[..., None]
        def km(value, oracle_value):
            return sweep.metrics(
                np.asarray(value)[..., None], oracle_value[..., None],
                wet2_3d, FOCUS, ACCUMULATION_BAR)
        for name in ("zn", "zah", "zhw", "zRo", "zaeiw"):
            kappa_operand_metrics[name] = km(diagnostics[name], oracle2[name])
        oracle_face = (0.5 * (oracle2["zaeiw"]
                              + np.roll(oracle2["zaeiw"], -1, axis=1))
                       * np.asarray(u_mask_native[:, 1:], dtype=bool))
        kappa_operand_metrics["aeiu_face"] = km(aeiu[..., 0], oracle_aeiu[..., 0])
        kappa_operand_metrics["oracle_zaeiw_to_aeiu"] = km(
            oracle_face, oracle_aeiu[..., 0])
        if args.capture_kappa_geometry:
            oracle_e3w = _load(held / "eiv_dump_e3w.bin")
            oracle_pn2 = _load(held / "eiv_dump_rn2b.bin")
            raw_e3w = raw_kappa["e3w"]
            if raw_e3w.ndim == 1:
                raw_e3w = np.broadcast_to(
                    raw_e3w[None, None, :], raw_kappa["pn2"].shape)
            wet_w = np.asarray(raw_kappa["wmask"][..., :35], dtype=bool)
            def gm(value, oracle_value):
                return sweep.metrics(
                    np.asarray(value)[..., :35], oracle_value, wet_w,
                    FOCUS, ACCUMULATION_BAR)
            kappa_geometry_metrics = {
                "e3w_Kmm": gm(raw_e3w, oracle_e3w),
                "rn2b": gm(raw_kappa["pn2"], oracle_pn2),
            }
        if args.post_chain_factorial:
            dtype = diagnostics["zRo"].dtype
            ff_t = jnp.asarray(raw_kappa["f_coriolis"], dtype=dtype)
            ssmask = jnp.asarray(raw_kappa["surface_mask"], dtype=dtype)
            omega = jnp.asarray(raw_kappa["omega"], dtype=dtype)
            aei0 = jnp.asarray(raw_kappa["aei0"], dtype=dtype)
            rad = jnp.asarray(np.pi / 180.0, dtype=dtype)
            z1_f20 = (jnp.asarray(1.0, dtype=dtype)
                      / (jnp.asarray(2.0, dtype=dtype) * omega
                         * jnp.sin(rad * jnp.asarray(20.0, dtype=dtype))))

            def literal_post(ro_value, zah_value, zhw_value):
                zaeiw_pre = ((ro_value * ro_value)
                              * jnp.sqrt(zah_value / zhw_value)) * ssmask
                zzaei = jnp.minimum(
                    jnp.asarray(1.0, dtype=dtype),
                    jnp.abs(ff_t * z1_f20)) * zaeiw_pre
                return jnp.minimum(zzaei, aei0)

            own = {name: jnp.asarray(diagnostics[name], dtype=dtype)
                   for name in ("zRo", "zah", "zhw")}
            ora = {name: jnp.asarray(oracle2[name], dtype=dtype)
                   for name in ("zRo", "zah", "zhw")}
            arms = {
                "literal_own": literal_post(
                    own["zRo"], own["zah"], own["zhw"]),
                "literal_oracle_zRo": literal_post(
                    ora["zRo"], own["zah"], own["zhw"]),
                "literal_oracle_zah_zhw": literal_post(
                    own["zRo"], ora["zah"], ora["zhw"]),
                "literal_all_oracle": literal_post(
                    ora["zRo"], ora["zah"], ora["zhw"]),
            }
            u_surf_mask = jnp.asarray(bargs[5][:, 1:], dtype=dtype)
            def score_kappa_arm(arm_kappa):
                arm_args = list(bargs)
                arm_args[0] = arm_kappa
                arm_u = np.asarray(original_bolus(*arm_args, **bkwargs)[0])
                arm_aeiu = (0.5 * (arm_kappa
                                    + jnp.roll(arm_kappa, -1, axis=1))
                            * u_surf_mask)
                return {
                    "zaeiw": km(arm_kappa, oracle2["zaeiw"]),
                    "aeiu": km(arm_aeiu, oracle_aeiu[..., 0]),
                    "row8_8": sweep.metrics(
                        arm_u[..., :35], oracle_u_eiv, wet_psi,
                        FOCUS, ACCUMULATION_BAR),
                }
            for arm_name, arm_kappa in arms.items():
                post_chain_metrics[arm_name] = score_kappa_arm(arm_kappa)
            post_chain_metrics["production"] = {
                "zaeiw": kappa_operand_metrics["zaeiw"],
                "aeiu": kappa_operand_metrics["aeiu_face"],
                "row8_8": bolus_operand_metrics["captured_u_increment"],
            }
            if args.rossby_factorial:
                lat_t = jnp.asarray(raw_kappa["lat_t"], dtype=dtype)
                f_floor = jnp.asarray(1.0e-10, dtype=dtype)
                zfw_stored = jnp.maximum(jnp.abs(ff_t), f_floor)
                zfw_recomputed = jnp.maximum(
                    jnp.abs((jnp.asarray(2.0, dtype=dtype) * omega)
                            # NEMO gphit is degrees and uses rad*gphit;
                            # LatLonGrid.lat2d is already radians.
                            * jnp.sin(lat_t)), f_floor)
                ro_min = jnp.asarray(2.0e3, dtype=dtype)
                ro_max = jnp.asarray(40.0e3, dtype=dtype)

                def literal_ro(zn_value, zfw_value):
                    return jnp.maximum(
                        ro_min,
                        jnp.minimum(
                            (jnp.asarray(0.4, dtype=dtype) * zn_value)
                            / zfw_value,
                            ro_max))

                own_zn = jnp.asarray(diagnostics["zn"], dtype=dtype)
                oracle_zn = jnp.asarray(oracle2["zn"], dtype=dtype)
                ro_arms = {
                    "literal_stored_f": literal_ro(own_zn, zfw_stored),
                    "literal_recomputed_f": literal_ro(
                        own_zn, zfw_recomputed),
                    "literal_oracle_zn": literal_ro(
                        oracle_zn, zfw_stored),
                    "literal_oracle_zn_recomputed_f": literal_ro(
                        oracle_zn, zfw_recomputed),
                    "direct_oracle_zRo": jnp.asarray(
                        oracle2["zRo"], dtype=dtype),
                }
                for arm_name, arm_ro in ro_arms.items():
                    arm_kappa = literal_post(
                        arm_ro, own["zah"], own["zhw"])
                    rossby_metrics[arm_name] = {
                        "zRo": km(arm_ro, oracle2["zRo"]),
                        **score_kappa_arm(arm_kappa),
                    }
                rossby_metrics["production"] = {
                    "zRo": kappa_operand_metrics["zRo"],
                    **post_chain_metrics["production"],
                }
                if args.zn_sqrt_factorial:
                    raw_pn2 = jnp.asarray(raw_kappa["pn2"], dtype=dtype)
                    raw_e3w = jnp.asarray(raw_kappa["e3w"], dtype=dtype)
                    if raw_e3w.ndim == 1:
                        raw_e3w = jnp.broadcast_to(raw_e3w, raw_pn2.shape)
                    zn_term_exact = jnp.sqrt(jnp.maximum(
                        raw_pn2, jnp.asarray(0.0, dtype=dtype))) * raw_e3w
                    zn_term_guarded = jnp.sqrt(jnp.maximum(
                        raw_pn2, jnp.asarray(1.0e-30, dtype=dtype))) * raw_e3w
                    zeros = jnp.zeros_like(zn_term_exact)
                    zn_exact, _, _ = gm_module._nemo_treguier_left_reductions(
                        zn_term_exact, zeros, zeros,
                        jnp.asarray(0.0, dtype=dtype))
                    zn_guarded, _, _ = gm_module._nemo_treguier_left_reductions(
                        zn_term_guarded, zeros, zeros,
                        jnp.asarray(0.0, dtype=dtype))
                    for arm_name, arm_zn in {
                            "guarded_floor": zn_guarded,
                            "exact_forward": zn_exact}.items():
                        arm_ro = literal_ro(arm_zn, zfw_stored)
                        arm_kappa = literal_post(
                            arm_ro, own["zah"], own["zhw"])
                        zn_sqrt_metrics[arm_name] = {
                            "zn": km(arm_zn, oracle2["zn"]),
                            "zRo": km(arm_ro, oracle2["zRo"]),
                            **score_kappa_arm(arm_kappa),
                        }
                    zn_sqrt_metrics["production"] = {
                        "zn": kappa_operand_metrics["zn"],
                        "zRo": kappa_operand_metrics["zRo"],
                        "zaeiw": kappa_operand_metrics["zaeiw"],
                        "aeiu": kappa_operand_metrics["aeiu_face"],
                        "row8_8": bolus_operand_metrics["captured_u_increment"],
                    }
                    pn2_np = np.asarray(raw_kappa["pn2"])
                    zn_sqrt_metrics["operand_receipts"] = {
                        "n_zero_n2_slots": int(np.count_nonzero(pn2_np == 0.0)),
                        "n_negative_n2_slots": int(np.count_nonzero(pn2_np < 0.0)),
                        "exact_differs_from_guarded": bool(
                            np.any(np.asarray(zn_exact)
                                   != np.asarray(zn_guarded))),
                        "negative_forward_clipped_zero": bool(np.all(
                            np.asarray(jnp.sqrt(jnp.maximum(
                                raw_pn2, jnp.asarray(0.0, dtype=dtype))))
                            [pn2_np < 0.0] == 0.0)),
                    }
    if args.capture_redi_tail:
        wet_t = np.broadcast_to(
            np.asarray(state.land_mask.data, dtype=bool)[..., None],
            np.asarray(state.T_before.data).shape).copy()
        if getattr(model.z_coord, "is_active", None) is not None:
            wet_t &= np.asarray(model.z_coord.is_active, dtype=bool)
        wet_t = wet_t[..., :35]
        if int(np.any(wet_t, axis=-1).sum()) != 9920:
            raise SystemExit("registered Redi T-column census changed")
        matched = {}
        targets = {
            "temperature": np.asarray(state.T_before.data),
            "salinity": np.asarray(state.S_before.data),
        }
        matched_calls = {}
        for capture in redi_captures:
            q_in = capture["q"]
            for name, target in targets.items():
                if name not in matched and np.array_equal(
                        q_in[..., :35][wet_t], target[..., :35][wet_t]):
                    matched[name] = capture["output"][..., :35]
                    matched_calls[name] = capture
                    break
        if set(matched) != set(targets) or len(redi_captures) != 2:
            raise SystemExit(
                "expected exactly the T_before/S_before Redi calls; got "
                f"{len(redi_captures)} calls and matches {sorted(matched)}")
        redi_oracle = {}
        for name, suffix in (("temperature", "tem"), ("salinity", "sal")):
            redi_oracle[name] = (
                _load(redi_dir / f"stp_dump_23_after_traldf_{suffix}.bin")
                - _load(redi_dir / f"stp_dump_22_before_traldf_{suffix}.bin"))
            redi_metrics[name] = sweep.metrics(
                matched[name], redi_oracle[name], wet_t,
                FOCUS, ACCUMULATION_BAR)
        if args.redi_e3w_factorial:
            exact_e3w = np.asarray(raw_kappa["e3w"])
            if exact_e3w.shape != np.asarray(state.T_before.data).shape:
                raise SystemExit(
                    "registered live Kmm e3w must have full tracer shape; "
                    f"got {exact_e3w.shape}")
            for name in ("temperature", "salinity"):
                capture = matched_calls[name]
                replay_kwargs = dict(capture["kwargs"])
                replay_kwargs["msc_e3w_override"] = jnp.asarray(
                    exact_e3w, dtype=capture["args"][0].dtype)
                if args.redi_flux_ladder:
                    replay_kwargs["return_diagnostics"] = True
                    replay_kwargs["return_operand_diagnostics"] = bool(
                        args.redi_zfu_operand_ladder
                        and name == "temperature")
                replay = original_redi(*capture["args"], **replay_kwargs)
                diagnostics_replay = None
                if args.redi_flux_ladder:
                    if replay_kwargs.get("return_bolus"):
                        replay, _, diagnostics_replay = replay
                    else:
                        replay, diagnostics_replay = replay
                elif isinstance(replay, tuple) and replay_kwargs.get("return_bolus"):
                    replay = replay[0]
                exact_metric = sweep.metrics(
                    np.asarray(replay)[..., :35], redi_oracle[name], wet_t,
                    FOCUS, ACCUMULATION_BAR)
                legacy_metric = redi_metrics[name]
                legacy_error = legacy_metric["max_column_error"]
                exact_error = exact_metric["max_column_error"]
                redi_e3w_metrics[name] = {
                    "legacy_t_average": legacy_metric,
                    "live_kmm_e3w": exact_metric,
                    "max_error_removal_fraction": float(
                        (legacy_error - exact_error) / legacy_error),
                }
                if args.redi_flux_ladder:
                    short = "tem" if name == "temperature" else "sal"
                    active = jnp.asarray(capture["args"][10], dtype=bool)
                    umask, vmask, _ = gm_module.nemo_iso_face_masks(
                        capture["args"][4], capture["args"][5], active)
                    below = jnp.concatenate(
                        [active[..., 1:], jnp.zeros_like(active[..., :1])],
                        axis=-1)
                    masks = {
                        "zfu": np.asarray(umask[..., :35], dtype=bool),
                        "zfv": np.asarray(vmask[..., :35], dtype=bool),
                        "zfw": np.asarray(
                            (active * below)[..., :35], dtype=bool),
                    }
                    for flux, key in (("zfu", "zfu"), ("zfv", "zfv"),
                                      ("zfw", "zfw_kp1")):
                        oracle_flux = _load(
                            flux_dir / f"redi_dump_{flux}_{short}.bin")
                        metric = sweep.metrics(
                            np.asarray(diagnostics_replay[key])[..., :35],
                            oracle_flux, masks[flux], FOCUS, POINTWISE_BAR)
                        redi_flux_metrics[f"{flux}_{short}"] = metric
                        for control_name, control_value in _controls(
                                oracle_flux, masks[flux], POINTWISE_BAR).items():
                            controls_key = (
                                f"redi_flux_{flux}_{short}_{control_name}")
                            # Applied below after the base control dictionary is built.
                            redi_flux_metrics[f"_{controls_key}"] = control_value
                    if args.redi_zfu_operand_ladder and name == "temperature":
                        own = diagnostics_replay["zfu_operands"]
                        own = {key: jnp.asarray(value)
                               for key, value in own.items()}
                        nlev = own["e3t"].shape[-1]
                        oracle_ahtu = _load_levels(
                            flux_dir / "ldftra_dump_ahtu.bin", 36)
                        oracle_uslp = _load(
                            flux_dir / "eiv_dump_uslp.bin")
                        oracle_e3u = _load(
                            held / "fct_entry_dump_e3u.bin")

                        def pad_operand(value, current):
                            value = jnp.asarray(value, dtype=current.dtype)
                            if value.shape[-1] == nlev:
                                return value
                            if value.shape[-1] != nlev - 1:
                                raise SystemExit(
                                    "Redi zfu held operand level count changed")
                            return jnp.concatenate(
                                [value, current[..., -1:]], axis=-1)

                        oracle_ahtu = pad_operand(oracle_ahtu, own["ahtu"])
                        oracle_uslp = pad_operand(oracle_uslp, own["uslp"])
                        oracle_e3u = pad_operand(
                            oracle_e3u, own["e3u_flux"])

                        def compose(slope_oracle, thickness_oracle,
                                    ahtu_oracle, literal_association=False):
                            slope = oracle_uslp if slope_oracle else own["uslp"]
                            e3u = (oracle_e3u if thickness_oracle
                                   else own["e3u_flux"])
                            ahtu = oracle_ahtu if ahtu_oracle else own["ahtu"]
                            if literal_association:
                                ratio = own["e2u"] * (1.0 / own["e1u"])
                                wm_ip1 = jnp.roll(own["wmask"], -1, axis=1)
                                wm_kp1 = jnp.roll(own["wmask"], -1, axis=2)
                                wm_ip1_kp1 = jnp.roll(wm_ip1, -1, axis=2)
                                zmsku = 1.0 / jnp.maximum(
                                    (wm_ip1 + wm_kp1)
                                    + (wm_ip1_kp1 + own["wmask"]), 1.0)
                                zdkt_kp1 = jnp.roll(own["zdkt"], -1, axis=2)
                                avg4 = (
                                    (jnp.roll(own["zdkt"], -1, axis=1)
                                     + zdkt_kp1)
                                    + (jnp.roll(zdkt_kp1, -1, axis=1)
                                       + own["zdkt"]))
                            else:
                                ratio = own["e2u"] / own["e1u"]
                                zmsku = own["zmsku"]
                                avg4 = own["avg4_u"]
                            za11 = ratio[:, :, None] * e3u
                            za13 = (-own["e2u"][:, :, None]
                                    * slope * zmsku)
                            return ahtu * (
                                za11 * own["zdit"] + za13 * avg4)

                        oracle_zfu = _load(
                            flux_dir / "redi_dump_zfu_tem.bin")
                        wet_u = masks["zfu"]
                        arms = {}
                        arm_values = {}
                        for use_slope in (0, 1):
                            for use_thickness in (0, 1):
                                for use_ahtu in (0, 1):
                                    arm = (f"S{use_slope}H{use_thickness}"
                                           f"K{use_ahtu}")
                                    value = compose(
                                        use_slope, use_thickness, use_ahtu)
                                    arm_values[arm] = np.asarray(value)[..., :35]
                                    arms[arm] = sweep.metrics(
                                        arm_values[arm], oracle_zfu, wet_u,
                                        FOCUS, POINTWISE_BAR)
                        literal_value = np.asarray(
                            compose(1, 1, 1, literal_association=True))[..., :35]
                        literal_metric = sweep.metrics(
                            literal_value, oracle_zfu, wet_u,
                            FOCUS, POINTWISE_BAR)
                        baseline_error = arms["S0H0K0"]["max_column_error"]
                        # Once production is exact, the historical relative-
                        # removal denominator is zero. Keep the diagnostic
                        # finite by normalizing counterfactual deltas to the
                        # oracle RMS; this cannot affect the later all-pass
                        # classifier and is stamped in the ladder receipt.
                        normalizer = (baseline_error if baseline_error != 0.0
                                      else arms["S0H0K0"]["reference_rms"])
                        for metric in arms.values():
                            metric["max_error_removal_fraction"] = float(
                                (baseline_error - metric["max_column_error"])
                                / normalizer)
                        literal_metric["max_error_removal_fraction"] = float(
                            (baseline_error
                             - literal_metric["max_column_error"])
                            / normalizer)
                        errors = {key: value["max_column_error"]
                                  for key, value in arms.items()}
                        interactions = {
                            "SxH": float((errors["S1H0K0"]
                                          + errors["S0H1K0"]
                                          - errors["S1H1K0"]
                                          - errors["S0H0K0"])
                                         / normalizer),
                            "SxK": float((errors["S1H0K0"]
                                          + errors["S0H0K1"]
                                          - errors["S1H0K1"]
                                          - errors["S0H0K0"])
                                         / normalizer),
                            "HxK": float((errors["S0H1K0"]
                                          + errors["S0H0K1"]
                                          - errors["S0H1K1"]
                                          - errors["S0H0K0"])
                                         / normalizer),
                        }
                        perturb = np.array(arm_values["S0H0K0"], copy=True)
                        first_wet = tuple(
                            int(index) for index in np.argwhere(wet_u)[0])
                        perturb[first_wet] += max(
                            4.0 * POINTWISE_BAR
                            * arms["S0H0K0"]["reference_rms"],
                            4.0 * abs(float(np.spacing(perturb[first_wet]))))
                        thickness_substitution_noninert = not bool(
                            np.array_equal(
                                arm_values["S0H1K0"],
                                arm_values["S0H0K0"]))
                        ladder_controls = {
                            "production_recompose_identity": bool(
                                np.array_equal(
                                    arm_values["S0H0K0"],
                                    np.asarray(diagnostics_replay["zfu"])[..., :35])),
                            "eight_factorial_arms_present": len(arms) == 8,
                            "wet_point_plant_red": not sweep.metrics(
                                perturb, oracle_zfu, wet_u, FOCUS,
                                POINTWISE_BAR)["pass"],
                            **({"live_thickness_substitution_inert_after_fix":
                                not thickness_substitution_noninert}
                               if args.redi_zfu_postfix else
                               {"live_thickness_substitution_noninert":
                                thickness_substitution_noninert}),
                            "literal_recompose_finite": bool(
                                np.isfinite(literal_value[wet_u]).all()),
                        }
                        redi_zfu_operand_ladder = {
                            "normalization": (
                                "baseline_max_column_error"
                                if baseline_error != 0.0 else
                                "oracle_reference_rms_zero_baseline"),
                            "arms": arms,
                            "literal_all_oracle": literal_metric,
                            "interactions": interactions,
                            "operand_metrics": {
                                "final_uslp": sweep.metrics(
                                    np.asarray(own["uslp"])[..., :35],
                                    np.asarray(oracle_uslp)[..., :35], wet_u,
                                    FOCUS, POINTWISE_BAR),
                                "live_e3u": sweep.metrics(
                                    np.asarray(own["e3u_flux"])[..., :35],
                                    np.asarray(oracle_e3u)[..., :35], wet_u,
                                    FOCUS, POINTWISE_BAR),
                                "ahtu": sweep.metrics(
                                    np.asarray(own["ahtu"])[..., :35],
                                    np.asarray(oracle_ahtu)[..., :35], wet_u,
                                    FOCUS, POINTWISE_BAR),
                            },
                            "controls": ladder_controls,
                        }
                        if args.redi_zfw_association_factorial:
                            zfw = {
                                key: jnp.asarray(value)
                                for key, value in
                                diagnostics_replay["zfw_operands"].items()
                            }
                            zdit_kp1 = jnp.roll(zfw["zdit"], -1, axis=2)
                            zdjt_kp1 = jnp.roll(zfw["zdjt"], -1, axis=2)
                            wi = (
                                zfw["zdit"],
                                jnp.roll(zdit_kp1, +1, axis=1),
                                jnp.roll(zfw["zdit"], +1, axis=1),
                                zdit_kp1,
                            )
                            wj = (
                                zfw["zdjt"],
                                jnp.roll(zdjt_kp1, +1, axis=0),
                                jnp.roll(zfw["zdjt"], +1, axis=0),
                                zdjt_kp1,
                            )

                            def zfw_compose(source_gradient, source_a33):
                                def four(values):
                                    a, b, c, d = values
                                    return ((a + b) + (c + d)
                                            if source_gradient else
                                            ((a + b) + c) + d)
                                term31 = zfw["zA31"] * four(wi)
                                term32 = zfw["zA32"] * four(wj)
                                if source_a33:
                                    a33 = (((zfw["e1e2t"][..., None]
                                             / zfw["e3w_kp1"])
                                            * jnp.roll(
                                                zfw["wmask"], -1, axis=2))
                                           * (zfw["ah_wslp2"] - zfw["akz"]))
                                    a33 = a33 * zfw["qdiff_kp1"]
                                else:
                                    a33 = zfw["a33_current"]
                                return ((term31 + term32) + a33) * zfw["act_below"]

                            oracle_zfw = _load(
                                flux_dir / "redi_dump_zfw_tem.bin")
                            zfw_arms = {}
                            zfw_values = {}
                            for use_g in (0, 1):
                                for use_a in (0, 1):
                                    arm = f"G{use_g}A{use_a}"
                                    value = np.asarray(zfw_compose(
                                        use_g, use_a))[..., :35]
                                    zfw_values[arm] = value
                                    zfw_arms[arm] = sweep.metrics(
                                        value, oracle_zfw, masks["zfw"],
                                        FOCUS, POINTWISE_BAR)
                            base = zfw_arms["G0A0"]["max_column_error"]
                            interaction = float(
                                (zfw_arms["G1A0"]["max_column_error"]
                                 + zfw_arms["G0A1"]["max_column_error"]
                                 - zfw_arms["G1A1"]["max_column_error"]
                                 - base) / base)
                            redi_zfw_association_factorial = {
                                "arms": zfw_arms,
                                "interaction_GxA": interaction,
                                "controls": {
                                    "production_recompose_identity": bool(
                                        np.array_equal(
                                            zfw_values[
                                                "G1A0" if
                                                args.redi_zfw_skew_postfix
                                                else "G0A0"],
                                            np.asarray(diagnostics_replay[
                                                "zfw_kp1"])[..., :35])),
                                    "four_arms_present": len(zfw_arms) == 4,
                                    "source_arm_finite": bool(np.isfinite(
                                        zfw_values["G1A1"]
                                        [masks["zfw"]]).all()),
                                },
                            }
                            if args.redi_zfw_component_score:
                                own_components = {
                                    "skew": np.asarray(
                                        zfw["skew_current"]
                                        * zfw["act_below"])[..., :35],
                                    "a33": np.asarray(
                                        zfw["a33_current"]
                                        * zfw["act_below"])[..., :35],
                                }
                                oracle_components = {
                                    term: _load(component_dir /
                                        f"redi_dump_zfw_{term}_tem.bin")
                                    for term in ("skew", "a33")}
                                redi_zfw_component_metrics = {
                                    term: sweep.metrics(
                                        own_components[term],
                                        oracle_components[term], masks["zfw"],
                                        FOCUS, POINTWISE_BAR)
                                    for term in ("skew", "a33")}
                                redi_zfw_component_metrics["oracle_closure"] = (
                                    sweep.metrics(
                                        oracle_components["skew"]
                                        + oracle_components["a33"],
                                        oracle_zfw, masks["zfw"], FOCUS,
                                        POINTWISE_BAR))
                                if (args.redi_zfw_skew_factorial
                                        or args.redi_zfw_skew_operand_factorial):
                                    nlev = zfw["ahtu"].shape[-1]

                                    def pad_w_slope(value):
                                        value = jnp.asarray(value, dtype=dtype)
                                        if value.shape[-1] != nlev - 1:
                                            raise SystemExit(
                                                "held W-slope level count changed")
                                        return jnp.concatenate([
                                            value,
                                            jnp.zeros_like(value[..., :1])],
                                            axis=-1)

                                    oracle_ahtu = jnp.asarray(_load_levels(
                                        flux_dir / "ldftra_dump_ahtu.bin", nlev),
                                        dtype=dtype)
                                    oracle_ahtv = jnp.asarray(_load_levels(
                                        flux_dir / "ldftra_dump_ahtv.bin", nlev),
                                        dtype=dtype)
                                    oracle_wslpi = pad_w_slope(_load(
                                        flux_dir / "eiv_dump_wslpi.bin"))
                                    oracle_wslpj = pad_w_slope(_load(
                                        flux_dir / "eiv_dump_wslpj.bin"))
                                    full_umask = jnp.asarray(umask, dtype=dtype)
                                    full_vmask = jnp.asarray(vmask, dtype=dtype)
                                    full_wmask = zfw["wmask"]

                                    def pair4(a, b, c, d):
                                        return (a + b) + (c + d)

                                    ahtu_w = jnp.roll(
                                        oracle_ahtu, +1, axis=1)
                                    ahtu_kp1 = jnp.roll(
                                        oracle_ahtu, -1, axis=2)
                                    ahtu_w_kp1 = jnp.roll(
                                        ahtu_kp1, +1, axis=1)
                                    ahtv_s = jnp.roll(
                                        oracle_ahtv, +1, axis=0)
                                    ahtv_kp1 = jnp.roll(
                                        oracle_ahtv, -1, axis=2)
                                    ahtv_s_kp1 = jnp.roll(
                                        ahtv_kp1, +1, axis=0)
                                    umask_w = jnp.roll(
                                        full_umask, +1, axis=1)
                                    umask_kp1 = jnp.roll(
                                        full_umask, -1, axis=2)
                                    umask_w_kp1 = jnp.roll(
                                        umask_kp1, +1, axis=1)
                                    vmask_s = jnp.roll(
                                        full_vmask, +1, axis=0)
                                    vmask_kp1 = jnp.roll(
                                        full_vmask, -1, axis=2)
                                    vmask_s_kp1 = jnp.roll(
                                        vmask_kp1, +1, axis=0)
                                    zmsku_literal = full_wmask / jnp.maximum(
                                        pair4(full_umask, umask_w_kp1,
                                              umask_w, umask_kp1), 1.0)
                                    zmskv_literal = full_wmask / jnp.maximum(
                                        pair4(full_vmask, vmask_s_kp1,
                                              vmask_s, vmask_kp1), 1.0)
                                    zahu_literal = pair4(
                                        oracle_ahtu, ahtu_w_kp1,
                                        ahtu_w, ahtu_kp1) * zmsku_literal
                                    zahv_literal = pair4(
                                        oracle_ahtv, ahtv_s_kp1,
                                        ahtv_s, ahtv_kp1) * zmskv_literal
                                    skew_geom = ensure_geometry(
                                        capture["args"][8])
                                    zA_literal = (
                                        (-zahu_literal
                                         * skew_geom.dy_T[:, :, None]
                                         * zmsku_literal
                                         * jnp.roll(oracle_wslpi, -1, axis=2)),
                                        (-zahv_literal
                                         * skew_geom.dx_T[:, :, None]
                                         * zmskv_literal
                                         * jnp.roll(oracle_wslpj, -1, axis=2)),
                                    )

                                    def gradients(q_value):
                                        q_value = jnp.asarray(q_value, dtype=dtype)
                                        return (
                                            (jnp.roll(q_value, -1, axis=1)
                                             - q_value) * full_umask,
                                            (jnp.roll(q_value, -1, axis=0)
                                             - q_value) * full_vmask,
                                        )

                                    gradient_levels = {
                                        0: gradients(capture["q"]),
                                        1: gradients(state.T.data),
                                    }

                                    def four(values, source_grouping):
                                        a, b, c, d = values
                                        if source_grouping:
                                            return (a + b) + (c + d)
                                        return ((a + b) + c) + d

                                    def skew_arm(use_literal_coefficient,
                                                 source_grouping,
                                                 use_now_gradient):
                                        za31, za32 = (zA_literal
                                            if use_literal_coefficient else
                                            (zfw["zA31"], zfw["zA32"]))
                                        zdit_arm, zdjt_arm = gradient_levels[
                                            use_now_gradient]
                                        zdit_kp1_arm = jnp.roll(
                                            zdit_arm, -1, axis=2)
                                        zdjt_kp1_arm = jnp.roll(
                                            zdjt_arm, -1, axis=2)
                                        wi_arm = (
                                            zdit_arm,
                                            jnp.roll(zdit_kp1_arm, +1, axis=1),
                                            jnp.roll(zdit_arm, +1, axis=1),
                                            zdit_kp1_arm,
                                        )
                                        wj_arm = (
                                            zdjt_arm,
                                            jnp.roll(zdjt_kp1_arm, +1, axis=0),
                                            jnp.roll(zdjt_arm, +1, axis=0),
                                            zdjt_kp1_arm,
                                        )
                                        return ((za31 * four(
                                            wi_arm, source_grouping)
                                            + za32 * four(
                                                wj_arm, source_grouping))
                                            * zfw["act_below"])

                                    skew_arms = {}
                                    skew_values = {}
                                    for use_c in (0, 1):
                                        for use_g in (0, 1):
                                            for use_t in (0, 1):
                                                arm = f"C{use_c}G{use_g}T{use_t}"
                                                value = np.asarray(skew_arm(
                                                    use_c, use_g, use_t))[..., :35]
                                                skew_values[arm] = value
                                                skew_arms[arm] = sweep.metrics(
                                                    value,
                                                    oracle_components["skew"],
                                                    masks["zfw"], FOCUS,
                                                    POINTWISE_BAR)
                                    baseline = skew_arms["C0G0T0"][
                                        "max_column_error"]
                                    errors = {name: metric[
                                        "max_column_error"]
                                        for name, metric in skew_arms.items()}
                                    removals = {name: float(
                                        (baseline - error) / baseline)
                                        for name, error in errors.items()}
                                    interactions = {
                                        "CxG_at_T0": float((
                                            errors["C1G0T0"]
                                            + errors["C0G1T0"]
                                            - errors["C1G1T0"]
                                            - baseline) / baseline),
                                        "CxT_at_G0": float((
                                            errors["C1G0T0"]
                                            + errors["C0G0T1"]
                                            - errors["C1G0T1"]
                                            - baseline) / baseline),
                                        "GxT_at_C0": float((
                                            errors["C0G1T0"]
                                            + errors["C0G0T1"]
                                            - errors["C0G1T1"]
                                            - baseline) / baseline),
                                        "CxGxT": float((
                                            errors["C1G1T1"]
                                            - errors["C1G1T0"]
                                            - errors["C1G0T1"]
                                            - errors["C0G1T1"]
                                            + errors["C1G0T0"]
                                            + errors["C0G1T0"]
                                            + errors["C0G0T1"]
                                            - baseline) / baseline),
                                    }
                                    perturb = np.array(
                                        skew_values["C0G0T0"], copy=True)
                                    wet_index = tuple(int(x) for x in
                                        np.argwhere(masks["zfw"])[0])
                                    perturb[wet_index] += max(
                                        4.0 * POINTWISE_BAR
                                        * skew_arms["C0G0T0"]["reference_rms"],
                                        4.0 * abs(float(np.spacing(
                                            perturb[wet_index]))))
                                    redi_zfw_skew_factorial = {
                                        "arms": skew_arms,
                                        "max_error_removal_fraction": removals,
                                        "interactions": interactions,
                                        "controls": {
                                            "eight_arms_present":
                                                len(skew_arms) == 8,
                                            "source_inputs_hash_bound": True,
                                            "baseline_recomposes_production":
                                                bool(np.array_equal(
                                                    skew_values["C0G0T0"],
                                                    own_components["skew"])),
                                            "faithful_arm_finite": bool(
                                                np.isfinite(skew_values[
                                                    "C1G1T0"][masks[
                                                        "zfw"]]).all()),
                                            "wet_point_plant_red": not
                                                sweep.metrics(
                                                    perturb,
                                                    oracle_components["skew"],
                                                    masks["zfw"], FOCUS,
                                                    POINTWISE_BAR)["pass"],
                                            "now_gradient_distinct": not bool(
                                                np.array_equal(
                                                    skew_values["C1G1T0"],
                                                    skew_values["C1G1T1"])),
                                            **{f"oracle_{name}": value
                                               for name, value in _controls(
                                                   oracle_components["skew"],
                                                   masks["zfw"],
                                                   POINTWISE_BAR).items()},
                                        },
                                    }
                                    if args.redi_zfw_skew_operand_factorial:
                                        def source_zA(au, av, si, sj):
                                            au = au * full_umask
                                            av = av * full_vmask
                                            au_w = jnp.roll(au, +1, axis=1)
                                            au_kp1 = jnp.roll(au, -1, axis=2)
                                            au_w_kp1 = jnp.roll(
                                                au_kp1, +1, axis=1)
                                            av_s = jnp.roll(av, +1, axis=0)
                                            av_kp1 = jnp.roll(av, -1, axis=2)
                                            av_s_kp1 = jnp.roll(
                                                av_kp1, +1, axis=0)
                                            zahu = pair4(
                                                au, au_w_kp1,
                                                au_w, au_kp1) * zmsku_literal
                                            zahv = pair4(
                                                av, av_s_kp1,
                                                av_s, av_kp1) * zmskv_literal
                                            return (
                                                (-zahu
                                                 * skew_geom.dy_T[:, :, None]
                                                 * zmsku_literal
                                                 * jnp.roll(si, -1, axis=2)),
                                                (-zahv
                                                 * skew_geom.dx_T[:, :, None]
                                                 * zmskv_literal
                                                 * jnp.roll(sj, -1, axis=2)),
                                            )

                                        def mixed_zA(use_u, use_v,
                                                     use_i, use_j):
                                            return source_zA(
                                                oracle_ahtu if use_u else
                                                zfw["ahtu"],
                                                oracle_ahtv if use_v else
                                                zfw["ahtv"],
                                                oracle_wslpi if use_i else
                                                zfw["wslpi"],
                                                oracle_wslpj if use_j else
                                                zfw["wslpj"])

                                        zdit_before, zdjt_before = (
                                            gradient_levels[0])
                                        zdit_before_kp1 = jnp.roll(
                                            zdit_before, -1, axis=2)
                                        zdjt_before_kp1 = jnp.roll(
                                            zdjt_before, -1, axis=2)
                                        wi_before = (
                                            zdit_before,
                                            jnp.roll(
                                                zdit_before_kp1, +1, axis=1),
                                            jnp.roll(
                                                zdit_before, +1, axis=1),
                                            zdit_before_kp1,
                                        )
                                        wj_before = (
                                            zdjt_before,
                                            jnp.roll(
                                                zdjt_before_kp1, +1, axis=0),
                                            jnp.roll(
                                                zdjt_before, +1, axis=0),
                                            zdjt_before_kp1,
                                        )
                                        sum_i = four(wi_before, True)
                                        sum_j = four(wj_before, True)
                                        operand_arms = {}
                                        operand_values = {}
                                        for use_u in (0, 1):
                                            for use_v in (0, 1):
                                                for use_i in (0, 1):
                                                    for use_j in (0, 1):
                                                        arm = (f"U{use_u}V{use_v}"
                                                               f"I{use_i}J{use_j}")
                                                        za31, za32 = mixed_zA(
                                                            use_u, use_v,
                                                            use_i, use_j)
                                                        value = np.asarray(
                                                            (za31 * sum_i
                                                             + za32 * sum_j)
                                                            * zfw["act_below"]
                                                        )[..., :35]
                                                        operand_values[arm] = value
                                                        operand_arms[arm] = (
                                                            sweep.metrics(
                                                                value,
                                                                oracle_components[
                                                                    "skew"],
                                                                masks["zfw"],
                                                                FOCUS,
                                                                POINTWISE_BAR))
                                        operand_base = operand_arms[
                                            "U0V0I0J0"]["max_column_error"]
                                        operand_errors = {
                                            name: metric["max_column_error"]
                                            for name, metric in
                                            operand_arms.items()}
                                        operand_removals = {
                                            name: float((operand_base - error)
                                                        / operand_base)
                                            for name, error in
                                            operand_errors.items()}
                                        pair_interactions = {}
                                        def operand_key(enabled):
                                            return "".join(
                                                f"{name}{int(name in enabled)}"
                                                for name in "UVIJ")

                                        for left_index, left in enumerate(
                                                "UVIJ"):
                                            for right in "UVIJ"[
                                                    left_index + 1:]:
                                                e_left = operand_errors[
                                                    operand_key({left})]
                                                e_right = operand_errors[
                                                    operand_key({right})]
                                                e_pair = operand_errors[
                                                    operand_key({left, right})]
                                                pair_interactions[
                                                    f"{left}x{right}"] = float(
                                                        (e_left + e_right
                                                         - e_pair
                                                         - operand_base)
                                                        / operand_base)
                                        raw_operand_metrics = {
                                            "ahtu": sweep.metrics(
                                                np.asarray(zfw["ahtu"]
                                                    * full_umask)[..., :35],
                                                np.asarray(oracle_ahtu)[..., :35],
                                                np.asarray(full_umask[..., :35],
                                                           dtype=bool),
                                                FOCUS, POINTWISE_BAR),
                                            "ahtv": sweep.metrics(
                                                np.asarray(zfw["ahtv"]
                                                    * full_vmask)[..., :35],
                                                np.asarray(oracle_ahtv)[..., :35],
                                                np.asarray(full_vmask[..., :35],
                                                           dtype=bool),
                                                FOCUS, POINTWISE_BAR),
                                            "wslpi": sweep.metrics(
                                                np.asarray(zfw["wslpi"])[..., :35],
                                                np.asarray(oracle_wslpi)[..., :35],
                                                np.asarray(full_wmask[..., :35],
                                                           dtype=bool),
                                                FOCUS, POINTWISE_BAR),
                                            "wslpj": sweep.metrics(
                                                np.asarray(zfw["wslpj"])[..., :35],
                                                np.asarray(oracle_wslpj)[..., :35],
                                                np.asarray(full_wmask[..., :35],
                                                           dtype=bool),
                                                FOCUS, POINTWISE_BAR),
                                        }
                                        redi_zfw_skew_operand_factorial = {
                                            "arms": operand_arms,
                                            "raw_operand_metrics":
                                                raw_operand_metrics,
                                            "max_error_removal_fraction":
                                                operand_removals,
                                            "pair_interactions":
                                                pair_interactions,
                                            "controls": {
                                                "sixteen_arms_present":
                                                    len(operand_arms) == 16,
                                                "baseline_recomposes_production":
                                                    bool(np.array_equal(
                                                        operand_values[
                                                            "U0V0I0J0"],
                                                        own_components["skew"])),
                                                "all_oracle_exact":
                                                    operand_arms[
                                                        "U1V1I1J1"]["pass"],
                                                "all_oracle_finite": bool(
                                                    np.isfinite(operand_values[
                                                        "U1V1I1J1"][masks[
                                                            "zfw"]]).all()),
                                                "certified_coefficients_inert":
                                                    all(np.array_equal(
                                                        operand_values[
                                                            operand_key({name})],
                                                        operand_values[
                                                            "U0V0I0J0"])
                                                        for name in "UV")
                                                    and raw_operand_metrics[
                                                        "ahtu"]["pass"]
                                                    and raw_operand_metrics[
                                                        "ahtv"]["pass"],
                                                "slope_substitutions_noninert":
                                                    all(not np.array_equal(
                                                        operand_values[
                                                            operand_key({name})],
                                                        operand_values[
                                                            "U0V0I0J0"])
                                                        for name in "IJ"),
                                                "source_inputs_hash_bound": True,
                                                **{f"oracle_{name}": value
                                                   for name, value in _controls(
                                                       oracle_components[
                                                           "skew"],
                                                       masks["zfw"],
                                                       POINTWISE_BAR).items()},
                                            },
                                        }
                                        if args.redi_zfw_wslp_stage_factorial:
                                            stage_arms = {}
                                            stage_values = {}
                                            for use_i in (0, 1):
                                                for use_j in (0, 1):
                                                    arm = f"I{use_i}J{use_j}"
                                                    si = (zfw["bolus_wslpi"]
                                                          if use_i else
                                                          zfw["wslpi"])
                                                    sj = (zfw["bolus_wslpj"]
                                                          if use_j else
                                                          zfw["wslpj"])
                                                    za31, za32 = source_zA(
                                                        zfw["ahtu"],
                                                        zfw["ahtv"], si, sj)
                                                    value = np.asarray(
                                                        (za31 * sum_i
                                                         + za32 * sum_j)
                                                        * zfw["act_below"]
                                                    )[..., :35]
                                                    stage_values[arm] = value
                                                    stage_arms[arm] = (
                                                        sweep.metrics(
                                                            value,
                                                            oracle_components[
                                                                "skew"],
                                                            masks["zfw"], FOCUS,
                                                            POINTWISE_BAR))
                                            stage_base = stage_arms["I0J0"][
                                                "max_column_error"]
                                            stage_errors = {
                                                name: metric["max_column_error"]
                                                for name, metric in
                                                stage_arms.items()}
                                            redi_zfw_wslp_stage_factorial = {
                                                "arms": stage_arms,
                                                "max_error_removal_fraction": {
                                                    name: float(
                                                        (stage_base - error)
                                                        / stage_base)
                                                    for name, error in
                                                    stage_errors.items()},
                                                "interaction_IxJ": float((
                                                    stage_errors["I1J0"]
                                                    + stage_errors["I0J1"]
                                                    - stage_errors["I1J1"]
                                                    - stage_base) / stage_base),
                                                "raw_stage_metrics": {
                                                    "bolus_wslpi": sweep.metrics(
                                                        np.asarray(zfw[
                                                            "bolus_wslpi"])[
                                                                ..., :35],
                                                        np.asarray(
                                                            oracle_wslpi)[
                                                                ..., :35],
                                                        np.asarray(full_wmask[
                                                            ..., :35],
                                                            dtype=bool),
                                                        FOCUS, POINTWISE_BAR),
                                                    "bolus_wslpj": sweep.metrics(
                                                        np.asarray(zfw[
                                                            "bolus_wslpj"])[
                                                                ..., :35],
                                                        np.asarray(
                                                            oracle_wslpj)[
                                                                ..., :35],
                                                        np.asarray(full_wmask[
                                                            ..., :35],
                                                            dtype=bool),
                                                        FOCUS, POINTWISE_BAR),
                                                },
                                                "controls": {
                                                    "four_arms_present":
                                                        len(stage_arms) == 4,
                                                    "baseline_recomposes_production":
                                                        bool(np.array_equal(
                                                            stage_values[
                                                                "I0J0"],
                                                            own_components[
                                                                "skew"])),
                                                    "joint_arm_finite": bool(
                                                        np.isfinite(stage_values[
                                                            "I1J1"][masks[
                                                                "zfw"]]).all()),
                                                    **({
                                                        "both_stage_substitutions_inert_after_fix":
                                                            all(np.array_equal(
                                                                stage_values[
                                                                    name],
                                                                stage_values[
                                                                    "I0J0"])
                                                                for name in (
                                                                    "I1J0",
                                                                    "I0J1")),
                                                    } if args.redi_zfw_a33_floor_factorial
                                                       else {
                                                        "both_stage_substitutions_noninert":
                                                            all(not np.array_equal(
                                                                stage_values[
                                                                    name],
                                                                stage_values[
                                                                    "I0J0"])
                                                                for name in (
                                                                    "I1J0",
                                                                    "I0J1")),
                                                    }),
                                                    **{f"oracle_{name}": value
                                                       for name, value in
                                                       _controls(
                                                           oracle_components[
                                                               "skew"],
                                                           masks["zfw"],
                                                           POINTWISE_BAR).items()},
                                                },
                                            }
                                        if args.redi_zfw_a33_floor_factorial:
                                            # Round 93 holds the winning exact
                                            # NEMO raw-W -> production-Shapiro
                                            # pair and splits the remaining
                                            # traldf_iso.F90:285-332 A33
                                            # arithmetic into H and K groups.
                                            au = np.asarray(oracle_ahtu)
                                            av = np.asarray(oracle_ahtv)
                                            um = np.asarray(full_umask)
                                            vm = np.asarray(full_vmask)
                                            wm = np.asarray(full_wmask)
                                            wi_exact = np.asarray(oracle_wslpi)
                                            wj_exact = np.asarray(oracle_wslpj)
                                            up = lambda value: np.roll(
                                                value, +1, axis=2)
                                            au_m = au * um
                                            av_m = av * vm
                                            au_w = np.roll(au_m, +1, axis=1)
                                            av_s = np.roll(av_m, +1, axis=0)
                                            um_w = np.roll(um, +1, axis=1)
                                            vm_s = np.roll(vm, +1, axis=0)
                                            cnt_u = ((up(um) + um_w)
                                                     + (up(um_w) + um))
                                            cnt_v = ((up(vm) + vm_s)
                                                     + (up(vm_s) + vm))
                                            zmsku_ab = wm / np.maximum(cnt_u, 1.0)
                                            zmskv_ab = wm / np.maximum(cnt_v, 1.0)
                                            sum_u = ((up(au_m) + au_w)
                                                     + (up(au_w) + au_m))
                                            sum_v = ((up(av_m) + av_s)
                                                     + (up(av_s) + av_m))
                                            zahu_ab = sum_u * zmsku_ab
                                            zahv_ab = sum_v * zmskv_ab
                                            ah_current = (zahu_ab * wi_exact ** 2
                                                          + zahv_ab * wj_exact ** 2)
                                            ah_source = ((zahu_ab * wi_exact)
                                                         * wi_exact
                                                         + (zahv_ab * wj_exact)
                                                         * wj_exact)
                                            geom93 = ensure_geometry(
                                                capture["args"][8])
                                            e1u93 = np.asarray(
                                                geom93.dx_u[:, 1:])[:, :, None]
                                            e2v93 = np.asarray(
                                                geom93.dy_v[1:, :])[:, :, None]
                                            e1u_w = np.roll(e1u93, +1, axis=1)
                                            e2v_s = np.roll(e2v93, +1, axis=0)
                                            e3w_ab = np.roll(np.asarray(
                                                zfw["e3w_kp1"]), +1, axis=2)
                                            dt93 = float(capture["kwargs"]["dt"])

                                            def k_current(ah_value):
                                                inv_e1 = 1.0 / (e1u93 ** 2)
                                                inv_e2 = 1.0 / (e2v93 ** 2)
                                                akh = 0.25 * (
                                                    (au_m + up(au_m)) * inv_e1
                                                    + (au_w + up(au_w))
                                                    * np.roll(inv_e1, +1, axis=1)
                                                    + (av_m + up(av_m)) * inv_e2
                                                    + (av_s + up(av_s))
                                                    * np.roll(inv_e2, +1, axis=0))
                                                e3sq = e3w_ab ** 2
                                                coef = dt93 * (
                                                    akh + ah_value / e3sq)
                                                return (np.maximum(coef - 0.5, 0.0)
                                                        * e3sq / dt93)

                                            def k_source(ah_value):
                                                u0 = ((au_m + up(au_m))
                                                      / (e1u93 * e1u93))
                                                u1 = ((au_w + up(au_w))
                                                      / (e1u_w * e1u_w))
                                                v0 = ((av_m + up(av_m))
                                                      / (e2v93 * e2v93))
                                                v1 = ((av_s + up(av_s))
                                                      / (e2v_s * e2v_s))
                                                akh = ((u0 + u1) + (v0 + v1)) * 0.25
                                                e3sq = e3w_ab * e3w_ab
                                                coef = dt93 * (
                                                    akh + ah_value / e3sq)
                                                return (np.maximum(coef - 0.5, 0.0)
                                                        * e3sq * (1.0 / dt93))

                                            za31, za32 = source_zA(
                                                oracle_ahtu, oracle_ahtv,
                                                oracle_wslpi, oracle_wslpj)
                                            zdi = zfw["zdit"]
                                            zdj = zfw["zdjt"]
                                            zdi_kp1 = jnp.roll(zdi, -1, axis=2)
                                            zdj_kp1 = jnp.roll(zdj, -1, axis=2)
                                            skew93 = np.asarray((
                                                za31 * pair4(
                                                    zdi,
                                                    jnp.roll(zdi_kp1, +1, axis=1),
                                                    jnp.roll(zdi, +1, axis=1),
                                                    zdi_kp1)
                                                + za32 * pair4(
                                                    zdj,
                                                    jnp.roll(zdj_kp1, +1, axis=0),
                                                    jnp.roll(zdj, +1, axis=0),
                                                    zdj_kp1))
                                                * zfw["act_below"])[..., :35]
                                            a33_values = {}
                                            full_values = {}
                                            ah_values = {}
                                            akz_values = {}
                                            for use_h in (0, 1):
                                                for use_k in (0, 1):
                                                    arm = f"H{use_h}K{use_k}"
                                                    ah_ab = (ah_source if use_h
                                                             else ah_current)
                                                    akz_ab = (k_source(ah_ab)
                                                              if use_k else
                                                              k_current(ah_ab))
                                                    ah_flux = np.roll(
                                                        ah_ab, -1, axis=2)
                                                    akz_flux = np.roll(
                                                        akz_ab, -1, axis=2)
                                                    if use_k:
                                                        a33 = (((np.asarray(
                                                            zfw["e1e2t"])[..., None]
                                                            / np.asarray(zfw[
                                                                "e3w_kp1"]))
                                                            * np.roll(wm, -1, axis=2))
                                                            * (ah_flux - akz_flux))
                                                        a33 = a33 * np.asarray(
                                                            zfw["qdiff_kp1"])
                                                    else:
                                                        a33 = ((np.asarray(
                                                            zfw["e1e2t"])[..., None]
                                                            / np.asarray(zfw[
                                                                "e3w_kp1"]))
                                                            * (ah_flux - akz_flux)
                                                            * np.asarray(zfw[
                                                                "qdiff_kp1"]))
                                                    a33 = (a33 * np.asarray(
                                                        zfw["act_below"]))[..., :35]
                                                    ah_values[arm] = ah_flux[..., :35]
                                                    akz_values[arm] = akz_flux[..., :35]
                                                    a33_values[arm] = a33
                                                    full_values[arm] = skew93 + a33
                                            oracle_ah = ah_values["H1K1"]
                                            oracle_akz = akz_values["H1K1"]
                                            a33_arms = {}
                                            for arm in a33_values:
                                                a33_arms[arm] = {
                                                    "ah_wslp2": sweep.metrics(
                                                        ah_values[arm], oracle_ah,
                                                        masks["zfw"], FOCUS,
                                                        POINTWISE_BAR),
                                                    "akz": sweep.metrics(
                                                        akz_values[arm], oracle_akz,
                                                        masks["zfw"], FOCUS,
                                                        POINTWISE_BAR),
                                                    "a33": sweep.metrics(
                                                        a33_values[arm],
                                                        oracle_components["a33"],
                                                        masks["zfw"], FOCUS,
                                                        POINTWISE_BAR),
                                                    "skew": sweep.metrics(
                                                        skew93,
                                                        oracle_components["skew"],
                                                        masks["zfw"], FOCUS,
                                                        POINTWISE_BAR),
                                                    "full_zfw": sweep.metrics(
                                                        full_values[arm], oracle_zfw,
                                                        masks["zfw"], FOCUS,
                                                        POINTWISE_BAR),
                                                }
                                            base93 = a33_arms["H0K0"]["full_zfw"][
                                                "max_column_error"]
                                            errors93 = {arm: value["full_zfw"][
                                                "max_column_error"]
                                                for arm, value in a33_arms.items()}
                                            plant93 = np.roll(
                                                oracle_components["a33"], 1, axis=1)
                                            redi_zfw_a33_floor_factorial = {
                                                "arms": a33_arms,
                                                "max_error_removal_fraction": {
                                                    arm: float((base93 - error)
                                                               / base93)
                                                    for arm, error in
                                                    errors93.items()},
                                                "interaction_HxK": float((
                                                    errors93["H1K0"]
                                                    + errors93["H0K1"]
                                                    - errors93["H1K1"]
                                                    - base93) / base93),
                                                "controls": {
                                                    "wslp_winner_admitted": True,
                                                    "four_arms_present":
                                                        len(a33_arms) == 4,
                                                    "derived_oracle_ah_identity":
                                                        a33_arms["H1K1"][
                                                            "ah_wslp2"]["pass"],
                                                    "derived_oracle_akz_identity":
                                                        a33_arms["H1K1"]["akz"][
                                                            "pass"],
                                                    "oracle_component_closure":
                                                        redi_zfw_component_metrics[
                                                            "oracle_closure"]["pass"],
                                                    "zonal_roll_plant_red": not
                                                        sweep.metrics(
                                                            plant93,
                                                            oracle_components["a33"],
                                                            masks["zfw"], FOCUS,
                                                            POINTWISE_BAR)["pass"],
                                                },
                                            }
    specs = (
        ("8.3", "uu(Kmm) / zptu", "traadv.F90:301-304", "un", POINTWISE_BAR),
        ("8.4", "e2u", "traadv.F90:329", "e2u", POINTWISE_BAR),
        ("8.5", "e3u(Kmm)", "traadv.F90:329", "e3u", POINTWISE_BAR),
        ("8.6", "e2u*e3u(Kmm)", "traadv.F90:329", "e2e3u", POINTWISE_BAR),
        ("8.7", "Eulerian pU", "traadv.F90:328-331", "pu_euler", ACCUMULATION_BAR),
        ("8.8", "GM U increment", "ldftra.F90:894-902", "pu_bolus", ACCUMULATION_BAR),
        ("8.9", "total pU entering FCT", "traadv.F90:343-362", "pu_total", ACCUMULATION_BAR),
        ("8.10", "temperature U upstream flux", "traadv_fct.F90:528-530", "upstream", ACCUMULATION_BAR),
    )
    rows = []
    blocked = False
    first = None
    controls = {
        "hook_restored": restored,
        "solver_substitution_count": (
            solver_calls == 1 if args.hold_slow_forcing else solver_calls == 0),
        "transport_substitution_count": (
            transport_substitutions == 1 if args.oracle_transport
            else transport_substitutions == 0),
        "round59_production_debt_admitted": (
            prior59 is None
            or prior59["rows"][0]["metrics"]["n_diverged_columns"] == 104),
        "round60_literal_local_exact": (
            prior60 is None
            or prior60["metrics"]["raw_metric"]["pass"]),
        "round61_null_admitted": (
            prior61 is None
            or prior61["rows"][0]["metrics"]["n_diverged_columns"] == 104),
        "round62_direct_cycle_exact": (
            prior62 is None
            or prior62["cycle_capture_metrics"]["cycle_corrected_u"]["pass"]),
        "round63_thickness_debt_admitted": (
            prior63 is None
            or prior63["rows"][2]["metrics"]["n_diverged_columns"] == 9758),
        "cycle_capture_count": (
            len(cycle_captures) >= 1 if args.capture_cycle
            else len(cycle_captures) == 0),
        "solver_transport_capture_count": (
            len(solver_transport_captures) == 1 if args.capture_cycle
            else len(solver_transport_captures) == 0),
        "consumed_transport_at_bar": (
            capture_metrics["consumed_Hu_avg"]["pass"]
            if args.capture_cycle else True),
        "mass_flux_division_proxy_red_104": (
            prior63["controls"]["mass_flux_division_proxy_red_104"]
            if args.live_thickness_entry else
            sweep.metrics(proxy_un, oracle["un"], wet, FOCUS,
                          POINTWISE_BAR)["n_diverged_columns"] == 104
            if args.direct_cycle_entry else True),
        "live_thickness_capture_count": (
            len(thickness_captures) == 1 if args.live_thickness_entry
            else len(thickness_captures) == 0),
        "generic_min_thickness_plant_red_9758": (
            sweep.metrics(h_u[:, 1:, :35], oracle["e3u"], wet, FOCUS,
                          POINTWISE_BAR)["n_diverged_columns"] == 9758
            if args.live_thickness_entry else True),
        "round64_row8_8_debt_admitted": (
            prior64 is None
            or prior64["rows"][5]["metrics"]["n_diverged_columns"] == 8938),
        "bolus_capture_count": (
            len(bolus_captures) >= 1 if args.capture_bolus_operands
            else len(bolus_captures) == 0),
        "kappa_capture_count": (
            len(kappa_captures) == 1 if args.capture_kappa_operands
            else len(kappa_captures) == 0),
        "round65_aeiu_debt_admitted": (
            prior65 is None
            or not prior65["bolus_operand_metrics"]["aeiu_face"]["pass"]),
        "round66_reduction_debt_admitted": (
            prior66 is None
            or not prior66["kappa_operand_metrics"]["zn"]["pass"]),
        "round67_reduction_null_admitted": (
            prior67 is None
            or prior67["kappa_operand_metrics"]["zhw"]["max_column_error"]
            == prior66["kappa_operand_metrics"]["zhw"]["max_column_error"]),
        "round68_coupled_debt_admitted": (
            prior68 is None
            or (not prior68["kappa_geometry_metrics"]["e3w_Kmm"]["pass"]
                and not prior68["kappa_geometry_metrics"]["rn2b"]["pass"])),
        "round69_surface_debt_admitted": (
            prior69 is None
            or (not prior69["kappa_geometry_metrics"]["e3w_Kmm"]["pass"]
                and prior69["kappa_geometry_metrics"]["rn2b"]["pass"])),
        "round70_e3t_surface_proxy_plant_red": (
            prior70 is None
            or (not prior70["kappa_geometry_metrics"]["e3w_Kmm"]["pass"]
                and prior70["rows"][5]["metrics"]["max_column_error"]
                >= 8.0e-4)),
        "round71_exact_geometry_debt_admitted": (
            prior71 is None
            or (all(prior71["kappa_geometry_metrics"][name]["pass"]
                    for name in ("e3w_Kmm", "rn2b"))
                and not prior71["kappa_operand_metrics"]["zaeiw"]["pass"])),
        "round72_zro_debt_admitted": (
            prior72 is None
            or (all(prior72["post_chain_metrics"]["literal_oracle_zRo"]
                           [name]["pass"]
                    for name in ("zaeiw", "aeiu", "row8_8"))
                and not prior72["post_chain_metrics"]["literal_own"]
                               ["row8_8"]["pass"])),
        "round73_zn_debt_admitted": (
            prior73 is None
            or (prior73["rossby_metrics"]["literal_oracle_zn"]
                      ["row8_8"]["pass"]
                and not prior73["rossby_metrics"]["literal_stored_f"]
                               ["row8_8"]["pass"])),
        "round74_sqrt_owner_admitted": (
            prior74 is None
            or (all(prior74["zn_sqrt_metrics"]["exact_forward"]
                           [name]["pass"] for name in
                           ("zn", "zRo", "zaeiw", "aeiu", "row8_8"))
                and not prior74["zn_sqrt_metrics"]["guarded_floor"]
                               ["row8_8"]["pass"])),
        "round75_tracer_entry_admitted": (
            prior75 is None
            or (prior75["first_diverged_subrow"] is None
                and all(row["status"] == "AT_BAR"
                        for row in prior75["rows"]))),
        "round56_unheld_red": (
            prior56 is None
            or prior56["rows"][0]["status"] == "DIVERGED"),
    }
    for subrow, name, source, key, bar in specs:
        row = {"subrow": subrow, "name": name, "nemo_source": source, "bar": bar}
        if blocked:
            row["status"] = "ORDERED_BLOCKED"
        else:
            metric = sweep.metrics(values[key], oracle[key], wet, FOCUS, bar)
            row.update(status="AT_BAR" if metric["pass"] else "DIVERGED",
                       metrics=metric)
            row_controls = _controls(oracle[key], wet, bar)
            if subrow == "8.4":
                # Mercator e2u is exactly zonally uniform on the registered
                # wet population, so a zonal roll cannot be a red-capable
                # plant.  Preserve that geometry fact as a positive receipt;
                # wet-point, meridional-roll, and sign plants remain red.
                controls["8.4_zonal_roll_structurally_inert"] = not (
                    row_controls.pop("zonal_roll"))
            controls.update({f"{subrow}_{k}": v
                             for k, v in row_controls.items()})
            if not metric["pass"]:
                first = subrow
                blocked = True
        rows.append(row)
    controls["gm_sign_plant"] = not sweep.metrics(
        -values["pu_bolus"], oracle["pu_bolus"], wet, FOCUS,
        ACCUMULATION_BAR)["pass"]
    if args.capture_bolus_operands:
        controls.update({
            "bolus_current_normalized_or_guarded_plant_red": (
                not zn_sqrt_metrics["guarded_floor"]["row8_8"]["pass"]
                if args.exact_sqrt_production else
                not bolus_operand_metrics["psi_current_normalized"]["pass"]),
            "bolus_sign_plant": not sweep.metrics(
                -literal_psi[..., :35], oracle_psi, wet_psi, FOCUS,
                ACCUMULATION_BAR)["pass"],
            "bolus_meridional_roll_plant": not sweep.metrics(
                np.roll(literal_psi[..., :35], 1, axis=0), oracle_psi,
                wet_psi, FOCUS, ACCUMULATION_BAR)["pass"],
        })
    if args.capture_kappa_operands:
        oracle_zaeiw = _load2(held / "eiv_dump_zaeiw.bin")
        perturbed_zaeiw = np.array(oracle_zaeiw, copy=True)
        first_wet = tuple(int(x) for x in np.argwhere(wet2)[0])
        perturb = max(
            4.0 * ACCUMULATION_BAR
            * kappa_operand_metrics["zaeiw"]["reference_rms"],
            4.0 * abs(float(np.spacing(perturbed_zaeiw[first_wet]))))
        perturbed_zaeiw[first_wet] += perturb
        controls.update({
            "kappa_diagnostic_replay_identity":
                np.array_equal(kappa_result, kappa_replay),
            "oracle_zaeiw_face_closure":
                kappa_operand_metrics["oracle_zaeiw_to_aeiu"]["pass"],
            "zaeiw_wet_point_plant": not km(
                perturbed_zaeiw, oracle_zaeiw)["pass"],
            "zaeiw_zonal_roll_plant": not km(
                np.roll(oracle_zaeiw, 1, axis=1), oracle_zaeiw)["pass"],
        })
    if args.capture_kappa_geometry:
        controls.update({
            "kappa_geometry_capture_present": bool(raw_kappa),
            "e3w_level_roll_plant": not sweep.metrics(
                np.roll(raw_e3w[..., :35], 1, axis=2), oracle_e3w,
                wet_w, FOCUS, ACCUMULATION_BAR)["pass"],
            "rn2b_level_roll_plant": not sweep.metrics(
                np.roll(raw_kappa["pn2"][..., :35], 1, axis=2), oracle_pn2,
                wet_w, FOCUS, ACCUMULATION_BAR)["pass"],
        })
    if args.post_chain_factorial:
        controls.update({
            "post_chain_production_or_guarded_zaeiw_red": (
                not zn_sqrt_metrics["guarded_floor"]["zaeiw"]["pass"]
                if args.exact_sqrt_production else
                not post_chain_metrics["production"]["zaeiw"]["pass"]),
            "post_chain_production_or_guarded_row8_8_red": (
                not zn_sqrt_metrics["guarded_floor"]["row8_8"]["pass"]
                if args.exact_sqrt_production else
                not post_chain_metrics["production"]["row8_8"]["pass"]),
        })
    if args.rossby_factorial:
        controls.update({
            "rossby_latitude_is_radians":
                float(np.max(np.abs(raw_kappa["lat_t"]))) <= np.pi / 2.0,
            "rossby_production_or_guarded_downstream_red": (
                not zn_sqrt_metrics["guarded_floor"]["row8_8"]["pass"]
                if args.exact_sqrt_production else
                not rossby_metrics["production"]["row8_8"]["pass"]),
            "rossby_direct_oracle_downstream_pass":
                rossby_metrics["direct_oracle_zRo"]["row8_8"]["pass"],
        })
    if args.zn_sqrt_factorial:
        controls.update({
            "zn_sqrt_guarded_downstream_red":
                not zn_sqrt_metrics["guarded_floor"]["row8_8"]["pass"],
            "zn_sqrt_zero_slots_present":
                zn_sqrt_metrics["operand_receipts"]["n_zero_n2_slots"] > 0,
            "zn_sqrt_negative_slots_present":
                zn_sqrt_metrics["operand_receipts"]["n_negative_n2_slots"] > 0,
            "zn_sqrt_forward_arms_distinguished":
                zn_sqrt_metrics["operand_receipts"]
                               ["exact_differs_from_guarded"],
            "zn_sqrt_negative_forward_clipped_zero":
                zn_sqrt_metrics["operand_receipts"]
                               ["negative_forward_clipped_zero"],
        })
    if args.capture_redi_tail:
        for name in ("temperature", "salinity"):
            controls.update({
                f"redi_{name}_{key}": value for key, value in
                _controls(redi_oracle[name], wet_t, ACCUMULATION_BAR).items()
            })
            controls[f"redi_{name}_sign_plant"] = not sweep.metrics(
                -redi_oracle[name], redi_oracle[name], wet_t,
                FOCUS, ACCUMULATION_BAR)["pass"]
    if args.redi_e3w_factorial:
        controls.update({
            "round76_temperature_debt_admitted": (
                prior76["redi_metrics"]["temperature"]
                       ["n_diverged_columns"] == 9920),
            "redi_e3w_two_tracer_arms_complete": (
                set(redi_e3w_metrics) == {"temperature", "salinity"}),
            "redi_e3w_substitution_noninert": any(
                arm["max_error_removal_fraction"] != 0.0
                for arm in redi_e3w_metrics.values()),
        })
    if args.redi_flux_ladder:
        flux_control_keys = [
            key for key in redi_flux_metrics if key.startswith("_redi_flux_")]
        for key in flux_control_keys:
            controls[key[1:]] = bool(redi_flux_metrics.pop(key))
        controls.update({
            "round77_majority_admitted": (
                prior77["disposition"] == "REDI_MSC_E3W_MAJORITY"),
            "redi_flux_six_operands_scored": (
                set(redi_flux_metrics) == {
                    f"{flux}_{tracer}"
                    for flux in ("zfu", "zfv", "zfw")
                    for tracer in ("tem", "sal")}),
        })
    if args.redi_zfu_operand_ladder:
        controls.update({
            "round78_zfu_temperature_admitted": (
                prior78["disposition"] == "REDI_DIVERGED_ZFU_T"),
            **{f"redi_zfu_{name}": value for name, value in
               redi_zfu_operand_ladder["controls"].items()},
        })
    if args.redi_zfw_association_factorial:
        controls.update({
            "round85_zfw_temperature_admitted": (
                prior85["redi_flux_metrics"]["zfw_tem"]["pass"] is False),
            **{f"redi_zfw_{name}": value for name, value in
               redi_zfw_association_factorial["controls"].items()},
        })
    if args.redi_zfw_component_score:
        controls.update({
            "round86_components_released": (
                json.loads(args.round86.read_text()).get("disposition")
                == "REDI_ZFW_T_COMPONENTS_HELD_OPEN"),
            "zfw_oracle_component_closure":
                redi_zfw_component_metrics["oracle_closure"]["pass"],
        })
    if args.redi_zfw_skew_factorial:
        controls.update({
            "round87_ordered_skew_admitted": (
                prior87["disposition"] ==
                "REDI_ZFW_T_DIVERGED_A31_A32"),
            **{f"redi_zfw_skew_{name}": value for name, value in
               redi_zfw_skew_factorial["controls"].items()},
        })
    if args.redi_zfw_skew_postfix:
        controls.update({
            "round88_literal_skew_admitted": (
                prior88["disposition"] ==
                "REDI_ZFW_T_SKEW_LOCALIZED_TO_COEFFICIENT_ASSEMBLY"),
            "production_literal_skew_selector_reached": (
                matched_calls["temperature"]["kwargs"].get(
                    "vertical_skew_evaluation") == "nemo_literal"),
        })
    if args.redi_zfw_skew_operand_factorial:
        controls.update({
            "round89_postfix_regression_admitted": (
                prior89["disposition"] ==
                "REDI_ZFW_T_SKEW_POSTFIX_REGRESSION"),
            **{f"redi_zfw_skew_operand_{name}": value
               for name, value in
               redi_zfw_skew_operand_factorial["controls"].items()},
        })
    if args.redi_zfw_wslp_stage_factorial:
        controls.update({
            "round90_wslp_pair_admitted": (
                prior90["disposition"] ==
                "REDI_ZFW_T_SKEW_OWNED_WSLPI_WSLPJ"),
            **{f"redi_zfw_wslp_stage_{name}": value
               for name, value in
               redi_zfw_wslp_stage_factorial["controls"].items()},
        })
    if args.redi_zfw_a33_floor_factorial:
        controls.update({
            "round93_wslp_at_bar_admitted": (
                prior93_wslp["disposition"] ==
                "WSLOPE_ASSOCIATION_AT_BAR_R1S0"),
            **{f"redi_zfw_a33_floor_{name}": value
               for name, value in
               redi_zfw_a33_floor_factorial["controls"].items()},
        })
    controls["all_scored_finite"] = all(
        row.get("status") == "ORDERED_BLOCKED"
        or row["metrics"]["n_nonfinite_wet_elements"] == 0 for row in rows)
    controls["registered_population_exact"] = (
        int(np.any(wet, axis=-1).sum()) == 9758 and int(wet.sum()) == 336338)
    valid = all(controls.values())
    disposition = (("TRACER_ENTRY_ROW8_AT_BAR_LIVE_QCO"
                    if args.live_thickness_entry else
                    "TRACER_ENTRY_ROW8_AT_BAR_DIRECT_KMM"
                    if args.direct_cycle_entry else
                    "TRACER_ENTRY_ROW8_AT_BAR_ORACLE_TRANSPORT"
                    if args.oracle_transport else
                    "TRACER_ENTRY_ROW8_AT_BAR_UPSTREAM_FORCING_EXACT"
                    if args.hold_slow_forcing else "TRACER_ENTRY_ROW8_AT_BAR")
                   if valid and first is None
                   else "INVALID" if not valid else f"TRACER_ENTRY_DIVERGED_{first}")
    if args.capture_cycle and valid and not args.direct_cycle_entry:
        production_at_bar = capture_metrics["production_Hu_avg"]["pass"]
        consumed_at_bar = capture_metrics["consumed_Hu_avg"]["pass"]
        cycle_at_bar = capture_metrics["cycle_corrected_u"]["pass"]
        if production_at_bar and consumed_at_bar and not cycle_at_bar:
            disposition = "ROW8_3_ACCUMULATOR_EXONERATED_KMM_COMPOSITION_OPEN"
        elif not consumed_at_bar:
            disposition = "INVALID_ORACLE_TRANSPORT_LAYOUT"
        elif not production_at_bar and not cycle_at_bar:
            disposition = "INVALID_TRANSPORT_SUBSTITUTION_PLUMBING"
        else:
            disposition = "ROW8_3_CYCLE_CAPTURE_UNRESOLVED"
    if args.capture_bolus_operands and valid:
        aeiu_at_bar = bolus_operand_metrics["aeiu_face"]["pass"]
        slope_at_bar = bolus_operand_metrics["wslpi_kp1_face_sum"]["pass"]
        literal_at_bar = bolus_operand_metrics["psi_literal_same_operands"]["pass"]
        current_red = not bolus_operand_metrics["psi_current_normalized"]["pass"]
        if aeiu_at_bar and slope_at_bar and literal_at_bar and current_red:
            disposition = "ROW8_8_LOCALIZED_TO_PSI_ASSOCIATION"
        elif not slope_at_bar:
            disposition = "ROW8_8_LOCALIZED_TO_WSPLPI_OPERAND"
        elif not aeiu_at_bar:
            disposition = "ROW8_8_LOCALIZED_TO_AEIU_OPERAND"
        else:
            disposition = "ROW8_8_PSI_COMPOSITION_OPEN"
    if args.capture_kappa_operands and valid:
        ladder = ("zn", "zah", "zhw", "zRo", "zaeiw", "aeiu_face")
        first_kappa = next(
            (name for name in ladder
             if not kappa_operand_metrics[name]["pass"]), None)
        disposition = (("TRACER_ENTRY_ROW8_AT_BAR_LITERAL_GM"
                        if first is None else
                        f"ROW8_8_GM_COEFFICIENT_AT_BAR_DOWNSTREAM_{first}")
                       if first_kappa is None else
                       f"ROW8_8_GM_COEFFICIENT_DIVERGED_{first_kappa.upper()}")
    if args.capture_kappa_geometry and valid:
        e3w_pass = kappa_geometry_metrics["e3w_Kmm"]["pass"]
        rn2_pass = kappa_geometry_metrics["rn2b"]["pass"]
        disposition = (("TRACER_ENTRY_ROW8_AT_BAR_EXACT_SURFACE_KMM"
                        if args.exact_surface_kmm_carry and first is None
                        and all(metric["pass"] for metric in
                                kappa_operand_metrics.values()) else
                        "TRACER_ENTRY_ROW8_AT_BAR_CARRIED_KMM"
                        if args.surface_kmm_carry and first is None
                        and all(metric["pass"] for metric in
                                kappa_operand_metrics.values()) else
                        "ROW8_8_KAPPA_GEOMETRY_AT_BAR")
                       if e3w_pass and rn2_pass else
                       "ROW8_8_LOCALIZED_TO_KMM_E3W"
                       if not e3w_pass and rn2_pass else
                       "ROW8_8_LOCALIZED_TO_RN2B"
                       if not rn2_pass else "ROW8_8_KAPPA_GEOMETRY_AT_BAR")
    if args.post_chain_factorial and valid:
        def arm_pass(name):
            return all(metric["pass"]
                       for metric in post_chain_metrics[name].values())

        if not arm_pass("literal_all_oracle"):
            disposition = "INVALID_LITERAL_REPLAY"
        elif arm_pass("literal_own"):
            disposition = "ROW8_8_LOCALIZED_TO_POST_CHAIN_ASSOCIATION"
        elif arm_pass("literal_oracle_zRo"):
            disposition = "ROW8_8_LOCALIZED_TO_ZRO_PRECURSOR"
        elif arm_pass("literal_oracle_zah_zhw"):
            disposition = "ROW8_8_LOCALIZED_TO_ZAH_ZHW_PRECURSOR"
        else:
            disposition = "ROW8_8_POST_CHAIN_COMPOSITION_OPEN"
    if args.rossby_factorial and valid:
        def rossby_arm_pass(name):
            return all(metric["pass"]
                       for metric in rossby_metrics[name].values())

        if not rossby_arm_pass("literal_oracle_zn_recomputed_f"):
            disposition = "INVALID_ROSSBY_LITERAL_REPLAY"
        elif rossby_arm_pass("literal_recomputed_f"):
            disposition = "ROW8_8_LOCALIZED_TO_ZFW_COMPOSITION"
        elif rossby_arm_pass("literal_oracle_zn"):
            disposition = "ROW8_8_LOCALIZED_TO_ZN_PRECURSOR"
        else:
            disposition = "ROW8_8_LOCALIZED_TO_ZN_X_ZFW_COMPOSITION"
    if args.zn_sqrt_factorial and valid:
        exact_pass = all(metric["pass"] for metric in
                         zn_sqrt_metrics["exact_forward"].values())
        disposition = ("ROW8_8_LOCALIZED_TO_ZN_SQRT_FORWARD_FLOOR"
                       if exact_pass else
                       "ROW8_8_ZN_SQRT_FORWARD_OPEN")
    if args.exact_sqrt_production and valid:
        production_operands_pass = all(
            metric["pass"] for metric in
            zn_sqrt_metrics["production"].values())
        disposition = ("TRACER_ENTRY_ROW8_AT_BAR_EXACT_GM_SQRT"
                       if production_operands_pass and first is None else
                       f"TRACER_ENTRY_DIVERGED_{first}"
                       if first is not None else
                       "ROW8_8_EXACT_SQRT_PRODUCTION_OPEN")
    if args.capture_redi_tail and valid:
        if not redi_metrics["temperature"]["pass"]:
            disposition = "TRACER_TAIL_DIVERGED_REDI_T"
        elif not redi_metrics["salinity"]["pass"]:
            disposition = "TRACER_TAIL_DIVERGED_REDI_S"
        else:
            disposition = "TRACER_TAIL_REDI_AT_BAR"
    if args.redi_e3w_factorial and valid:
        temp = redi_e3w_metrics["temperature"]
        salt = redi_e3w_metrics["salinity"]
        exact_pass = (temp["live_kmm_e3w"]["pass"]
                      and salt["live_kmm_e3w"]["pass"])
        owned = (exact_pass
                 and temp["max_error_removal_fraction"] >= 0.99
                 and salt["max_error_removal_fraction"] >= 0.99)
        improved = (temp["max_error_removal_fraction"] >= 0.50
                    and salt["max_error_removal_fraction"] >= 0.0)
        disposition = ("REDI_MSC_E3W_OWNED" if owned else
                       "REDI_MSC_E3W_MAJORITY" if improved else
                       "REDI_MSC_E3W_REFUTED")
    if args.redi_flux_ladder:
        flux_order = (
            ("zfu_tem", "78.T.1", "temperature zfu",
             "traldf_iso_scheme.h90:55-70"),
            ("zfv_tem", "78.T.2", "temperature zfv",
             "traldf_iso_scheme.h90:72-90"),
            ("zfw_tem", "78.T.3", "temperature zfw_kp1",
             "traldf_iso_scheme.h90:104-129"),
            ("zfu_sal", "78.S.1", "salinity zfu",
             "traldf_iso_scheme.h90:55-70"),
            ("zfv_sal", "78.S.2", "salinity zfv",
             "traldf_iso_scheme.h90:72-90"),
            ("zfw_sal", "78.S.3", "salinity zfw_kp1",
             "traldf_iso_scheme.h90:104-129"),
        )
        flux_blocked = False
        for operand, subrow, name, source in flux_order:
            row = {"subrow": subrow, "name": name,
                   "nemo_source": source, "bar": POINTWISE_BAR}
            if flux_blocked:
                row["status"] = "ORDERED_BLOCKED"
            else:
                metric = redi_flux_metrics[operand]
                row.update(status="AT_BAR" if metric["pass"] else "DIVERGED",
                           metrics=metric)
                if not metric["pass"]:
                    first_flux = operand
                    first_flux_subrow = subrow
                    flux_blocked = True
            redi_flux_rows.append(row)
        if valid:
            if first_flux is not None:
                flux, tracer = first_flux.split("_")
                disposition = (
                    f"REDI_DIVERGED_{flux.upper()}_"
                    f"{'T' if tracer == 'tem' else 'S'}")
            elif not redi_e3w_metrics["temperature"]["live_kmm_e3w"]["pass"]:
                disposition = "REDI_DIVERGED_DIVERGENCE_VOLUME_T"
            else:
                disposition = "TRACER_TAIL_REDI_AT_BAR_LIVE_E3W"
    if args.redi_zfu_operand_ladder and valid:
        arms = redi_zfu_operand_ladder["arms"]
        literal = redi_zfu_operand_ladder["literal_all_oracle"]
        slope_removal = arms["S1H0K0"]["max_error_removal_fraction"]
        thickness_removal = arms["S0H1K0"]["max_error_removal_fraction"]
        ahtu_removal = arms["S0H0K1"]["max_error_removal_fraction"]
        full_current = arms["S1H1K1"]
        if (thickness_removal >= 0.90 and slope_removal < 0.50
                and ahtu_removal < 0.50 and literal["pass"]):
            disposition = "REDI_ZFU_T_LOCALIZED_TO_LIVE_E3U"
        elif (slope_removal >= 0.90 and thickness_removal < 0.50
              and ahtu_removal < 0.50 and literal["pass"]):
            disposition = "REDI_ZFU_T_LOCALIZED_TO_FINAL_USLP"
        elif (ahtu_removal >= 0.90 and slope_removal < 0.50
              and thickness_removal < 0.50 and literal["pass"]):
            disposition = "REDI_ZFU_T_LOCALIZED_TO_AHTU"
        elif (not full_current["pass"] and literal["pass"]
              and (full_current["max_column_error"]
                   - literal["max_column_error"])
              / full_current["max_column_error"] >= 0.90):
            disposition = "REDI_ZFU_T_LOCALIZED_TO_ASSOCIATION"
        elif (max(slope_removal, thickness_removal, ahtu_removal) < 0.90
              and literal["pass"]):
            disposition = "REDI_ZFU_T_OPERAND_COMPOSITION"
        elif (full_current["max_error_removal_fraction"] >= 0.90
              and not literal["pass"]):
            disposition = "REDI_ZFU_T_OPERANDS_BOUNDED_ASSOCIATION_OPEN"
        else:
            disposition = "REDI_ZFU_T_OPERAND_LADDER_OPEN"
        first_flux_subrow = "79.T.1"
    if args.redi_zfu_postfix and valid:
        current = redi_zfu_operand_ladder["arms"]["S0H0K0"]
        e3u = redi_zfu_operand_ladder["operand_metrics"]["live_e3u"]
        full = redi_zfu_operand_ladder["arms"]["S1H1K1"]
        if current["pass"]:
            disposition = "REDI_ZFU_T_POSTFIX_AT_BAR"
        elif (e3u["pass"] and full["pass"]
              and current["max_column_error"] < 1.0e-5):
            disposition = "REDI_ZFU_T_POSTFIX_RESIDUAL_FINAL_USLP"
        else:
            disposition = "REDI_ZFU_T_POSTFIX_REGRESSION"
        first_flux_subrow = "80.T.1"
    if args.redi_zfu_kmm_postfix and valid:
        current = redi_zfu_operand_ladder["arms"]["S0H0K0"]
        e3u = redi_zfu_operand_ladder["operand_metrics"]["live_e3u"]
        full = redi_zfu_operand_ladder["arms"]["S1H1K1"]
        if current["pass"]:
            disposition = "REDI_ZFU_T_KMM_FACE_FIXED_AT_BAR"
        elif (e3u["pass"] and full["pass"]
              and current["max_column_error"] < 1.0e-5):
            disposition = (
                "REDI_ZFU_T_KMM_FACE_FIXED_RESIDUAL_FINAL_USLP")
        else:
            disposition = "REDI_ZFU_T_KMM_FACE_FIXED_REGRESSION"
        first_flux_subrow = "81.T.1"
    if args.redi_zfu_slope_kmm_postfix and valid:
        current = redi_zfu_operand_ladder["arms"]["S0H0K0"]
        e3u = redi_zfu_operand_ladder["operand_metrics"]["live_e3u"]
        uslp = redi_zfu_operand_ladder["operand_metrics"]["final_uslp"]
        if current["pass"] and e3u["pass"] and uslp["pass"]:
            disposition = "REDI_ZFU_T_KMM_SLOPE_AT_BAR"
        else:
            disposition = "REDI_ZFU_T_KMM_SLOPE_REGRESSION"
        first_flux_subrow = "82.T.1"
    if args.redi_zfu_kmm_operator_postfix and valid:
        current = redi_zfu_operand_ladder["arms"]["S0H0K0"]
        uslp = redi_zfu_operand_ladder["operand_metrics"]["final_uslp"]
        ahtu = redi_zfu_operand_ladder["arms"]["S0H0K1"]
        if current["pass"] and uslp["pass"]:
            disposition = "REDI_ZFU_T_KMM_OPERATOR_AT_BAR"
        elif (uslp["pass"] and ahtu["pass"]
              and current["max_column_error"] < 1.0e-12):
            disposition = "REDI_ZFU_T_KMM_OPERATOR_FIXED_AHTU_RESIDUAL"
        else:
            disposition = "REDI_ZFU_T_KMM_OPERATOR_REGRESSION"
        first_flux_subrow = "83.T.1"
    if args.redi_zfu_bolus_stage_split_postfix and valid:
        current = redi_zfu_operand_ladder["arms"]["S0H0K0"]
        uslp = redi_zfu_operand_ladder["operand_metrics"]["final_uslp"]
        ahtu = redi_zfu_operand_ladder["arms"]["S0H0K1"]
        if current["pass"] and uslp["pass"]:
            disposition = "REDI_ZFU_T_BOLUS_STAGE_SPLIT_AT_BAR"
        elif (uslp["pass"] and ahtu["pass"]
              and current["max_column_error"] < 1.0e-12):
            disposition = (
                "REDI_ZFU_T_BOLUS_STAGE_SPLIT_FIXED_AHTU_RESIDUAL")
        else:
            disposition = "REDI_ZFU_T_BOLUS_STAGE_SPLIT_REGRESSION"
        first_flux_subrow = "84.T.1"
    if args.redi_zfu_ahtu_postfix and valid:
        current = redi_zfu_operand_ladder["arms"]["S0H0K0"]
        ahtu = redi_zfu_operand_ladder["operand_metrics"]["ahtu"]
        if current["pass"] and ahtu["pass"]:
            disposition = "REDI_ZFU_T_AHTU_AT_BAR"
        else:
            disposition = "REDI_ZFU_T_AHTU_REGRESSION"
        first_flux_subrow = "85.T.1"
    if args.redi_zfw_association_factorial and valid:
        arms = redi_zfw_association_factorial["arms"]
        if arms["G1A0"]["pass"] and not arms["G0A1"]["pass"]:
            disposition = "REDI_ZFW_T_GRADIENT_ASSOCIATION_OWNED"
        elif arms["G0A1"]["pass"] and not arms["G1A0"]["pass"]:
            disposition = "REDI_ZFW_T_A33_ASSOCIATION_OWNED"
        elif arms["G1A1"]["pass"]:
            disposition = "REDI_ZFW_T_ASSOCIATION_COMPOSITION_OWNED"
        else:
            disposition = "REDI_ZFW_T_COMPONENTS_HELD_OPEN"
        first_flux_subrow = "86.T.3"
    if args.redi_zfw_component_score and valid:
        if not redi_zfw_component_metrics["skew"]["pass"]:
            disposition = "REDI_ZFW_T_DIVERGED_A31_A32"
            first_flux_subrow = "87.T.3a"
        elif not redi_zfw_component_metrics["a33"]["pass"]:
            disposition = "REDI_ZFW_T_DIVERGED_A33"
            first_flux_subrow = "87.T.3b"
        else:
            disposition = "REDI_ZFW_T_COMPONENTS_AT_BAR"
            first_flux_subrow = "87.T.3c"
    if args.redi_zfw_skew_factorial and valid:
        arms = redi_zfw_skew_factorial["arms"]
        removal = redi_zfw_skew_factorial[
            "max_error_removal_fraction"]
        if arms["C0G0T1"]["pass"]:
            disposition = "INVALID_REDI_ZFW_SKEW_MATCHES_KMM_NOT_KBB"
        elif arms["C1G1T0"]["pass"]:
            if (removal["C1G0T0"] >= 0.90
                    and removal["C0G1T0"] < 0.90):
                disposition = "REDI_ZFW_T_SKEW_LOCALIZED_TO_COEFFICIENT_ASSEMBLY"
            elif (removal["C0G1T0"] >= 0.90
                  and removal["C1G0T0"] < 0.90):
                disposition = "REDI_ZFW_T_SKEW_LOCALIZED_TO_GRADIENT_ASSOCIATION"
            else:
                disposition = "REDI_ZFW_T_SKEW_COEFFICIENT_GRADIENT_COMPOSITION"
        else:
            disposition = "REDI_ZFW_T_SKEW_A31_A32_HELD_SPLIT_REQUIRED"
        first_flux_subrow = "88.T.3a"
    if args.redi_zfw_skew_postfix and valid:
        skew_pass = redi_zfw_component_metrics["skew"]["pass"]
        a33_pass = redi_zfw_component_metrics["a33"]["pass"]
        total_pass = redi_flux_metrics["zfw_tem"]["pass"]
        if not skew_pass:
            disposition = "REDI_ZFW_T_SKEW_POSTFIX_REGRESSION"
        elif not a33_pass:
            disposition = "REDI_ZFW_T_SKEW_FIXED_A33_NEXT"
        elif total_pass:
            disposition = "REDI_ZFW_T_AT_BAR"
        else:
            disposition = "REDI_ZFW_T_COMPONENT_SUM_ASSOCIATION_OPEN"
        first_flux_subrow = "89.T.3a"
    if args.redi_zfw_skew_operand_factorial and valid:
        factor_names = ("U", "V", "I", "J")
        label = {
            "U": "AHTU", "V": "AHTV",
            "I": "WSLPI", "J": "WSLPJ",
        }
        passing = []
        for arm, metric in redi_zfw_skew_operand_factorial["arms"].items():
            enabled = tuple(name for index, name in enumerate(factor_names)
                            if arm[2 * index + 1] == "1")
            if enabled and metric["pass"]:
                passing.append((len(enabled), enabled, arm))
        if passing:
            _, enabled, _ = min(passing)
            disposition = ("REDI_ZFW_T_SKEW_OWNED_"
                           + "_".join(label[name] for name in enabled))
        else:
            disposition = "REDI_ZFW_T_SKEW_A31_A32_HELD_SPLIT_REQUIRED"
        first_flux_subrow = "90.T.3a"
    if args.redi_zfw_wslp_stage_factorial and valid:
        joint = redi_zfw_wslp_stage_factorial["arms"]["I1J1"]
        removal = redi_zfw_wslp_stage_factorial[
            "max_error_removal_fraction"]["I1J1"]
        if joint["pass"] and removal >= 0.90:
            disposition = "REDI_ZFW_T_SKEW_OWNED_NAA_W_SLOPE_PAIR"
        else:
            disposition = "REDI_ZFW_T_SKEW_NEEDS_EXISTING_ROW30_LADDER"
        first_flux_subrow = "91.T.3a"
    if args.redi_zfw_a33_floor_factorial and valid:
        faithful93 = redi_zfw_a33_floor_factorial["arms"]["H1K1"]
        if faithful93["full_zfw"]["pass"]:
            disposition = "REDI_ZFW_T_AT_BAR_ROUND93"
            t3_status = "AT_BAR"
        elif (faithful93["ah_wslp2"]["pass"]
              and faithful93["akz"]["pass"]
              and redi_zfw_a33_floor_factorial["controls"][
                  "oracle_component_closure"]
              and faithful93["full_zfw"]["max_column_error"] <= 2.0e-14):
            disposition = "REDI_ZFW_T_CLEARED_RULE1B_ORACLE_ARITHMETIC"
            t3_status = "CLEARED_RULE1B"
        else:
            disposition = "REDI_ZFW_T_ASSOCIATION_UNRESOLVED"
            t3_status = "DIVERGED"
        if t3_status != "DIVERGED":
            # Release the already measured salinity siblings without
            # rerunning NEMO. Temperature T.3 retains the faithful-arm metric
            # that supports either the strict or Rule-1b clearance.
            redi_flux_rows = []
            first_flux = None
            first_flux_subrow = None
            release_order = (
                ("zfu_tem", "78.T.1", "temperature zfu",
                 "traldf_iso_scheme.h90:55-70"),
                ("zfv_tem", "78.T.2", "temperature zfv",
                 "traldf_iso_scheme.h90:72-90"),
                ("zfw_tem", "78.T.3", "temperature zfw_kp1",
                 "traldf_iso_scheme.h90:104-129"),
                ("zfu_sal", "78.S.1", "salinity zfu",
                 "traldf_iso_scheme.h90:55-70"),
                ("zfv_sal", "78.S.2", "salinity zfv",
                 "traldf_iso_scheme.h90:72-90"),
                ("zfw_sal", "78.S.3", "salinity zfw_kp1",
                 "traldf_iso_scheme.h90:104-129"),
            )
            blocked93 = False
            rule1b93 = t3_status == "CLEARED_RULE1B"
            for operand, subrow, name, source in release_order:
                row = {"subrow": subrow, "name": name,
                       "nemo_source": source, "bar": POINTWISE_BAR}
                if blocked93:
                    row["status"] = "ORDERED_BLOCKED"
                elif subrow == "78.T.3":
                    row.update(status=t3_status,
                               clearance=("bar" if t3_status == "AT_BAR"
                                          else "Rule-1b proven-oracle-arithmetic"),
                               metrics=faithful93["full_zfw"])
                else:
                    metric = redi_flux_metrics[operand]
                    salinity_arithmetic_floor = (
                        operand == "zfw_sal"
                        and not metric["pass"]
                        and metric["max_column_error"] <= 2.0e-14
                        and abs(metric["correlation"] - 1.0) <= 2.0e-15
                        and abs(metric["rms_ratio"] - 1.0) <= 2.0e-15
                        and faithful93["full_zfw"]["pass"]
                        and faithful93["ah_wslp2"]["pass"]
                        and faithful93["akz"]["pass"])
                    if metric["pass"]:
                        row.update(status="AT_BAR", metrics=metric)
                    elif salinity_arithmetic_floor:
                        row.update(
                            status="CLEARED_RULE1B",
                            clearance="Rule-1b proven-oracle-arithmetic",
                            metrics=metric)
                        rule1b93 = True
                    else:
                        row.update(status="DIVERGED", metrics=metric)
                        blocked93 = True
                        first_flux = operand
                        first_flux_subrow = subrow
                redi_flux_rows.append(row)
            if not blocked93:
                disposition = ("TRACER_TAIL_REDI_CLEARED_ROUND93_RULE1B"
                               if rule1b93 else
                               "TRACER_TAIL_REDI_CLEARED_ROUND93")
        first_flux_subrow = (first_flux_subrow
                             if t3_status != "DIVERGED" else "93.T.3b")
    receipt_first = (first_flux_subrow
                     if args.redi_flux_ladder and first_flux_subrow is not None
                     else first)
    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": ("dino-split-explicit-momentum-chain-round93-v1"
                   if args.redi_zfw_a33_floor_factorial else
                   "dino-split-explicit-momentum-chain-round91-v1"
                   if args.redi_zfw_wslp_stage_factorial else
                   "dino-split-explicit-momentum-chain-round90-v1"
                   if args.redi_zfw_skew_operand_factorial else
                   "dino-split-explicit-momentum-chain-round89-v1"
                   if args.redi_zfw_skew_postfix else
                   "dino-split-explicit-momentum-chain-round88-v1"
                   if args.redi_zfw_skew_factorial else
                   "dino-split-explicit-momentum-chain-round87-v1"
                   if args.redi_zfw_component_score else
                   "dino-split-explicit-momentum-chain-round86-v1"
                   if args.redi_zfw_association_factorial else
                   "dino-split-explicit-momentum-chain-round85-v1"
                   if args.redi_zfu_ahtu_postfix else
                   "dino-split-explicit-momentum-chain-round84-v1"
                   if args.redi_zfu_bolus_stage_split_postfix else
                   "dino-split-explicit-momentum-chain-round83-v1"
                   if args.redi_zfu_kmm_operator_postfix else
                   "dino-split-explicit-momentum-chain-round82-v1"
                   if args.redi_zfu_slope_kmm_postfix else
                   "dino-split-explicit-momentum-chain-round81-v1"
                   if args.redi_zfu_kmm_postfix else
                   "dino-split-explicit-momentum-chain-round80-v1"
                   if args.redi_zfu_postfix else
                   "dino-split-explicit-momentum-chain-round79-v1"
                   if args.redi_zfu_operand_ladder else
                   "dino-split-explicit-momentum-chain-round78-v1"
                   if args.redi_flux_ladder else
                   "dino-split-explicit-momentum-chain-round77-v1"
                   if args.redi_e3w_factorial else
                   "dino-split-explicit-momentum-chain-round76-v1"
                   if args.capture_redi_tail else
                   "dino-split-explicit-momentum-chain-round75-v1"
                   if args.exact_sqrt_production else
                   "dino-split-explicit-momentum-chain-round74-v1"
                   if args.zn_sqrt_factorial else
                   "dino-split-explicit-momentum-chain-round73-v1"
                   if args.rossby_factorial else
                   "dino-split-explicit-momentum-chain-round72-v1"
                   if args.post_chain_factorial else
                   "dino-split-explicit-momentum-chain-round71-v1"
                   if args.exact_surface_kmm_carry else
                   "dino-split-explicit-momentum-chain-round64-v1"
                   if args.live_thickness_entry else
                   "dino-split-explicit-momentum-chain-round63-v1"
                   if args.direct_cycle_entry else
                   "dino-split-explicit-momentum-chain-round62-v1"
                   if args.capture_cycle else
                   "dino-split-explicit-momentum-chain-round61-v1"
                   if args.oracle_transport else
                   "dino-split-explicit-momentum-chain-round59-v1"),
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "disposition": disposition,
        "first_diverged_subrow": receipt_first,
        "rows": rows,
        "controls": controls,
        **({"cycle_capture_metrics": capture_metrics}
           if args.capture_cycle else {}),
        **({"bolus_operand_metrics": bolus_operand_metrics}
           if args.capture_bolus_operands else {}),
        **({"kappa_operand_metrics": kappa_operand_metrics}
           if args.capture_kappa_operands else {}),
        **({"kappa_geometry_metrics": kappa_geometry_metrics}
           if args.capture_kappa_geometry else {}),
        **({"post_chain_metrics": post_chain_metrics}
           if args.post_chain_factorial else {}),
        **({"rossby_metrics": rossby_metrics}
           if args.rossby_factorial else {}),
        **({"zn_sqrt_metrics": zn_sqrt_metrics}
           if args.zn_sqrt_factorial else {}),
        **({"redi_metrics": redi_metrics}
           if args.capture_redi_tail else {}),
        **({"redi_e3w_metrics": redi_e3w_metrics}
           if args.redi_e3w_factorial else {}),
        **({"redi_flux_metrics": redi_flux_metrics}
           if args.redi_flux_ladder else {}),
        **({"redi_flux_rows": redi_flux_rows,
            "first_diverged_flux_operand": first_flux}
           if args.redi_flux_ladder else {}),
        **({"redi_zfu_operand_ladder": redi_zfu_operand_ladder}
           if args.redi_zfu_operand_ladder else {}),
        **({"redi_zfw_association_factorial":
            redi_zfw_association_factorial}
           if args.redi_zfw_association_factorial else {}),
        **({"redi_zfw_component_metrics": redi_zfw_component_metrics}
           if args.redi_zfw_component_score else {}),
        **({"redi_zfw_skew_factorial": redi_zfw_skew_factorial}
           if args.redi_zfw_skew_factorial else {}),
        **({"redi_zfw_skew_operand_factorial":
            redi_zfw_skew_operand_factorial}
           if args.redi_zfw_skew_operand_factorial else {}),
        **({"redi_zfw_wslp_stage_factorial":
            redi_zfw_wslp_stage_factorial}
           if args.redi_zfw_wslp_stage_factorial else {}),
        **({"redi_zfw_a33_floor_factorial":
            redi_zfw_a33_floor_factorial}
           if args.redi_zfw_a33_floor_factorial else {}),
        "focus_ji": [list(x) for x in FOCUS],
        "bindings": {
            "round54": _sha(args.round54.resolve()),
            **({"round56": _sha(args.round56.resolve())}
               if args.hold_slow_forcing else {}),
            **({"round59": _sha(args.round59.resolve()),
                "round60": _sha(args.round60.resolve())}
               if args.oracle_transport else {}),
            **({"round61": _sha(args.round61.resolve())}
               if args.capture_cycle else {}),
            **({"round62": _sha(args.round62.resolve())}
               if args.direct_cycle_entry else {}),
            **({"round63": _sha(args.round63.resolve())}
               if args.live_thickness_entry else {}),
            **({"round64": _sha(args.round64.resolve()),
                **{name: _sha(held / name) for name in BOLUS_HELD_SHA}}
               if args.capture_bolus_operands else {}),
            **({"round65": _sha(args.round65.resolve()),
                **{name: _sha(held / name) for name in KAPPA_HELD_SHA}}
               if args.capture_kappa_operands else {}),
            **({"round66": _sha(args.round66.resolve())}
               if args.literal_kappa_reduction else {}),
            **({"round67": _sha(args.round67.resolve()),
                **{name: _sha(held / name) for name in KAPPA_GEOMETRY_SHA}}
               if args.capture_kappa_geometry else {}),
            **({"round68": _sha(args.round68.resolve())}
               if args.coupled_kappa_carry else {}),
            **({"round69": _sha(args.round69.resolve())}
               if args.surface_kmm_carry else {}),
            **({"round70": _sha(args.round70.resolve())}
               if args.exact_surface_kmm_carry else {}),
            **({"round71": _sha(args.round71.resolve())}
               if args.post_chain_factorial else {}),
            **({"round72": _sha(args.round72.resolve())}
               if args.rossby_factorial else {}),
            **({"round73": _sha(args.round73.resolve())}
               if args.zn_sqrt_factorial else {}),
            **({"round74": _sha(args.round74.resolve())}
               if args.exact_sqrt_production else {}),
            **({"round75": _sha(args.round75.resolve()),
                **{name: _sha(redi_dir / name) for name in REDI_HELD_SHA}}
               if args.capture_redi_tail else {}),
            **({"round76": _sha(args.round76.resolve())}
               if args.redi_e3w_factorial else {}),
            **({"round77": _sha(args.round77.resolve()),
                "redi_flux_bracket": _sha(args.redi_flux_bracket.resolve()),
                **{name: _sha(flux_dir / name)
                   for name in sorted(expected_flux)}}
               if args.redi_flux_ladder else {}),
            **({"round78": _sha(args.round78.resolve()),
                "ldftra_dump_ahtu.bin": _sha(
                    flux_dir / "ldftra_dump_ahtu.bin"),
                "eiv_dump_uslp.bin": _sha(
                    flux_dir / "eiv_dump_uslp.bin")}
               if args.redi_zfu_operand_ladder else {}),
            **({"round79": _sha(args.round79.resolve())}
               if args.redi_zfu_postfix else {}),
            **({"round80": _sha(args.round80.resolve())}
               if args.redi_zfu_kmm_postfix else {}),
            **({"round81": _sha(args.round81.resolve())}
               if args.redi_zfu_slope_kmm_postfix else {}),
            **({"round82": _sha(args.round82.resolve())}
               if args.redi_zfu_kmm_operator_postfix else {}),
            **({"round83": _sha(args.round83.resolve())}
               if args.redi_zfu_bolus_stage_split_postfix else {}),
            **({"round84": _sha(args.round84.resolve())}
               if args.redi_zfu_ahtu_postfix else {}),
            **({"round85": _sha(args.round85.resolve())}
               if args.redi_zfw_association_factorial else {}),
            **({"round86": _sha(args.round86.resolve()),
                "redi_zfw_component_bracket": _sha(
                    args.redi_zfw_component_bracket.resolve()),
                **{name: _sha(component_dir / name)
                   for name in sorted(expected_components)}}
               if args.redi_zfw_component_score else {}),
            **({"round87": _sha(args.round87.resolve()),
                **{f"skew_input_{name}": _sha(flux_dir / name)
                   for name in sorted(REDI_SKEW_INPUT_SHA)}}
               if args.redi_zfw_skew_factorial else {}),
            **({"round88": _sha(args.round88.resolve())}
               if args.redi_zfw_skew_postfix else {}),
            **({"round89": _sha(args.round89.resolve()),
                **{f"skew_operand_input_{name}": _sha(flux_dir / name)
                   for name in sorted(REDI_SKEW_INPUT_SHA)}}
               if args.redi_zfw_skew_operand_factorial else {}),
            **({"round90": _sha(args.round90.resolve())}
               if args.redi_zfw_wslp_stage_factorial else {}),
            **({"round93_wslp": _sha(args.round93_wslp.resolve())}
               if args.redi_zfw_a33_floor_factorial else {}),
            "held_raw_artifact": _sha(args.raw_artifact.resolve()),
            **{name: _sha(held / name) for name in HELD_SHA},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity" /
                ("PREREG_split_explicit_momentum_chain_round93.md"
                 if args.redi_zfw_a33_floor_factorial else
                 "PREREG_split_explicit_momentum_chain_round91.md"
                 if args.redi_zfw_wslp_stage_factorial else
                 "PREREG_split_explicit_momentum_chain_round90.md"
                 if args.redi_zfw_skew_operand_factorial else
                 "PREREG_split_explicit_momentum_chain_round89.md"
                 if args.redi_zfw_skew_postfix else
                 "PREREG_split_explicit_momentum_chain_round88.md"
                 if args.redi_zfw_skew_factorial else
                 "PREREG_split_explicit_momentum_chain_round87.md"
                 if args.redi_zfw_component_score else
                 "PREREG_split_explicit_momentum_chain_round86.md"
                 if args.redi_zfw_association_factorial else
                 "PREREG_split_explicit_momentum_chain_round85.md"
                 if args.redi_zfu_ahtu_postfix else
                 "PREREG_split_explicit_momentum_chain_round84.md"
                 if args.redi_zfu_bolus_stage_split_postfix else
                 "PREREG_split_explicit_momentum_chain_round83.md"
                 if args.redi_zfu_kmm_operator_postfix else
                 "PREREG_split_explicit_momentum_chain_round82.md"
                 if args.redi_zfu_slope_kmm_postfix else
                 "PREREG_split_explicit_momentum_chain_round81.md"
                 if args.redi_zfu_kmm_postfix else
                 "PREREG_split_explicit_momentum_chain_round80.md"
                 if args.redi_zfu_postfix else
                 "PREREG_split_explicit_momentum_chain_round79.md"
                 if args.redi_zfu_operand_ladder else
                 "PREREG_split_explicit_momentum_chain_round78.md"
                 if args.redi_flux_ladder else
                 "PREREG_split_explicit_momentum_chain_round77.md"
                 if args.redi_e3w_factorial else
                 "PREREG_split_explicit_momentum_chain_round76.md"
                 if args.capture_redi_tail else
                 "PREREG_split_explicit_momentum_chain_round75.md"
                 if args.exact_sqrt_production else
                 "PREREG_split_explicit_momentum_chain_round74.md"
                 if args.zn_sqrt_factorial else
                 "PREREG_split_explicit_momentum_chain_round73.md"
                 if args.rossby_factorial else
                 "PREREG_split_explicit_momentum_chain_round72.md"
                 if args.post_chain_factorial else
                 "PREREG_split_explicit_momentum_chain_round71.md"
                 if args.exact_surface_kmm_carry else
                 "PREREG_split_explicit_momentum_chain_round70.md"
                 if args.surface_kmm_carry else
                 "PREREG_split_explicit_momentum_chain_round69.md"
                 if args.coupled_kappa_carry else
                 "PREREG_split_explicit_momentum_chain_round68.md"
                 if args.capture_kappa_geometry else
                 "PREREG_split_explicit_momentum_chain_round67.md"
                 if args.literal_kappa_reduction else
                 "PREREG_split_explicit_momentum_chain_round66.md"
                 if args.capture_kappa_operands else
                 "PREREG_split_explicit_momentum_chain_round65.md"
                 if args.capture_bolus_operands else
                 "PREREG_split_explicit_momentum_chain_round64.md"
                 if args.live_thickness_entry else
                 "PREREG_split_explicit_momentum_chain_round63.md"
                 if args.direct_cycle_entry else
                 "PREREG_split_explicit_momentum_chain_round62.md"
                 if args.capture_cycle else
                 "PREREG_split_explicit_momentum_chain_round61.md"
                 if args.oracle_transport else
                 "PREREG_split_explicit_momentum_chain_round59.md")),
            **({"preregistration_amendment": _sha(
                root / "docs/ocean/fidelity" /
                "PREREG_split_explicit_momentum_chain_round90_amendment.md")}
               if args.redi_zfw_skew_operand_factorial else {}),
            "production_model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "production_barotropic": _sha(root / "packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
            "nemo_traadv": _sha(nemo / "src/OCE/TRA/traadv.F90"),
            "nemo_traadv_fct": _sha(nemo / "src/OCE/TRA/traadv_fct.F90"),
            "nemo_ldftra": _sha(nemo / "cfgs/DINO/MY_SRC/ldftra.F90"),
        },
        "arm": ("redi_zfw_a33_floor_factorial"
                if args.redi_zfw_a33_floor_factorial else
                "redi_zfw_wslp_stage_factorial"
                if args.redi_zfw_wslp_stage_factorial else
                "redi_zfw_skew_operand_factorial"
                if args.redi_zfw_skew_operand_factorial else
                "redi_zfw_skew_postfix"
                if args.redi_zfw_skew_postfix else
                "redi_zfw_skew_factorial"
                if args.redi_zfw_skew_factorial else
                "redi_zfw_component_score"
                if args.redi_zfw_component_score else
                "redi_zfw_association_factorial"
                if args.redi_zfw_association_factorial else
                "redi_zfu_ahtu_postfix"
                if args.redi_zfu_ahtu_postfix else
                "redi_zfu_bolus_stage_split_postfix"
                if args.redi_zfu_bolus_stage_split_postfix else
                "redi_zfu_kmm_operator_postfix"
                if args.redi_zfu_kmm_operator_postfix else
                "redi_zfu_slope_kmm_postfix"
                if args.redi_zfu_slope_kmm_postfix else
                "redi_zfu_kmm_postfix"
                if args.redi_zfu_kmm_postfix else
                "redi_zfu_postfix"
                if args.redi_zfu_postfix else
                "redi_zfu_operand_ladder"
                if args.redi_zfu_operand_ladder else
                "redi_flux_ladder"
                if args.redi_flux_ladder else
                "redi_e3w_factorial"
                if args.redi_e3w_factorial else
                "capture_redi_tail"
                if args.capture_redi_tail else
                "exact_sqrt_production"
                if args.exact_sqrt_production else
                "zn_sqrt_factorial"
                if args.zn_sqrt_factorial else
                "rossby_factorial"
                if args.rossby_factorial else
                "post_chain_factorial"
                if args.post_chain_factorial else
                "exact_surface_kmm_carry"
                if args.exact_surface_kmm_carry else
                "surface_kmm_carry" if args.surface_kmm_carry else
                "coupled_kappa_carry" if args.coupled_kappa_carry else
                "kappa_geometry_capture" if args.capture_kappa_geometry else
                "literal_kappa_reduction" if args.literal_kappa_reduction else
                "kappa_operand_capture" if args.capture_kappa_operands else
                "bolus_operand_capture" if args.capture_bolus_operands else
                "live_thickness_entry" if args.live_thickness_entry else
                "direct_cycle_entry" if args.direct_cycle_entry else
                "cycle_capture" if args.capture_cycle else
                "oracle_transport" if args.oracle_transport else
                "held_slow_forcing" if args.hold_slow_forcing else
                "production"),
        "ordered_next": ("redi_zfw_mixed_native_slope_carry" if disposition ==
                          "REDI_ZFW_T_SKEW_OWNED_NAA_W_SLOPE_PAIR" else
                          "redi_zfw_existing_row30_slope_ladder" if disposition ==
                          "REDI_ZFW_T_SKEW_NEEDS_EXISTING_ROW30_LADDER" else
                          "redi_zfw_skew_operand_carry" if
                          disposition.startswith(
                              "REDI_ZFW_T_SKEW_OWNED_") else
                          "stop_skew_postfix_regression" if disposition ==
                          "REDI_ZFW_T_SKEW_POSTFIX_REGRESSION" else
                          "redi_zfw_a33_operand_peel" if disposition ==
                          "REDI_ZFW_T_SKEW_FIXED_A33_NEXT" else
                          "redi_zfw_component_sum_association" if disposition ==
                          "REDI_ZFW_T_COMPONENT_SUM_ASSOCIATION_OPEN" else
                          "redi_salinity_flux_ladder" if disposition ==
                          "REDI_ZFW_T_AT_BAR" else
                          "stop_invalid_kbb_registration" if disposition ==
                          "INVALID_REDI_ZFW_SKEW_MATCHES_KMM_NOT_KBB" else
                          "redi_zfw_skew_source_fix" if disposition in (
                          "REDI_ZFW_T_SKEW_LOCALIZED_TO_COEFFICIENT_ASSEMBLY",
                          "REDI_ZFW_T_SKEW_LOCALIZED_TO_GRADIENT_ASSOCIATION",
                          "REDI_ZFW_T_SKEW_COEFFICIENT_GRADIENT_COMPOSITION") else
                          "redi_zfw_a31_vs_a32_held_split" if disposition ==
                          "REDI_ZFW_T_SKEW_A31_A32_HELD_SPLIT_REQUIRED" else
                          "redi_zfw_a31_a32_operand_peel" if disposition ==
                          "REDI_ZFW_T_DIVERGED_A31_A32" else
                          "redi_zfw_a33_operand_peel" if disposition ==
                          "REDI_ZFW_T_DIVERGED_A33" else
                          "redi_zfw_component_sum_association" if disposition ==
                          "REDI_ZFW_T_COMPONENTS_AT_BAR" else
                          "redi_zfw_source_association_fix" if disposition in (
                          "REDI_ZFW_T_GRADIENT_ASSOCIATION_OWNED",
                          "REDI_ZFW_T_A33_ASSOCIATION_OWNED",
                          "REDI_ZFW_T_ASSOCIATION_COMPOSITION_OWNED") else
                          "redi_zfw_component_held_instrumentation" if disposition ==
                          "REDI_ZFW_T_COMPONENTS_HELD_OPEN" else
                          "redi_zfu_ahtu_last_bits" if disposition in (
                          "REDI_ZFU_T_KMM_OPERATOR_FIXED_AHTU_RESIDUAL",
                          "REDI_ZFU_T_BOLUS_STAGE_SPLIT_FIXED_AHTU_RESIDUAL") else
                          "redi_zfu_temperature_vflux" if disposition in (
                          "REDI_ZFU_T_AHTU_AT_BAR",
                          "REDI_ZFU_T_KMM_OPERATOR_AT_BAR",
                          "REDI_ZFU_T_BOLUS_STAGE_SPLIT_AT_BAR",
                          "REDI_ZFU_T_KMM_SLOPE_AT_BAR") else
                          "stop_postfix_regression" if disposition in (
                          "REDI_ZFU_T_KMM_SLOPE_REGRESSION",
                          "REDI_ZFU_T_KMM_OPERATOR_REGRESSION",
                          "REDI_ZFU_T_BOLUS_STAGE_SPLIT_REGRESSION",
                          "REDI_ZFU_T_AHTU_REGRESSION") else
                          "redi_zfu_final_uslp_residual" if disposition in (
                          "REDI_ZFU_T_POSTFIX_RESIDUAL_FINAL_USLP",
                          "REDI_ZFU_T_KMM_FACE_FIXED_RESIDUAL_FINAL_USLP") else
                          "redi_zfu_temperature_vflux" if disposition in (
                          "REDI_ZFU_T_POSTFIX_AT_BAR",
                          "REDI_ZFU_T_KMM_FACE_FIXED_AT_BAR") else
                          "stop_postfix_regression" if disposition in (
                          "REDI_ZFU_T_POSTFIX_REGRESSION",
                          "REDI_ZFU_T_KMM_FACE_FIXED_REGRESSION") else
                          "redi_zfu_live_e3u_production" if disposition ==
                          "REDI_ZFU_T_LOCALIZED_TO_LIVE_E3U" else
                          "redi_zfu_operand_followup" if disposition.startswith(
                          "REDI_ZFU_T_") else
                          "redi_flux_subpeel" if disposition.startswith(
                          "REDI_DIVERGED_ZF") else
                          "redi_divergence_volume_peel" if disposition ==
                          "REDI_DIVERGED_DIVERGENCE_VOLUME_T" else
                          "redi_msc_e3w_production" if disposition ==
                          "REDI_MSC_E3W_OWNED" else
                          "redi_flux_ladder" if disposition in {
                              "REDI_MSC_E3W_MAJORITY",
                              "REDI_MSC_E3W_REFUTED"} else
                          "tracer_zdf" if disposition in {
                              "TRACER_TAIL_REDI_AT_BAR",
                              "TRACER_TAIL_REDI_AT_BAR_LIVE_E3W"} else
                          "redi_t" if disposition in {
            "TRACER_ENTRY_ROW8_AT_BAR",
            "TRACER_ENTRY_ROW8_AT_BAR_UPSTREAM_FORCING_EXACT",
            "TRACER_ENTRY_ROW8_AT_BAR_ORACLE_TRANSPORT",
            "TRACER_ENTRY_ROW8_AT_BAR_DIRECT_KMM",
            "TRACER_ENTRY_ROW8_AT_BAR_LIVE_QCO",
            "TRACER_ENTRY_ROW8_AT_BAR_LITERAL_GM",
            "TRACER_ENTRY_ROW8_AT_BAR_CARRIED_KMM",
            "TRACER_ENTRY_ROW8_AT_BAR_EXACT_SURFACE_KMM",
            "TRACER_ENTRY_ROW8_AT_BAR_EXACT_GM_SQRT"} else first),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} "
          f"first_diverged_subrow={receipt_first}")
    for row in rows + redi_flux_rows:
        if "metrics" in row:
            print(row["subrow"], row["status"],
                  row["metrics"]["n_diverged_columns"],
                  row["metrics"]["max_column_error"])
        else:
            print(row["subrow"], row["status"])
    if args.redi_zfu_operand_ladder:
        for name, metric in sorted(
                redi_zfu_operand_ladder["arms"].items()):
            print(name, "AT_BAR" if metric["pass"] else "DIVERGED",
                  metric["n_diverged_columns"], metric["max_column_error"],
                  metric["max_error_removal_fraction"])
        literal = redi_zfu_operand_ladder["literal_all_oracle"]
        print("S1H1K1L1", "AT_BAR" if literal["pass"] else "DIVERGED",
              literal["n_diverged_columns"], literal["max_column_error"],
              literal["max_error_removal_fraction"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
