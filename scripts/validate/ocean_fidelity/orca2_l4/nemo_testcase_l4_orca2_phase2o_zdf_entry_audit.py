#!/usr/bin/env python3
"""Fail-closed inventory of the ORCA2 TKE/EVD/DDM/IWM entry boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
)

NX, NY, NZ, HALO = 94, 152, 31, 2
N3 = NX * NY * NZ
NI3 = (NX - 2 * HALO) * (NY - 2 * HALO) * NZ


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_zdf_record(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        payload = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L4_ZDF___2", f"ZDF magic {magic!r}")
    require(header == (2, 1, 1, NX, NY, NZ, 64), f"ZDF header {header}")
    require(payload.size == N3 + 3 * NI3,
            f"ZDF payload {payload.size} != {N3 + 3 * NI3}")
    require(np.isfinite(payload).all(), "non-finite ZDF payload")
    fields = []
    cursor = 0
    for name, count, allocation in (
        ("avm", N3, "full jpi*jpj*jpk"),
        ("avt", NI3, "rank0 interior (jpi-4)*(jpj-4)*jpk"),
        ("avs", NI3, "rank0 interior (jpi-4)*(jpj-4)*jpk"),
        ("en", NI3, "rank0 interior (jpi-4)*(jpj-4)*jpk"),
    ):
        values = payload[cursor:cursor + count]
        fields.append({
            "name": name,
            "allocation": allocation,
            "count": int(values.size),
            "finite": int(np.isfinite(values).sum()),
        })
        cursor += count
    require(cursor == payload.size, "ZDF schema did not reach exact EOF")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "magic": magic,
        "header": list(header),
        "payload_f64": int(payload.size),
        "bytes": path.stat().st_size,
        "sampling": "after complete zdf_phy(kt=1,Kbb=1,Kmm=1,Krhs=3)",
        "fields": fields,
    }


def _surface_field(path: Path, wanted: str) -> np.ndarray:
    """Read one field from the canonical post-SBC frame."""
    fields = (
        ("utau", "full"), ("vtau", "full"), ("utauU", "full"),
        ("vtauV", "full"), ("utau_b", "full"), ("vtau_b", "full"),
        ("utau_icb", "full"), ("vtau_icb", "full"),
        ("taum", "reduced"), ("wndm", "reduced"),
        ("qsr", "reduced"), ("qns", "reduced"),
        ("qns_b", "reduced"), ("qsr_tot", "reduced"),
        ("qns_tot", "reduced"), ("emp", "full"), ("emp_b", "full"),
        ("sfx", "reduced"), ("sfx_b", "reduced"),
        ("emp_tot", "reduced"), ("fwfice", "reduced"),
        ("rnf", "full"), ("rnf_b", "full"), ("fwficb", "reduced"),
        ("fr_i", "full"), ("snwice_mass", "full"),
        ("snwice_mass_b", "full"), ("snwice_fmass", "full"),
        ("rCdU_ice", "halo1"), ("icb_calving", "full"),
        ("icb_calving_hflx", "full"), ("icb_floating_melt", "full"),
        ("icb_stored_heat", "full"), ("rnf_tsc", "reduced3d"),
        ("rnf_tsc_b", "reduced3d"),
    )
    with path.open("rb") as handle:
        require(handle.read(16).decode("ascii").rstrip() == "NEMO_L4_SBCIN_1",
                "post-SBC magic")
        header = struct.unpack("=13i", handle.read(52))
        require(header[:7] == (1, 1, 1, NX, NY, 2, 10),
                f"post-SBC header {header}")
        payload = np.fromfile(handle, np.float64)
    sizes = {
        "full": NX * NY,
        "reduced": (NX - 2 * HALO) * (NY - 2 * HALO),
        "halo1": (NX - 2) * (NY - 2),
        "reduced3d": 2 * (NX - 2 * HALO) * (NY - 2 * HALO),
    }
    cursor = 0
    for name, allocation in fields:
        count = sizes[allocation]
        if name == wanted:
            require(allocation == "full", f"{wanted} is not full-domain")
            return payload[cursor:cursor + count].reshape((NX, NY), order="F").T
        cursor += count
    raise AuditError(f"post-SBC field {wanted!r} not found")


def _require_runtime_line(text: str, literal: str) -> None:
    require(literal in text, f"resolved runtime output lacks {literal!r}")


def audit(deck: Path, run: Path) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "ZDF audit is CPU-only")
    output_path = run / "ocean.output"
    output = output_path.read_text(errors="replace")
    for literal in (
        "ln_zdftke =  T", "ln_zdfevd =  T", "nn_evdm =            0",
        "rn_evd  =    100.00000000000000", "ln_zdfddm =  T",
        "ln_zdfiwm =  T", "nn_pdl    =            1",
        "nn_mxl    =            3", "ln_mxl0   =  T", "ln_lc     =  T",
        "nn_etau   =            1", "nn_htau   =            1",
        "nn_eice =            1",
        "Variable (T) or constant (F) mixing efficiency            =  F",
        "Differential internal wave-driven mixing (T) or not (F)   =  T",
    ):
        _require_runtime_line(output, literal)

    card = build_orca2_zps_card(deck)
    physics = card.recipe.model_config.physics
    vmix = physics.vertical_mixing
    tke = vmix.tke
    evd = physics.convection.enhanced_diffusion
    zdf = _validate_zdf_record(run / "oracle_zdf_entry_kt00000001.bin")

    fr_i = _surface_field(run / "oracle_ocean_surface_input_kt00000001.bin",
                          "fr_i")
    expected_effective_ice = np.tanh(10.0 * fr_i)
    production_effective_ice = np.zeros_like(expected_effective_ice)
    wet_ice = expected_effective_ice != 0.0
    differing = wet_ice & (
        expected_effective_ice.view(np.uint64)
        != production_effective_ice.view(np.uint64))
    first = np.argwhere(differing)[0]

    card_rows = {
        "closure": {"nemo": "ln_zdftke=T", "card": vmix.scheme,
                    "status": "SELECTOR_ALIGNED"},
        "shear_geometry": {
            "nemo": "zdfsh2 face-native now-times-before",
            "card": tke.tke_shear_production,
            "status": "CARD_SELECTOR_DEBT",
        },
        "iwm_rebased_avm_floor_m2_s": {
            "nemo": 1.4e-6, "card": tke.kappaM_min,
            "status": ("CARD_MATCHES_NEMO" if tke.kappaM_min == 1.4e-6
                       else "CARD_SELECTOR_DEBT"),
        },
        "iwm_rebased_avt_floor_m2_s": {
            "nemo": 1.0e-10, "card": tke.kappaH_min,
            "status": ("CARD_MATCHES_NEMO" if tke.kappaH_min == 1.0e-10
                       else "CARD_SELECTOR_DEBT"),
        },
        "evd_momentum": {
            "nemo_nn_evdm": 0, "card_nu_conv_m2_s": evd.nu_conv,
            "status": "CARD_SELECTOR_DEBT",
        },
        "double_diffusion": {
            "nemo": True, "card": bool(vmix.ddm.enabled),
            "status": "CARD_SELECTOR_DEBT",
        },
        "internal_wave_mixing": {
            "nemo": True, "card": bool(vmix.iwm.enabled),
            "status": ("CARD_MATCHES_NEMO" if vmix.iwm.enabled
                       else "UNMEASURED_ORCA2_OWNER"),
        },
        "iwm_differential_ts": {
            "nemo": True, "card": bool(vmix.iwm.tsdiff),
            "status": "UNMEASURED_ORCA2_OWNER",
        },
        "tke_bottom_boundary": {
            "nemo": "active when bottom drag is active",
            "card": bool(tke.bottom_tke_bc),
            "status": "GYRE_OWNER_SHARED_TKE_DEBT",
        },
        "tke_etau": {
            "nemo_nn_etau": 1, "card": tke.etau_mode,
            "status": "GYRE_OWNER_SHARED_TKE_DEBT",
        },
        "tke_ice_attenuation": {
            "nemo_nn_eice": 1, "card_eice": int(tke.eice),
            "nemo_effective": "tanh(10*fr_i)",
            "card_effective": "zero/no attenuation",
            "status": "GYRE_OWNER_SHARED_TKE_DEBT",
            "unequal": int(differing.sum()),
            "count": int(wet_ice.sum()),
            "first_zero_based_j_i": [int(first[0]), int(first[1])],
            "first_nemo": float(expected_effective_ice[tuple(first)]),
            "first_card": 0.0,
        },
    }

    return {
        "status": "UNMEASURED_NEEDS_WRITE_ONLY_FRAMES",
        "execution": {"backend": jax.default_backend(), "dtype": "float64",
                      "transcendentals": "libm"},
        "runtime_provenance": {
            "ocean_output": str(output_path), "sha256": sha256(output_path),
            "selectors_verified": 15,
        },
        "existing_record": zdf,
        "card_rows": card_rows,
        "first_unscored_boundary": {
            "operator": "zdf_sh2",
            "owner": "GYRE_OWNER_SHARED_TKE",
            "source": "zdfphy.F90:264-270; zdfsh2.F90:80-100",
            "reason": (
                "the only ZDF stream is sampled after TKE, EVD, DDM, and IWM; "
                "it contains neither sh2 nor its pre-closure avm_k operand"
            ),
        },
        "required_write_only_frames": [
            {
                "site": "after zdf_sh2, immediately before zdf_tke",
                "fields": [
                    "sh2", "avm_k", "avt_k", "en_pre", "rn2", "rn2b",
                    "uu_Kbb", "uu_Kmm", "vv_Kbb", "vv_Kmm",
                    "e3uw_Kbb", "e3uw_Kmm", "e3vw_Kbb", "e3vw_Kmm",
                    "umask", "vmask", "wumask", "wvmask", "taum", "fr_i",
                    "rCdU_bot", "gdepw_Kmm", "e3t_Kmm", "e3w_Kmm",
                    "mbkt",
                ],
                "time_levels": "kt=1, Kbb=1, Kmm=1, Krhs=3",
            },
            {
                "site": "after zdf_tke, before copy to avm/avt and zdf_evd",
                "fields": ["en_post_tke", "avm_k", "avt_k", "p_pdlr"],
                "time_levels": "kt=1, Kbb=1, Kmm=1",
            },
            {
                "site": "before and after zdf_evd; after zdf_ddm",
                "fields": ["rn2", "rn2b", "avm", "avt", "avs"],
                "time_levels": "kt=1, Kmm=1, Krhs=3",
            },
            {
                "site": "inside zdf_iwm before coefficient updates and after return",
                "fields": [
                    "ebot_iwm", "ecri_iwm", "ensq_iwm", "esho_iwm",
                    "hbot_iwm", "hcri_iwm", "ht_Kmm", "gdept_Kmm",
                    "e3w_Kmm", "rn2", "zfact1", "zfact2", "zfact3",
                    "zfact4", "zReb", "zemx_iwm", "zav_wave",
                    "zav_ratio", "avm", "avt", "avs",
                ],
                "time_levels": "kt=1, Kmm=1",
            },
        ],
        "source_order": [
            "zdf_sh2", "zdf_drg", "zdf_mxl", "zdf_tke",
            "copy avm_k/avt_k", "river-mouth avt increment", "zdf_evd",
            "zdf_ddm", "zdf_iwm", "zdf_mxl_turb", "lbc_lnk(avm)",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.deck_root, args.oracle_root)
    except (AuditError, OSError, ValueError, struct.error) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
