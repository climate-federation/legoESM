#!/usr/bin/env python3
"""Enumerate every namelist variable read by the compiled NEMO SWE engine."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/"
    "SWG_OMIP_L1/BLD/ppsrc/nemo"
)

# The executed initialization path is compiled nemogcm.f90:332-352.  These
# are the 13 NAMELIST declarations reached by that path and enumerated by the
# operator's second adversarial review.  dyn_ldf_init delegates to ldf_dyn_init
# instead of reading a second group (compiled dynldf.f90:120-121).
SOURCE_SPECS: tuple[tuple[str, str], ...] = (
    ("namctl", "nemogcm.f90"),
    ("namcfg", "nemogcm.f90"),
    ("namrun", "domain.f90"),
    ("namdom", "domain.f90"),
    ("namtile", "domain.f90"),
    ("namnc4", "domain.f90"),
    ("namlbc", "dommsk.f90"),
    ("nambdy", "dommsk.f90"),
    ("nam_vvl", "domqco.f90"),
    ("namsbc", "sbcmod.f90"),
    ("namdyn_adv", "dynadv.f90"),
    ("namdyn_vor", "dynvor.f90"),
    ("namdyn_ldf", "ldfdyn.f90"),
)

# NEMO spellings map one-to-one to explicit card fields except for unit-bearing
# card names and the public SBC alias.  This frozen list is independent of the
# NamedTuple under test: removing any card field remains observable.
EXPECTED_CARD_READS: dict[str, tuple[str, ...]] = {
    "namctl": (
        "sn_cfctl", "ln_timing", "ln_diacfl", "nn_isplt", "nn_jsplt",
        "nn_ictls", "nn_npfchk", "nn_ictle", "nn_jctls", "nn_jctle",
    ),
    "namcfg": (
        "ln_read_cfg", "cn_domcfg", "ln_closea", "ln_write_cfg",
        "cn_domcfg_out",
    ),
    "namrun": (
        "cn_ocerst_indir", "cn_ocerst_outdir", "nn_stocklist", "ln_rst_list",
        "nn_no", "cn_exp", "cn_ocerst_in", "cn_ocerst_out", "ln_rstart",
        "nn_rstctl", "nn_it000", "nn_itend", "nn_date0", "nn_time0",
        "nn_leapy", "nn_istate", "nn_stock", "nn_write", "ln_mskland",
        "ln_clobber", "nn_chunksz", "ln_1st_euler", "ln_cfmeta",
        "ln_xios_read", "nn_wxios", "ln_top",
    ),
    "namdom": ("rn_Dt_s", "rn_atfp", "ln_c1d", "ln_meshmask", "ln_shuman"),
    "namtile": ("ln_tile", "nn_ltile_i", "nn_ltile_j"),
    "namnc4": ("nn_nchunks_i", "nn_nchunks_j", "nn_nchunks_k", "ln_nc4zip"),
    "namlbc": ("rn_shlat", "ln_vorlat"),
    "nambdy": (
        "ln_bdy", "nb_bdy", "ln_coords_file", "cn_coords_file",
        "ln_mask_file", "cn_mask_file", "cn_dyn2d", "nn_dyn2d_dta",
        "cn_dyn3d", "nn_dyn3d_dta", "cn_tra", "nn_tra_dta", "ln_tra_dmp",
        "ln_dyn3d_dmp", "rn_time_dmp_days", "rn_time_dmp_out_days", "cn_ice",
        "nn_ice_dta", "ln_vol", "nn_volctl", "nn_rimwidth",
    ),
    "nam_vvl": (
        "ln_vvl_zstar", "ln_vvl_ztilde", "ln_vvl_layer",
        "ln_vvl_ztilde_as_zstar", "ln_vvl_zstar_at_eqtor", "rn_ahe3",
        "rn_rst_e3t_days", "rn_lf_cutoff_days", "rn_zdef_max", "ln_vvl_dbg",
        "nn_vvl_interp",
    ),
    "namsbc": (
        "nn_fsbc", "ln_usr_sbc", "ln_flx", "ln_blk", "ln_abl", "ln_cpl",
        "ln_mixcpl", "nn_components", "nn_ice", "ln_ice_embd", "ln_traqsr",
        "ln_dm2dc", "ln_rnf", "nn_fwb", "ln_ssr", "ln_apr_dyn", "ln_wave",
        "nn_lsm",
    ),
    "namdyn_adv": (
        "ln_dynadv_OFF", "ln_dynadv_vec", "nn_dynkeg", "ln_dynadv_cen2",
        "ln_dynadv_up3",
    ),
    "namdyn_vor": (
        "ln_dynvor_ens", "ln_dynvor_ene", "ln_dynvor_enT", "ln_dynvor_een",
        "nn_e3f_typ", "ln_dynvor_mix", "ln_dynvor_msk",
    ),
    "namdyn_ldf": (
        "ln_dynldf_OFF", "nn_dynldf_typ", "ln_dynldf_lap", "ln_dynldf_blp",
        "ln_dynldf_lev", "ln_dynldf_hor", "ln_dynldf_iso", "nn_ahm_ijk_t",
        "rn_Uv_m_s", "rn_Lv_m", "rn_ahm_b_m2_s", "rn_csmc", "rn_minfac",
        "rn_maxfac",
    ),
}

_ALIASES = {
    "rn_dt": "rn_Dt_s",
    "rn_time_dmp": "rn_time_dmp_days",
    "rn_time_dmp_out": "rn_time_dmp_out_days",
    "rn_rst_e3t": "rn_rst_e3t_days",
    "rn_lf_cutoff": "rn_lf_cutoff_days",
    "ln_usr": "ln_usr_sbc",
    "ln_dynadv_off": "ln_dynadv_OFF",
    "ln_dynvor_ent": "ln_dynvor_enT",
    "ln_dynldf_off": "ln_dynldf_OFF",
    "rn_uv": "rn_Uv_m_s",
    "rn_lv": "rn_Lv_m",
    "rn_ahm_b": "rn_ahm_b_m2_s",
}


def parse_namelist_declaration(path: Path, group: str) -> tuple[str, ...]:
    """Return names from one preprocessed Fortran ``NAMELIST/group/`` line."""
    needle = f"namelist/{group}/"
    lines = path.read_text().splitlines()
    for index, raw in enumerate(lines):
        code = raw.split("!", 1)[0]
        lower = code.lower()
        if needle not in lower:
            continue
        parts = [code[lower.index(needle) + len(needle):]]
        while parts[-1].rstrip().endswith("&"):
            index += 1
            if index >= len(lines):
                raise ValueError(f"unterminated {group} declaration in {path}")
            parts.append(lines[index].split("!", 1)[0])
        joined = " ".join(parts).replace("&", " ")
        names = tuple(name.strip().lower() for name in joined.split(",") if name.strip())
        if not names:
            raise ValueError(f"empty {group} declaration in {path}")
        return names
    raise ValueError(f"missing NAMELIST/{group}/ in {path}")


def compiled_card_inventory(source_root: Path) -> dict[str, tuple[str, ...]]:
    """Parse and convert all 13 compiled declarations to card-field names."""
    inventory = {}
    for group, filename in SOURCE_SPECS:
        raw = parse_namelist_declaration(source_root / filename, group)
        inventory[group] = tuple(_ALIASES.get(name, name) for name in raw)
    return inventory


def validate_inventory(inventory: dict[str, tuple[str, ...]]) -> None:
    if inventory != EXPECTED_CARD_READS:
        missing = {
            group: sorted(set(EXPECTED_CARD_READS[group]) - set(inventory.get(group, ())))
            for group in EXPECTED_CARD_READS
        }
        extra = {
            group: sorted(set(inventory.get(group, ())) - set(EXPECTED_CARD_READS[group]))
            for group in EXPECTED_CARD_READS
        }
        raise ValueError(f"compiled inventory mismatch: missing={missing}, extra={extra}")
    fields = [field for names in inventory.values() for field in names]
    if len(fields) != 131 or len(set(fields)) != 131:
        raise ValueError(
            f"compiled inventory is not 131 unique reads: total={len(fields)} "
            f"unique={len(set(fields))}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--plant-drop", choices=tuple(
        field for names in EXPECTED_CARD_READS.values() for field in names))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        inventory = compiled_card_inventory(args.source_root)
        if args.plant_drop:
            inventory = {
                group: tuple(field for field in fields if field != args.plant_drop)
                for group, fields in inventory.items()
            }
        validate_inventory(inventory)
    except Exception as exc:  # noqa: BLE001 - fail-closed validation CLI
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    result = {
        "groups": len(inventory),
        "total": sum(len(fields) for fields in inventory.values()),
        "inventory": inventory,
    }
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"PASS groups={result['groups']} total={result['total']} unique=131")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
