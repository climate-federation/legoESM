#!/usr/bin/env python3
"""Which jpl=1 ridging / ice_cor kernel is NEMO-literal on lane 3's ICE_RHEO record.

Lane 3's exact-dynamics arm (``nemo_si3_phase2_round14_seed_probe``) starts from
the oracle step-8 entry and step-7 restart, replaces only the rheology output by
NEMO's, advances one production-JIT card step and scores the oracle step-9 entry.
This probe runs that arm three ways, changing one kernel at a time:

- ``lane3``: the card as committed (``apply_si3_jpl1_ridging`` + the card's ice_cor);
- ``o6_rdgrft``: ridging through the O6 kernel ``si3_ice_dyn_rdgrft`` at jpl=1;
- ``o6_rdgrft_cor``: also ice_cor through the O6 kernel ``si3_ice_cor_dyn``.

The adapters are harness glue (axis reshape and the run's resolved namelist), not
model code.  Every row is NEMO-vs-candidate, bitwise, over the 1000 x 1000 interior.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K
from legoesm.ice.fidelity import nemo_rheo_testcase_recipe as recipe
from legoesm.ice.itd import SI3IceCorConfig, SI3IceCorState, si3_ice_cor_dyn
from legoesm.ice.ridging import (
    SI3JPL1RidgingState,
    SI3RdgrftConfig,
    SI3RdgrftState,
    si3_ice_dyn_rdgrft,
    si3_rdgrft_config_after_init,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp

from legoesm import constants

HERE = Path(__file__).resolve().parent
SEED_PROBE = HERE / "nemo_si3_phase2_round14_seed_probe.py"
ARMS = ("lane3", "o6_rdgrft", "o6_rdgrft_cor")
# Oracle-zero receivers on this ln_icethd=F card (lane-3 receipt: UNINFORMATIVE).
UNINFORMATIVE = ("oa_i", "a_ip", "v_ip", "v_il")
# ICE_RHEO resolved namelist (final/output.namelist.ice, namelist_ice_cfg).
RN_HIMAX = 99.0
RN_HIMIN = 0.1
RN_AMAX = 0.997
RN_SIMIN = 0.1
RN_SINEW = 0.75
SSS_M = 35.0
RHOW = constants.rho_water  # phycst.F90:59 rhow = 1000


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rdgrft_config(card) -> SI3RdgrftConfig:
    """ICE_RHEO's resolved namdyn_rdgrft, through NEMO's ln_icethd=F override."""
    rc = card.ridging_config
    raw = SI3RdgrftConfig(
        rn_csrdg=rc.cs_ridging,
        ln_partf_exp=rc.exponential_participation,
        rn_astar=rc.astar,
        ln_distf_lin=not rc.exponential_distribution,
        ln_distf_exp=rc.exponential_distribution,
        rn_murdg=rc.mu_ridging,
        ln_ridging=rc.ridging,
        rn_hstar=rc.hstar_m,
        rn_porordg=0.0,
        rn_fsnwrdg=0.5,
        rn_fpndrdg=0.5,
        ln_rafting=rc.rafting,
        rn_hraft=rc.hraft_m,
        rn_craft=rc.craft,
        rn_fsnwrft=0.5,
        rn_fpndrft=0.5,
        ln_rhg_evp_or_vp=True,
        ln_pnd=True,
        ln_pnd_lids=True,
        nn_icesal=card.nn_icesal,
        rdt_ice=card.dt_s,
        rhoi=K.rho_ice,
        rhos=K.rho_snow,
        rhow=RHOW,
        rcp=K.c_ocean,
    )
    return si3_rdgrft_config_after_init(raw, ln_icethd=card.thermodynamics)


def cor_config(card) -> SI3IceCorConfig:
    return SI3IceCorConfig(
        rn_himin=RN_HIMIN,
        rn_simin=RN_SIMIN,
        rn_sinew=RN_SINEW,
        nn_icesal=card.nn_icesal,
        ln_pnd=True,
        ln_pnd_lids=True,
        rdt_ice=card.dt_s,
        rhoi=K.rho_ice,
        rhos=K.rho_snow,
        rhow=RHOW,
        rt0=constants.T_freeze,
    )


def o6_ridging_adapter(card, sst_m, *, hstar_scale: float = 1.0):
    """Drop-in for ``apply_si3_jpl1_ridging`` that runs the O6 kernel at jpl=1."""
    cfg = rdgrft_config(card)
    cfg = cfg._replace(rn_hstar=cfg.rn_hstar * hstar_scale)
    hi_max = jnp.asarray([0.0, RN_HIMAX])
    sss = jnp.full_like(sst_m, SSS_M)

    def run(state: SI3JPL1RidgingState, divergence, deformation, dt, *, config=None):
        del config
        assert dt == cfg.rdt_ice
        z = jnp.zeros_like(state.ice_area)
        c = lambda x: x[..., None]  # noqa: E731
        s = SI3RdgrftState(
            a_i=c(state.ice_area),
            v_i=c(state.ice_volume),
            v_s=c(state.snow_volume),
            oa_i=c(state.age_content),
            a_ip=c(state.pond_area),
            v_ip=c(state.pond_volume),
            v_il=c(state.pond_lid_volume),
            e_s=c(state.snow_enthalpy),
            e_i=c(state.ice_enthalpy),
            szv_i=c(state.ice_salt_content),
            at_i=state.ice_area,
            ato_i=state.open_water_area,
            wfx_dyn=z,
            sfx_dyn=z,
            hfx_dyn=z,
            wfx_snw_dyn=z,
            wfx_pnd=z,
        )
        out = si3_ice_dyn_rdgrft(
            s,
            divu_i=divergence,
            delta_i=deformation,
            sst_m=sst_m,
            sss_m=sss,
            hi_max=hi_max,
            nn_hls=card.halo_width,
            config=cfg,
        ).state
        return SI3JPL1RidgingState(
            ice_area=out.a_i[..., 0],
            open_water_area=out.ato_i,
            ice_volume=out.v_i[..., 0],
            snow_volume=out.v_s[..., 0],
            age_content=out.oa_i[..., 0],
            pond_area=out.a_ip[..., 0],
            pond_volume=out.v_ip[..., 0],
            pond_lid_volume=out.v_il[..., 0],
            snow_enthalpy=out.e_s[..., 0],
            ice_enthalpy=out.e_i[..., 0],
            ice_salt_content=out.szv_i[..., 0],
        ), None

    return run


def o6_cor_adapter(sst_m):
    """Drop-in for the card's ``_ice_cor`` that runs ``si3_ice_cor_dyn`` at jpl=1."""
    index = recipe.ICE_RHEO_TRACERS.index

    def layers(contents, name, n):
        return jnp.stack([contents[..., index(f"{name}_l{k:02d}")] for k in range(1, n + 1)], -1)

    def run(card, contents, t_surface):
        cfg = cor_config(card)
        f = lambda n: contents[..., index(n)][..., None]  # noqa: E731
        z = jnp.zeros_like(t_surface)
        e_s, e_i, szv = (
            layers(contents, "e_s", 5),
            layers(contents, "e_i", 10),
            layers(contents, "szv_i", 10),
        )
        st = SI3IceCorState(
            a_i=f("a_i"),
            v_i=f("v_i"),
            v_s=f("v_s"),
            oa_i=f("oa_i"),
            a_ip=f("a_ip"),
            v_ip=f("v_ip"),
            v_il=f("v_il"),
            e_s=e_s[..., None],
            e_i=e_i[..., None],
            szv_i=szv[..., None],
            sz_i=jnp.zeros_like(szv)[..., None],
            t_su=t_surface[..., None],
            t_i=jnp.zeros_like(e_i)[..., None],
            t_s=jnp.zeros_like(e_s)[..., None],
            h_i=z[..., None],
            h_s=z[..., None],
            h_ip=z[..., None],
            h_il=z[..., None],
            at_i=z,
            vt_i=z,
            ato_i=z,
            sfx_res=z,
            hfx_res=z,
            wfx_res=z,
        )
        out = si3_ice_cor_dyn(
            st,
            rn_amax_2d=jnp.full_like(z, RN_AMAX),
            sss_m=jnp.full_like(z, SSS_M),
            sst_m=sst_m,
            hi_max=jnp.asarray([0.0, RN_HIMAX]),
            hi_mean=jnp.asarray([0.5 * RN_HIMAX]),
            nn_hls=card.halo_width,
            config=cfg,
        )
        for name in ("a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il"):
            contents = contents.at[..., index(name)].set(getattr(out, name)[..., 0])
        for name, n in (("e_s", 5), ("e_i", 10), ("szv_i", 10)):
            for k in range(1, n + 1):
                contents = contents.at[..., index(f"{name}_l{k:02d}")].set(
                    getattr(out, name)[..., k - 1, 0]
                )
        return recipe._replace_periodic_halo(contents), recipe._replace_periodic_halo(
            out.t_su[..., 0]
        )

    return run


def run_arm(seed, card, state, target, arm: str, sst_m, *, hstar_scale: float = 1.0):
    saved_rdg, saved_cor = recipe.apply_si3_jpl1_ridging, recipe._ice_cor
    try:
        if arm in ("o6_rdgrft", "o6_rdgrft_cor"):
            recipe.apply_si3_jpl1_ridging = o6_ridging_adapter(card, sst_m, hstar_scale=hstar_scale)
        if arm == "o6_rdgrft_cor":
            recipe._ice_cor = o6_cor_adapter(sst_m)
        out = seed._exact_dynamics_arm(card, state, target)
    finally:
        recipe.apply_si3_jpl1_ridging, recipe._ice_cor = saved_rdg, saved_cor
    return seed._entry_rows(card, out, target), out


def summarize(rows) -> dict[str, object]:
    informative = [r for r in rows if not str(r["name"]).endswith(UNINFORMATIVE)]
    exact = [r for r in informative if r["status"] == "BIT-EXACT"]
    return {
        "informative_rows": len(informative),
        "bit_exact_rows": len(exact),
        "nonexact": {
            str(r["name"]).split(".")[-1]: r["bitwise_nonzero_over_n"]
            for r in informative
            if r["status"] != "BIT-EXACT"
        },
    }


def run_probe(root: Path, restart_root: Path, *, plant: bool = False) -> tuple[dict, int]:
    seed = _load("nemo_si3_round14_seed_probe_o6", SEED_PROBE)
    if jax.default_backend() != "cpu":
        raise RuntimeError("lane-3 kernel probe requires CPU")
    ot = seed.oracle_gate
    entry_header, entry = ot.read_frame(root / f"oracle_ice_step_entry_kt{seed.ENTRY_STEP:08d}.bin")
    target_header, target = ot.read_frame(
        root / f"oracle_ice_step_entry_kt{seed.TARGET_STEP:08d}.bin"
    )
    assert entry_header["kt"] == seed.ENTRY_STEP and target_header["kt"] == seed.TARGET_STEP
    with netCDF4.Dataset(root / "mesh_mask.nc") as ds:
        mesh = {n: np.asarray(ds[n][:]) for n in seed.rung34.MESH_FIELDS}
    with netCDF4.Dataset(root / "output.init_ice.nc") as ds:
        sst_c = np.asarray(ds["sst"][0])
    card = recipe.build_ice_rheo_card(entry, mesh, sst_c + constants.T_freeze)
    entry_restart = restart_root / f"ICE_RHEO_OMIP_L3_{seed.ENTRY_RESTART_STEP:08d}_restart_ice.nc"
    state = card.initial_state._replace(
        moments=seed.trajectory._restart_moments(entry_restart, card)
    )
    sst_m = recipe._periodic_halo(sst_c.T)
    hstar_scale = 1.0 + 1.0e-10 if plant else 1.0  # plant: O6 rn_hstar moved by 1e-10
    arms = {}
    for arm in ARMS:
        rows, _ = run_arm(seed, card, state, target, arm, sst_m, hstar_scale=hstar_scale)
        arms[arm] = {"summary": summarize(rows), "rows": rows}
    exact = {
        a: v["summary"]["bit_exact_rows"] == v["summary"]["informative_rows"]
        for a, v in arms.items()
    }
    report = {
        "format": "nemo-si3-o6-lane3-kernel-probe-v1",
        "worktree": worktree_stamp(),
        "execution": "CPU/fp64 production JIT; lane-3 exact-dynamics arm (oracle step-8 rheology)",
        "clock": {"oracle_entry": seed.ENTRY_STEP, "oracle_target_entry": seed.TARGET_STEP},
        "plant": plant,
        "arms": arms,
        "all_informative_rows_bit_exact": exact,
    }
    code = 0 if (exact["lane3"] and not plant) or (plant and not exact["o6_rdgrft"]) else 1
    return report, code


def main() -> int:
    seed_defaults = _load("nemo_si3_round14_seed_probe_o6_defaults", SEED_PROBE)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=seed_defaults.ROOT)
    parser.add_argument("--restart-root", type=Path, default=seed_defaults.RESTART_ROOT)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    report, code = run_probe(args.root, args.restart_root, plant=args.plant)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(json.dumps({a: v["summary"] for a, v in report["arms"].items()}, indent=2))
    if args.artifact:
        args.artifact.write_text(payload + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
