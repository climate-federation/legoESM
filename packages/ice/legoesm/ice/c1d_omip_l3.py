"""Immutable legoESM column card for the NEMO C1D_OMIP_L3 oracle."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy
from legoesm.ice.config import (
    SeaIceConfig,
    validate_si3_bulk_config,
    validate_si3_thermo_config,
)
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from legoesm.coupler.ocean_forcing import NemoSI3ExchangeConfig

FORCING_SHA256 = "e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe"
THERMO_STREAM_SHA256 = "7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b"
ORACLE_VERSION = "V2_SCALAR_MATH"
ORACLE_V1_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_phase2_inputs"
)
ORACLE_V2_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/"
    "c1d_omip_l3_sasice_scalarmath_v2_a"
)
# Retain the superseded V1 pins explicitly; V2 happens to be byte-identical,
# but its separate build/root provenance is part of the oracle identity.
ORACLE_V1_EXCHANGE_STREAM_SHA256 = (
    "7f22ca914bc25890c7ccda810553eefae97625e437974c2208399e9afc5b2c50"
)
ORACLE_V1_ZDF_INPUT_STREAM_SHA256 = (
    "5522eadce595408b00065fa30d8b41fccb3815bee76d6fbf5ba3adbb2656cb27"
)
ORACLE_V1_ZDF_OPERAND_STREAM_SHA256 = (
    "587454974cb07590454d2bb63d745dc73488c220e84785175c9c110d26954632"
)
EXCHANGE_STREAM_SHA256 = "091395cf604e83d88fbf458c4ef76ac3df2d5a65dac9cdc502e224cc1d1af2e4"
STABLE_EXCHANGE_STREAM_SHA256 = "091395cf604e83d88fbf458c4ef76ac3df2d5a65dac9cdc502e224cc1d1af2e4"
BULK_STREAM_SHA256 = "57868f3212646bdf6b0c4add0f48701c0082718331bc76a153050c6d1d44dfe9"
ZDF_INPUT_STREAM_SHA256 = "cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd"
ZDF_OPERAND_STREAM_SHA256 = "aad46579fb2d2cc19299d7a25992802525603bb9858adf1e892ff5d85bf40442"
_DOCUMENTED_DURATION_STEPS = 8760  # C1D EXP_SASICE README; 365 days hourly


class C1DOMIPL3Card(NamedTuple):
    name: str
    dt_seconds: float
    nsteps: int
    forcing_path: Path
    oracle_root: Path
    precision_policy: PrecisionPolicy
    config: SeaIceConfig
    selector_sources: tuple[tuple[str, str], ...]


class C1DOMIPL3CoupledCard(NamedTuple):
    """Reviewed rung-3.6 construction layered on the existing SI3 card."""

    ice: C1DOMIPL3Card
    slab_depth_m: float
    ocean_dt_seconds: float
    ice_dt_seconds: float
    ice_cadence: int
    exchange: NemoSI3ExchangeConfig
    chlorophyll_path: Path
    chlorophyll_sha256: str
    weights_path: Path
    weights_sha256: str


def build_c1d_omip_l3_card(
    *,
    forcing_path: Path = Path(
        "/data/abyssal/dbalwada/nemo-inputs/C1D_v5.0.0/C1D_v5.0.0/"
        "ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc"
    ),
    oracle_root: Path = ORACLE_V2_ROOT,
) -> C1DOMIPL3Card:
    """Return the scope-exact 1-hour, one-category SI3 column card."""

    config = SeaIceConfig(
        dynamics="none",
        transport="none",
        n_categories=1,
        bulk_scheme="nemo_si3_constant",
        Cd_ice=1.0e-3,  # const-ok: ORCA1 namelist_cfg:140, rn_Cd_ia
        Ch_ice=1.0e-3,  # const-ok: ORCA1 namelist_cfg:142, rn_Ch_ia
        Ce_ice=1.0e-3,  # const-ok: ORCA1 namelist_cfg:141, rn_Ce_ia
        drag_ocean=5.0e-3,  # const-ok: ice ref:136; output:733
        snow_blow_exponent=0.66,  # const-ok: ice ref:141; output:735
        albedo_snow_dry=0.85,  # const-ok: ice ref:305; output:873
        albedo_snow_melt=0.75,  # const-ok: ice ref:306; output:874
        albedo_ice_dry=0.64,  # const-ok: ice ref:307; output:875
        albedo_ice_melt=0.53,  # const-ok: ice ref:308; output:876
        albedo_ice_pivot=1.0,  # const-ok: ice ref:311; output:879 [m]
        emissivity_ice=constants.emissivity_ice_nemo,
        thermo_scheme="si3_bl99",
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    validate_si3_thermo_config(config)
    validate_si3_bulk_config(config)
    return C1DOMIPL3Card(
        name="C1D_OMIP_L3/EXP_SASICE",
        dt_seconds=3600.0,
        nsteps=_DOCUMENTED_DURATION_STEPS,
        forcing_path=forcing_path,
        oracle_root=oracle_root,
        precision_policy=PrecisionPolicy.fp64(transcendentals="libm"),
        config=config,
        selector_sources=(
            ("jpl=1 HFN", "iceitd.F90:129-180; accepted output.namelist.ice"),
            ("BL99", "icethd.F90:148; icethd_zdf_bl99.F90:34-590"),
            ("nlay_i=nlay_s=3", "par_ice.F90; accepted ocean.output:615-627"),
            ("P07", "icethd_zdf_bl99.F90:261-275"),
            ("nn_icesal=2, rn_sinew=.75", "icethd_sal.F90:204-249; icethd_dh.F90:328-362"),
            ("ln_pnd=.false.", "icethd.F90:176-177; accepted output.namelist.ice"),
            ("ln_icedA=.false.", "icethd.F90:161; accepted output.namelist.ice"),
            ("aEVP (C1D-inert)", "icestp.F90:167-171; accepted ocean.output:843-846"),
            ("Prather (C1D-inert)", "icestp.F90:167-171; accepted ocean.output:861-862"),
            ("ridging/rafting (C1D-inert)", "icestp.F90:167-171; accepted ocean.output:820-837"),
        ),
    )


def build_c1d_omip_l3_coupled_card(
    *, oracle_root: Path = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l3/"
        "c1d_omip_l3_coupled10m_r13_oracle_i"
    ),
    chlorophyll_path: Path = Path(
        "/data/abyssal/dbalwada/ORCA1-omip/INPUTS/orca1_inputs/"
        "data_repository/input_fields/merged_ESACCI_BIOMER4V1R1_CHL_REG05.nc"
    ),
    weights_path: Path = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l3/"
        "c1d_omip_l3_coupled10m_r13_oracle_i/"
        "weights_reg05_C1D_OMIP_L3_bilinear.nc"
    ),
) -> C1DOMIPL3CoupledCard:
    """Return the indivisible 10 m, real-freshwater rung-3.6 card."""

    return C1DOMIPL3CoupledCard(
        ice=build_c1d_omip_l3_card(oracle_root=oracle_root),
        slab_depth_m=10.0,  # const-ok: user Decision 6 free construction parameter
        ocean_dt_seconds=3600.0,  # C1D hourly ocean step
        ice_dt_seconds=14400.0,  # ORCA1 nn_fsbc=4 times hourly ocean step
        ice_cadence=4,
        exchange=NemoSI3ExchangeConfig(),
        chlorophyll_path=chlorophyll_path,
        chlorophyll_sha256=(
            "f43c5a1e8ce75e52bfc8edfa4318e68215c76c4b252cb0fb70fa00b39a40d6fe"
        ),
        weights_path=weights_path,
        weights_sha256=(
            "715c51c4528feb7cb1f1d409584e5257d1b680f350682e2444b478217ec3b4dc"
        ),
    )


__all__ = (
    "C1DOMIPL3Card",
    "C1DOMIPL3CoupledCard",
    "FORCING_SHA256",
    "ORACLE_VERSION",
    "ORACLE_V1_ROOT",
    "ORACLE_V2_ROOT",
    "ORACLE_V1_EXCHANGE_STREAM_SHA256",
    "ORACLE_V1_ZDF_INPUT_STREAM_SHA256",
    "ORACLE_V1_ZDF_OPERAND_STREAM_SHA256",
    "THERMO_STREAM_SHA256",
    "EXCHANGE_STREAM_SHA256",
    "STABLE_EXCHANGE_STREAM_SHA256",
    "BULK_STREAM_SHA256",
    "ZDF_INPUT_STREAM_SHA256",
    "ZDF_OPERAND_STREAM_SHA256",
    "build_c1d_omip_l3_card",
    "build_c1d_omip_l3_coupled_card",
)
