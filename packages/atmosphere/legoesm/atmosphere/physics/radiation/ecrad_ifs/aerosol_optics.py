"""ecRad 49r1 aerosol optics, IFS-RRTMG configuration (JAX port).

Port of ``radiation_aerosol_optics.F90:add_aerosol_optics`` (OpenIFS 49r1,
8f6f722, compiled file lines 478-817) for the IFS configuration only:
mixing-ratio aerosols (``is_direct = false``), RRTMG gas optics so
``do_cloud_aerosol_per_sw/lw_g_point = false`` (per-band tables mapped to
g-points, radiation_config.F90:1053-1063), ``do_lw_aerosol_scattering =
false`` (NLWSCATTERING=1, su_radiation_scheme.F90:582-593: LW optical depth
weighted by the single-scattering co-albedo) and
``do_sw_delta_scaling_with_gases = false`` (delta-Eddington on the aerosol
alone, ``radiation_delta_eddington.h:delta_eddington_extensive_vec``).  No
other configuration is implemented.

The per-band tables are the arrays the 49r1 oracle holds after its own setup
(``setup_general_aerosol_optics`` averaging aerosol_ifs_49R1_20230119.nc to
the RRTMG bands; the 48r1 file stands in for the 49r1 name
aerosol_ifs_49R1_20230725.nc, a declared departure), plus the 49r1 type map
the IFS set-up computes (su_radiation_scheme.F90:426-485), exported by
``scripts/validate/oracle/ecrad_ifs/export_aerosol_optics_npz.py`` and
vendored as ``data/aerosol_optics_48r1.npz`` (vote A3a).

Layout is column first, top-down: ``od_sw``/``ssa_sw`` ``(ncol, nlev, 112)``,
``od_lw`` ``(ncol, nlev, 140)``, ``mixing_ratio`` ``(ncol, nlev, ntype)``.

Smoothing (votes A6-A8, B4): one static ``sharpness`` (``math.inf`` = the
oracle's hard form exactly).
* RH bin (``calc_rh_index``): the bin index is ``1 + #{j >= 2: rh >
  rh_lower(j)}``; each edge is a strict :func:`.two_stream.step` of scale
  :data:`RH_EDGE_SCALE`, and the table rows are blended with the telescoping
  weights (one-hot at ``sharpness = inf``).
* ``max(scat_od, 1e-24)`` in delta-Eddington: :func:`.two_stream.lower_clamp`
  with scale = the floor (the fp32 literal, as the oracle promotes it).
* The SW combine gates ``local_od > 0 .and. od_aer > 0`` and ``local_scat >
  0`` are dropped (vote A6a): the combining formulae are applied everywhere
  with where-before-divide guards; where the oracle's gate is false the
  formulae return the gas values to rounding (``od_aer = 0`` exactly).
* Smooth form only: ratio denominators below sqrt(finfo(dtype).tiny) count as
  zero, so reverse-mode AD cannot overflow in fp32 (vote D2a, cloud-optics CO8).
"""

from __future__ import annotations

import functools
import json
import os
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_solvers as ms
from legoesm.atmosphere.physics.radiation.ecrad_ifs.two_stream import lower_clamp, step

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 add_aerosol_optics (IFS, RRTMG bands, no LW aerosol "
        "scattering): per-type mass-extinction optical depth of aerosol mixing "
        "ratios, RH-binned hydrophilic tables, delta-Eddington of the aerosol, "
        "combined with the gas optics per g-point."
    ),
    "inputs": {
        "od_sw, ssa_sw": "1 (ncol, nlev, 112) gas optics",
        "od_lw": "1 (ncol, nlev, 140) gas optics",
        "pressure_hl": "Pa (ncol, nlev+1) top-down",
        "h2o_mmr": "kg/kg (ncol, nlev)",
        "h2o_sat_liq": "kg/kg (ncol, nlev) saturation w.r.t. liquid",
        "mixing_ratio": "kg/kg (ncol, nlev, ntype) aerosol mass mixing ratio",
    },
    "outputs": {
        "od_sw, ssa_sw, g_sw": "1 (ncol, nlev, 112) gas+aerosol",
        "od_lw": "1 (ncol, nlev, 140) gas+aerosol absorption",
    },
    "sign_convention": (
        "od >= 0, 0 <= ssa <= 1; aerosol adds optical depth, never removes it "
        "(delta-Eddington removes forward-scattered part only). Optical "
        "properties only (no budget)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) ifs-source/radiation/radiation_aerosol_optics.F90 "
        "add_aerosol_optics, radiation_aerosol_optics_data.F90 calc_rh_index, "
        "radiation_delta_eddington.h; Bozzo et al. (2020) GMD 13."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_aerosol_oracle.py: matches the 49r1 "
        "stage dump around add_aerosol_optics to 1e-12 (fp64, hard form)."
    ),
}

# --- ecRad constants ---
# OneOverAccelDueToGravity = 1/AccelDueToGravity (radiation_aerosol_optics.F90:494,
# radiation_constants.F90:26): the oracle MULTIPLIES by the reciprocal.
_ONE_OVER_G = 1.0 / constants.g_ecrad
# max(scat_od, 1.0e-24) in delta_eddington_extensive_vec: a DEFAULT-REAL
# (single-precision) literal in the oracle (radiation_delta_eddington.h), promoted.
_DE_SCAT_FLOOR = float(np.float32(1.0e-24))

# --- smoothing scales (vote B4: interim, replaced by the per-quantity widths) ---
RH_EDGE_SCALE = 1.0          # relative humidity is a fraction (port "fractions" rule)

_DATA = os.path.join(os.path.dirname(__file__), "data")
TABLE_FILE = os.path.join(_DATA, "aerosol_optics_48r1.npz")
PROVENANCE_FILE = os.path.join(_DATA, "aerosol_optics_48r1.json")


class AerosolTables(NamedTuple):
    """Per user type, RH-binned (hydrophobic rows repeated over the bins)."""
    mass_ext_sw: np.ndarray   # (ntype, nrh, 14) m2/kg
    ssa_sw: np.ndarray        # (ntype, nrh, 14)
    g_sw: np.ndarray          # (ntype, nrh, 14)
    mass_ext_lw: np.ndarray   # (ntype, nrh, 16) m2/kg
    ssa_lw: np.ndarray        # (ntype, nrh, 16)
    rh_lower: np.ndarray      # (nrh,)
    active: np.ndarray        # (ntype,) bool: class hydrophobic/hydrophilic


@functools.lru_cache(maxsize=1)
def load_raw():
    """The vendored npz as a dict of numpy arrays (Fortran shapes)."""
    if not os.path.exists(TABLE_FILE):
        raise FileNotFoundError(f"aerosol optics table file missing: {TABLE_FILE}")
    with np.load(TABLE_FILE) as z:
        return {k: z[k] for k in z.files}


def provenance():
    with open(PROVENANCE_FILE) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def load_tables() -> AerosolTables:
    """Map the file types to the user types with the 49r1 type map
    (radiation_aerosol_optics_data.F90 set_types: >0 hydrophobic index,
    <0 hydrophilic index, 0 ignored)."""
    z = load_raw()
    nrh = z["rh_lower"].shape[0]
    rows = {k: [] for k in ("mass_ext_sw", "ssa_sw", "g_sw", "mass_ext_lw", "ssa_lw")}
    active = []
    for m in z["type_map"].astype(np.int64):
        for k in rows:
            band, var = k.rsplit("_", 1)[1], k.rsplit("_", 1)[0]
            if m > 0:
                t = z[f"{var}_{band}_phobic"][:, m - 1]                    # (nband,)
                rows[k].append(np.broadcast_to(t[None, :], (nrh, t.shape[0])))
            elif m < 0:
                rows[k].append(z[f"{var}_{band}_philic"][:, :, -m - 1].T)  # (nrh, nband)
            else:
                nb = z[f"{var}_{band}_phobic"].shape[0]
                rows[k].append(np.zeros((nrh, nb)))
        active.append(m != 0)
    return AerosolTables(**{k: np.ascontiguousarray(np.stack(v)) for k, v in rows.items()},
                         rh_lower=np.asarray(z["rh_lower"], np.float64),
                         active=np.asarray(active))


def background_mass_ext():
    """550-nm dry mass extinction (m2/kg) and one-based type index of the
    tropospheric and stratospheric background types, as su_radiation_scheme.F90
    computes them (:723-745), keyword-ready for :func:`.ifs_aerosol.macc_mixing_ratio`."""
    z = load_raw()
    return dict(trop_bg_mass_ext=float(z["trop_bg_mass_ext"]),
                strat_bg_mass_ext=float(z["strat_bg_mass_ext"]),
                itype_trop_bg=int(z["itype_trop_bg"]), itype_strat_bg=int(z["itype_strat_bg"]))


def rh_bin_weights(rh, rh_lower, sharpness):
    """calc_rh_index (radiation_aerosol_optics_data.F90:634-658) as bin weights
    (..., nrh): one-hot at the oracle index when sharpness = inf."""
    nrh = rh_lower.shape[0]
    above = [jnp.ones_like(rh)] + [step(rh, rh_lower[j], RH_EDGE_SCALE, sharpness, True)
                                   for j in range(1, nrh)] + [jnp.zeros_like(rh)]
    return jnp.stack([above[i] - above[i + 1] for i in range(nrh)], axis=-1)


def _tiny(x, sharpness):
    """Denominator threshold: 0 in the hard form; sqrt(finfo.tiny) when smoothed, so
    den**2 stays normal in reverse-mode AD (vote D2a, the cloud-optics rule CO8)."""
    return 0.0 if _hard(sharpness) else float(np.sqrt(np.finfo(np.dtype(x.dtype)).tiny))


def _ratio(num, den, sharpness):
    """num/den where den > threshold, else 0 (where-before-divide, votes A6a, D2a)."""
    ok = den > _tiny(den, sharpness)
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def add_aerosol_optics(od_sw, ssa_sw, od_lw, pressure_hl, h2o_mmr, h2o_sat_liq,
                       mixing_ratio, sharpness=100.0, tables=None):
    """add_aerosol_optics (see module docstring).  Returns (od_sw, ssa_sw, g_sw, od_lw)."""
    t = load_tables() if tables is None else tables
    if mixing_ratio.shape[-1] != t.mass_ext_sw.shape[0]:
        raise ValueError(f"mixing_ratio has {mixing_ratio.shape[-1]} aerosol types, "
                         f"the 49r1 type map has {t.mass_ext_sw.shape[0]}")
    # rh w.r.t. liquid and its bin (:619-620); layer mass (:622-624)
    w = rh_bin_weights(h2o_mmr / h2o_sat_liq, t.rh_lower, sharpness)          # (c, l, nrh)
    factor = (pressure_hl[..., 1:] - pressure_hl[..., :-1]) * _ONE_OVER_G     # (c, l)

    def pick(table, jtype):        # table (ntype, nrh, nband) -> (c, l, nband)
        return jnp.einsum("clr,rb->clb", w, table[jtype]) if not _hard(sharpness) \
            else jnp.take(table[jtype], jnp.argmax(w, axis=-1), axis=0)

    od_a = scat_a = scat_g_a = od_lw_a = 0.0
    for jtype in range(t.mass_ext_sw.shape[0]):           # sequential, as the oracle
        if not t.active[jtype]:
            continue                                      # class ignored: adds nothing (:712-713)
        mr = (factor * mixing_ratio[..., jtype])[..., None]
        local_od = mr * pick(t.mass_ext_sw, jtype)                            # :639-640
        ssa = pick(t.ssa_sw, jtype)
        od_a = od_a + local_od
        scat_a = scat_a + local_od * ssa
        scat_g_a = scat_g_a + local_od * ssa * pick(t.g_sw, jtype)
        od_lw_a = od_lw_a + mr * pick(t.mass_ext_lw, jtype) * (1.0 - pick(t.ssa_lw, jtype))  # :663-668

    # delta_eddington_extensive_vec (:723)
    g = _ratio(scat_g_a, lower_clamp(scat_a, _DE_SCAT_FLOOR, _DE_SCAT_FLOOR, sharpness), sharpness)
    f = g * g
    od_a = od_a - scat_a * f
    scat_a = scat_a * (1.0 - f)
    scat_g_a = scat_a * g / (1.0 + g)

    # SW combine per g-point through the band map (:744-762; gates dropped, A6a)
    band_sw = ms.band_from_g_sw()
    od_ab, scat_ab, scat_g_ab = od_a[..., band_sw], scat_a[..., band_sw], scat_g_a[..., band_sw]
    local_od = od_sw + od_ab
    local_scat = ssa_sw * od_sw + scat_ab
    g_sw = _ratio(scat_g_ab, local_scat, sharpness)   # oracle g_sw = 0 where its gates are false
    pos = local_od > _tiny(local_od, sharpness)       # local_od = 0: oracle leaves ssa_sw unchanged
    ssa_out = jnp.where(pos, local_scat / jnp.where(pos, local_od, 1.0), ssa_sw)
    # LW, no aerosol scattering (:801-806)
    od_lw_out = od_lw + od_lw_a[..., ms.band_from_g_lw()]
    return local_od, ssa_out, g_sw, od_lw_out


def _hard(sharpness):
    return sharpness == float("inf")
