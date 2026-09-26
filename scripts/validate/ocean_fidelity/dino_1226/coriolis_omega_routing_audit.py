"""#1455 next-action #1, STEP 0: which Earth does each Coriolis consumer see?

The DINO campaign's synthesis (docs/ocean/fidelity/dino_campaign_synthesis.md
sec 3 item 1) records that a THIRD of the measured vertex-Coriolis gap against
NEMO's own ``ff_f`` is not a discretisation convention at all but a uniform
-1.578e-05 -- legoESM's ``constants.Omega`` (7.292e-05, four significant
figures) against NEMO's ``phycst.F90`` value (7.2921150830e-05).  It also
records that the shipped DINO card ALREADY pins NEMO's rate as a top-level
``omega`` field, and that one consumer of the truncated literal was fixed
earlier in the campaign.  So the run may be carrying TWO rotation rates at
once, and which sites see which is a question about wiring, not physics.

THIS SCRIPT ANSWERS THAT BY INSTANTIATION, NOT BY READING THE SOURCE.  It
builds the SAME objects the campaign's own probes build -- the bridged
day-0 entry state from ``multistep_replay.build_replay_ic`` and the model
config from ``dino_lat_lon_model_config`` -- and then, for every enumerated
Coriolis-consuming site, RECOVERS the rotation rate that site actually
received.  For a scalar site that is the stored float; for an array site it
is the inversion omega_eff = f / (2 sin(phi)) evaluated on the site's OWN
latitudes at rows where |sin(phi)| is large enough that the inversion is
well-conditioned (the equator row is excluded by construction, and the
conditioning bound is printed next to every number).

CO-LOCATION.  Every recovered omega is printed together with the array it
came from, that array's shape, the latitude vector used as the divisor, and
the row window scored -- because the campaign's own canon says an f recovered
from a v-face array against T-point latitudes is a different quantity from
the same f recovered against face latitudes, and the two differ by exactly
the placement error this audit is meant to separate FROM the constant.

WHAT IS AND IS NOT SCORED.  This script scores the ROUTING (which constant
reaches which site).  It reports the placement gap (row-average f_v against
NEMO's ff_f) only after DIVIDING OUT the recovered rotation rate, so the two
halves of the pair never appear mixed in one number.  It runs no model step
and issues no verdict about the barotropic residual.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 LEGOESM_NEMO_E3T=both JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/\
coriolis_omega_routing_audit.py

Writes ``results/dino_1455/coriolis_omega_routing_audit.json``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from legoesm.ocean.constants_config import NEMO_OMEGA

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The inversion omega = f / (2 sin(phi)) blows up at the equator.  Score only
# rows whose |sin(phi)| clears this bound; the resulting worst-case relative
# amplification of a 1-ulp f error is printed, so the reported precision is
# never finer than the instrument.
_SIN_FLOOR = 0.10          # |sin(phi)| >= 0.1  <=>  |phi| >= 5.74 deg


def _omega_eff(f: np.ndarray, lat_rad: np.ndarray, *, name: str,
               lat_name: str) -> dict:
    """Recover omega from an f array against the latitudes it was built on.

    ``f`` is (n, m) or (n,); ``lat_rad`` is (n,).  Rows with
    ``|sin(lat)| < _SIN_FLOOR`` are dropped (ill-conditioned inversion) and
    the dropped count is reported rather than hidden.
    """
    f = np.asarray(f, dtype=np.float64)
    lat_rad = np.asarray(lat_rad, dtype=np.float64)
    if f.ndim == 2:
        # f on a lat-lon grid must not vary with longitude; measure it rather
        # than assume it, else a row reduction would hide zonal structure.
        zonal_spread = float(np.abs(f - f[:, :1]).max())
        f1d = f[:, 0]
    else:
        zonal_spread = 0.0
        f1d = f
    if f1d.shape != lat_rad.shape:
        raise SystemExit(
            f"FATAL: {name} has row count {f1d.shape} but {lat_name} has "
            f"{lat_rad.shape}; the inversion would be off by a row and the "
            "recovered omega would absorb the placement error.")
    s = np.sin(lat_rad)
    keep = np.abs(s) >= _SIN_FLOOR
    if not keep.any():
        raise SystemExit(f"FATAL: no rows of {name} clear the sin floor.")
    om = f1d[keep] / (2.0 * s[keep])
    return {
        "array": name,
        "shape": list(np.asarray(f).shape),
        "latitudes": lat_name,
        "rows_scored": int(keep.sum()),
        "rows_dropped_equatorial": int((~keep).sum()),
        "zonal_spread_abs": zonal_spread,
        "omega_median": float(np.median(om)),
        "omega_min": float(om.min()),
        "omega_max": float(om.max()),
        "omega_spread_rel": float((om.max() - om.min())
                                  / max(abs(np.median(om)), 1e-300)),
    }


#: Sites whose recovered rate is NOT a pure rotation rate: legoESM builds the
#: v-point f as a ROW AVERAGE, so the placement error rides along and the
#: inversion returns a blend.  They are audited and printed like any other
#: site, and excluded from the COUNT of distinct rates -- counting them would
#: report a discretisation convention as if it were a third planet.
_PLACEMENT_CONTAMINATED = (
    "geometry.f_v", "vertex_coriolis(grid)", "coriolis_at_faces -> f_v")


def _classify(om: float, omega_lego: float, omega_nemo: float,
              omega_phycst: float = NEMO_OMEGA) -> str:
    """Name the Earth a recovered rate belongs to, or refuse to name one.

    THREE references, not two, and the distinction is the point of the audit:
    the DINO card pins a SEVEN-FIGURE literal that NEMO uses only under
    ``key_cice``, while DINO takes the sidereal-day branch.  Calling the card's
    pin "NEMO's omega" would erase exactly the discrepancy this probe exists to
    surface -- so the pin is named as a pin and phycst's computed value is a
    separate reference.  (Adversarial review, round 2.)
    """
    # The phycst reference DEFAULTS to NEMO's real rate rather than to None.
    # With a None default a three-argument call silently dropped the reference
    # and resolved every NEMO-side rate to the card-pin label -- which is how
    # this file shipped a red test (adversarial review, round 2).  There is now
    # no argument list that turns the distinction off.
    refs = [("legoESM constants.Omega", omega_lego),
            ("NEMO card pin (7-figure key_cice literal)", omega_nemo),
            ("NEMO phycst.F90 2*pi/rsiday (what DINO runs)", omega_phycst)]
    # Most specific first: if the card's pin has been corrected to phycst's
    # value the two references coincide, and the phycst name is the true one.
    for tag, ref in reversed(refs):
        if abs(om - ref) <= 1e-9 * abs(ref):
            return tag
    return "NEITHER (unrecognised)"


def main() -> None:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    import jax.numpy as jnp  # noqa: F401  (imported for side-effect parity)
    import multistep_replay as mr

    from legoesm import constants as _C
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG

    omega_lego = float(_C.Omega)
    omega_nemo = float(NEMO_CONSTANTS_CONFIG.Omega)
    # NEMO's phycst.F90 non-key_cice branch computes omega = 2*pi/rsiday with
    # rsiday the sidereal day; NEMO_CONSTANTS_CONFIG carries the 7-figure
    # literal.  Both are printed so a reader can see which one this audit's
    # "NEMO" column means.
    # TRANSCRIBED, not a truncated copy -- a probe whose whole finding is
    # "transcribe the expression, do not paste a rounded literal" must not
    # paste a rounded literal (the 6-figure form is 6.3e-12 off).
    omega_phycst_full = float(NEMO_OMEGA)

    print("=" * 78)
    print("STEP 0 -- Coriolis rotation-rate ROUTING audit (instantiated)")
    print("=" * 78)
    print(f"  legoESM constants.Omega        = {omega_lego:.12e}")
    print(f"  NEMO_CONSTANTS_CONFIG.Omega    = {omega_nemo:.12e}")
    print(f"  NEMO phycst.F90 (2pi/rsiday)   = {omega_phycst_full:.12e}")
    print(f"  lego vs NEMO card rel gap      = "
          f"{(omega_lego - omega_nemo) / omega_nemo:+.6e}")
    print(f"  NEMO card vs phycst rel gap    = "
          f"{(omega_nemo - omega_phycst_full) / omega_phycst_full:+.6e}")
    print(f"  inversion conditioning: rows scored need |sin(phi)| >= "
          f"{_SIN_FLOOR}, i.e. |phi| >= "
          f"{np.degrees(np.arcsin(_SIN_FLOOR)):.2f} deg")

    # ---------------------------------------------------------------- build
    # EXACTLY the campaign's own entry point: every fidelity probe in this
    # directory, and the 90-day twin, build the geometry this way.
    g, br, cfg, st0 = mr.build_replay_ic()
    from legoesm.ocean.experiments.dino import dino_lat_lon_model_config
    mc, _phys = dino_lat_lon_model_config(br.geometry, cfg)
    geom = br.geometry

    sites: list[dict] = []

    def _scalar_site(name, where, value, note=""):
        rec = {"site": name, "where": where, "kind": "scalar",
               "omega_received": float(value),
               "earth": _classify(float(value), omega_lego, omega_nemo,
                                  omega_phycst_full),
               "note": note}
        sites.append(rec)
        return rec

    def _array_site(name, where, f, lat_rad, lat_name, note=""):
        rec = _omega_eff(f, lat_rad, name=name, lat_name=lat_name)
        rec.update({"site": name, "where": where, "kind": "array",
                    "omega_received": rec["omega_median"],
                    "earth": _classify(rec["omega_median"], omega_lego,
                                       omega_nemo, omega_phycst_full),
                    "note": note})
        sites.append(rec)
        return rec

    # -- the config scalars ------------------------------------------------
    _scalar_site(
        "DINOConfig.omega (the card's pin)",
        "ocean/experiments/dino.py DINO_RECIPES['nemo_dino_kamm_mlf']",
        cfg.omega, "top-level card field")
    _scalar_site(
        "LatLonCGridOceanConfig.omega (property)",
        "ocean/state.py LatLonCGridOceanConfig.omega",
        mc.omega, "read-through to constants.Omega")
    _scalar_site(
        "LatLonCGridOceanConfig.constants.Omega",
        "ocean/state.py ConstantsConfig.Omega",
        mc.constants.Omega,
        "GM/Redi Treguier f20 + EKE/GEOMETRIC call sites read this")
    # physics_with_constants explicitly permits physics.constants to be None,
    # so a getattr default of NaN here would manufacture a spurious extra
    # "Earth" AND emit a bare NaN token into the JSON (adversarial review,
    # round 2).  Absent is recorded as absent.
    _phys_c = getattr(mc.physics, "constants", None)
    if _phys_c is None:
        sites.append({"site": "physics block constants.Omega",
                      "where": "ocean/state.py physics_with_constants",
                      "kind": "scalar", "omega_received": None,
                      "earth": "N/A (physics carries no constants block)",
                      "note": "not propagated on this card"})
    else:
        _scalar_site(
            "physics block constants.Omega",
            "ocean/state.py physics_with_constants",
            _phys_c.Omega, "propagated set")
    _scalar_site(
        "geometry.omega (stored scalar)",
        "grids/latlon.py LatLonCGridGeometry.omega",
        geom.omega, "the rate the f_T/f_u/f_v arrays were built from")

    # -- the geometry arrays ----------------------------------------------
    # CO-LOCATION, stated explicitly: f_T sits at TRACER latitudes (geom.lat),
    # f_u at u-points which share the tracer latitude on a lat-lon grid, and
    # f_v at the V-FACE latitudes.  legoESM stores no lat_v on the geometry;
    # the face latitudes are NEMO's own gphiv, which the bridge used to BUILD
    # this geometry (nemo_state_bridge lat_face), so they are the correct and
    # non-circular divisor for f_v.
    lat_T = np.asarray(geom.lat, dtype=np.float64)
    gphit = np.deg2rad(np.asarray(g.gphit, dtype=np.float64)[:, 0])
    gphiv = np.deg2rad(np.asarray(g.gphiv, dtype=np.float64)[:, 0])
    lat_face = np.concatenate([[2.0 * gphit[0] - gphiv[0]], gphiv])

    # PREMISE, verified here rather than relayed: the bridged geometry's
    # tracer latitudes ARE NEMO's gphit.  If they were not, every omega
    # recovered below would absorb a latitude error.
    lat_gap_deg = float(np.degrees(np.abs(lat_T - gphit)).max())
    print(f"\n  premise: max|geom.lat - NEMO gphit| = {lat_gap_deg:.3e} deg "
          f"({'OK' if lat_gap_deg < 1e-12 else 'FAILS -- see below'})")
    if lat_gap_deg >= 1e-12:
        raise SystemExit(
            "FATAL: the bridged geometry's tracer latitudes are not NEMO's "
            f"gphit (max {lat_gap_deg:.3e} deg); the omega inversion would "
            "absorb a latitude error and this audit cannot separate the "
            "constant from the placement.")

    _array_site("geometry.f_T", "grids/latlon.py f_legacy",
                geom.f_T, lat_T, "geom.lat (= NEMO gphit)",
                "QG-Leith f_h; GM/Redi f_coriolis source")
    _array_site("geometry.f_u", "grids/latlon.py f_u",
                geom.f_u, lat_T, "geom.lat (= NEMO gphit)",
                "coriolis_at_faces -> semi-implicit + explicit_ab2 Coriolis")
    # f_v against the FACE latitudes.  The recovered omega here is NOT purely
    # the constant: legoESM builds f_v as a row AVERAGE of f_T, so the
    # placement error rides along.  It is reported as such, and separated
    # below.
    _array_site("geometry.f_v", "grids/latlon.py f_v (row average of f_T)",
                geom.f_v, lat_face, "NEMO gphiv face latitudes",
                "vertex_coriolis -> EEN barotropic + 3-D een/ene_total; "
                "CARRIES THE PLACEMENT ERROR, see the separation below")

    # -- the operator-level sites, obtained by CALLING the helpers ---------
    from legoesm.ocean.dynamics.latlon_cgrid_operators import vertex_coriolis
    from legoesm.ocean.dynamics.barotropic_common import coriolis_at_faces
    f_vtx = np.asarray(vertex_coriolis(geom), dtype=np.float64)
    _array_site("vertex_coriolis(grid)",
                "ocean/dynamics/latlon_cgrid_operators.py vertex_coriolis",
                f_vtx, lat_face, "NEMO gphiv face latitudes",
                "the array the EEN pre-block stores as f_vtx")
    f_u_op, f_v_op = coriolis_at_faces(geom, np.float64)
    _array_site("coriolis_at_faces -> f_u",
                "ocean/dynamics/barotropic_common.py coriolis_at_faces",
                np.asarray(f_u_op), lat_T, "geom.lat (= NEMO gphit)", "")
    _array_site("coriolis_at_faces -> f_v",
                "ocean/dynamics/barotropic_common.py coriolis_at_faces",
                np.asarray(f_v_op), lat_face, "NEMO gphiv face latitudes",
                "CARRIES THE PLACEMENT ERROR")

    # -- the table ---------------------------------------------------------
    ff_f = np.asarray(g.ff_f, dtype=np.float64)
    # WHICH EARTH IS NEMO'S OWN ff_f ON?  Recover it from NEMO's own arrays
    # -- ff_f against gphiv (the F-point and the v-face share a latitude on a
    # Mercator lat-lon grid).  This is the non-circular check that the "NEMO
    # omega" column above really is the rate the oracle integrated with, and
    # it simultaneously confirms the F-point latitude convention: a wrong
    # latitude would show up as a recovered rate that is not a clean constant.
    _ff_om = _omega_eff(ff_f, gphiv, name="NEMO ff_f",
                        lat_name="NEMO gphiv (F-point latitude)")
    print("\n  NEMO's own ff_f, inverted against its own gphiv:")
    print(f"    omega recovered = {_ff_om['omega_median']:.12e}  "
          f"(spread over rows {_ff_om['omega_spread_rel']:.2e} relative)")
    print(f"    vs phycst 2pi/rsiday  {(_ff_om['omega_median'] - omega_phycst_full) / omega_phycst_full:+.3e}")
    print(f"    vs the card's pin     {(_ff_om['omega_median'] - omega_nemo) / omega_nemo:+.3e}")
    print(f"    vs constants.Omega    {(_ff_om['omega_median'] - omega_lego) / omega_lego:+.3e}")
    _ff_om["site"] = "NEMO ff_f (the ORACLE's own array)"
    _ff_om["where"] = "RUN_TRAJ/mesh_mask.nc ff_f"
    _ff_om["kind"] = "array"
    _ff_om["omega_received"] = _ff_om["omega_median"]
    _ff_om["earth"] = _classify(_ff_om["omega_median"], omega_lego,
                                omega_nemo, omega_phycst_full)
    sites.append(_ff_om)

    print("\n" + "-" * 78)
    print("ROUTING TABLE -- the rate each site ACTUALLY received")
    print("-" * 78)
    print(f"  {'site':44s} {'omega received':>18s}  Earth")
    for rec in sites:
        print(f"  {rec['site'][:44]:44s} {rec['omega_received']:18.12e}  "
              f"{rec['earth']}")

    # COUNT THE RATES, NOT THE LABELS.  Every unrecognised value used to land
    # in one "NEITHER" bucket, so N genuinely different rates read as 1 and the
    # printed count was right only by luck (adversarial review, round 2).  The
    # placement-contaminated sites are excluded because their recovered value
    # is a blend of a rate and a discretisation convention, not a rate.
    _rate_sites = [r for r in sites
                   if r.get("omega_received") is not None
                   and r["site"] not in _PLACEMENT_CONTAMINATED
                   and r["site"] != "NEMO ff_f (the ORACLE's own array)"]
    _rates = sorted({float(f"{r['omega_received']:.15e}") for r in _rate_sites})
    earths = sorted({r["earth"] for r in _rate_sites})
    print(f"\n  DISTINCT ROTATION RATES REACHING legoESM's OWN SITES: "
          f"{len(_rates)}")
    for _r in _rates:
        _who = [x["site"] for x in _rate_sites
                if float(f"{x['omega_received']:.15e}") == _r]
        print(f"    {_r:.12e}  <- {len(_who)} site(s): {', '.join(_who[:3])}"
              + (" ..." if len(_who) > 3 else ""))
    print(f"    (the oracle's own ff_f, for comparison: "
          f"{_ff_om['omega_median']:.12e})")
    print(f"    labels in use: {earths}")
    print("    The three placement-contaminated sites (a row average, so the "
          "recovered value blends rate and convention) are audited above and "
          "EXCLUDED from this count.")

    # -- separate the constant from the placement -------------------------
    # NEMO's own ff_f is the oracle for the vertex Coriolis.  Divide BOTH
    # sides by the rotation rate each was built with, so what remains is the
    # pure placement (discretisation) gap and the constant never contaminates
    # it.  Row map: legoESM v row j+1 <-> NEMO f row j (the same -1 shift the
    # v-face metric arm calibrated); it is re-calibrated here against a known
    # answer rather than transferred.
    jpj, jpi = ff_f.shape
    f_v_lego = np.asarray(geom.f_v, dtype=np.float64)
    if f_v_lego.shape != (jpj + 1, jpi):
        raise SystemExit(f"FATAL: geom.f_v is {f_v_lego.shape}, want "
                         f"({jpj + 1}, {jpi})")
    ff_izonal = float(np.abs(ff_f - ff_f[:, :1]).max())
    if ff_izonal != 0.0:
        raise SystemExit(f"FATAL: NEMO ff_f varies along i by {ff_izonal:.3e}")

    def _shift_score(shift: int) -> float:
        j = np.arange(1, jpj + 1) + shift
        ok = (j >= 0) & (j < jpj)
        cand = ff_f[np.clip(j, 0, jpj - 1), 0]
        a = f_v_lego[1:jpj + 1, 0][ok]
        b = cand[ok]
        keep = np.abs(b) > 1e-6
        return float(np.median(np.abs(a[keep] - b[keep]) / np.abs(b[keep])))

    scores = {s: _shift_score(s) for s in (-2, -1, 0, 1, 2)}
    best_wrong = min(v for s, v in scores.items() if s != -1)
    print("\n  row-map calibration (lego v row j <- NEMO f row j+shift):")
    for s in sorted(scores):
        tag = "   <- the map used" if s == -1 else \
            f"   ({scores[s] / max(scores[-1], 1e-300):.0f}x worse)"
        print(f"    shift {s:+d}: median rel {scores[s]:.3e}{tag}")
    if not (scores[-1] < 1e-3 and scores[-1] * 100.0 < best_wrong):
        raise SystemExit(
            f"FATAL: the f row map is not established (chosen {scores[-1]:.3e},"
            f" best wrong {best_wrong:.3e}); the separation below would be "
            "reading the wrong rows against each other.")


    j = np.arange(1, jpj + 1) - 1
    ok = (j >= 0) & (j < jpj)
    lego_col = f_v_lego[1:jpj + 1, 0][ok]
    nemo_col = ff_f[j[ok], 0]
    lat_col = lat_face[1:jpj + 1][ok]
    keep = np.abs(np.sin(lat_col)) >= _SIN_FLOOR

    om_geom = float(geom.omega)
    total_rel = (lego_col - nemo_col) / nemo_col
    # Placement-only: rebuild legoESM's own construction on NEMO's rate, so
    # the constant divides out exactly and what remains is discretisation.
    lego_on_nemo = lego_col * (omega_nemo / om_geom)
    place_rel = (lego_on_nemo - nemo_col) / nemo_col
    const_rel = (om_geom - omega_nemo) / omega_nemo

    print("\n" + "-" * 78)
    print("THE VERTEX-CORIOLIS GAP, SPLIT (rows with |sin(phi)| >= "
          f"{_SIN_FLOOR}; n={int(keep.sum())})")
    print("-" * 78)
    print(f"  total    (as shipped)        median {np.median(total_rel[keep]):+.4e}"
          f"   RMS {np.sqrt((total_rel[keep] ** 2).mean()):.4e}")
    print(f"  constant (rotation rate)     uniform {const_rel:+.4e}")
    print(f"  placement (row average vs    median {np.median(place_rel[keep]):+.4e}"
          f"   RMS {np.sqrt((place_rel[keep] ** 2).mean()):.4e}")
    print("             f at the f-point)")
    resid = np.median(total_rel[keep]) - const_rel - np.median(place_rel[keep])
    print(f"  additivity residual (median) {resid:+.4e}   "
          "(second order in the two small gaps)")

    out = {
        "provenance": mr.provenance("coriolis_omega_routing_audit"),
        "omega_lego_constants": omega_lego,
        "omega_nemo_card": omega_nemo,
        "omega_nemo_phycst_full": omega_phycst_full,
        "sin_floor": _SIN_FLOOR,
        "rows_scored": int(keep.sum()),
        "sites": sites,
        "distinct_earths": earths,
        "distinct_rates_reaching_legoesm_sites": _rates,
        "n_distinct_rates": len(_rates),
        "oracle_own_rate": _ff_om["omega_median"],
        "placement_contaminated_sites_excluded_from_count":
            list(_PLACEMENT_CONTAMINATED),
        "row_map_scores": {str(k): v for k, v in scores.items()},
        "gap_split_relative": {
            "total_median": float(np.median(total_rel[keep])),
            "total_rms": float(np.sqrt((total_rel[keep] ** 2).mean())),
            "constant_uniform": float(const_rel),
            "placement_median": float(np.median(place_rel[keep])),
            "placement_rms": float(np.sqrt((place_rel[keep] ** 2).mean())),
        },
    }
    os.makedirs("results/dino_1455", exist_ok=True)
    path = "results/dino_1455/coriolis_omega_routing_audit.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {path}")
    print(f"  {out['provenance']}")


if __name__ == "__main__":
    main()
