"""Field-by-field audit of the ORCA1 zdftke card against the TKEConfig defaults.

WHY.  In one session, FIVE NEMO-faithful options turned out to be implemented,
documented, and never selected by ``orca1_zdftke_config``: the K amplitude
(``kappa_convention``), NEMO's nn_mxl=2 (``tke_mxl_choice=4``), the face-native
shear production with NEMO's coastal doubling
(``tke_shear_production="nemo_face_native"``), the z=0 surface-BC placement
(``tke_surface_bc_level="nemo_z0"``), the TKE diffusion coefficient
(``alpha_tke``), and NEMO's dissipation Newton split
(``dissipation_discretization="nemo_1p5_split"``).  Two of those were worth
1.4x and 30x in the diffusivity.  Finding them one at a time, each costing a
GPU arm, is the wrong process.

WHAT THIS DOES.  Prints EVERY ``TKEConfig`` field in three groups:

  SET      the card overrides the dataclass default -- a deliberate mapping,
           printed with both values so the mapping can be re-checked.
  DEFAULT  the card is silent and the field keeps the TKEConfig default. This
           is the dangerous group: silence is only correct when the default
           happens to be NEMO's, and nothing in the code says which.
  N/A      not a fidelity question (see _NOT_FIDELITY).

For the DEFAULT group it also flags any field whose type is ``str`` and whose
name appears in ``_KNOWN_NEMO_OPTIONS`` -- i.e. a selector with a documented
NEMO-shaped alternative that the card is not choosing.  That is exactly the
signature all five gaps had.

This is a REPORT, not a gate: it cannot know NEMO's intent for every field.
The companion gate is ``tests/unit/test_run_omip_core2_kappa_convention_cli.py``
and its siblings, which PIN the values already audited so they cannot drift
back.

Usage (CPU, seconds; sbatch it per the login-node policy):
    python scripts/validate/ocean_fidelity/audit_orca1_tke_card.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_REPO / "packages" / _p))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "run"))

# Fields that are numerics hygiene or legoESM-internal, not a NEMO mapping.
# THIS SET IS ITSELF A CLAIM, and it was wrong once already: `veros_dz_slots`
# was excluded here as "numerics hygiene" until codex 9405117 pointed out that
# it selects the DISCRETE OPERATOR.  NEMO's zzd = -0.5*dt*mean(avm)/(e3t*e3w)
# is flux-form -- the gradient is taken across the intervening T cell (/e3t)
# and the divergence into the W control volume (/e3w).  legoESM reproduces
# that mapping (dz_cell<->e3t, dz_half<->e3w) ONLY in the veros_dz_slots
# branch; the default legacy operator uses dz_half for BOTH.  So matching
# alpha_tke=1 fixes the coefficient while leaving the stencil different.
# Anything that changes WHICH discretisation runs belongs in the audit.
_NOT_FIDELITY = {
    "mxl_min", "tke_dry_wmask", "debug_precision",
}

# Non-str fields whose default is known NOT to be NEMO's. Same flagging as the
# str selectors -- the five gaps found on 2026-08-13 were mostly str, but
# alpha_tke (float) and veros_dz_slots (bool) show the pattern is not
# type-specific, so restricting the audit to strings would hide them.
_KNOWN_NEMO_NON_STR = {
    "veros_dz_slots": (True,
                       "NEMO zzd = -0.5*dt*mean(avm)/(e3t*e3w) is flux-form; "
                       "only the slots branch maps dz_cell->e3t, "
                       "dz_half->e3w. The legacy default uses dz_half twice."),
    "alpha_tke": (1.0, "NEMO diffuses en with avm x1 (zdftke tridiagonal)"),
}

# str-valued selectors known to carry a NEMO-shaped alternative. Keeping the
# alternatives here means a future reader sees WHAT the card is declining.
_KNOWN_NEMO_OPTIONS = {
    "kappa_convention": ("veros_sqrte", "NEMO avm = rn_ediff*zmxlm*sqrt(en)"),
    "surface_bc": ("nemo_dirichlet", "NEMO nn_bc_surf=1"),
    "tke_surface_bc_level": ("nemo_z0", "NEMO holds en(1) at z=0, solves jk>=2"),
    "tke_shear_production": ("nemo_face_native",
                             "NEMO zdf_sh2: face-native, now x before, "
                             "COASTAL DOUBLING (2-umask*umask)"),
    "dissipation_discretization": ("nemo_1p5_split",
                                   "NEMO zfact2=1.5*dt*rn_ediss diagonal + "
                                   "zfact3=0.5*rn_ediss explicit add-back"),
    "prandtl_mode": ("richardson", "NEMO nn_pdl=1"),
    # n2_mode / n2_eos_form: RESOLVED 2026-08-14. This entry used to read
    # "<TEOS-10 rn2, NOT IMPLEMENTED> ... Do not wire nemo_bn2 as if it were
    # ORCA1's" -- correct when written (codex 9405307 caught me wiring
    # nemo_bn2 alone, which silently took the S-EOS branch), and its own
    # stated exit condition was "closing it needs a TEOS-10 rab/bn2
    # implementation". That now exists: eos.nemo_roquet_alpha_beta ports the
    # Roquet et al. 2015 polynomial with the TEOS-10 coefficient set, and
    # compute_buoyancy_frequency_nemo_bn2(eos_form=...) selects it. So the
    # NEMO value is the PAIR -- nemo_bn2 alone is still the wrong N2, which
    # is why both are listed.
    "n2_mode": ("nemo_bn2",
                "ORCA1 zdftke consumes eosbn2's rn2; needs n2_eos_form="
                "'teos10' WITH it -- nemo_bn2 alone is the S-EOS bn2, a "
                "DIFFERENT N2 (codex 9405307)."),
    "n2_eos_form": ("teos10",
                    "ORCA1 runs ln_teos10=.true. (namelist_cfg:308), so the "
                    "bn2 alpha/beta are the Roquet polynomial on the TEOS-10 "
                    "coefficient set, not the 3-term S-EOS fit."),
    # Added after codex 9405307 named it and this auditor had missed it: NEMO
    # puts -avt*rn2 EXPLICITLY on the RHS with the carried avt
    # (zdftke.F90:417-418); the default splits it sign-aware onto the implicit
    # diagonal instead.
    "tke_buoyancy_sink": ("nemo_explicit",
                          "NEMO zdftke:417 puts -p_avt*rn2 on the RHS "
                          "explicitly with the carried avt"),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iwm", action="store_true",
                   help="build the card with iwm_enabled=True (what the "
                        "--iwm production arms get)")
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
    from run_omip_core2 import orca1_zdftke_config

    base = TKEConfig()
    card = orca1_zdftke_config(iwm_enabled=a.iwm)
    rows_set, rows_default, rows_na = [], [], []
    for f in card._fields:
        cv, dv = getattr(card, f), getattr(base, f)
        if f in _NOT_FIDELITY:
            rows_na.append((f, dv, cv))
        elif cv != dv:
            rows_set.append((f, dv, cv))
        else:
            rows_default.append((f, dv, cv))

    print(f"ORCA1 zdftke card audit (iwm_enabled={a.iwm})")
    print(f"  {len(rows_set)} SET, {len(rows_default)} left at DEFAULT, "
          f"{len(rows_na)} not-a-fidelity-question, "
          f"{len(card._fields)} fields total\n")

    print("--- SET by the card (deliberate NEMO mappings; re-check each) ---")
    for f, dv, cv in rows_set:
        print(f"  {f:32s} default {dv!r:24s} -> card {cv!r}")

    print("\n--- LEFT AT DEFAULT (silence is only correct if the default IS "
          "NEMO's) ---")
    flagged = []
    for f, dv, cv in rows_default:
        note = ""
        known = _KNOWN_NEMO_OPTIONS.get(f) or _KNOWN_NEMO_NON_STR.get(f)
        if known is not None:
            alt, why = known
            if cv != alt:
                note = f"   <== NEMO value {alt!r} NOT selected: {why}"
                flagged.append(f)
        print(f"  {f:32s} {cv!r}{note}")

    print("\n--- not a fidelity question ---")
    print("  " + ", ".join(f for f, _, _ in rows_na))

    if flagged:
        print(f"\nFLAGGED {len(flagged)}: {', '.join(flagged)}")
        print("Each is a str selector with a documented NEMO alternative the "
              "card declines. That is the exact signature of all five gaps "
              "found on 2026-08-13. Justify or wire each one.")
    else:
        print("\nNo str selector with a known NEMO alternative is left "
              "unselected.")

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_json, "w") as fh:
            json.dump({"iwm_enabled": a.iwm, "flagged": flagged,
                       "set": {f: [repr(d), repr(c)] for f, d, c in rows_set},
                       "default": {f: repr(c) for f, _, c in rows_default}},
                      fh, indent=1)
        print(f"[json] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
