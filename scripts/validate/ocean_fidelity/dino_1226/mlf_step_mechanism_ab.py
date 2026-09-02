#!/usr/bin/env python3
"""M-01 one-variable A/B: NEMO ``stp_MLF`` written twice, on the DINO card.

Branch-isomorphism row M-01 (``docs/ocean/fidelity/nemo_branch_isomorphism_map.md``)
records TWO legoESM implementations of one NEMO routine, ``stpmlf.F90:108-473``:

  * ``_leapfrog_step``  — TWO ``_step_impl`` passes; the dissipative increment
    is whatever a whole second pass evaluated at Nbb produced
    (``_ab2_scope_override="advective"`` withholds it from the Nnn pass).
    This is what the certified DINO card selects (``outer_integrator="leapfrog"``).
  * ``_nemo_mlf_step`` — ONE pass, with ``_ldf_state=(T,S,u,v)_before`` so only
    ``dyn_ldf``/``tra_ldf``/Redi read Nbb as their OWN call argument, exactly as
    ``stpmlf.F90:275`` (``dyn_ldf(Kbb,Kmm,...)``) and ``:437``
    (``tra_ldf(..., pts(:,:,:,:,Kbb), ...)``) do. Structurally the faithful one;
    selected by no card.

The registry's collapse question is whether the certified card can be repointed
at the faithful implementation without moving a certified number. That is a
MEASUREMENT, and this is it: at every step both methods are applied to the SAME
state, under the SAME jit, differing in one thing only — the method.

Why the PRIVATE methods and not ``outer_integrator="nemo_mlf"``: that public
value hard-requires ``implicit_vmix_e3t_now_divisor=True``
(``ocean_model_latlon_cgrid.py:3019``), which the certified DINO card sets
False. Going through the dispatch would therefore change TWO things at once
(the step composition and S-34's implicit-solve divisor). Calling the two
methods directly at the card's own config is what makes this one-variable.

Run (fp64, CPU):

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python mlf_step_mechanism_ab.py \
        --recipe nemo_dino_kamm_mlf --steps 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import numpy as np

import kamm_twin_90d as twin
from legoesm.ocean.experiments.dino import apply_dino_lat_lon_surface_forcing


FIELDS = ("T", "S", "u", "v", "eta")
# One float64 ulp at unit scale; a row is normalised by its own max|value|,
# exactly as legoesm.ocean.fidelity.ulp_move_gate does.
ULP = 2.0 ** -52


def field_move(a, b):
    """(max abs difference, that difference in ulps of the field's own scale)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = float(np.max(np.abs(a - b)))
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    ulps = diff / (ULP * scale) if scale > 0.0 else 0.0
    return diff, ulps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--run-traj", default=twin.RUN_TRAJ)
    p.add_argument("--run-stepdump", default=twin.RUN_STEPDUMP)
    p.add_argument("--restart-file", default=twin.RESTART_FILE)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--no-bridge-tke", dest="bridge_tke", action="store_false",
        help=("skip seeding state.tke/tke_avm from the NEMO restart. The "
              "kamm_mlf card selects tke_shear_evaluation_stage='step_entry' "
              "and tke_preclosure_coeff_source='carried_previous_step', which "
              "REQUIRE a carried avm_k, and only the restart bridge supplies "
              "it -- so the default here is on. Both arms see the same seeded "
              "state either way, so this is not the variable under test."))
    p.set_defaults(bridge_tke=True)
    p.add_argument(
        "--no-gm-redi", dest="use_gm_redi", action="store_false",
        help=("MECHANISM CONTROL: rebuild the card with GM/Redi off. The two "
              "implementations differ only in what the dissipative terms are "
              "evaluated on, and GM/Redi's tracer source is the one term "
              "whose operand the two mechanisms disagree about "
              "(tests/ocean/unit/test_nemo_mlf_step_transcription.py). With "
              "GM/Redi off the difference must fall to the fp64/XLA-fusion "
              "noise floor; if it does not, the probe is measuring something "
              "else and its headline number is not the mechanism."))
    p.set_defaults(use_gm_redi=None)
    args = p.parse_args(argv)

    if os.environ.get("JAX_ENABLE_X64") != "1":
        raise SystemExit("run with JAX_ENABLE_X64=1 (Rule 1c: oracle work is fp64)")
    # Rule 1c: JAX_ENABLE_X64 permits f64 arrays, it does not set legoESM's
    # precision policy -- the twin runner's own _precision_gate does this, and
    # without it the bridge builds the vertical ladder in f32.
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    br, cfg, mc, model, forcing, sf, st = twin._build_twin_state(
        args.recipe, args.run_traj, args.run_stepdump,
        bridge_tke=args.bridge_tke, use_gm_redi=args.use_gm_redi,
        restart_file=args.restart_file)
    if getattr(cfg, "outer_integrator", None) != "leapfrog":
        raise SystemExit(f"{args.recipe} does not select the leapfrog arm")
    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    dt = twin.DT
    t0_sec = twin.seasonal_t0_seconds(f"{args.run_stepdump}/{args.restart_file}")

    # SAME jit wrapper on both arms: an unjitted-vs-jitted pair would report XLA
    # fusion noise as a mechanism difference (see
    # tests/ocean/unit/test_nemo_mlf_step_transcription.py's docstring).
    lf = jax.jit(lambda s, e: model._leapfrog_step(
        s, dt, surface_forcing=sf, external_tracer_rate=e))
    mlf = jax.jit(lambda s, e: model._nemo_mlf_step(
        s, dt, surface_forcing=sf, external_tracer_rate=e))

    from legoesm.ocean.fidelity.precision_gate import require_fp64
    require_fp64(br.geometry, st, context="mlf_step_mechanism_ab")

    print(f"recipe={args.recipe} dt={dt} steps={args.steps} "
          f"gm_redi={'ON' if getattr(cfg, 'use_gm_redi', False) else 'off'} "
          f"placement={placement} bridge_tke={args.bridge_tke} u_before="
          f"{'bridged' if st.u_before is not None else 'None (Euler start)'}",
          flush=True)

    rows = []
    for k in range(args.steps):
        if placement == "leapfrog_rhs":
            st_f, ext = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, dt,
                t_seconds=t0_sec + (k + 1) * dt, return_rate=True)
        else:
            st_f = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, dt,
                t_seconds=t0_sec + (k + 1) * dt)
            ext = None
        a = lf(st_f, ext)
        b = mlf(st_f, ext)
        if k == 0:
            # Determinism control -- and ONLY that.  The same arm twice must be
            # exactly zero, which rules out a nondeterministic harness.  It
            # does NOT rule out the two arms differing because they are two
            # separate jit traces with different fusion; the discriminator for
            # that is --no-gm-redi, which must collapse the tracer difference
            # if the GM/Redi operand swap is the mechanism.
            a2 = lf(st_f, ext)
            for name in FIELDS:
                d, _ = field_move(getattr(a, name).data, getattr(a2, name).data)
                if d != 0.0:
                    raise SystemExit(
                        f"determinism control failed: leapfrog vs leapfrog "
                        f"moved {name} by {d:.3e}; the harness is not "
                        "deterministic")
            print("determinism control: leapfrog vs leapfrog = 0.0 on all "
                  "fields (does not exclude cross-trace fusion; use "
                  "--no-gm-redi for that)", flush=True)
        row = {"step": k + 1}
        for name in FIELDS:
            diff, ulps = field_move(getattr(a, name).data,
                                    getattr(b, name).data)
            row[name] = {"max_abs_diff": diff, "ulps_at_field_scale": ulps}
        rows.append(row)
        worst = max(FIELDS, key=lambda n: row[n]["ulps_at_field_scale"])
        print(f"step {k + 1}: worst={worst} "
              f"max|d|={row[worst]['max_abs_diff']:.6e} "
              f"({row[worst]['ulps_at_field_scale']:.3e} ulp)  "
              + "  ".join(f"{n}={row[n]['max_abs_diff']:.3e}" for n in FIELDS),
              flush=True)
        st = a          # advance on the CERTIFIED arm; one-step A/B each time

    first = next((r for r in rows
                  if any(r[n]["max_abs_diff"] > 0.0 for n in FIELDS)), None)
    report = {
        "recipe": args.recipe, "dt": dt, "steps": args.steps,
        "restart_file": args.restart_file,
        "producer_git_sha": twin_git_sha(),
        "gm_redi": bool(getattr(cfg, "use_gm_redi", False)),
        "surface_tendency_placement": placement,
        "bridge_tke": bool(args.bridge_tke),
        "rows": rows,
        "first_differing_step": None if first is None else first["step"],
        "worst_ulps": max(r[n]["ulps_at_field_scale"]
                          for r in rows for n in FIELDS),
        "verdict": ("BIT-IDENTICAL" if first is None
                    else "MEASURED DIFFERENCE"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def twin_git_sha() -> str:
    from legoesm.ocean.fidelity.provenance import git_sha
    try:
        return git_sha(allow_dirty=True)
    except RuntimeError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
