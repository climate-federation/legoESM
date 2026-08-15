"""Single-column (SCM) RCE **convection-scheme** intercomparison vs the plane CRM.

For every convection scheme in the campaign sweep this driver measures how well
an SCM RCE column reproduces the RCEMIP1 plane-CRM reference, both **a priori**
(scheme defaults) and **a posteriori** (after derivative-free tuning of the
scheme's ``tunable_tier<=extended`` parameters against the CRM profiles).

It is a thin orchestration layer over ``run_scm_rce_campaign`` — the CRM
reference extraction, the JIT SCM RCE evaluation, the normalized profile score
and the derivative-free tuner are all reused verbatim (see CLAUDE.md: "reuse
scripts/run/run_scm_rce_campaign.py for CRM reference extraction / SCM
evaluation"; no RCE profile numerics are re-derived here).

Outputs (under ``--outdir``, default ``results/scm_rce_convection_intercomparison``):

* ``summary.md``            — a-priori vs tuned RMSE table + tuned parameters,
* ``intercomparison.csv``   — machine-readable per-scheme metrics,
* ``tuned_parameters.json`` — per-scheme tuned overrides (recommended, never
  mutates production ``*Config`` defaults),
* ``profiles_<scheme>.png`` — CRM (black), a-priori (red **dashed**), tuned
  (red **solid**) T / q_v / condensate profiles,
* ``profiles_all_convection.png`` — every scheme in one grid.

Tuning objective is the combined profile+precip score against the CRM (lower is
better). The realism/equilibrium gate is *reported* per scheme but not used to
reject tuning trials, so every scheme is tuned toward the best CRM match the
user asked for; unphysical equilibria are flagged in the ``realism`` column
rather than silently hidden.

The schemes do not share a compensating-subsidence kernel by default, so an
as-shipped ranking partly measures the KERNEL rather than the scheme.
``--subsidence-solve`` selects the arm (via the campaign's shared
``apply_subsidence_solve_override`` selector); run it twice into distinct
``--outdir`` and read the PRIMARY arm for scheme physics.

Example (GPU) — the two arms::

    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/run/run_scm_rce_convection_intercomparison.py \
        --tune-evals 48 --subsidence-solve implicit_flux \
        --outdir results/scm_rce_convection_intercomparison_matched   # PRIMARY

    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/run/run_scm_rce_convection_intercomparison.py \
        --tune-evals 48 --subsidence-solve as_shipped \
        --outdir results/scm_rce_convection_intercomparison_shipped   # SECONDARY
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _load_campaign():
    """Import the campaign module as the shared SCM-RCE numerics library.

    The campaign lives in ``scripts/run`` as an executable driver, not an
    installed package; load it by path and register it in ``sys.modules`` so its
    module-level ``@dataclass`` definitions resolve.
    """
    path = Path(__file__).resolve().parent / "run_scm_rce_campaign.py"
    spec = importlib.util.spec_from_file_location("scm_rce_campaign", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


camp = _load_campaign()

DEFAULT_OUTDIR = Path("results/scm_rce_convection_intercomparison")
# Convection schemes to intercompare (the campaign's convection sweep).
CONVECTION_SCHEMES = camp.SCHEME_SWEEPS["convection"]


@dataclass
class SchemeResult:
    scheme: str
    prior: "camp.RunDiagnostics"
    tuned: "camp.RunDiagnostics"
    records: list  # list[camp.TuneRecord]
    # Which vertical-transport kernel ARM produced these numbers, and what the
    # shared selector actually did for THIS scheme.  Carried on every result and
    # written into the JSON / CSV / summary so a score can never be read without
    # knowing whether the schemes were kernel-matched (a cross-arm comparison is
    # a confound, not a result).
    subsidence_solve: str = "as_shipped"
    subsidence_solve_status: str = ""
    # The run signature the checkpoint was written under.  Carried on the
    # result (not just on disk) because the merge stage must be able to refuse
    # a checkpoint produced against a DIFFERENT reference: the physical-unit
    # RMSE columns are computed at merge time against the CURRENT reference,
    # so a stale profile with the same level count yields a plausible, wrong
    # number rather than a NaN.
    signature: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Per-scheme persistence (each scheme runs in its own process; recompilation of
# the static-config SCM graph on every tune eval makes expensive-compile schemes
# — zhang_mcfarlane, emanuel, … — slow, so schemes are parallelized across cores
# and results checkpointed to disk so a killed/parallel run is resumable).
# --------------------------------------------------------------------------- #
from dataclasses import asdict  # noqa: E402


def _scheme_json_path(outdir: Path, scheme: str) -> Path:
    return outdir / f"scheme_{scheme}.json"


# Config fields that change a scheme's numerical result.  A checkpoint written
# under one signature must not be silently reused under another — the classic
# footgun is a ``--quick`` smoke checkpoint (0.05 day, 2 tune evals) reused in a
# full-length run and merged as if it were a real result.  Guards skip + merge.
#
# ``subsidence_solve`` is in here for the same reason and is the sharpest case:
# the PRIMARY (matched-kernel) and SECONDARY (as-shipped) arms are two different
# experiments, so reusing one arm's per-scheme checkpoint in the other would
# silently FABRICATE that arm's number.  Removing it from this tuple is a
# correctness regression, gated by
# tests/unit/test_scm_rce_convection_intercomparison_cli.py.
_SIGNATURE_FIELDS = (
    "days", "dt", "analysis_days", "tune_evals", "tune_seed", "radiation",
    "radiation_update_interval_steps", "large_scale_forcing",
    "surface_wind_m_s", "coriolis_s_inv",
    "scm_microphysics_substeps", "scm_convection_substeps",
    "subsidence_solve",
    "reference_dir", "last_reference_files",
    # The column's saturation treatment is part of the experiment: the ice
    # super-saturation allowance only exists in the ice-capable schemes, and
    # the in-scheme liquid guard changes the condensation rate every step.  A
    # checkpoint from one setting must not be reused under another.
    "microphysics", "hard_saturation_adjustment",
    # A column whose lowest level is pinned to the SST has no sensible heat
    # flux; that is a different experiment, not a tuning detail.
    "bl_anchor_top_m",
    # WHICH boundary-layer scheme maintains the sub-cloud layer, WHAT the tuner
    # minimises, WHICH parameters it is allowed to move, and where the
    # sub-cloud layer is taken to end.  Each changes the experiment outright, so
    # a checkpoint from one setting must never be reused under another.
    "turbulence", "tune_mode", "objective", "focused_include", "subcloud_top_m",
    # Emanuel's downdraft re-evaporation ships OFF behind a static branch;
    # turning it on is different PHYSICS, not a tuning detail.
    "emanuel_unsaturated_downdraft",
    # WHICH parameter set the tuner may move (registry tier, or the
    # derivative-free `physical` set) and how much of the budget goes to the
    # local refinement.  An `extended`-tier checkpoint reused under an
    # `aggressive` request would report a search that never happened.
    "param_set", "tune_refine_frac",
    # The DEFINITION of the thermodynamic objective: its two tolerances, the
    # pressure bound that separates troposphere from stratosphere, and the
    # liquid/ice blend width of the saturation curve behind RH.  Each changes
    # what "close to the CRM" means, so a checkpoint written under one must
    # never be merged into a table labelled with another.
    "thermo_T_scale_K", "thermo_rh_scale", "thermo_min_p_Pa",
    "thermo_rh_blend_width_K",
)


#: What the tuner is allowed to move.  ``convection`` is the historical arm
#: (every extended-tier parameter of the active convection scheme);
#: ``focused`` searches ONE named cross-category set instead.  A typo must
#: select nothing, so both the driver and ``evaluate_scheme`` raise.
TUNE_MODES = ("convection", "focused")

#: Categories the focused mode is allowed to touch.  A category whose active
#: scheme has no tunable sub-config (convection="none") is skipped with a note.
FOCUSED_TUNE_CATEGORIES = ("turbulence", "microphysics", "convection")

#: THE SUB-CLOUD HYPOTHESIS, written down as parameters.
#:
#: Measured (job 9403110): our tuned columns sit at RH 0.82-0.99 with the
#: lowest level within ~1 K of the SST, against the CRM's RH 0.752 and 3.06 K.
#: The air-sea humidity difference that drives evaporation is therefore 26-59 %
#: of the reference's, while the transfer coefficient residual is already
#: 1.6-1.8x too LARGE — so the deficit is the state of the sub-cloud layer, not
#: the surface exchange, and only two families of process set that state in an
#: RCE column with no large-scale forcing:
#:
#: 1. how vigorously the boundary-layer scheme ventilates the layer, and
#: 2. how much falling precipitation re-evaporates into it (directly in the
#:    microphysics, and through the convection scheme's downdrafts).
#:
#: Each name below is registry-qualified; ``tune_focused_params`` raises if one
#: matches no parameter of an active scheme, so this list cannot silently rot.
#: EVERY name here was checked to be READ by numerical code.  Four more were
#: proposed and REMOVED after that check: ``Lscale_mu_coef``, ``mult_coef``,
#: ``C_invrs_tau_sfc`` and ``C_invrs_tau_bkgnd`` exist as CLUBBParams fields and
#: are referenced by nothing but tuning include-lists — the same defect class
#: as the ``a_const``/``up2_sfc_coef`` that PR #1601 found unread.
#: ``scripts/validate/audit_param_gradients.py`` is the mechanical form of that
#: check; run it before adding a name here.
SUBCLOUD_TURBULENCE_INCLUDE = {
    "clubb": (
        # Eddy diffusivity: Km = c_K * L * sqrt(TKE), and the scalar variants.
        "atm.turb.CLUBBParams.c_K",
        "atm.turb.CLUBBParams.c_K1",
        "atm.turb.CLUBBParams.c_K2",
        # Mixing length: the parcel entrainment rate and the floor/stability
        # limiters that decide how deep the surface layer's mixing reaches.
        "atm.turb.CLUBBParams.mu",
        "atm.turb.CLUBBParams.lmin_coef",
        "atm.turb.CLUBBParams.lambda0_stability_coef",
    ),
}

#: Rain re-evaporation below cloud base -- and the direction that matters here
#: is DOWNWARD.  Rain falling into a sub-saturated layer evaporates, cooling it
#: (helping the 3 K air-sea deficit) but ADDING VAPOUR to it, and too much of
#: that drives the layer towards saturation, which is the measured defect
#: (RH 0.82-0.99 against the reference's 0.752).  So these are tuned as a
#: SUSPECTED EXCESS, not as a missing process.  Consistent with the columns
#: themselves: the low-level condensate is precipitation-dominated
#: (0.011 g/kg rain against 0.006 g/kg cloud in the smoke column).
#:
#: ``MorrisonConfig.evap_coeff`` was proposed and REMOVED: the default
#: ``rain_evap_scheme="m2005"`` calls ``rain_evaporation_m2005``, which uses
#: the two ventilation coefficients; ``evap_coeff`` is read only by the legacy
#: ``"bulk"`` branch (morrison.py:316-320, config.py:610).
SUBCLOUD_MICROPHYSICS_INCLUDE = {
    "morrison": (
        "atm.micro.MorrisonConfig.rain_vent_f1",
        "atm.micro.MorrisonConfig.rain_vent_f2",
    ),
}

#: The convective downdraft that re-evaporates precipitation below cloud base.
#: Only the schemes that expose such a knob appear; for the others the focused
#: set is turbulence + microphysics, reported rather than silently assumed.
#:
#: EMANUEL IS ABSENT BY DEFAULT AND THAT IS A FINDING, NOT AN OVERSIGHT.  Its
#: ``downdraft_efficiency`` is read only inside
#: ``if config.enable_unsaturated_downdraft:`` (emanuel.py:465), a STATIC
#: Python branch whose field defaults to ``False`` (config.py:1073) -- so
#: Emanuel currently has NO convective downdraft re-evaporation at all, and
#: there is nothing to slow until the branch is switched on.  (The field's own
#: docstring at config.py:995-1011 still claims "default ``True``"; the code is
#: the authority and the docstring is stale.)  Enabling it is a one-variable
#: experiment of its own -- the branch MOISTENS the sub-cloud layer, per the
#: comment at emanuel.py:463 -- so it is exposed as
#: ``--emanuel-unsaturated-downdraft`` and enters the checkpoint signature,
#: rather than being turned on silently as part of a tuning set.
SUBCLOUD_CONVECTION_INCLUDE = {
    "bechtold": ("atm.conv.BechtoldConfig.downdraft_evap_efficiency",),
    "tiedtke": ("atm.conv.TiedtkeConfig.downdraft_evap_efficiency",),
}

#: Curated only when the gating flag is on; see the note above.
EMANUEL_DOWNDRAFT_INCLUDE = ("atm.conv.EmanuelConfig.downdraft_efficiency",)


def default_focused_include(
    *, turbulence: str, microphysics: str, convection: str,
    emanuel_unsaturated_downdraft: bool = False,
) -> tuple[str, ...]:
    """The focused parameter set for one (turbulence, microphysics, convection).

    Built per configuration rather than as one flat list because a name that
    belongs to an INACTIVE scheme is a hard error in the tuner (by design), so
    the default must contain exactly the knobs the active schemes own.

    Emanuel's downdraft efficiency is included ONLY when its gating flag is on,
    because with the flag off the parameter is read by no executed code and
    would be a search dimension that cannot move anything.
    """
    include = (
        SUBCLOUD_TURBULENCE_INCLUDE.get(turbulence, ())
        + SUBCLOUD_MICROPHYSICS_INCLUDE.get(microphysics, ())
        + SUBCLOUD_CONVECTION_INCLUDE.get(convection, ())
    )
    if convection == "emanuel" and emanuel_unsaturated_downdraft:
        include = include + EMANUEL_DOWNDRAFT_INCLUDE
    return include


def _run_signature(args) -> dict:
    # Path -> str so the signature is JSON-serializable; every other field the
    # argparse namespace stores here is already a JSON scalar/list.  Includes the
    # tune seed and the reference selection — different seed or CRM reference is a
    # different experiment and must invalidate a checkpoint.
    def _norm(v):
        return str(v) if isinstance(v, Path) else v
    sig = {}
    for k in _SIGNATURE_FIELDS:
        if hasattr(args, k):
            sig[k] = _norm(getattr(args, k))
        elif k in _OBJECTIVE_DEFINITION_CONSTANTS:
            # Not a flag: the objective's tolerances are module constants with
            # written-down reasons, deliberately not per-run knobs.  They still
            # belong in the signature — editing one redefines "close to the
            # CRM", and a checkpoint scored under the old value would otherwise
            # merge into a table labelled with the new one.
            sig[k] = _OBJECTIVE_DEFINITION_CONSTANTS[k]
        else:
            raise AttributeError(
                f"_run_signature: {k!r} is in _SIGNATURE_FIELDS but is neither "
                "an argparse field nor a declared objective constant")
    return sig


#: Objective-definition constants that enter the signature without being flags.
#: Read from the shared metrics module by NAME so an edit there propagates here
#: instead of this file carrying a second copy of the number.
_OBJECTIVE_DEFINITION_CONSTANTS = {
    "thermo_T_scale_K": camp.DEFAULT_THERMO_T_SCALE_K,
    "thermo_rh_scale": camp.DEFAULT_THERMO_RH_SCALE,
    "thermo_min_p_Pa": camp.DEFAULT_THERMO_MIN_P_PA,
    "thermo_rh_blend_width_K": camp.DEFAULT_THERMO_RH_BLEND_WIDTH_K,
}


def _checkpoint_signature(path: Path) -> dict:
    try:
        return json.loads(path.read_text()).get("signature", {}) or {}
    except (OSError, ValueError):
        return {}


def _signature_compatible(ckpt_sig: dict, run_sig: dict) -> bool:
    # An UNSTAMPED checkpoint ({}) is no longer grandfathered: it predates the
    # signature, so nothing establishes which protocol produced it, and
    # accepting it is how a stale result reaches a published table.  Delete it
    # or re-run with --force.
    return bool(ckpt_sig) and ckpt_sig == run_sig


def save_scheme_result(outdir: Path, res: SchemeResult, signature: dict | None = None) -> None:
    payload = {
        "scheme": res.scheme,
        "signature": signature or {},
        "subsidence_solve": res.subsidence_solve,
        "subsidence_solve_status": res.subsidence_solve_status,
        "prior": asdict(res.prior),
        "tuned": asdict(res.tuned),
        "records": [asdict(r) for r in res.records],
    }
    # Checkpoints use Python's extended JSON (NaN/Infinity permitted) so a
    # crashed scheme's non-finite score round-trips back through
    # load_scheme_result; they are Python-self-consumed, not strict JSON.
    _scheme_json_path(outdir, res.scheme).write_text(json.dumps(payload, indent=2))


def load_scheme_result(path: Path) -> SchemeResult:
    payload = json.loads(path.read_text())
    return SchemeResult(
        scheme=payload["scheme"],
        prior=camp.RunDiagnostics(**payload["prior"]),
        tuned=camp.RunDiagnostics(**payload["tuned"]),
        records=[camp.TuneRecord(**r) for r in payload["records"]],
        # Pre-arm checkpoints predate the flag; they were necessarily run with
        # the shipped defaults, so that is the honest label for them.  The empty
        # status distinguishes "never stamped" from a stamped "as_shipped".
        subsidence_solve=payload.get("subsidence_solve", "as_shipped"),
        subsidence_solve_status=payload.get("subsidence_solve_status", ""),
        signature=payload.get("signature", {}) or {},
    )


# Fields whose disagreement makes two checkpoints incomparable AT MERGE TIME.
# Not the whole signature: `tune_evals` is deliberately per-scheme (the budget
# scales with the scheme's parameter count), and a different tuning budget does
# not make two rows incomparable — it is reported per row instead.
_MERGE_CRITICAL_SIGNATURE_FIELDS = tuple(
    f for f in _SIGNATURE_FIELDS if f != "tune_evals")


def _guard_merge_inputs(results, run_sig: dict | None, *, allow_partial: bool,
                        reference_dir=None) -> None:
    """Refuse to publish a ranking that is partial or built from mixed runs.

    Two failure modes, both silent without this:

    * a ranking table and figures assembled from whichever schemes happened to
      finish — one surviving checkpoint is enough to produce a plausible
      "ranking" of one scheme; and
    * checkpoints written under DIFFERENT protocols merged into one table.
      Since the physical-unit RMSE columns are computed at merge time against
      the CURRENT reference, a stale profile with the same level count
      produces a believable wrong number, not a NaN.

    ``run_sig`` is the signature of THIS invocation, used as the baseline when
    the merge happens inside a computing run.  In ``--merge-only`` there is no
    meaningful current signature — the merge job legitimately does not repeat
    the arm's twenty flags — so pass ``None`` and the checkpoints are required
    to agree with EACH OTHER instead, plus with ``reference_dir``, which is the
    one flag the merge really does supply and the one the physical-unit columns
    are computed against.

    ``allow_partial`` prints what is wrong and continues, so an operator
    debugging a subset has said out loud that the artifacts are partial.
    """
    def _fail(msg: str) -> None:
        if not allow_partial:
            raise SystemExit(msg)
        print(f"[warn] {msg}", flush=True)

    present = {r.scheme for r in results}
    missing = [s for s in CONVECTION_SCHEMES if s not in present]
    if missing:
        _fail(f"MERGE REFUSED: {len(present)}/{len(CONVECTION_SCHEMES)} scheme "
              f"checkpoints present; missing {missing}. A ranking over a "
              "subset is not a ranking of the campaign. Re-run the missing "
              "arms, or pass --allow-partial if you know the table is partial.")

    unstamped = [r.scheme for r in results if not r.signature]
    if unstamped:
        _fail("MERGE REFUSED: unstamped (pre-signature) checkpoint(s) "
              f"{unstamped} — nothing records which protocol produced them.")

    stamped = [r for r in results if r.signature]
    baseline = run_sig if run_sig is not None else (
        stamped[0].signature if stamped else {})
    mismatched = []
    for r in stamped:
        diffs = {k: (r.signature.get(k), baseline.get(k))
                 for k in _MERGE_CRITICAL_SIGNATURE_FIELDS
                 if r.signature.get(k) != baseline.get(k)}
        if diffs:
            mismatched.append(f"{r.scheme}: {diffs}")
    if mismatched:
        _fail("MERGE REFUSED: checkpoint(s) written under a different protocol "
              "than the rest — the physical-unit RMSE columns are computed "
              "against ONE reference, so merging these would produce "
              "believable wrong numbers:\n  " + "\n  ".join(mismatched))

    if reference_dir is not None:
        wrong_ref = [f"{r.scheme}: {r.signature.get('reference_dir')}"
                     for r in stamped
                     if r.signature.get("reference_dir") != str(reference_dir)]
        if wrong_ref:
            _fail("MERGE REFUSED: checkpoint(s) were scored against a "
                  f"different reference than {reference_dir}:\n  "
                  + "\n  ".join(wrong_ref))


def load_all_scheme_results(outdir: Path, schemes) -> list[SchemeResult]:
    # Merge aggregates EVERY checkpoint present (its documented job); the
    # signature guard lives on the compute-path skip, which recomputes a
    # mismatched checkpoint before it can reach the merge.
    results = []
    for scheme in schemes:
        path = _scheme_json_path(outdir, scheme)
        if path.exists():
            results.append(load_scheme_result(path))
    return results


def evaluate_scheme(
    scheme: str,
    ref,
    *,
    days: float,
    dt: float,
    analysis_days: float,
    tune_evals: int,
    seed: int,
    scm_microphysics_substeps: int,
    scm_convection_substeps: int,
    surface_wind_m_s: float,
    coriolis_s_inv: float,
    large_scale_forcing: str,
    radiation: str,
    radiation_update_interval_steps: int,
    subsidence_solve: str = "as_shipped",
    microphysics: str = camp.BASELINE_SCHEMES["microphysics"],
    hard_saturation_adjustment: bool = False,
    bl_anchor_top_m: float = camp.DEFAULT_SCM_RCE_BL_TOP_M,
    turbulence: str = camp.BASELINE_SCHEMES["turbulence"],
    tune_mode: str = "convection",
    objective: str = "combined",
    focused_include: tuple[str, ...] = (),
    subcloud_top_m: float = camp.DEFAULT_SUBCLOUD_TOP_M,
    emanuel_unsaturated_downdraft: bool = False,
    param_set: str = "extended",
    tune_refine_frac: float = 0.0,
) -> SchemeResult:
    """A-priori run + derivative-free tuning for one convection scheme.

    The realism/equilibrium gate is disabled (``require_*=False``) so the tuner
    minimizes the CRM profile score freely; realism diagnostics are still
    computed and reported.

    ``subsidence_solve`` selects the vertical-transport kernel arm via the shared
    ``camp.apply_subsidence_solve_override`` selector (never re-implemented
    here).  It is applied to ``base_cfg`` IMMEDIATELY, before the a-priori run
    and before tuning, so BOTH see the same kernel — applying it later would
    tune under one kernel and report under another.
    """
    if tune_mode not in TUNE_MODES:
        raise ValueError(
            f"evaluate_scheme: unknown tune_mode {tune_mode!r}; "
            f"expected one of {TUNE_MODES}")
    cache: dict[str, "camp.RunDiagnostics"] = {}
    base_cfg = camp.make_physics_config(
        radiation=radiation,
        radiation_update_interval_steps=radiation_update_interval_steps,
        convection=scheme,
        turbulence=turbulence,
        microphysics=microphysics,
        hard_saturation_adjustment=hard_saturation_adjustment,
    )
    base_cfg, solve_status = camp.apply_subsidence_solve_override(
        base_cfg, subsidence_solve, category="convection")
    if emanuel_unsaturated_downdraft:
        # Emanuel's downdraft re-evaporation is a STATIC Python branch that
        # ships OFF, so its efficiency parameter is read by no executed code
        # until this is set.  Applied BEFORE the a-priori run so both
        # conditions see the same branch — switching it later would tune under
        # one physics and report under another.
        if scheme != "emanuel":
            raise ValueError(
                "emanuel_unsaturated_downdraft was requested with "
                f"convection={scheme!r}; the flag belongs to EmanuelConfig and "
                "silently ignoring it would leave the run label wrong.")
        _component, _s, sub = camp._active_subconfig(base_cfg, "convection")
        base_cfg = camp._set_active_subconfig(
            base_cfg, "convection",
            sub._replace(enable_unsaturated_downdraft=True))
    common = dict(
        days=days,
        dt=dt,
        analysis_days=analysis_days,
        require_equilibrium=False,
        require_realism=False,
        equil_T_tol_K=camp.EQUIL_T_TOL_K,
        equil_qv_tol=camp.EQUIL_QV_TOL,
        equil_qcond_tol=camp.EQUIL_QCOND_TOL,
        scm_microphysics_substeps=scm_microphysics_substeps,
        scm_convection_substeps=scm_convection_substeps,
        surface_wind_m_s=surface_wind_m_s,
        coriolis_s_inv=coriolis_s_inv,
        large_scale_forcing=large_scale_forcing,
        bl_anchor_top_m=bl_anchor_top_m,
        subcloud_top_m=subcloud_top_m,
    )
    prior = camp.run_cached(cache, base_cfg, ref, label=f"prior:{scheme}", **common)
    if tune_mode == "focused":
        # The sub-cloud experiment: ONE named parameter set spanning the
        # boundary-layer scheme, the microphysics' rain re-evaporation and the
        # convection scheme's downdrafts.  ``prior`` above and the tuner's own
        # default run are the same config, so they hit the same cache entry —
        # the a-priori column is not paid for twice.
        _best_cfg, records, _default_run, tuned = camp.tune_focused_params(
            base_cfg,
            ref,
            cache,
            categories=FOCUSED_TUNE_CATEGORIES,
            include=focused_include,
            tune_evals=tune_evals,
            seed=seed,
            objective=objective,
            **common,
        )
    else:
        _best_cfg, records, tuned = camp.tune_category_winner(
            "convection",
            base_cfg,
            ref,
            cache,
            tune_evals=tune_evals,
            seed=seed,
            objective=objective,
            param_set=param_set,
            refine_frac=tune_refine_frac,
            **common,
        )
    return SchemeResult(
        scheme=scheme, prior=prior, tuned=tuned, records=records,
        subsidence_solve=subsidence_solve, subsidence_solve_status=solve_status,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float, prec: str = ".4g") -> str:
    return "inf" if not np.isfinite(x) else format(float(x), prec)


def _physical_verdict(run) -> str:
    """Physical/unphysical verdict from the always-computed realism diagnostics.

    Mirrors ``run_scm_rce``'s equilibrium+realism gate — same campaign threshold
    constants (reused by name, no re-derived numerics) and same NaN-fails-closed
    semantics as ``realism_reasons_from_diagnostics``. We evaluate it here because
    the tuning runs use ``require_realism=False`` (so the tuner minimizes the CRM
    score freely), which leaves ``run.realism_status`` as ``not_checked`` even
    though the underlying moist-adiabat / cold-point / drift diagnostics are still
    populated.

    Covers every gate condition whose scalar is stored on ``RunDiagnostics``:
    equilibrium drift (T/qv/qcond), moist-adiabat mean+max deviation, and
    cold-point T+z. The one gate condition it cannot see is the free-troposphere
    level count (``n_free_trop_levels >= MIN_FREE_TROP_LEVELS = 3``), which is not
    a stored field — negligible for a 30-level RCE column but noted for honesty:
    a ``physical`` verdict here means "physical modulo the level-count check".
    """
    if not np.isfinite(run.score):
        return "nonfinite"
    # Honor the campaign's own hard-failure status FIRST: a run the campaign
    # marked failed (negative q_v/condensate, out-of-range T, invalid precip,
    # non-finite profile — see run_scm_rce_campaign's realism gate) must never
    # be reported "physical" just because the stored realism scalars happen to
    # pass.  ``status == "ok"`` means the campaign's hard checks passed.
    if run.status != "ok":
        return "unphysical"
    reasons = (
        run.drift_T_rmse_K > camp.EQUIL_T_TOL_K
        or run.drift_qv_rmse > camp.EQUIL_QV_TOL
        or run.drift_qcond_rmse > camp.EQUIL_QCOND_TOL
        or not (run.moist_adiabat_mean_abs_K <= camp.MADIAB_MEAN_TOL_K)
        or not (run.moist_adiabat_max_abs_K <= camp.MADIAB_MAX_TOL_K)
        or not (camp.COLD_POINT_MIN_K <= run.cold_point_T_K <= camp.COLD_POINT_MAX_K)
        or not (camp.TROP_MIN_Z_KM <= run.cold_point_z_km <= camp.TROP_MAX_Z_KM)
    )
    return "unphysical" if reasons else "physical"


CSV_FIELDS = (
    "scheme",
    # Kernel arm, immediately after the scheme name: every downstream reader of
    # this CSV sees which arm produced the row before it sees any metric.
    "subsidence_solve", "subsidence_solve_status",
    "prior_score", "prior_T_rmse", "prior_qv_rmse", "prior_cloud_rmse",
    "prior_precip_rmse", "prior_precip_mm_day", "prior_verdict",
    "tuned_score", "tuned_T_rmse", "tuned_qv_rmse", "tuned_cloud_rmse",
    "tuned_precip_rmse", "tuned_precip_mm_day", "tuned_verdict",
    "score_improvement_pct", "crm_precip_mm_day", "n_tuned_params",
    # The tuning budget is per-scheme (it scales with the parameter count), so
    # it belongs on the ROW; a single campaign-wide number in the preamble
    # would be the last finisher's value.
    "tune_evals", "evals_per_param",
    "tuned_drift_T_K", "tuned_madiab_mean_K", "tuned_cold_point_T_K",
    "tuned_cold_point_z_km",
    # PHYSICAL-unit RMSE (K, g/kg).  The scores above are normalised by the
    # reference's mass-weighted standard deviation — commensurable for the
    # optimiser, uninterpretable in a figure caption.  Column names match what
    # scripts/plot/plot_scm_rce_convection_paper.py reads.
    "apriori_T_rmse_K", "tuned_T_rmse_K",
    "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg",
    "apriori_qcond_rmse_g_kg", "tuned_qcond_rmse_g_kg",
    # SUB-CLOUD LAYER.  Reported for every arm, not only the sub-cloud one, so
    # the column that was never scored can still be read off the historical
    # rows.  The bulk state is what the surface fluxes see; the CRM reference
    # values it should be compared against are RH 0.752, SST-T_air 3.06 K.
    "apriori_subcloud_score", "tuned_subcloud_score",
    "apriori_sfc_rh", "tuned_sfc_rh",
    "apriori_sfc_delta_T_K", "tuned_sfc_delta_T_K",
    "apriori_sfc_driver_g_kg", "tuned_sfc_driver_g_kg",
    "apriori_evap_mm_day", "tuned_evap_mm_day",
    # THE OBJECTIVE THAT WAS ACTUALLY MINIMISED, by name, next to its value on
    # both conditions and next to the improvement computed FROM IT.  Without
    # this the table carries `score`/`score_improvement_pct` from the historical
    # combined score whatever the tuner was asked to minimise, which is how a
    # thermo-tuned campaign gets read as a condensate result.
    "objective", "prior_objective", "tuned_objective",
    "objective_improvement_pct",
    # TEMPERATURE AND TROPOSPHERIC HUMIDITY, the target of --objective thermo.
    # The `_term` columns are the two halves of that score (dimensionless,
    # 1 K and 5 % RH per unit); the RH/qv columns are physical.
    "apriori_thermo_score", "tuned_thermo_score",
    "apriori_thermo_T_term", "tuned_thermo_T_term",
    "apriori_thermo_rh_term", "tuned_thermo_rh_term",
    "apriori_trop_rh_rmse", "tuned_trop_rh_rmse",
    "apriori_trop_qv_rmse_g_kg", "tuned_trop_qv_rmse_g_kg",
    # HELD-OUT WINDOW: the same thermo score one analysis window earlier.  A
    # tuned column that matches only where it was scored shows up here.
    "apriori_heldout_thermo_score", "tuned_heldout_thermo_score",
    "tuned_heldout_T_rmse_K", "tuned_heldout_qv_rmse_g_kg",
    # WATER BUDGET of the tuned column: an "equilibrium" accumulating water is
    # not one, and the previous arm's winner was doing exactly that.
    "tuned_P_minus_E_mm_day",
)

KG_KG_TO_G_KG = 1_000.0


def _improvement_pct(prior: float, tuned: float) -> float:
    return (
        100.0 * (prior - tuned) / prior
        if np.isfinite(prior) and prior > 0 and np.isfinite(tuned)
        else float("nan")
    )


def _row_objective(res: SchemeResult) -> str:
    """The objective this scheme's checkpoint was tuned under.

    Read from the ROW's own signature, never from the current invocation's
    arguments: a merge aggregates checkpoints, and labelling each row with the
    merging process's flags is how a table gets a name its numbers do not have.
    An unstamped row says so instead of guessing.
    """
    return str(res.signature.get("objective", "unstamped"))


def _row(res: SchemeResult, ref=None) -> dict:
    p, t = res.prior, res.tuned
    impr = _improvement_pct(p.score, t.score)
    objective = _row_objective(res)
    prior_obj = (
        camp.objective_value(p, objective)
        if objective in camp.TUNE_OBJECTIVES else float("nan"))
    tuned_obj = (
        camp.objective_value(t, objective)
        if objective in camp.TUNE_OBJECTIVES else float("nan"))
    thermo = {
        "objective": objective,
        "prior_objective": prior_obj,
        "tuned_objective": tuned_obj,
        "objective_improvement_pct": _improvement_pct(prior_obj, tuned_obj),
        "apriori_thermo_score": p.thermo_score,
        "tuned_thermo_score": t.thermo_score,
        "apriori_thermo_T_term": p.thermo_T_term,
        "tuned_thermo_T_term": t.thermo_T_term,
        "apriori_thermo_rh_term": p.thermo_rh_term,
        "tuned_thermo_rh_term": t.thermo_rh_term,
        "apriori_trop_rh_rmse": p.trop_rh_rmse,
        "tuned_trop_rh_rmse": t.trop_rh_rmse,
        "apriori_trop_qv_rmse_g_kg": p.trop_qv_rmse_g_kg,
        "tuned_trop_qv_rmse_g_kg": t.trop_qv_rmse_g_kg,
        "apriori_heldout_thermo_score": p.heldout_thermo_score,
        "tuned_heldout_thermo_score": t.heldout_thermo_score,
        "tuned_heldout_T_rmse_K": t.heldout_T_rmse_K,
        "tuned_heldout_qv_rmse_g_kg": t.heldout_qv_rmse_g_kg,
        "tuned_P_minus_E_mm_day": t.precip_mm_day - t.evap_mm_day,
    }
    if ref is None:
        # No reference in scope (unit tests of the row shape): the physical
        # columns are NaN rather than absent, so the CSV header never changes
        # shape between call sites.
        nan = float("nan")
        phys = {k: nan for k in (
            "apriori_T_rmse_K", "tuned_T_rmse_K",
            "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg",
            "apriori_qcond_rmse_g_kg", "tuned_qcond_rmse_g_kg")}
    else:
        pr = camp.physical_profile_rmse(ref, p)
        tr = camp.physical_profile_rmse(ref, t)
        phys = {
            "apriori_T_rmse_K": pr["T_rmse_K"],
            "tuned_T_rmse_K": tr["T_rmse_K"],
            "apriori_qv_rmse_g_kg": pr["qv_rmse_kg_kg"] * KG_KG_TO_G_KG,
            "tuned_qv_rmse_g_kg": tr["qv_rmse_kg_kg"] * KG_KG_TO_G_KG,
            "apriori_qcond_rmse_g_kg": pr["qcond_rmse_kg_kg"] * KG_KG_TO_G_KG,
            "tuned_qcond_rmse_g_kg": tr["qcond_rmse_kg_kg"] * KG_KG_TO_G_KG,
        }
    subcloud = {
        "apriori_subcloud_score": p.subcloud_score,
        "tuned_subcloud_score": t.subcloud_score,
        "apriori_sfc_rh": p.sfc_relative_humidity,
        "tuned_sfc_rh": t.sfc_relative_humidity,
        "apriori_sfc_delta_T_K": p.sfc_delta_T_K,
        "tuned_sfc_delta_T_K": t.sfc_delta_T_K,
        "apriori_sfc_driver_g_kg": p.sfc_driver_kg_kg * KG_KG_TO_G_KG,
        "tuned_sfc_driver_g_kg": t.sfc_driver_kg_kg * KG_KG_TO_G_KG,
        "apriori_evap_mm_day": p.evap_mm_day,
        "tuned_evap_mm_day": t.evap_mm_day,
    }
    return {
        **phys,
        **subcloud,
        **thermo,
        "scheme": res.scheme,
        "subsidence_solve": res.subsidence_solve,
        "subsidence_solve_status": res.subsidence_solve_status,
        "prior_score": p.score, "prior_T_rmse": p.T_rmse,
        "prior_qv_rmse": p.qv_rmse, "prior_cloud_rmse": p.cloud_rmse,
        "prior_precip_rmse": p.precip_rmse, "prior_precip_mm_day": p.precip_mm_day,
        "prior_verdict": _physical_verdict(p),
        "tuned_score": t.score, "tuned_T_rmse": t.T_rmse,
        "tuned_qv_rmse": t.qv_rmse, "tuned_cloud_rmse": t.cloud_rmse,
        "tuned_precip_rmse": t.precip_rmse, "tuned_precip_mm_day": t.precip_mm_day,
        "tuned_verdict": _physical_verdict(t),
        "score_improvement_pct": impr, "crm_precip_mm_day": p.precip_ref_mm_day,
        "n_tuned_params": len(res.records),
        "tuned_drift_T_K": t.drift_T_rmse_K,
        "tuned_madiab_mean_K": t.moist_adiabat_mean_abs_K,
        "tuned_cold_point_T_K": t.cold_point_T_K,
        "tuned_cold_point_z_km": t.cold_point_z_km,
        "tune_evals": res.signature.get("tune_evals", ""),
        "evals_per_param": (
            round(res.signature["tune_evals"] / len(res.records), 1)
            if res.signature.get("tune_evals") and res.records else ""),
    }


def write_csv(path: Path, results: list[SchemeResult], ref=None) -> None:
    """Machine-readable per-scheme metrics.

    ``ref`` is what turns the physical-unit columns from NaN into numbers, so
    the merge stage passes it; a caller that only wants the normalised scores
    may omit it.

    Rows are ordered by the objective each was TUNED under, not by the
    historical combined score: sorting a thermo-tuned table by a
    condensate-dominated number ranks it on something nobody minimised.  A
    non-finite or unstamped objective sorts last rather than first, so a
    crashed scheme cannot head the table.
    """
    def _sort_key(res: SchemeResult) -> float:
        objective = _row_objective(res)
        if objective not in camp.TUNE_OBJECTIVES:
            return float("inf")
        return camp.objective_value(res.tuned, objective)

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for res in sorted(results, key=_sort_key):
            writer.writerow(_row(res, ref))


def _finite_or_none(x: float):
    # A crashed scheme can carry a non-finite score; null keeps this advertised
    # artifact valid for strict JSON consumers (jq) while losing no real value.
    return x if np.isfinite(x) else None


def write_tuned_parameters(path: Path, results: list[SchemeResult]) -> None:
    payload = {}
    for res in results:
        if not res.records:
            continue
        payload[res.scheme] = {
            rec.scheme_key + "." + rec.parameter: {
                "default": rec.default, "tuned": rec.tuned,
                "bounds": [rec.lower, rec.upper], "units": rec.units,
                "score_default": _finite_or_none(rec.score_default),
                "score_tuned": _finite_or_none(rec.score_tuned),
            }
            for rec in res.records
        }
    # allow_nan=False: tuned_parameters.json is an externally consumed artifact,
    # not self-reloaded checkpoint state, so it must be strict JSON.
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


# One blurb per kernel arm, stated at the TOP of the summary so no table in this
# file can be read without knowing which arm produced it.
_ARM_BLURB = {
    "implicit_flux": (
        "**PRIMARY (matched-kernel) arm.** Every convection scheme that owns a "
        "compensating-subsidence mass-flux kernel is forced onto the "
        "conservative `implicit_flux` solve, so a score gap between two such "
        "schemes measures the SCHEME, not the transport kernel."
    ),
    "as_shipped": (
        "**SECONDARY (as-shipped) arm.** Every scheme keeps its own shipped "
        "`subsidence_solve` default — this is what users get today. Because "
        "Bechtold/EDMF/Kain-Fritsch ship `implicit_flux` while "
        "Tiedtke/Zhang-McFarlane/mass_flux ship the leaky `advective` solve, "
        "part of any score gap here measures the KERNEL rather than the scheme; "
        "read the PRIMARY arm for scheme physics."
    ),
    "advective": (
        "**Symmetric control arm.** Every kernel-capable scheme is forced onto "
        "the leaky `advective` solve, bounding the kernel's contribution from "
        "the other direction."
    ),
}
_MIXED_ARM_BLURB = (
    "**WARNING — CONFOUNDED TABLE.** These rows were NOT produced under one "
    "common kernel arm, so the ranking mixes scheme physics with transport-"
    "kernel differences and must NOT be read as a scheme intercomparison. "
    "Re-run each arm into its own `--outdir`."
)


def write_summary(path: Path, ref, results: list[SchemeResult], meta: dict) -> None:
    ordered = sorted(results, key=lambda r: r.tuned.score)
    lines: list[str] = []
    lines.append("# SCM RCE Convection-Scheme Intercomparison vs CRM (RCEMIP1)\n")
    lines.append(
        "Each convection scheme is run in the SCM RCE column and scored against "
        "the plane-CRM reference (horizontal/time mean of the last "
        f"{meta['last_reference_files']} CRM 3-D daily volumes). Metrics are the "
        "std-normalized, mass-weighted profile RMSE (T, q_v, condensate) plus a "
        "surface-precip term; **score** is their combination (lower = closer to "
        "CRM). *A priori* = scheme defaults; *tuned* = after derivative-free "
        "tuning of the scheme's extended-tier parameters against the CRM "
        "profiles. The evaluation budget is PER SCHEME (it scales with the "
        "scheme's parameter count, so a 1-parameter scheme is not compared "
        "against a 19-parameter one at the same number of draws) and is "
        "reported in the `#evals` column, not here.\n"
    )
    lines.append(
        f"SCM: radiation `{meta['radiation']}`, fixed SST 300 K, dt {meta['dt']:.0f} s, "
        f"{meta['days']:.0f} d ({meta['analysis_days']:.0f} d analysis window), "
        f"surface wind {meta['surface_wind_m_s']:.1f} m/s, large-scale forcing "
        f"`{meta['large_scale_forcing']}`. CRM equilibrium surface precip "
        f"{ref.precip_ref_mm_day:.3g} mm/day.\n"
    )
    # Kernel arm, stated before any number. Derived from the RESULTS (not from
    # meta) so a merged outdir that accidentally mixes arms is reported as mixed
    # rather than mislabelled with the current invocation's flag.
    arms = sorted({r.subsidence_solve for r in results})
    single_arm = len(arms) == 1
    arm_label = arms[0] if single_arm else "MIXED(" + ",".join(arms) + ")"
    lines.append(
        f"Kernel arm: `--subsidence-solve {arm_label}`. "
        + (_ARM_BLURB[arm_label] if single_arm and arm_label in _ARM_BLURB
           else _MIXED_ARM_BLURB)
        + " The per-scheme `kernel` column below reports what the shared "
        "override actually did: `forced:<scheme>=<solve>` was kernel-matched, "
        "while `not_applicable:<scheme>` has no such knob (sbm/dca/kuo have no "
        "mass-flux kernel; emanuel's shipped buoyancy-sorting path never calls "
        "it) and so was **not** kernel-matched in either arm.\n"
    )
    lines.append(
        "> The realism/equilibrium gate is **reported** (`verdict` column, derived "
        "from the campaign's own moist-adiabat / cold-point / equilibrium-drift "
        "thresholds) but not used to reject tuning trials, so each scheme is tuned "
        "to the best CRM match; `unphysical` marks a good-RMSE but non-RCE column.\n"
    )
    lines.append("## A priori vs tuned RMSE\n")
    lines.append(
        "| rank | scheme | kernel | score (prior→tuned) | T RMSE (p→t) | "
        "qv RMSE (p→t) | cloud RMSE (p→t) | precip mm/d (p→t) | Δscore % | "
        "verdict (p→t) | cold-pt T,z (tuned) | #params | #evals |"
    )
    lines.append("|---:|---|---|---|---|---|---|---|---:|---|---|---:|---:|")
    for i, res in enumerate(ordered, 1):
        p, t = res.prior, res.tuned
        row = _row(res, ref)
        lines.append(
            f"| {i} | {res.scheme} "
            f"| {res.subsidence_solve_status or res.subsidence_solve} "
            f"| {_fmt(p.score)}→{_fmt(t.score)} "
            f"| {_fmt(p.T_rmse)}→{_fmt(t.T_rmse)} "
            f"| {_fmt(p.qv_rmse)}→{_fmt(t.qv_rmse)} "
            f"| {_fmt(p.cloud_rmse)}→{_fmt(t.cloud_rmse)} "
            f"| {_fmt(p.precip_mm_day, '.3g')}→{_fmt(t.precip_mm_day, '.3g')} "
            f"| {_fmt(row['score_improvement_pct'], '.1f')} "
            f"| {row['prior_verdict']}→{row['tuned_verdict']} "
            f"| {_fmt(t.cold_point_T_K, '.0f')} K, {_fmt(t.cold_point_z_km, '.1f')} km "
            f"| {len(res.records)} "
            f"| {row['tune_evals']} |"
        )
    lines.append(f"\nCRM reference surface precip: {ref.precip_ref_mm_day:.3g} mm/day.\n")
    lines.append(
        "Verdict thresholds (campaign RCE realism gate): equilibrium drift_T ≤ "
        f"{camp.EQUIL_T_TOL_K:g} K, |T−T_moist| mean ≤ {camp.MADIAB_MEAN_TOL_K:g} K / "
        f"max ≤ {camp.MADIAB_MAX_TOL_K:g} K, cold point in "
        f"[{camp.COLD_POINT_MIN_K:g}, {camp.COLD_POINT_MAX_K:g}] K at "
        f"[{camp.TROP_MIN_Z_KM:g}, {camp.TROP_MAX_Z_KM:g}] km.\n"
    )

    lines.append("## Tuned parameters\n")
    for res in ordered:
        if not res.records:
            lines.append(f"- **{res.scheme}**: no extended-tier tunable parameters.")
            continue
        lines.append(f"- **{res.scheme}** (score {_fmt(res.prior.score)}→{_fmt(res.tuned.score)}):")
        for rec in res.records:
            lines.append(
                f"  - `{rec.scheme_key}.{rec.parameter}` "
                f"{_fmt(rec.default)} → {_fmt(rec.tuned)} {rec.units} "
                f"(bounds [{_fmt(rec.lower)}, {_fmt(rec.upper)}])"
            )
    lines.append("")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Plotting — CRM (black), a-priori (red dashed), tuned (red solid)
# --------------------------------------------------------------------------- #
_CRM_COLOR = "#000000"
_RED = "#d62728"
_M_PER_KM = camp.M_PER_KM
_KG_TO_G = camp.MSE_KJ_TO_J  # kg/kg -> g/kg is also x1000


def _panels(ref, run):
    T = np.asarray(run.T_profile)
    qv = np.asarray(run.qv_profile) * _KG_TO_G
    qc = np.asarray(run.qcond_profile) * _KG_TO_G
    return [
        ("T [K]", ref.T_ref, T),
        ("q$_v$ [g/kg]", ref.qv_ref * _KG_TO_G, qv),
        ("condensate [g/kg]", ref.qcond_ref * _KG_TO_G, qc),
    ]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_scheme(path: Path, ref, res: SchemeResult) -> None:
    if not res.prior.T_profile:
        return
    plt = _plt()
    z_km = ref.z_m / _M_PER_KM
    ztop = float(np.nanmax(z_km))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0))
    prior_panels = _panels(ref, res.prior)
    tuned_panels = _panels(ref, res.tuned)
    for ax, (label, ref_prof, prior_prof), (_l, _r, tuned_prof) in zip(
        axes, prior_panels, tuned_panels
    ):
        ax.plot(ref_prof, z_km, color=_CRM_COLOR, lw=2.2, label="CRM (RCEMIP1)")
        ax.plot(prior_prof, z_km, color=_RED, lw=1.8, ls="--", label="SCM a priori")
        ax.plot(tuned_prof, z_km, color=_RED, lw=2.0, ls="-", label="SCM tuned")
        ax.set_xlabel(label)
        ax.set_ylim(0.0, ztop)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("z [km]")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"convection = {res.scheme}   "
        f"score {_fmt(res.prior.score)} → {_fmt(res.tuned.score)}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_all(path: Path, ref, results: list[SchemeResult]) -> None:
    # Only plot schemes that actually produced profiles: a crashed scheme has
    # empty T/qv/qcond profiles and would raise "x and y must have same first
    # dimension" in the shared-axes montage.  Filter EVERY result, not just the
    # first, since a valid scheme can sort ahead of a crashed one.
    plottable = [r for r in results if r.prior.T_profile and r.tuned.T_profile]
    ordered = sorted(plottable, key=lambda r: r.tuned.score)
    if not ordered:
        return
    plt = _plt()
    z_km = ref.z_m / _M_PER_KM
    ztop = float(np.nanmax(z_km))
    n = len(ordered)
    fig, axes = plt.subplots(n, 3, figsize=(11.0, 3.1 * n), squeeze=False)
    for i, res in enumerate(ordered):
        prior_panels = _panels(ref, res.prior)
        tuned_panels = _panels(ref, res.tuned)
        for j, ((label, ref_prof, prior_prof), (_l, _r, tuned_prof)) in enumerate(
            zip(prior_panels, tuned_panels)
        ):
            ax = axes[i][j]
            ax.plot(ref_prof, z_km, color=_CRM_COLOR, lw=2.0, label="CRM")
            ax.plot(prior_prof, z_km, color=_RED, lw=1.6, ls="--", label="a priori")
            ax.plot(tuned_prof, z_km, color=_RED, lw=1.8, ls="-", label="tuned")
            ax.set_ylim(0.0, ztop)
            ax.grid(alpha=0.2)
            if i == n - 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel(f"{res.scheme}\nz [km]", fontsize=9)
        axes[i][2].text(
            1.02, 0.5,
            f"score\n{_fmt(res.prior.score)}\n→ {_fmt(res.tuned.score)}",
            transform=axes[i][2].transAxes, fontsize=8, va="center",
        )
    axes[0][0].legend(loc="best", fontsize=7)
    fig.suptitle("SCM RCE convection intercomparison vs CRM — a priori (dashed) vs tuned (solid)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, exposed so tests exercise the REAL parser (a hand-rolled
    namespace would keep passing after a flag is renamed or dropped)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=camp.DEFAULT_REFERENCE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--days", type=float, default=camp.DEFAULT_DAYS)
    parser.add_argument("--dt", type=float, default=camp.DEFAULT_DT_S)
    parser.add_argument("--analysis-days", type=float, default=camp.DEFAULT_ANALYSIS_DAYS)
    parser.add_argument("--last-reference-files", type=int,
                        default=camp.DEFAULT_LAST_REFERENCE_FILES)
    parser.add_argument("--tune-evals", type=int, default=48)
    parser.add_argument("--tune-seed", type=int, default=20260705)
    parser.add_argument(
        "--radiation", default="rrtmgp", choices=("rrtmgp", "gray"),
        help=(
            "SCM radiation. Default `rrtmgp` is the RCEMIP standard (broadband) and "
            "matches the `rrtmgp`-default plane CRM (run_rcemip_plane.py); the CRM "
            "reference MUST be regenerated with rrtmgp for an apples-to-apples "
            "stratosphere. `gray` is a fast approximation that under-drives "
            "convection and leaves a warm cold point."
        ),
    )
    parser.add_argument("--radiation-update-interval-steps", type=int, default=None)
    parser.add_argument(
        "--turbulence", default=camp.BASELINE_SCHEMES["turbulence"],
        choices=camp.SCHEME_SWEEPS["turbulence"],
        help=(
            "Boundary-layer scheme, held FIXED across every convection scheme. "
            "It is what maintains the sub-cloud layer, so switching it is a "
            "one-variable experiment in its own right and is recorded in the "
            "checkpoint signature."
        ),
    )
    parser.add_argument(
        "--tune-mode", default="convection", choices=TUNE_MODES,
        help=(
            "`convection` searches every extended-tier parameter of the active "
            "convection scheme (the historical arm). `focused` searches ONE "
            "named cross-category set instead — see --focused-include."
        ),
    )
    parser.add_argument(
        "--param-set", default="extended", choices=camp.PARAM_SETS,
        help=(
            "WHICH parameters of the convection scheme the tuner may move. "
            "`core`/`extended`/`aggressive` are the registry's own tiers. "
            "`physical` is the DERIVATIVE-FREE calibration set: every "
            "aggressive-tier parameter EXCEPT those categorised `numerics`, "
            "PLUS the tier-0 parameters excluded only because their AD "
            "gradient vanishes (the CAPE trigger of eight of the ten schemes). "
            "Use it when the campaign claims to have tuned the scheme's "
            "physics; `aggressive` is neither a superset nor a subset of that. "
            "Enters the checkpoint signature."
        ),
    )
    parser.add_argument(
        "--tune-refine-frac", type=float, default=0.0,
        help=(
            "Fraction of the evaluation budget spent on a LOCAL COORDINATE "
            "REFINEMENT around the best random draw instead of on more random "
            "draws. 0.0 (default) reproduces the historical pure random "
            "search. A uniform search over ~20 parameters leaves points ~0.76 "
            "of the range apart on every axis, so without this the reported "
            "optimum is the best of N lottery tickets rather than a point that "
            "no single-parameter move improves. Enters the signature."
        ),
    )
    parser.add_argument(
        "--objective", default="combined", choices=camp.TUNE_OBJECTIVES,
        help=(
            "What the tuner MINIMISES. `combined` is the full-column score, "
            "measured to be 87-100 %% condensate, which cannot see the "
            "sub-cloud layer. `subcloud` is the T/q_v error below "
            "--subcloud-top-m, normalized by the reference's own spread there. "
            "Both are reported for every run whichever is selected."
        ),
    )
    parser.add_argument(
        "--focused-include", action="append", default=None, metavar="SCHEME_KEY.FIELD",
        help=(
            "Registry-qualified parameter to tune in --tune-mode focused; "
            "repeatable. Omit to use the sub-cloud default set for the active "
            "(turbulence, microphysics, convection) triple. A name that matches "
            "no parameter of an active scheme is a hard error."
        ),
    )
    parser.add_argument(
        "--emanuel-unsaturated-downdraft", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Switch ON Emanuel's unsaturated-downdraft re-evaporation, which "
            "ships OFF behind a static Python branch. With it off, "
            "EmanuelConfig.downdraft_efficiency is read by no executed code. "
            "The branch re-evaporates a fraction of the column condensate "
            "below the LCL, which COOLS and MOISTENS the sub-cloud layer, so "
            "it is different physics and enters the checkpoint signature."
        ),
    )
    parser.add_argument(
        "--subcloud-top-m", type=float, default=camp.DEFAULT_SUBCLOUD_TOP_M,
        help=(
            "Top of the layer scored as sub-cloud [m]. The default sits below "
            "the CRM reference's own shallow-cumulus condensate maximum at "
            "1.3 km."
        ),
    )
    parser.add_argument(
        "--microphysics", default=camp.BASELINE_SCHEMES["microphysics"],
        choices=camp.SCHEME_SWEEPS["microphysics"],
        help=(
            "SCM microphysics, held FIXED across every convection scheme. The "
            "default `kessler` is WARM-RAIN ONLY: it carries no ice, so the "
            "IFS/SAM homogeneous-freezing ice-super-saturation allowance is "
            "inert and the upper troposphere is biased for every scheme "
            "alike. Use `morrison` (SAM M2005 flavor) to score the cold point "
            "against an ice-carrying CRM reference."
        ),
    )
    parser.add_argument(
        "--hard-saturation-adjustment", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the in-scheme iterated saturation adjustment (the IFS "
            "'no liquid super-saturation' half) for the selected microphysics. "
            "Default off = the smooth-sigmoid path, which leaves a few percent "
            "standing super-saturation. sdm/fast_sbm reject the flag."
        ),
    )
    parser.add_argument(
        "--bl-anchor-top-m", type=float,
        default=camp.DEFAULT_SCM_RCE_BL_TOP_M,
        help=(
            "Depth [m] over which the boundary layer is anchored to an "
            "SST-rooted lapse profile. The shipped 0.0 anchors EXACTLY the "
            "lowest level (the mask is <=, and z is 0 there), so T_a == SST "
            "and the sensible heat flux is identically zero. Pass a NEGATIVE "
            "value to disable the anchor entirely, which is what a zero depth "
            "was meant to express."
        ),
    )
    parser.add_argument("--schemes", default=None,
                        help="comma-separated convection scheme subset")
    parser.add_argument("--scm-microphysics-substeps", type=int,
                        default=camp.DEFAULT_SCM_MICROPHYSICS_SUBSTEPS)
    parser.add_argument("--scm-convection-substeps", type=int,
                        default=camp.DEFAULT_SCM_CONVECTION_SUBSTEPS)
    parser.add_argument("--surface-wind-m-s", type=float,
                        default=camp.DEFAULT_SCM_RCE_SURFACE_WIND_M_S)
    parser.add_argument("--coriolis-s-inv", type=float,
                        default=camp.DEFAULT_SCM_RCE_CORIOLIS_S_INV)
    parser.add_argument("--large-scale-forcing", default=camp.DEFAULT_SCM_RCE_LARGE_SCALE_FORCING,
                        choices=camp.SCM_RCE_LARGE_SCALE_FORCING_CHOICES)
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke run (short, 2 schemes, few evals)")
    parser.add_argument("--merge-only", action="store_true",
                        help="skip running; build summary/CSV/combined plot from "
                             "existing scheme_*.json checkpoints")
    parser.add_argument("--no-merge", action="store_true",
                        help="run schemes + write per-scheme json/png but skip the "
                             "aggregate summary/CSV/combined plot (for parallel workers)")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help=(
            "Merge even when scheme checkpoints are missing or were written "
            "under a different protocol. Prints what is wrong and continues; "
            "the resulting table is NOT a campaign ranking."
        ),
    )
    parser.add_argument("--force", action="store_true",
                        help="re-run schemes even if a scheme_*.json checkpoint exists")
    parser.add_argument(
        "--subsidence-solve", default="as_shipped",
        choices=list(camp.SUBSIDENCE_SOLVE_MODES),
        help=(
            "Vertical-transport kernel ARM. Convection schemes do not share a "
            "compensating-subsidence kernel by default — Bechtold/EDMF/"
            "Kain-Fritsch ship the conservative `implicit_flux` solve while "
            "Tiedtke/Zhang-McFarlane/mass_flux ship the leaky `advective` one — "
            "so an as-shipped ranking partly measures the KERNEL, not the "
            "scheme. Run the campaign TWICE into distinct --outdir: "
            "`implicit_flux` = PRIMARY arm, every kernel-capable scheme forced "
            "onto the conservative solve (isolates scheme physics); "
            "`as_shipped` (default) = SECONDARY arm, every scheme keeps its own "
            "shipped default (what users get today); `advective` = symmetric "
            "control. Schemes with no such knob (sbm/dca/kuo, and emanuel whose "
            "shipped path bypasses the kernel) are reported "
            "`not_applicable:<scheme>`, never silently skipped. The arm is part "
            "of the checkpoint signature, so one arm's result can never be "
            "reused for another."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.radiation_update_interval_steps is None:
        args.radiation_update_interval_steps = (
            camp.DEFAULT_RRTMGP_UPDATE_INTERVAL_STEPS if args.radiation == "rrtmgp" else 1
        )

    if args.schemes:
        selected = tuple(s.strip() for s in args.schemes.split(",") if s.strip())
        unknown = [s for s in selected if s not in CONVECTION_SCHEMES]
        if unknown:
            raise SystemExit(f"Unknown convection schemes {unknown}; "
                             f"known={list(CONVECTION_SCHEMES)}")
        schemes = selected
    else:
        schemes = CONVECTION_SCHEMES
    if args.quick:
        args.days = min(args.days, 0.05)
        args.analysis_days = min(args.analysis_days, args.days)
        args.tune_evals = min(args.tune_evals, 2)
        schemes = schemes[:2]

    args.outdir.mkdir(parents=True, exist_ok=True)
    meta_path = args.outdir / "run_meta.json"

    # In --merge-only we aggregate checkpoints from a PRIOR run: rebuild the CRM
    # reference from that run's recorded config (run_meta.json), not the current
    # CLI defaults, so the summary/plots compare against the same target.
    saved_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if args.merge_only and saved_meta:
        args.analysis_days = saved_meta.get("analysis_days", args.analysis_days)
        args.last_reference_files = saved_meta.get(
            "last_reference_files", args.last_reference_files)
        if "reference_dir" in saved_meta:
            args.reference_dir = Path(saved_meta["reference_dir"])

    ref = camp.build_reference_profiles(
        args.reference_dir, args.last_reference_files,
        precip_analysis_days=args.analysis_days,
    )

    meta = dict(
        radiation=args.radiation, dt=args.dt, days=args.days,
        analysis_days=args.analysis_days,
        surface_wind_m_s=args.surface_wind_m_s,
        large_scale_forcing=args.large_scale_forcing,
        last_reference_files=args.last_reference_files,
        reference_dir=str(args.reference_dir),
        subsidence_solve=args.subsidence_solve,
        microphysics=args.microphysics,
        hard_saturation_adjustment=args.hard_saturation_adjustment,
        bl_anchor_top_m=args.bl_anchor_top_m,
        scm_microphysics_substeps=args.scm_microphysics_substeps,
        scm_convection_substeps=args.scm_convection_substeps,
        turbulence=args.turbulence,
        tune_mode=args.tune_mode,
        objective=args.objective,
        subcloud_top_m=args.subcloud_top_m,
    )
    if args.merge_only and saved_meta:
        meta = {**meta, **saved_meta}

    run_sig = _run_signature(args)
    if not args.merge_only:
        for scheme in schemes:
            ckpt = _scheme_json_path(args.outdir, scheme)
            if ckpt.exists() and not args.force:
                if _signature_compatible(_checkpoint_signature(ckpt), run_sig):
                    print(f"[scheme] convection={scheme} — checkpoint exists, skip", flush=True)
                    continue
                print(f"[scheme] convection={scheme} — checkpoint config mismatch "
                      f"(e.g. --quick vs full), recomputing", flush=True)
            print(f"[scheme] convection={scheme} ...", flush=True)
            res = evaluate_scheme(
                scheme, ref,
                days=args.days, dt=args.dt, analysis_days=args.analysis_days,
                tune_evals=args.tune_evals, seed=args.tune_seed,
                scm_microphysics_substeps=args.scm_microphysics_substeps,
                scm_convection_substeps=args.scm_convection_substeps,
                surface_wind_m_s=args.surface_wind_m_s,
                coriolis_s_inv=args.coriolis_s_inv,
                large_scale_forcing=args.large_scale_forcing,
                radiation=args.radiation,
                radiation_update_interval_steps=args.radiation_update_interval_steps,
                subsidence_solve=args.subsidence_solve,
                microphysics=args.microphysics,
                hard_saturation_adjustment=args.hard_saturation_adjustment,
                bl_anchor_top_m=args.bl_anchor_top_m,
                turbulence=args.turbulence,
                tune_mode=args.tune_mode,
                objective=args.objective,
                focused_include=tuple(
                    args.focused_include
                    if args.focused_include
                    else default_focused_include(
                        turbulence=args.turbulence,
                        microphysics=args.microphysics,
                        convection=scheme,
                        emanuel_unsaturated_downdraft=(
                            args.emanuel_unsaturated_downdraft),
                    )
                ),
                subcloud_top_m=args.subcloud_top_m,
                param_set=args.param_set,
                tune_refine_frac=args.tune_refine_frac,
                emanuel_unsaturated_downdraft=(
                    args.emanuel_unsaturated_downdraft and scheme == "emanuel"),
            )
            res.signature = run_sig
            save_scheme_result(args.outdir, res, run_sig)  # checkpoint before plotting
            plot_scheme(args.outdir / f"profiles_{scheme}.png", ref, res)
            print(f"    prior score={_fmt(res.prior.score)} "
                  f"-> tuned score={_fmt(res.tuned.score)} "
                  f"({len(res.records)} params) "
                  f"[kernel {res.subsidence_solve_status}]", flush=True)
            # The sub-cloud arm's actual target, next to the score it was tuned
            # on, so a reader never has to infer whether the layer moved.
            print(f"    sub-cloud score={_fmt(res.prior.subcloud_score)} "
                  f"-> {_fmt(res.tuned.subcloud_score)} | "
                  f"RH {_fmt(res.prior.sfc_relative_humidity, '.3f')} "
                  f"-> {_fmt(res.tuned.sfc_relative_humidity, '.3f')} | "
                  f"SST-T_air {_fmt(res.prior.sfc_delta_T_K, '.2f')} "
                  f"-> {_fmt(res.tuned.sfc_delta_T_K, '.2f')} K | "
                  f"E {_fmt(res.prior.evap_mm_day, '.2f')} "
                  f"-> {_fmt(res.tuned.evap_mm_day, '.2f')} mm/day "
                  f"(CRM RH 0.752, 3.06 K, 2.73)", flush=True)
        # ATOMIC: the campaign runs one process per scheme against a shared
        # --outdir, so several finish at once and write this same file. Two
        # interleaved write_text calls can leave a truncated file, and
        # --merge-only reads it to rebuild the reference. Write-then-rename is
        # atomic within a directory on POSIX.
        #
        # `meta` deliberately holds only PROTOCOL fields that are identical
        # across schemes. The tuning budget is NOT one of them — it scales with
        # each scheme's parameter count — so it lives on the per-scheme
        # checkpoint signature and is reported per row. Putting it here would
        # publish the last finisher's budget as if it were the campaign's.
        _tmp = meta_path.with_suffix(f".json.{os.getpid()}.tmp")
        _tmp.write_text(json.dumps(meta, indent=2))
        _tmp.replace(meta_path)

    if args.no_merge:
        print(f"[done] ran {len(schemes)} scheme(s); merge skipped (--no-merge)")
        return 0

    # Merge: aggregate every scheme checkpoint present in outdir (covers schemes
    # run by parallel worker processes, not just this invocation's subset).
    results = load_all_scheme_results(args.outdir, CONVECTION_SCHEMES)
    if not results:
        raise SystemExit(f"No scheme_*.json checkpoints found in {args.outdir}")
    # In --merge-only the current args are NOT the arm's args (the merge job
    # supplies only the outdir and the reference), so the checkpoints are
    # required to agree with each other and with the reference being scored
    # against, rather than with this invocation.
    _guard_merge_inputs(
        results, None if args.merge_only else run_sig,
        allow_partial=args.allow_partial, reference_dir=args.reference_dir)
    write_csv(args.outdir / "intercomparison.csv", results, ref)
    # Second copy under the name the paper figure script reads, so the
    # figures are built from THIS table rather than a hand-copied one.
    write_csv(args.outdir / "summary_table.csv", results, ref)
    write_tuned_parameters(args.outdir / "tuned_parameters.json", results)
    write_summary(args.outdir / "summary.md", ref, results, meta)
    plot_all(args.outdir / "profiles_all_convection.png", ref, results)
    print(f"[done] merged {len(results)} scheme(s) -> {args.outdir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
