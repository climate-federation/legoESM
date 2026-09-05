#!/usr/bin/env python3
"""Does NEMO's REFERENCE thickness weighting own GYRE's kt2 first-over-bar?

Round 27 left this as its open question 1 and named it the GYRE merge blocker:
NEMO forms the barotropic slow forcing from REFERENCE thicknesses,

``Ue_rhs = SUM( e3u_0(:,:,1:jpkm1) * uu(:,:,1:jpkm1,Krhs) * umask ) * r1_hu_0``

while legoESM weights with LIVE thicknesses (``min_cell_to_uface`` of
``compute_layer_thickness``) and divides by the live column sum.  Round 27
measured that on LOCK at kt=1, where NEMO's entry ``ssh`` is exactly ``0.0`` so
the two coincide bitwise, and asserted -- without measuring -- that the
difference is ``O(eta/H)`` at kt >= 2.

Rule 0 first, and it moves the citation.  ``stp2d.F90:177`` opens a
``SELECT CASE( n_dynadv )`` with two arms: ``:178`` ``CASE( np_VEC_c2,
np_LIN_dyn )`` whose body ``:180-181`` ASSIGNS the depth mean, and ``:183``
``CASE ( np_FLX_c2, np_FLX_up3 )`` whose body ``:185-186`` CUMULATES it onto
the 2-D advective RHS.  GYRE sets ``ln_dynadv_vec = .true.``, so GYRE executes
``stp2d.F90:180``; LOCK and OVERFLOW set ``ln_dynadv_up3 = .true.`` and execute
``:185``.  Both arms use the reference ``e3u_0`` and the reference
``r1_hu_0``, so the question is unchanged and only the line moves.

WHAT THIS PROBE CAN AND CANNOT SCORE, stated before any number.  The GYRE
oracle V2 record set holds ``oracle_bt_substeps_kt00000001.bin`` and
``oracle_rhs_kt00000001.bin`` ONLY, and ``read_bt_substeps`` requires
``kt == 1``.  There is no dumped GYRE slow-forcing frame and no dumped GYRE
3-D momentum RHS at kt=2, so a kt=2 arm CANNOT be scored against the oracle.
Two measurements are therefore reported and labelled differently:

* **kt=1, ORACLE-SCORED** -- the production ``slow_u``/``slow_v`` frame with
  a BIT census, the production 3-D momentum RHS, and the whole kt=1
  external-mode trace (16 frames x 50 substeps = 800 rows) scored in NEMO's
  own recurrence order, so the first frame over bar is NAMED by measurement
  rather than guessed.
* **kt=2 and kt=1, ARM-TO-ARM, NOT oracle-scored** -- the depth-mean
  SUB-STATEMENT under three weightings: the model's divide, the model's
  reciprocal, and a ``reference`` arm built from NEMO's OWN ``mesh_mask``
  ``e3u_0``/``e3v_0``/``umask``/``vmask`` with ``hu_0``/``hv_0`` accumulated
  exactly as ``domain.F90:142`` does and ``r1_hu_0`` formed exactly as
  ``domain.F90:159`` does.

  These arms are NOT scored against NEMO's frame, and a first draft of this
  probe wrongly did score them: legoESM's ``F_slow`` is the depth mean PLUS
  wind, drag and biharmonic increments, so an arm holding only the depth mean
  compares an INCOMPLETE statement against a COMPLETE frame.  That is Rule 2's
  second blind spot -- two things of the same name are not the same quantity
  -- and it made every arm read 580/580 unequal at ONE shared magnitude, which
  is the missing wind term and not the weighting.  What the arms measure is
  the difference between two WEIGHTINGS of one shared operand.

The mechanism the sizing tests: under z-star the live face thickness is
``h_u(k) = e3u_0(k) * s_u`` with ``s_u`` a SINGLE per-column scalar --
``min_cell_to_uface`` of ``dz_ref(k) * (1 + eta/H)`` selects the same
neighbouring column at every ``k`` because ``dz_ref(k) > 0`` -- so ``s_u``
cancels between the numerator and the denominator of the depth mean.  If that
holds, the reference-versus-live residual is ROUNDOFF and not ``O(eta/H)``.

Scope limits, written down rather than discovered later:

* the reference arm accumulates ``hu_0`` sequentially over ``jk = 1..jpkm1``
  as ``domain.F90:142`` does, but the inner ``SUM(...)`` intrinsic of
  ``stp2d.F90:180`` has an unspecified association order and legoESM's
  reduction is XLA's.  Bit-level agreement of the two reductions is therefore
  NOT claimed by this probe; what it measures is the WEIGHTING.
* ``mesh_mask.nc`` supplies ``e3u_0``/``umask`` on the interior only, which is
  exactly the scored set, so no halo convention enters.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from nemo_testcase_overflow_barotropic_gate import state_from_oracle_entry
from nemo_testcase_phase3_stage_sweep_gate import GateError, git_sha, require, sha256
from nemo_testcase_l2_gyre_phase3_gate import (
    DIMS,
    _trace_native,
    _xyz,
    expected_masks,
    read_bt_substeps,
    read_entry,
    score,
)

CASE = "GYRE-zco"
# NEMO's own recurrence order inside one barotropic substep, as the oracle
# record writes it; scoring in this order is what makes "first over bar" a
# statement about NEMO's execution rather than about a dict's iteration.
BT_TRACE_ORDER = (
    "eta_entry", "u_entry", "v_entry", "eta_mid", "u_mid", "v_mid",
    "slow_u", "slow_v", "eta_exit", "eta_pgf",
    "pgf_u", "pgf_v", "trd_u", "trd_v", "u_exit", "v_exit",
)
ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round19_oracle_v2_external")


def read_rhs_payload(path: Path) -> dict:
    """The 3-D momentum RHS arrays the GYRE gate's ``read_rhs`` validates but drops.

    Same header contract as ``nemo_testcase_l2_gyre_phase3_gate.read_rhs``;
    that function returns only the registry level, so the payload is read here
    rather than duplicating the validation.
    """
    import struct

    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nrhs, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic")
    require((version, kt, nrhs, nx, ny, nz, bits) == (1, 1, 3, *DIMS, 64),
            f"{path}: bad header")
    count = nx * ny * nz
    require(values.size == 2 * count, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {"u": _xyz(values[:count], nx, ny, nz),
            "v": _xyz(values[count:], nx, ny, nz)}


def nemo_reference_weights(mesh: Path, nlev: int) -> dict:
    """``e3u_0``/``umask`` and NEMO's own ``r1_hu_0``, read from the oracle mesh.

    ``domain.F90:140-146`` accumulates ``hu_0 = hu_0 + e3u_0(jk)*umask(jk)``
    sequentially over ``jk = 1..jpkm1``; ``domain.F90:159`` then forms
    ``r1_hu_0 = ssumask / ( hu_0 + 1 - ssumask )``, which is NOT ``1/hu_0`` on
    a face where that shift rounds.  Both are reproduced literally.
    """
    import netCDF4

    with netCDF4.Dataset(mesh) as data:
        def field(name):
            return np.asarray(data.variables[name][0], dtype=np.float64)

        e3u_0 = field("e3u_0").transpose(1, 2, 0)          # (j, i, k)
        e3v_0 = field("e3v_0").transpose(1, 2, 0)
        umask = field("umask").transpose(1, 2, 0)
        vmask = field("vmask").transpose(1, 2, 0)
        ssumask = field("umaskutil")
        ssvmask = field("vmaskutil")
    require(e3u_0.shape[2] > nlev, f"{mesh}: only {e3u_0.shape[2]} levels")

    out = {}
    for tag, e3, mask, ss in (("u", e3u_0, umask, ssumask),
                              ("v", e3v_0, vmask, ssvmask)):
        h0 = np.zeros(e3.shape[:2], dtype=np.float64)
        for jk in range(nlev):                     # domain.F90:142, jk=1..jpkm1
            h0 = h0 + e3[:, :, jk] * mask[:, :, jk]
        out[f"e3{tag}_0"] = e3[:, :, :nlev]
        out[f"{tag}mask"] = mask[:, :, :nlev]
        out[f"h{tag}_0"] = h0
        out[f"r1_h{tag}_0"] = ss / (h0 + 1.0 - ss)      # domain.F90:159/160
        out[f"ss{tag}mask"] = ss
    return out


def live_depth_mean(du_dt, eta, h_bathy, z_coord, config, grid, face_mask,
                    *, tag: str, association: str = "model"):
    """legoESM's own depth-mean statement, through the model's own helpers.

    Identical in shape to the LOCK walk's ``depth_mean_statement``; the ``v``
    face is added here because GYRE has wet V faces and LOCK does not.  The
    arm-equality check against the captured production frame is what proves
    nothing is re-derived.
    """
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface,
        min_cell_to_vface,
        nemo_source_round,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    if association not in ("model", "nemo_reciprocal"):
        raise ValueError(f"unknown association {association!r}")
    if tag not in ("u", "v"):
        raise ValueError(f"unknown face {tag!r}")

    h_k = compute_layer_thickness(eta, h_bathy, z_coord,
                                  min_water_column_m=config.min_water_column_m)
    h_f = min_cell_to_uface(h_k) if tag == "u" else min_cell_to_vface(h_k, grid)
    pair = jnp.sum(jnp.stack([h_f, jnp.asarray(du_dt) * h_f], axis=-1), axis=-2)
    depth = jnp.maximum(pair[..., 0], 1e-10)
    if association == "model":
        mean = pair[..., 1] / depth
    else:
        mean = pair[..., 1] * nemo_source_round(1.0 / depth)
    return np.asarray(mean * face_mask)


def reference_depth_mean(du_dt, weights, tag: str):
    """``stp2d.F90:180`` literally: reference thickness, reference reciprocal.

    ``Ue_rhs(ji,jj) = SUM( e3u_0(ji,jj,1:jpkm1) * uu(ji,jj,1:jpkm1,Krhs)
    * umask(ji,jj,1:jpkm1) ) * r1_hu_0(ji,jj)``
    """
    if tag not in ("u", "v"):
        raise ValueError(f"unknown face {tag!r}")
    e3 = weights[f"e3{tag}_0"]
    mask = weights[f"{tag}mask"]
    total = np.sum(e3 * np.asarray(du_dt) * mask, axis=-1)
    return total * weights[f"r1_h{tag}_0"]


def _bits(a, b, mask):
    a = np.asarray(a, dtype=np.float64)[mask]
    b = np.asarray(b, dtype=np.float64)[mask]
    return int(np.count_nonzero(a.view(np.int64) != b.view(np.int64)))


def run(*, plant: bool = False, allow_dirty: bool = False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    legoesm_git_sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from nemo_testcase_l2_gyre_phase3_gate import _surface_forcings

    substeps_path = ROOT / "oracle_bt_substeps_kt00000001.bin"
    rhs_path = ROOT / "oracle_rhs_kt00000001.bin"
    entry1_path = ROOT / "oracle_step_entry_kt00000001.bin"
    entry2_path = ROOT / "oracle_step_entry_kt00000002.bin"
    mesh_path = ROOT / "mesh_mask.nc"
    for path in (substeps_path, rhs_path, entry1_path, entry2_path, mesh_path):
        require(path.is_file(), f"missing {path}")

    # The record set bounds the claim; assert it rather than assume it.
    kt2_records = sorted(
        p.name for p in ROOT.glob("oracle_bt_substeps_kt*.bin"))
    require(kt2_records == ["oracle_bt_substeps_kt00000001.bin"],
            f"a kt>=2 GYRE slow-forcing record now exists ({kt2_records}); "
            "the kt=2 arm must be SCORED against it instead of sized")

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config
    require(cfg.pgf_scheme is not None, "card has no resolved pgf_scheme")
    nlev = card.recipe.z_coord.n_levels
    masks = expected_masks(card)
    u_mask2d = masks["u"][..., 0]
    v_mask2d = masks["v"][..., 0]

    oracle = read_bt_substeps(substeps_path)
    rhs = read_rhs_payload(rhs_path)
    weights = nemo_reference_weights(mesh_path, nlev)

    entry1 = read_entry(entry1_path)
    entry2 = read_entry(entry2_path)
    # P1: the fact that decides whether kt=1 can separate the arms at all.
    rest = {"kt1_entry_abs_max_ssh": float(np.max(np.abs(entry1["ssh"]))),
            "kt2_entry_abs_max_ssh": float(np.max(np.abs(entry2["ssh"])))}

    # Reuse the campaign's shared seeder rather than a second stagger map.
    plain = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    base = plain._seed_tke_preclosure_carry(card.recipe.initial_state)
    seeded1 = state_from_oracle_entry(base, entry1, masks)
    seeded2 = state_from_oracle_entry(base, entry2, masks)
    freshwater1, surface1 = _surface_forcings(card, seeded1, 1)

    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_barotropic_substeps=True))
    trace = trace_model.step(seeded1, card.dt_s, freshwater=freshwater1,
                             surface_forcing=surface1)
    require(isinstance(trace.substeps, dict),
            "compiled step did not expose keyed barotropic substeps")
    trace_frames = {name: _trace_native(np.asarray(trace.substeps[name]), name)
                    for name in BT_TRACE_ORDER}
    captured = {name: trace_frames[name][0] for name in ("slow_u", "slow_v")}
    del trace_model
    jax.clear_caches()

    model = plain
    tend1 = model.tendencies(seeded1, dt=card.dt_s, momentum_only=True)
    model_du = np.asarray(tend1.du_dt.data)
    model_dv = np.asarray(tend1.dv_dt.data)

    nemo_du = np.zeros_like(model_du)
    nemo_dv = np.zeros_like(model_dv)
    nemo_du[:, 1:, :] = rhs["u"][..., :nlev]
    nemo_dv[1:, :, :] = rhs["v"][..., :nlev]
    # PRECONDITION, measured, not assumed.  GYRE enters kt=1 at rest, so
    # dyn_keg / dyn_zad / dyn_vor are products with a zero entry velocity and
    # the dumped 3-D RHS is dyn_hpg alone -- but it is NOT zero: GYRE's
    # usrdef_istate temperature varies with latitude, so the pressure gradient
    # has something to act on.  A first draft of this probe asserted the frame
    # was identically zero (it had targeted the DEEPEST wet face, whose RHS is
    # zero, and read that as the whole array).  The measured value is
    # published here so the claim cannot drift again.
    nemo_rhs_abs_max = {"u": float(np.max(np.abs(nemo_du))),
                        "v": float(np.max(np.abs(nemo_dv)))}
    kt1_arms_informative = max(nemo_rhs_abs_max.values()) > 0.0

    # kt=1 ARM-TO-ARM, not oracle-scored.  NEMO's kt=1 entry ssh is exactly
    # 0.0, so the live and reference weightings should coincide BITWISE here;
    # that is the control which shows the kt=2 separation below is eta, not a
    # difference of code paths.
    def _live1(du, tag, association):
        arr = live_depth_mean(
            du, seeded1.eta.data, seeded1.H_bathy.data, card.recipe.z_coord,
            cfg, card.recipe.grid,
            np.asarray(getattr(seeded1, f"{tag}_mask").data),
            tag=tag, association=association)
        return arr[:, 1:] if tag == "u" else arr[1:, :]

    kt1_arms = {}
    for tag, du, mask, trim in (("u", nemo_du, u_mask2d, lambda a: a[:, 1:, :]),
                                ("v", nemo_dv, v_mask2d, lambda a: a[1:, :, :])):
        ref = reference_depth_mean(trim(du), weights, tag) * mask
        kt1_arms[tag] = {
            "arm_abs_max": float(np.max(np.abs(ref[mask]))),
            "reference_vs_live_divide_bit_unequal":
                _bits(ref, _live1(du, tag, "model"), mask),
            "reference_vs_live_reciprocal_bit_unequal":
                _bits(ref, _live1(du, tag, "nemo_reciprocal"), mask),
            "n": int(mask.sum()),
        }

    # ---- ORACLE-SCORED rows.  The production slow-forcing frame is the one
    # row that carries a fidelity claim about this statement.
    rows = []
    bit_census = {}
    for tag, mask in (("u", u_mask2d), ("v", v_mask2d)):
        rows.append(score(f"{CASE}.kt1.substep1.slow_{tag}.model_captured",
                          oracle[f"slow_{tag}"][0], captured[f"slow_{tag}"],
                          mask, plant=plant))
        # score() uses np.array_equal, which is VALUE equality (-0.0 == 0.0).
        # The bar is BITS, so publish both (round-27 Rule-11 record).
        bit_census[f"slow_{tag}"] = {
            "bit_unequal": _bits(oracle[f"slow_{tag}"][0],
                                 captured[f"slow_{tag}"], mask),
            "n": int(mask.sum())}

    # The kt=1 3-D momentum RHS NEMO dumps after stp_2D.  GYRE enters kt=1 at
    # rest (u = v = 0 measured below), and dyn_keg / dyn_zad / dyn_vor are all
    # products with the entry velocity, so this frame is dyn_hpg alone on the
    # vector-form arm exactly as it is on LOCK/OVERFLOW's flux-form arm.
    for tag, cand, ref, mask, trim in (
            ("u", model_du, rhs["u"], masks["u"], lambda a: a[:, 1:, :]),
            ("v", model_dv, rhs["v"], masks["v"], lambda a: a[1:, :, :])):
        rows.append(score(f"{CASE}.kt1.stp2d.momentum_rhs.{tag}",
                          ref[..., :nlev], trim(cand), mask))

    # The whole kt=1 external-mode trace, in NEMO's own recurrence order, so
    # the first frame over bar is NAMED by measurement rather than guessed.
    trace_masks = {"eta": masks["ssh"], "u": u_mask2d, "v": v_mask2d}
    bt_rows = []
    bt_first_over_bar = None
    for jn in range(oracle["ncycle"]):
        for name in BT_TRACE_ORDER:
            stagger = ("u" if name.startswith("u_") or name.endswith("_u")
                       else "v" if name.startswith("v_") or name.endswith("_v")
                       else "eta")
            row = score(f"{CASE}.kt1.bt.jn{jn + 1:02d}.{name}",
                        oracle[name][jn], trace_frames[name][jn],
                        trace_masks[stagger])
            row["substep"], row["boundary"] = jn + 1, name
            bt_rows.append(row)
            if row["status"] != "AT-BAR" and bt_first_over_bar is None:
                bt_first_over_bar = {"substep": jn + 1, "boundary": name,
                                     "absolute_max": row["absolute_max"],
                                     "n_unequal": row["n_unequal"]}

    # ---- kt=2 sizing.  NOT an oracle comparison: there is no dumped GYRE
    # slow-forcing frame at kt=2.  This is the reference-minus-live difference
    # on NEMO's own kt=2 entry state, which is what sizes the O(eta/H) claim.
    tend2 = model.tendencies(seeded2, dt=card.dt_s, momentum_only=True)
    kt2 = {}
    for tag, du, mask, trim in (
            ("u", np.asarray(tend2.du_dt.data), u_mask2d, lambda a: a[:, 1:]),
            ("v", np.asarray(tend2.dv_dt.data), v_mask2d, lambda a: a[1:, :])):
        def live_arm(association, _du=du, _tag=tag, _trim=trim):
            return _trim(live_depth_mean(
                _du, seeded2.eta.data, seeded2.H_bathy.data,
                card.recipe.z_coord, cfg, card.recipe.grid,
                np.asarray(getattr(seeded2, f"{_tag}_mask").data),
                tag=_tag, association=association))

        face = {"live_divide": live_arm("model"),
                "live_reciprocal": live_arm("nemo_reciprocal"),
                "reference": reference_depth_mean(trim(du), weights, tag) * mask}
        scale = float(np.max(np.abs(face["live_divide"][mask])))
        require(scale > 0.0,
                f"kt=2 {tag} operand is zero; the arms would perturb a zero")
        delta = float(np.max(np.abs(
            face["reference"][mask] - face["live_divide"][mask])))
        kt2[tag] = {
            "arm_scale_abs_max": scale,
            "n": int(mask.sum()),
            "reference_minus_live_divide_abs_max": delta,
            "reference_minus_live_divide_relative": delta / scale,
            "reference_vs_live_divide_bit_unequal":
                _bits(face["reference"], face["live_divide"], mask),
            "reference_vs_live_reciprocal_bit_unequal":
                _bits(face["reference"], face["live_reciprocal"], mask),
            "live_divide_vs_live_reciprocal_bit_unequal":
                _bits(face["live_divide"], face["live_reciprocal"], mask),
        }
        if tag == "u":
            # Non-vacuity for the NEW arm, always on: it must actually READ
            # NEMO's reference thickness.  Two weaker plants were tried first
            # and each is a lesson kept here.  The deepest wet face carries a
            # momentum RHS of exactly 0.0, so a plant there perturbs a zero
            # and proves nothing; and ONE ULP on e3u_0 at the largest
            # contributing cell is below one ULP of the column sum, so it
            # leaves every scored cell bit-identical.  That second number is
            # this control's operand resolution.  The plant is therefore
            # +1.0 m on the largest contributing face, the same convention the
            # rest of this campaign's controls use: it demonstrates the arm
            # READS e3u_0, not that it resolves a one-ULP effect.
            probe_weights = dict(weights)
            e3 = np.array(weights["e3u_0"], copy=True)
            contribution = np.abs(e3 * trim(du) * weights["umask"])
            deep = np.unravel_index(int(np.argmax(np.where(
                np.broadcast_to(mask[..., None], e3.shape),
                contribution, -np.inf))), e3.shape)
            require(contribution[deep] > 0.0,
                    "planted e3u_0 target contributes zero")
            e3[deep] = e3[deep] + 1.0
            probe_weights["e3u_0"] = e3
            moved = _bits(reference_depth_mean(trim(du), probe_weights, "u")
                          * mask, face["reference"], mask)
            require(moved > 0,
                    "a one-ULP change to NEMO's e3u_0 did not move the "
                    "reference arm; the arm is not reading e3u_0")
            kt2["reference_arm_reads_e3u_0_cells_moved"] = moved
    h_col = np.asarray(seeded2.H_bathy.data)
    eta2 = np.asarray(entry2["ssh"])
    wet = masks["ssh"]
    kt2["eta_over_H_abs_max"] = float(np.max(np.abs(eta2[wet] / h_col[wet])))
    kt2["comparison"] = "ARM-TO-ARM DIFFERENCE, NOT ORACLE-SCORED"

    # Corroboration against the campaign's own number rather than a private
    # one: kt=1 is exact on every field, so one step from NEMO's seeded kt=1
    # entry lands where the trajectory lands, and these rows must reproduce
    # the round-23 kt2 register.
    stepped = plain.step(seeded1, card.dt_s, freshwater=freshwater1,
                         surface_forcing=surface1)
    kt2_entry_rows = []
    for field in ("T", "S", "u", "v", "ssh"):
        cand = np.asarray(
            getattr(stepped, field if field != "ssh" else "eta").data)
        if field == "u":
            cand = cand[:, 1:, :]
        elif field == "v":
            cand = cand[1:, :, :]
        ref = entry2[field] if field == "ssh" else entry2[field][..., :nlev]
        kt2_entry_rows.append(score(f"{CASE}.kt2.before.{field}", ref, cand,
                                    masks[field]))

    over = [row["name"] for row in rows if row["status"] != "AT-BAR"]
    at_rest = {"kt1_entry_abs_max_u": float(np.max(np.abs(entry1["u"]))),
               "kt1_entry_abs_max_v": float(np.max(np.abs(entry1["v"])))}
    report = {
        "format": "nemo-testcase-gyre-slow-forcing-weighting-v1",
        "case": CASE,
        "legoesm_git_sha": legoesm_git_sha,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "transcendentals": "libm",
        "jax_backend": jax.default_backend(),
        "oracle_root": str(ROOT),
        "artifacts": {p.name: sha256(p) for p in
                      (substeps_path, rhs_path, entry1_path, entry2_path,
                       mesh_path)},
        "nemo_source": [
            "stp2d.F90:177 SELECT CASE( n_dynadv ); :178 vector-form arm, "
            ":183 flux-form arm",
            "stp2d.F90:180 GYRE's executed statement (ln_dynadv_vec = .true.)",
            "stp2d.F90:185 LOCK/OVERFLOW's statement (ln_dynadv_up3 = .true.)",
            "domain.F90:142 hu_0 accumulation over jk = 1..jpkm1",
            "domain.F90:159 r1_hu_0 = ssumask / ( hu_0 + 1 - ssumask )",
            "stprk3.F90:186 stp_2D is called once per step, from the Nbb entry",
        ],
        "precondition_kt1_and_kt2_entry_ssh": rest,
        "precondition_kt1_entry_at_rest": at_rest,
        "precondition_nemo_kt1_momentum_rhs_abs_max": nemo_rhs_abs_max,
        "kt1_arms_informative": kt1_arms_informative,
        "kt1_oracle_scored_rows": rows,
        "kt1_oracle_scored_bit_census": bit_census,
        "kt1_rows_over_bar": over,
        "kt1_barotropic_trace_rows_scored": len(bt_rows),
        "kt1_barotropic_first_over_bar": bt_first_over_bar,
        "kt1_arm_to_arm_bit_separation": kt1_arms,
        "kt2_entry_rows_after_one_step": kt2_entry_rows,
        "arm_to_arm_note": (
            "the four depth-mean arms are NOT scored against NEMO's frame: "
            "legoESM's F_slow is the depth mean PLUS wind, drag and "
            "biharmonic increments, so the sub-statement is not that frame. "
            "What is measured is the difference between two WEIGHTINGS of one "
            "shared operand, at kt=2, because NEMO's kt=1 3-D momentum RHS "
            "is identically zero on this card."),
        "kt2_reference_vs_live_sizing": kt2,
        "planted_control": plant,
    }
    if plant:
        planted = [row for row in rows if row["name"].endswith(
            "slow_u.model_captured")]
        require(planted and not planted[0]["exact"],
                "the planted slow_u violation did not fire")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(plant=args.plant, allow_dirty=args.allow_dirty)
    except GateError as error:
        print(json.dumps({"status": "GATE-ERROR", "error": str(error)}, indent=2))
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.plant:
        return 1
    return 0 if not report["kt1_rows_over_bar"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
