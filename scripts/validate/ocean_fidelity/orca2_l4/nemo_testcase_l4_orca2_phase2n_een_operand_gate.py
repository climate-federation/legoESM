#!/usr/bin/env python3
"""First-divergence gate for ORCA2's frozen EEN primitive operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_gyre_zco_card,
    build_lock_exchange_zco_card,
    build_orca2_zps_card,
    build_overflow_zps_card,
)
from legoesm.ocean.vertical import nemo_qco_live_vorticity_e3f_cgrid  # noqa: E402

LOCAL_NX, LOCAL_NY, NZ = 94, 152, 31
HALO = 2
GLOBAL_NX, GLOBAL_NY, ACTIVE_NZ = 180, 148, 30
FIELDS = (
    "u_nw", "u_ne", "u_sw", "u_se",
    "v_nw", "v_ne", "v_sw", "v_se",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_stream(path: Path, magic: str, fields: int) -> list[np.ndarray]:
    with path.open("rb") as handle:
        got_magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=10i", handle.read(40))
        values = np.fromfile(handle, np.float64)
    n3 = LOCAL_NX * LOCAL_NY * NZ
    require(got_magic == magic, f"{path.name}: bad magic {got_magic!r}")
    require(header == (1, 1, 1, 3, LOCAL_NX, LOCAL_NY, NZ, fields, 64,
                       fields * n3), f"{path.name}: bad header {header}")
    require(values.size == fields * n3, f"{path.name}: bad payload size")
    arrays = []
    for index in range(fields):
        local = values[index * n3:(index + 1) * n3].reshape(
            (LOCAL_NX, LOCAL_NY, NZ), order="F")
        arrays.append(local[HALO:-HALO, HALO:-HALO, :].transpose(1, 0, 2))
    return arrays


def _triads(q: jax.Array) -> tuple[jax.Array, ...]:
    b = nemo_source_round

    def shift(value, di=0, dj=0):
        out = jnp.roll(value, di, axis=1) if di else value
        return jnp.roll(out, dj, axis=0) if dj else out

    def triad(a, c, d):
        return b(b(a + c) + d)

    return (
        triad(shift(q, 1, 0), q, shift(q, 0, 1)),
        triad(shift(q, 0, 1), q, shift(q, -1, 0)),
        triad(q, shift(q, 0, 1), shift(q, 1, 1)),
        triad(shift(q, -1, 1), shift(q, 0, 1), q),
        triad(q, shift(q, 1, 0), shift(q, 1, -1)),
        triad(shift(q, 0, -1), q, shift(q, 1, 0)),
        triad(shift(q, 1, 1), shift(q, 1, 0), q),
        triad(shift(q, 1, 0), q, shift(q, 0, 1)),
    )


def _production_primitives(eta, z_coord):
    """Mirror the currently selected shared builder through its zpvo triads."""
    b = nemo_source_round
    dtype = jnp.float64
    raw = z_coord.nemo_een_barotropic
    one = jnp.asarray(1.0, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    area_eta = b(b(jnp.asarray(raw.e1t) * jnp.asarray(raw.e2t)) * eta)
    east = jnp.roll(area_eta, -1, axis=1)
    north = jnp.roll(area_eta, -1, axis=0)
    northeast = jnp.roll(north, -1, axis=1)
    quad = b(b(area_eta + east) + b(north + northeast))
    hf0 = jnp.asarray(raw.hf_0, dtype=dtype)
    wet_f = (hf0 > 0.0).astype(dtype)
    r1_hf0 = b(wet_f / b(hf0 + one - wet_f))
    area_f = b(jnp.asarray(raw.e1f) * jnp.asarray(raw.e2f))
    r3f = b(b(quarter * quad) * r1_hf0 / area_f)
    e3f0 = jnp.asarray(raw.e3f_0, dtype=dtype)
    live = b(e3f0 * b(one + r3f[..., None] * jnp.asarray(raw.fmask)))
    q = b(jnp.asarray(raw.ff_f)[..., None] / live)
    return (e3f0, live, q, *_triads(q))


def _round21_live_divisor_primitives(eta, z_coord):
    """Reproduce the GYRE Round-21 one-variable live-divisor arm."""
    # The committed arm called this existing shared helper.  Its uncommitted
    # probe diff adds source-round/order/reciprocal experiments; Round 21 says
    # those extra arms made zero further movement, so they are not folded in.
    raw = z_coord.nemo_een_barotropic
    dtype = jnp.float64
    b0 = jax.lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    e3t0 = jnp.asarray(z_coord.nemo_e3t_0, dtype=dtype)
    tmask = jnp.asarray(z_coord.is_active, dtype=dtype)

    def east(value):
        return jnp.roll(value, -1, axis=1)

    def north_closed(value):
        return jnp.concatenate([value[1:], jnp.zeros_like(value[:1])], axis=0)

    masked = b0(e3t0 * tmask)
    masked_n = north_closed(masked)
    ref_sum = b0(b0(masked + east(masked))
                 + b0(masked_n + east(masked_n)))
    e3f0 = b0(ref_sum / jnp.asarray(4.0, dtype=dtype))
    e3f0 = jnp.where(e3f0 == 0.0, jnp.asarray(raw.e3f_0), e3f0)
    live_vertex = nemo_qco_live_vorticity_e3f_cgrid(
        eta, z_coord, dtype, nn_e3f_typ=0)
    live = live_vertex[1:, 1:]
    q = nemo_source_round(jnp.asarray(raw.ff_f)[..., None] / live)
    return (e3f0, live, q, *_triads(q))


def _orca_fold_primitives(eta, z_coord, grid):
    """Round-21 divisor plus NEMO's T-pivot F-field fold overwrite."""
    base = _round21_live_divisor_primitives(eta, z_coord)
    e3f0 = base[0]
    perm_f = jnp.arange(e3f0.shape[1] - 1, -1, -1, dtype=jnp.int32)
    e3f0 = e3f0.at[-1].set(e3f0[-2, perm_f])
    live_vertex = nemo_qco_live_vorticity_e3f_cgrid(
        eta, z_coord, jnp.float64, nn_e3f_typ=0, grid=grid)
    live = live_vertex[1:, 1:]
    q = nemo_source_round(
        jnp.asarray(z_coord.nemo_een_barotropic.ff_f)[..., None] / live)
    return (e3f0, live, q, *_triads(q))


def _phase2n_legacy_primitives(eta, z_coord, grid):
    """Reproduce the admitted Phase-2n F-fold plus wrong ``fmask`` arm."""
    b = nemo_source_round
    raw = z_coord.nemo_een_barotropic
    dtype = jnp.float64
    one = jnp.asarray(1.0, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    fixed = _orca_fold_primitives(eta, z_coord, grid)
    e3f0 = fixed[0]
    eta = jnp.asarray(eta, dtype=dtype)
    area_eta = b(b(jnp.asarray(raw.e1t) * jnp.asarray(raw.e2t)) * eta)
    east = jnp.roll(area_eta, -1, axis=1)
    north = jnp.concatenate([area_eta[1:], jnp.zeros_like(area_eta[:1])], axis=0)
    northeast = jnp.roll(north, -1, axis=1)
    quad = b(b(area_eta + east) + b(north + northeast))
    hf0 = jnp.asarray(raw.hf_0, dtype=dtype)
    wet_f = (hf0 > 0.0).astype(dtype)
    r1_hf0 = b(wet_f / b(hf0 + one - wet_f))
    area_f = b(jnp.asarray(raw.e1f) * jnp.asarray(raw.e2f))
    r3f = b(b(quarter * quad) * r1_hf0 / area_f)
    live = b(e3f0 * b(one + r3f[..., None] * jnp.asarray(raw.fmask)))
    q = b(jnp.asarray(raw.ff_f)[..., None] / live)
    return (e3f0, live, q, *_triads(q))


def _source_masks(raw) -> tuple[np.ndarray, np.ndarray]:
    umask = np.asarray(raw.umask, dtype=bool)[:, :90]
    vmask = np.asarray(raw.vmask, dtype=bool)[:, :90]
    levels = np.arange(ACTIVE_NZ)[None, None, :]
    # NEMO mbku/mbkv retain level one even for dry columns.
    u_owned = levels < np.maximum(umask.sum(axis=2), 1)[:, :, None]
    v_owned = levels < np.maximum(vmask.sum(axis=2), 1)[:, :, None]
    return u_owned, v_owned


def _classify(index: tuple[int, int, int], wet: np.ndarray) -> str:
    j, i, k = index
    if j == GLOBAL_NY - 1:
        return "north_fold_row"
    here = wet[j, i, min(k, ACTIVE_NZ - 1)]
    if not here:
        return "coast_or_land_loop_cell"
    column_bottom = max(int(wet[j, i].sum()), 1) - 1
    if k == column_bottom and column_bottom < ACTIVE_NZ - 1:
        return "partial_cell_bottom"
    neighbours = (
        wet[j, (i - 1) % 90, min(k, ACTIVE_NZ - 1)],
        wet[j, (i + 1) % 90, min(k, ACTIVE_NZ - 1)],
        wet[max(j - 1, 0), i, min(k, ACTIVE_NZ - 1)],
        wet[min(j + 1, GLOBAL_NY - 1), i, min(k, ACTIVE_NZ - 1)],
    )
    if not all(neighbours):
        return "coast_or_land_loop_cell"
    return "interior"


def score(candidate, oracle, mask, wet) -> dict[str, object]:
    actual = np.asarray(candidate, np.float64)[:, :90, :ACTIVE_NZ]
    expected = np.asarray(oracle, np.float64)[:, :, :ACTIVE_NZ]
    require(actual.shape == expected.shape == mask.shape, "operand shape mismatch")
    unequal_grid = actual.view(np.uint64) != expected.view(np.uint64)
    unequal = unequal_grid & mask
    points = np.argwhere(unequal)
    result = {
        "status": "AT_BAR" if not len(points) else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(mask.sum()),
        "max_abs": float(np.abs(actual[mask] - expected[mask]).max(initial=0.0)),
    }
    if len(points):
        index = tuple(int(value) for value in points[0])
        result["first_global_jik_zero_based"] = list(index)
        result["first_nemo_ijk_one_based"] = [index[1] + 1, index[0] + 1, index[2] + 1]
        result["first_location_class"] = _classify(index, wet)
        classes: dict[str, int] = {}
        for point in points:
            label = _classify(tuple(int(value) for value in point), wet)
            classes[label] = classes.get(label, 0) + 1
        result["location_counts"] = classes
        per_level = unequal.sum(axis=(0, 1))
        per_row = unequal.sum(axis=(1, 2))
        result["per_level_unequal"] = [int(value) for value in per_level]
        result["per_row_unequal"] = [int(value) for value in per_row]
        result["bottom_vs_interior"] = {
            "partial_cell_bottom": int(classes.get("partial_cell_bottom", 0)),
            "interior_wet": int(classes.get("interior", 0)),
            "coast_or_land_loop_cell": int(
                classes.get("coast_or_land_loop_cell", 0)
            ),
            "north_fold_row": int(classes.get("north_fold_row", 0)),
        }
    return result


def _coefficient_complete_masks(u_source, v_source):
    """Masks whose three-q stencil is present in the de-haloed record."""
    u = [u_source.copy() for _ in range(4)]
    v = [v_source.copy() for _ in range(4)]
    for mask in u:
        mask[0, :, :] = False  # every U triad reads the unrecorded south halo
    for mask in v[:2]:
        mask[-1, :, :] = False  # north-reading V triads
    for mask in v[2:]:
        mask[0, :, :] = False  # south-reading V triads
    return (*u, *v)


# ---------------------------------------------------------------------------
# Round 8: the vertex thickness on the tripolar fold row.
#
# NEMO gives that row no formula of its own.  ``dyn_vor_init`` evaluates the
# masked four-cell average over the OWNED domain (dynvor.f90:913-919), then
# completes the field with the ordinary F-point north-fold exchange, sign +1
# (dynvor.f90:935), and only then substitutes the reference thickness for any
# remaining zero (dynvor.f90:937).  Under a T pivot that exchange rewrites the
# LAST OWNED row from the row immediately below it at the mirrored longitude:
# the compiled row loop runs to ``ipj - ihls`` with source ``ipj - ihls - 1``
# and the compiled longitude loop pairs ``ii1 + ii2 = ipi + 1``
# (lbcnfd.f90:722-746).
# ---------------------------------------------------------------------------

# Local (haloed) extent of the recorded rank-zero array, and the owned block
# inside it.  The instrumented writer fills owned cells only -- its halo slots
# are left at zero -- so ``_assert_record_is_owned_only`` makes that coverage
# limit mechanical instead of assumed.
FOLD_ROW = GLOBAL_NY - 1


def _read_local_with_halo(path: Path, magic: str) -> np.ndarray:
    """The recorded rank-zero array WITHOUT stripping its halo slots."""
    with path.open("rb") as handle:
        got_magic = handle.read(16).decode("ascii").rstrip()
        handle.read(40)
        values = np.fromfile(handle, np.float64)
    require(got_magic == magic, f"{path.name}: bad magic {got_magic!r}")
    require(values.size == LOCAL_NX * LOCAL_NY * NZ,
            f"{path.name}: bad payload size")
    return values.reshape((LOCAL_NX, LOCAL_NY, NZ), order="F").transpose(1, 0, 2)


def _assert_record_is_owned_only(path: Path, magic: str) -> dict[str, object]:
    """Prove the record carries NO value outside rank zero's owned block.

    The writer copies ``DO_3D( 0, 0, 0, 0, ... )`` into a zero-initialised
    buffer, so every halo slot is an absence, not a NEMO number.  Asserting it
    here is what stops a later round from reading those zeros as the other
    rank's half of the fold.
    """
    local = _read_local_with_halo(path, magic)
    halo_slots = np.concatenate([
        local[:, :HALO, :].ravel(), local[:, -HALO:, :].ravel(),
        local[:HALO, HALO:-HALO, :].ravel(),
        local[-HALO:, HALO:-HALO, :].ravel(),
    ])
    require(np.all(halo_slots == 0.0),
            f"{path.name}: a halo slot is non-zero; the record's coverage is "
            "not the owned block the writer claims")
    # The two writers differ in vertical reach -- the frozen operand is copied
    # over ``1, jpk`` and the live one over ``1, jpkm1`` -- so the occupancy
    # assertion covers the active levels every comparison here uses.
    owned = local[HALO:-HALO, HALO:-HALO, :]
    require(np.all(owned[:, :, :ACTIVE_NZ] != 0.0),
            f"{path.name}: an owned cell is zero; the writer did not fill the "
            "owned block")
    return {
        "recorded_longitudes_owned": int(owned.shape[1]),
        "recorded_longitudes_total": GLOBAL_NX,
        "halo_slots_all_zero": True,
        "unrecorded_fold_row_longitudes": GLOBAL_NX - owned.shape[1],
    }


def _fold_arm(primitives, *, perm: str, source_row: str, z_coord, grid, eta):
    """The frozen vertex thickness with a DELIBERATELY WRONG fold rule.

    ``perm='f'`` / ``source_row='below'`` is NEMO's rule; the other
    combinations are the controls that must not pass.
    """
    e3f0 = primitives[0]
    n_lon = e3f0.shape[1]
    if perm == "f":                       # lbcnfd.f90:722-746, ii1+ii2 = ipi+1
        index = jnp.arange(n_lon - 1, -1, -1, dtype=jnp.int32)
    else:                                 # the T-origin mirror, ii1+ii2 = ipi+2
        index = (-jnp.arange(n_lon, dtype=jnp.int32)) % n_lon
    source = e3f0[-2] if source_row == "below" else e3f0[-1]
    return e3f0.at[-1].set(source[index])


MAGICS = (("e3f_0vor", "NEMO_L4_E3F0_1"), ("live_e3f_vor", "NEMO_L4_E3FV_1"))


def _stitch_per_rank(per_rank_root: Path) -> tuple[dict, dict]:
    """Both ranks' owned halves of the fold row, joined into the full domain.

    The round-8 acquisition removed the writer's rank guard and tagged each
    file with its rank, so the two owned blocks together cover all 180
    longitudes.  Rank zero owns i = 0..89 and rank one i = 90..179, the
    longitude-only split the run prints (``ocean.output:196-202``).
    """

    recorded, coverage = {}, {}
    for name, magic in MAGICS:
        halves = []
        for rank in (0, 1):
            path = per_rank_root / (
                ("oracle_een_e3f0vor" if name == "e3f_0vor"
                 else "oracle_een_e3fvor")
                + f"_kt00000001_r{rank:04d}.bin")
            coverage[f"{name}_rank{rank}"] = _assert_record_is_owned_only(
                path, magic)
            halves.append(_read_local_with_halo(path, magic)[
                HALO:-HALO, HALO:-HALO, :ACTIVE_NZ])
        joined = np.concatenate(halves, axis=1)
        require(joined.shape[1] == GLOBAL_NX,
                f"{name}: the two ranks join to {joined.shape[1]} longitudes, "
                f"not {GLOBAL_NX}")
        recorded[name] = joined
    return recorded, coverage


def _fold_row_rows(card, oracle_paths, per_rank_root=None) -> dict[str, object]:
    """Score the fold row alone, with its ablation and its two wrong rules.

    With ``per_rank_root`` the record covers ALL 180 fold-row longitudes, so
    the exchange is certified in BOTH directions; without it the record is
    rank zero's 90 owned destinations only.
    """
    z_coord = card.recipe.z_coord
    grid = card.recipe.grid
    coverage = {
        name: _assert_record_is_owned_only(oracle_paths[name], magic)
        for name, magic in MAGICS
    }
    recorded = {
        name: _read_local_with_halo(oracle_paths[name], magic)[
            HALO:-HALO, HALO:-HALO, :ACTIVE_NZ]
        for name, magic in MAGICS
    }
    n_cols = GLOBAL_NX // 2
    rank0_agreement = None
    if per_rank_root is not None:
        full, per_rank_coverage = _stitch_per_rank(Path(per_rank_root))
        # The new acquisition is a different build, so its rank-zero half must
        # reproduce the admitted record before its rank-one half is believed.
        rank0_agreement = {
            name: int(np.sum(
                full[name][:, :n_cols].view(np.uint64)
                != recorded[name].view(np.uint64)))
            for name in full
        }
        for name, unequal in rank0_agreement.items():
            require(unequal == 0,
                    f"{name}: the per-rank record's rank-zero half disagrees "
                    f"with the admitted record in {unequal} cells")
        coverage = {"rank_zero_only_record": coverage,
                    "per_rank_record": per_rank_coverage}
        recorded = full
        n_cols = GLOBAL_NX
    own_eta = jnp.asarray(card.recipe.initial_state.eta.data, jnp.float64)
    # eta = 0 makes r3f exactly zero, so the live builder returns its own
    # FROZEN e3f_0vor bit for bit (x * 1.0 is exact).  That is how the frozen
    # operand is read out of the one shared builder instead of re-derived.
    zero_eta = jnp.zeros_like(own_eta)
    built = {
        "e3f_0vor": np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
            zero_eta, z_coord, jnp.float64, nn_e3f_typ=0, grid=grid)[1:, 1:]),
        "live_e3f_vor": np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
            own_eta, z_coord, jnp.float64, nn_e3f_typ=0, grid=grid)[1:, 1:]),
    }

    def _unequal(candidate: np.ndarray, expected: np.ndarray) -> dict[str, object]:
        actual = np.asarray(candidate, np.float64)[:, :n_cols, :ACTIVE_NZ]
        rows = {}
        for label, sl in (("owned_block", slice(None)),
                          ("north_fold_row", slice(FOLD_ROW, FOLD_ROW + 1))):
            a, b = actual[sl], expected[sl]
            unequal = a.view(np.uint64) != b.view(np.uint64)
            rows[label] = {
                "status": "AT_BAR" if not unequal.any() else "DEBT",
                "unequal": int(unequal.sum()),
                "count": int(unequal.size),
                "max_abs": float(np.abs(a - b).max(initial=0.0)),
            }
        return rows

    scored = {name: _unequal(built[name], recorded[name]) for name in built}

    # Controls.  Each rebuilds the SAME arm with one rule changed and must
    # leave the fold row unequal; a control that passes means the gate cannot
    # tell the rules apart and is refused here rather than reported.
    zero_arm = jax.jit(lambda value: _round21_live_divisor_primitives(
        value, z_coord))(zero_eta)
    controls = {}
    for label, kwargs in (
        ("no_fold", None),
        ("t_origin_mirror", {"perm": "t", "source_row": "below"}),
        ("fold_row_as_its_own_source", {"perm": "f", "source_row": "self"}),
    ):
        candidate = (zero_arm[0] if kwargs is None else
                     _fold_arm(zero_arm, z_coord=z_coord, grid=grid,
                               eta=zero_eta, **kwargs))
        row = _unequal(np.asarray(candidate), recorded["e3f_0vor"])
        require(row["north_fold_row"]["unequal"] > 0,
                f"fold control {label!r} is vacuous: it reproduces NEMO's "
                "recorded fold row even though its rule is wrong")
        controls[label] = row["north_fold_row"]

    # The rule as NEMO states it, applied by the production vertex-thickness
    # helper that used to refuse this row.  The pairing is restated from the
    # compiled longitude loop, so an inverted shift in that helper fails here.
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx
    h_k = (jnp.asarray(z_coord.dz_ref, jnp.float64)
           * jnp.asarray(z_coord.is_active, jnp.float64))
    h_vtx = np.asarray(jax.jit(lambda value: een_e3f_h_vtx(
        value, None, None, grid, card.recipe.model_config.een_e3f_scheme,
        dz_ref=z_coord.dz_ref)[0])(h_k))
    n_lon = GLOBAL_NX
    pairing = np.asarray([(n_lon + 1 - c) % n_lon for c in range(n_lon)])
    helper_row = h_vtx[-1, :n_lon]
    helper_source = h_vtx[-2, :n_lon][pairing]
    require(np.array_equal(helper_row.view(np.uint64),
                           helper_source.view(np.uint64)),
            "the production vertex-thickness helper's fold row is not the "
            "row below it at the mirrored longitude")

    return {
        "claim_label": "INDEPENDENT",
        "record_coverage": coverage,
        "scored": scored,
        "controls": controls,
        "production_helper_fold_row": {
            "refused": False,
            "matches_compiled_pairing": True,
            "pairing": "destination + source = Ni0glo + 1 (one-based)",
        },
        "source": {
            "owned_average": (
                "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"
                "dynvor.f90:913-919"),
            "fold_exchange": (
                "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"
                "dynvor.f90:935"),
            "zero_substitution": (
                "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"
                "dynvor.f90:937"),
            "t_pivot_f_point_rule": (
                "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"
                "lbcnfd.f90:722-746"),
        },
        "recorded_fold_row_longitudes": n_cols,
        "per_rank_rank_zero_agreement": rank0_agreement,
        "unmeasured": {
            "other_half_of_the_fold_row": (
                "rank one owns the remaining 90 fold-row longitudes and the "
                "instrumented writer is rank-zero only, so those destinations "
                "are UNRECORDED; every recorded destination's SOURCE does lie "
                "in rank one's half, so the exchange across the rank boundary "
                "is exercised in one direction only"
            ) if per_rank_root is None else (
                "NOTHING: the per-rank record covers all 180 fold-row "
                "longitudes, so both directions of the exchange are certified"
            ),
        },
    }


def _rule12_rows():
    rows = {}
    gyre = build_gyre_zco_card()
    # Cross-card means old and new *production* paths, not the diagnostic
    # source-rounding arm above.  The only old-path operand distinction is
    # that it consumed fmask where the new path consumes fe3mask.  GYRE's
    # free-slip closed box has those arrays bit-identical, and its inactive
    # fold descriptor makes the new F-fold operation an identity.
    old_coord = gyre.recipe.z_coord._replace(
        nemo_een_barotropic=gyre.recipe.z_coord.nemo_een_barotropic._replace(
            fe3mask=gyre.recipe.z_coord.nemo_een_barotropic.fmask
        )
    )
    old_fn = jax.jit(lambda value: nemo_qco_live_vorticity_e3f_cgrid(
        value, old_coord, jnp.float64, nn_e3f_typ=0,
        grid=gyre.recipe.grid)[1:, 1:])
    new_fn = jax.jit(lambda value: nemo_qco_live_vorticity_e3f_cgrid(
        value, gyre.recipe.z_coord, jnp.float64, nn_e3f_typ=0,
        grid=gyre.recipe.grid)[1:, 1:])
    # The original 0/21,120 row sampled only the card's cold-start eta.  Close
    # the review request on actual oracle states: all ten GYRE kt=1..10 BEFORE
    # frames, before any RK stage at that kt.  This remains an operand-builder
    # row, not a ten-step legoESM trajectory score.
    from scripts.validate.ocean_fidelity.testcases.nemo_testcase_l2_gyre_phase3_gate import (  # noqa: E501
        read_entry,
    )
    gyre_root = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10")
    old_frames, new_frames, entry_hashes = [], [], []
    for kt in range(1, 11):
        path = gyre_root / f"oracle_step_entry_kt{kt:08d}.bin"
        require(path.is_file(), f"missing GYRE Rule-12 entry frame {path}")
        entry = read_entry(path)
        require(entry["kt"] == kt, f"wrong GYRE entry kt in {path}")
        eta = jnp.asarray(entry["ssh"], dtype=jnp.float64)
        old_frames.append(np.asarray(old_fn(eta)))
        new_frames.append(np.asarray(new_fn(eta)))
        entry_hashes.append({"kt": kt, "sha256": sha256(path)})
    old_np = np.stack(old_frames)
    new_np = np.stack(new_frames)
    rows[gyre.case] = {
        "selector": gyre.recipe.model_config.vorticity_scheme,
        "unequal": int(np.count_nonzero(old_np.view(np.uint64)
                                         != new_np.view(np.uint64))),
        "count": int(old_np.size),
        "status": "AT_BAR" if np.array_equal(old_np, new_np) else "DEBT",
        "scope": (
            "GYRE oracle kt=1..10 BEFORE entry eta; ten operand-builder "
            "evaluations, before all RK stages at each kt"),
        "scope_reason": (
            "the Rule-12 arm compares static old/new e3f builders on ten "
            "observed NEMO states; it does not execute a legoESM trajectory"),
        "entry_records": entry_hashes,
    }
    for card in (build_lock_exchange_zco_card(), build_overflow_zps_card()):
        rows[card.case] = {
            "selector": card.recipe.model_config.vorticity_scheme,
            "unequal": 0,
            "count": int(np.prod(card.recipe.initial_state.eta.data.shape)),
            "status": "AT_BAR_UNREACHABLE_EEN_SELECTOR",
        }
    return rows


def validate(deck_root: Path, oracle_root: Path, plant: bool,
             per_rank_root: Path | None = None) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "EEN operand gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    paths = {
        "e3f_0vor": oracle_root / "oracle_een_e3f0vor_kt00000001.bin",
        "live_e3f_vor": oracle_root / "oracle_een_e3fvor_kt00000001.bin",
        "q": oracle_root / "oracle_een_q_kt00000001.bin",
        "zpvo": oracle_root / "oracle_een_zpvo_kt00000001.bin",
    }
    oracle = (
        read_stream(paths["e3f_0vor"], "NEMO_L4_E3F0_1", 1)[0],
        read_stream(paths["live_e3f_vor"], "NEMO_L4_E3FV_1", 1)[0],
        read_stream(paths["q"], "NEMO_L4_QEEN_1", 1)[0],
        *read_stream(paths["zpvo"], "NEMO_L4_ZPVO_1", 8),
    )
    names = ("e3f_0vor", "live_e3f_vor", "q", *FIELDS)
    card = build_orca2_zps_card(deck_root)
    require(card.recipe.model_config.vorticity_scheme == "een_total",
            "ORCA2 card does not select EEN")
    eta = card.recipe.initial_state.eta.data
    production = jax.jit(lambda value: _production_primitives(
        value, card.recipe.z_coord))(eta)
    live_arm = jax.jit(lambda value: _round21_live_divisor_primitives(
        value, card.recipe.z_coord))(eta)
    orca_fold = jax.jit(lambda value: _orca_fold_primitives(
        value, card.recipe.z_coord, card.recipe.grid))(eta)
    phase2n_legacy = jax.jit(lambda value: _phase2n_legacy_primitives(
        value, card.recipe.z_coord, card.recipe.grid))(eta)
    raw = card.recipe.z_coord.nemo_een_barotropic
    u_source, v_source = _source_masks(raw)
    full = np.ones_like(u_source, dtype=bool)
    coefficient_masks = _coefficient_complete_masks(u_source, v_source)
    masks = (full, full, full, *coefficient_masks)
    wet = (np.asarray(raw.fmask, dtype=bool)[:, :90],) * 3 + (
        (np.asarray(raw.umask, dtype=bool)[:, :90],) * 4
        + (np.asarray(raw.vmask, dtype=bool)[:, :90],) * 4)

    rows = []
    for index, name in enumerate(names):
        expected = oracle[index].copy()
        if plant and index == 0:
            expected[0, 0, 0] = np.nextafter(expected[0, 0, 0], np.inf)
        rows.append({
            "operand": name,
            "production": score(production[index], expected, masks[index], wet[index]),
            "round21_live_divisor": score(live_arm[index], expected, masks[index], wet[index]),
            "phase2n_legacy": score(
                phase2n_legacy[index], expected, masks[index], wet[index]
            ),
            "orca_fold_fixed": score(orca_fold[index], expected, masks[index], wet[index]),
        })
    if plant:
        # An oracle/oracle arm proves the scorer rejects one changed bit.
        planted_expected = oracle[0][:, :, :ACTIVE_NZ].copy()
        planted_expected[0, 0, 0] = np.nextafter(
            planted_expected[0, 0, 0], np.inf)
        exact = score(oracle[0][:, :, :ACTIVE_NZ], planted_expected,
                      full, wet[0])
        require(exact["unequal"] == 1, "one-bit EEN operand plant did not fire")
        raise GateError("planted EEN operand bit rejected through production scorer")

    def first(arm: str):
        return next((row for row in rows if row[arm]["status"] == "DEBT"), None)

    return {
        "status": "PASS",
        "boundary": "O4-EXT-A/EEN-primitive-first-divergence",
        "production_first": first("production"),
        "round21_live_divisor_first": first("round21_live_divisor"),
        "phase2n_legacy_first": first("phase2n_legacy"),
        "orca_fold_fixed_first": first("orca_fold_fixed"),
        "rows": rows,
        "records": {name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in paths.items()},
        "execution": {
            "backend": jax.default_backend(), "jax_disable_jit": False,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
            "domain": "rank0 owned cells; global j=0:148, i=0:90, k=0:30",
        },
        "operand_coverage": {
            "coefficient_exclusions": [
                "U triads at global j=0 require the unrecorded south halo q",
                "V north-reading triads at global j=147 require the unrecorded north halo q",
                "V south-reading triads at global j=0 require the unrecorded south halo q",
            ],
        },
        "cross_card_rule12": _rule12_rows(),
        "north_fold_row": _fold_row_rows(card, paths, per_rank_root),
        "source": {
            "e3f_0vor": "dynvor.F90:918-950",
            "live_e3f_vor": "domqco.F90:233-246; domzgr_substitute.h90:130",
            "fe3mask": "dommsk.F90:146-198 (before fmask changes at :207-243)",
            "q_and_zpvo": "dynspg_ts.F90:1520-1569",
            "production_builder": "barotropic_latlon_cgrid.py:_nemo_literal_een_coefficients",
            "round21_arm": "vertical.py:nemo_qco_live_vorticity_e3f_cgrid",
            "ens_caveat": (
                "NEMO ENS also consumes e3f_vor at dynvor.F90:623,719; "
                "LOCK/OVERFLOW unreachable means only that legoESM's current "
                "helper is dispatched by een_e3f_scheme='nemo_avg4'. Any "
                "future ENS/QCO card must re-check fe3mask."),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument(
        "--per-rank-root", type=Path,
        help="round-8 per-rank EEN record; certifies BOTH directions of "
             "the tripolar fold instead of rank zero's half alone")
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, args.plant,
                          args.per_rank_root)
    except (GateError, OSError, ValueError, IndexError, struct.error) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
