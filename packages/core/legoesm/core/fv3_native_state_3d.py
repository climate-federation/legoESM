"""Six-face, km-general state container for the FV3 duo-grid dycore.

STAGE 1 of the staged port (arXiv 2606.11356, FESOM2 Fortran -> C ->
C++/Kokkos): a loop-faithful intermediate representation, verified
against the original, before any target-language work. Here the
intermediate is NumPy/fp64 and the target is JAX.

WHY THIS FILE EXISTS
--------------------
The repository already holds two certified slices of the duo dycore and
they do not overlap:

    ============  ==================  =========================
                  km = 1              km = 2/3
    ============  ==================  =========================
    one face      --                  pressure chain, bit-exact
    six faces     duo acoustic path,  **the target: missing**
                  bit-exact
    ============  ==================  =========================

``fv3_native_pgrad`` is km-general but scoped to ONE face with "no
six-face exchange or averaging cadence"; ``fv3_native_duo_stepper`` has
the full six-face duo cadence but only at km=1. The missing intersection
is six-face x km>1, and that is what this container is the foundation
for.

THE km <= 4 WINDOW IS NOT A SHORTCUT
------------------------------------
``fv_dynamics.F90:568`` opens ``if ( npz > 4 ) then`` and the
``Lagrangian_to_Eulerian`` call at ``:618`` sits inside it (verified by
block-depth counting; the block closes at ``:674``). So at ``km <= 4``
the ORACLE ITSELF never remaps. A km=3 model is therefore genuinely
three-dimensional and faithful with no vertical-remap port at all.

``require_no_remap_needed`` below makes that a hard guard rather than a
comment, because silently skipping the remap at km>4 would leave ``delp``
as a deformed Lagrangian thickness and ``pt`` stuck in theta_v -- its only
inverse lives inside ``Lagrangian_to_Eulerian``
(``fv_dynamics.F90:396-410``), and hydrostatic never recomputes ``pkz``
(``:281-293``), it reuses what the previous remap left.

...BUT THE WINDOW IS NOT REACHABLE IN THE ORACLE ITSELF, which matters
for how the reference is generated. ``tools/fv_eta.F90:101`` selects the
vertical coordinate on ``km`` and has no table below 5: ``:241`` handles
``km == 5 .or. km == 10`` analytically, and ``case(20)`` upward covers the
rest. Asking the oracle for ``npz = 3`` yields NaN ak/bk and an abort
("var_hi: computed model top ... bottom/top dz = NaN NaN", then
``free(): invalid size``). So every RUNNABLE oracle 3-D column has
``npz >= 5``, i.e. the remap always fires there.

The reference is therefore taken at the ``dyn_core`` BOUNDARY rather than
the outer-step boundary: ``Lagrangian_to_Eulerian`` runs after
``dyn_core`` returns (``fv_dynamics.F90:618``, inside the ``npz > 4``
block opened at ``:568``), so one complete acoustic loop is remap-free by
construction. Generate at ``npz = 5`` with the analytic coordinate of
``fv_eta.F90:241`` -- ``ptop = 500e2``, ``ks = 0``,
``bk(k) = (k-1)/km``, ``ak(k) = ptop*(1 - bk(k))`` -- which is exactly
reproducible on our side.

This module's guard still refuses ``km > 4`` because OUR stepper has no
remap; it is a statement about our lane, not a claim that the oracle can
be run there.

LAYOUT CONTRACT
---------------
Transcribed from ``fv3_native_pgrad``'s module docstring, which declares
these shapes load-bearing for its bit-exact oracle compare. With
``m_a = n + 2*ng`` and ``m_b = m_a + 1``, Fortran origin in brackets:

    delp, pt, w, q_con   (m_a, m_a, km)      [isd, jsd, 1]
    u,  vc               (m_a, m_b, km)      [isd, jsd, 1]
    v,  uc               (m_b, m_a, km)      [isd, jsd, 1]
    pk, gz               (m_a, m_a, km+1)    [isd, jsd, 1]
    pe                   (n+2,  km+1, n+2)   [is-1, 1, js-1]
    peln                 (n,    km+1, n)     [is,   1, js]
    pkz                  (n, n, km)          [is, js, 1]

``pe``/``peln`` keep the upstream ``(i, k, j)`` axis order. That order is
LOAD-BEARING and must not be restaged to ``(i, j, k)`` for convenience --
``fv3_native_pgrad`` says so explicitly, and any restaging needs its own
equivalence gate.

Note ``pk``/``gz``/``pe``/``peln`` carry ``km+1``, not ``km``. Assuming a
single vertical extent for every field is the obvious way to get this
wrong, so :func:`validate_state_3d` checks each one separately.
"""
from __future__ import annotations

import numpy as np

# fv_dynamics.F90:568 -- above this the oracle calls Lagrangian_to_Eulerian
# every outer split, and we have not ported fv_mapz.
MAX_KM_WITHOUT_REMAP = 4

# Cell-centred / A-grid scalars carried by the acoustic step.
SCALAR_FIELDS = ("delp", "pt", "w")
# D-grid prognostic winds. fv_arrays.F90 declares u,v D-grid PROGNOSTIC and
# says the C grid "is diagnostic in that it is predicted every time step
# from the D grid variables" -- so uc/vc must NOT persist here.
WIND_FIELDS = ("u", "v")
STATE_FIELDS = SCALAR_FIELDS + WIND_FIELDS


def require_no_remap_needed(km: int, *, remap_follows: bool = False) -> None:
    """Refuse a km the oracle would have remapped, unless one follows.

    Silently skipping ``Lagrangian_to_Eulerian`` does not degrade
    gracefully: ``delp`` stays a deformed Lagrangian thickness while
    ``pt`` stays in theta_v, and interfaces drift off the reference
    coordinate until some ``delp`` goes non-positive.

    ``remap_follows=True`` is the CALLER'S PROMISE that
    ``fv3_native_mapz.lagrangian_to_eulerian`` runs on this state before
    the next outer step -- i.e. the caller is
    ``fv3_native_dynamics.fv_dynamics_step``, which owns the
    ``fv_dynamics.F90:568`` ``npz > 4`` block.  It is deliberately a
    keyword the acoustic stages FORWARD rather than infer: a stage cannot
    see its own caller, and inferring "someone will remap" from ``km``
    alone is exactly the silent skip this guard exists to stop.
    """
    if not isinstance(km, (int, np.integer)) or km < 1:
        raise ValueError(f"km must be a positive integer, got {km!r}")
    if not isinstance(remap_follows, bool):
        # Any truthy value would bypass the guard, so a typo'd kwarg or a
        # string sentinel would silently license a deformed-delp state.
        raise TypeError(
            f"remap_follows must be a bool, got {type(remap_follows).__name__}"
            f" ({remap_follows!r}). It is a promise that "
            f"fv3_native_mapz.lagrangian_to_eulerian runs on this state, "
            f"not a flag to be set loosely.")
    if remap_follows:
        return
    if km > MAX_KM_WITHOUT_REMAP:
        raise ValueError(
            f"km={km} > {MAX_KM_WITHOUT_REMAP}: fv_dynamics.F90:568 gates "
            f"Lagrangian_to_Eulerian on `npz > 4` and the call at :618 is "
            f"INSIDE that block, so the oracle remaps at this km and we have "
            f"not ported fv_mapz. Running without it would leave delp a "
            f"deformed Lagrangian thickness and pt in theta_v (its only "
            f"inverse is inside the remap, fv_dynamics.F90:396-410). Use "
            f"km <= {MAX_KM_WITHOUT_REMAP}, or port the remap first.")


def field_shape(name: str, n: int, ng: int, km: int) -> tuple[int, ...]:
    """Declared shape of one field. Single source of truth for the layout."""
    m_a = n + 2 * ng
    m_b = m_a + 1
    if name in ("delp", "pt", "w", "q_con", "delpc", "ptc", "ua", "va",
                "ut", "vt"):
        return (m_a, m_a, km)
    if name in ("u", "vc"):
        return (m_a, m_b, km)
    if name in ("v", "uc"):
        return (m_b, m_a, km)
    if name in ("pk", "gz"):
        return (m_a, m_a, km + 1)
    if name == "divgd":
        return (m_b, m_b, km)
    if name == "pe":
        return (n + 2, km + 1, n + 2)     # (i, k, j) -- load-bearing
    if name == "peln":
        return (n, km + 1, n)             # (i, k, j) -- load-bearing
    if name == "pkz":
        return (n, n, km)
    if name == "ps":
        # init_hydro.F90:62 declares ps(ifirst-ng:ilast+ng, jfirst-ng:
        # jlast+ng), i.e. the full padded plane. It was allocated at the
        # call site instead of here, which is exactly the "undeclared
        # shape at a call site" this function exists to stop.
        return (n + 2 * ng, n + 2 * ng)
    raise KeyError(
        f"unknown field {name!r}; add it to field_shape rather than "
        f"allocating an undeclared shape at a call site")


def build_state_3d(n: int, ng: int, km: int, *, fill: float = 0.0,
                   remap_follows: bool = False) -> list:
    """Six per-face dicts of km-general fp64 arrays.

    fp64 throughout: Lane A is the fp64 reference the JAX stage is
    verified against, so a float32 array here would silently cap the
    achievable agreement at ~1e-7.
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    if n < 1 or ng < 1:
        raise ValueError(f"n and ng must be >= 1, got n={n}, ng={ng}")
    return [
        {f: np.full(field_shape(f, n, ng, km), fill, dtype=np.float64)
         for f in STATE_FIELDS}
        for _ in range(6)
    ]


def validate_state_3d(state: list, n: int, ng: int, km: int, *,
                      require_finite: bool = True,
                      remap_follows: bool = False) -> None:
    """Fail loudly on anything the stage kernels would accept silently.

    The motivating hazard: ``fort.__getitem__`` splits its subscript as
    ``i, j, *k``, so a 2-subscript read of a 3-D array returns the whole
    column and BROADCASTS. A rank or extent slip therefore produces
    numbers rather than an exception unless something checks first.
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    if not isinstance(state, (list, tuple)) or len(state) != 6:
        raise ValueError(
            f"state must be 6 per-face dicts, got "
            f"{type(state).__name__} of length "
            f"{len(state) if hasattr(state, '__len__') else '?'}")
    for t, face in enumerate(state, start=1):
        missing = [f for f in STATE_FIELDS if f not in face]
        if missing:
            raise KeyError(f"face {t}: missing field(s) {missing}")
        for name, arr in face.items():
            a = np.asarray(arr)
            try:
                want = field_shape(name, n, ng, km)
            except KeyError:
                raise KeyError(
                    f"face {t}: undeclared field {name!r} in the state; "
                    f"every carried field needs a shape in field_shape") \
                    from None
            if a.shape != want:
                raise ValueError(
                    f"face {t} field {name!r}: shape {a.shape}, expected "
                    f"{want} for n={n} ng={ng} km={km}")
            if a.dtype != np.float64:
                raise TypeError(
                    f"face {t} field {name!r}: dtype {a.dtype}, expected "
                    f"float64 -- this is the fp64 reference the JAX stage "
                    f"is verified against")
            if require_finite and not np.all(np.isfinite(a)):
                bad = int(np.count_nonzero(~np.isfinite(a)))
                raise ValueError(
                    f"face {t} field {name!r}: {bad} non-finite value(s)")


def level_slice(face: dict, k: int, km: int) -> dict:
    """One level of a face, as 2-D arrays the certified kernels accept.

    The oracle calls ``c_sw``/``d_sw*`` inside ``do k=1,npz`` on arrays
    declared ``dimension(bd%isd:bd%ied, bd%jsd:bd%jed)``
    (``sw_core.F90:84-87``), so slicing per level and calling the existing
    2-D certified stage is the faithful lift -- not new stage math.

    Returns VIEWS, so a kernel that writes in place updates the parent
    state, matching the Fortran's intent(inout) dummies.
    """
    if not (0 <= k < km):
        raise IndexError(f"level {k} out of range for km={km}")
    return {name: arr[:, :, k] for name, arr in face.items()}


def state_signature(state: list) -> dict:
    """Cheap deterministic fingerprint, for trace checkpoints.

    Sums are fp64 and order-fixed, so two runs of the same code agree
    exactly; this is a change-detector between stages, not a physics
    diagnostic.
    """
    out = {}
    for t, face in enumerate(state, start=1):
        for name in sorted(face):
            a = np.asarray(face[name], dtype=np.float64)
            out[f"t{t}.{name}"] = (
                float(a.sum()), float(np.abs(a).max()), a.shape)
    return out
