"""D-grid transport phase with duo BARRIER 1, km-general -- JAX lane.

Functional, jit-compilable, differentiable twin of
``legoesm.core.fv3_native_dsw_phase_3d``.  Like its NumPy
specification, this module introduces **no numerics of its own**: every
arithmetic operation happens inside a kernel that is already ported and
gated elsewhere (``fv3_duo_sw_core.d_sw1_duo`` / ``d_sw2_duo``,
``fv3_duo_halos.average_allflux_shared_edges``,
``fv3_duo_stepper.exchange_post_pgrad_sixface``).  What it contributes
is the CADENCE -- which kernel runs at which level, on which face, and
in which order -- and, above all, WHERE THE BARRIER SITS.

Authority chain (``docs/atmosphere/fv3_duo_jax_lane_strategy.md`` §2):
``fv3_native_dsw_phase_3d`` is the SPECIFICATION (hop B).  The pinned
Fortran
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean``
is quoted only where a line number says WHY a step exists.  **Every
citation in this file was re-derived with ``sed``/``grep`` against that
tree** (R7), including the ones the spec already carries; the tree's
``model/dyn_core.F90`` has md5 ``e5a5fab97b0fa3066daf4d00b1e5591d``,
which is the pinned value (three copies of this source exist and they
are not the same file -- STATE lesson 3).

ORACLE CADENCE, verified line by line
-------------------------------------
::

    dyn_core.F90:744   do k=1,npz                        <- d_sw1 nest
                :831     call d_sw1( delp(isd,jsd,k), pt(isd,jsd,k), ...
                :853   if (duogrid) then
                :855     do iq=1,4+nq ! 1delp 2w 3qcon 4temp 5>q
                :856       if (iq==1 .or. iq==4 .or. iq>4 ) then
                :857         do k=1,npz
                :872         call mpp_get_boundary(fxx_delp, fyy_delp, ...
                :874              ... gridtype=CGRID_NE )     <- BARRIER 1
                :877         do k=1,npz            (the 0.5*(a+b) blend)
                :914   do k=1,npz                        <- d_sw2 nest
                :950     call d_sw2( delp(isd,jsd,k), ptc(isd,jsd,k), ...

The enclosing subroutine of every line above is ``dyn_core``
(``dyn_core.F90:92``) -- i.e. the acoustic driver itself, entered on
every substep, with the barrier block gated only by ``if (duogrid)``
at ``:853``.  It is not a dead branch on this deck.

**THE THING THIS MODULE IS FOR.**  ``d_sw1`` computes FLUXES ONLY; the
``delp``/``pt`` updates happen in ``d_sw2`` AFTER the averaging.  That
ordering is the whole reason the oracle splits ``d_sw`` into six
routines instead of one: averaging the seam fluxes BEFORE they are
applied is what makes the panel-edge flux single-valued, hence
conservative and imprint-free across the seam.  A monolithic ``d_sw``
structurally cannot express it, because there would be no stage
boundary for the barrier to sit in.

BARRIER 1'S SLOT SELECTION -- the trap
--------------------------------------
``dyn_core.F90:855`` opens ``do iq=1,4+nq`` over the flux slots
(``! 1delp 2w 3qcon 4temp 5>q``) and ``:856`` reads, verbatim::

    if (iq==1 .or. iq==4 .or. iq>4 ) then

so slots **2 (``w``) and 3 (``q_con``) are deliberately NOT averaged**
and must come back BYTE-IDENTICAL across the barrier.  The exclusion is
not a runtime guard here: ``build_jax_duo_halo_tables`` bakes it into
``tab.allflux_slots`` (``fv3_duo_halos.py:1398``) so an excluded slot
never enters the selected list at all.  This module does not re-derive
it -- it calls ``average_allflux_shared_edges``, which owns it -- and
``test_fv3_dsw_phase_3d`` asserts the property survives the 3-D lift,
with a mutation control showing the gate fails when the exclusion is
removed.  (The same mutation was injected in this campaign's census
against ``test_fv3_duo_halos.test_barrier1_skips_slot2_w_and_slot3_
qcon_exactly`` and was CAUGHT; this file's gate is modelled on it.)

THE TWO BARRIERS HAVE DIFFERENT EXTENTS, which is why barrier 2 is a
different routine and is NOT in this unit::

    barrier 1 (CGRID_NE, :872)  fyy at j in {js, je+1} over i = is..ie
                                fxx at i in {is, ie+1} over j = js..je
    barrier 2 (BGRID_NE, :984)  i = is..ie+1, j = js..je+1  (B corners)

CONVENTIONS FOLLOWED (set by ``fv3_cgrid_phase_3d``)
----------------------------------------------------
C1  Face-stacked dicts, never lists of six per-face dicts.  Face ``t``
    is Fortran tile ``t+1`` and Fortran index ``1-ng`` sits at numpy
    index 0 on both horizontal axes.  Lists appear inside a function
    body only, as the per-face operands the single-face kernels take by
    construction.  The level axis is inserted at position 2, so a
    ``d_sw1`` output is ``(6, i, j, km)`` and an ``allflux`` stack is
    ``(6, i, j, km, 4+nq)`` -- which is the oracle's own
    ``allflux_x(i,j,k,iq)`` layout (``dyn_core.F90:860``), and the
    spec's ``np.stack(..., axis=2)``.
C2  The face loop and the level loop are Python loops, not ``vmap``.
    Both are proven independent below (R1a), so vectorising would be
    LEGAL -- it is declined because it is an OPTIMISATION, not a
    translation, and batching adds an axis XLA may contract
    differently.  Cost, stated: at ``km = 3`` this traces 18 ``d_sw1``
    and 18 ``d_sw2`` bodies plus 3 barrier applications.
C3  ``ctx``, ``km``, ``cfg`` and every deck constant / Python-branch
    selector are STATIC; the field data, the capacitors and ``dt`` are
    DYNAMIC, so a new time step does not recompile the phase.  Verified
    rather than assumed: ``d_sw1_duo`` and ``d_sw2_duo`` use ``dt``
    arithmetically only (their Python branches read ``hord_*``,
    ``nord_*``, ``damp_*``, ``hydrostatic``, ``inline_q``, never the
    time step).
C4  In-place mutation in the spec becomes a RETURNED key (R4).  The
    spec mutates ``csw_outs[t]["uc"]``/``["vc"]``/``["divg_d"]`` in
    ``_exchange_post_pgrad`` and mutates the ``flux_cap`` capacitors in
    place; both come back in the returned dict here and the CALLER
    threads them.
C5  A data-dependent Python guard cannot exist under ``jit``.  This
    module contains none -- every guard it adds reads a shape, a dtype
    or a static Python value.
C6  Stage observation is by RETURN, not by callback.  The spec takes a
    ``stage_hook`` for S07 (post exchanges), S08 (post ``d_sw1``, i.e.
    PRE-barrier fluxes), S09 (post barrier) and S10 (post ``d_sw2``).
    Under ``jit`` that hook would be handed tracers, exactly what
    ``build_jax_duo_stepper_context`` refuses for ``ctx['step_dump']``,
    so it is NOT ported.  S07 is the returned ``uc``/``vc``/``divg_d``,
    S09 the returned ``allflux_x``/``allflux_y`` and S10 the returned
    ``delp``/``pt``.  **S08 is the one that is not recoverable from the
    others** -- the barrier overwrites the fluxes -- so the pre-barrier
    stacks are returned explicitly as ``allflux_x_prebarrier`` /
    ``allflux_y_prebarrier``.  Without them the slot-exclusion gate
    could not be written as a single-call comparison, which is the one
    property in this module that most needs a cheap, exact test.
C7  Name and shape tables are IMPORTED from the spec, never retyped:
    ``DUO_DECK_CFG``, ``STATE_FIELDS``, ``field_shape``,
    ``require_no_remap_needed``, and ``CSW_OUT_LIKE`` from the sibling
    3-D module.  :data:`DSW_DUO_DECK` is BUILT from the spec's dict via
    ``SWConfig.from_mapping``, so there is no restatement to drift and
    an unknown deck knob RAISES at import instead of being dropped.

DEVIATIONS FROM THE SPEC, each with its reason
----------------------------------------------
D1. **The helper is PUBLIC.**  The spec's is ``_exchange_post_pgrad``.
    Here it is :func:`exchange_post_pgrad_3d`, because it needs a
    ``make_*_jit`` factory and a direct test, and a private
    cross-module import is banned by the empty-allowlist ratchet
    ``tests/test_no_private_cross_imports.py``.  The km=1 lane already
    exports the same routine publicly
    (``fv3_duo_stepper.exchange_post_pgrad_sixface``), which this one
    delegates to per level -- so the two lanes cannot drift.
D2. **The spec's per-level ``np.copy`` is dropped.**  It exists because
    ``csw_outs[t][name][:, :, k]`` is a strided VIEW and the NumPy
    exchange chain is certified only against ordinary 2-D arrays.  This
    lane is functional: a slice is a fresh value, there is no buffer to
    alias, and the copy has nothing left to protect.
D3. **``cfg`` is an ``SWConfig``, not a ``dict``.**  A dict is not
    hashable and cannot be a jit-static argument; every field is read
    by a Python ``if`` inside ``d_sw1``/``d_sw2``.  ``kgb``, ``nord_w``
    and ``damp_w`` are NOT ``SWConfig`` fields, so they stay explicit
    keywords whose ``None`` default reproduces the spec's own
    ``c.get("nord_w", c["nord_v"])`` / ``c.get("damp_w", c["damp_v"])``
    / ``c.get("kgb", 0.0)`` fallbacks exactly.  ``SWConfig`` is NOT
    extended with them: that would change another unit's public config
    for this unit's convenience.
D4. **The capacitor windows are sliced, and this is load-bearing.**
    ``alloc_flux_capacitors`` allocates ``mfx (npx, m_a, km)`` while
    ``d_sw1_duo``'s ``xflux`` dummy is ``(npx, n)``: the NumPy kernel
    wraps its argument in a 1-based ``fort`` view and never checks the
    shape, so the oversized slab is harmless there, whereas the JAX
    kernel's ``_check_shape`` refuses it.  The used window is exactly
    ``mfx[:, :n]`` (origin ``(is, js)``, written ``i = is..ie+1``,
    ``j = js..je``; the columns beyond ``je`` "exist only because the
    caller-side arrays are allocated rectangular -- they stay zero",
    per the allocator's own docstring).  ``cx``/``cy`` need no slice:
    their declared shapes already match the kernel's dummies.  A test
    asserts the unused region is still zero afterwards.

Non-hydrostatic
---------------
``hydrostatic=False`` threads the NH ``w`` path: ``d_sw1``'s
``fv_tp_2d(w)`` into allflux slot 2 -- which the barrier deliberately
does NOT average (``dyn_core.F90:856``) -- then ``d_sw2``'s del-6
``dw`` increment and the mass-weighted ``w`` update on the OLD
``delp``.  ``w`` and ``dw`` ride the returned dict for ``d_sw5``'s
finalisation, exactly as the spec carries them.  On the hydrostatic
arm neither key is present, mirroring the spec (an absent key is an
ABSENCE; materialising it as zeros would be a plausible-looking value
where the twin has nothing).

Differentiability
-----------------
No ``donate_argnums`` (it conflicts with reverse-mode AD, which is the
point of this lane).  **No ``jnp.where`` is added by this module**, so
R1b's dead-arm hazard has no site here: every branch is a Python ``if``
on a static value, and every array operation is a slice, a
``jnp.stack`` or an ``.at[...].set()`` index copy.  The selects this
phase INHERITS are ``d_sw1_duo``'s four panel-edge ``uc*dt > 0`` /
``vc*dt > 0` arms and its upwind ``crx``/``cry`` pair; that kernel's
docstring records that both arms of the dividing selects are total on
an admitted gridstruct (``sin_sg`` strictly positive) and they are
gated in ``test_fv3_duo_sw_core.py``.  ``d_sw2_duo`` contains no
data-dependent branch at all.  Grid metrics, halo tables and ``hs`` are
STATIC (carried by ``DuoStepperContext``, hashed by identity), so
differentiating this phase means differentiating with respect to the
state fields, the C-grid winds and the capacitors only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.fv3_cgrid_phase_3d import CSW_OUT_LIKE
from legoesm.core.fv3_duo_stepper import (
    SWConfig,
    exchange_post_pgrad_sixface,
)
from legoesm.core.fv3_duo_sw_core import d_sw1_duo, d_sw2_duo
from legoesm.core.fv3_native_dsw_phase_3d import DUO_DECK_CFG
from legoesm.core.fv3_native_state_3d import (
    STATE_FIELDS,
    field_shape,
    require_no_remap_needed,
)
from legoesm.grids.fv3_duo_halos import average_allflux_shared_edges

__all__ = [
    "DUO_DECK_CFG",
    "DSW_DUO_DECK",
    "DSW1_OUT_2D",
    "CAPACITOR_FIELDS",
    "exchange_post_pgrad_3d",
    "dsw_transport_phase_3d",
    "make_exchange_post_pgrad_3d_jit",
    "make_dsw_transport_phase_3d_jit",
]

# The shipped duo deck, as an SWConfig.  BUILT from the spec's dict
# rather than restated (C7): `SWConfig.from_mapping` raises on a knob
# that is not an SWConfig field, so a deck key added on the NumPy side
# fails at IMPORT here instead of being silently dropped.
#
# The four fields the deck does not name -- `hord_mt`, `dddmp`,
# `d2_bg`, `d4_bg` -- keep their SWConfig defaults and are NOT read by
# this phase: `hord_mt` belongs to d_sw3 and the damping trio to d_sw5,
# both of which are the TAIL unit.  Stated so a reader does not take
# their presence as a claim that this phase honours them.
DSW_DUO_DECK = SWConfig.from_mapping(DUO_DECK_CFG)

# `d_sw1_duo`'s output names, in the order its return dict lists them.
# Every one of these is stacked over levels and faces and returned.
# `delp`/`pt`/`w` are EXCLUDED here on purpose: d_sw1 returns the
# corner-ghost-rotated inputs, and what this phase returns under those
# names is d_sw2's UPDATE, so carrying both would put two different
# quantities under one name.
DSW1_OUT_2D = ("crx_adv", "cry_adv", "xfx_adv", "yfx_adv",
               "ra_x", "ra_y", "ut", "vt",
               "allflux_x", "allflux_y")

# The flux capacitors, as `fv3_native_tracer2d.alloc_flux_capacitors`
# names and shapes them, paired with the `d_sw1_duo` output that feeds
# each one (sw_core.F90:903-920 accumulates them every substep, and the
# oracle's dummies are intent(inout)).
#
#   name : (d_sw1 key, declared 2-D shape, kernel-window slice)
#
# The window slice is D4: `mfx`/`mfy` are allocated one halo-padded
# axis wider than the kernel's dummy, `cx`/`cy` are not.
CAPACITOR_FIELDS = ("mfx", "mfy", "cx", "cy")
_CAP_FROM_DSW1 = {"mfx": "xflux", "mfy": "yflux", "cx": "cx", "cy": "cy"}

# The exchanges this phase applies per level, by the csw_outs key each
# one writes.  Named as data so the entry gate can report a missing key
# instead of raising a bare KeyError inside the halo tables.
_EXCHANGE_FIELDS = ("divg_d", "uc", "vc")


# ---------------------------------------------------------------------
# entry gates
# ---------------------------------------------------------------------

def _require_f64_jax(fname: str, arrays: dict) -> None:
    """Static-dtype gate mirroring the NumPy lane's ``_require_f64``.

    Reads only ``.dtype`` (static under jit): a float32 operand would
    otherwise be silently upcast -- or, with ``jax_enable_x64``
    disabled, the whole phase would silently run in float32 -- and the
    oracle build is ``-fdefault-real-8``.

    DUPLICATION, deliberate and already the campaign's convention:
    character-identical to ``fv3_cgrid_phase_3d``'s /
    ``fv3_duo_stepper``'s / ``fv3_duo_halos``'s /
    ``fv3_duo_sw_core``'s, which are private and therefore not
    importable across modules (the empty-allowlist ratchet
    ``tests/test_no_private_cross_imports.py``).  FOLLOW-UP, already
    open on the other four: promote ONE definition to a public name and
    delete the copies -- this is now the fifth.
    """
    for name, a in arrays.items():
        if a is None:
            continue
        if jnp.asarray(a).dtype != jnp.float64:
            raise TypeError(
                f"{fname}: {name} must be float64 (got "
                f"{jnp.asarray(a).dtype}); enable jax_enable_x64 and pass "
                f"f64 operands (oracle build is -fdefault-real-8)")


def _require_bool(fname: str, name: str, value) -> None:
    """A truthy non-bool would silently select a branch.

    ``remap_follows`` in particular is a PROMISE that the vertical remap
    runs later; a string sentinel or a stray ``1`` must not license a
    deformed-``delp`` state.  ``hydrostatic`` chooses whether allflux
    slot 2 is written at all, i.e. what the barrier is handed.
    """
    if not isinstance(value, bool):
        raise TypeError(
            f"{fname}: {name} must be a bool, got "
            f"{type(value).__name__} ({value!r})")


def _require_km(fname: str, km) -> int:
    """``km`` is a Python trip count and a shape -- never traced."""
    if isinstance(km, bool) or not isinstance(km, (int, np.integer)):
        raise TypeError(
            f"{fname}: km must be a Python int (it is a loop trip count "
            f"and an array extent, so it cannot be traced), got "
            f"{type(km).__name__} ({km!r})")
    if km < 1:
        raise ValueError(f"{fname}: km must be >= 1, got {km}")
    return int(km)


def _require_nord(fname: str, name: str, nord) -> int:
    """A damping ORDER is integral by construction.

    ``int()`` on 2.7 would round to 2 without a word, and
    ``exchange_post_pgrad_sixface`` branches on ``nord > 0`` -- so a
    float here silently selects whether the ``divgd`` halo is exchanged
    at all.  Same guard, same reason, as the km=1 lane's.
    """
    if isinstance(nord, bool) or nord != int(nord):
        raise ValueError(
            f"{fname}: {name} must be an integral damping order, got "
            f"{nord!r}")
    nord = int(nord)
    if nord < 0:
        raise ValueError(f"{fname}: {name} must be >= 0, got {nord}")
    return nord


def _resolve_cfg(fname: str, cfg) -> SWConfig:
    """``None``/``SWConfig`` -> ``SWConfig``; anything else RAISES.

    Dispatch hardening on a STATIC value.  A plain ``dict`` is the
    NumPy lane's container and is REFUSED rather than converted here:
    it is unhashable, so it could not be a jit-static argument, and
    silently converting it at the eager entry would make the eager and
    compiled lanes accept different types.  ``SWConfig.from_mapping``
    is the explicit converter, and it raises on an unknown knob.
    """
    if cfg is None:
        return DSW_DUO_DECK
    if isinstance(cfg, SWConfig):
        return cfg
    raise TypeError(
        f"{fname}: cfg must be an SWConfig (hashable, so it can be a "
        f"jit-static argument) or None for the shipped duo deck "
        f"{DSW_DUO_DECK}; a plain dict cannot cross the jit boundary. "
        f"Convert with SWConfig.from_mapping(<dict>), which raises on "
        f"an unknown knob instead of dropping it. Got "
        f"{type(cfg).__name__}.")


def _require_barrier_nq(fname: str, ctx, nslot: int) -> None:
    """The barrier's slot axis and the halo table must agree.

    ``d_sw1_duo`` allocates its allflux stacks with ``nq = 1``, so
    ``4 + nq = 5`` slots; ``build_jax_duo_stepper_context`` builds the
    tables with ``nq = 1`` too.  They are not free parameters -- a
    context built at a different ``nq`` would make
    ``average_allflux_shared_edges`` refuse the stack.  Checked HERE,
    at the join the two meet, so the message names both sides.
    """
    want = 4 + int(ctx.tab.nq)
    if nslot != want:
        raise ValueError(
            f"{fname}: d_sw1 produced a {nslot}-slot allflux stack but "
            f"the halo tables were built with nq={ctx.tab.nq}, i.e. "
            f"4+nq = {want} slots (dyn_core.F90:855 `do iq=1,4+nq`). "
            f"Rebuild the context at the kernel's nq; a mismatch would "
            f"either average the wrong slots or refuse the stack one "
            f"frame deeper.")


def _require_barrier_layout(fname: str, ctx, afx, afy, km: int) -> None:
    """The barrier's index tables are baked for ONE leading layout.

    ``average_allflux_shared_edges`` blends through a FLAT index table
    built for ``(6, npx, n)`` / ``(6, n, npx)`` planes.  A wrong
    leading shape does NOT raise there: ``jnp.ndarray.at[idx].set()``
    CLAMPS out-of-bounds indices instead of erroring, so a mis-shaped
    operand would be silently blended at the wrong cells.  That makes
    this check load-bearing rather than defensive politeness.
    """
    n, npx = ctx.n, ctx.npx
    want_x = (6, npx, n, km)
    want_y = (6, n, npx, km)
    if afx.shape[:4] != want_x or afy.shape[:4] != want_y:
        raise ValueError(
            f"{fname}: barrier-1 operands have leading shapes "
            f"{afx.shape[:4]} / {afy.shape[:4]}, expected {want_x} / "
            f"{want_y} for n={n} km={km}. The blend's index table is "
            f"baked for that layout and `.at[].set()` CLAMPS rather "
            f"than raising, so a mismatch would blend the wrong cells "
            f"silently.")


def _validate_stacked(fname: str, container: dict, ctx, km: int,
                      required: tuple, *, what: str) -> None:
    """Tier-0 gate: every required key present, every shape declared.

    Shapes are ``(6,) + field_shape(name, n, ng, km)`` with
    ``CSW_OUT_LIKE`` mapping a C-grid output name onto the field whose
    shape it shares.  Checking is not optional politeness: a stagger
    slip between ``(m_a, m_b)`` and ``(m_b, m_a)`` BROADCASTS in a
    later arithmetic op instead of raising.
    """
    if not isinstance(container, dict):
        raise TypeError(
            f"{fname}: {what} must be the face-stacked dict of arrays "
            f"(convention C1), got {type(container).__name__}. A list of "
            f"six per-face dicts is the NumPy lane's container -- convert "
            f"it with fv3_cgrid_phase_3d.state_3d_to_jax().")
    missing = [k for k in required if k not in container]
    if missing:
        raise KeyError(
            f"{fname}: {what} is missing {missing}; keys are "
            f"{sorted(container)}")
    n, ng = ctx.n, ctx.ng
    for name in required:
        a = jnp.asarray(container[name])
        want = (6,) + field_shape(CSW_OUT_LIKE.get(name, name), n, ng, km)
        if a.shape != want:
            raise ValueError(
                f"{fname}: {what}[{name!r}] has shape {a.shape}, expected "
                f"{want} for n={n} ng={ng} km={km}")


def _cap_shape(name: str, n: int, ng: int, km: int) -> tuple:
    """Declared capacitor shape -- ``alloc_flux_capacitors``' contract.

    Transcribed from that allocator's docstring rather than re-derived
    (``fv3_native_tracer2d.py:75-78``)::

        mfx (npx, m_a, km)  origin (is,  js)   used i=is..ie+1, j=js..je
        mfy (m_a, npx, km)  origin (is,  js)   used i=is..ie,   j=js..je+1
        cx  (npx, m_a, km)  origin (is,  jsd)  used i=is..ie+1, j=jsd..jed
        cy  (m_a, npx, km)  origin (isd, js)   used i=isd..ied, j=js..je+1
    """
    npx, m_a = n + 1, n + 2 * ng
    if name in ("mfx", "cx"):
        return (6, npx, m_a, km)
    if name in ("mfy", "cy"):
        return (6, m_a, npx, km)
    raise KeyError(f"unknown capacitor {name!r}; expected one of "
                   f"{list(CAPACITOR_FIELDS)}")


def _validate_flux_cap(fname: str, flux_cap: dict, ctx, km: int) -> None:
    """Every capacitor present and at its declared shape.

    A missing one cannot be defaulted to zeros: the capacitors are an
    ACCUMULATOR the caller owns across substeps, so silently starting
    one from zero would discard the transport already banked in it.
    """
    if not isinstance(flux_cap, dict):
        raise TypeError(
            f"{fname}: flux_cap must be the face-stacked capacitor dict "
            f"(convention C1) or None, got {type(flux_cap).__name__}")
    missing = [k for k in CAPACITOR_FIELDS if k not in flux_cap]
    if missing:
        raise KeyError(
            f"{fname}: flux_cap is missing {missing}; the capacitors are "
            f"exactly {list(CAPACITOR_FIELDS)} and a missing one cannot "
            f"be defaulted to zeros -- it is an accumulator the caller "
            f"owns across substeps (sw_core.F90:903-920, intent(inout)). "
            f"Allocate with "
            f"fv3_native_tracer2d.alloc_flux_capacitors(n, ng, km).")
    for name in CAPACITOR_FIELDS:
        a = jnp.asarray(flux_cap[name])
        want = _cap_shape(name, ctx.n, ctx.ng, km)
        if a.shape != want:
            raise ValueError(
                f"{fname}: flux_cap[{name!r}] has shape {a.shape}, "
                f"expected {want} (alloc_flux_capacitors' contract for "
                f"n={ctx.n} ng={ctx.ng} km={km})")


# ---------------------------------------------------------------------
# assembly helpers
# ---------------------------------------------------------------------

def _stack_levels(fname: str, name: str, per_level: list):
    """``km`` per-level outputs -> one array with ``km`` at AXIS 2.

    Axis 2 rather than -1 so an ``allflux`` slab ``(i, j, slot)``
    becomes ``(i, j, km, slot)`` -- the oracle's own
    ``allflux_x(i,j,k,iq)`` layout (``dyn_core.F90:860``) and the
    spec's ``np.stack(..., axis=2)``.  For a plain 2-D output the two
    conventions coincide.

    The reference shape is level 0's, and every other level is required
    to match it.  A per-level shape TABLE is not used because
    ``d_sw1``/``d_sw2`` publish no such table -- re-deriving ten Fortran
    bound expressions here would be exactly the kind of restatement
    that drifts.  What this catches is the defect that could actually
    occur: a level whose output shape differs from its neighbours'.

    ``jnp.stack`` is a pure index copy -- no ``x*y + z`` for XLA to
    contract into an FMA -- which is why the jit-vs-eager gate on the
    assembly step alone may be BITWISE while the gates on the kernels'
    arithmetic may not.
    """
    want = per_level[0].shape
    for k, arr in enumerate(per_level[1:], start=1):
        if arr.shape != want:
            raise ValueError(
                f"{fname}: level {k} output {name!r} has shape "
                f"{arr.shape}, level 0 has {want}. A stagger or window "
                f"mismatch here would broadcast, not raise.")
    return jnp.stack(per_level, axis=2)


def _stack_faces(fname: str, per_face: list) -> dict:
    """Six per-face dicts of 3-D arrays -> one dict of ``(6, …)`` stacks.

    All six dicts carry the same keys and, within one key, the same
    per-face shape -- the FACE axis is stackable, the stagger axis is
    not, which is why this stacks per key and never across keys.
    """
    keys = tuple(per_face[0])
    for t, d in enumerate(per_face):
        if tuple(d) != keys:
            raise KeyError(
                f"{fname}: face {t + 1} produced keys {sorted(d)}, face 1 "
                f"produced {sorted(keys)} -- a per-face lane split")
    return {k: jnp.stack([d[k] for d in per_face], axis=0) for k in keys}


# ---------------------------------------------------------------------
# the post-p_grad_c duo exchanges (dyn_core.F90:652 / :655), per level
# ---------------------------------------------------------------------

def exchange_post_pgrad_3d(ctx, csw_outs: dict, km, *, nord: int) -> dict:
    """Apply the shared post-``p_grad_c`` duo exchanges at every level.

    ``dyn_core.F90:652``::

        if(duogrid .and. flagstruct%nord > 0) call ext_scalar(divgd, ...)

    and ``:655``::

        if (duogrid) call ext_vector(uc, vc, ..., 1,0,0,1)

    (``:653`` is blank and ``:654`` is the ``.not. duogrid`` group-halo
    completion; ``:706``/``:709`` are the REGIONAL branch and are not
    on this lane at all.)

    THE GATE IS ``nord``, the DIVERGENCE-damping order ``d_sw5`` runs,
    not ``nord_v``, the vorticity-damping order.  The shipped decks set
    both to 2, so gating on the wrong one is invisible there -- which
    is exactly why it is stated.  ``exchange_post_pgrad_sixface`` owns
    the gate; this routine only supplies the per-level cadence.

    Omitting these leaves the ``uc``/``vc``/``divgd`` halos stale and
    ``d_sw1`` transports garbage: in the NumPy lane the first sub-step
    produced ``delp ~ -1.5e39`` and 267 non-finite ``u`` values before
    any exchange was added.

    The oracle's exchanges are 3-D calls over the whole column; this
    lane holds one 2-D plane per level, so it drives the shared
    six-face helper once per ``k``.  Level-independent by construction
    -- ``ext_scalar`` / ``ext_vector`` are horizontal operators.

    Returns ``{"divg_d", "uc", "vc"}`` face-stacked and
    ``(6, i, j, km)``-shaped.  The spec MUTATES those three keys of
    ``csw_outs`` in place; here they come back and the caller threads
    them (R4 / convention C4), and the spec's per-level ``np.copy`` is
    dropped because a functional slice has no buffer to alias (D2).
    """
    km = _require_km("exchange_post_pgrad_3d", km)
    nord = _require_nord("exchange_post_pgrad_3d", "nord", nord)
    _validate_stacked("exchange_post_pgrad_3d", csw_outs, ctx, km,
                      _EXCHANGE_FIELDS, what="csw_outs")
    _require_f64_jax("exchange_post_pgrad_3d",
                     {k: csw_outs[k] for k in _EXCHANGE_FIELDS})

    per_level = {name: [] for name in _EXCHANGE_FIELDS}
    # R1a, level axis: `ext_scalar`/`ext_vector` are HORIZONTAL
    # operators -- each level's exchange reads and writes only that
    # level's plane, and this lane is functional, so the NumPy lane's
    # strided `fort` views onto one buffer do not exist here.  No
    # iteration reads a location another writes.  Independent,
    # therefore vectorisable, therefore NOT vectorised (C2): the halo
    # tables index a FLAT (6, m, m) layout, so a batched level axis
    # would need its own table and its own equivalence gate.
    for k in range(km):
        dg_k, uc_k, vc_k = exchange_post_pgrad_sixface(
            ctx, csw_outs["divg_d"][:, :, :, k], csw_outs["uc"][:, :, :, k],
            csw_outs["vc"][:, :, :, k], nord=nord)
        per_level["divg_d"].append(dg_k)
        per_level["uc"].append(uc_k)
        per_level["vc"].append(vc_k)
    # Each per-level entry is already face-stacked `(6, i, j)`, so the
    # level axis goes to position 3 here, not 2 -- the result is
    # `(6, i, j, km)`, which is `(6,) + field_shape(name, n, ng, km)`.
    return {name: jnp.stack(per_level[name], axis=3)
            for name in _EXCHANGE_FIELDS}


# ---------------------------------------------------------------------
# d_sw1 -> BARRIER 1 -> d_sw2, km-general (dyn_core.F90:744-950)
# ---------------------------------------------------------------------

def dsw_transport_phase_3d(ctx, states: dict, csw_outs: dict, dt, km, *,
                           cfg=None,
                           hydrostatic: bool = True,
                           remap_follows: bool = False,
                           kgb: float | None = None,
                           nord_w: int | None = None,
                           damp_w: float | None = None,
                           flux_cap: dict | None = None) -> dict:
    """``d_sw1`` (per k) -> BARRIER 1 (per k) -> ``d_sw2`` (per k).

    The exchanges at ``dyn_core.F90:652``/``:655`` run FIRST, before
    ``d_sw1``, via :func:`exchange_post_pgrad_3d`.

    ``states`` supplies the D winds and the A-grid scalars;
    ``csw_outs`` supplies ``uc``/``vc`` as updated by ``p_grad_c`` in
    the C-grid phase, plus ``divg_d`` from ``c_sw``.

    ``dt`` is the FULL acoustic step (``dyn_core.F90:831`` passes
    ``dt``, not ``dt2``) and is DYNAMIC (C3).

    ``cfg`` is an :class:`~legoesm.core.fv3_duo_stepper.SWConfig`;
    ``None`` selects :data:`DSW_DUO_DECK`, the shipped duo deck built
    from the spec's own ``DUO_DECK_CFG``.  ``kgb``/``nord_w``/
    ``damp_w`` are the three ``d_sw2`` knobs ``SWConfig`` does not
    carry; ``None`` reproduces the spec's fallbacks exactly
    (``kgb -> 0.0``, ``nord_w -> cfg.nord_v``, ``damp_w -> cfg.damp_v``
    on the NH arm and ``0.0`` on the hydrostatic one).

    ``flux_cap``, when given, is the face-stacked ``mfx``/``mfy``/
    ``cx``/``cy`` capacitor bundle
    (``fv3_native_tracer2d.alloc_flux_capacitors``, stacked over faces)
    that ``d_sw1`` accumulates into every sub-step
    (``sw_core.F90:903-920``) for ``tracer_2d``'s large-time-step
    transport.  The accumulation happens BEFORE barrier 1 edits
    ``allflux``, so the capacitors carry the UN-averaged fluxes,
    exactly as the oracle's do -- the averaging block touches
    ``allflux_x/y``, never ``mfx``/``xflux``.  ``None`` keeps the
    spec's discard-the-capacitors behaviour for callers that advect no
    tracers, and then no capacitor key is returned.

    Returns, face-stacked with ``km`` at axis 2:

    * every :data:`DSW1_OUT_2D` output, with ``allflux_x``/
      ``allflux_y`` AVERAGED by the barrier (stage S09);
    * ``allflux_x_prebarrier`` / ``allflux_y_prebarrier``, the same
      stacks BEFORE the barrier (stage S08 -- the only stage not
      recoverable from the others, C6);
    * ``delp`` / ``pt``, ``d_sw2``-updated (stage S10);
    * ``uc`` / ``vc`` / ``divg_d``, exchanged (stage S07);
    * ``w`` and ``dw`` on the NH arm only;
    * ``mfx`` / ``mfy`` / ``cx`` / ``cy`` when ``flux_cap`` was given.
    """
    km = _require_km("dsw_transport_phase_3d", km)
    for nm, vv in (("hydrostatic", hydrostatic),
                   ("remap_follows", remap_follows)):
        _require_bool("dsw_transport_phase_3d", nm, vv)
    require_no_remap_needed(km, remap_follows=remap_follows)
    c = _resolve_cfg("dsw_transport_phase_3d", cfg)

    # The spec's `c.get(...)` fallbacks, made explicit.  `damp_w` is
    # forced to 0.0 on the hydrostatic arm exactly as the spec does --
    # d_sw2's del-6 `dw` block is NH-only, and a non-zero damp_w there
    # would additionally make the kernel demand `del6_u`/`del6_v`.
    kgb = 0.0 if kgb is None else float(kgb)
    # NOT `int(...)` before the guard: `int(2.7)` rounds to 2 without a
    # word, which is exactly what `_require_nord` exists to refuse.
    nord_w = c.nord_v if nord_w is None else nord_w
    nord_w = _require_nord("dsw_transport_phase_3d", "nord_w", nord_w)
    if hydrostatic:
        damp_w = 0.0
    else:
        damp_w = float(c.damp_v) if damp_w is None else float(damp_w)

    _validate_stacked("dsw_transport_phase_3d", states, ctx, km,
                      STATE_FIELDS, what="states")
    _validate_stacked("dsw_transport_phase_3d", csw_outs, ctx, km,
                      _EXCHANGE_FIELDS, what="csw_outs")
    _require_f64_jax("dsw_transport_phase_3d",
                     {k: states[k] for k in STATE_FIELDS}
                     | {k: csw_outs[k] for k in _EXCHANGE_FIELDS}
                     | {"dt": jnp.asarray(dt)})
    if flux_cap is not None:
        _validate_flux_cap("dsw_transport_phase_3d", flux_cap, ctx, km)
        _require_f64_jax("dsw_transport_phase_3d",
                         {f"flux_cap[{k}]": flux_cap[k]
                          for k in CAPACITOR_FIELDS})

    bd, npx, n, m_a = ctx.bd, ctx.npx, ctx.n, ctx.m_a

    # --- POST-p_grad_c duo exchanges, BEFORE d_sw1 (:652 / :655) ------
    # `c.nord` is forwarded RAW: `int(c.nord)` here would round a 2.7
    # to 2 before `exchange_post_pgrad_3d`'s guard could refuse it.
    ex = exchange_post_pgrad_3d(ctx, csw_outs, km, nord=c.nord)
    uc6, vc6, divgd6 = ex["uc"], ex["vc"], ex["divg_d"]

    # --- d_sw1 at every level, on every face (:744 / :831) ------------
    # Held as [face][k] rather than merged: the barrier consumes one
    # level at a time, and the oracle's own barrier block is a
    # `do k=1,npz` (dyn_core.F90:857 / :877).
    per_face_levels: list[list[dict]] = []
    zx = jnp.zeros((npx, n), dtype=jnp.float64)
    zy = jnp.zeros((n, npx), dtype=jnp.float64)
    zcx = jnp.zeros((npx, m_a), dtype=jnp.float64)
    zcy = jnp.zeros((m_a, npx), dtype=jnp.float64)
    # R1a, face axis: `d_sw1_duo` reads only face t's own fields,
    # gridstruct and flags and writes only face t's outputs -- no
    # iteration reads a location another writes, and this lane is
    # functional, so the NumPy lane's `fort` 1-based views onto one
    # buffer do not exist here at all.  Cross-face coupling enters ONLY
    # at the exchanges above and at the barrier below, both of which
    # see all six faces at once.  Kept as a literal loop (C2).
    for t in range(6):
        levels = []
        # R1a, level axis: `d_sw1`'s dummies are 2-D --
        # `dimension(bd%isd:bd%ied, bd%jsd:bd%jed)` at
        # sw_core.F90:516 -- so a level CANNOT read another level, and
        # the oracle's own `do k=1,npz` (dyn_core.F90:744) is what
        # makes the per-level call the faithful lift.  Independent,
        # therefore vectorisable, therefore NOT vectorised (C2).
        for k in range(km):
            if flux_cap is None:
                xflux, yflux, cxk, cyk = zx, zy, zcx, zcy
            else:
                # D4: `mfx`/`mfy` are allocated one halo-padded axis
                # wider than the kernel's dummy; slice to the WINDOW
                # the kernel writes (`mfx[:, :n]`, `mfy[:n, :]`).
                # `cx`/`cy` already match and are passed whole.
                xflux = flux_cap["mfx"][t][:, :n, k]
                yflux = flux_cap["mfy"][t][:n, :, k]
                cxk = flux_cap["cx"][t][:, :, k]
                cyk = flux_cap["cy"][t][:, :, k]
            levels.append(d_sw1_duo(
                states["delp"][t][:, :, k], states["pt"][t][:, :, k],
                states["w"][t][:, :, k],
                uc6[t][:, :, k], vc6[t][:, :, k],
                xflux, yflux, cxk, cyk,
                ctx.gs6[t], ctx.flags6[t], bd, npx, npx, dt=dt,
                hord_tr=c.hord_tr, hord_vt=c.hord_vt,
                hord_tm=c.hord_tm, hord_dp=c.hord_dp,
                nord_v=c.nord_v, nord_t=0,
                damp_v=c.damp_v, damp_t=0.0,
                hydrostatic=hydrostatic,
                workspace_sentinel=0.0))
        per_face_levels.append(levels)

    # --- capacitor write-back (R4 / C4) -------------------------------
    caps = None
    if flux_cap is not None:
        # R1a, write-back: level k writes only the `[..., k]` slot of
        # each capacitor and face t only face t's slab, so every write
        # lands in a disjoint location and no iteration reads one
        # another wrote.  Functional `.at[].set()` throughout -- the
        # spec's `cap["mfx"][:, :, k] = ...` in-place stores become
        # these (R4), and the oracle's dummies are intent(inout).
        caps = {}
        for name in CAPACITOR_FIELDS:
            src = _CAP_FROM_DSW1[name]
            per_face = []
            for t in range(6):
                arr = jnp.asarray(flux_cap[name][t])
                for k in range(km):
                    v = per_face_levels[t][k][src]
                    if name == "mfx":
                        arr = arr.at[:, :n, k].set(v)
                    elif name == "mfy":
                        arr = arr.at[:n, :, k].set(v)
                    else:
                        arr = arr.at[:, :, k].set(v)
                per_face.append(arr)
            caps[name] = jnp.stack(per_face, axis=0)

    # --- BARRIER 1: inter-panel flux average, one level at a time -----
    # dyn_core.F90:872 (`mpp_get_boundary`, `gridtype=CGRID_NE` at
    # :874) then the 0.5*(mine + neighbour) blend at :877-885.  The
    # SLOT SELECTION is :856, `if (iq==1 .or. iq==4 .or. iq>4)` -- slot
    # 2 (w) and slot 3 (q_con) are NOT averaged.  This module does not
    # re-derive that: `average_allflux_shared_edges` selects through
    # `tab.allflux_slots`, which bakes the exclusion in structurally.
    #
    # Averaging must happen with ALL SIX FACES present for that level;
    # doing it inside the per-face loop above would average a face
    # against a stale neighbour.
    #
    # LOOP ORDER, stated because it differs from the oracle's: upstream
    # runs `do iq` OUTSIDE and `do k` inside (:855 / :857, :877); this
    # runs `for k` outside and all selected `iq` at once inside the
    # helper.  Legal because the blend at (iq, k) reads and writes only
    # (iq, k) -- the buffers `mpp_get_boundary` fills are indexed
    # (edge, k) per iq and both operands come from the PRE-update
    # buffer -- so the two loops commute.  R1a: no iteration reads a
    # location another writes, in either order.
    afx_pre = [_stack_levels("dsw_transport_phase_3d", "allflux_x",
                             [per_face_levels[t][k]["allflux_x"]
                              for k in range(km)]) for t in range(6)]
    afy_pre = [_stack_levels("dsw_transport_phase_3d", "allflux_y",
                             [per_face_levels[t][k]["allflux_y"]
                              for k in range(km)]) for t in range(6)]
    afx_pre = jnp.stack(afx_pre, axis=0)
    afy_pre = jnp.stack(afy_pre, axis=0)
    _require_barrier_nq("dsw_transport_phase_3d", ctx,
                        int(afx_pre.shape[-1]))
    _require_barrier_layout("dsw_transport_phase_3d", ctx, afx_pre,
                            afy_pre, km)

    afx_lv, afy_lv = [], []
    for k in range(km):
        ax, ay = average_allflux_shared_edges(afx_pre[:, :, :, k, :],
                                              afy_pre[:, :, :, k, :],
                                              ctx.tab)
        afx_lv.append(ax)
        afy_lv.append(ay)
    afx6 = jnp.stack(afx_lv, axis=3)
    afy6 = jnp.stack(afy_lv, axis=3)

    # --- d_sw2 at every level, on the AVERAGED fluxes (:914 / :950) ---
    names = ("delp", "pt") + (() if hydrostatic else ("w", "dw"))
    per_face = []
    # R1a, face axis: `d_sw2_duo` is a pure cell update given the
    # fluxes -- it reads face t's delp/pt/w and face t's averaged
    # allflux slabs and writes face t's outputs.  Level axis: d_sw2's
    # dummies are 2-D as well (sw_core.F90:1000-1012 declares the same
    # `bd%isd:bd%ied, bd%jsd:bd%jed` shapes), and the oracle's own
    # `do k=1,npz` at :914 wraps the :950 call.  Both independent, both
    # left as literal loops (C2).
    for t in range(6):
        per_level = {name: [] for name in names}
        for k in range(km):
            s1 = per_face_levels[t][k]
            s2 = d_sw2_duo(
                s1["delp"], s1["pt"], afx6[t][:, :, k, :],
                afy6[t][:, :, k, :], ctx.gs6[t], ctx.flags6[t], bd,
                w=(None if hydrostatic else s1["w"]),
                npx=npx, npy=npx, dt=dt, kgb=kgb, nord_w=nord_w,
                damp_w=damp_w, hydrostatic=hydrostatic)
            for name in names:
                if name not in s2 or s2[name] is None:
                    raise KeyError(
                        f"d_sw2 returned no {name!r} (keys "
                        f"{sorted(s2)}); the 3-D assembler must not "
                        f"silently drop a stage output")
                per_level[name].append(jnp.asarray(s2[name]))
        per_face.append({
            name: _stack_levels(f"dsw_transport_phase_3d[face {t + 1}]",
                                name, per_level[name])
            for name in names})
    s2s = _stack_faces("dsw_transport_phase_3d", per_face)

    # `allflux_x`/`allflux_y` are EXCLUDED from this comprehension --
    # they were already stacked into `afx_pre`/`afy_pre` above, and
    # re-stacking them here would trace a second copy of the two
    # largest arrays only to overwrite it with the averaged version.
    out = {name: jnp.stack(
        [_stack_levels(f"dsw_transport_phase_3d[face {t + 1}]", name,
                       [per_face_levels[t][k][name] for k in range(km)])
         for t in range(6)], axis=0)
        for name in DSW1_OUT_2D if not name.startswith("allflux_")}
    out["allflux_x"] = afx6
    out["allflux_y"] = afy6
    out["allflux_x_prebarrier"] = afx_pre
    out["allflux_y_prebarrier"] = afy_pre
    out.update(s2s)
    out["uc"], out["vc"], out["divg_d"] = uc6, vc6, divgd6
    if caps is not None:
        out.update(caps)
    return out


# ---------------------------------------------------------------------
# jit factories -- ONE policy per public routine, declared here so a
# call site can never invent a different static split.
#
# `ctx` is STATIC (hashed by identity; it carries the six gridstructs,
# the static flags, `hs` and the halo tables, all constants in this
# lane); `km`, `cfg` and every deck knob a Python `if` reads are
# STATIC; the field data, the capacitors and `dt` are DYNAMIC (C3).
# Each `km`-style argument appears in BOTH `static_argnums` and
# `static_argnames` so it is static whether the caller passes it
# positionally or by keyword -- jax resolves the two independently when
# both are given.  No `donate_argnums` anywhere: buffer donation
# conflicts with reverse-mode AD, which is the point of the lane.
# ---------------------------------------------------------------------

def make_exchange_post_pgrad_3d_jit(fn=exchange_post_pgrad_3d):
    """Static: ``ctx``, ``km``, ``nord``.  ``csw_outs`` dynamic."""
    return jax.jit(fn, static_argnums=(0, 2),
                   static_argnames=("km", "nord"))


def make_dsw_transport_phase_3d_jit(fn=dsw_transport_phase_3d):
    """Static: ``ctx``, ``km``, ``cfg``, ``hydrostatic``,
    ``remap_follows``, ``kgb``, ``nord_w``, ``damp_w``.  ``states``,
    ``csw_outs``, ``flux_cap`` and ``dt`` dynamic.

    ``cfg`` must be an ``SWConfig`` (hashable by value) -- every one of
    its fields is read by a Python ``if`` or a Python power inside
    ``d_sw1``/``d_sw2``, so it cannot be traced.  ``damp_w`` is static
    for the same reason (``d_sw2_duo`` selects which grid metrics it
    associates on ``damp_w <= 1e-5``); ``kgb`` and ``nord_w`` are
    static because they arrive from the same deck, which is a jit
    POLICY and not numerics -- the kernel's own
    ``make_d_sw2_duo_jit`` keeps ``kgb`` dynamic, and the values are
    identical either way.

    ``flux_cap=None`` changes the pytree and correctly retraces.
    """
    return jax.jit(fn, static_argnums=(0, 4),
                   static_argnames=("km", "cfg", "hydrostatic",
                                    "remap_follows", "kgb", "nord_w",
                                    "damp_w"))
