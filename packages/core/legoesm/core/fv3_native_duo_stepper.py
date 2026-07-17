"""Six-face integrated duo SW stepper — phase 4c production assembly.

Wires the certified duo stage ports (c_sw duo, d_sw1..d_sw6 duo) and
the six-face exchange/averaging analogs into the dyn_core duo sequence
on all six cube faces.  This is the production caller the per-stage
TRANSLATION certificates deliberately excluded (codex avg-r1 P0): the
two inter-panel averaging sites (dyn_core.F90:853-900 C-ring,
968-1020 BGRID_NE) and the c_sw output exchanges run on REAL neighbor
data here instead of the single-face sentinel contracts.

Assembly ladder (each sub-brick gated before the next):
  SB1  build_six_face_duo_context + csw_step_sixface   (this module)
  SB2  PG-C update + d_sw1 + allflux averaging + d_sw2
  SB3  d_sw3 + BGRID averaging + kee + d_sw4/5/6 = one acoustic step
  SB4  time loop + Williamson-2 run -> the duo-target gate
       (scripts/validate/fv3_native/w2_duo_oracle_gate.py)

Workspace semantics: dyn_core's utt/vtt are UNINITIALIZED stack
upstream (the always-fire d_sw1 edge blocks read them; benign there
because the panel-edge cosa metrics are near-zero).  The stepper makes
the DEFINED choice ut/vt = 0 at entry each step — documented, and the
Williamson-2 duo-target gate is the arbiter.
"""

from __future__ import annotations

from legoesm.grids.fv3_native_gridstruct import (
    analytic_swcore_state,
    build_fv3_native_gridstruct,
    exchange_bgrid_scalar_halos,
    exchange_cgrid_vector_halos,
)


def build_six_face_duo_context(n: int, ng: int = 3) -> dict:
    """Gridstructs + Bounds for all six faces (certified builders)."""
    from legoesm.core.fv3_native_sw_core import Bounds

    gs6 = [build_fv3_native_gridstruct(n, ng, tile=t) for t in range(1, 7)]
    for gs in gs6:
        gs.setdefault("bounded_domain", False)
        gs.setdefault("grid_type", 0)
        gs.setdefault("sw_corner", True)
        gs.setdefault("se_corner", True)
        gs.setdefault("ne_corner", True)
        gs.setdefault("nw_corner", True)
    return {"n": n, "ng": ng, "gs6": gs6,
            "bd": Bounds.single_tile(n, ng)}


def analytic_six_face_state(ctx: dict, **kw) -> list:
    """Per-face analytic solid-body SW state (Williamson-2-like)."""
    return [analytic_swcore_state(gs, **kw) for gs in ctx["gs6"]]


def csw_step_sixface(ctx: dict, states: list, dt2: float,
                     duogrid: bool = True) -> list:
    """SB1: certified duo c_sw on every face + the two post-c_sw
    exchanges dyn_core performs before d_sw (divgd CORNER-scalar,
    uc/vc CGRID_NE vector) on the six-face neighbor machinery.

    Returns the per-face c_sw output dicts with exchanged halos.
    """
    from legoesm.core.fv3_native_sw_core import c_sw

    n, ng = ctx["n"], ctx["ng"]
    bd = ctx["bd"]
    npx = n + 1
    outs = []
    for t in range(1, 7):
        st = states[t - 1]
        outs.append(c_sw(st["delp"], st["pt"], st.get("w", st["pt"] * 0.0),
                         st["u"], st["v"], ctx["gs6"][t - 1], bd,
                         npx, npx, dt2, duogrid=duogrid))

    divgd6 = [o["divg_d"] for o in outs]
    uc6 = [o["uc"] for o in outs]
    vc6 = [o["vc"] for o in outs]
    for t in range(1, 7):
        exchange_bgrid_scalar_halos(divgd6, t, n, ng)
        exchange_cgrid_vector_halos(uc6, vc6, t, n, ng)
    return outs
