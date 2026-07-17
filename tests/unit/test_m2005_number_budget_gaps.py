"""M2005 double-moment number-budget gaps vs the SAM oracle — the follow-up
scoped OUT of the fall-speed audit (m2005-double-moment-numberbudget-followup),
now closed.  Oracle: gSAM MICRO_M2005 module_mp_graupel.f90.

1. COOPER counts the TOTAL frozen number (SAM :3396-3399):
       IF (KC2 > NI3D+NS3D+NG3D) NNUCCD = (KC2-NI3D-NS3D-NG3D)/DT
   legoESM subtracted only N_i, so a column whose prognostic snow/graupel
   number already satisfied the Cooper target kept nucleating cloud ice
   (and its MI0 mass source).
2. NSMLTR/NGMLTR (SAM :2189-2203, :2211): melted snow-flake / graupel-
   particle number becomes RAIN number.  legoESM debited N_s/N_g on melt
   but the number then VANISHED — rain in melting layers under-counted
   drops (too-large mean size -> too-fast fallout, too-little evaporation).

(The third memo item, NPRCI, needed no code change: it was already wired as
``dN_i_autoconv`` and pinned in test_m_double_moment_snow.py — only the
comments claimed otherwise.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio_ice

from legoesm import constants

jax.config.update("jax_enable_x64", True)


def _run(*, T=250.0, p=4.0e4, ssat_ice=0.15, q_i=0.0, N_i=0.0,
         q_s=0.0, N_s=None, q_g=0.0, N_g=None, q_r=0.0, N_r=0.0,
         dt=20.0, config=None):
    """One-level column; every unrelated hydrometeor zero so the budget under
    test is the only active term."""
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(p)))
    rho = p / (constants.R_d * T)
    z = jnp.zeros((1, 1))

    def _f(v):
        return None if v is None else jnp.full((1, 1), float(v))

    hm = HydrometeorState(
        q_c=z, q_r=_f(q_r), q_i=_f(q_i), q_s=_f(q_s), q_g=_f(q_g),
        N_c=z, N_r=_f(N_r), N_i=_f(N_i), N_s=_f(N_s), N_g=_f(N_g),
    )
    cfg = config if config is not None else MorrisonConfig()
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), (1.0 + ssat_ice) * qsi), hm,
        jnp.full((1, 1), p), jnp.full((1, 2), p), jnp.full((1, 1), rho),
        jnp.full((1, 1), 300.0), dt, cfg)
    return out, rho


# ---------------------------------------------------------------------------
# 1. Cooper target counts NI3D + NS3D + NG3D
# ---------------------------------------------------------------------------

class TestCooperCountsTotalFrozenNumber:
    """Isolation: q_i = N_i = 0 and q_c = 0, so dN_i_dt has no autoconversion
    (gated on q_i > 1e-14), no homogeneous freezing, no sedimentation — the
    nucleation term is the only contributor."""

    def test_saturated_snow_number_suppresses_nucleation(self):
        # N_s far above any Cooper target (cap is 500 L^-1 = 5e5 m^-3
        # ~ 1e6/kg at this rho): the column already holds the frozen number,
        # so SAM would nucleate NOTHING.
        out_rich, _ = _run(q_s=5.0e-4, N_s=1.0e7)
        out_poor, _ = _run(q_s=5.0e-4, N_s=0.0)
        assert float(out_rich.dN_i_dt[0, 0]) == 0.0
        assert float(out_poor.dN_i_dt[0, 0]) > 0.0

    def test_saturated_graupel_number_suppresses_nucleation(self):
        cfg = MorrisonConfig(do_graupel=True)
        out_rich, _ = _run(q_g=5.0e-4, N_g=1.0e7, config=cfg)
        out_poor, _ = _run(q_g=5.0e-4, N_g=0.0, config=cfg)
        assert float(out_rich.dN_i_dt[0, 0]) == 0.0
        assert float(out_poor.dN_i_dt[0, 0]) > 0.0

    def test_single_moment_path_unchanged(self):
        """No prognostic N_s/N_g -> the target is reduced by N_i only, the
        exact prior behavior (byte-identical old AD path)."""
        out_sm, _ = _run(q_s=5.0e-4, N_s=None)
        out_dm0, _ = _run(q_s=5.0e-4, N_s=0.0)
        # A zero prognostic snow number contributes zero to the frozen count,
        # so the NUCLEATION term agrees with single-moment exactly.
        assert float(out_sm.dN_i_dt[0, 0]) == float(out_dm0.dN_i_dt[0, 0])
        assert float(out_sm.dN_i_dt[0, 0]) > 0.0

    def test_mass_source_suppressed_with_the_number(self):
        """MNUCCD = NNUCCD*MI0 rides on the number deficit: no deficit, no
        nucleation mass source either (dq_i from nucleation ~ 0)."""
        out_rich, _ = _run(q_s=5.0e-4, N_s=1.0e7)
        # dq_i_dt in this isolated column = nucleation mass only (no ice, no
        # deposition target without q_i? deposition needs N_i>0 -> 0 here).
        assert abs(float(out_rich.dq_i_dt[0, 0])) < 1.0e-30

    def test_negative_snow_number_cannot_inflate_nucleation(self):
        """A transient N_s < 0 is clipped at 0 in the frozen count: nucleation
        must not EXCEED the N_s=0 rate."""
        out_neg, _ = _run(q_s=5.0e-4, N_s=-1.0e6)
        out_zero, _ = _run(q_s=5.0e-4, N_s=0.0)
        assert (float(out_neg.dN_i_dt[0, 0])
                <= float(out_zero.dN_i_dt[0, 0]) + 1e-12)


# ---------------------------------------------------------------------------
# 2. NSMLTR / NGMLTR — melted number becomes rain number
# ---------------------------------------------------------------------------

class TestMeltNumberBecomesRainNumber:
    """Isolation: warm subsaturated column, q_r = N_r = q_c = 0 -> rain
    autoconversion/self-collection/evaporation/freezing/sedimentation all
    zero; the melt route is dN_r_dt's only possible source."""

    def test_melting_snow_number_arrives_in_rain(self):
        out, rho = _run(T=280.0, p=9.0e4, ssat_ice=-0.5,
                        q_s=1.0e-3, N_s=2.0e4)
        dnr = float(out.dN_r_dt[0, 0])
        assert dnr > 0.0
        # Cannot create more drops than there are flakes (per-volume).
        assert dnr * 20.0 <= 2.0e4 * rho * (1.0 + 1e-12)
        # And the snow number budget records a matching sink direction.
        assert float(out.dN_s_dt[0, 0]) < 0.0

    def test_melting_graupel_number_arrives_in_rain(self):
        cfg = MorrisonConfig(do_graupel=True)
        out, rho = _run(T=280.0, p=9.0e4, ssat_ice=-0.5,
                        q_g=1.0e-3, N_g=2.0e4, config=cfg)
        dnr = float(out.dN_r_dt[0, 0])
        assert dnr > 0.0
        assert dnr * 20.0 <= 2.0e4 * rho * (1.0 + 1e-12)
        assert float(out.dN_g_dt[0, 0]) < 0.0

    def test_cold_column_no_spurious_rain_number(self):
        """Same snow column at 250 K: nothing melts, no rain-number source."""
        out, _ = _run(T=250.0, ssat_ice=-0.5, q_s=1.0e-3, N_s=2.0e4)
        assert float(out.dN_r_dt[0, 0]) == 0.0

    def test_single_moment_snow_has_no_number_to_route(self):
        """N_s=None: no prognostic number exists, dN_r_dt stays 0 — the melt
        route only fires on the double-moment branches."""
        out, _ = _run(T=280.0, p=9.0e4, ssat_ice=-0.5, q_s=1.0e-3, N_s=None)
        assert float(out.dN_r_dt[0, 0]) == 0.0
