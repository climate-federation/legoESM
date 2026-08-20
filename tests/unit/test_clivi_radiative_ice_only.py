"""CMOR ``clivi`` must report only the frozen condensate radiation SEES.

CMIP6 defines ``clivi`` as the column ice mass, "including precipitating frozen
hydrometeors ONLY IF the precipitating hydrometeor affects the calculation of
radiative transfer in model". This model's radiation reads cloud ice alone --
``radiation/integration.py`` takes ``tracers["q_i"]`` / tracer slot 3 and never
receives snow (slot 4) or graupel (slot 5) -- so snow and graupel are
radiatively inert here and must not be reported.

They previously were, and it is not a small correction: snow+graupel were
roughly half the published clivi on lat-lon and ~70% on MPAS in this campaign's
checkpoints. (Indicative only -- observational IWP products differ in whether
they include precipitating ice, so the size of any model bias is NOT settled by
this change; what is settled is that the model was reporting a quantity its own
radiation never used.)

If snow is ever wired INTO the radiation it becomes radiatively active and
CMIP6 then requires it here; ``test_snow_is_excluded_only_while_radiation_
ignores_it`` documents that coupling so the exclusion is revisited rather than
silently kept.
"""
from __future__ import annotations



def test_snow_and_graupel_are_not_summed_into_the_ice_path():
    """The defect: q_frozen must not accumulate q_s/q_g.

    Asserted against ``DiagnosticCollector.collect``, the method that actually
    computes clivi -- not a wrapper.
    """
    import inspect

    from legoesm.driver.diagnostics import DiagnosticCollector

    src = inspect.getsource(DiagnosticCollector.collect)
    # The old code looped over (q_i, q_s, q_g) accumulating into q_frozen.
    assert "for q_frz in (q_i, q_s, q_g)" not in src, (
        "snow and graupel are being summed into clivi again; radiation does "
        "not see them (integration.py reads tracers['q_i'] only)")
    assert "q_frozen = q_i" in src, (
        "clivi's frozen path is no longer plain cloud ice")


def test_snow_is_excluded_only_while_radiation_ignores_it():
    """Couple the exclusion to the radiation it is justified by.

    If radiation ever starts consuming snow, CMIP6 requires snow in clivi and
    this test goes red, forcing the diagnostic to be revisited instead of
    silently staying wrong.
    """
    import inspect

    from legoesm.atmosphere.physics.radiation import integration

    src = inspect.getsource(integration)
    # The radiation builds its ice column from q_i / tracer slot 3 only.
    assert 'tracers["q_i"]' in src or "tracers['q_i']" in src, (
        "radiation no longer reads q_i by name — re-derive which frozen "
        "species are radiatively active before trusting clivi")
    for frozen in ('tracers["q_s"]', 'tracers["q_g"]'):
        assert frozen not in src, (
            f"radiation now consumes {frozen}; it is radiatively ACTIVE, so "
            "CMIP6 requires it in clivi — re-add it there")


def test_snow_and_graupel_do_not_change_the_EMITTED_clivi():
    """Numerical end-to-end: run ``collect`` twice and compare emitted planes.

    The source assertions above are SYNTACTIC -- `q_frozen = q_i` followed by
    `q_frozen += q_s` would satisfy them (codex review). This runs the real
    collector with zero vs large snow/graupel and requires the published
    ``clivi``/``clwvi`` to be byte-identical, which no syntactic dodge passes.
    """
    import jax.numpy as jnp
    import numpy as np

    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.diagnostics import DiagnosticCollector

    NLAT, NLON, NLEV = 36, 72, 8

    def _run(q_s, q_g):
        coll = DiagnosticCollector(
            nlev=NLEV, sigma_full=np.linspace(0.06, 0.97, NLEV),
            dsigma=np.full(NLEV, 1.0 / NLEV), experiment_id="amip",
            monthly_means=True, cmip_output=True, n_days=30,
            cmip_resolution_deg=5.0, start_year=1979)

        class _Grid:
            lat = np.deg2rad(np.linspace(-87.5, 87.5, NLAT))
            lon = np.deg2rad(np.linspace(2.5, 357.5, NLON))

        coll.set_cmip_grid_info("latlon", grid=_Grid(), start_year=1979)
        s3, s2 = (NLAT, NLON, NLEV), (NLAT, NLON)
        state = HydrostaticState(
            u=Field(jnp.zeros(s3), name="u", dims=("lat", "lon", "level"),
                    units="m/s"),
            v=Field(jnp.zeros(s3), name="v", dims=("lat", "lon", "level"),
                    units="m/s"),
            T=Field(jnp.full(s3, 265.0), name="T",
                    dims=("lat", "lon", "level"), units="K"),
            p_s=Field(jnp.full(s2, 101325.0), name="p_s", dims=("lat", "lon"),
                      units="Pa"),
            phis=Field(jnp.zeros(s2), name="phis", dims=("lat", "lon"),
                       units="m2/s2"),
        )
        coll.collect(
            elapsed_day=5.0, day=5.0, state=state,
            q_v=jnp.full(s3, 0.003), q_c=jnp.full(s3, 2.0e-4),
            q_r=jnp.zeros(s3), sst=jnp.full(s2, 290.0), sic=jnp.zeros(s2),
            precip_total=jnp.zeros(s2), sw_up_toa=jnp.full(s2, 100.0),
            lw_up_toa=jnp.full(s2, 240.0), sw_net_sfc=jnp.full(s2, 160.0),
            lw_net_sfc=jnp.full(s2, -60.0), sw_down_toa=jnp.full(s2, 340.0),
            T_ice=271.35, q_i=jnp.full(s3, 1.0e-4), q_s=q_s, q_g=q_g,
        )
        d = coll._spatial_monthly._data_2d
        out = {}
        for month in d.values():
            for k in ("clivi", "clwvi"):
                if k in month:
                    e = month[k]
                    out[k] = np.asarray(e[0] if isinstance(e, tuple) else e,
                                        dtype=float)
        return out

    z3 = jnp.zeros((NLAT, NLON, NLEV))
    dry = _run(z3, z3)
    # snow 5x the cloud ice, graupel 2x -- if summed, clivi would jump ~8x
    wet = _run(jnp.full((NLAT, NLON, NLEV), 5.0e-4),
               jnp.full((NLAT, NLON, NLEV), 2.0e-4))

    assert "clivi" in dry and "clwvi" in dry, "clivi/clwvi were not emitted"
    assert float(np.nanmax(dry["clivi"])) > 0.0, (
        "fixture emits zero ice — the comparison would be vacuous")
    for k in ("clivi", "clwvi"):
        np.testing.assert_array_equal(
            dry[k], wet[k],
            err_msg=f"{k} changed when radiatively-inert snow/graupel were "
                    "added; they are being summed into the ice path again")
