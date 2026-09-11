#!/usr/bin/env python
"""Rule 12 for the two ``traldf_iso`` transcription fixes: which cards move?

SECTION A -- the A33 ``e3w`` divisor (PR #1728 round 3).
SECTION B -- the stretch's TIME LEVEL (round 4): every ``e3``/``r3`` operand of
``traldf_iso`` is indexed ``Kmm`` (``traldf_iso.f90:231-232``, ``:284``,
``:292``, ``:305``, ``:823``, ``:828``), and the shared step now hands the
operator the step-entry height instead of the post-barotropic one.


The fix swaps the A33 vertical-flux divisor from the interface midpoint
``0.5*(e3t(k-1)+e3t(k))`` to NEMO's ``e3w_0(k)*(1+r3t)`` with
``e3w_0(k) = gdept_0(k) - gdept_0(k-1)`` (``traldf_iso.f90:285``, ``:831-833``,
``domzgr_substitute.h90:131``), resolved by ``nemo_iso_a33_e3w`` and shared by
BOTH halves of the explicit/implicit A33 split.

This enumerates every DINO recipe and every ocean recipe in the catalog and
reports, per card:

  reaches?   does it run the isoneutral operator whose A33 block changed
  msc        ``ln_traldf_msc`` -- with msc=F the explicit A33 coefficient is
             identically zero and the divisor cannot matter
  e3w moves  max relative change of the divisor this card would see
  raises?    whether ``nemo_e3w0_reference`` now REFUSES a coordinate that
             previously got the midpoint silently (a new hard error in a
             previously-working path is a regression, and it is reported here
             rather than discovered in a run)

A card with ``reaches=yes`` and ``e3w moves > 0`` has its trajectory changed
and must be registered.
"""
import sys

import numpy as np



def _reach(gmc, sch, ffm):
    """Does this card execute either changed block, and how many operands move?

    TWO blocks changed, and they are gated DIFFERENTLY -- a diff reviewer
    demonstrated that the first draft's single ``slope_scheme`` predicate
    misses the implicit half:

      explicit ``nemo_iso_lap_tracer_tendency_latlon_cgrid``
          reached when ``slope_scheme == 'nemo_iso_lap'``.
      implicit ``compute_isoneutral_K33_latlon``'s ``_J_vol``
          reached when ``slope_positions == 'nemo_native'`` -- which the
          reviewer showed is satisfiable with ``slope_scheme='triads'``
          (measured: max|K33(kmm) - K33(base)| = 6.73e-05 against max|K33|
          = 0.178 on a 6x7x5 partial-cell case).

    Operand count: ``nemo_qco_live`` already carried its u/v flux faces on the
    Kmm height, so four VOLUME operands move; the default ``tpoint_jacobian``
    sets ``e3u_flux = e3v_flux = e3t``, so six move.
    """
    if gmc is None:
        return False, 0, "card runs no GM/Redi at all"
    pos = getattr(gmc, "slope_positions", "mode_b")
    msc = bool(getattr(gmc, "msc_stabilize", False))
    expl = (sch == "nemo_iso_lap")
    impl = (pos == "nemo_native" and msc)
    if not (expl or impl):
        return False, 0, "does not run either changed block"
    nop = (4 if ffm == "nemo_qco_live" else 6) if expl else 0
    which = ("explicit + implicit" if (expl and impl)
             else ("explicit only" if expl else "implicit (K33) only"))
    return True, nop, f"REGISTERED: trajectory moves ({which})"

def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import jax.numpy as jnp
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        nemo_iso_a33_e3w)

    print(f"{'card':30s}{'scheme':14s}{'msc':>5s}{'reaches':>9s}"
          f"{'max rel d(e3w)':>16s}  note")
    moved, failed = [], []
    for name in sorted(dm.DINO_RECIPES):
        try:
            cfg = dm.dino_config_for_recipe(name)
            if name.startswith("nemo_dino"):
                cfg = dm.nemo_faithful_dino_config(base=cfg)
            grid = dm.dino_lat_lon_grid(cfg)
            z = dm.dino_lat_lon_vertical(grid, cfg)
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{'?':14s}{'?':>5s}{'BUILD FAIL':>9s}"
                  f"{'-':>16s}  {type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        scheme = getattr(cfg, "gm_redi_slope_scheme", "?")
        msc = scheme == "nemo_iso_lap"     # the card's msc_stabilize=True arm
        reaches = scheme == "nemo_iso_lap"
        # The card's OWN horizontal shape: nemo_e3w_0 is a 3-D mesh field,
        # so a stand-in shape would only measure a broadcast error.
        _ny, _nx = np.asarray(grid.lat).size, np.asarray(grid.lon).size
        e3t = (jnp.asarray(z.dz_ref)[None, None, :]
               * jnp.ones((_ny, _nx, 1), dtype=jnp.float64))
        jac = jnp.ones((_ny, _nx), dtype=jnp.float64)
        note = ""
        try:
            new = np.asarray(nemo_iso_a33_e3w(z, e3t, jac, jnp.float64))
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{scheme:14s}{str(msc):>5s}{str(reaches):>9s}"
                  f"{'RAISES':>16s}  {type(exc).__name__}")
            failed.append(name)
            continue
        e3t_n = np.asarray(e3t)
        old = 0.5 * (np.roll(e3t_n, 1, 2) + e3t_n)
        old[..., 0] = e3t_n[..., 0]
        rel = np.abs(new[..., 1:] - old[..., 1:]) / np.maximum(
            np.abs(new[..., 1:]), 1e-300)
        mx = float(rel.max())
        if reaches and mx > 0.0:
            moved.append(name)
            note = "REGISTERED: trajectory moves"
        elif not reaches:
            note = "does not run the changed operator"
        elif mx == 0.0:
            note = "byte-unchanged (midpoint IS e3w_0 here)"
        print(f"{name:30s}{scheme:14s}{str(msc):>5s}{str(reaches):>9s}"
              f"{mx:16.4e}  {note}")

    # --- the NEMO test-case cards (LOCK_EXCHANGE zco, OVERFLOW zps) -------
    from legoesm.ocean.fidelity import nemo_testcase_recipe as ntc
    for case in ("LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        try:
            card = ntc.build_nemo_testcase_card(case)
        except Exception as exc:                           # noqa: BLE001
            print(f"{case:30s}{'?':14s}{'?':>5s}{'BUILD FAIL':>9s}"
                  f"{'-':>16s}  {type(exc).__name__}: {exc}")
            failed.append(case)
            continue
        gmc = getattr(getattr(card, "model_config", None), "gm_redi", None)
        sch = getattr(gmc, "slope_scheme", "n/a")
        m = bool(getattr(gmc, "msc_stabilize", False))
        zc = card.recipe.z_coord if hasattr(card, "recipe") else None
        try:
            _n = np.asarray(nemo_iso_a33_e3w(
                zc, jnp.asarray(np.broadcast_to(
                    np.asarray(zc.dz_ref), (2, 2, zc.n_levels)).copy()),
                jnp.ones((2, 2), dtype=jnp.float64), jnp.float64))
            _e = np.broadcast_to(np.asarray(zc.dz_ref),
                                 (2, 2, zc.n_levels)).copy()
            _o = 0.5 * (np.roll(_e, 1, 2) + _e)
            _o[..., 0] = _e[..., 0]
            gap = float(np.abs(_n[..., 1:] - _o[..., 1:]).max())
            tag = f"{gap:16.4e}"
        except Exception as exc:                           # noqa: BLE001
            tag, gap = f"{'RAISES':>16s}", -1.0
            failed.append(case)
        note = ("does not run the changed block (ln_traldf_msc=F)"
                if not (m and sch == "nemo_iso_lap")
                else "REGISTERED: trajectory moves")
        print(f"{case:30s}{str(sch):14s}{str(m):>5s}"
              f"{str(m and sch == 'nemo_iso_lap'):>9s}{tag}  {note}")
        if m and sch == "nemo_iso_lap" and gap > 0.0:
            moved.append(case)

    # --- the enumeration itself, grep-backed rather than asserted ---------
    import subprocess
    print("\nEvery place ln_traldf_msc (GMRediConfig.msc_stabilize) is "
          "turned ON in production code -- the ONLY cards that execute the "
          "changed A33 block at all:")
    out = subprocess.run(
        ["grep", "-rn", "--include=*.py", "--include=*.yaml",
         "msc_stabilize", "packages/", "scripts/"],
        capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if "=True" in ln.replace(" ", "") or ": True" in ln:
            print("  " + ln.strip())

    # =================================================================
    # SECTION B -- the stretch's TIME LEVEL
    # =================================================================
    # The Kmm change is scoped to the ``nemo_iso_lap`` branch of
    # gm_redi_tracer_tendency_latlon, and BOTH model call sites pass it
    # unconditionally -- so "reaches" is exactly "runs nemo_iso_lap", and the
    # number of operands that move depends on the card's flux-face mode:
    #   nemo_qco_live   the u/v faces were ALREADY on the Kmm height, so the
    #                   four VOLUME operands move (e3t x2, A33 e3w, ze3w_2)
    #   tpoint_jacobian e3u_flux = e3v_flux = e3t, so SIX operands move
    print("\nSECTION B -- the stretch's TIME LEVEL (Kmm), per card")
    print(f"{'card':30s}{'scheme':14s}{'flux faces':18s}"
          f"{'reaches':>9s}{'operands':>10s}  note")
    moved_b = []
    for name in sorted(dm.DINO_RECIPES):
        try:
            cfg = dm.dino_config_for_recipe(name)
            if name.startswith("nemo_dino"):
                cfg = dm.nemo_faithful_dino_config(base=cfg)
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{'?':14s}{'?':18s}{'BUILD FAIL':>9s}"
                  f"{'-':>10s}  {type(exc).__name__}")
            failed.append(name)
            continue
        # Rule 10: the card's RESOLVED GMRediConfig, built the way the driver
        # builds it -- not the ExperimentConfig's string and not a getattr
        # default.  An earlier draft of this section read the defaults and
        # reported "tpoint_jacobian" for a card that selects nemo_qco_live.
        try:
            _gg = dm.dino_lat_lon_grid(cfg)
            _mc, _ = dm.dino_lat_lon_model_config(_gg, cfg)
            gmc = _mc.gm_redi
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{'?':14s}{'?':18s}{'CFG FAIL':>9s}"
                  f"{'-':>10s}  {type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        if gmc is None:
            print(f"{name:30s}{'(no gm_redi)':14s}{'-':18s}{'False':>9s}"
                  f"{0:>10d}  card runs no GM/Redi at all")
            continue
        sch = gmc.slope_scheme
        ffm = gmc.redi_flux_face_thickness_evaluation
        reaches, nop, note = _reach(gmc, sch, ffm)
        if reaches:
            moved_b.append(name)
        print(f"{name:30s}{str(sch):14s}{str(ffm):18s}{str(reaches):>9s}"
              f"{nop:>10d}  {note}")
    for case in ("LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        try:
            card = ntc.build_nemo_testcase_card(case)
        except Exception:                                  # noqa: BLE001
            continue
        gmc = getattr(getattr(card, "model_config", None), "gm_redi", None)
        sch = "n/a" if gmc is None else gmc.slope_scheme
        ffm = ("-" if gmc is None
               else gmc.redi_flux_face_thickness_evaluation)
        reaches, nop, note = _reach(gmc, sch, ffm)
        if reaches:
            moved_b.append(case)
        print(f"{case:30s}{str(sch):14s}{str(ffm):18s}{str(reaches):>9s}"
              f"{nop:>10d}  {note}")

    # --- the NEMO GYRE card and the OMIP driver: a diff reviewer showed the
    # --- first draft of this section enumerated the DINO recipes ONLY, so a
    # --- card selecting nemo_iso_lap elsewhere would have been invisible.
    try:
        from legoesm.ocean.fidelity import nemo_recipe as nrc
        _gy = getattr(nrc, "_NEMO_GYRE_CARD_CONFIG", None)
        if _gy is not None:
            _lo = getattr(_gy, "lateral_operator", None)
            # Rule 10: linear_free_surface is on the card's Z-COORDINATE, not
            # on its recipe config (nemo_recipe.py:872-875 builds the z_coord
            # then ``._replace(linear_free_surface=True)``).  The first draft
            # of this row read it off the recipe, got False, and printed
            # "trajectory moves" for a card that cannot move.
            _bld = getattr(nrc, "build_nemo_gyre_card", None)
            _zc = None
            if _bld is not None:
                try:
                    _zc = getattr(_bld(), "z_coord", None)
                except Exception:                              # noqa: BLE001
                    _zc = None
            if _zc is None:
                _src = open("packages/ocean/legoesm/ocean/fidelity/"
                            "nemo_recipe.py").read()
                _lfs = "_replace(linear_free_surface=True)" in _src
            else:
                _lfs = bool(getattr(_zc, "linear_free_surface", False))
            print(f"{'NEMO GYRE card':30s}{str(_lo):14s}"
                  f"{'-':18s}{str(_lo == 'nemo_iso_lap' and not _lfs):>9s}"
                  f"{0 if _lfs else 4:>10d}  "
                  + ("selects nemo_iso_lap but sets linear_free_surface "
                     "(key_linssh): compute_ocean_jacobian short-circuits "
                     "(vertical.py:1443) so BOTH time levels are the eta=0 "
                     "reference and the change is inert BY CONSTRUCTION, not "
                     "by luck" if _lfs else "REGISTERED: trajectory moves"))
            if _lo == "nemo_iso_lap" and not _lfs:
                moved_b.append("NEMO GYRE card")
    except Exception as exc:                                   # noqa: BLE001
        print(f"{'NEMO GYRE card':30s}{'?':14s}{'-':18s}{'CFG FAIL':>9s}"
              f"{'-':>10s}  {type(exc).__name__}: {exc}")
        failed.append("NEMO GYRE card")
    print(f"{'run_omip_core2 --gm-slope-':30s}{'nemo_iso_lap':14s}{'-':18s}"
          f"{'True':>9s}{4:>10d}  "
          "REGISTERED: selectable from the CLI (run_omip_core2.py:1107), so "
          "any OMIP run that chooses it moves; no committed OMIP config does")

    # --- two instruments that rebuild the operator OUTSIDE the model step ---
    # They call the changed functions WITHOUT the Kmm height, so on the DINO
    # card they now sit on the post-barotropic level while the model sits on
    # the step-entry one.  Registered, not fixed here: they are comparison
    # harnesses, and moving them is a separate one-variable change.
    import subprocess as _sp
    _hits = _sp.run(
        ["grep", "-rn", "--include=*.py",
         "-e", "gm_redi_tracer_tendency_latlon(",
         "-e", "compute_isoneutral_K33_latlon(",
         "packages/", "scripts/"],
        capture_output=True, text=True).stdout.splitlines()
    print("\nEvery caller of the two changed functions, and whether it "
          "passes the Kmm height:")
    for ln in _hits:
        loc = ":".join(ln.split(":")[:2])
        f = ln.split(":")[0]
        if f.endswith("gm_redi_latlon_cgrid.py"):
            continue                       # the definitions themselves
        try:
            txt = open(f).read()
        except OSError:
            continue
        tag = ("carries redi_kmm_eta" if "redi_kmm_eta" in txt
               else "NO Kmm height -- stays on the positional level "
                    "(REGISTERED: this instrument now rebuilds a different "
                    "operator than the model runs on the DINO card)")
        print(f"  {loc:70s} {tag}")
    print("\nEvery place slope_scheme='nemo_iso_lap' is selected in "
          "production code -- the ONLY cards Section B can reach:")
    out_b = subprocess.run(
        ["grep", "-rn", "--include=*.py", "--include=*.yaml",
         "nemo_iso_lap", "packages/", "scripts/"],
        capture_output=True, text=True).stdout
    for ln in out_b.splitlines():
        if "slope_scheme" in ln:
            print("  " + ln.strip())

    # NON-VACUITY for Section B, functional rather than structural: the new
    # argument must be INERT when it names the height the operator already
    # had, and LOAD-BEARING when it names a different one.  A sweep that
    # cannot tell those apart says nothing about which cards move.
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        gm_redi_tracer_tendency_latlon as _gmt)
    _c = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    _g = dm.dino_lat_lon_grid(_c)
    _mcb, _ = dm.dino_lat_lon_model_config(_g, _c)
    _z = dm.dino_lat_lon_vertical(_g, _c)
    _st = dm.dino_lat_lon_state(_g, _z, _c)
    _kw = dict(eos=_mcb.eos, eos_linear=_mcb.eos_linear,
               mask=_st.land_mask.data, u_mask=_st.u_mask.data,
               v_mask=_st.v_mask.data, rho_0=_mcb.constants.rho_0,
               g=_mcb.constants.g, omega=_mcb.omega, dt=2700.0,
               eos_depth=getattr(_mcb, "eos_depth", "insitu"),
               native_bolus_slope_eta=_st.eta.data)
    _eta = _st.eta.data
    _args = (_st.T.data, _st.S.data, _eta, _st.H_bathy.data, _g, _z,
             _mcb.gm_redi)
    _base = np.asarray(_gmt(*_args, **_kw)[0])
    _same = np.asarray(_gmt(*_args, redi_kmm_eta=_eta, **_kw)[0])
    _other = np.asarray(_gmt(*_args, redi_kmm_eta=_eta + 1.0, **_kw)[0])
    _d_same = float(np.max(np.abs(_same - _base)))
    _d_other = float(np.max(np.abs(_other - _base)))
    print(f"\nSECTION B non-vacuity: redi_kmm_eta=eta is inert "
          f"(max|diff| = {_d_same:.3e}, bar 0.0); redi_kmm_eta=eta+1 m moves "
          f"the tendency (max|diff| = {_d_other:.3e}, must be > 0)")
    if _d_same != 0.0:
        print("  ^^ the new argument is NOT inert when it names the height "
              "the operator already had; every 'does not move' row above is "
              "unsafe")
        failed.append("section-B-inertness")
    if _d_other <= 0.0:
        print("  ^^ the new argument does NOT reach the operator; every "
              "'REGISTERED' row above is unsupported")
        failed.append("section-B-liveness")
    print(f"\nSection B: {len(moved_b)} card(s) move: {moved_b}")

    # =================================================================
    # SECTION C -- WHICH TIME LEVEL the Nbb dissipative pass hands over
    # =================================================================
    # PR #1728 round 5.  ``_leapfrog_step`` runs the dissipative operators on
    # a state whose WHOLE geometry is substituted to the before level, but
    # NEMO moves only the TRACER (traldf_iso.f90:184-205) and the before
    # DENSITY (stpmlf.f90:201 -> eosbn2.f90:362 with Knn=Nbb) there; every
    # sea-level stretch inside ldf_slp (ldfslp.f90:177-345) and traldf_iso
    # (traldf_iso.f90:231-828) stays Kmm.  The step now hands that pass the
    # step-entry height through ``_step_impl(_kmm_geometry_eta=...)``.
    #
    # A card can only be touched if it RUNS that pass, so the predicate is the
    # resolved outer integrator, printed rather than assumed (Rule 10):
    #   leapfrog       builds the substituted Nbb pass -> REGISTERED
    #   nemo_mlf       ONE pass at Nnn, Kbb tracers handed over via _ldf_state
    #                  -> already on NEMO's split, cannot move
    #   anything else  no Nbb pass at all -> cannot move
    print("\nSECTION C -- the Nbb pass's geometry time level, per card")
    print(f"{'card':30s}{'integrator':16s}{'nbb pass':>10s}"
          f"{'reaches':>9s}  note")
    moved_c, seen_int = [], {}
    for name in sorted(dm.DINO_RECIPES):
        try:
            cfg = dm.dino_config_for_recipe(name)
            if name.startswith("nemo_dino"):
                cfg = dm.nemo_faithful_dino_config(base=cfg)
            grid = dm.dino_lat_lon_grid(cfg)
            mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{'?':16s}{'BUILD FAIL':>10s}{'-':>9s}  "
                  f"{type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        integ = getattr(mc, "outer_integrator", "forward_euler")
        seen_int[name] = integ
        nbb = integ == "leapfrog"
        gmc = getattr(mc, "gm_redi", None)
        sch = getattr(gmc, "slope_scheme", "n/a") if gmc is not None else "n/a"
        pos = (getattr(gmc, "slope_positions", "mode_b")
               if gmc is not None else "n/a")
        reaches = nbb and (sch == "nemo_iso_lap" or pos == "nemo_native")
        if reaches:
            moved_c.append(name)
            note = "REGISTERED: trajectory moves"
        elif nbb:
            note = "runs the Nbb pass but no operand of it reads the height"
        elif integ == "nemo_mlf":
            note = "single Nnn pass (_ldf_state) -- already NEMO's split"
        else:
            note = f"no Nbb dissipative pass ({integ})"
        print(f"{name:30s}{integ:16s}{str(nbb):>10s}{str(reaches):>9s}  {note}")

    # The predicate above is a claim about the CODE, so it is measured on the
    # code: run one step of a REGISTERED card and of a non-registered card,
    # spying on what ``_step_impl`` is actually handed.  A non-registered card
    # that is handed a height would move silently; a registered card that is
    # handed None would make every REGISTERED row above a fiction.
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    real_impl = LatLonCGridOceanModel._step_impl
    print("\n  spy: what _step_impl is handed, per pass, on one real step")
    for probe in ("nemo_dino_kamm_mlf", "legoesm_default"):
        if probe not in dm.DINO_RECIPES:
            print(f"    {probe:22s} NOT A RECIPE -- the spy proves nothing")
            failed.append(f"section-C-{probe}")
            continue
        seen = []

        def _spy(self, state, dt, *a, **kw):
            seen.append(kw.get("_kmm_geometry_eta"))
            return real_impl(self, state, dt, *a, **kw)

        try:
            cfg = dm.dino_config_for_recipe(probe)
            if probe.startswith("nemo_dino"):
                cfg = dm.nemo_faithful_dino_config(base=cfg)
            grid = dm.dino_lat_lon_grid(cfg)
            z = dm.dino_lat_lon_vertical(grid, cfg)
            mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
            model = LatLonCGridOceanModel(grid, z, mc)
            st = dm.dino_lat_lon_state(grid, z, cfg)
            # The before level MUST differ from the now level, or this spy
            # cannot tell the two apart and "handed a height" is satisfied by
            # handing the WRONG one -- a diff reviewer demonstrated exactly
            # that, wiring the keyword to eta_before and keeping the row at
            # AGREES.  From rest eta == eta_before == 0, so the whole state's
            # before level is seeded here with a displaced sea surface.
            _eta_b = (jnp.asarray(st.eta.data) - 0.37
                      * jnp.asarray(st.land_mask.data))
            st = st._replace(
                eta_before=st.eta.replace(data=_eta_b),
                u_before=st.u, v_before=st.v, T_before=st.T, S_before=st.S)
            LatLonCGridOceanModel._step_impl = _spy
            with jax.disable_jit():
                model.step(st, dt=float(getattr(cfg, "dt", 2700.0)))
        except Exception as exc:                       # noqa: BLE001
            print(f"    {probe:22s} STEP FAILED {type(exc).__name__}: {exc}")
            failed.append(f"section-C-{probe}")
            continue
        finally:
            LatLonCGridOceanModel._step_impl = real_impl
        handed = [("None" if v is None else "height") for v in seen]
        want = (probe in moved_c)
        got = any(v is not None for v in seen)
        ok = (want == got)
        # WHICH height, not merely SOME height.  ``_kmm_eta`` is the operand
        # this whole round is about, so the spy compares the handed array
        # against BOTH candidates and refuses to pass on the wrong one.
        _now = np.asarray(st.eta.data)
        _bef = np.asarray(_eta_b)
        _sep = float(np.abs(_now - _bef).max())
        which = []
        for v in seen:
            if v is None:
                which.append("None")
            elif np.array_equal(np.asarray(v), _now):
                which.append("NOW")
            elif np.array_equal(np.asarray(v), _bef):
                which.append("BEFORE")
            else:
                which.append("OTHER")
        if _sep <= 0.0:
            print("    the two candidate heights are identical, so the "
                  "NOW/BEFORE column below proves nothing")
            failed.append("section-C-degenerate-seed")
        if "BEFORE" in which or "OTHER" in which:
            ok = False
        print(f"    {probe:22s} passes={handed}  which={which}  "
              f"|now-before|={_sep:.3f} m  registered={want}  "
              f"handed-a-height={got}  {'AGREES' if ok else 'CONTRADICTS'}")
        if not ok:
            failed.append(f"section-C-{probe}")

    print(f"\nSection C: {len(moved_c)} card(s) move: {moved_c}")

    print(f"\n{len(moved)} card(s) move: {moved}")
    if failed:
        print(f"{len(failed)} card(s) could not be scored: {failed}")
    # A sweep that can never report a move is not a sweep.  Plant: hand the
    # resolver a coordinate with the mesh reference switched off and require
    # the reported move to collapse to zero.
    from legoesm.ocean.experiments import dino as _dm
    _c = _dm.nemo_faithful_dino_config(
        base=_dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    _g = _dm.dino_lat_lon_grid(_c)
    _z = _dm.dino_lat_lon_vertical(_g, _c)._replace(
        nemo_e3w_mesh_reference=False, t_depth_ref=None)
    _ny, _nx = np.asarray(_g.lat).size, np.asarray(_g.lon).size
    _e3t = (jnp.asarray(_z.dz_ref)[None, None, :]
            * jnp.ones((_ny, _nx, 1), dtype=jnp.float64))
    _new = np.asarray(nemo_iso_a33_e3w(_z, _e3t, jnp.ones((_ny, _nx),
                                                          jnp.float64),
                                       jnp.float64))
    _e = np.asarray(_e3t)
    _old = 0.5 * (np.roll(_e, 1, 2) + _e)
    _old[..., 0] = _e[..., 0]
    _pm = float(np.abs(_new - _old).max())
    print(f"PLANT: with the mesh reference switched off the resolver returns "
          f"the midpoint, max|new - old| = {_pm:.3e} (bar 0.0)")
    if _pm != 0.0:
        print("  ^^ the midpoint arm is not the midpoint; the sweep's zero "
              "rows prove nothing")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
