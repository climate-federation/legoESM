"""Per-frame LES statistics + column budgets shared by every reference emitter.

One implementation for the dry suite driver (``run_les_suite.py``: CBL, SBL) and
the moist case drivers (``run_bomex_les.py``, ``run_dycoms_les.py``,
``run_rico_les.py``), so every artifact carries the same statistics computed the
same way:

* :func:`make_frame_fields_fn` — device fields for one snapshot: the eddy
  viscosity the core applied (incl. Lilly suppression), the applied surface BC,
  the SGS momentum stress on faces, the kinematic pressure and its gradients.
* :func:`frame_statistics` — horizontal-mean truth profiles + turbulence
  statistics (``bridge.TURBULENCE_STAT_FIELDS``) and, for a moist state, the
  moisture channels (``qt``, ``wqt_*``, ``qc``, ``bridge.MOIST_STAT_FIELDS``).
  For a moist state the thermodynamic variable is the liquid-water potential
  temperature θ_l (the SCM's conserved variable), and its SGS flux is built from
  the face fluxes the core applied to θ and q_c: ``J_θ − (L_v/(c_pd Π_f)) J_qc``.
  Because Π varies with height, the θ_l SGS tendency the core applied,
  ``R_θ − (L_v/(c_pd Π_c)) R_qc``, equals minus the divergence of that flux PLUS
  a variable-Exner source that is not a flux and is not emitted (see
  :func:`theta_l_sgs_faces`). A gradient flux ``−K ∂θ_l/∂z`` would instead invent
  transport where the core applies none (uniform θ and q_c, varying Π).
* :class:`FrameRecorder` — appends frames and stacks them for
  ``emit.build_reference_artifact``.
* :func:`column_contents` / :func:`budget_report` — column-integrated planar
  means and the RULE-0 closure check ``Δcolumn − Σ sources``.

Conventions: heights increase with index; fluxes positive upward; kinematic
units (heat K m/s, momentum m²/s², water (kg/kg) m/s); the moist budget is in
the transport norm ``∫q dz`` the Boussinesq core conserves.
"""
from __future__ import annotations

import numpy as np
from legoesm.atmosphere.physics._shared import exner_function

from legoesm import constants

from .emit import (
    column_share,
    covariance,
    faces_to_centres,
    horizontal_mean,
    resolved_vertical_flux,
    sgs_flux_from_face_stress,
    sgs_vertical_scalar_flux_faces,
    third_moment,
)
from .scm_coupling import liquid_water_theta

# Cloudy-cell threshold on q_c [kg/kg]: the convention ``les_record`` and the
# moist drivers' ``moist_profiles`` already use for cloud fraction.
CLOUD_QC_THRESHOLD_KG_KG = 1.0e-5

# Tracer slots carried into q_t = q_v + q_c + q_r (the same species
# ``les_record`` and the SCM scorer ``scm_final_moist_on`` sum). The column water
# budget spans every water slot (``N_WATER_SLOTS``); the two agree only while the
# ice slots are empty (true for Kessler).
_QT_SLOTS = (0, 1, 2)


def make_frame_fields_fn(g, u_geo, f_cor):
    """Jitted per-frame device field set ``fields(st, sfc_theta_flux, sfc_qv_flux,
    t_sfc=None, q_sfc=None, u_geo=None) -> dict``.

    ``sfc_theta_flux`` / ``sfc_qv_flux`` are the surface fluxes the driver passes
    the core at this state (prescribed, or interactive bulk); ``t_sfc`` a
    prescribed surface temperature (MOST BC) or ``None``; ``q_sfc`` the prescribed
    surface-reference humidity of that MOST layer (its moisture flux replaces
    ``sfc_qv_flux``); ``u_geo`` the geostrophic wind at this time when it varies
    (default: the one given here). ``nu_t`` is the
    viscosity ``rhs`` applies (``effective_eddy_viscosity``: Lilly suppression
    included when ``cfg.sgs_buoyancy``), so the SGS fluxes rebuilt from it are the
    ones the model integrated."""
    import jax
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl

    @jax.jit
    def _fields(st, sfc_theta_flux, sfc_qv_flux, t_sfc=None, q_sfc=None, ug=None):
        ug = u_geo if ug is None else ug
        sb = sl.surface_boundary(st.u, st.v, st.theta, g, sfc_theta_flux, t_sfc,
                                 q_sfc=q_sfc, tracers=st.tracers)
        cd_surf, q = sb[0], sb[1]
        if q_sfc is not None:
            sfc_qv_flux = sb[2]
        nu_t = sl.effective_eddy_viscosity(st.u, st.v, st.w, g, theta=st.theta,
                                           tracers=st.tracers)
        tau13, tau23, spd1, cd = sl.sgs_vertical_momentum_flux_faces(
            st.u, st.v, st.w, nu_t, g, cd_surf=cd_surf)
        p = sl.kinematic_pressure(st, g, ug, f_cor, sfc_theta_flux=sfc_theta_flux,
                                  t_sfc=t_sfc, sfc_qv_flux=sfc_qv_flux, q_sfc=q_sfc)
        dpdz = sl.f2c(jnp.pad(sl.ddz_c2f(p, g.dz), ((0, 0), (0, 0), (1, 1))))
        floor = jnp.full_like(nu_t, g.cfg.nu_floor)
        fl13, fl23, _s, _c = sl.sgs_vertical_momentum_flux_faces(
            st.u, st.v, st.w, floor, g, cd_surf=cd_surf)
        fl13 = fl13.at[..., 0].set(0.0)   # the wall stress is not the floor's
        fl23 = fl23.at[..., 0].set(0.0)
        return dict(
            q=jnp.asarray(q, st.u.dtype),
            lid=jnp.asarray(sl.lid_heat_flux(st.theta, nu_t, g), st.u.dtype),
            qv=jnp.asarray(sfc_qv_flux, st.u.dtype),
            nu_t=nu_t, tau13=tau13, tau23=tau23,
            tau_x_s=-cd * spd1 * jnp.mean(st.u[..., 0]),
            tau_y_s=-cd * spd1 * jnp.mean(st.v[..., 0]),
            p=p, dpdx=sl.ddx(p, g), dpdy=sl.ddy(p, g), dpdz=dpdz,
            dudx=sl.ddx(st.u, g), dvdy=sl.ddy(st.v, g),
            dwdz=sl.ddz_f2c(st.w, g.dz), wc=sl.f2c(st.w),
            fl13=fl13, fl23=fl23)
    return _fields


def theta_l_sgs_faces(theta, q_c, nu, dz, pr_sgs, q_surface, exner_f, q_top=0.0):
    """θ_l SGS flux on the ``nz+1`` faces from the fluxes the core applied: θ and
    q_c are diffused separately with the same diffusivity, so
    ``J_θl = J_θ − (L_v/(c_pd Π_f)) J_qc`` (q_c has no surface flux).

    ``−∂J_θl/∂z`` differs from the applied θ_l tendency
    ``R_θ − (L_v/(c_pd Π_c)) R_qc`` by the source
    ``(L_v/c_pd)/dz · [J_qc,k+1 (1/Π_c,k − 1/Π_f,k+1) − J_qc,k (1/Π_c,k − 1/Π_f,k)]``
    (the q_c flux crossing a level of different Π), which no flux can carry."""
    j_th = sgs_vertical_scalar_flux_faces(theta, nu, dz, pr_sgs=pr_sgs,
                                          surface_flux=q_surface, top_flux=q_top)
    j_qc = sgs_vertical_scalar_flux_faces(q_c, nu, dz, pr_sgs=pr_sgs)
    return j_th - (constants.L_v / constants.c_pd) * j_qc / np.asarray(exner_f)  # latent-ok: liquid_water_theta's constant L_v


def frame_statistics(st, g, fields, pr_sgs, *, exner_c=None, exner_f=None):
    """Horizontal-mean truth profiles + turbulence statistics for one snapshot.

    Returns ``(profiles, surface)``: ``profiles`` maps artifact field names to
    ``(nz,)`` arrays, ``surface`` the applied surface fluxes / stresses at this
    state. A moist state (``st.tracers`` set) needs the reference Exner at centres
    (``exner_c``) and faces (``exner_f``): its thermodynamic channel is θ_l."""
    a = {k: np.asarray(v) for k, v in fields.items()}
    u, v, wc = np.asarray(st.u), np.asarray(st.v), a["wc"]
    wf = np.asarray(st.w)
    dz = float(g.dz)
    q = float(a["q"])
    # the lid face: heat entering from above (cfg.theta_top_gradient) is a downward
    # turbulent flux; the background floor's share uses the floor viscosity
    q_top = -float(a["lid"])
    q_top_floor = -g.cfg.nu_floor / pr_sgs * g.cfg.theta_top_gradient
    moist = st.tracers is not None and st.tracers.shape[-1] > 0
    theta = np.asarray(st.theta)
    floor = np.full_like(theta, g.cfg.nu_floor)
    out = {}
    if moist:
        if exner_c is None or exner_f is None:
            raise ValueError("a moist frame needs exner_c and exner_f (θ_l)")
        tr = np.asarray(st.tracers)
        q_c = tr[..., 1] if tr.shape[-1] > 1 else np.zeros_like(theta)
        th = np.asarray(liquid_water_theta(
            theta, q_c, np.asarray(exner_c)[None, None, :]))
        wtheta_sgs = faces_to_centres(theta_l_sgs_faces(
            theta, q_c, a["nu_t"], dz, pr_sgs, q, exner_f, q_top=q_top))
        wtheta_floor = faces_to_centres(theta_l_sgs_faces(
            theta, q_c, floor, dz, pr_sgs, 0.0, exner_f, q_top=q_top_floor))
        qt3 = sum(tr[..., k] for k in _QT_SLOTS if k < tr.shape[-1])
        qv_flux = float(a["qv"])
        wqt_sgs_f = sum(
            sgs_vertical_scalar_flux_faces(tr[..., k], a["nu_t"], dz, pr_sgs=pr_sgs,
                                           surface_flux=(qv_flux if k == 0 else 0.0))
            for k in _QT_SLOTS if k < tr.shape[-1])
        out.update(
            qt=horizontal_mean(qt3),
            wqt_resolved=resolved_vertical_flux(wc, qt3),
            wqt_sgs=faces_to_centres(wqt_sgs_f),
            qc=horizontal_mean(q_c),
            qr=(horizontal_mean(tr[..., 2]) if tr.shape[-1] > 2
                else np.zeros(theta.shape[-1])),
            cloud_frac=(q_c > CLOUD_QC_THRESHOLD_KG_KG).mean(axis=(0, 1)),
        )
    else:
        th = theta
        wtheta_sgs = faces_to_centres(sgs_vertical_scalar_flux_faces(
            th, a["nu_t"], dz, pr_sgs=pr_sgs, surface_flux=q, top_flux=q_top))
        wtheta_floor = faces_to_centres(sgs_vertical_scalar_flux_faces(
            th, floor, dz, pr_sgs=pr_sgs, top_flux=q_top_floor))
    ww_f = covariance(wf, wf)                 # faces (nz+1) via the same reducer
    out.update(
        theta=horizontal_mean(th), u=horizontal_mean(u), v=horizontal_mean(v),
        wtheta_resolved=resolved_vertical_flux(wc, th),
        wtheta_sgs=wtheta_sgs,
        uw_resolved=covariance(u, wc), vw_resolved=covariance(v, wc),
        uw_sgs=sgs_flux_from_face_stress(a["tau13"]),
        vw_sgs=sgs_flux_from_face_stress(a["tau23"]),
        uu=covariance(u, u), vv=covariance(v, v),
        ww=0.5 * (ww_f[:-1] + ww_f[1:]), thth=covariance(th, th),
        wwtheta=third_moment(wc, wc, th),
        wp=covariance(wc, a["p"]), theta_dpdz=covariance(th, a["dpdz"]),
        pstrain_11=covariance(a["p"], a["dudx"]),
        pstrain_22=covariance(a["p"], a["dvdy"]),
        pstrain_33=covariance(a["p"], a["dwdz"]),
        pres_uw=covariance(u, a["dpdz"]) + covariance(wc, a["dpdx"]),
        pres_vw=covariance(v, a["dpdz"]) + covariance(wc, a["dpdy"]),
        wtheta_floor=wtheta_floor,
        uw_floor=sgs_flux_from_face_stress(a["fl13"]),
        vw_floor=sgs_flux_from_face_stress(a["fl23"]),
    )
    out["tke_resolved"] = 0.5 * (out["uu"] + out["vv"] + out["ww"])
    surface = dict(w_theta_s=q, tau_x_s=float(a["tau_x_s"]),
                   tau_y_s=float(a["tau_y_s"]))
    if moist:
        surface["w_qv_s"] = float(a["qv"])
    return out, surface


def floor_shares(prof):
    """Column share of the background-viscosity floor in the total (resolved+SGS)
    heat and momentum flux profiles."""
    return (
        column_share(prof["wtheta_floor"], prof["wtheta_resolved"] + prof["wtheta_sgs"]),
        column_share(prof["uw_floor"], prof["uw_resolved"] + prof["uw_sgs"]),
        column_share(prof["vw_floor"], prof["vw_resolved"] + prof["vw_sgs"]),
    )


class FrameRecorder:
    """Collects per-frame profiles, applied surface series and cumulative surface
    integrals, then stacks them for ``emit.build_reference_artifact``.

    ``record`` is called with the snapshot time, the state, the device fields
    (:func:`make_frame_fields_fn`) and the running time integrals of the surface
    terms the step applied, plus any other per-frame scalar or profile series (``series``:
    artifact field name → float)."""

    # profile channels passed to the artifact constructor by keyword, not via **stats
    _CORE = ("theta", "u", "v", "wtheta_resolved", "wtheta_sgs",
             "qt", "wqt_resolved", "wqt_sgs", "qc")

    def __init__(self, g, pr_sgs, *, exner_c=None, exner_f=None):
        self.g = g
        self.pr_sgs = pr_sgs
        self.exner_c = exner_c
        self.exner_f = exner_f
        self.profs: dict[str, list] = {}
        self.surface: dict[str, list] = {}
        self.series: dict[str, list] = {}
        self.times: list[float] = []

    def record(self, t, st, fields, series=None):
        prof, srf = frame_statistics(st, self.g, fields, self.pr_sgs,
                                     exner_c=self.exner_c, exner_f=self.exner_f)
        for k, val in prof.items():
            self.profs.setdefault(k, []).append(val)
        for k, val in srf.items():
            self.surface.setdefault(k, []).append(val)
        for k, val in (series or {}).items():
            self.series.setdefault(k, []).append(np.asarray(val, dtype=np.float64))
        self.times.append(float(t))
        return prof, srf

    def artifact_kwargs(self):
        """Keyword arguments for ``emit.build_reference_artifact``: the stacked
        truth/flux profiles, the applied surface series, their integrals and the
        remaining statistics."""
        stack = {k: np.stack(v) for k, v in self.profs.items()}
        kw = {k: stack.pop(k) for k in self._CORE if k in stack}
        kw.update(stack)
        kw["times_s"] = np.asarray(self.times)
        kw["z"] = np.asarray(self.g.z_c)
        kw.update({k: np.asarray(v) for k, v in self.surface.items()})
        kw.update({k: np.asarray(v) for k, v in self.series.items()})
        return kw


def column_contents(state, g, ref=None):
    """Device vector of column-integrated planar means ``∫⟨φ⟩dz`` for
    ``(θ, u, v, water)`` — water = the summed water-species slots (0 when dry).
    Differences of this vector across an operator are that operator's exact
    contribution to the column budgets.

    With the anelastic reference state ``ref`` (moist drivers) two density-weighted
    invariants follow: total water mass ``∫ρ q_w dz`` [kg/m²] (changed by the
    microphysics only through surface precipitation, by positivity not at all) and
    moist enthalpy ``∫ρ (c_pd Π θ + L_v q_v − L_f q_ice) dz`` [J/m²] (latent
    heating ``c_pd Π Δθ`` balances ``−L_v Δq_v`` and the fusion heat of ice;
    precipitating liquid carries none; q_ice = slots 3-5)."""
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl

    def col(f):
        return g.dz * jnp.sum(f.mean((0, 1)))

    moist = state.tracers is not None and state.tracers.shape[-1] > 0
    qw = jnp.sum(state.tracers[..., :sl.N_WATER_SLOTS], axis=-1) if moist else None
    water = col(qw) if moist else jnp.zeros((), state.u.dtype)
    cols = [col(state.theta), col(state.u), col(state.v), water]
    if ref is not None:
        rho, exner = jnp.asarray(ref.rho_c), jnp.asarray(ref.exner_c)
        cols.append(col(rho * qw))
        ice = jnp.sum(state.tracers[..., 3:sl.N_WATER_SLOTS], axis=-1)
        cols.append(col(rho * (constants.c_pd * exner * state.theta
                               + constants.L_v * state.tracers[..., 0]  # latent-ok: the microphysics adapter's constant L_v
                               - constants.L_f * ice)))  # latent-ok: frozen condensate holds −L_f
    return jnp.stack(cols)


def budget_report(label, d_column, sources, column):
    """Print and return ``Δcolumn − Σ sources`` for one quantity.

    ``sources`` maps term name → its time-integrated column contribution;
    ``column`` is the final column content (round-off scale)."""
    total = float(sum(sources.values()))
    residual = float(d_column) - total
    terms = " ".join(f"{k}={float(v):+.6e}" for k, v in sources.items())
    print(f"  budget {label}: d(column)={float(d_column):+.6e} {terms} "
          f"residual={residual:+.3e} column={float(column):+.6e}")
    return residual


# Composite per-step budget of a moist case driver: the dycore step's own terms
# (``spectral_les_plane.BUDGET_TERMS``) followed by the split operators the driver
# applies around it, each the exact change of the column contents it caused.
SPLIT_TERMS = ("ls_theta", "ls_water", "rad_theta", "micro_theta", "micro_water",
               "positivity_water", "sponge_u", "sponge_v", "precip_mass",
               "micro_water_rho", "micro_enthalpy_rho", "positivity_water_rho",
               "positivity_theta", "positivity_enthalpy_rho")


def moist_step_budget(dycore, *, ls=None, rad=None, micro=None, positivity=None,
                      sponge=None, precip_mass=0.0):
    """Device vector in ``BUDGET_TERMS + SPLIT_TERMS`` order for one driver step.

    ``dycore`` is ``step(return_budget=True)``'s vector; each other argument is a
    ``column_contents(state, g, ref)`` difference across that operator (``None`` =
    not applied); ``precip_mass`` the domain-mean surface precipitation removed in
    the step [kg/m²]."""
    import jax.numpy as jnp
    z = jnp.zeros(6, dycore.dtype)
    ls, rad, micro, positivity, sponge = (
        z if d is None else d for d in (ls, rad, micro, positivity, sponge))
    split = jnp.stack([ls[0], ls[3], rad[0], micro[0], micro[3], positivity[3],
                       sponge[1], sponge[2], jnp.asarray(precip_mass, dycore.dtype),
                       micro[4], micro[5], positivity[4], positivity[0],
                       positivity[5]])
    return jnp.concatenate([dycore, split.astype(dycore.dtype)])


class MoistRunEmitter:
    """Live reference emission for a moist LES case driver: frame statistics on the
    driver's record cadence, the running composite budget, its RULE-0 closure
    report, and the artifact.

    The driver builds one per run, calls :meth:`start` with the initial state,
    :meth:`add` with each step's :func:`moist_step_budget` vector, :meth:`record`
    at each output time, then :meth:`close_budgets` and :meth:`artifact`."""

    def __init__(self, g, ref, u_geo, f_cor):
        import jax.numpy as jnp
        from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl
        self.g = g
        self.terms = tuple(sl.BUDGET_TERMS) + SPLIT_TERMS
        self.fields_fn = make_frame_fields_fn(g, u_geo, f_cor)
        self.rec = FrameRecorder(g, g.cfg.pr_sgs, exner_c=np.asarray(ref.exner_c),
                                 exner_f=np.asarray(exner_function(np.asarray(ref.p_f))))
        self.ref = ref
        self.acc = jnp.zeros(len(self.terms), g.z_c.dtype)
        self.col0 = None

    def start(self, state):
        self.col0 = np.asarray(column_contents(state, self.g, self.ref))

    def add(self, step_budget):
        self.acc = self.acc + step_budget

    def state_dict(self):
        """Everything accumulated so far (budgets, recorded frames), for a
        checkpoint; :meth:`load_state_dict` restores it exactly."""
        return {"acc": self.acc, "col0": self.col0, "profs": self.rec.profs,
                "surface": self.rec.surface, "series": self.rec.series,
                "times": self.rec.times}

    def load_state_dict(self, d):
        import jax.numpy as jnp
        self.acc = jnp.asarray(d["acc"])
        self.col0 = np.asarray(d["col0"])
        self.rec.profs = {k: list(v) for k, v in d["profs"].items()}
        self.rec.surface = {k: list(v) for k, v in d["surface"].items()}
        self.rec.series = {k: list(v) for k, v in d["series"].items()}
        self.rec.times = [float(t) for t in d["times"]]

    def _named(self):
        return dict(zip(self.terms, np.asarray(self.acc)))

    def record(self, t, state, sfc_theta_flux, sfc_qv_flux, precip_rate, series=None,
               *, t_sfc=None, q_sfc=None, u_geo=None):
        """One output frame: statistics of ``state`` with the surface fluxes the
        driver applies at it, the domain-mean surface precipitation rate of the
        most recent microphysics call [kg/m²/s] (``bridge`` ``precip_s``), and the
        running exact integrals of the applied surface fluxes and of the
        precipitation. ``t_sfc``/``q_sfc``/``u_geo``: a prescribed-surface MOST BC
        and a time-varying geostrophic wind, passed as the step applies them."""
        n = self._named()
        prof, srf = self.rec.record(
            t, state, self.fields_fn(state, sfc_theta_flux, sfc_qv_flux, t_sfc, q_sfc,
                                     u_geo),
            series={"w_theta_s_cum": n["sfc_heat_flux"], "tau_x_s_cum": n["sfc_stress_x"],
                  "tau_y_s_cum": n["sfc_stress_y"], "w_qv_s_cum": n["sfc_qv_flux"],
                  "precip_s_cum": n["precip_mass"],
                  "precip_s": float(precip_rate), **(series or {})})
        return prof, srf

    def close_budgets(self, state):
        """Print and return the column-budget residuals ``Δcolumn − Σ terms`` for
        θ, u, v and total water (transport norm ``∫q dz``), plus the microphysics'
        own density-weighted closures: its water-mass change against the surface
        precipitation it removed (``micro_water``) and its moist-enthalpy change
        (``micro_enthalpy``, no source); and the positivity fixer's water-mass change
        (``positivity_water``, no source: a clip that creates water fails it).

        The split-operator terms are measured column changes, so they route every
        change to a named term but cannot detect an error inside that operator;
        the independent checks are the dycore terms and the two microphysics
        closures."""
        n = self._named()
        col1 = np.asarray(column_contents(state, self.g, self.ref))
        d = col1 - self.col0
        groups = {
            "theta": ("sfc_heat_flux", "fv_theta", "ls_theta", "rad_theta", "micro_theta",
                      "positivity_theta", "lid_heat_flux"),
            "u": ("sfc_stress_x", "coriolis_u", "body_force_u", "sponge_u"),
            "v": ("sfc_stress_y", "coriolis_v", "body_force_v", "sponge_v"),
            "water": ("sfc_qv_flux", "fv_water", "ls_water", "micro_water",
                      "positivity_water"),
        }
        res = {lab: budget_report(lab, d[i], {k: n[k] for k in keys}, col1[i])
               for i, (lab, keys) in enumerate(groups.items())}
        res["micro_water"] = budget_report(
            "micro_water", n["micro_water_rho"], {"precip": -n["precip_mass"]}, col1[4])
        res["micro_enthalpy"] = budget_report(
            "micro_enthalpy", n["micro_enthalpy_rho"], {}, col1[5])
        res["positivity_water"] = budget_report(
            "positivity_water", n["positivity_water_rho"], {}, col1[4])
        res["positivity_enthalpy"] = budget_report(
            "positivity_enthalpy", n["positivity_enthalpy_rho"], {}, col1[5])
        return res

    def artifact(self, **kw):
        """``emit.build_reference_artifact`` with the recorded frames (``kw``: case
        name, sgs, forcing profiles, ``les_setup``)."""
        from .emit import build_reference_artifact
        return build_reference_artifact(prescribe="fluxes", **self.rec.artifact_kwargs(),
                                        **kw)
