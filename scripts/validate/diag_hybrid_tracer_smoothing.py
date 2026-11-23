"""Reproduce the T18 hybrid van Leer gaps and T17 condensate diagnostics.

Run on PBS develop@desched1, account UCLB0017, CPU, JAX_ENABLE_X64=1::

    python scripts/validate/diag_hybrid_tracer_smoothing.py --out <artifact-dir>

The fixture, species/decks, pressure regions and dt=112.5 s are inherited from
T17. Compare defaults (q width 1e-10 kg/kg; flux width 1e-6 Pa/s) with the
frozen hard path (both zero). Report paired budgets, reconstructed effective
faces, and vertical-only forward-Euler negatives before conservative borrowing.
These are fixture diagnostics, not production-climate statistics.
"""

import argparse
import os
from pathlib import Path
import json
import csv
import runpy
import subprocess
import hashlib


def countmin(x, mask):
    """Return the masked count and minimum; the minimum is None for an empty mask."""
    vals = x[mask]
    return int(vals.size), float(vals.min()) if vals.size else None


def main():
    if hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64())
    from legoesm.grids import vertical as v
    from legoesm.core.smooth import smooth_abs
    from legoesm.core.field import Field
    from legoesm.atmosphere.standard_atmosphere import standard_atmosphere_temperature
    from legoesm.driver.config import ExperimentConfig
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as pe
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import tracer_horizontal_advection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    OUT = parser.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    W = Path(__file__).resolve().parents[2]
    assert os.environ.get("PBS_JOBID") and jax.config.x64_enabled
    assert all(d.platform == "cpu" for d in jax.devices())
    h = runpy.run_path(str(W / "tests/unit/test_mpas_atmosphere.py"))
    coord = v.make_cam6_l32_levels()
    nlev = 32
    mesh = h["_make_mesh"](level=2)
    state = h["_add_perturbation_hydro"](h["_make_hydrostatic_state"](mesh, nlev), mesh, nlev)
    T = standard_atmosphere_temperature(mesh.latCell, coord.sigma_full)
    qv = (
        ExperimentConfig().rh_init
        * saturation_mixing_ratio(T, state.p_s.data[:, None] * coord.sigma_full)
        * coord.sigma_full**2
    )
    state = state._replace(
        T=state.T.replace(data=T),
        tracers={"q_v": Field(data=qv, name="q_v", dims=("nCells", "nlev"), units="kg/kg")},
    )
    calls = []
    real = pe.vertical_advection_hybrid_van_leer


    def spy(*a, **kw):
        calls.append((a, kw))
        return real(*a, **kw)


    pe.vertical_advection_hybrid_van_leer = spy
    smooth = pe.mpas_model_tendencies(state, mesh, coord, pe.MPASPrimitiveEquationConfig(nu_del2=1e5))
    pe.vertical_advection_hybrid_van_leer = real
    hard = pe.mpas_model_tendencies(
        state,
        mesh,
        coord,
        pe.MPASPrimitiveEquationConfig(
            nu_del2=1e5, tracer_van_leer_width=0.0, tracer_van_leer_flux_width=0.0
        ),
    )
    _, F, ps, coord = calls[0][0]
    area = np.asarray(mesh.areaCell)[:, None]


    def norm(x, mask):
        return np.sqrt(np.sum(area * np.where(mask, x * x, 0.0)))


    gap = np.asarray(smooth.tracer_tendencies["q_v"].data - hard.tracer_tendencies["q_v"].data)
    reference = [
        float(np.max(abs(gap))),
        float(norm(gap, True) / norm(np.asarray(hard.tracer_tendencies["q_v"].data), True)),
    ]
    (OUT / "full_gap.json").write_text(json.dumps(reference))
    dp = v.dp_from_hybrid(coord, ps)
    p = v.pressure_from_hybrid(coord, ps)
    ph = v.pressure_from_hybrid(coord, ps, full=False)
    dc = jnp.diff(p, axis=-1)
    dt = 112.5
    # Decks have latitude-dependent amplitudes, exact zero outside. Levels top -> bottom.
    amplitude = (1.0 + 0.25 * jnp.cos(mesh.latCell))[:, None]


    def deck(start, values):
        return (
            jnp.zeros_like(qv)
            .at[:, start : start + len(values)]
            .set(amplitude * jnp.array(values)[None, :])
        )


    qs = {
        "q_v": qv,
        "q_c": deck(23, [1e-5, 8e-5, 1e-4, 2e-5]),
        "q_i": deck(9, [1e-6, 8e-6, 1e-5, 2e-6]),
        "q_r": deck(24, [1e-7, 1e-6, 3e-7]),
        "q_s": deck(12, [1e-8, 1e-7, 3e-8]),
        "small": qv * 1e-6,
    }
    widths = [("hard", 0.0, 0.0), ("default", 1e-10, 1e-6)]


    def tendency(q, w, fw):
        return real(q, F, ps, coord, smoothing_width=w, flux_smoothing_width=fw)


    def faces(q, w, fw):
        pos, neg = v._van_leer_face_values(q, dc, dp, w)
        edge = jnp.zeros(nlev + 1, dtype=bool).at[jnp.array([1, nlev - 1])].set(True)
        pos = jnp.where(edge, jnp.concatenate([q[:, :1], q], -1), pos)
        neg = jnp.where(edge, jnp.concatenate([q, q[:, -1:]], -1), neg)
        if fw:
            af = smooth_abs(F, fw * fw) - fw
            selected = 0.5 * (pos + neg) + 0.5 * (af / jnp.where(F != 0, F, 1.0)) * (pos - neg)
        else:
            selected = jnp.where(F > 0, pos, neg)
        return pos, neg, selected


    rows = []
    controls = {"T15_full_tendency": reference}
    raw = {}
    for name, q in qs.items():
        baseline = tendency(q, 0.0, 0.0)
        hf = np.asarray(faces(q, 0.0, 0.0)[2][:, 1:-1])
        horizontal = tracer_horizontal_advection(q[..., None], state.u.data, mesh)[..., 0]
        for label, w, relative in widths:
            tend = tendency(q, w, relative)
            pos, neg, face = faces(q, w, relative)
            # Independent face-flux reconstruction validates the diagnostic, not physics.
            ft = -(F[:, 1:] * (face[:, 1:] - q) - F[:, :-1] * (face[:, :-1] - q)) / dp
            np.testing.assert_allclose(np.asarray(tend), np.asarray(ft), rtol=2e-8, atol=2e-20)
            residual = jnp.sum(dp * tend - q * jnp.diff(F, axis=-1), axis=-1)
            assert float(jnp.max(jnp.abs(residual))) < 1e-14
            weights = jnp.asarray(np.random.default_rng(1700).normal(size=q.shape))
            gradient = jax.grad(lambda x: jnp.sum(weights * tendency(x, w, relative)))(q)
            grad_ok = bool(np.isfinite(np.asarray(gradient)).all())
            raw[name + "_" + label] = np.asarray(tend)
            diff = np.asarray(tend - baseline)
            updated = np.asarray(q + dt * tend)
            full = np.asarray(q + dt * (tend + horizontal))
            for region in ("above200", "below200"):
                cm = np.asarray(p < 20000 if region == "above200" else p >= 20000)
                fm = np.asarray(ph[:, 1:-1] < 20000 if region == "above200" else ph[:, 1:-1] >= 20000)
                f = np.asarray(face[:, 1:-1])
                lo = np.minimum(np.asarray(q[:, :-1]), np.asarray(q[:, 1:]))
                hi = np.maximum(np.asarray(q[:, :-1]), np.asarray(q[:, 1:]))

                outside = ((f < lo) | (f > hi)) & fm
                outn, outmin = countmin(f, outside)
                fn, fmin = countmin(f, (f < 0) & fm)
                un, umin = countmin(updated, (updated < 0) & cm)
                fulln, fullmin = countmin(full, (full < 0) & cm)
                denom = norm(np.asarray(baseline), cm)
                hardrange = float(np.ptp(hf[fm]))
                fgap = float(np.max(abs(f - hf)[fm]))
                row = dict(
                    species=name,
                    region=region,
                    width=label,
                    q_min=float(np.asarray(q)[cm].min()),
                    q_max=float(np.asarray(q)[cm].max()),
                    max_abs_tendency_gap=float(np.max(abs(diff)[cm])),
                    relative_L2=float(norm(diff, cm) / denom)
                    if denom
                    else (0.0 if norm(diff, cm) == 0 else None),
                    outside_count=outn,
                    outside_min=outmin,
                    negative_face_count=fn,
                    negative_face_min=fmin,
                    negative_step_count=un,
                    negative_step_min=umin,
                    full_horizontal_plus_vertical_negative_count=fulln,
                    full_horizontal_plus_vertical_negative_min=fullmin,
                    face_gap=fgap,
                    hard_face_range=hardrange,
                    face_gap_over_hard_range=fgap / hardrange
                    if hardrange
                    else (0.0 if fgap == 0 else None),
                    gradient_finite=grad_ok,
                    gradient_max=float(jnp.max(abs(gradient))),
                    sample_cells=int(cm.sum()),
                    sample_faces=int(fm.sum()),
                )
                full_denom = norm(np.asarray(baseline + horizontal), cm)
                row["full_area_relative_L2"] = (
                    float(norm(diff, cm) / full_denom)
                    if full_denom
                    else (0.0 if norm(diff, cm) == 0 else None)
                )
                row["effective_max_bound_excess"] = float(
                    np.max(np.maximum(0.0, np.maximum(lo - f, f - hi))[fm])
                )
                for tag, branch in [("pos", pos), ("neg", neg)]:
                    b = np.asarray(branch[:, 1:-1])
                    mask = ((b < lo) | (b > hi)) & fm
                    row[tag + "_outside_count"], row[tag + "_outside_min"] = countmin(b, mask)
                    row[tag + "_max_bound_excess"] = float(
                        np.max(np.maximum(0.0, np.maximum(lo - b, b - hi))[fm])
                    )
                    row[tag + "_negative_count"], row[tag + "_negative_min"] = countmin(b, (b < 0) & fm)
                rows.append(row)
        print(name, "done", flush=True)
    # Positive controls for exact diagnostic comparisons; hard baseline must be bounded.
    assert all(
        r["outside_count"] == 0 and r["negative_face_count"] == 0 and r["negative_step_count"] == 0
        for r in rows
        if r["width"] == "hard"
    )
    assert np.any(np.asarray(qs["q_s"]) == 0)
    # Planted violation: negative face below adjacent zeros must be detected.
    lo = np.array([0.0, 1.0])
    hi = np.array([0.0, 2.0])
    planted = np.array([-1e-9, 3.0])
    controls["planted_outside_count"] = int(((planted < lo) | (planted > hi)).sum())
    assert controls["planted_outside_count"] == 2
    controls["diagnostic_reconstruction"] = "passed"
    with (OUT / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(
        OUT / "arrays.npz",
        F=np.asarray(F),
        p=np.asarray(p),
        ph=np.asarray(ph),
        dp=np.asarray(dp),
        area=np.asarray(area),
        **{k: np.asarray(q) for k, q in qs.items()},
        **raw,
    )
    provenance = {
        "sha": subprocess.check_output(["git", "-C", str(W), "rev-parse", "HEAD"], text=True).strip(),
        "job": os.environ["PBS_JOBID"],
        "host": os.uname().nodename,
        "jax": jax.__version__,
        "dtype": str(qv.dtype),
        "devices": str(jax.devices()),
        "dt": dt,
        "max_courant_pair": float(jnp.max(dt * (abs(F[:, :-1]) + abs(F[:, 1:])) / dp)),
        "profiles": {
            "q_c": [23, [1e-5, 8e-5, 1e-4, 2e-5]],
            "q_i": [9, [1e-6, 8e-6, 1e-5, 2e-6]],
            "q_r": [24, [1e-7, 1e-6, 3e-7]],
            "q_s": [12, [1e-8, 1e-7, 3e-8]],
            "small": "q_v * 1e-6",
            "amplitude": "1 + .25*cos(latitude)",
        },
        "controls": controls,
        "source_hashes": {
            str(path.relative_to(W)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (W / "packages").rglob("*.py")
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("COMPLETE", controls, flush=True)

    hh = runpy.run_path(str(W / "tests/unit/test_vertical_advection_hybrid_van_leer.py"))
    gaps = {}
    for case in range(4):
        cc, pp, ff, qq = hh["_limiter_case"](case)
        hard = real(qq, ff, pp, cc, smoothing_width=0.0, flux_smoothing_width=0.0)
        smooth = real(qq, ff, pp, cc)
        gaps[case] = np.max(abs(np.asarray(smooth - hard)), axis=-1).tolist()
    (OUT / "gaps.json").write_text(json.dumps(gaps, indent=2))
    print("GAPS", gaps, flush=True)
    print("FULL_GAP", reference, flush=True)

    # Aggregate the exact regional diagnostics and verify resolved production borrow.
    rows = json.loads((OUT / "metrics.json").read_text())
    rows = [r for r in rows if r["width"] == "default"]


    def summary(rs):
        def nm(key):
            vals = [r[key + "_min"] for r in rs if r[key + "_min"] is not None]
            return dict(count=sum(r[key + "_count"] for r in rs), minimum=min(vals) if vals else None)

        return dict(
            negative_faces=nm("negative_face"),
            negative_step=nm("negative_step"),
            face_error_over_hard_range=max(r["face_gap_over_hard_range"] for r in rs),
            max_tendency_gap=max(r["max_abs_tendency_gap"] for r in rs),
            max_region_vertical_relL2=max(r["relative_L2"] for r in rs),
            max_region_full_relL2=max(r["full_area_relative_L2"] for r in rs),
        )


    result = dict(
        all=summary(rows),
        species={
            n: summary([r for r in rows if r["species"] == n])
            for n in dict.fromkeys(r["species"] for r in rows)
        },
    )
    arrays = np.load(OUT / "arrays.npz")
    area = arrays["area"]
    for name, stats in result["species"].items():
        hard = arrays[name + "_hard"]
        smooth = arrays[name + "_default"]
        stats["whole_species_vertical_relL2"] = float(
            np.sqrt(np.sum(area * (smooth - hard) ** 2) / np.sum(area * hard**2))
        )
    from scripts.run.run_amip import build_arg_parser, build_config_from_args, _postprocess_args
    from legoesm.driver.run_config_yaml import load_yaml_config
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import make_cam6_l32_levels

    h = runpy.run_path(str(W / "tests/unit/test_run_amip_cli.py"))
    result["borrow"] = {}
    for deck in ("amip_production.yaml", "amip_production_latlon48.yaml"):
        parser = build_arg_parser()
        parser.set_defaults(**load_yaml_config(str(W / "config/amip" / deck), parser))
        cfg = build_config_from_args(
            _postprocess_args(parser.parse_args(h["_AMIP_DUMMY_PATHS"]), parser)
        )
        kind = "mpas" if deck == "amip_production.yaml" else "latlon"
        grid = create_grid(
            kind, 2 if kind == "mpas" else 4, **({"lloyd_iterations": 5} if kind == "mpas" else {})
        )
        model = create_atmosphere_dycore(cfg, grid, make_cam6_l32_levels(), latlon_native_grid=grid)
        result["borrow"][deck] = dict(
            driver=cfg.dycore.mpas_conservative_tracer_clamp,
            dycore=model.config.conservative_tracer_clamp,
            q_width=model.config.tracer_van_leer_width,
            flux_width=model.config.tracer_van_leer_flux_width,
            scheme=cfg.dycore.tracer_vertical_scheme,
        )
        assert cfg.dycore.mpas_conservative_tracer_clamp and model.config.conservative_tracer_clamp
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

    (OUT / "measured_source.patch").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=W, text=True)
    )
    (OUT / "probe_sha256.txt").write_text(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "\n"
    )


def paired_column_budget(m, q, dq, dm):
    """Per-column instantaneous product rule, without a denominator floor."""
    import numpy as np
    a, b = np.asarray(m) * np.asarray(dq), np.asarray(q) * np.asarray(dm)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("nonfinite model budget; no conservation verdict")
    numerator = np.abs(np.sum(a + b, axis=-1))
    denominator = np.sum(np.abs(a) + np.abs(b), axis=-1)
    relative = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                         where=denominator != 0)
    relative = np.where((denominator == 0) & (numerator != 0), np.inf, relative)
    return {"numerator": numerator.tolist(), "denominator": denominator.tolist(),
            "max_relative": float(relative.max()),
            "verdict": "CONSERVES" if np.all(relative <= 1e-12) else "LEAKS"}


def global_inventory_budget(volume, rho_old, q_old, rho_new, q_new, boundary_budget):
    """Signed domain inventory change per tracer; boundary content is positive in.

    Host-side diagnostic only. All inputs must be fp64; no implicit broadcasting,
    masking, denominator floor or conversion from fp32 is permitted.
    """
    import numpy as np

    volume, rho_old, q_old, rho_new, q_new, boundary_budget = arrays = [
        np.asarray(a) for a in (volume, rho_old, q_old, rho_new, q_new, boundary_budget)
    ]
    if any(a.dtype != np.float64 for a in arrays):
        raise ValueError("fp64 arrays required")
    if (not volume.size or rho_old.shape != volume.shape or rho_new.shape != volume.shape
            or q_old.ndim != volume.ndim + 1 or q_old.shape[:-1] != volume.shape
            or q_new.shape != q_old.shape or q_old.shape[-1] == 0
            or boundary_budget.shape != (q_old.shape[-1],)):
        raise ValueError("incompatible or empty inventory shapes")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("nonfinite inventory input")
    if any(np.any(a <= 0) for a in (volume, rho_old, rho_new)):
        raise ValueError("volume and density must be positive")
    old = (volume * rho_old)[..., None] * q_old
    new = (volume * rho_new)[..., None] * q_new
    axes = tuple(range(volume.ndim))
    initial = np.sum(old, axis=axes)
    final = np.sum(new, axis=axes)
    scale = np.sum(np.abs(old), axis=axes)
    if not (np.isfinite(initial).all() and np.isfinite(final).all()
            and np.isfinite(scale).all() and np.all(scale > 0)):
        raise ValueError("nonfinite or zero initial inventory")
    residual = final - initial - boundary_budget
    relative = np.abs(residual) / scale
    if not (np.isfinite(residual).all() and np.isfinite(relative).all()):
        raise ValueError("nonfinite inventory residual")
    return dict(initial_inventory=initial.tolist(), final_inventory=final.tolist(),
                initial_abs_inventory=scale.tolist(), boundary_budget=boundary_budget.tolist(),
                signed_residual=residual.tolist(), relative_residual=relative.tolist(),
                verdicts=np.where(relative <= 1e-12, "CONSERVES", "LEAKS").tolist())


def inventory_controls(volume, rho, q):
    """Identity, weighted internal transfer and the same amount added as a source."""
    import numpy as np

    zero = np.zeros(q.shape[-1], dtype=np.float64)
    identity = global_inventory_budget(volume, rho, q, rho, q, zero)
    mass = (np.asarray(volume) * np.asarray(rho)).reshape(-1)
    content = mass[:, None] * np.asarray(q).reshape(-1, q.shape[-1])
    transfer, source = np.array(q, copy=True), np.array(q, copy=True)
    expected, donors, receivers = [], [], []
    for tracer in range(q.shape[-1]):
        positive = np.flatnonzero(content[:, tracer] > 0)
        if positive.size < 2:
            raise ValueError("control requires two positive tracer contents")
        donor = int(positive[np.argmax(content[positive, tracer])])
        remaining = positive[positive != donor]
        receiver = int(remaining[np.argmax(content[remaining, tracer])])
        amount = 0.01 * content[donor, tracer]
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("nonfinite or zero control amount")
        if amount / identity["initial_abs_inventory"][tracer] <= 1e-12:
            raise ValueError("planted source below inventory bar")
        transfer.reshape(-1, q.shape[-1])[donor, tracer] -= amount / mass[donor]
        transfer.reshape(-1, q.shape[-1])[receiver, tracer] += amount / mass[receiver]
        source.reshape(-1, q.shape[-1])[receiver, tracer] += amount / mass[receiver]
        expected.append(float(amount))
        donors.append(donor)
        receivers.append(receiver)
    results = []
    for name, final_q, target in (("identity", q, zero), ("conservative transfer", transfer, zero),
                                   ("planted source", source, expected)):
        result = global_inventory_budget(volume, rho, q, rho, final_q, zero)
        error = np.abs(np.asarray(result["signed_residual"]) - target)
        assert np.all(error / result["initial_abs_inventory"] <= 1e-12), name
        assert result["verdicts"] == ["LEAKS" if name == "planted source" else "CONSERVES"] * q.shape[-1]
        if name == "planted source":
            assert np.all(np.asarray(result["signed_residual"]) > 0)
        result.update(control=name, expected_signed_residual=np.asarray(target).tolist(),
                      donor_indices=donors, receiver_indices=receivers, amount=expected)
        results.append(result)
    return results


def height_les_fixture(root, q):
    """Exact T24 passive LES fixture and its copied q/w profiles."""
    import jax.numpy as jnp

    model, ns, pg, hc, metric = runpy.run_path(
        str(root / "tests/unit/test_plane_nh_tracers.py"))["_setup_with_tracers"](1)
    qp = jnp.broadcast_to(q[0, 22:26], ns.tracers.data.shape[:-1])
    wp = jnp.broadcast_to(jnp.sin(jnp.linspace(0, jnp.pi, 5)).at[-1].set(0.0), ns.w.data.shape)
    ns = ns._replace(w=ns.w.replace(data=wp), tracers=ns.tracers.replace(data=qp[..., None]))
    return model, ns, pg, hc, metric


def spectral_level_quadrature_budget(area, field):
    """Signed horizontal integral at each level, using native Gaussian areas.

    Exact-zero fields are unscored identities, never a conservation verdict.
    This host-side instrument does not average levels or floor denominators.
    """
    import numpy as np

    area, field = np.asarray(area), np.asarray(field)
    if (area.dtype != np.float64 or field.dtype != np.float64
            or area.ndim != 2 or field.ndim != 3
            or field.shape[:2] != area.shape or not field.size
            or not np.isfinite(area).all() or not np.isfinite(field).all()
            or not np.all(area > 0)):
        raise ValueError("finite fp64 field(lat,lon,level) and positive native area required")
    signed = np.sum(area[..., None] * field, axis=(0, 1))
    absolute = np.sum(area[..., None] * np.abs(field), axis=(0, 1))
    relative = [None if norm == 0 else float(abs(value) / norm)
                for value, norm in zip(signed, absolute)]
    return dict(signed_integral=signed.tolist(), abs_integral=absolute.tolist(),
                relative=relative,
                verdicts=["ZERO_UNSCORED" if r is None else
                          "AT_BAR" if r <= 1e-12 else "DEBT" for r in relative])


def spectral_horizontal_quadrature_probe(grid, hc, metric, state, label, datasets):
    """Compare actual horizontal continuity with the canonical flux divergence.

    Horizontal divergence is positive out; density tendency is its negative / J.
    No acoustic step, tracer-mass closure or temporal-order claim is made here.
    """
    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.dynamics.gcm import spectral_nh as sn
    from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
    from legoesm.grids.gaussian import (
        dealiasing_mask, sh_analysis_3d, sh_synthesis_3d,
        uv_from_vordiv_3d, vordiv_from_uv_3d,
    )

    cfg = sn.SpectralNHConfig()
    tendency = sn.spectral_nh_slow_tendencies(
        state, grid, hc, metric, cfg, physics_tendency=None)
    theta_p = sh_synthesis_3d(grid, state.theta_prime_hat.data)
    rho_p = sh_synthesis_3d(grid, state.rho_prime_hat.data)
    _, rho = sanitize_theta_rho(hc.theta_ref + theta_p, hc.rho_ref + rho_p)
    u_cos, v_cos = uv_from_vordiv_3d(grid, state.vor_hat.data, state.div_hat.data)
    cos_lat = jnp.clip(grid.cos_lat[:, None, None], sn._COS_LAT_MIN, None)
    u, vv = u_cos / cos_lat, v_cos / cos_lat
    jac = metric.jacobian[..., None]
    div_hat = vordiv_from_uv_3d(grid, jac * rho * u, jac * rho * vv)[1]
    divergence = sh_synthesis_3d(grid, div_hat)
    # Match the actual caller's grid->spectral projection and final de-aliasing.
    expected_hat = sh_analysis_3d(grid, -divergence / jac)
    expected_hat = expected_hat * dealiasing_mask(grid, cfg.dealiasing_fraction)[:, None]
    expected = sh_synthesis_3d(grid, expected_hat)
    actual = sh_synthesis_3d(grid, tendency.rho_prime_hat.data)
    area = np.asarray(grid.grid_area)
    idx00 = np.flatnonzero((np.asarray(grid.ls) == 0) & (np.asarray(grid.ms) == 0))
    assert len(idx00) == 1
    for key, value in dict(area=area, rho=rho, u=u, v=vv, divergence=divergence,
                           divergence_hat=div_hat, actual_drho=actual,
                           expected_drho=expected).items():
        datasets[f"spectral_quadrature/{label}/{key}"] = np.asarray(value)
    result = dict(
        path="spectral NH/height", statistic="horizontal quadrature per level",
        fixture=label, config=cfg._asdict(), scope="horizontal continuity only; no time step",
        actual=spectral_level_quadrature_budget(area, np.asarray(jac * actual)),
        canonical=spectral_level_quadrature_budget(area, np.asarray(divergence)),
        divergence_mode00_real=np.asarray(div_hat[idx00[0]].real).tolist(),
        divergence_mode00_imag=np.asarray(div_hat[idx00[0]].imag).tolist(),
        continuity_matches=bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-12)),
    )
    if label == "census_identity":
        result["accepted"] = (result["continuity_matches"] and all(
            verdict == "ZERO_UNSCORED" for key in ("actual", "canonical")
            for verdict in result[key]["verdicts"]))
        return result

    amplitude = np.max(np.abs(np.asarray(divergence)), axis=(0, 1))
    assert np.all(np.isfinite(amplitude) & (amplitude > 0)), "vacuous supplemental divergence"
    offset = 0.01 * amplitude
    planted = np.asarray(divergence) + offset[None, None, :]
    datasets[f"spectral_quadrature/{label}/planted_divergence"] = planted
    result["planted"] = spectral_level_quadrature_budget(area, planted)
    expected_added = offset * np.sum(area)
    actual_added = (np.asarray(result["planted"]["signed_integral"])
                    - np.asarray(result["canonical"]["signed_integral"]))
    result["expected_added_integral"] = expected_added.tolist()
    result["actual_added_integral"] = actual_added.tolist()
    result["control_matches"] = bool(np.allclose(
        actual_added, expected_added, rtol=1e-12, atol=0.0))
    result["accepted"] = (result["continuity_matches"] and result["control_matches"]
        and all(np.all(np.isfinite(result[key]["abs_integral"]))
                and np.all(np.asarray(result[key]["abs_integral"]) > 0)
                for key in ("actual", "canonical"))
        and all(verdict == "AT_BAR" for key in ("actual", "canonical")
                for verdict in result[key]["verdicts"])
        and all(verdict == "DEBT" for verdict in result["planted"]["verdicts"]))
    return result


def height_artifacts(out, provenance, datasets, objects, configs):
    """Persist the measured height arrays, geometry and complete resolved configs."""
    import jax
    import numpy as np

    arrays = dict(datasets)
    for name, obj in objects.items():
        for path, leaf in jax.tree_util.tree_flatten_with_path(obj)[0]:
            arrays[name + jax.tree_util.keystr(path)] = np.asarray(leaf)
    np.savez(out / "height_inputs.npz", **arrays)
    provenance["height_inputs"] = dict(
        sha256=hashlib.sha256((out / "height_inputs.npz").read_bytes()).hexdigest(),
        members={key: dict(shape=list(np.shape(a)), dtype=str(np.asarray(a).dtype))
                 for key, a in arrays.items()})
    provenance["height_configs"] = {key: value._asdict() for key, value in configs.items()}
    # Pin actual imported legoESM source files, including indirect helper imports.
    import sys
    root = Path(__file__).resolve().parents[2]
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("legoesm.") and filename and filename.endswith(".py"):
            path = Path(filename).resolve()
            assert path.is_relative_to(root), (name, path)
            provenance["module_sources"][name] = str(path)
            provenance["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    fixture_paths = ("tests/unit/test_mpas_atmosphere.py",
                     "tests/unit/test_vertical_advection_hybrid_van_leer.py",
                     "tests/unit/test_plane_nh_tracers.py",
                     "tests/unit/test_spectral_dealiasing.py")
    provenance["height_fixture_sources"] = {
        str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in fixture_paths}
    provenance["height_dt"] = 1.0
    provenance["height_boundary"] = "closed lids, sphere or doubly periodic plane; net boundary content=0"
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))


def record_height_inventory(rows, path, area, hc, metric, rho_old, q_old, rho_new, q_new, datasets):
    """Measure full-step inventory separately from the legacy gross-increment scale."""
    import numpy as np

    area, dz, jacobian = map(np.asarray, (area, hc.dz, metric.jacobian))
    volume = area[..., None] * jacobian[..., None] * dz
    rho_old, q_old, rho_new, q_new = map(np.asarray, (rho_old, q_old, rho_new, q_new))
    zero = np.zeros(q_old.shape[-1], dtype=np.float64)
    result = global_inventory_budget(volume, rho_old, q_old, rho_new, q_new, zero)
    result.update(path=path, scheme="full split step", statistic="global inventory budget", scope="full domain")
    rows.append(result)
    print(path, result["statistic"], result["relative_residual"], result["verdicts"], flush=True)
    for control in inventory_controls(volume, rho_old, q_old):
        control.update(path=path, scheme=control["control"], statistic="global inventory budget", scope="full domain instrument control")
        rows.append(control)
    for name, a in dict(area=area, dz=dz, jacobian=jacobian, volume=volume,
                        rho_old=rho_old, q_old=q_old, rho_new=rho_new, q_new=q_new).items():
        datasets[path + "/" + name] = a


def assert_height_inventory_conserves(rows, paths):
    """Require full-step inventory rows for every named fixed path."""
    assert paths, "STOP: no fixed inventory paths requested"
    for path in paths:
        matched = [r for r in rows if r.get("path") == path
                   and r.get("scheme") == "full split step"
                   and r.get("statistic") == "global inventory budget"
                   and r.get("scope") == "full domain"]
        assert matched, ("STOP: missing full-step inventory row", path)
        for row in matched:
            assert row["verdicts"] and all(v == "CONSERVES" for v in row["verdicts"]), row


def realistic_ocean_columns():
    """32 layers, 5--300 m, temperature/salinity, varying column stretch."""
    import jax.numpy as jnp
    import numpy as np
    from legoesm.ocean import vertical as ov

    zc = ov.create_z_star_from_thicknesses(np.geomspace(5.0, 300.0, 32))
    J = jnp.asarray([0.97, 1.0, 1.03])
    h = zc.dz_ref * J[:, None]
    depth = -zc.z_full_ref * J[:, None]
    T = 2.0 + 18.0 * jnp.exp(-depth / 500.0)
    S = 34.0 + 0.7 * (1.0 - jnp.exp(-depth / 1000.0))
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, 33)
    w = 1e-5 * jnp.asarray([1.0, -0.7, 0.4])[:, None] * jnp.sin(phase)
    w = w.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return zc, J, h, T, S, w


def height_vertical_paths(out, rows, provenance, record):
    """Port T24 height rows verbatim; add distinct whole-domain inventory rows."""
    import contextlib
    import inspect
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.grids import vertical as v
    from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis_3d, sh_synthesis_3d

    root = Path(__file__).resolve().parents[2]
    datasets, objects, configs = {}, {}, {}

    @contextlib.contextmanager
    def observe(module, name, path, kind, rho, density_mass_tendency, tracer_only=False):
        real = getattr(module, name)
        sig = inspect.signature(real)
        observed = []

        def spy(*args, **kwargs):
            result = real(*args, **kwargs)
            if tracer_only and not any(f.function in ("_vert_one", "_tracer_tend_one", "<lambda>") for f in inspect.stack()):
                return result
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            a = list(bound.arguments.values())
            q, w = a[:2]
            dz, J = (a[2], a[4]) if kind == "height" else (a[2].dz, a[3])
            m = rho * dz * J[..., None]

            def capture(m_, q_, dq_, dm_, flux_):
                assert np.all(np.asarray(flux_)[..., (0, -1)] == 0), "nonzero boundary flux"
                observed.append(True)
                record(path, kwargs.get("scheme", name), m_, q_, dq_, dm_)
            jax.debug.callback(capture, m, q, result, density_mass_tendency, w, ordered=True)
            return result

        setattr(module, name, spy)
        try:
            yield
            jax.effects_barrier()
            assert observed, f"unexecuted route {path}/{name}"
        finally:
            setattr(module, name, real)

    h = runpy.run_path(str(root / "tests/unit/test_mpas_atmosphere.py"))
    mesh = h["_make_mesh"](level=2)
    _, _, _, q = runpy.run_path(str(root / "tests/unit/test_vertical_advection_hybrid_van_leer.py"))["_column"]()
    from legoesm.atmosphere.dynamics.gcm import compressible_euler_mpas as nh
    from legoesm.atmosphere.dynamics.gcm import spectral_nh as sn
    from legoesm.atmosphere.dynamics.les import compressible_euler_plane as plane
    from legoesm.atmosphere.dynamics.les import compressible_euler_plane_halo as halo
    hc = v.create_height_coordinate(10, 30000.0)
    metric = v.compute_terrain_metric(jnp.zeros(mesh.nCells), hc)
    ns = h["_make_nh_state"](mesh, hc, metric, 10)
    profile = q[0, :10] + q[0, 20:30]
    ww = jnp.sin(jnp.linspace(0, jnp.pi, 11)).at[-1].set(0.0)
    ns = ns._replace(w=ns.w.replace(data=jnp.broadcast_to(ww, (mesh.nCells, 11))), tracers=ns.tracers.replace(data=jnp.broadcast_to(profile, (mesh.nCells, 10))[..., None]))
    objects.update(mpas_mesh=mesh, mpas_hc=hc, mpas_metric=metric, mpas_initial=ns)
    rho = jnp.broadcast_to(hc.rho_ref, (mesh.nCells, 10))
    from legoesm.timestepping.split_explicit import SplitExplicitConfig
    nhcfg = nh.MPASCompressibleEulerConfig()
    def native_acoustic_mp(dt):
        return nh.mpas_acoustic_substeps(ns, None, dt, 1, SplitExplicitConfig(), hc, metric, nhcfg)
    drho = jax.jvp(lambda dt: native_acoustic_mp(dt).rho_prime.data,
                   (jnp.array(0.0),), (jnp.array(1.0),))[1]
    dm_nh = drho * hc.dz * metric.jacobian[..., None]
    with observe(nh, "_vertical_advection_height_1d", "MPAS NH/height", "height", rho, dm_nh, True):
        nh.mpas_compressible_euler_slow_tendencies(ns, mesh, hc, metric, nh.MPASCompressibleEulerConfig())
    acoustic = native_acoustic_mp(jnp.array(1.0))
    assert np.array_equal(acoustic.tracers.data, ns.tracers.data)
    record("MPAS NH/height", "acoustic-only step; tracers unchanged", 1.0, ns.tracers.data[..., 0],
           acoustic.rho_prime.data * hc.dz * metric.jacobian[..., None] * ns.tracers.data[..., 0], 0.0, "finite product increment")
    # Full step is evaluated outside spies; only the native tracer call above
    # is attributed. No semidiscrete result is promoted to a time-step claim.
    nn = nh.MPASCompressibleEulerModel(mesh, hc, metric, nh.MPASCompressibleEulerConfig(fix_mass=False)).step(ns, 1.0)
    m0 = rho * hc.dz * metric.jacobian[..., None]
    m1 = (hc.rho_ref + nn.rho_prime.data) * hc.dz * metric.jacobian[..., None]
    record("MPAS NH/height", "full split step", 1.0, ns.tracers.data[..., 0],
           m1 * nn.tracers.data[..., 0] - m0 * ns.tracers.data[..., 0], 0.0, "finite product increment")

    objects["mpas_final"] = nn
    configs.update(mpas_slow=nhcfg, mpas_acoustic=nhcfg,
                   mpas_full=nh.MPASCompressibleEulerConfig(fix_mass=False),
                   mpas_acoustic_split=SplitExplicitConfig())
    record_height_inventory(rows, "MPAS NH/height", mesh.areaCell, hc, metric,
                            hc.rho_ref + ns.rho_prime.data, ns.tracers.data,
                            hc.rho_ref + nn.rho_prime.data, nn.tracers.data, datasets)

    # Same physical vertical profiles on a Gaussian grid; invoke the real
    # spectral slow caller and separately its full split step.
    from legoesm.grids.gaussian import sh_synthesis_3d
    gg = create_gaussian_grid(10)
    hcg = hc
    gm = v.compute_terrain_metric(jnp.zeros((gg.n_lat, gg.n_lon)), hcg)
    ss = sn.nh_rest_state_spectral(gg, hcg, n_tracers=1)
    qh = jnp.broadcast_to(profile, (gg.n_lat, gg.n_lon, 10))
    wh = jnp.broadcast_to(ww, (gg.n_lat, gg.n_lon, 11))
    ss = ss._replace(w_hat=ss.w_hat.replace(data=sh_analysis_3d(gg, wh)),
                     tracers_hat=ss.tracers_hat.replace(data=sh_analysis_3d(gg, qh)[..., None]))
    objects.update(spectral_grid=gg, spectral_hc=hcg, spectral_metric=gm, spectral_initial=ss)
    rh = jnp.broadcast_to(hcg.rho_ref, qh.shape)
    def native_acoustic_sp(dt):
        return sn._acoustic_substeps_grid(wh, jnp.zeros_like(qh), jnp.zeros_like(qh),
                                         dt, 1, hcg, gm, sn.SpectralNHConfig())
    drho = jax.jvp(lambda dt: native_acoustic_sp(dt)[2],
                   (jnp.array(0.0),), (jnp.array(1.0),))[1]
    dm_sn = drho * hcg.dz * gm.jacobian[..., None]
    with observe(sn, "vertical_advection_height", "spectral NH/height", "height", rh, dm_sn, True):
        sn.spectral_nh_slow_tendencies(ss, gg, hcg, gm, sn.SpectralNHConfig())
    record("spectral NH/height", "acoustic-only step; tracers unchanged", 1.0, qh,
           native_acoustic_sp(jnp.array(1.0))[2] * hcg.dz * gm.jacobian[..., None] * qh, 0.0, "finite product increment")
    nsn = sn.SpectralCompressibleEulerModel(gg, hcg, gm, sn.SpectralNHConfig()).step(ss, 1.0)
    qn = sh_synthesis_3d(gg, nsn.tracers_hat.data[..., 0]).real
    rn = hcg.rho_ref + sh_synthesis_3d(gg, nsn.rho_prime_hat.data).real
    record("spectral NH/height", "full split step", 1.0, qh,
           (rn * hcg.dz * qn - rh * hcg.dz * qh) * gm.jacobian[..., None], 0.0, "finite product increment")

    objects["spectral_final"] = nsn
    configs["spectral"] = sn.SpectralNHConfig()
    initial_q = sh_synthesis_3d(gg, ss.tracers_hat.data[..., 0]).real[..., None]
    initial_rho = hcg.rho_ref + sh_synthesis_3d(gg, ss.rho_prime_hat.data).real
    record_height_inventory(rows, "spectral NH/height", gg.grid_area, hcg, gm,
                            initial_rho, initial_q, rn, qn[..., None], datasets)

    # Keep the exact T24 identity separate from a nonzero existing rough fixture.
    rough = runpy.run_path(str(root / "tests/unit/test_spectral_dealiasing.py"))[
        "TestAliasingStabilityNH"]()._rough_nh_state(gg, hcg, n_tracers=1, seed=1)
    objects["spectral_quadrature_rough_initial"] = rough
    quadrature = [spectral_horizontal_quadrature_probe(
        gg, hcg, gm, state, label, datasets) for label, state in
        (("census_identity", ss), ("supplemental_rough_seed1", rough))]
    rows.extend(quadrature)
    provenance["spectral_quadrature"] = quadrature
    # Save even refuting evidence before the mandatory failure stop.
    height_artifacts(out, provenance, datasets, objects, configs)
    (out / "vertical_paths.json").write_text(json.dumps(rows, indent=2))
    assert all(row["accepted"] for row in quadrature), "spectral horizontal quadrature REFUTED"

    # Native LES local operators are also used on halo columns. Their scope
    # here is explicitly local, not an MPI horizontal conservation claim.
    model, ns, pg, hc, metric = height_les_fixture(root, q)
    qp, wp = ns.tracers.data[..., 0], ns.w.data
    objects.update(les_grid=pg, les_hc=hc, les_metric=metric, les_initial=ns)
    configs["les"] = model.config
    rp = jnp.broadcast_to(hc.rho_ref, qp.shape)
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import acoustic_column_kernel
    def native_acoustic_plane(dt):
        return acoustic_column_kernel(wp, ns.theta_prime.data, ns.rho_prime.data,
                    hc, metric.jacobian, dt, model.config.acoustic_off_centering, model.config.g)
    drho = jax.jvp(lambda dt: native_acoustic_plane(dt)[2],
                   (jnp.array(0.0),), (jnp.array(1.0),))[1]
    dm_plane = drho * hc.dz * metric.jacobian[..., None]
    for mod, label in ((plane, "LES serial"), (halo, "LES halo local")):
        for name in ("_vertical_advection_plane", "vertical_advection_van_leer_plane"):
            with observe(mod, name, label, "plane", rp, dm_plane):
                getattr(mod, name)(qp, wp, hc, metric.jacobian)
    record("LES serial/height", "acoustic-only step; tracers unchanged", 1.0, qp,
           native_acoustic_plane(jnp.array(1.0))[2] * hc.dz * metric.jacobian[..., None] * qp, 0.0, "finite product increment")
    from legoesm.parallel.plane_mpi import make_plane_pencil_layout
    layout = make_plane_pencil_layout(rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
                                      ny_global=pg.ny, nx_global=pg.nx, halo=1)
    for scheme, name in (("centered", "_vertical_advection_plane"),
                         ("van_leer", "vertical_advection_van_leer_plane")):
        cfg = model.config._replace(vertical_tracer_advection=scheme)
        configs["les_slow_" + scheme] = cfg
        with observe(plane, name, "LES serial native caller/" + scheme, "plane", rp, dm_plane, True):
            plane.plane_compressible_euler_slow_tendencies(ns, pg, hc, metric, cfg)
        with observe(halo, name, "LES halo native caller/" + scheme, "plane", rp, dm_plane, True):
            halo.plane_compressible_euler_slow_tendencies_halo(ns, pg, hc, metric, cfg, layout)
    nn = model.step(ns, 1.0)
    record("LES serial/height", "full split step", 1.0, qp,
           ((hc.rho_ref + nn.rho_prime.data) * hc.dz * nn.tracers.data[..., 0] - rp * hc.dz * qp) * metric.jacobian[..., None],
           0.0, "finite product increment")

    objects["les_final"] = nn
    record_height_inventory(rows, "LES serial/height", pg.grid_area, hc, metric,
                            hc.rho_ref + ns.rho_prime.data, ns.tracers.data,
                            hc.rho_ref + nn.rho_prime.data, nn.tracers.data, datasets)
    # Saved initial/final states include the actual lid velocities.
    for name in ("mpas_initial", "mpas_final", "les_initial", "les_final"):
        assert np.all(np.asarray(objects[name].w.data)[..., (0, -1)] == 0), name
    for state in (ss, nsn):
        assert np.all(np.asarray(sh_synthesis_3d(gg, state.w_hat.data).real)[..., (0, -1)] == 0)
    # Reproduce retained standalone T24 diagnostics only. MPAS/LES full-step
    # transport now changes by design; their legacy paired rows remain reported,
    # but acceptance belongs to the distinct global inventory budget below.
    expected = {
        ("MPAS NH/height", "_vertical_advection_height_1d"): 2.6e-2,
        ("MPAS NH/height", "acoustic-only step; tracers unchanged"): 4.5e-1,
        ("spectral NH/height", "vertical_advection_height"): 4.9e-2,
        ("spectral NH/height", "acoustic-only step; tracers unchanged"): 4.5e-1,
        ("spectral NH/height", "full split step"): 5.8e-2,
        ("LES serial", "_vertical_advection_plane"): 2.9e-2,
        ("LES serial", "vertical_advection_van_leer_plane"): 1.7e-3,
        ("LES serial/height", "acoustic-only step; tracers unchanged"): 8.5e-2,
    }
    for label in ("LES halo local", "LES serial native caller/centered", "LES halo native caller/centered"):
        expected[label, "_vertical_advection_plane"] = 2.9e-2
    for label in ("LES serial native caller/van_leer", "LES halo native caller/van_leer"):
        expected[label, "vertical_advection_van_leer_plane"] = 1.7e-3
    expected["LES halo local", "vertical_advection_van_leer_plane"] = 1.7e-3
    legacy = {(r["path"], r["scheme"]): r for r in rows if "max_relative" in r}
    reproduction = []
    for key, target in expected.items():
        value = legacy[key]["max_relative"]
        reproduction.append(dict(path=key[0], scheme=key[1], measured=value,
                                 issue2641=target, agrees=float(format(value, ".2g")) == target))
    provenance["issue2641_reproduction"] = reproduction
    height_artifacts(out, provenance, datasets, objects, configs)
    (out / "vertical_paths.json").write_text(json.dumps(rows, indent=2))
    assert all(r["agrees"] for r in reproduction), "STOP: T24 issue2641 reproduction mismatch; inspect artifacts"
    assert_height_inventory_conserves(rows, ("MPAS NH/height", "LES serial/height"))



def all_vertical_paths(out):
    """Preserve ocean census and append the T29 height-core measurements."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean import vertical as ov
    from legoesm.ocean.dynamics import ocean_pe_cdgrid as cd
    from legoesm.ocean.dynamics import spectral_ocean_pe as so

    assert os.environ.get("PBS_JOBID"), "PBS required"
    assert jax.config.x64_enabled and all(d.platform == "cpu" for d in jax.devices())
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = {mod.__name__: str(Path(mod.__file__).resolve()) for mod in (ov, cd, so)}
    assert all(Path(path).is_relative_to(root) for path in sources.values()), sources
    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "PBS_JOBID": os.environ["PBS_JOBID"], "precision": "fp64", "device": "cpu",
        "tolerance": 1e-12, "complete": False, "module_sources": sources,
        "source_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in sources.items()},
        "scope": "ocean vertical subsystem only; closed lids; positive-up w; dh=diff(w)",
        "fixture": "PR2597 _column(32,seed=0); 32-layer 5--300 m T/S; J=[0.97,1,1.03]",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "measured_source.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True))
    rows = []

    def record(path, scheme, m, q, dq, dm, statistic="instantaneous"):
        for value in (m, q, dq, dm):
            if np.ndim(value):
                assert np.asarray(value).dtype == np.float64, "fp64 arrays required"
        r = paired_column_budget(m, q, dq, dm)
        assert np.any(np.asarray(r["denominator"]) > 0), f"vacuous route {path}"
        r.update(path=path, scheme=scheme, statistic=statistic,
                 boundary_budget=0.0, scope="vertical subsystem; per column")
        rows.append(r)
        print(path, scheme, statistic, r["max_relative"], r["verdict"], flush=True)
        (out / "vertical_paths.json").write_text(json.dumps(rows, indent=2))
        return r

    qo, wo, m = runpy.run_path(str(root / "tests/ocean/unit/test_adaptive_implicit_vertadv.py"))["_column"](32, seed=0)
    np.testing.assert_array_equal(np.asarray(wo)[..., [0, -1]], 0.0)
    zc = ov.create_z_star_from_thicknesses(np.asarray(m[0]))
    # The builder recovers dz from snapped interfaces: use its actual thickness.
    np.testing.assert_allclose(zc.dz_ref, m[0], rtol=1e-12, atol=0.0)
    m = zc.dz_ref[None, :]
    J = jnp.ones(1)
    # These are the actual tracer bindings used by the two tendency functions.
    operators = (("CD-grid", cd._vertical_advection_ocean),
                 ("spectral", so.vertical_advection_ocean))
    provenance["operators"] = {path: op.__module__ + "." + op.__qualname__ for path, op in operators}
    for path, op in operators:
        record(path + " original", op.__name__, m, qo, op(qo, wo, zc, J), jnp.diff(wo))
    zc, J, h, T, S, w = realistic_ocean_columns()
    np.savez(out / "inputs.npz", h=h, T=T, S=S, w=w, J=J, original_q=qo, original_w=wo, original_h=m)
    provenance["inputs_sha256"] = hashlib.sha256((out / "inputs.npz").read_bytes()).hexdigest()
    for path, op in operators:
        for name, q in (("T", T), ("S", S)):
            record(path + " realistic " + name, op.__name__, h, q, op(q, w, zc, J), jnp.diff(w))
    qnew = ov.implicit_vertical_advection_ocean(qo, wo, m, 600.0)
    control = record("ocean", "implicit flux form fixed thickness", 1.0, qo,
                     m*qnew-m*qo, 0.0, "finite product increment")
    assert control["verdict"] == "CONSERVES"
    dq = ov.vertical_advection_ocean(T, w, zc, J)
    # Add a nonzero unbalanced source, independent of the model's own residual.
    leak = jnp.zeros_like(T).at[:, 0].set(1e-3)
    control = record("planted leak", "surface source 1e-3 K/s", h, T, dq+leak, jnp.diff(w))
    assert control["verdict"] == "LEAKS"
    # Golden primitive momentum arrays; production wiring is checked separately.
    momentum_op = cd.advective_vertical_momentum_ocean
    spectral_momentum = so.advective_vertical_momentum_ocean
    provenance["momentum_operators"] = {"CD-grid": momentum_op.__module__ + "." + momentum_op.__qualname__,
                                        "spectral": spectral_momentum.__module__ + "." + spectral_momentum.__qualname__}
    np.savez(out / "momentum.npz", cd=momentum_op(T,w,zc,J), spectral=spectral_momentum(T,w,zc,J))
    provenance["lane"] = "serial vertical census"
    height_vertical_paths(out, rows, provenance, record)
    provenance["complete"] = True
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))


def height_halo_mpi(out):
    """Genuine two-rank T24 full step; MPI failures abort the whole lane."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    try:
        if comm.Get_size() != 2:
            raise ValueError("--height-halo-mpi requires exactly two MPI ranks")
        # Rank-count check deliberately precedes JAX, mpi4jax and collectives.
        import mpi4jax  # noqa: F401
        import jax
        import jax.numpy as jnp
        import numpy as np
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64())
        from legoesm.atmosphere.dynamics.les import compressible_euler_plane as plane
        from legoesm.grids.plane import create_plane_grid
        from legoesm.parallel.plane_mpi import (
            gather_plane_field, make_plane_pencil_layout, scatter_plane_field,
        )

        assert os.environ.get("PBS_JOBID"), "PBS required"
        assert jax.config.x64_enabled and all(d.platform == "cpu" for d in jax.devices())
        root = Path(__file__).resolve().parents[2]
        _, _, _, q = runpy.run_path(str(root / "tests/unit/test_vertical_advection_hybrid_van_leer.py"))["_column"]()
        model, ns, pg, hc, metric = height_les_fixture(root, q)
        fields = ("u", "v", "w", "theta_prime", "rho_prime", "tracers", "phis")
        layout = make_plane_pencil_layout(
            rank=comm.Get_rank(), n_ranks=2, n_ranks_y=2, n_ranks_x=1,
            ny_global=pg.ny, nx_global=pg.nx,
            halo=plane.horizontal_advection_halo_width(model.config))
        assert layout.n_ranks == 2
        local_grid = create_plane_grid(nx=layout.nx_local, ny=layout.ny_local, nlev=pg.nlev,
                                       dx=pg.dx, dy=pg.dy, dtype=jnp.float64)
        local_metric = plane.make_flat_plane_terrain_metric(local_grid, hc)
        local_model = plane.PlaneCompressibleEulerModel(local_grid, hc, local_metric, model.config)
        local_state = ns._replace(**{name: getattr(ns, name).replace(
            data=scatter_plane_field(getattr(ns, name).data, layout)) for name in fields})
        provenance = dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            PBS_JOBID=os.environ["PBS_JOBID"], precision="fp64", device="cpu",
            tolerance=1e-12, complete=False, lane="height halo MPI",
            module_sources={}, source_sha256={})
        rank_provenance = dict(provenance, rank=layout.rank, layout=layout._asdict(),
                               global_initial_sha256={name: hashlib.sha256(
                                   np.ascontiguousarray(np.asarray(getattr(ns, name).data)).tobytes()).hexdigest()
                                   for name in fields})
        # Source hashes must be collected independently on every rank.
        import sys
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if name.startswith("legoesm.") and filename and filename.endswith(".py"):
                path = Path(filename).resolve()
                assert path.is_relative_to(root), (name, path)
                rank_provenance["module_sources"][name] = str(path)
                rank_provenance["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rank_provenance["fixture_sha256"] = {
            str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("tests/unit/test_vertical_advection_hybrid_van_leer.py",
                         "tests/unit/test_plane_nh_tracers.py")}
        ranks = comm.gather(rank_provenance, root=0)
        if layout.rank == 0:
            assert ranks[0]["global_initial_sha256"] == ranks[1]["global_initial_sha256"], "rank fixture mismatch"
            provenance["ranks"] = ranks
            out.mkdir(parents=True, exist_ok=True)
            (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
            (out / "measured_source.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True))
        # Root assertion failures above reach comm.Abort via the handler below.
        rho_old = gather_plane_field(hc.rho_ref + local_state.rho_prime.data, layout)
        q_old = gather_plane_field(local_state.tracers.data, layout)
        final = local_model.step_halo(local_state, dt=1.0, layout=layout)
        jax.effects_barrier()
        assert np.all(np.asarray(local_state.w.data)[..., (0, -1)] == 0)
        assert np.all(np.asarray(final.w.data)[..., (0, -1)] == 0)
        rho_new = gather_plane_field(hc.rho_ref + final.rho_prime.data, layout)
        q_new = gather_plane_field(final.tracers.data, layout)
        w_new = gather_plane_field(final.w.data, layout)
        if layout.rank == 0:
            np.testing.assert_array_equal(rho_old, hc.rho_ref + ns.rho_prime.data)
            np.testing.assert_array_equal(q_old, ns.tracers.data)
            rows, datasets = [], {"halo_final_w": np.asarray(w_new)}
            record_height_inventory(rows, "LES halo full/height", pg.grid_area, hc, metric,
                                    rho_old, q_old, rho_new, q_new, datasets)
            objects = dict(les_grid=pg, les_hc=hc, les_metric=metric, les_initial=ns)
            height_artifacts(out, provenance, datasets, objects, {"les": model.config})
            (out / "vertical_paths.json").write_text(json.dumps(rows, indent=2))
            assert_height_inventory_conserves(rows, ("LES halo full/height",))
            provenance["complete"] = True
            (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    except Exception:
        import traceback
        traceback.print_exc()
        comm.Abort(1)
        raise


if __name__ == "__main__":
    import sys
    if "--all-vertical-paths" in sys.argv:
        parser = argparse.ArgumentParser(description="Ocean and height-coordinate vertical census (T28/T29)")
        parser.add_argument("--all-vertical-paths", action="store_true", required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--height-halo-mpi", action="store_true")
        args = parser.parse_args()
        if args.height_halo_mpi:
            height_halo_mpi(args.out)
        else:
            all_vertical_paths(args.out)
    else:
        main()
