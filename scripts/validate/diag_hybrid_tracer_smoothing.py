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


def all_vertical_paths(out):
    """Reuse the original paired statistic and ocean controls only."""
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
    # Momentum shares the paired tracer operator (#2662); wiring is checked separately.
    momentum_op = cd._vertical_advection_ocean
    spectral_momentum = so.vertical_advection_ocean
    provenance["momentum_operators"] = {"CD-grid": momentum_op.__module__ + "." + momentum_op.__qualname__,
                                        "spectral": spectral_momentum.__module__ + "." + spectral_momentum.__qualname__}
    np.savez(out / "momentum.npz", cd=momentum_op(T,w,zc,J), spectral=spectral_momentum(T,w,zc,J))
    provenance["complete"] = True
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    import sys
    if "--all-vertical-paths" in sys.argv:
        parser = argparse.ArgumentParser(description="Ocean-only vertical census (T28)")
        parser.add_argument("--all-vertical-paths", action="store_true", required=True)
        parser.add_argument("--out", type=Path, required=True)
        all_vertical_paths(parser.parse_args().out)
    else:
        main()
