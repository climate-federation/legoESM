"""Colliding-modons nonlinear SW driver + faithfulness diagnostics (issue #521).

Runs the FV3 case-8 twin-vortex ("Colliding Modons", doi:10.1002/2017MS000965)
on the cubed-sphere FV3 shallow-water solver, non-rotating (f=0), and tracks the
collision/exchange/return plus conservation and cube-symmetry artifacts.

The two equatorial modons start a half-circumference apart (90E westerly /
270E easterly), self-propagate, collide near a cube face boundary, exchange
vortices, and return to their initial longitudes after ~100 days.  A cube
pathology shows up as (a) energy/mass drift, (b) loss of the lon->lon+pi
antisymmetry, or (c) grid-scale noise at the seams.

Run:
  PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python \
    scripts/validate/run_colliding_modons.py --n 48 --days 60 --report-every 5
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _total_energy(h, u_d, v_d, area, g):
    """Shallow-water energy ~ 0.5*(h*|U|^2) + 0.5*g*h^2, integrated.  Winds are
    on edges; average to centres for a scalar diagnostic (monitor, not budget)."""
    uc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])      # (6,n,n)
    vc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    ke = 0.5 * h * (uc ** 2 + vc ** 2)
    pe = 0.5 * g * h ** 2
    return float(jnp.sum((ke + pe) * area))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--days", type=float, default=60.0)
    p.add_argument("--dt", type=float, default=300.0)
    p.add_argument("--report-every", type=float, default=5.0)
    p.add_argument("--div-damp", type=float, default=8.0)   # iter1009 default
    p.add_argument("--damp-v", type=float, default=0.030)
    p.add_argument("--hyperdiff-factor", type=float, default=0.0)  # x _hyperdiff_cube(n)
    args = p.parse_args()

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel, iter1009_dual_target_config)
    from tests.test_cases.modons import colliding_modons_cubesphere

    n = args.n
    grid = create_cubed_sphere(n)
    try:
        from scripts.matrix.run_atmosphere_test_matrix import _hyperdiff_cube
        hyperdiff = args.hyperdiff_factor * _hyperdiff_cube(n)
    except Exception:
        hyperdiff = 0.0
    cfg = iter1009_dual_target_config(
        n, div_damp_factor=args.div_damp, damp_v=args.damp_v,
        hyperdiff_coeff=hyperdiff)
    model = FV3EdgeShallowWaterModel(grid, cfg)
    state, cd_nr = colliding_modons_cubesphere(grid, model.cdgrid)
    model.cdgrid = cd_nr
    model.set_initial_mass(state)
    state = model.step(state, args.dt)         # settle storage dtype (1 step)
    g = model.config.g
    area = model.cdgrid.base.area
    lon_c = grid.lon

    mass0 = float(jnp.sum(state.h * area))
    en0 = _total_energy(state.h, state.u_d, state.v_d, area, g)

    steps_per_day = int(round(86400 / args.dt))
    block_days = args.report_every
    n_blocks = int(round(args.days / block_days))
    n_steps_block = int(steps_per_day * block_days)

    step = jax.jit(lambda s: model.step(s, args.dt))

    def day_block(s):
        for _ in range(n_steps_block):
            s = step(s)
        return s

    # centre-cell zonal wind for tracking the modons (edge u_d -> centre)
    def diag(s):
        uc = 0.5 * (s.u_d[:, :, :-1] + s.u_d[:, :, 1:])
        # longitude of the strongest westerly and easterly centre-cell wind
        flat = np.asarray(uc).ravel()
        lo = np.asarray(lon_c).ravel()
        i_w = int(np.argmax(flat)); i_e = int(np.argmin(flat))
        lon_w = np.degrees(lo[i_w]) % 360
        lon_e = np.degrees(lo[i_e]) % 360
        # lon->lon+pi antisymmetry of the centre zonal wind on a lat-lon probe
        return lon_w, lon_e, float(flat[i_w]), float(flat[i_e])

    print(f"# colliding modons C{n}  days={args.days}  dt={args.dt}  "
          f"f=0 (non-rotating)")
    print(f"{'day':>5} {'mass_err':>10} {'energy_err':>11} {'max|u|':>8} "
          f"{'lon_W':>7} {'lon_E':>7} {'minh':>8}")
    print(f"{0:>5} {0.0:>10.2e} {0.0:>11.2e} "
          f"{float(jnp.max(jnp.abs(state.u_d))):>8.2f} "
          f"{diag(state)[0]:>7.1f} {diag(state)[1]:>7.1f} "
          f"{float(jnp.min(state.h)):>8.1f}")

    s = state
    for b in range(1, n_blocks + 1):
        s = day_block(s)
        jax.block_until_ready(s.h)
        if not np.all(np.isfinite(np.asarray(s.h))):
            print(f"# BLEW UP at day {b*block_days}"); break
        mass_err = float(jnp.sum(s.h * area) / mass0 - 1.0)
        en_err = _total_energy(s.h, s.u_d, s.v_d, area, g) / en0 - 1.0
        lon_w, lon_e, uw, ue = diag(s)
        print(f"{b*block_days:>5.0f} {mass_err:>10.2e} {en_err:>11.2e} "
              f"{float(jnp.max(jnp.abs(s.u_d))):>8.2f} {lon_w:>7.1f} "
              f"{lon_e:>7.1f} {float(jnp.min(s.h)):>8.1f}")

    np.save("/tmp/modon_h.npy", np.asarray(s.h))
    np.save("/tmp/modon_h0.npy", np.asarray(state.h))
    print("# saved /tmp/modon_h{,0}.npy")


if __name__ == "__main__":
    main()
