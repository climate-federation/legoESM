#!/usr/bin/env python
"""Test JAX differentiability of MOST bulk flux algorithms.

Verifies that jax.grad works through the Obukhov length iteration
(jax.lax.fori_loop) for both COARE 3.0 and Large & Yeager 2004.
"""

import jax
import jax.numpy as jnp

from legoesm.coupler.bulk_flux import compute_most_fluxes, psi_m, psi_h


if __name__ == "__main__":
    print("=" * 70)
    print("MOST Bulk Flux Differentiability Test")
    print("=" * 70)

    # Test stability functions first
    print("\n--- Stability function values ---")
    zetas = jnp.array([-2.0, -0.5, -0.01, 0.0, 0.01, 0.5, 2.0])
    for z in zetas:
        print(f"  ζ={float(z):+6.2f}  ψ_m={float(psi_m(z)):+8.4f}  ψ_h={float(psi_h(z)):+8.4f}")

    # Gradient of stability functions
    print("\n--- Stability function gradients ---")
    dpsi_m = jax.grad(lambda z: psi_m(z).sum())
    dpsi_h = jax.grad(lambda z: psi_h(z).sum())
    for z in [-1.0, -0.1, 0.1, 1.0]:
        z_arr = jnp.array(z)
        print(f"  ζ={z:+5.1f}  dψ_m/dζ={float(dpsi_m(z_arr)):+8.4f}"
              f"  dψ_h/dζ={float(dpsi_h(z_arr)):+8.4f}")


    # Test data: array of SST values with spatial variation
    shape = (6, 4, 4)
    T_sfc = jnp.full(shape, 300.0) + 5.0 * jnp.sin(jnp.linspace(0, 3, 4))[None, :, None]
    T_atm = jnp.full(shape, 290.0)
    q_sfc = jnp.full(shape, 0.020)
    q_atm = jnp.full(shape, 0.008)
    u_rel = jnp.full(shape, 8.0)
    v_rel = jnp.full(shape, 2.0)
    rho = jnp.full(shape, 1.2)

    schemes = ["coare3", "large_yeager"]
    n_iters = [3, 5, 10]

    all_pass = True

    for scheme in schemes:
        print(f"\n--- {scheme.upper()} ---")

        for n_iter in n_iters:
            def loss(T_sfc_in):
                tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
                    u_rel, v_rel, T_atm, q_atm,
                    T_sfc_in, q_sfc, rho,
                    z_ref=10.0, z0_init=1e-4,
                    scheme=scheme, n_iter=n_iter,
                )
                return jnp.mean(shflx ** 2 + lhflx ** 2 + tau_x ** 2)

            grads = jax.grad(loss)(T_sfc)
            is_finite = bool(jnp.all(jnp.isfinite(grads)))
            is_nonzero = bool(not jnp.allclose(grads, 0.0))
            grad_norm = float(jnp.max(jnp.abs(grads)))

            if is_finite and is_nonzero:
                print(f"  PASS  n_iter={n_iter:2d}  |grad|_max={grad_norm:.4e}")
            elif is_finite:
                print(f"  WARN  n_iter={n_iter:2d}  gradients are all zero")
                all_pass = False
            else:
                print(f"  FAIL  n_iter={n_iter:2d}  non-finite gradients, |grad|_max={grad_norm}")
                all_pass = False

        # Test gradient w.r.t. wind speed
        def loss_wind(u_in):
            tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
                u_in, v_rel, T_atm, q_atm,
                T_sfc, q_sfc, rho,
                z_ref=10.0, z0_init=1e-4,
                scheme=scheme, n_iter=5,
            )
            return jnp.mean(tau_x ** 2)

        grads_u = jax.grad(loss_wind)(u_rel)
        is_finite_u = bool(jnp.all(jnp.isfinite(grads_u)))
        is_nonzero_u = bool(not jnp.allclose(grads_u, 0.0))
        grad_norm_u = float(jnp.max(jnp.abs(grads_u)))

        if is_finite_u and is_nonzero_u:
            print(f"  PASS  grad w.r.t. wind  |grad|_max={grad_norm_u:.4e}")
        else:
            print(f"  FAIL  grad w.r.t. wind  finite={is_finite_u} nonzero={is_nonzero_u}")
            all_pass = False


    # Test physical sanity: unstable vs neutral
    print("\n--- Physical sanity checks ---")
    for scheme in schemes:
        tau_x_n, _, _, _, ustar_n = compute_most_fluxes(
            jnp.array(10.0), jnp.array(0.0),
            jnp.array(290.0), jnp.array(0.008),
            jnp.array(290.0), jnp.array(0.008),  # neutral: T_sfc = T_atm
            jnp.array(1.2),
            scheme=scheme, n_iter=5,
        )
        tau_x_u, _, shflx_u, _, ustar_u = compute_most_fluxes(
            jnp.array(10.0), jnp.array(0.0),
            jnp.array(290.0), jnp.array(0.008),
            jnp.array(300.0), jnp.array(0.020),  # unstable: warm SST
            jnp.array(1.2),
            scheme=scheme, n_iter=5,
        )
        tau_x_s, _, shflx_s, _, ustar_s = compute_most_fluxes(
            jnp.array(10.0), jnp.array(0.0),
            jnp.array(290.0), jnp.array(0.008),
            jnp.array(280.0), jnp.array(0.004),  # stable: cold SST
            jnp.array(1.2),
            scheme=scheme, n_iter=5,
        )
        # Unstable should enhance turbulence (larger u*, |τ|)
        u_enhanced = float(jnp.abs(tau_x_u)) > float(jnp.abs(tau_x_n))
        # Stable should suppress turbulence (smaller u*, |τ|)
        s_suppressed = float(jnp.abs(tau_x_s)) < float(jnp.abs(tau_x_n))
        # Positive SH when surface warmer
        sh_positive = float(shflx_u) > 0.0
        # Negative SH when surface cooler
        sh_negative = float(shflx_s) < 0.0

        ok = u_enhanced and s_suppressed and sh_positive and sh_negative
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {scheme:12s}  |τ_u|={float(jnp.abs(tau_x_u)):.4f}"
              f"  |τ_n|={float(jnp.abs(tau_x_n)):.4f}"
              f"  |τ_s|={float(jnp.abs(tau_x_s)):.4f}"
              f"  SH_u={float(shflx_u):.1f}  SH_s={float(shflx_s):.1f}")
        if not ok:
            all_pass = False


    # Test coupler integration
    print("\n--- Coupler ocean_tile_response with MOST ---")
    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.coupling_fields import AtmToSurface
    from legoesm.coupler.coupler import ocean_tile_response

    for scheme in ["constant", "coare3", "large_yeager"]:
        try:
            config = CouplerConfig(bulk_scheme=scheme, z_ref=10.0, bulk_n_iter=5)
            forcing = AtmToSurface(
                sw_down=jnp.full(shape, 200.0),
                lw_down=jnp.full(shape, 300.0),
                precip_total=jnp.zeros(shape),
                precip_snow=jnp.zeros(shape),
                T_lowest=jnp.full(shape, 290.0),
                q_lowest=jnp.full(shape, 0.008),
                u_lowest=jnp.full(shape, 8.0),
                v_lowest=jnp.full(shape, 2.0),
                p_lowest=jnp.full(shape, 95000.0),
                p_surface=jnp.full(shape, 101325.0),
                rho_lowest=jnp.full(shape, 1.2),
                cos_zenith=jnp.full(shape, 0.5),
                co2_ppmv=jnp.array(400.0),
                has_radiation=jnp.array(1.0),
                has_precipitation=jnp.array(1.0),
            )
            sst = jnp.full(shape, 300.0) + 3.0 * jnp.sin(jnp.linspace(0, 3, 4))[None, :, None]
            ocean_u = jnp.zeros(shape)
            ocean_v = jnp.zeros(shape)

            resp = ocean_tile_response(forcing, sst, ocean_u, ocean_v, config)

            # Test differentiability through coupler
            def coupler_loss(sst_in):
                r = ocean_tile_response(forcing, sst_in, ocean_u, ocean_v, config)
                return jnp.mean(r.shflx ** 2)

            grads = jax.grad(coupler_loss)(sst)
            is_finite = bool(jnp.all(jnp.isfinite(grads)))
            is_nonzero = bool(not jnp.allclose(grads, 0.0))
            grad_norm = float(jnp.max(jnp.abs(grads)))

            if is_finite and is_nonzero:
                print(f"  PASS  {scheme:12s}  |grad|_max={grad_norm:.4e}"
                      f"  SH_mean={float(jnp.mean(resp.shflx)):.1f} W/m²")
            else:
                print(f"  FAIL  {scheme:12s}  finite={is_finite} nonzero={is_nonzero}")
                all_pass = False
        except Exception as e:
            print(f"  ERROR {scheme:12s}  {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            all_pass = False

    print()
    if all_pass:
        print("All MOST bulk flux tests PASSED.")
    else:
        print("Some tests FAILED — see above.")
