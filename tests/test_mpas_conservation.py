"""Conservation diagnostics for MPAS icosahedral grid.

Tests mass/energy/enstrophy conservation for shallow water
and volume/heat/salt conservation for the ocean model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.core.conservation import global_integral_voronoi
from legoesm.core.operators_voronoi import (
    kinetic_energy_cell,
    curl_vertex,
    vertex_thickness,
)
from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
    MPASShallowWaterConfig,
    MPASShallowWaterModel,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.init_mpas import rest_state_mpas_ocean

# Import test cases
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "atmosphere", "shallow_water"))
from test_cases.williamson_mpas import (
    williamson_test2_mpas,
    williamson_test5_mpas,
    williamson_test6_mpas,
)


# ============================================================================
# Helpers
# ============================================================================

def sw_diagnostics(state, mesh, g):
    """Compute shallow water conservation diagnostics."""
    h = state.h.data
    u = state.u.data
    h_s = state.h_s.data
    area = mesh.areaCell

    # Mass
    mass = jnp.sum(h * area)

    # Energy: KE + PE
    ke_cell = kinetic_energy_cell(u, mesh)  # (nCells,)
    KE = jnp.sum(ke_cell * h * area)
    PE = jnp.sum(0.5 * g * (h + h_s) ** 2 * area)
    energy = KE + PE

    # Potential enstrophy: 0.5 * sum(q^2 * h_v * area_v)
    zeta = curl_vertex(u, mesh)
    h_v = vertex_thickness(h, mesh)
    q_v = (zeta + mesh.fVertex) / jnp.maximum(h_v, 1e-10)
    enstrophy = 0.5 * jnp.sum(q_v ** 2 * h_v * mesh.areaTriangle)

    return {
        "mass": float(mass),
        "energy": float(energy),
        "KE": float(KE),
        "PE": float(PE),
        "enstrophy": float(enstrophy),
    }


def ocean_diagnostics(state, mesh, z_coord, config):
    """Compute ocean conservation diagnostics."""
    mask = state.land_mask.data
    area = mesh.areaCell
    H_bathy = state.H_bathy.data
    eta = state.eta.data

    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # Volume: integral of eta over ocean
    volume = jnp.sum(eta * mask * area)

    # Heat: integral of T * h_k over ocean
    heat = jnp.sum(
        state.T.data * h_k * mask[:, jnp.newaxis] * area[:, jnp.newaxis]
    )

    # Salt: integral of S * h_k over ocean
    salt = jnp.sum(
        state.S.data * h_k * mask[:, jnp.newaxis] * area[:, jnp.newaxis]
    )

    # Total kinetic energy
    u_3d = state.u.data  # (nEdges, nlev)
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    h_e_k = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
    edge_mask = mask[c1] * mask[c2]
    # KE per edge per level
    dc = mesh.dcEdge
    dv = mesh.dvEdge
    area_e = dc * dv
    KE = 0.5 * jnp.sum(u_3d ** 2 * h_e_k * area_e[:, jnp.newaxis] * edge_mask[:, jnp.newaxis])

    return {
        "volume": float(volume),
        "heat": float(heat),
        "salt": float(salt),
        "KE": float(KE),
    }


def relative_change(val, ref):
    """Compute relative change, handling zero reference."""
    if abs(ref) < 1e-30:
        return abs(val - ref)
    return abs(val - ref) / abs(ref)


# ============================================================================
# Shallow Water Conservation
# ============================================================================

def check_sw_conservation():
    """Check conservation for shallow water on icosahedral grid."""
    print("=" * 70)
    print("SHALLOW WATER CONSERVATION ON ICOSAHEDRAL GRID")
    print("=" * 70)

    mesh = create_voronoi_mesh(subdivision_level=3)
    print(f"\nMesh: level 3, {mesh.nCells} cells, {mesh.nEdges} edges")

    g = constants.g

    for tc_name, tc_fn, dt, n_steps, fix_mass, fix_energy in [
        ("TC2 (geostrophic, no fixers)", williamson_test2_mpas, 600.0, 100, False, False),
        ("TC2 (geostrophic, mass fixer)", williamson_test2_mpas, 600.0, 100, True, False),
        ("TC2 (geostrophic, mass+energy fixer)", williamson_test2_mpas, 600.0, 100, True, True),
        ("TC5 (mountain, no fixers)", williamson_test5_mpas, 600.0, 100, False, False),
        ("TC5 (mountain, mass fixer)", williamson_test5_mpas, 600.0, 100, True, False),
        ("TC6 (RH wave, no fixers)", williamson_test6_mpas, 300.0, 200, False, False),
        ("TC6 (RH wave, mass+energy fixer)", williamson_test6_mpas, 300.0, 200, True, True),
    ]:
        print(f"\n--- {tc_name} ---")
        print(f"  dt={dt}s, {n_steps} steps, duration={dt*n_steps/86400:.2f} days")

        state0 = tc_fn(mesh)
        config = MPASShallowWaterConfig(
            g=g,
            nu_del2=0.0,
            nu_del4=0.0,
            fix_mass=fix_mass,
            fix_energy=fix_energy,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)

        diag0 = sw_diagnostics(state0, mesh, g)

        state = state0
        for i in range(n_steps):
            state = model.step(state, dt)

        diag_final = sw_diagnostics(state, mesh, g)

        mass_change = relative_change(diag_final["mass"], diag0["mass"])
        energy_change = relative_change(diag_final["energy"], diag0["energy"])
        enstrophy_change = relative_change(diag_final["enstrophy"], diag0["enstrophy"])

        print(f"  Mass:      {diag0['mass']:.6e} -> {diag_final['mass']:.6e}  "
              f"rel change: {mass_change:.2e}")
        print(f"  Energy:    {diag0['energy']:.6e} -> {diag_final['energy']:.6e}  "
              f"rel change: {energy_change:.2e}")
        print(f"  Enstrophy: {diag0['enstrophy']:.6e} -> {diag_final['enstrophy']:.6e}  "
              f"rel change: {enstrophy_change:.2e}")
        print(f"  KE: {diag0['KE']:.6e} -> {diag_final['KE']:.6e}")
        print(f"  PE: {diag0['PE']:.6e} -> {diag_final['PE']:.6e}")

        # Height field check
        h_err = jnp.max(jnp.abs(state.h.data - state0.h.data))
        h_ref = jnp.max(jnp.abs(state0.h.data))
        print(f"  Max |h - h0|: {float(h_err):.4e} m "
              f"(rel: {float(h_err/h_ref):.2e})")


# ============================================================================
# Ocean Conservation
# ============================================================================

def check_ocean_conservation():
    """Check conservation for ocean PE on icosahedral grid."""
    print("\n" + "=" * 70)
    print("OCEAN PE CONSERVATION ON ICOSAHEDRAL GRID")
    print("=" * 70)

    mesh = create_voronoi_mesh(subdivision_level=2)
    z_coord = create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0,
    )
    print(f"\nMesh: level 2, {mesh.nCells} cells, {mesh.nEdges} edges")
    print(f"Vertical: {z_coord.n_levels} levels, H_max={z_coord.H_max}m")

    for test_name, fix_vol, fix_heat, fix_salt, dt, n_steps, A_h in [
        ("Rest state, no fixers", False, False, False, 60.0, 50, 1e3),
        ("Rest state, all fixers", True, True, True, 60.0, 50, 1e3),
        ("Perturbed, no fixers", False, False, False, 60.0, 50, 1e3),
        ("Perturbed, all fixers", True, True, True, 60.0, 50, 1e3),
    ]:
        print(f"\n--- {test_name} ---")
        print(f"  dt={dt}s, {n_steps} steps, duration={dt*n_steps/3600:.2f} hours")

        config = MPASOceanConfig(
            A_h=A_h,
            K_h=1e2,
            A_v=1e-3,
            K_v=1e-4,
            n_barotropic_substeps=5,
            use_conservation_fixer=(fix_vol or fix_heat or fix_salt),
            fix_volume=fix_vol,
            fix_heat=fix_heat,
            fix_salt=fix_salt,
        )

        state0 = rest_state_mpas_ocean(
            mesh, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=500.0, land_lat_threshold=85.0,
        )

        mask = state0.land_mask.data

        # Add perturbation for non-rest tests
        if "Perturbed" in test_name:
            T_pert = state0.T.data + 0.5 * jnp.sin(
                4 * mesh.latCell
            )[:, jnp.newaxis] * mask[:, jnp.newaxis]
            eta_pert = state0.eta.data + 0.01 * jnp.sin(
                3 * mesh.lonCell
            ) * mask
            state0 = state0._replace(
                T=state0.T.replace(data=T_pert),
                eta=state0.eta.replace(data=eta_pert),
            )

        model = MPASOceanModel(mesh, z_coord, config)
        diag0 = ocean_diagnostics(state0, mesh, z_coord, config)

        state = state0
        for i in range(n_steps):
            state = model.step(state, dt)

        diag_final = ocean_diagnostics(state, mesh, z_coord, config)

        total_area = jnp.sum(state0.land_mask.data * mesh.areaCell)

        vol_change = relative_change(diag_final["volume"], diag0["volume"]) if abs(diag0["volume"]) > 1e-20 else abs(diag_final["volume"] - diag0["volume"]) / float(total_area)
        heat_change = relative_change(diag_final["heat"], diag0["heat"])
        salt_change = relative_change(diag_final["salt"], diag0["salt"])

        print(f"  Volume:  {diag0['volume']:.6e} -> {diag_final['volume']:.6e}  "
              f"change: {vol_change:.2e}")
        print(f"  Heat:    {diag0['heat']:.6e} -> {diag_final['heat']:.6e}  "
              f"rel change: {heat_change:.2e}")
        print(f"  Salt:    {diag0['salt']:.6e} -> {diag_final['salt']:.6e}  "
              f"rel change: {salt_change:.2e}")
        print(f"  KE:      {diag0['KE']:.6e} -> {diag_final['KE']:.6e}")

        # Check all fields finite
        all_finite = (
            jnp.all(jnp.isfinite(state.u.data))
            and jnp.all(jnp.isfinite(state.T.data))
            and jnp.all(jnp.isfinite(state.S.data))
            and jnp.all(jnp.isfinite(state.eta.data))
        )
        print(f"  All fields finite: {bool(all_finite)}")

        # Max eta change (ocean cells only)
        eta_err = jnp.max(jnp.abs(
            (state.eta.data - state0.eta.data) * mask
        ))
        print(f"  Max |eta change| (ocean): {float(eta_err):.4e} m")

        # Max T change (ocean cells only)
        T_err = jnp.max(jnp.abs(
            (state.T.data - state0.T.data) * mask[:, jnp.newaxis]
        ))
        print(f"  Max |T change| (ocean): {float(T_err):.4e} degC")

        # Max S change (ocean cells only)
        S_err = jnp.max(jnp.abs(
            (state.S.data - state0.S.data) * mask[:, jnp.newaxis]
        ))
        print(f"  Max |S change| (ocean): {float(S_err):.4e} PSU")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    check_sw_conservation()
    check_ocean_conservation()


def test_shared_core_mpas_fixers_signature_and_conserve():
    """The MPAS mass/energy conservation fixers shared via
    ``legoesm.core.conservation`` (federation dedup) take
    ``(state_new, state_old, mesh, ...)`` and restore the energy / mass of
    ``state_old`` after a perturbation.  The fp64 conservation accumulator keeps
    the energy and mass budgets tight (this also pins the new signature so the
    legacy ``(state, target, mesh)`` shape cannot silently come back)."""
    from legoesm.core.conservation import fix_mass_mpas, fix_energy_mpas

    mesh = create_voronoi_mesh(subdivision_level=2)
    g = constants.g
    state = williamson_test5_mpas(mesh)  # nonzero KE + topography
    diag0 = sw_diagnostics(state, mesh, g)

    bad = state._replace(
        u=state.u.replace(data=state.u.data * 1.10),     # changes KE
        h=state.h.replace(data=state.h.data + 5.0),      # changes mass
    )

    # Energy fixer: rescale u so E(new) matches E(state_old=state).
    e_fixed = fix_energy_mpas(bad, state, mesh, g)
    rel_e = abs(sw_diagnostics(e_fixed, mesh, g)["energy"] - diag0["energy"]) / diag0["energy"]
    assert rel_e < 1e-6, f"energy not restored: rel err {rel_e:.2e}"

    # Mass fixer, anchor-to-target mode.
    m_fixed = fix_mass_mpas(bad, state, mesh, target_mass=diag0["mass"])
    rel_m = abs(sw_diagnostics(m_fixed, mesh, g)["mass"] - diag0["mass"]) / diag0["mass"]
    assert rel_m < 1e-10, f"mass not anchored: rel err {rel_m:.2e}"

    # Mass fixer, match-previous-state mode (target_mass=None).  Tolerance is
    # fp32-realistic: the fixer's internal mass uses the fp64 accumulator, while
    # ``sw_diagnostics`` re-measures in the (fp32) storage dtype — that
    # native-sum rounding, not the fixer, sets the floor here.
    m_fixed2 = fix_mass_mpas(bad, state, mesh)
    rel_m2 = abs(sw_diagnostics(m_fixed2, mesh, g)["mass"] - diag0["mass"]) / diag0["mass"]
    assert rel_m2 < 1e-6, f"mass not matched to state_old: rel err {rel_m2:.2e}"
