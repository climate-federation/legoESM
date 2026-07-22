"""
Soil Temperature Calculations.

Translated from CTSM's SoilTemperatureMod.F90

This module calculates soil temperature using a simplified approach from CLM5
that is compatible with CLMml. It solves the heat diffusion equation in soil
layers using thermal conductivity and heat capacity.

Key components:
    - soil_temperature: Main routine to compute soil temperature profile
    - soil_therm_prop: Calculate thermal conductivity and heat capacity
    - tridiag: Solve tridiagonal system for temperature updates

Physical basis:
    Heat diffusion equation:
        ∂T/∂t = ∂/∂z(κ ∂T/∂z) + S

    Where:
        - T: Temperature [K]
        - κ: Thermal diffusivity [m²/s]
        - z: Depth [m]
        - S: Source/sink terms [K/s]

Thermal conductivity:
    Soil (Johansen algorithm, Farouki 1981):
        thk = dke*dksat + (1-dke)*tkdry
        dksat = tkmg * tkwat^(fl*watsat) * tkice^((1-fl)*watsat)
        dke = log10(satw) + 1.0  (unfrozen)
        dke = satw               (frozen)

    Snow (Jordan 1991):
        thk = tkair + (7.75e-5*bw + 1.105e-6*bw^2)*(tkice-tkair)

    Interface (harmonic mean):
        tk(j) = thk(j)*thk(j+1)*(z(j+1)-z(j)) /
                (thk(j)*(z(j+1)-zi(j)) + thk(j+1)*(zi(j)-z(j)))

Heat capacity:
    cv = csol*(1-watsat)*dz + h2osoi_ice*cpice + h2osoi_liq*cpliq

Reference: SoilTemperatureMod.F90, lines 1-398
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

# ============================================================================
# Module Constants
# ============================================================================

# Threshold for thin surface layer [m] (line 30)
THIN_SFCLAYER = 1.0e-6


# ============================================================================
# Type Definitions
# ============================================================================


class SoilTemperatureParams(NamedTuple):
    """Parameters for soil temperature calculations.

    Attributes:
        thin_sfclayer: Threshold for thin surface layer [m]
        denh2o: Density of liquid water [kg/m3]
        denice: Density of ice [kg/m3]
        tfrz: Freezing temperature [K]
        tkwat: Thermal conductivity of water [W/m/K]
        tkice: Thermal conductivity of ice [W/m/K]
        tkair: Thermal conductivity of air [W/m/K]
        cpice: Specific heat of ice [J/kg/K]
        cpliq: Specific heat of liquid water [J/kg/K]
        thk_bedrock: Thermal conductivity of bedrock [W/m/K]
        csol_bedrock: Heat capacity of bedrock [J/m3/K]
    """

    thin_sfclayer: float = THIN_SFCLAYER
    denh2o: float = 1000.0
    denice: float = 917.0
    tfrz: float = 273.15
    tkwat: float = 0.57
    tkice: float = 2.29
    tkair: float = 0.023
    cpice: float = 2.11727e3
    cpliq: float = 4.188e3
    thk_bedrock: float = 3.0
    csol_bedrock: float = 2.0e6


class ColumnGeometry(NamedTuple):
    """Column geometry information.

    Attributes:
        dz: Layer thickness [m] [n_cols, n_levgrnd]
        z: Layer depth (center) [m] [n_cols, n_levgrnd]
        zi: Layer depth (interface) [m] [n_cols, n_levgrnd+1]
        snl: Number of snow layers (negative) [n_cols]
        nbedrock: Index of bedrock layer [n_cols]
    """

    dz: jnp.ndarray
    z: jnp.ndarray
    zi: jnp.ndarray
    snl: jnp.ndarray
    nbedrock: jnp.ndarray


class SoilState(NamedTuple):
    """Soil state variables.

    Attributes:
        tkmg: Thermal conductivity of soil minerals [W/m/K] [n_cols, n_levgrnd]
        tkdry: Thermal conductivity of dry soil [W/m/K] [n_cols, n_levgrnd]
        csol: Heat capacity of soil solids [J/m3/K] [n_cols, n_levgrnd]
        watsat: Volumetric soil water at saturation [m3/m3] [n_cols, n_levgrnd]
    """

    tkmg: jnp.ndarray
    tkdry: jnp.ndarray
    csol: jnp.ndarray
    watsat: jnp.ndarray


class WaterState(NamedTuple):
    """Water state variables.

    Attributes:
        h2osoi_liq: Liquid water content [kg/m2] [n_cols, n_levgrnd]
        h2osoi_ice: Ice content [kg/m2] [n_cols, n_levgrnd]
        h2osfc: Surface water depth [mm] [n_cols]
        h2osno: Snow water equivalent [kg/m2] [n_cols]
        frac_sno_eff: Effective snow cover fraction [n_cols]
    """

    h2osoi_liq: jnp.ndarray
    h2osoi_ice: jnp.ndarray
    h2osfc: jnp.ndarray
    h2osno: jnp.ndarray
    frac_sno_eff: jnp.ndarray


class ThermalProperties(NamedTuple):
    """Thermal properties output.

    Attributes:
        tk: Thermal conductivity at layer interface [W/m/K] [n_cols, n_levgrnd]
        cv: Heat capacity [J/m2/K] [n_cols, n_levgrnd]
        tk_h2osfc: Thermal conductivity of surface water [W/m/K] [n_cols]
        thk: Thermal conductivity of each layer [W/m/K] [n_cols, n_levgrnd]
        bw: Partial density of water in snow pack [kg/m3] [n_cols, n_levgrnd]
    """

    tk: jnp.ndarray
    cv: jnp.ndarray
    tk_h2osfc: jnp.ndarray
    thk: jnp.ndarray
    bw: jnp.ndarray


class SoilTemperatureResult(NamedTuple):
    """Results from soil temperature calculation.

    Attributes:
        t_soisno: Updated soil temperature [K] [n_cols, n_levgrnd]
        energy_error: Energy conservation error [W/m2] [n_cols]
    """

    t_soisno: jnp.ndarray
    energy_error: jnp.ndarray


# ============================================================================
# Tridiagonal System Solver
# ============================================================================


def tridiag(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    r: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a tridiagonal system of equations.

    Implements the Thomas algorithm for solving F x U = R where F is a
    tridiagonal matrix. This is a stable O(n) algorithm that performs
    forward elimination followed by backward substitution.

    The tridiagonal matrix F is defined by vectors A, B, C:
        | b(1) c(1)   0  ...                      |   | u(1)   |   | r(1)   |
        | a(2) b(2) c(2) ...                      |   | u(2)   |   | r(2)   |
        |                ...                      | x | ...    | = | ...    |
        |                ... a(n-1) b(n-1) c(n-1) |   | u(n-1) |   | r(n-1) |
        |                ...   0    a(n)   b(n)   |   | u(n)   |   | r(n)   |

    Args:
        a: Lower diagonal coefficients [n]. Note: a[0] is not used.
        b: Main diagonal coefficients [n]
        c: Upper diagonal coefficients [n]. Note: c[-1] is not used.
        r: Right-hand side vector [n]

    Returns:
        u: Solution vector [n]

    Reference:
        SoilTemperatureMod.F90, lines 349-396

    Note:
        The algorithm is vectorized using jax.lax.scan for JIT compatibility.
    """
    n = a.shape[0]

    # Initialize arrays
    gam = jnp.zeros(n)
    u = jnp.zeros(n)

    # Forward elimination (lines 387-391)
    # First step: bet = b(1), u(1) = r(1) / bet
    bet = b[0]
    u = u.at[0].set(r[0] / bet)

    def forward_step(
        carry: Tuple[jnp.ndarray, float], j: int
    ) -> Tuple[Tuple[jnp.ndarray, float], jnp.ndarray]:
        """Forward elimination step for layer j.

        Args:
            carry: Tuple of (gam, bet) from previous step
            j: Current layer index (1-based, so actual index is j)

        Returns:
            Updated (gam, bet) and u_j value
        """
        gam_arr, bet_prev = carry

        # gam(j) = c(j-1) / bet
        gam_j = c[j - 1] / bet_prev
        gam_arr = gam_arr.at[j].set(gam_j)

        # bet = b(j) - a(j) * gam(j)
        bet_new = b[j] - a[j] * gam_j

        # u(j) = (r(j) - a(j) * u(j-1)) / bet
        u_j = (r[j] - a[j] * u[j - 1]) / bet_new

        return (gam_arr, bet_new), u_j

    # Run forward elimination for layers 1 to n-1 (indices 1 to n-1)
    if n > 1:
        (gam, _), u_forward = lax.scan(forward_step, (gam, bet), jnp.arange(1, n))
        # Update u with forward elimination results
        u = u.at[1:].set(u_forward)
    else:
        # Single layer case - already solved
        return u

    # Backward substitution (lines 392-394)
    def backward_step(u_arr: jnp.ndarray, j: int) -> Tuple[jnp.ndarray, None]:
        """Backward substitution step for layer j.

        Args:
            u_arr: Current solution vector
            j: Current layer index (counting backwards from n-2 to 0)

        Returns:
            Updated u_arr and None (no output to collect)
        """
        # u(j) = u(j) - gam(j+1) * u(j+1)
        u_j = u_arr[j] - gam[j + 1] * u_arr[j + 1]
        u_arr = u_arr.at[j].set(u_j)
        return u_arr, None

    # Run backward substitution for layers n-2 down to 0 (indices n-2 to 0)
    u, _ = lax.scan(backward_step, u, jnp.arange(n - 2, -1, -1))

    return u


# ============================================================================
# Thermal Properties
# ============================================================================


def soil_therm_prop(
    geom: ColumnGeometry,
    t_soisno: jnp.ndarray,
    water: WaterState,
    soil: SoilState,
    params: SoilTemperatureParams,
    nlevgrnd: int,
) -> ThermalProperties:
    """Calculate thermal conductivities and heat capacities of soil layers.

    Implements the Johansen algorithm for soil thermal conductivity (Farouki 1981).

    Args:
        geom: Column geometry information
        t_soisno: Soil temperature [K] [n_cols, n_levgrnd]
        water: Water state variables (all arrays [n_cols, n_levgrnd])
        soil: Soil state variables (all arrays [n_cols, n_levgrnd])
        params: Soil temperature parameters
        nlevgrnd: Number of ground layers

    Returns:
        ThermalProperties with tk, cv, tk_h2osfc, thk, bw

    Reference: Fortran lines 188-346

    Note:
        This function operates only on soil layers. Input arrays should have
        shape [n_cols, n_levgrnd], not including snow layers.
    """
    n_cols = geom.dz.shape[0]

    # Initialize outputs - all for soil layers only
    thk = jnp.zeros((n_cols, nlevgrnd))
    bw = jnp.zeros((n_cols, nlevgrnd))
    tk = jnp.zeros((n_cols, nlevgrnd))
    cv = jnp.zeros((n_cols, nlevgrnd))

    # Extract needed variables
    nbedrock = geom.nbedrock
    snl = geom.snl
    dz = geom.dz
    zi = geom.zi
    z = geom.z
    h2osfc = water.h2osfc
    h2osno = water.h2osno
    h2osoi_liq = water.h2osoi_liq
    h2osoi_ice = water.h2osoi_ice
    tkmg = soil.tkmg
    tkdry = soil.tkdry
    csol = soil.csol
    watsat = soil.watsat

    # Soil thermal conductivity (Fortran lines 221-241)
    # Calculate saturation (lines 222-223)
    satw = (h2osoi_liq / params.denh2o + h2osoi_ice / params.denice) / (dz * watsat + 1e-30)
    satw = jnp.minimum(1.0, satw)

    # Calculate Kersten number (lines 224-229)
    # Use natural log instead of log10 to avoid JAX compatibility issues: log10(x) = ln(x) / ln(10)
    satw_safe = satw + 1e-30
    ln10 = 2.302585093  # ln(10)
    dke_unfrozen = jnp.maximum(0.0, jnp.log(satw_safe) / ln10 + 1.0)
    dke_frozen = satw
    dke = jnp.where(t_soisno >= params.tfrz, dke_unfrozen, dke_frozen)

    # Calculate liquid fraction (lines 230-231)
    liq_vol = h2osoi_liq / (params.denh2o * dz + 1e-30)
    ice_vol = h2osoi_ice / (params.denice * dz + 1e-30)
    fl = liq_vol / (liq_vol + ice_vol + 1e-30)

    # Calculate saturated thermal conductivity (line 232)
    dksat = tkmg * (params.tkwat ** (fl * watsat)) * (params.tkice ** ((1.0 - fl) * watsat))

    # Calculate thermal conductivity (lines 233-236)
    thk_soil = dke * dksat + (1.0 - dke) * tkdry
    thk_soil = jnp.where(satw > 0.1e-6, thk_soil, tkdry)

    # Apply bedrock conductivity (line 237)
    # Create layer indices (1-based for comparison with nbedrock)
    j_indices = jnp.arange(1, nlevgrnd + 1)
    is_bedrock = j_indices[None, :] > nbedrock[:, None]
    thk = jnp.where(is_bedrock, params.thk_bedrock, thk_soil)

    # Thermal conductivity at layer interface (Fortran lines 250-259)
    # Use harmonic mean for interface conductivity
    # For layers 0 to nlevgrnd-2 (Python indexing)
    thk_j = thk[:, :-1]
    thk_jp1 = thk[:, 1:]
    z_j = z[:, :-1]
    z_jp1 = z[:, 1:]
    zi_j = zi[:, 1:-1]  # Interface depths between layers

    tk_interface = (thk_j * thk_jp1 * (z_jp1 - z_j)) / (
        thk_j * (z_jp1 - zi_j) + thk_jp1 * (zi_j - z_j) + 1e-30
    )

    # Set interface conductivity for layers 0 to nlevgrnd-2
    tk = tk.at[:, :-1].set(tk_interface)

    # Bottom boundary (lines 255-256) - last layer has zero flux
    tk = tk.at[:, -1].set(0.0)

    # Calculate thermal conductivity of h2osfc (Fortran lines 262-266)
    zh2osfc = 1.0e-3 * (0.5 * h2osfc)  # Convert mm to m
    tk_h2osfc = (params.tkwat * thk[:, 0] * (z[:, 0] + zh2osfc)) / (
        params.tkwat * z[:, 0] + thk[:, 0] * zh2osfc + 1e-30
    )

    # Soil heat capacity (Fortran lines 270-279)
    # Basic heat capacity (line 273)
    cv = csol * (1.0 - watsat) * dz + (h2osoi_ice * params.cpice + h2osoi_liq * params.cpliq)

    # Bedrock heat capacity (line 274)
    cv = jnp.where(is_bedrock, params.csol_bedrock * dz, cv)

    # Add snow water heat capacity to top layer (lines 275-278)
    # Only if no snow layers (snl+1 == 1) and h2osno > 0
    add_snow_heat = (snl + 1 == 1) & (h2osno > 0.0)
    cv_top = jnp.where(add_snow_heat, cv[:, 0] + params.cpice * h2osno, cv[:, 0])
    cv = cv.at[:, 0].set(cv_top)

    return ThermalProperties(
        tk=tk,
        cv=cv,
        tk_h2osfc=tk_h2osfc,
        thk=thk,
        bw=bw,
    )


# ============================================================================
# Main Soil Temperature Calculation
# ============================================================================


def soil_temperature(
    geom: ColumnGeometry,
    t_soisno: jnp.ndarray,
    gsoi: jnp.ndarray,
    thermal_props: ThermalProperties,
    dtime: float,
    nlevgrnd: int,
) -> SoilTemperatureResult:
    """Compute soil temperature using implicit finite difference method.

    Solves the 1D heat diffusion equation in soil layers using a tridiagonal
    matrix approach. The system is:

        A[j] * T[j-1] + B[j] * T[j] + C[j] * T[j+1] = R[j]

    for j = 1 to nlevgrnd.

    Args:
        geom: Column geometry information
        t_soisno: Soil temperature at start of timestep [K] [n_cols, n_levgrnd]
        gsoi: Ground heat flux (positive downward) [W/m2] [n_cols]
        thermal_props: Thermal properties (tk, cv)
        dtime: Model timestep [s]
        nlevgrnd: Number of ground layers

    Returns:
        SoilTemperatureResult containing:
            - t_soisno: Updated soil temperature [K] [n_cols, n_levgrnd]
            - energy_error: Energy conservation error [W/m2] [n_cols]

    Note:
        Fortran source lines 36-185 in SoilTemperatureMod.F90

        The tridiagonal coefficients are:
        - Layer 1 (top): Boundary condition from gsoi
        - Layers 2 to nlevgrnd-1: Interior layers
        - Layer nlevgrnd (bottom): Zero flux boundary condition
    """
    n_cols = t_soisno.shape[0]

    # Save initial temperature for energy conservation check (line 82-87)
    tssbef = t_soisno

    # Extract thermal properties
    cv = thermal_props.cv
    tk = thermal_props.tk
    z = geom.z

    # Compute fact for all layers (line 99)
    fact = dtime / cv  # [n_cols, nlevgrnd]

    # Compute layer thicknesses
    dzm = jnp.zeros_like(z)
    dzp = jnp.zeros_like(z)

    # dzm[j] = z[j] - z[j-1] for j > 0
    dzm = dzm.at[:, 1:].set(z[:, 1:] - z[:, :-1])

    # dzp[j] = z[j+1] - z[j] for j < nlevgrnd-1
    dzp = dzp.at[:, :-1].set(z[:, 1:] - z[:, :-1])

    # Initialize tridiagonal arrays
    atri = jnp.zeros((n_cols, nlevgrnd))
    btri = jnp.zeros((n_cols, nlevgrnd))
    ctri = jnp.zeros((n_cols, nlevgrnd))
    rtri = jnp.zeros((n_cols, nlevgrnd))

    # Top layer (j=0) (lines 100-107)
    atri_top = jnp.zeros((n_cols, 1))
    btri_top = 1.0 + fact[:, 0:1] * tk[:, 0:1] / (dzp[:, 0:1] + 1e-30)
    ctri_top = -fact[:, 0:1] * tk[:, 0:1] / (dzp[:, 0:1] + 1e-30)
    rtri_top = t_soisno[:, 0:1] + fact[:, 0:1] * gsoi[:, None]

    # Interior layers (0 < j < nlevgrnd-1) (lines 109-117)
    if nlevgrnd > 2:
        atri_interior = -fact[:, 1:-1] * tk[:, :-2] / (dzm[:, 1:-1] + 1e-30)
        btri_interior = 1.0 + fact[:, 1:-1] * (
            tk[:, :-2] / (dzm[:, 1:-1] + 1e-30) + tk[:, 1:-1] / (dzp[:, 1:-1] + 1e-30)
        )
        ctri_interior = -fact[:, 1:-1] * tk[:, 1:-1] / (dzp[:, 1:-1] + 1e-30)
        rtri_interior = t_soisno[:, 1:-1]
    else:
        atri_interior = jnp.zeros((n_cols, 0))
        btri_interior = jnp.zeros((n_cols, 0))
        ctri_interior = jnp.zeros((n_cols, 0))
        rtri_interior = jnp.zeros((n_cols, 0))

    # Bottom layer (j=nlevgrnd-1) (lines 119-127)
    atri_bottom = -fact[:, -1:] * tk[:, -2:-1] / (dzm[:, -1:] + 1e-30)
    btri_bottom = 1.0 + fact[:, -1:] * tk[:, -2:-1] / (dzm[:, -1:] + 1e-30)
    ctri_bottom = jnp.zeros((n_cols, 1))
    rtri_bottom = t_soisno[:, -1:]

    # Assemble tridiagonal matrix
    atri = jnp.concatenate([atri_top, atri_interior, atri_bottom], axis=1)
    btri = jnp.concatenate([btri_top, btri_interior, btri_bottom], axis=1)
    ctri = jnp.concatenate([ctri_top, ctri_interior, ctri_bottom], axis=1)
    rtri = jnp.concatenate([rtri_top, rtri_interior, rtri_bottom], axis=1)

    # Solve tridiagonal system (lines 141-145)
    t_soisno_new = jax.vmap(tridiag)(atri, btri, ctri, rtri)

    # Energy conservation check (lines 147-158)
    edif = jnp.sum(cv * (t_soisno_new - tssbef), axis=1) / dtime
    energy_error = jnp.abs(gsoi - edif)

    return SoilTemperatureResult(
        t_soisno=t_soisno_new,
        energy_error=energy_error,
    )


# ============================================================================
# Combined Workflow Function
# ============================================================================


def compute_soil_temperature(
    geom: ColumnGeometry,
    t_soisno: jnp.ndarray,
    gsoi: jnp.ndarray,
    water: WaterState,
    soil: SoilState,
    params: SoilTemperatureParams,
    dtime: float,
    nlevsno: int,
    nlevgrnd: int,
) -> Tuple[SoilTemperatureResult, ThermalProperties]:
    """Complete soil temperature calculation workflow.

    This function combines thermal property calculation and temperature
    update into a single workflow.

    Args:
        geom: Column geometry information (arrays have shape [n_cols, n_levgrnd])
        t_soisno: Initial soil temperature [K] [n_cols, n_levgrnd]
        gsoi: Ground heat flux [W/m2] [n_cols]
        water: Water state variables (arrays have shape [n_cols, n_levgrnd])
        soil: Soil state variables (arrays have shape [n_cols, n_levgrnd])
        params: Soil temperature parameters
        dtime: Model timestep [s]
        nlevsno: Maximum number of snow layers (not used in current implementation)
        nlevgrnd: Number of ground layers

    Returns:
        Tuple of (SoilTemperatureResult, ThermalProperties)

    Note:
        This function operates only on soil layers. All input arrays should
        have shape [n_cols, n_levgrnd], not including snow layers.
    """
    # Calculate thermal properties
    thermal_props = soil_therm_prop(
        geom=geom,
        t_soisno=t_soisno,
        water=water,
        soil=soil,
        params=params,
        nlevgrnd=nlevgrnd,
    )

    # Update soil temperature
    result = soil_temperature(
        geom=geom,
        t_soisno=t_soisno,
        gsoi=gsoi,
        thermal_props=thermal_props,
        dtime=dtime,
        nlevgrnd=nlevgrnd,
    )

    return result, thermal_props  # Backward compatibility aliases


SoilTemperature = soil_temperature
SoilThermProp = compute_soil_temperature  # Alias for thermal properties computation
