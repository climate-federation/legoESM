"""Online diagnostics for legoESM."""

from legoesm.diagnostics.angular_momentum import (
    aam_drift_nh,
    aam_drift_pe,
    aam_from_nh_state,
    aam_from_pe_state,
    apply_aam_correction_nh,
    apply_aam_correction_pe,
    compute_atmospheric_angular_momentum,
)
from legoesm.diagnostics.total_energy_nh import (
    apply_te_correction_nh,
    compute_total_energy_nh,
    te_drift_nh,
)
from legoesm.diagnostics.total_energy_pe import (
    apply_te_correction_pe,
    compute_total_energy_pe,
    te_drift_pe,
)
from legoesm.diagnostics.cloud_overlap import maximum_random_overlap
from legoesm.diagnostics.column_integrals import (
    column_d_ext_field,
    column_mass_weighted_mean,
    column_water_vapor,
)
from legoesm.diagnostics.conservation_drift import (
    DAYS_REQUIRED,
    DEFAULT_MIN_BASELINE,
    apply_drift_tolerance,
    apply_value_threshold,
    compute_relative_drift,
    relative_drift_series,
)
from legoesm.diagnostics.energy_budget import (
    EnergyBudget,
    EnergyBudgetTracker,
    area_weighted_mean,
    area_weighted_profile,
    column_dry_static_energy,
    column_moist_static_energy,
    surface_energy_flux,
    surface_net_radiation,
    toa_net_radiation,
)
from legoesm.diagnostics.monthly_means import MonthlyAccumulator
from legoesm.diagnostics.precision_drift import (
    DriftSnapshot,
    PrecisionDriftChecker,
    compare_states,
    precision_health_report,
    check_tracer_negativity,
)
