"""Diagnostic collection for the composable model driver.

Extracts the diagnostic accumulation, snapshot capture, and output
logic from run_amip.py into a reusable class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.diagnostics.column_integrals import column_water_vapor
from legoesm.diagnostics.energy_budget import EnergyBudgetTracker
from legoesm.forcing.time_utils import day_to_calendar


class DiagnosticCollector:
    """Accumulates diagnostics during a simulation.

    Manages time-series arrays, vertical profiles, 2D snapshots,
    energy budget tracking, and optional monthly-mean accumulation.

    Parameters
    ----------
    nlev : int
        Number of vertical levels.
    sigma_full : array
        Sigma at full levels.
    dsigma : array
        Layer thickness in sigma.
    monthly_means : bool
        Enable monthly-mean zonal diagnostics.
    cmip_output : bool
        Enable CF/CMOR NetCDF output.
    clear_sky_diag : bool
        Track clear-sky radiation diagnostics.
    n_days : int
        Total simulation days (for snapshot selection).
    output_dir : str or Path
        Output directory for diagnostics and CMOR files.
    """

    def __init__(
        self,
        nlev: int,
        sigma_full,
        dsigma,
        monthly_means: bool = False,
        cmip_output: bool = False,
        clear_sky_diag: bool = False,
        n_days: int = 200,
        output_dir: str | Path = "",
    ):
        self.nlev = nlev
        self.sigma_full = sigma_full
        self.dsigma = dsigma
        self.clear_sky_diag = clear_sky_diag

        # Time-series storage
        self.times: list[float] = []
        self.sst: list[float] = []
        self.sic: list[float] = []
        self.T_atm: list[float] = []
        self.T_low: list[float] = []
        self.max_wind: list[float] = []
        self.precip: list[float] = []
        self.CWV: list[float] = []
        self.sw_up_toa: list[float] = []
        self.lw_up_toa: list[float] = []
        self.sw_net_sfc: list[float] = []
        self.lw_net_sfc: list[float] = []
        self.dry_mass: list[float] = []
        self.profiles_T: list[np.ndarray] = []
        self.profiles_qv: list[np.ndarray] = []

        # Energy budget tracker
        self.energy_tracker = EnergyBudgetTracker()

        # Moisture budget tracker
        from legoesm.diagnostics.energy_budget import MoistureBudgetTracker
        self.moisture_tracker = MoistureBudgetTracker()

        # Monthly means
        self.monthly_means = monthly_means
        self.monthly_accum = None
        if monthly_means:
            from legoesm.diagnostics.monthly_means import MonthlyAccumulator
            self.monthly_accum = MonthlyAccumulator(nlev=nlev, n_lat_bins=90)

        # CFWriter
        self.cf_writer = None
        if cmip_output:
            from legoesm.io.cmor_output import CFWriter
            cmor_dir = str(Path(output_dir) / "cmor") if output_dir else "cmor"
            self.cf_writer = CFWriter(
                output_dir=cmor_dir,
                experiment_id="amip",
                model_id="legoESM-1-0",
                freq="mon",
                calendar="noleap",
                ref_date="0001-01-01",
            )
            if not monthly_means:
                from legoesm.diagnostics.monthly_means import MonthlyAccumulator
                self.monthly_means = True
                self.monthly_accum = MonthlyAccumulator(nlev=nlev, n_lat_bins=90)

        # Snapshots
        self.snapshot_days: set[int] = set()
        for d in [5, 10, 15, 20, 25, 30, 60, 100, 200, 300]:
            if d <= n_days:
                self.snapshot_days.add(d)
        if n_days not in self.snapshot_days:
            self.snapshot_days.add(n_days)
        self.snapshots: dict[int, dict[str, np.ndarray]] = {}

    def collect(
        self,
        elapsed_day: float,
        day: float,
        state,
        q_v,
        q_c,
        q_r,
        sst,
        sic,
        precip_total,
        sw_up_toa,
        lw_up_toa,
        sw_net_sfc,
        lw_net_sfc,
        sw_down_toa,
        T_ice: float,
        lat_deg_grid=None,
    ) -> None:
        """Collect diagnostics at a diagnostic interval.

        Parameters
        ----------
        elapsed_day : float
            Days since simulation start.
        day : float
            Absolute simulation day.
        state : HydrostaticState
        q_v, q_c, q_r : jax.Array
        sst, sic : jax.Array
        precip_total : jax.Array
            Total precipitation [kg/m2/s].
        sw_up_toa, lw_up_toa : jax.Array
        sw_net_sfc, lw_net_sfc : jax.Array
        sw_down_toa : jax.Array
        T_ice : float
        lat_deg_grid : array, optional
            Latitude in degrees for monthly means.
        """
        from legoesm.forcing.surface_utils import blend_surface_temperature

        mean_sst = float(jnp.mean(sst))
        mean_sic = float(jnp.mean(sic))
        mean_T = float(jnp.mean(state.T.data))
        mean_T_low = float(jnp.mean(state.T.data[..., -1]))
        if hasattr(state, 'v'):
            max_v = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))
        else:
            max_v = float(jnp.max(jnp.abs(state.u.data)))
        mean_precip = float(jnp.mean(precip_total)) * 86400.0
        cwv = column_water_vapor(q_v, state.p_s.data, self.dsigma)
        mean_cwv = float(jnp.mean(cwv))
        mean_sw_toa = float(jnp.mean(sw_up_toa))
        mean_lw_toa = float(jnp.mean(lw_up_toa))
        mean_ps = float(jnp.mean(state.p_s.data))
        mean_sw_sfc = float(jnp.mean(sw_net_sfc))
        mean_lw_sfc = float(jnp.mean(lw_net_sfc))

        self.times.append(elapsed_day)
        self.sst.append(mean_sst)
        self.sic.append(mean_sic)
        self.T_atm.append(mean_T)
        self.T_low.append(mean_T_low)
        self.max_wind.append(max_v)
        self.precip.append(mean_precip)
        self.CWV.append(mean_cwv)
        self.sw_up_toa.append(mean_sw_toa)
        self.lw_up_toa.append(mean_lw_toa)
        self.sw_net_sfc.append(mean_sw_sfc)
        self.lw_net_sfc.append(mean_lw_sfc)
        self.dry_mass.append(mean_ps)

        self.profiles_T.append(
            np.asarray(jnp.mean(state.T.data, axis=(0, 1, 2)))
        )
        self.profiles_qv.append(
            np.asarray(jnp.mean(q_v, axis=(0, 1, 2))) * 1000.0
        )

        # Snapshots
        iday = int(round(elapsed_day))
        if iday in self.snapshot_days:
            T_sfc_snap = blend_surface_temperature(sst, sic, T_ice)
            self.snapshots[iday] = {
                'SST': np.asarray(sst),
                'SIC': np.asarray(sic),
                'T_sfc': np.asarray(T_sfc_snap),
                'T_low': np.asarray(state.T.data[..., -1]),
                'q_v_low': np.asarray(q_v[..., -1]) * 1000.0,
                'precip': np.asarray(precip_total) * 86400.0,
                'wind': np.asarray(
                    jnp.sqrt(state.u.data[..., -1] ** 2 + state.v.data[..., -1] ** 2)
                ),
            }

        # Energy budget
        elapsed_s = elapsed_day * 86400.0
        self.energy_tracker.update(
            state.T.data, q_v, state.u.data, state.v.data,
            state.phis.data, state.p_s.data,
            self.dsigma, self.sigma_full,
            sw_down_toa, sw_up_toa, lw_up_toa, sw_net_sfc, lw_net_sfc,
            elapsed_seconds=elapsed_s,
        )

        # Moisture budget
        self.moisture_tracker.update(
            q_v, state.p_s.data, self.dsigma,
            precip_total, elapsed_seconds=elapsed_s,
        )

        # Monthly means
        if self.monthly_means and self.monthly_accum is not None and lat_deg_grid is not None:
            doy, _ = day_to_calendar(day)
            year = int(day // 365.0)
            fields_2d = {
                'T_low': np.asarray(state.T.data[..., -1]),
                'precip': np.asarray(precip_total) * 86400.0,
                'sw_up_toa': np.asarray(sw_up_toa),
                'lw_up_toa': np.asarray(lw_up_toa),
                'sw_net_sfc': np.asarray(sw_net_sfc),
                'lw_net_sfc': np.asarray(lw_net_sfc),
            }
            self.monthly_accum.add_2d(doy, year, fields_2d, lat_deg_grid)
            self.monthly_accum.add_3d(doy, year, {
                'T': np.asarray(state.T.data),
                'u': np.asarray(state.u.data),
                'q_v': np.asarray(q_v) * 1000.0,
            }, lat_deg_grid)
            ebudget = self.energy_tracker
            self.monthly_accum.add_scalar(doy, year, {
                'T_atm': mean_T,
                'T_low': mean_T_low,
                'precip': mean_precip,
                'sw_up_toa': mean_sw_toa,
                'lw_up_toa': mean_lw_toa,
            })

        return {
            'mean_sst': mean_sst,
            'mean_sic': mean_sic,
            'mean_T': mean_T,
            'mean_T_low': mean_T_low,
            'max_v': max_v,
            'mean_precip': mean_precip,
            'mean_cwv': mean_cwv,
            'mean_sw_toa': mean_sw_toa,
            'mean_lw_toa': mean_lw_toa,
            'mean_sw_sfc': mean_sw_sfc,
            'mean_lw_sfc': mean_lw_sfc,
        }

    def flush_to_disk(self, output_dir: str | Path) -> None:
        """Flush accumulated timeseries to disk and clear in-memory lists.

        Appends to ``timeseries_incremental/`` directory as numbered
        chunks.  Called periodically (e.g. annually) during long runs
        to prevent unbounded memory growth.
        """
        output_dir = Path(output_dir)
        incr_dir = output_dir / "timeseries_incremental"
        incr_dir.mkdir(parents=True, exist_ok=True)

        # Find next chunk number
        existing = sorted(incr_dir.glob("chunk_*.npz"))
        chunk_idx = len(existing)

        if not self.times:
            return  # nothing to flush

        sigma = np.asarray(self.sigma_full)

        # Save current batch
        np.savez(
            incr_dir / f"chunk_{chunk_idx:04d}.npz",
            days=np.array(self.times),
            T_atm=np.array(self.T_atm),
            T_low=np.array(self.T_low),
            max_wind=np.array(self.max_wind),
            precip=np.array(self.precip),
            CWV=np.array(self.CWV),
            sw_up_toa=np.array(self.sw_up_toa),
            lw_up_toa=np.array(self.lw_up_toa),
            sw_net_sfc=np.array(self.sw_net_sfc),
            lw_net_sfc=np.array(self.lw_net_sfc),
            dry_mass_ps=np.array(self.dry_mass),
            profiles_T=np.array(self.profiles_T) if self.profiles_T else np.array([]),
            profiles_qv=np.array(self.profiles_qv) if self.profiles_qv else np.array([]),
        )

        # Also flush energy tracker
        energy_data = self.energy_tracker.flush_to_lists()
        if energy_data["times"]:
            np.savez(
                incr_dir / f"energy_chunk_{chunk_idx:04d}.npz",
                **{k: np.array(v) for k, v in energy_data.items()},
            )

        # Clear all timeseries lists (snapshots and monthly accum preserved)
        self.times.clear()
        self.sst.clear()
        self.sic.clear()
        self.T_atm.clear()
        self.T_low.clear()
        self.max_wind.clear()
        self.precip.clear()
        self.CWV.clear()
        self.sw_up_toa.clear()
        self.lw_up_toa.clear()
        self.sw_net_sfc.clear()
        self.lw_net_sfc.clear()
        self.dry_mass.clear()
        self.profiles_T.clear()
        self.profiles_qv.clear()

        # Also flush moisture tracker
        moisture_data = self.moisture_tracker.flush_to_lists()
        if moisture_data["times"]:
            np.savez(
                incr_dir / f"moisture_chunk_{chunk_idx:04d}.npz",
                **{k: np.array(v) for k, v in moisture_data.items()},
            )

    def save(self, output_dir: str | Path) -> None:
        """Save all accumulated diagnostics to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sigma = np.asarray(self.sigma_full)

        np.savez(
            output_dir / "timeseries.npz",
            days=np.array(self.times),
            sst=np.array(self.sst),
            sic=np.array(self.sic),
            T_atm=np.array(self.T_atm),
            T_low=np.array(self.T_low),
            max_wind=np.array(self.max_wind),
            precip=np.array(self.precip),
            CWV=np.array(self.CWV),
            sw_up_toa=np.array(self.sw_up_toa),
            lw_up_toa=np.array(self.lw_up_toa),
            sw_net_sfc=np.array(self.sw_net_sfc),
            lw_net_sfc=np.array(self.lw_net_sfc),
            dry_mass_ps=np.array(self.dry_mass),
            sigma=sigma,
            profiles_T=np.array(self.profiles_T) if self.profiles_T else np.array([]),
            profiles_qv=np.array(self.profiles_qv) if self.profiles_qv else np.array([]),
            energy_toa_net=np.array(self.energy_tracker.toa_net),
            energy_column=np.array(self.energy_tracker.column_energy),
            energy_dE_dt=np.array(self.energy_tracker.dE_dt),
            energy_residual=np.array(self.energy_tracker.residual),
            moisture_column_water=np.array(self.moisture_tracker.column_water),
            moisture_precip_rate=np.array(self.moisture_tracker.precip_rate),
            moisture_residual=np.array(self.moisture_tracker.residual),
        )

        if self.monthly_means and self.monthly_accum is not None:
            self.monthly_accum.save(
                output_dir / "monthly_means.npz",
                sigma=sigma,
            )

        if self.cf_writer is not None:
            self.cf_writer.close()

    def print_summary(self) -> str:
        """Return end-of-run summary string."""
        parts = [self.energy_tracker.summary()]
        if len(self.moisture_tracker.times) >= 2:
            parts.append(self.moisture_tracker.summary())
        return "\n\n".join(parts)

    def check_stability(self, state, elapsed_day: float) -> str | None:
        """Check for blow-up conditions.

        Returns
        -------
        str or None
            Error message if blow-up detected, None if stable.
        """
        if not jnp.all(jnp.isfinite(state.u.data)):
            return f"BLOWUP at day {elapsed_day:.0f}: non-finite winds"
        if hasattr(state, 'v'):
            max_v = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))
        else:
            max_v = float(jnp.max(jnp.abs(state.u.data)))
        if max_v > 500:
            return f"BLOWUP at day {elapsed_day:.0f}: max wind {max_v:.1f} m/s"
        if not jnp.all(jnp.isfinite(state.T.data)):
            return f"BLOWUP at day {elapsed_day:.0f}: non-finite T"

        # Temperature bounds (physical range for Earth atmosphere)
        T_min_val = float(jnp.min(state.T.data))
        T_max_val = float(jnp.max(state.T.data))
        if T_min_val < 100.0 or T_max_val > 400.0:
            return (
                f"BLOWUP at day {elapsed_day:.0f}: temperature out of physical bounds "
                f"(min={T_min_val:.1f}K, max={T_max_val:.1f}K). "
                f"Check dt, hyperdiffusion, and physics configuration."
            )

        # Surface pressure bounds
        if hasattr(state, 'p_s'):
            ps_min = float(jnp.min(state.p_s.data))
            ps_max = float(jnp.max(state.p_s.data))
            if ps_min < 40000.0 or ps_max > 115000.0:
                return (
                    f"BLOWUP at day {elapsed_day:.0f}: surface pressure out of bounds "
                    f"(min={ps_min:.0f}Pa, max={ps_max:.0f}Pa). "
                    f"Check dt and dynamics configuration."
                )

        return None


class EnsembleDiagnosticCollector:
    """Collects ensemble-specific diagnostics: spread, CRPS, rank histograms.

    Wraps a base ``DiagnosticCollector`` (which handles ensemble-mean diagnostics)
    and adds per-member spread tracking, per-member checkpointing, and
    probabilistic verification scores.

    Parameters
    ----------
    base_collector : DiagnosticCollector
        The base collector (operates on ensemble-mean state).
    n_members : int
        Number of ensemble members.
    """

    def __init__(self, base_collector: DiagnosticCollector, n_members: int):
        self.base = base_collector
        self.n_members = n_members

        # Time-series of ensemble spread
        self.spread_T: list[float] = []
        self.spread_u: list[float] = []
        self.spread_ps: list[float] = []
        self.spread_times: list[float] = []

    def collect_ensemble(
        self,
        elapsed_day: float,
        full_carry,
        mean_carry,
    ) -> dict[str, float]:
        """Collect ensemble spread diagnostics from the full ensemble carry.

        Parameters
        ----------
        elapsed_day : float
            Days since simulation start.
        full_carry : SegmentCarry with leading (n_members, ...) dimension
            Full ensemble state.
        mean_carry : SegmentCarry without ensemble dimension
            Ensemble mean (for base collector).

        Returns
        -------
        dict with spread metrics.
        """
        from legoesm.parallel.ensemble import ensemble_spread

        spread = ensemble_spread(full_carry)
        self.spread_times.append(elapsed_day)
        self.spread_T.append(spread.get('T', 0.0))
        self.spread_u.append(spread.get('u', 0.0))
        self.spread_ps.append(spread.get('p_s', 0.0))

        return {
            'spread_T': spread.get('T', 0.0),
            'spread_u': spread.get('u', 0.0),
            'spread_ps': spread.get('p_s', 0.0),
        }

    def save_member_checkpoint(
        self,
        full_carry,
        step: int,
        day: float,
        output_dir,
        config,
    ) -> None:
        """Save per-member checkpoints for the full ensemble.

        Each member is saved as a separate checkpoint file:
        ``ensemble_member_NNN_day_DDDD.npz``.
        """
        from legoesm.io.restart import save_restart
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState

        output_dir = Path(output_dir)
        n = self.n_members

        for m in range(n):
            # Extract single member from batched carry
            member_carry = jax.tree.map(lambda x: x[m], full_carry)

            # Build a minimal state for save_restart
            dims_3d = ("face", "x", "y", "level")
            dims_2d = ("face", "x", "y")
            member_state = HydrostaticState(
                u=Field(member_carry.u, name="u", dims=dims_3d, units="m/s"),
                v=Field(member_carry.v, name="v", dims=dims_3d, units="m/s"),
                T=Field(member_carry.T, name="T", dims=dims_3d, units="K"),
                p_s=Field(member_carry.p_s, name="p_s", dims=dims_2d, units="Pa"),
                phis=Field(member_carry.phis, name="phis", dims=dims_2d, units="m2/s2"),
            )

            elapsed = day - config.start_day
            path = output_dir / f"ensemble_member_{m:03d}_day_{int(elapsed):04d}.npz"
            save_restart(
                path=path,
                state=member_state,
                q_v=member_carry.q_v,
                step=step,
                day=day,
                config=config,
                q_c=member_carry.q_c,
                q_r=member_carry.q_r,
            )

    def save(self, output_dir) -> None:
        """Save ensemble diagnostics alongside base diagnostics."""
        output_dir = Path(output_dir)

        # Save spread timeseries
        if self.spread_times:
            np.savez(
                output_dir / "ensemble_spread.npz",
                days=np.array(self.spread_times),
                spread_T=np.array(self.spread_T),
                spread_u=np.array(self.spread_u),
                spread_ps=np.array(self.spread_ps),
            )

        # Base diagnostics
        self.base.save(output_dir)
