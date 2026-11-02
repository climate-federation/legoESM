"""MARBL vertical mixing on the partial-cell geometry (conserves sum(h * c)).
For the MARBL tracers only: actual layer thicknesses (partial cells), K = 0 at every
interface touching an inactive cell, inactive cells left untouched.  T, S, momentum unchanged.
Backup: ocean_model_latlon_cgrid.py.bak_vmix"""
import sys
from pathlib import Path

P = Path("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py")
if len(sys.argv) > 1:
    P = Path(sys.argv[1])
OLD = '''            def _mix(f):
                f_new = implicit_vertical_diffusion_ocean(
                    f, K_v_cell.astype(f.dtype), dz_cell.astype(f.dtype),
                    dz_half_cell.astype(f.dtype), dt,
                )
                return jnp.where(mask_3d > 0.5, f_new, f)
'''
NEW = '''            # Partial-cell geometry, as in the tracer advection: actual layer
            # thicknesses and no mixing across the sea floor -> sum(h * c) conserved.
            _act = (self.z_coord.is_active if hasattr(self.z_coord, "is_active")
                    else jnp.broadcast_to(mask_3d, state.T.data.shape)) > 0.5
            _act = _act & (mask_3d > 0.5)
            _h = compute_layer_thickness(state.eta.data, state.H_bathy.data, self.z_coord,
                                         min_water_column_m=self.config.min_water_column_m)
            _dz_m = jnp.where(_act, jnp.maximum(_h, 1e-3), dz_cell)
            _dzh_m = build_dz_half(_dz_m)
            if K_v_cell.shape[-1] == _act.shape[-1] - 1:
                _K_m = K_v_cell * (_act[..., :-1] & _act[..., 1:])
            else:
                _K_m = K_v_cell * _act

            def _mix(f):
                f_new = implicit_vertical_diffusion_ocean(
                    f, _K_m.astype(f.dtype), _dz_m.astype(f.dtype),
                    _dzh_m.astype(f.dtype), dt,
                )
                return jnp.where(_act, f_new, f)
'''
s = P.read_text()
if "Partial-cell geometry, as in the tracer advection" in s:
    sys.exit("already patched; nothing changed")
if s.count(OLD) != 1:
    sys.exit(f"not patched, nothing changed: MARBL _mix block found {s.count(OLD)}x")
if "compute_layer_thickness" not in s:
    sys.exit("not patched: compute_layer_thickness not available in the module")
P.with_name(P.name + ".bak_vmix").write_text(s)
P.write_text(s.replace(OLD, NEW))
print(f"patched {P} (backup {P.name}.bak_vmix)")
