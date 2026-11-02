"""Diagnostic switches for the MARBL tracer transport (default: everything on).
  MARBL_NO_HDIFF=1 / MARBL_NO_ADV=1 / MARBL_NO_VMIX=1 skip that part for MARBL tracers.
Backup: ocean_model_latlon_cgrid.py.bak_switch"""
import sys
from pathlib import Path

P = Path("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py")
if len(sys.argv) > 1:
    P = Path(sys.argv[1])
EDITS = [
    ("    _diff = _marbl_flux_form_diffusion(grid, h_k_old, active_3d, K_h, dt) if K_h > 0.0 else None\n",
     "    _diff = (_marbl_flux_form_diffusion(grid, h_k_old, active_3d, K_h, dt)\n"
     "             if K_h > 0.0 and __import__(\"os\").environ.get(\"MARBL_NO_HDIFF\", \"0\") != \"1\" else None)\n"),
    ("        if config.tracer_time_integrator == \"rk3\":\n            tr_new = _ssp_rk3_tracer_step(\n",
     "        if __import__(\"os\").environ.get(\"MARBL_NO_ADV\", \"0\") == \"1\":\n"
     "            return tr.astype(dtype)\n"
     "        if config.tracer_time_integrator == \"rk3\":\n            tr_new = _ssp_rk3_tracer_step(\n"),
    ("        if _is_marbl_tracers(state.biogeo):\n            # Same backward-Euler K_v solve for every MARBL tracer.\n",
     "        if _is_marbl_tracers(state.biogeo) and __import__(\"os\").environ.get(\"MARBL_NO_VMIX\", \"0\") != \"1\":\n"
     "            # Same backward-Euler K_v solve for every MARBL tracer.\n"),
]
s = P.read_text()
if "MARBL_NO_HDIFF" in s:
    sys.exit("already patched; nothing changed")
bad = [o[:60] for o, n in EDITS if s.count(o) != 1]
if bad:
    sys.exit(f"not patched, nothing changed; anchors not found once: {bad}")
P.with_name(P.name + ".bak_switch").write_text(s)
for o, n in EDITS:
    s = s.replace(o, n)
P.write_text(s)
print(f"patched {P} (backup {P.name}.bak_switch)")
