"""MARBL budget diagnostics with the model's own layer thicknesses; MARBL_NO_SOURCES=1 = transport only.
Backup: omip_coupling.py.bak_inv"""
import sys
from pathlib import Path

P = Path("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm/ocean/biogeochemistry/marbl/omip_coupling.py")
if len(sys.argv) > 1:
    P = Path(sys.argv[1])

EDITS = [
    ("import sys\n", "import os\nimport sys\n"),
    ("        acc=None, n_acc=0, acc_day0=None,\n",
     "        acc=None, n_acc=0, acc_day0=None,\n"
     "        no_sources=os.environ.get(\"MARBL_NO_SOURCES\", \"0\") == \"1\",\n"),
    ("def _accumulate(diag, day, h_k, bgc):\n",
     '''def _inventories(bgc, h_k):
    """Inventories [mmol] with the model's layer thicknesses, ocean cells only."""
    vol = h_k * _M["area"][..., None] * _M["mask"][..., None]
    tr = {"P": bgc.PO4 + bgc.DOP + bgc.DOPr + bgc.spP + bgc.diatP + bgc.diazP + bgc.zooC / 117.0,
          "N": bgc.NO3 + bgc.NH4 + bgc.DON + bgc.DONr
               + (bgc.spC + bgc.diatC + bgc.diazC + bgc.zooC) * (16.0 / 117.0),
          "ALK": bgc.ALK, "DIC": bgc.DIC, "Si": bgc.SiO3 + bgc.diatSi, "Fe": bgc.Fe}
    return {k: float(jnp.sum(v * vol)) for k, v in tr.items()}


def _print_inventory(label, day, h_k, bgc):
    inv, inv0 = _inventories(bgc, h_k), _M["INV0"]
    print(f"    [MARBL{label}] day {day:.2f} inventory change since start: "
          + ", ".join(f"{k} {(inv[k] - inv0[k]) / inv0[k]:+.2e}" for k in inv), flush=True)


def _inventory_only(day, h_k, bgc):
    if _M.get("last_inv_day") is None:
        _M["last_inv_day"] = day - _M["dt_bgc_days"]
    if day - _M["last_inv_day"] + 1e-9 < _M["diag_days"]:
        return
    _M["last_inv_day"] = day
    _print_inventory(" transport only", day, h_k, bgc)


def _accumulate(diag, day, h_k, bgc):
'''),
    ('''          f"total P drift {(p_tot - p0) / p0:+.2e}  -> {fname.name}", flush=True)
''',
     '''          f"total P drift {(p_tot - p0) / p0:+.2e}  -> {fname.name}", flush=True)
    _print_inventory("", day, h_k, bgc)
'''),
    ('''    h_k = _layer_thickness(state)
    bgc_new, diag = _M["step"](state.biogeo, state.T.data, state.S.data, h_k, forcing)
''',
     '''    h_k = _layer_thickness(state)
    if "INV0" not in _M:
        _M["INV0"] = _inventories(state.biogeo, h_k)
        print(f"    [MARBL] budget diagnostics on{' (sources/sinks OFF: transport only)' if _M['no_sources'] else ''}",
              flush=True)
    if _M["no_sources"]:
        _inventory_only(day, h_k, state.biogeo)
        return state
    bgc_new, diag = _M["step"](state.biogeo, state.T.data, state.S.data, h_k, forcing)
'''),
]
s = P.read_text()
bad = [new[:40] for old, new in EDITS if s.count(old) != 1]
if "_inventories" in s:
    bad.append("already patched")
if bad:
    sys.exit(f"not patched, nothing changed: anchors not found exactly once for {bad}")
P.with_name(P.name + ".bak_inv").write_text(s)
for old, new in EDITS:
    s = s.replace(old, new)
P.write_text(s)
print(f"patched {P} (backup {P.name}.bak_inv)")
