"""Add sediment + hydrothermal Fe sources to the MARBL coupling (omip_coupling.py).

After each MARBL source/sink step:  Fe += (fe_sed + fe_vent) * dt_bgc / h   in the ocean cells,
with fe_* [mmol m-2 s-1] from the file in the environment variable MARBL_FE_SOURCES
(build_marbl_fe_sources.py).  Without the variable nothing changes.

Usage: python patch_marbl_fe_sources.py /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm/ocean/biogeochemistry/marbl/omip_coupling.py
"""
import re
import shutil
import sys

p = sys.argv[1]
s = open(p).read()
if "MARBL_FE_SOURCES" in s:
    sys.exit("already patched")
lines = s.splitlines(keepends=True)
hits = [i for i, l in enumerate(lines) if re.search(r'=\s*_M\["step"\]\(', l)]
assert len(hits) == 1, f"expected one MARBL step call, found {len(hits)}"
i = hits[0]
ind = re.match(r"\s*", lines[i]).group(0)
j, bal = i, lines[i].count("(") - lines[i].count(")")
while bal > 0:                                   # continuation lines of the call, if any
    j += 1
    bal += lines[j].count("(") - lines[j].count(")")
lhs = lines[i].split("=")[0].split(",")[0].strip()          # name of the new tracer state (e.g. bgc_new)
block = f'''{ind}# sediment + hydrothermal Fe (MARBL_FE_SOURCES, build_marbl_fe_sources.py) [patch_marbl_fe_sources]
{ind}if "fe_src" not in _M:
{ind}    _p = __import__("os").environ.get("MARBL_FE_SOURCES")
{ind}    _M["fe_src"] = None
{ind}    if _p:
{ind}        import xarray as _xr
{ind}        _d = _xr.open_dataset(_p)
{ind}        _src = (_d["fe_sed"].values + _d["fe_vent"].values) * __import__("numpy").asarray(_M["mask"])[..., None]
{ind}        _M["fe_src"] = jnp.asarray(_src)
{ind}        _tot = float((_src.sum(-1) * __import__("numpy").asarray(_M["area"])).sum()) * 365 * 86400 * 1e-12
{ind}        print(f"  MARBL: sediment + vent Fe from {{_p}}: {{_tot:.2f}} Gmol/yr", flush=True)
{ind}if _M["fe_src"] is not None:
{ind}    _act = h_k > 1e-3
{ind}    _dFe = jnp.where(_act, _M["fe_src"] * (_M["dt_bgc_days"] * 86400.0) / jnp.maximum(h_k, 1e-3), 0.0)
{ind}    {lhs} = {lhs}._replace(Fe={lhs}.Fe + _dFe.astype({lhs}.Fe.dtype))
'''
shutil.copy(p, p + ".bak_fesrc")
lines.insert(j + 1, block)
out = "".join(lines)
compile(out, p, "exec")
open(p, "w").write(out)
print(f"patched {p}: Fe sources added after line {i + 1} (tracer state '{lhs}'); backup {p}.bak_fesrc")
