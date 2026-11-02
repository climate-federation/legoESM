"""Let --marbl-forcing files omit 'ifrac' (then: sea ice where SST < -1 C).
Backup: omip_coupling.py.bak_ifrac"""
import sys
from pathlib import Path

P = Path("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm/ocean/biogeochemistry/marbl/omip_coupling.py")
if len(sys.argv) > 1:
    P = Path(sys.argv[1])
A_OLD = '''    for k in FORCING_VARS:
        a = np.nan_to_num(np.asarray(ds[k].values, np.float64))
'''
A_NEW = '''    for k in FORCING_VARS:
        if k not in ds:
            if k == "ifrac":
                print("  MARBL forcing: no ifrac in file -> sea ice where SST < -1 C")
                continue
            raise KeyError(f"{path}: missing {k!r}")
        a = np.nan_to_num(np.asarray(ds[k].values, np.float64))
'''
B_OLD = '''        ifrac = f["ifrac"][m]
'''
B_NEW = '''        ifrac = f["ifrac"][m] if "ifrac" in f else jnp.clip((-1.0 - sst) / 0.8, 0.0, 1.0)
'''
C_OLD = '''    print(f"  MARBL forcing: monthly {', '.join(FORCING_VARS)} from {path}")
'''
C_NEW = '''    print(f"  MARBL forcing: monthly {', '.join(out)} from {path}")
'''
s = P.read_text()
bad = [n for o, n in ((A_OLD, "_load_forcing loop"), (B_OLD, "ifrac lookup"), (C_OLD, "forcing print")) if s.count(o) != 1]
if bad:
    sys.exit(f"not patched, nothing changed: {bad}")
P.with_name(P.name + ".bak_ifrac").write_text(s)
P.write_text(s.replace(A_OLD, A_NEW).replace(B_OLD, B_NEW).replace(C_OLD, C_NEW))
print(f"patched {P} (backup {P.name}.bak_ifrac)")
