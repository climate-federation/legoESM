"""Add the BOMEX LES run's wall-drag law to its artifact record (user 2026-10-28).

run_bomex_les.py built SpectralLESConfig without sfc_flux_drag, so the run used the
config default "neutral" (log-law drag under the prescribed fluxes); the driver did not
record it. Only __meta__ changes; every array is copied byte-for-byte and verified.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

src = Path(sys.argv[1])
backup_dir = Path(sys.argv[2])
backup_dir.mkdir(parents=True, exist_ok=True)
backup = backup_dir / src.name
if backup.exists():
    raise SystemExit(f"backup {backup} exists; refusing to overwrite")
shutil.copy2(src, backup)

with zipfile.ZipFile(src) as zf:
    compressed = any(i.compress_type != zipfile.ZIP_STORED for i in zf.infolist())
old = np.load(backup)
arrays = {k: old[k] for k in old.files}
meta = json.loads(bytes(arrays["__meta__"]).decode())
if meta.get("case_name") != "bomex_cu" or src.name != "bomex_cu.npz":
    raise SystemExit(f"not the BOMEX artifact: case_name={meta.get('case_name')!r}")
setup = meta.get("les_setup")
if not isinstance(setup, dict):
    raise SystemExit("no les_setup dict in the record")
if "sfc_flux_drag" in setup:
    raise SystemExit(f"record already has sfc_flux_drag={setup['sfc_flux_drag']!r}")
setup["sfc_flux_drag"] = "neutral"
setup["sfc_flux_drag_source"] = (
    "added 2026-10-28: run_bomex_les.py left SpectralLESConfig.sfc_flux_drag at its "
    "default 'neutral' and did not record it; original artifact backed up")
arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)

tmp = src.with_name(src.name + ".tmp.npz")
(np.savez_compressed if compressed else np.savez)(tmp, **arrays)
new = np.load(tmp)
assert set(new.files) == set(old.files)
for k in old.files:
    if k == "__meta__":
        continue
    a, b = old[k], new[k]
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
m2 = json.loads(bytes(new["__meta__"]).decode())
m0 = json.loads(bytes(old["__meta__"]).decode())
diff = {k for k in set(m2["les_setup"]) | set(m0["les_setup"])
        if m2["les_setup"].get(k) != m0["les_setup"].get(k)}
assert diff == {"sfc_flux_drag", "sfc_flux_drag_source"}, diff
assert {k: v for k, v in m2.items() if k != "les_setup"} == {
    k: v for k, v in m0.items() if k != "les_setup"}
tmp.replace(src)
print(f"patched {src} (compressed={compressed}); backup {backup}; arrays identical: "
      f"{len(old.files) - 1}")
