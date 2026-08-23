"""Write the FESOM2 CORE2 INITIAL CONDITION as a comparator snapshot.

Decomposes the day-30 legoESM-vs-FESOM2 discrepancy: if the IC alone already
scores ~the day-30 rmse against the NEMO January reference, the FESOM2 gap is
dominated by PROTOCOL (PHC3.0 winter IC) rather than 30 days of divergent
model physics/forcing.  Uses the same mesh loader and snapshot writer as the
run driver, with a zero-dynamics stand-in state.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "run_fesom_core2", _ROOT / "scripts/run/run_fesom_core2.py")
_RFC = importlib.util.module_from_spec(_SPEC)
sys.modules["run_fesom_core2"] = _RFC
_SPEC.loader.exec_module(_RFC)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh-dir", required=True)
    p.add_argument("--ic-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(a.mesh_dir)
    T = np.load(Path(a.ic_dir) / "T_ic.npy")
    S = np.load(Path(a.ic_dir) / "S_ic.npy")
    n = mesh.nod2D
    state = types.SimpleNamespace(
        T=T, S=S,
        eta_n=np.zeros(n), a_ice=np.zeros(n), m_ice=np.zeros(n))
    path = _RFC.write_snapshot(out, "day0000_ic", state, mesh)
    print(f"[ic-snapshot] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
