"""Render the full coupled Earth-system model as a component-exchange graph.

Top altitude: atmosphere, land, ocean, and sea ice as boxes, with the
coupler as the central flux hub.  Each edge is a labelled bundle of the
physical quantities exchanged across that interface.  The coupler extracts
``AtmToSurface`` from the atmosphere, runs the four surface tiles (land /
ocean / sea-ice / lake), blends them by area fraction, and returns
``SurfaceToAtm`` as the atmosphere's lower boundary.

Exchange bundles transcribed from the canonical coupling structs in
``core/coupling_fields.py`` (AtmToSurface, TileResponse, SurfaceToAtm) and
the tile drivers in ``coupler/coupler.py``.  Like the other graphs in this
folder, this is hand-maintained documentation, not a parser.

Output: DOT + PNG under ``results/microphysics_graphs/`` (gitignored),
rendered with the system ``dot`` binary — no extra Python dependency.

ponytail: same DOT + `dot` pattern as the atmosphere/microphysics graphs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OUT_DIR = Path("results/microphysics_graphs")

# Component nodes: id -> (label, fillcolor).
COMPONENTS = {
    "atm": ("ATMOSPHERE\\ndynamics + radiation / convection /\\n"
            "turbulence / microphysics / GWD\\n(see atmosphere_model graph)", "#cce5ff"),
    "coupler": ("COUPLER\\nflux exchange + tile blend\\n(f_land, f_ocean, f_ice)", "#e0e0e0"),
    "land": ("LAND\\nsoil / snow / canopy / hydrology", "#cdebc0"),
    "ocean": ("OCEAN\\nbaroclinic + barotropic dycore", "#a9d0f0"),
    "ice": ("SEA ICE\\nthermodynamics + dynamics", "#d8f3ff"),
}

# Directed exchange edges: (src, dst, label, color).
EXCHANGES = [
    # Atmosphere <-> coupler (AtmToSurface down, SurfaceToAtm up).
    ("atm", "coupler",
     "SW↓, LW↓, precip,\\nT/q/u/v near-surface,\\np, ρ, cos(zenith), CO₂", "#3366aa"),
    ("coupler", "atm",
     "T_sfc, albedo, ε, z0,\\nsensible + latent flux,\\nwind stress τ, LW↑, q_sfc", "#aa3333"),
    # Coupler <-> land.
    ("coupler", "land",
     "SW↓, LW↓, precip,\\nnear-surface atm state", "#3366aa"),
    ("land", "coupler",
     "T_sfc, albedo, q_sfc,\\nsensible + latent flux,\\nrunoff", "#338833"),
    # Coupler <-> ocean.
    ("coupler", "ocean",
     "net heat flux,\\nfreshwater (P−E), salt flux,\\nwind stress τ", "#3366aa"),
    ("ocean", "coupler",
     "SST, surface currents,\\nCO₂ flux", "#225588"),
    # Coupler <-> sea ice.
    ("coupler", "ice",
     "SW↓, LW↓, precip,\\nnear-surface atm state", "#3366aa"),
    ("ice", "coupler",
     "T_sfc, albedo,\\nconductive / melt flux,\\nice fraction", "#5599bb"),
    # Direct ice <-> ocean interface (basal exchange).
    ("ice", "ocean",
     "basal heat / freeze-melt,\\nbrine salt flux", "#777777"),
    ("ocean", "ice",
     "freezing SST,\\nocean heat", "#777777"),
]


def _to_dot() -> str:
    L = [
        "digraph coupled {",
        "  rankdir=TB; bgcolor=white; nodesep=0.6; ranksep=1.0; splines=true;",
        '  labelloc="t"; fontsize=18; fontname="Helvetica-Bold";',
        '  label="Coupled Earth-system model — component exchange\\n'
        'edges = physical quantities exchanged · coupler mediates atm ↔ surface";',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]
    for cid, (label, fill) in COMPONENTS.items():
        L.append(f'  {cid} [label="{label}", fillcolor="{fill}"];')
    # Pin ranks: atmosphere top, coupler middle, surface row bottom.
    L.append('  {rank=source; atm;}')
    L.append('  {rank=same; land; ocean; ice;}')
    for src, dst, lab, col in EXCHANGES:
        L.append(f'  {src} -> {dst} [label="{lab}", color="{col}", '
                 f'fontcolor="{col}", penwidth=1.6];')
    L.append("}")
    return "\n".join(L)


def render(out_dir: Path = OUT_DIR) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / "coupled_model.dot"
    dot_path.write_text(_to_dot())
    dot_bin = shutil.which("dot")
    if not dot_bin:
        print("graphviz 'dot' not found; wrote .dot only:", dot_path)
        return None
    png_path = out_dir / "coupled_model.png"
    subprocess.run([dot_bin, "-Tpng", "-Gdpi=150", str(dot_path), "-o", str(png_path)],
                   check=True)
    return png_path


def _self_check() -> None:
    valid = set(COMPONENTS)
    for src, dst, *_ in EXCHANGES:
        assert src in valid and dst in valid, f"bad exchange {src}->{dst}"
    dot = _to_dot()
    assert "digraph" in dot and dot.count("->") >= 8, "graph too sparse"


if __name__ == "__main__":
    _self_check()
    p = render()
    if p:
        print(p)