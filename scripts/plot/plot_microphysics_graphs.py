"""Render cloud-microphysics schemes as causal (process-flow) graphs.

Each scheme is drawn as a directed graph whose nodes are water-substance
reservoirs (vapour, cloud, rain, ice, snow, graupel/rime, surface precip)
plus temperature, and whose edges are the microphysical processes that move
mass between reservoirs.  Dashed edges into ``T`` mark latent-heat feedback.

Process lists are transcribed from each backend's docstring + body in
``legoesm/atmosphere/physics/microphysics/`` (kessler, seifert_beheng,
sundqvist, morrison, thompson, p3).  This is documentation, not a parser:
update the dicts below if the schemes change.

Output: Graphviz DOT + rendered PNG under ``results/microphysics_graphs/``
(gitignored).  Rendering uses the system ``dot`` binary, so no extra Python
dependency is needed.

ponytail: DOT strings + `dot` subprocess, no python-graphviz/networkx dep.
Add a real graph lib only if these grow interactive.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OUT_DIR = Path("results/microphysics_graphs")

# Reservoir node styling, keyed by canonical species id.
NODES = {
    "qv": ("q_v\\nvapour", "#cfe8ff"),
    "qc": ("q_c\\ncloud water", "#bdf0c0"),
    "qr": ("q_r\\nrain", "#74c476"),
    "qi": ("q_i\\ncloud ice", "#e6d0ff"),
    "qs": ("q_s\\nsnow", "#c9a0ff"),
    "qg": ("q_g\\ngraupel", "#a878e0"),
    "qrim": ("q_rim / B_rim\\nrimed ice (P3)", "#a878e0"),
    "Nc": ("N_c\\ncloud number", "#eaffea"),
    "Nr": ("N_r\\nrain number", "#d9f5d9"),
    "Ni": ("N_i\\nice number", "#f3e8ff"),
    "P": ("precip\\nsurface", "#bbbbbb"),
    "T": ("T\\ntemperature", "#ffd9b3"),
}

# Each scheme: list of (src, dst, process-label).  src/dst are NODES keys.
# Edges into "T" are rendered dashed (latent-heat feedback).
SCHEMES: dict[str, dict] = {
    "kessler": {
        "title": "Kessler (1969) — 1-moment warm rain",
        "edges": [
            ("qv", "qc", "condensation"),
            ("qc", "qv", "evaporation"),
            ("qc", "qr", "autoconversion"),
            ("qc", "qr", "accretion"),
            ("qr", "qv", "rain evaporation"),
            ("qr", "P", "sedimentation"),
            ("qv", "T", "latent heat"),
            ("qr", "T", "latent heat"),
        ],
    },
    "sundqvist": {
        "title": "Sundqvist (1989) — diagnostic large-scale condensation",
        "edges": [
            ("qv", "qc", "RH-threshold\\ncondensation"),
            ("qc", "qr", "autoconversion"),
            ("qr", "qv", "sub-cloud\\nevaporation"),
            ("qr", "P", "large-scale precip"),
            ("qv", "T", "latent heat"),
        ],
    },
    "seifert_beheng": {
        "title": "Seifert-Beheng (2001) — 2-moment warm rain",
        "edges": [
            ("qv", "qc", "condensation"),
            ("qc", "qv", "evaporation"),
            ("qc", "qr", "autoconversion (mass-dep.)"),
            ("qc", "qr", "accretion"),
            ("qr", "qv", "rain evaporation"),
            ("qr", "P", "sedimentation"),
            ("Nc", "Nr", "autoconversion"),
            ("Nr", "Nr", "self-collection / breakup"),
            ("qv", "T", "latent heat"),
        ],
    },
    "morrison": {
        "title": "Morrison (2005) — 2-moment ice + liquid",
        "edges": [
            ("qv", "qc", "condensation"),
            ("qc", "qv", "evaporation"),
            ("qc", "qr", "autoconversion"),
            ("qc", "qr", "accretion"),
            ("qr", "qv", "rain evaporation"),
            ("Nr", "Nr", "self-collection / breakup"),
            ("qv", "qi", "ice nucleation (Cooper/homog.)"),
            ("qv", "qi", "deposition / sublimation"),
            ("qc", "qi", "Bergeron"),
            ("qi", "qs", "ice→snow autoconversion"),
            ("qc", "qs", "riming"),
            ("qi", "qs", "aggregation"),
            ("qs", "qr", "melting"),
            ("qi", "qc", "melting"),
            ("qr", "P", "rain sedimentation"),
            ("qs", "P", "snow sedimentation"),
            ("qv", "T", "latent heat"),
            ("qi", "T", "fusion / dep. heat"),
        ],
    },
    "thompson": {
        "title": "Thompson (2008) — hybrid-moment, adds graupel",
        "edges": [
            ("qv", "qc", "condensation"),
            ("qc", "qv", "evaporation"),
            ("qc", "qr", "autoconversion (gamma)"),
            ("qc", "qr", "accretion (gamma)"),
            ("qr", "qv", "rain evaporation"),
            ("qv", "qi", "ice nucleation"),
            ("qv", "qi", "deposition / sublimation"),
            ("qi", "qs", "ice→snow autoconv. / aggreg."),
            ("qc", "qs", "riming"),
            ("qc", "qg", "intense riming"),
            ("qr", "qg", "freezing / riming"),
            ("qs", "qg", "riming→graupel"),
            ("qs", "qr", "melting"),
            ("qg", "qr", "melting"),
            ("qr", "P", "rain sedimentation"),
            ("qs", "P", "snow sedimentation"),
            ("qg", "P", "graupel sedimentation"),
            ("qv", "T", "latent heat"),
            ("qi", "T", "fusion / dep. heat"),
        ],
    },
    "p3": {
        "title": "P3 (Morrison-Milbrandt 2015) — predicted particle properties",
        "edges": [
            ("qv", "qc", "condensation"),
            ("qc", "qv", "evaporation"),
            ("qc", "qr", "autoconversion"),
            ("qc", "qr", "accretion"),
            ("qr", "qv", "rain evaporation"),
            ("qv", "qrim", "nucleation / deposition"),
            ("qc", "qrim", "riming (predicts ρ_rim)"),
            ("qr", "qrim", "freezing"),
            ("qrim", "qr", "melting"),
            ("qr", "P", "rain sedimentation"),
            ("qrim", "P", "ice sedimentation"),
            ("qv", "T", "latent heat"),
            ("qrim", "T", "fusion / dep. heat"),
        ],
    },
}


def _to_dot(name: str, scheme: dict) -> str:
    used = {n for e in scheme["edges"] for n in e[:2]}
    lines = [
        f'digraph {name} {{',
        '  rankdir=LR;',
        '  bgcolor="white";',
        f'  labelloc="t"; fontsize=18; fontname="Helvetica-Bold";',
        f'  label="{scheme["title"]}";',
        '  node [style="filled,rounded", shape=box, fontname="Helvetica", fontsize=12];',
        '  edge [fontname="Helvetica", fontsize=9, color="#555555"];',
    ]
    for nid in NODES:
        if nid not in used:
            continue
        label, fill = NODES[nid]
        shape = "ellipse" if nid == "T" else "box"
        lines.append(f'  {nid} [label="{label}", fillcolor="{fill}", shape={shape}];')
    for src, dst, lab in scheme["edges"]:
        style = ' style=dashed color="#cc6600"' if dst == "T" else ""
        lines.append(f'  {src} -> {dst} [label="{lab}"{style}];')
    lines.append("}")
    return "\n".join(lines)


def render(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_bin = shutil.which("dot")
    pngs = []
    for name, scheme in SCHEMES.items():
        dot_path = out_dir / f"{name}.dot"
        dot_path.write_text(_to_dot(name, scheme))
        if dot_bin:
            png_path = out_dir / f"{name}.png"
            subprocess.run(
                [dot_bin, "-Tpng", "-Gdpi=150", str(dot_path), "-o", str(png_path)],
                check=True,
            )
            pngs.append(png_path)
    return pngs


def _self_check() -> None:
    # Every edge endpoint must be a known node; DOT must be non-trivial.
    for name, scheme in SCHEMES.items():
        for src, dst, _ in scheme["edges"]:
            assert src in NODES and dst in NODES, f"{name}: bad edge {src}->{dst}"
        dot = _to_dot(name, scheme)
        assert "digraph" in dot and "->" in dot, f"{name}: empty graph"


if __name__ == "__main__":
    _self_check()
    if shutil.which("dot") is None:
        print("graphviz 'dot' not found; wrote .dot files only.")
    outs = render()
    for p in outs:
        print(p)
