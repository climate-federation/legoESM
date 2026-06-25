"""Render the full atmospheric model as a component-coupling causal graph.

Altitude: one level up from ``plot_microphysics_graphs.py``.  Nodes are the
prognostic state variables (u, v, T, p_s, q_v, q_c, q_r, q_i) plus the
dynamical core and the five physics parameterization slots.  A SOLID coloured
edge ``component -> state`` is a tendency the component writes (its causal
effect on the state); each component also lists the fields it READS.  Dashed
grey edges are surface/boundary forcing.

Default config (PhysicsConfig defaults, the "first pass"):
    radiation   = gray          (ON)
    convection  = sbm           (ON)
    turbulence  = smagorinsky   (ON)
    microphysics= none          (disabled by default — drawn greyed)
    gravity_wave_drag = none    (disabled by default — drawn greyed)

Write-edges (which tendencies each slot emits) were confirmed against
``*/integration.py``; read-sets are the standard column inputs.  Like the
microphysics graphs this is hand-maintained documentation, not a parser.

Output: DOT + PNG under ``results/microphysics_graphs/`` (gitignored),
rendered with the system ``dot`` binary — no extra Python dependency.

ponytail: DOT + `dot` subprocess, reuses the pattern from the micro graphs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OUT_DIR = Path("results/microphysics_graphs")

# Prognostic state variables (graph sinks/sources).
STATE = {
    "u": "u, v\\nwind",
    "T": "T\\ntemperature",
    "ps": "p_s\\nsurface pressure",
    "qv": "q_v\\nvapour",
    "qc": "q_c\\ncloud water",
    "qr": "q_r\\nrain",
    "qi": "q_i\\nice",
}

# Each component: colour, default scheme, on/off, reads (label), writes (state ids).
COMPONENTS = {
    "dynamics": {
        "label": "Dynamical core\\n(advection, PGF,\\nCoriolis, hydrostatic)",
        "color": "#4477aa", "scheme": "spectral / FV", "on": True,
        "writes": ["u", "T", "ps", "qv", "qc", "qr", "qi"],
        "wlabel": "transport",
    },
    "radiation": {
        "label": "Radiation\\nreads: T, q_v, clouds",
        "color": "#ee6677", "scheme": "gray", "on": True,
        "writes": ["T"], "wlabel": "dT (heating)",
    },
    "convection": {
        "label": "Convection\\nreads: T, q_v, q_c, u, v",
        "color": "#228833", "scheme": "sbm", "on": True,
        "writes": ["T", "qv", "qc", "u"], "wlabel": "dT, dq, CMT",
    },
    "turbulence": {
        "label": "Turbulence / PBL\\nreads: u, v, T, q_v, sfc flux",
        "color": "#ccbb44", "scheme": "smagorinsky", "on": True,
        "writes": ["u", "T", "qv"], "wlabel": "diffusion",
    },
    "microphysics": {
        "label": "Microphysics\\nreads: T, q_v, q_c, q_r, q_i",
        "color": "#66ccee", "scheme": "none", "on": False,
        "writes": ["T", "qv", "qc", "qr", "qi"], "wlabel": "dT, dq",
    },
    "gwd": {
        "label": "Gravity-wave drag\\nreads: u, v, T",
        "color": "#aa3377", "scheme": "none", "on": False,
        "writes": ["u", "T"], "wlabel": "du, dv, dT",
    },
}

# Variants: name -> (title, {cid: (scheme, on)} overrides applied to COMPONENTS).
VARIANTS = {
    "atmosphere_model": (
        "Atmospheric model — component coupling (default config)\\n"
        "solid = tendency written · grey nodes = scheme \\\"none\\\" (off by default)",
        {},
    ),
    "atmosphere_model_all_on": (
        "Atmospheric model — component coupling (full physics suite)\\n"
        "solid = tendency written · every slot active",
        {"microphysics": ("morrison", True), "gwd": ("lindzen", True)},
    ),
}


def _to_dot(title: str, overrides: dict) -> str:
    # Apply per-variant scheme/on overrides onto a shallow copy.
    comps = {}
    for cid, c in COMPONENTS.items():
        c = dict(c)
        if cid in overrides:
            c["scheme"], c["on"] = overrides[cid]
        comps[cid] = c
    L = [
        "digraph atmosphere {",
        "  rankdir=LR; bgcolor=white; nodesep=0.35; ranksep=1.1;",
        f'  labelloc="t"; fontsize=18; fontname="Helvetica-Bold"; label="{title}";',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        # State variables clustered on the right.
        '  subgraph cluster_state {',
        '    label="prognostic state"; style="rounded,dashed"; color="#999999"; fontsize=12;',
    ]
    for sid, lab in STATE.items():
        L.append(f'    {sid} [label="{lab}", shape=box, style="filled,rounded", '
                 f'fillcolor="#eeeeee"];')
    L.append("  }")
    # Component nodes.
    for cid, c in comps.items():
        tag = c["scheme"] if c["on"] else f'{c["scheme"]} (off)'
        fill = c["color"] if c["on"] else "#dddddd"
        pen = "#222222" if c["on"] else "#999999"
        style = "filled,rounded" if c["on"] else "filled,rounded,dashed"
        L.append(f'  {cid} [label="{c["label"]}\\n[{tag}]", shape=box, '
                 f'style="{style}", fillcolor="{fill}", color="{pen}"];')
    # Surface forcing node.
    L.append('  surface [label="Surface / boundary\\nT_sfc, fluxes, albedo", '
             'shape=box, style="filled,rounded", fillcolor="#f0d9b5"];')
    L.append('  precip [label="precip\\nsurface", shape=box, '
             'style="filled,rounded", fillcolor="#bbbbbb"];')
    # Write edges: component -> state.
    for cid, c in comps.items():
        col = c["color"] if c["on"] else "#bbbbbb"
        sstyle = "solid" if c["on"] else "dashed"
        first = True
        for sid in c["writes"]:
            lab = f' label="{c["wlabel"]}"' if first else ""
            first = False
            L.append(f'  {cid} -> {sid} [color="{col}" style={sstyle} penwidth=1.6{lab}];')
    # Surface forcing (reads) into radiation + turbulence.
    L.append('  surface -> radiation [style=dashed color="#cc8844" label="T_sfc, albedo"];')
    L.append('  surface -> turbulence [style=dashed color="#cc8844" label="sfc fluxes"];')
    # Precip sinks.
    L.append('  convection -> precip [color="#228833" style=dashed label="conv. precip"];')
    L.append('  microphysics -> precip [color="#66ccee" style=dashed label="ls precip"];')
    L.append("}")
    return "\n".join(L)


def render(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_bin = shutil.which("dot")
    outs = []
    for name, (title, overrides) in VARIANTS.items():
        dot_path = out_dir / f"{name}.dot"
        dot_path.write_text(_to_dot(title, overrides))
        if not dot_bin:
            print("graphviz 'dot' not found; wrote .dot only:", dot_path)
            continue
        png_path = out_dir / f"{name}.png"
        subprocess.run([dot_bin, "-Tpng", "-Gdpi=150", str(dot_path), "-o", str(png_path)],
                       check=True)
        outs.append(png_path)
    return outs


def _self_check() -> None:
    valid = set(STATE)
    for cid, c in COMPONENTS.items():
        for sid in c["writes"]:
            assert sid in valid, f"{cid}: writes unknown state {sid}"
    for _name, (title, overrides) in VARIANTS.items():
        dot = _to_dot(title, overrides)
        assert "digraph" in dot and dot.count("->") >= 10, "graph too sparse"


if __name__ == "__main__":
    _self_check()
    for p in render():
        print(p)
