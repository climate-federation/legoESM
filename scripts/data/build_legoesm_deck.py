"""
build_legoesm_deck.py
=====================

Build the legoESM technical seminar deck (33 slides) as a real .pptx file
using ONLY the Python standard library. No pip install required.

Usage
-----
    python3 ~/Documents/legoESM/scripts/build_legoesm_deck.py

Output: ``~/Documents/legoESM/docs/assets/legoESM_seminar.pptx``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from html import escape as _esc
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile, ZIP_DEFLATED


# ---------------------------------------------------------------------------
# Geometry / theme
# ---------------------------------------------------------------------------

EMU = 914400
SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5
SLIDE_W = int(SLIDE_W_IN * EMU)
SLIDE_H = int(SLIDE_H_IN * EMU)
M = 0.55

TEAL = "0F4C5C"
TEAL_DARK = "0A3A47"
SLATE = "415A77"
RUST = "E36414"
INK = "1F2937"
MUTED = "475569"
SUBTLE = "94A3B8"
PAPER = "FFFFFF"
TINT = "F1F5F9"
BORDER = "CBD5E1"
LIGHT_GREY = "E2E8F0"
PALE_INK = "CBD5E1"

COL_ATM = "E0F2F1"
COL_OCN = "DBEAFE"
COL_LND = "ECF5DF"
COL_ICE = "E9EFF6"
COL_CPL = "FBE9DC"
COL_DA = "ECE6F7"

HEADER_FONT = "Georgia"
BODY_FONT = "Calibri"
MONO_FONT = "Consolas"

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "old"
DOCS = REPO / "docs"
DIAG = REPO / "diagnostics" / "fv3_visual"
DEBUG = REPO / "tests" / "debug"
LOGO_PATH = REPO / "docs" / "assets" / "legoESM.png"


def first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def emu(inches: float) -> int:
    return int(round(inches * EMU))


def pt(size: float) -> int:
    return int(round(size * 100))


def color(hexstr: str) -> str:
    return hexstr.replace("#", "").upper()


@dataclass
class IdGen:
    next_id: int = 2

    def __call__(self) -> int:
        v = self.next_id
        self.next_id += 1
        return v


@dataclass
class Run:
    text: str
    size: float = 14
    bold: bool = False
    italic: bool = False
    color: str = INK
    font: str = BODY_FONT


@dataclass
class Para:
    runs: list
    align: str = "l"
    bullet: str | None = None
    line_spacing: float = 1.15
    space_after_pt: float = 4.0
    indent_in: float = 0.0


def _para_xml(p):
    pPr_bits = []
    if p.align != "l":
        pPr_bits.append(f' algn="{p.align}"')
    if p.indent_in:
        pPr_bits.append(f' marL="{emu(p.indent_in)}" indent="0"')
    line_spc = ""
    if p.line_spacing and p.line_spacing != 1.0:
        line_spc = f'<a:lnSpc><a:spcPct val="{int(p.line_spacing*100000)}"/></a:lnSpc>'
    space_after = ""
    if p.space_after_pt:
        space_after = f'<a:spcAft><a:spcPts val="{int(p.space_after_pt*100)}"/></a:spcAft>'
    bullet_xml = '<a:buNone/>'
    pPr = "<a:pPr" + "".join(pPr_bits) + ">" + line_spc + space_after + bullet_xml + "</a:pPr>"
    if not p.runs:
        return f"<a:p>{pPr}<a:endParaRPr lang=\"en-US\"/></a:p>"
    runs_xml = []
    for r in p.runs:
        b = ' b="1"' if r.bold else ""
        i = ' i="1"' if r.italic else ""
        rPr = (f'<a:rPr lang="en-US" sz="{pt(r.size)}"{b}{i} dirty="0">'
               f'<a:solidFill><a:srgbClr val="{color(r.color)}"/></a:solidFill>'
               f'<a:latin typeface="{_esc(r.font, quote=True)}"/>'
               f'<a:ea typeface="{_esc(r.font, quote=True)}"/>'
               f'<a:cs typeface="{_esc(r.font, quote=True)}"/>'
               f'</a:rPr>')
        runs_xml.append(f"<a:r>{rPr}<a:t>{_esc(r.text)}</a:t></a:r>")
    return f"<a:p>{pPr}{''.join(runs_xml)}</a:p>"


def _txbody(paras, anchor="t", wrap="square",
            margin_l=0.0, margin_t=0.0, margin_r=0.0, margin_b=0.0):
    bodyPr = (f'<a:bodyPr wrap="{wrap}" rtlCol="0" anchor="{anchor}" '
              f'lIns="{emu(margin_l)}" tIns="{emu(margin_t)}" '
              f'rIns="{emu(margin_r)}" bIns="{emu(margin_b)}">'
              f'<a:normAutofit/></a:bodyPr>')
    return f"<p:txBody>{bodyPr}<a:lstStyle/>{''.join(_para_xml(p) for p in paras)}</p:txBody>"


def sp_textbox(idg, x, y, w, h, paras, *, anchor="t", margin=0.0):
    sid = idg()
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="TextBox {sid}"/>'
        f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
        f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
        + _txbody(paras, anchor=anchor, margin_l=margin, margin_t=margin,
                  margin_r=margin, margin_b=margin)
        + '</p:sp>'
    )


def sp_rect(idg, x, y, w, h, fill, *, line_color=None, line_w_pt=0.75):
    sid = idg()
    fill_xml = "<a:noFill/>" if fill is None else f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>'
    if line_color is None:
        line_xml = "<a:ln><a:noFill/></a:ln>"
    else:
        line_xml = (f'<a:ln w="{int(line_w_pt*12700)}"><a:solidFill>'
                    f'<a:srgbClr val="{color(line_color)}"/></a:solidFill></a:ln>')
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Rect {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>'
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>'
    )


def sp_rect_with_text(idg, x, y, w, h, fill, paras, *,
                      line_color=None, line_w_pt=0.75, anchor="ctr", margin=0.05):
    sid = idg()
    fill_xml = "<a:noFill/>" if fill is None else f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>'
    if line_color is None:
        line_xml = "<a:ln><a:noFill/></a:ln>"
    else:
        line_xml = (f'<a:ln w="{int(line_w_pt*12700)}"><a:solidFill>'
                    f'<a:srgbClr val="{color(line_color)}"/></a:solidFill></a:ln>')
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Rect {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>'
        + _txbody(paras, anchor=anchor, margin_l=margin, margin_t=margin,
                  margin_r=margin, margin_b=margin)
        + '</p:sp>'
    )


def sp_picture(idg, x, y, w, h, rId):
    sid = idg()
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="Picture {sid}"/>'
        f'<p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rId}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def image_size(path):
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return (int.from_bytes(head[16:20], "big"),
                        int.from_bytes(head[20:24], "big"))
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2:
                        return None
                    while marker[1] == 0xff:
                        marker = marker[1:] + f.read(1)
                    size_bytes = f.read(2)
                    if len(size_bytes) < 2:
                        return None
                    seg_size = int.from_bytes(size_bytes, "big")
                    if 0xc0 <= marker[1] <= 0xcf and marker[1] not in (0xc4, 0xc8, 0xcc):
                        f.read(1)
                        h = int.from_bytes(f.read(2), "big")
                        w = int.from_bytes(f.read(2), "big")
                        return (w, h)
                    f.seek(seg_size - 2, 1)
    except Exception:
        return None
    return None


def fit_image(path, x, y, w, h):
    sz = image_size(path)
    if not sz:
        return (x, y, w, h)
    pw, ph = sz
    if ph == 0:
        return (x, y, w, h)
    ratio = pw / ph
    if ratio > (w / h):
        new_w, new_h = w, w / ratio
    else:
        new_h, new_w = h, h * ratio
    return (x + (w - new_w) / 2, y + (h - new_h) / 2, new_w, new_h)


@dataclass
class SlideBuild:
    shapes_xml: list = field(default_factory=list)
    image_paths: list = field(default_factory=list)
    image_rids: list = field(default_factory=list)

    def add(self, xml):
        self.shapes_xml.append(xml)

    def add_image_rect(self, idg, x, y, w, h, path, *, caption=None,
                       caption_size=10, fit=True):
        if path.exists():
            rId = f"rId{len(self.image_rids) + 100}"
            self.image_paths.append(path)
            self.image_rids.append(rId)
            if fit:
                xi, yi, wi, hi = fit_image(path, x, y, w, h)
            else:
                xi, yi, wi, hi = x, y, w, h
            self.add(sp_picture(idg, xi, yi, wi, hi, rId))
        else:
            self.add(sp_rect(idg, x, y, w, h, TINT, line_color=BORDER))
            self.add(sp_textbox(idg, x, y + h / 2 - 0.2, w, 0.4,
                [Para([Run(f"[figure missing: {path.name}]",
                           size=11, color=SUBTLE, italic=True)], align="ctr")]))
        if caption:
            self.add(sp_textbox(idg, x, y + h + 0.05, w, 0.30,
                [Para([Run(caption, size=caption_size, color=MUTED, italic=True)], align="ctr")]))


TOTAL_SLIDES = 33


def add_content_chrome(b, idg, title, section, page, total):
    b.add(sp_rect(idg, 0, 0, SLIDE_W_IN, SLIDE_H_IN, PAPER))
    b.add(sp_rect(idg, M, 0.30, 0.42, 0.06, RUST))
    b.add(sp_rect_with_text(idg, M + 0.55, 0.24, 1.95, 0.24, TEAL,
        [Para([Run(section.upper(), size=9, bold=True, color=PAPER)], align="ctr")],
        anchor="ctr", margin=0.02))
    b.add(sp_textbox(idg, M, 0.55, SLIDE_W_IN - 2 * M, 0.85,
        [Para([Run(title, size=28, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_rect(idg, M, 1.40, SLIDE_W_IN - 2 * M, 0.012, BORDER))
    foot_y = SLIDE_H_IN - 0.40
    b.add(sp_textbox(idg, M, foot_y, 7.0, 0.3,
        [Para([Run("legoESM  ·  a differentiable Earth System Model in JAX",
                   size=9, color=SUBTLE, italic=True)])]))
    b.add(sp_textbox(idg, SLIDE_W_IN - M - 1.5, foot_y, 1.5, 0.3,
        [Para([Run(f"{page} / {total}", size=9, color=SUBTLE)], align="r")]))


def make_divider(section_no, section_name, blurb):
    b = SlideBuild(); idg = IdGen()
    b.add(sp_rect(idg, 0, 0, SLIDE_W_IN, SLIDE_H_IN, TEAL_DARK))
    b.add(sp_textbox(idg, 0.8, 1.5, 4.0, 2.5,
        [Para([Run(section_no, size=180, bold=True, color=RUST, font=HEADER_FONT)])]))
    b.add(sp_rect(idg, 5.0, 2.1, 0.04, 2.6, RUST))
    b.add(sp_textbox(idg, 5.4, 2.0, 7.5, 1.2,
        [Para([Run(section_name, size=44, bold=True, color=PAPER, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, 5.4, 3.2, 7.5, 2.5,
        [Para([Run(blurb, size=18, color=PALE_INK)], line_spacing=1.30)]))
    b.add(sp_textbox(idg, M, SLIDE_H_IN - 0.5, 8.0, 0.3,
        [Para([Run("legoESM  ·  Pierre Gentine et al.",
                   size=10, color=PALE_INK, italic=True)])]))
    return b


def bullets(items, *, size=13, color=INK, bullet_color=RUST,
            line_spacing=1.20, space_after_pt=5, bold_first=False):
    paras = []
    for item in items:
        if isinstance(item, tuple):
            lead, rest = item
            runs = [Run("▸  ", size=size, bold=True, color=bullet_color),
                    Run(lead, size=size, bold=True, color=color),
                    Run(rest, size=size, color=color)]
        else:
            txt = item
            if bold_first and ":" in txt:
                lead, rest = txt.split(":", 1)
                runs = [Run("▸  ", size=size, bold=True, color=bullet_color),
                        Run(lead + ":", size=size, bold=True, color=color),
                        Run(rest, size=size, color=color)]
            else:
                runs = [Run("▸  ", size=size, bold=True, color=bullet_color),
                        Run(txt, size=size, color=color)]
        paras.append(Para(runs, line_spacing=line_spacing, space_after_pt=space_after_pt))
    return paras


# ===========================================================================
# Slides
# ===========================================================================

def slide_title():
    b = SlideBuild(); idg = IdGen()
    b.add(sp_rect(idg, 0, 0, SLIDE_W_IN, SLIDE_H_IN, PAPER))
    b.add(sp_rect(idg, 0, SLIDE_H_IN - 1.4, SLIDE_W_IN, 1.4, TEAL_DARK))
    b.add(sp_rect(idg, 0, 0, SLIDE_W_IN, 0.16, RUST))
    logo_w = 4.6
    logo_h = 4.6
    logo_x = SLIDE_W_IN - M - logo_w
    logo_y = 0.85
    b.add_image_rect(idg, logo_x, logo_y, logo_w, logo_h, LOGO_PATH)
    text_w = logo_x - M - 0.35
    b.add(sp_textbox(idg, M, 1.1, text_w, 0.45,
        [Para([Run("DIFFERENTIABLE EARTH SYSTEM MODELING",
                   size=14, bold=True, color=RUST)])]))
    b.add(sp_textbox(idg, M, 1.6, text_w, 1.6,
        [Para([Run("legoESM", size=92, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, M, 3.4, text_w, 0.7,
        [Para([Run("A fully differentiable, modular",
                   size=24, italic=True, color=SLATE, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, M, 3.85, text_w, 0.7,
        [Para([Run("Earth System Model in JAX",
                   size=24, italic=True, color=SLATE, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, M, 4.75, text_w, 0.55,
        [Para([Run("Atmosphere · Ocean · Land · Sea ice · Coupler",
                   size=14, color=MUTED)])]))
    b.add(sp_textbox(idg, M, 5.15, text_w, 0.55,
        [Para([Run("End-to-end jax.grad through the coupled model",
                   size=14, color=MUTED)])]))
    b.add(sp_textbox(idg, M, SLIDE_H_IN - 1.10, 8.0, 0.5,
        [Para([Run("Pierre Gentine and collaborators",
                   size=18, bold=True, color=PAPER, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, M, SLIDE_H_IN - 0.60, 8.0, 0.3,
        [Para([Run("Columbia University  ·  2026",
                   size=12, italic=True, color=PALE_INK)])]))
    b.add(sp_textbox(idg, SLIDE_W_IN - 4.5 - M, SLIDE_H_IN - 0.85, 4.5, 0.5,
        [Para([Run("github.com/climate-federation/legoESM", size=12, color=PALE_INK)], align="r")]))
    return b


def slide_outline():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Talk outline", "intro", 2, TOTAL_SLIDES)
    parts = [
        ("01", "Why a differentiable ESM",
         "Motivation, design principles, where legoESM sits in the model landscape."),
        ("02", "Modularity across components",
         "Atmosphere · ocean · land · ice · coupler · grids — interchangeable lego bricks under a common JAX interface."),
        ("03", "Applications",
         "AMIP / AIMIP differentiable tuning · NeuralGCM + SFNO hybrids · slab-S2S · SCM · RCE / RCEMIP · CRM · ocean experiments."),
        ("04", "Validation strategy",
         "Dycore progression suite, Williamson + DCMIP + Held-Suarez, conservation diagnostics, ocean fidelity vs Veros."),
    ]
    y0 = 1.85; row_h = 1.20
    for i, (num, label, blurb) in enumerate(parts):
        y = y0 + i * row_h
        b.add(sp_textbox(idg, M, y, 1.0, 0.9,
            [Para([Run(num, size=46, bold=True, color=RUST, font=HEADER_FONT)])]))
        b.add(sp_rect(idg, M + 1.05, y + 0.10, 0.02, row_h - 0.30, TEAL))
        b.add(sp_textbox(idg, M + 1.25, y - 0.05, SLIDE_W_IN - 2 * M - 1.25, 0.55,
            [Para([Run(label, size=22, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
        b.add(sp_textbox(idg, M + 1.25, y + 0.55, SLIDE_W_IN - 2 * M - 1.25, 0.6,
            [Para([Run(blurb, size=13, color=MUTED)])]))
    return b


def slide_motivation():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Why a differentiable Earth System Model",
                       "motivation", 3, TOTAL_SLIDES)
    lx, lw = M, 6.6
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Three converging pressures on traditional ESMs",
                   size=17, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.15, lw, 4.5,
        bullets([
            "Parameter tuning is the bottleneck: cloud / convection / mixing constants are calibrated by hand against scarce observations, with no gradients.",
            "AI weather models (GraphCast, FourCastNet, ACE) are fast and skillful — but pure black boxes, no conservation, no prognostic ocean or land.",
            "Hybrid AI–physics requires differentiating through the dynamics: today this means writing a separate adjoint code (E3SM / MITgcm) or living without one.",
            "Modern accelerators (GPU / TPU) reward array-language ESMs; the Fortran / MPI stack is increasingly off the critical path.",
        ], size=13.5, line_spacing=1.22, space_after_pt=7)))
    rx = lx + lw + 0.35
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.70, rw, 4.95, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 1.85, rw - 0.5, 0.4,
        [Para([Run("Our hypothesis", size=15, bold=True, color=RUST, font=HEADER_FONT)])]))
    body = (
        "A modern Earth System Model should be:\n"
        "  · written in one differentiable array language\n"
        "  · modular across components, grids, and complexities\n"
        "  · GPU / TPU / CPU / MPI portable\n"
        "  · conservative by construction\n"
        "  · open to both classical physics and ML cores under the same interface\n\n"
        "Then parameter estimation, 4D-Var, sensitivity, and hybrid ML become "
        "first-class workflows — not retrofits."
    )
    paras = []
    for line in body.split("\n"):
        if line == "":
            paras.append(Para([], line_spacing=1.0, space_after_pt=4))
        else:
            paras.append(Para([Run(line, size=13, color=INK)],
                              line_spacing=1.25, space_after_pt=3))
    b.add(sp_textbox(idg, rx + 0.25, 2.30, rw - 0.5, 4.30, paras))
    return b


def slide_design_principles():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Five design principles", "principles", 4, TOTAL_SLIDES)
    principles = [
        ("Differentiable-first",
         "End-to-end jax.grad / eqx.filter_value_and_grad through the coupled model — including MPI halo exchange and global reductions."),
        ("Conservation by construction",
         "Mass conserved exactly; energy & momentum tracked diagnostically with explicit budget closure."),
        ("Modular & swappable",
         "Standard tensor-in / tendency-out interfaces; dynamical cores, physics, grids, and time integrators all hot-swappable."),
        ("Multi-grid native",
         "Cubed-sphere C-D · lat-lon C · Gaussian / spectral · MPAS Voronoi — shared FV / spectral operators where numerics permit."),
        ("AI-ready",
         "SFNO neural cores and ML parameterizations coexist with classical physics under the same interface. ERA5 ingestion built in."),
    ]
    card_w, card_h = 3.92, 2.05
    gap = 0.20
    x0_r1 = (SLIDE_W_IN - 3 * card_w - 2 * gap) / 2
    y_r1 = 1.75
    x0_r2 = (SLIDE_W_IN - 2 * card_w - gap) / 2
    y_r2 = y_r1 + card_h + 0.30

    def draw_card(x, y, title, body, idx):
        b.add(sp_rect(idg, x, y, card_w, card_h, PAPER, line_color=BORDER))
        b.add(sp_rect_with_text(idg, x + 0.16, y + 0.16, 0.30, 0.30, RUST,
            [Para([Run(str(idx), size=14, bold=True, color=PAPER)], align="ctr")],
            anchor="ctr", margin=0))
        b.add(sp_textbox(idg, x + 0.60, y + 0.16, card_w - 0.80, 0.40,
            [Para([Run(title, size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
        b.add(sp_textbox(idg, x + 0.20, y + 0.65, card_w - 0.40, card_h - 0.80,
            [Para([Run(body, size=11.5, color=MUTED)], line_spacing=1.20)]))

    for i, (t, body) in enumerate(principles[:3]):
        draw_card(x0_r1 + i * (card_w + gap), y_r1, t, body, i + 1)
    for i, (t, body) in enumerate(principles[3:]):
        draw_card(x0_r2 + i * (card_w + gap), y_r2, t, body, i + 4)
    return b


def slide_landscape():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Where legoESM sits in the modeling landscape",
                       "landscape", 5, TOTAL_SLIDES)
    rows = [
        ["E3SM / CESM / ICON",  "Fortran + MPI",          "Multiple, hand-coded",          "Separate adjoint",       "Yes",                               "Bolt-on"],
        ["MOM6 / NEMO",         "Fortran",                "Ocean only",                    "Limited",                "—",                                 "—"],
        ["NeuralGCM",           "JAX",                    "Spectral PE + NN physics",      "Yes (built-in)",         "No prognostic ocean / ice",         "Native"],
        ["GraphCast / ACE",     "PyTorch / JAX",          "Learned (no dycore)",           "Training only",          "No prognostic ocean / ice",         "Native"],
        ["legoESM",             "JAX (single lang.)",     "4 grids × {SW, hydro, NH, SFNO}", "End-to-end jax.grad",  "Atm + ocean + land + ice + lake",   "First-class"],
    ]
    headers = ["Model", "Language", "Dynamics", "Differentiability", "Prognostic components", "ML cores"]
    cx = M
    cw_raw = [2.10, 1.90, 2.45, 2.20, 2.40, 1.30]
    scale = (SLIDE_W_IN - 2 * M) / sum(cw_raw)
    cw = [c * scale for c in cw_raw]
    y0 = 1.80; rh = 0.78
    b.add(sp_rect(idg, cx, y0, sum(cw), 0.42, TEAL))
    x = cx
    for i, h in enumerate(headers):
        b.add(sp_textbox(idg, x + 0.10, y0, cw[i] - 0.15, 0.42,
            [Para([Run(h, size=12, bold=True, color=PAPER)], align="ctr")],
            anchor="ctr", margin=0))
        x += cw[i]
    for ri, row in enumerate(rows):
        y = y0 + 0.42 + ri * rh
        bg = PAPER if ri % 2 == 0 else TINT
        is_us = "legoESM" in row[0]
        if is_us:
            bg = "FDEADC"
        b.add(sp_rect(idg, cx, y, sum(cw), rh, bg))
        x = cx
        for i, cell in enumerate(row):
            run_color = TEAL_DARK if (is_us and i == 0) else INK
            bold = is_us and i == 0
            b.add(sp_textbox(idg, x + 0.10, y, cw[i] - 0.15, rh,
                [Para([Run(cell, size=12, bold=bold, color=run_color)], align="ctr")],
                anchor="ctr", margin=0))
            x += cw[i]
    cap_y = y0 + 0.42 + len(rows) * rh + 0.15
    b.add(sp_textbox(idg, M, cap_y, SLIDE_W_IN - 2 * M, 0.6,
        [Para([Run("legoESM is the only stack in this row that combines a full "
                   "classical-physics + dynamical-core ladder with a prognostic "
                   "ocean / ice / land and end-to-end differentiability.",
                   size=12, italic=True, color=MUTED)], align="ctr", line_spacing=1.25)]))
    return b


def slide_architecture():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "High-level architecture", "architecture", 6, TOTAL_SLIDES)
    blocks_top = [
        ("Atmosphere", COL_ATM, "atmosphere/",
         "SW · hydro · NH · SFNO   ·   radiation · convection · turbulence · clouds · microphysics · GWD"),
        ("Ocean", COL_OCN, "ocean/",
         "PE on lat-lon C · cubed-sphere · MPAS · spectral   ·   KPP · GM-Redi · BGC · barotropic split"),
        ("Land", COL_LND, "land/",
         "slab · multilayer Richards (8L) · soil thermal · stomata · DALEC-990 carbon"),
    ]
    blocks_bot = [
        ("Sea ice", COL_ICE, "ice/",
         "thermo · EVP / mEVP rheology · ITD (Lipscomb) · snow · brine · ridging · ponds"),
        ("Coupler", COL_CPL, "coupler/",
         "tile-based (ocean / ice / land / lake)   ·   COARE 3.0 · L–Y bulk fluxes   ·   surface energy + accumulator"),
        ("DA · ML · training", COL_DA, "da/ · ml/ · training/",
         "incremental 4D-Var   ·   SFNO · NeuralGCM · S2S slab   ·   ERA5 → state · trainable physics"),
    ]
    bw = (SLIDE_W_IN - 2 * M - 0.4) / 3
    bh = 1.95
    y_top = 1.70
    y_bot = y_top + bh + 0.35
    bar_y = y_top + bh + 0.05
    b.add(sp_rect_with_text(idg, M, bar_y, SLIDE_W_IN - 2 * M, 0.25, TEAL,
        [Para([Run("ParallelRuntime   ·   compiled_segments (lax.scan)   ·   pytree state   ·   MPI custom_vjp halo + allreduce(SUM)",
                   size=11, bold=True, color=PAPER)], align="ctr")],
        anchor="ctr", margin=0))

    def draw_block(x, y, w, h, title, fill, pkg, body):
        b.add(sp_rect(idg, x, y, w, h, fill, line_color=BORDER))
        b.add(sp_textbox(idg, x + 0.20, y + 0.12, w - 0.40, 0.40,
            [Para([Run(title, size=18, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
        b.add(sp_textbox(idg, x + 0.20, y + 0.55, w - 0.40, 0.30,
            [Para([Run(pkg, size=11, color=RUST, font=MONO_FONT)])]))
        b.add(sp_textbox(idg, x + 0.20, y + 0.90, w - 0.40, h - 1.00,
            [Para([Run(body, size=11, color=INK)], line_spacing=1.25)]))

    for i, (t, f, p, body) in enumerate(blocks_top):
        draw_block(M + 0.2 + i * bw, y_top, bw - 0.05, bh, t, f, p, body)
    for i, (t, f, p, body) in enumerate(blocks_bot):
        draw_block(M + 0.2 + i * bw, y_bot, bw - 0.05, bh, t, f, p, body)
    b.add(sp_textbox(idg, M, y_bot + bh + 0.10, SLIDE_W_IN - 2 * M, 0.35,
        [Para([Run("Every block ships its own configs and validation; cores, "
                   "schemes and grids are independently swappable.",
                   size=12, italic=True, color=MUTED)], align="ctr")]))
    return b


def slide_jax_foundations():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "JAX foundations: state · scan · autodiff",
                       "foundations", 7, TOTAL_SLIDES)
    lx, lw = M, 6.4
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Model as a pure function over pytrees",
                   size=16, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.15, lw, 4.6,
        bullets([
            "State as NamedTuple pytrees: SegmentCarry, OceanState, LatLonCGridOceanState, Field…",
            "Time integration is jax.lax.scan over a single-step function — no Python loops on traced values.",
            "Hot loop compiled in segments of length GCD(diagnostic interval, checkpoint interval).",
            "External forcings (SST, solar, ozone) passed as explicit traced args (SegmentForcing) so the kernel is reused, not retraced.",
            "Each segment has a JIT (donate_argnums) variant for production and a .raw non-donating variant for eqx.filter_value_and_grad.",
        ], size=13.5, line_spacing=1.22, space_after_pt=6)))
    rx = lx + lw + 0.40
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.70, rw, 4.95, TEAL_DARK))
    b.add(sp_textbox(idg, rx + 0.35, 1.85, rw - 0.7, 0.4,
        [Para([Run("training step  (sketch)", size=12, italic=True,
                   color=PALE_INK, font=HEADER_FONT)])]))
    code_lines = [
        "def loss_fn(theta, state0, forcing):",
        "    final = run_segment.raw(",
        "        theta, state0, forcing,",
        "        n_steps=N, dt=dt)",
        "    return area_weighted_mse(",
        "        final.state, target)",
        "",
        "grads = eqx.filter_value_and_grad(",
        "    loss_fn)(theta, state0, forcing)",
        "",
        "# end-to-end jax.grad — through",
        "# dycore + physics + halo + reductions",
    ]
    paras = [Para([Run(l, size=14, color=LIGHT_GREY, font=MONO_FONT)],
                  line_spacing=1.25, space_after_pt=0) for l in code_lines]
    b.add(sp_textbox(idg, rx + 0.35, 2.30, rw - 0.7, 4.4, paras))
    return b


def slide_ad_story():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "End-to-end differentiability through MPI",
                       "differentiability", 8, TOTAL_SLIDES)
    y = 1.80
    stages = ["state₀", "halo exchange\n(MPI)", "dynamics\n(lax.scan)",
              "physics\n(scan)", "global sum\n(allreduce)", "loss"]
    n = len(stages)
    sw = SLIDE_W_IN - 2 * M
    box_w = (sw - (n - 1) * 0.20) / n
    box_h = 0.95
    for i, name in enumerate(stages):
        x = M + i * (box_w + 0.20)
        fill = RUST if i in (1, 4) else TEAL
        b.add(sp_rect_with_text(idg, x, y, box_w, box_h, fill,
            [Para([Run(name, size=12, bold=True, color=PAPER)], align="ctr", line_spacing=1.10)],
            anchor="ctr", margin=0.05))
        if i < n - 1:
            ax = x + box_w - 0.05
            b.add(sp_textbox(idg, ax, y + 0.10, 0.30, 0.35,
                [Para([Run("→", size=18, bold=True, color=INK)], align="ctr")]))
            b.add(sp_textbox(idg, ax, y + 0.50, 0.30, 0.35,
                [Para([Run("←", size=18, bold=True, color=RUST)], align="ctr")]))
    b.add(sp_textbox(idg, M, y + 1.05, 6, 0.3,
        [Para([Run("→  primal trace          ", size=12, color=INK),
               Run("←  cotangent (jax.grad)", size=12, bold=True, color=RUST)])]))
    by = y + 1.5
    b.add(sp_textbox(idg, M, by, SLIDE_W_IN - 2 * M, 0.4,
        [Para([Run("What's non-trivial in this picture",
                   size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, M, by + 0.45, SLIDE_W_IN - 2 * M, 3.0,
        bullets([
            "MPI halo exchange has no native VJP. We wrap mpi4jax.sendrecv in a custom_vjp whose backward swaps source / dest endpoints — cotangents flow back to the sender (parallel/halo_exchange.py).",
            "Only allreduce(SUM) is AD-safe among reductions. global_max / global_min stay diagnostic-only; the differentiable path uses sums and softmax surrogates.",
            "Mass / moisture / energy fixers are written as projections — gradients flow through the global reductions used to compute the correction.",
            "JIT donate_argnums conflicts with reverse-mode AD; the training path uses non-donating .raw variants of compiled segments.",
        ], size=13, line_spacing=1.20, space_after_pt=4)))
    return b


def slide_conservation():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Conservation as a hard constraint",
                       "conservation", 9, TOTAL_SLIDES)
    lx, lw = M, 6.4
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Built into the time integrator",
                   size=16, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.15, lw, 4.5,
        bullets([
            "Mass exactly conserved by construction (target-anchored fixer in SegmentCarry).",
            "Moisture conserved via fix_moisture + zero_mean_tendency, with gradient-safe global means.",
            "Energy budget diagnostics (column DSE / MSE / TOA-SFC residual) — never just hidden clipping.",
            "Tracer monotonicity preserved by PPM transport with positive-definite limiter.",
            "ML cores get the same treatment: conservation correctors (dry mass, moisture, ocean volume / heat / salt) act as differentiable projections.",
        ], size=13.5, line_spacing=1.20, space_after_pt=6)))
    rx = lx + lw + 0.40
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.70, rw, 4.95, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 1.85, rw - 0.5, 0.4,
        [Para([Run("Diagnostics shipped in diagnostics/",
                   size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    items = [
        ("energy_budget.py",       "EnergyBudget, column DSE / MSE, TOA & SFC net radiation"),
        ("total_energy_pe.py",     "Hydrostatic total energy, te_correction, te_drift"),
        ("angular_momentum.py",    "Atmospheric angular momentum + corrections"),
        ("column_integrals.py",    "column_water_vapor, mass-weighted means"),
        ("conservation_drift.py",  "Relative-drift series + tolerance gating"),
        ("precision_drift.py",     "Tracer negativity + state diff tools"),
    ]
    y = 2.35
    for fname, desc in items:
        b.add(sp_textbox(idg, rx + 0.30, y, 2.5, 0.30,
            [Para([Run(fname, size=12, bold=True, color=TEAL_DARK, font=MONO_FONT)])]))
        b.add(sp_textbox(idg, rx + 2.85, y, rw - 3.0, 0.30,
            [Para([Run(desc, size=12, color=INK)])]))
        y += 0.55
    return b


def slide_modularity_intro():
    return make_divider("01", "Modularity across Earth components",
        "Every Earth-system block has its own swappable cores and physics "
        "schemes — same interfaces, different numerics, from the simplest "
        "aquaplanet to coupled CMIP-style runs.")


def slide_atmosphere_dycores():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Atmosphere — dynamical cores",
                       "modularity · atm", 11, TOTAL_SLIDES)
    headers = ["", "Lat-lon C-grid", "Cubed-sphere C-D", "Gaussian / spectral", "MPAS Voronoi"]
    rows = [
        ("Shallow water",      "shallow_water_latlon_cgrid", "shallow_water_fv3_cdgrid", "spectral_sw", "shallow_water_mpas"),
        ("Hydrostatic PE",     "primitive_eq_latlon_cgrid",   "primitive_eq_cdgrid",       "spectral_pe", "primitive_eq_mpas"),
        ("Non-hydrostatic CE", "—",                            "compressible_euler_cdgrid", "spectral_nh", "compressible_euler_mpas"),
        ("Neural (SFNO)",      "sfno_pe (channel-packed)",     "sfno_pe / sfno_sw",         "sfno (spherical)", "—"),
    ]
    cx, cy = M, 1.85
    cw_h = 2.30
    cw_b = (SLIDE_W_IN - 2 * M - cw_h) / 4
    rh = 0.85
    b.add(sp_rect_with_text(idg, cx, cy, cw_h, 0.45, TEAL,
        [Para([Run(headers[0], size=12, bold=True, color=PAPER)], align="ctr")],
        anchor="ctr", margin=0))
    for i, h in enumerate(headers[1:]):
        x = cx + cw_h + i * cw_b
        b.add(sp_rect_with_text(idg, x, cy, cw_b, 0.45, TEAL,
            [Para([Run(h, size=12, bold=True, color=PAPER)], align="ctr")],
            anchor="ctr", margin=0))
    for ri, row in enumerate(rows):
        y = cy + 0.45 + ri * rh
        bg = TINT if ri % 2 == 0 else PAPER
        b.add(sp_rect_with_text(idg, cx, y, cw_h, rh, bg,
            [Para([Run(row[0], size=14, bold=True, color=TEAL_DARK, font=HEADER_FONT)])],
            anchor="ctr", margin=0.15))
        for ci in range(1, 5):
            x = cx + cw_h + (ci - 1) * cw_b
            cell = row[ci]
            is_missing = cell == "—"
            b.add(sp_rect_with_text(idg, x, y, cw_b, rh, bg, line_color=BORDER,
                paras=[Para([Run(cell, size=10.5,
                                 color=SUBTLE if is_missing else INK,
                                 font=MONO_FONT)], align="ctr")],
                anchor="ctr", margin=0.05))
    cap_y = cy + 0.45 + len(rows) * rh + 0.15
    b.add(sp_textbox(idg, M, cap_y, SLIDE_W_IN - 2 * M, 0.55,
        [Para([Run("4 grid families × 4 complexities. Tracer transport "
                   "(RK3, 3rd-order PPM) and the physics interface are "
                   "grid-agnostic; SFNO cores use the same SegmentCarry as "
                   "the classical ones.",
                   size=12, italic=True, color=MUTED)],
              align="ctr", line_spacing=1.25)]))
    return b


def slide_atmosphere_physics():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Atmosphere — physics zoo",
                       "modularity · atm", 12, TOTAL_SLIDES)
    cats = [
        ("Radiation",        "radiation/",
         "gray (Frierson) · RRTMGP (LW g128 / g256, SW g112 / g224) · ozone_ml · solar (cos-zenith)"),
        ("Convection",       "convection/",
         "Bechtold · Tiedtke · Emanuel · Kain–Fritsch · Zhang–McFarlane · SBM (Frierson BM) · DCA · Kuo · shared mass_flux + plume + smooth triggers"),
        ("Microphysics",     "microphysics/",
         "Kessler · Sundqvist · Seifert–Beheng · Morrison · Thompson · ml_emulator (Equinox MLP)"),
        ("Cloud fraction",   "clouds/", "Sundqvist · Xu–Randall"),
        ("Turbulence / PBL", "turbulence/",
         "YSU · EDMF · TKE (MY2.5) · Louis · CLUBB-lite · Holtslag–Boville · Smagorinsky · MOST surface layer"),
        ("Gravity-wave drag", "gravity_wave_drag/",
         "Rayleigh · Lindzen · McFarlane · Hines · prognostic_spectral · ml_emulator"),
    ]
    cw, ch = 4.05, 1.65
    gx = (SLIDE_W_IN - 3 * cw - 2 * 0.20) / 2
    gy = 1.75
    for i, (name, pkg, body) in enumerate(cats):
        r, c = divmod(i, 3)
        x = gx + c * (cw + 0.20)
        y = gy + r * (ch + 0.25)
        b.add(sp_rect(idg, x, y, cw, ch, PAPER, line_color=BORDER))
        b.add(sp_rect(idg, x, y, 0.08, ch, RUST))
        b.add(sp_textbox(idg, x + 0.20, y + 0.10, cw - 0.30, 0.40,
            [Para([Run(name, size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
        b.add(sp_textbox(idg, x + 0.20, y + 0.50, cw - 0.30, 0.30,
            [Para([Run(pkg, size=10, color=RUST, font=MONO_FONT)])]))
        b.add(sp_textbox(idg, x + 0.20, y + 0.82, cw - 0.30, ch - 0.85,
            [Para([Run(body, size=10.5, color=INK)], line_spacing=1.22)]))
    b.add(sp_textbox(idg, M, gy + 2 * ch + 0.55, SLIDE_W_IN - 2 * M, 0.45,
        [Para([Run("Every scheme exposes a NamedTuple config → integration.py "
                   "factory → tendency function — interchangeable across all "
                   "atmosphere dycores.",
                   size=12, italic=True, color=MUTED)], align="ctr")]))
    return b


def slide_ocean_dycores():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Ocean — dynamical cores",
                       "modularity · ocean", 13, TOTAL_SLIDES)
    lx, lw = M, 6.5
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Same prognostic ocean PE across four grids",
                   size=16, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.20, lw, 4.6,
        bullets([
            "Lat-lon C-grid (FV, Sadourny EC vector-invariant; MOM6-style slow-forcing split): ocean_pe_latlon_cgrid.py",
            "Cubed-sphere C-D (Boussinesq, Lin 2004 + Griffies 2004): ocean_pe_cdgrid.py",
            "MPAS Voronoi (TRiSK, Ringler 2010): ocean_pe_mpas.py",
            "Spectral: spectral_ocean_pe.py",
            "Learned ocean (SFNO): sfno_ocean.py — packed PE state ↔ tensors via channel_packing",
            "Shared baroclinic helpers (ocean_tendency_common): EOS-pressure iteration, tracer sponge, freshwater virtual-salt, implicit bottom drag",
            "Shared barotropic helpers (barotropic_common): cosine / box filter, BEBT eta blend, MAXVEL clip",
        ], size=12.5, line_spacing=1.18, space_after_pt=4)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.70, rw, 4.95, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 1.85, rw - 0.5, 0.4,
        [Para([Run("Vertical & solvers", size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, rx + 0.25, 2.35, rw - 0.5, 4.30,
        bullets([
            "z-star vertical coordinate (ocean/vertical.py)",
            "Wright (1997) EOS + Boussinesq",
            "Pressure gradient: AHH08, SMC03 variants",
            "Implicit + explicit barotropic solvers across all grids",
            "Tridiagonal vertical implicit mixing",
            "Tripolar eORCA1 grid with pole-fold halo",
        ], size=12, line_spacing=1.22, space_after_pt=4)))
    return b


def slide_ocean_physics():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Ocean — physics & biogeochemistry",
                       "modularity · ocean", 14, TOTAL_SLIDES)
    cats = [
        ("Vertical mixing",      "vertical_mixing/",
         "KPP (LMD94, non-local) · Richardson · constant · Jayne–St-Laurent tidal · implicit tridiagonal"),
        ("Lateral mixing",       "lateral_mixing/",
         "Harmonic · biharmonic · GM-Redi (cube / lat-lon / MPAS) with DM95 slope tapering · Visbeck adaptive κ · eddy backscatter"),
        ("Bottom drag",          "bottom_drag/",
         "Linear & quadratic + implicit factor in ocean_tendency_common"),
        ("Surface forcing",      "surface_forcing/",
         "COARE 3.0 · Large–Yeager 09 · fixed-z0 · restoring · prescribed · wind_profiles"),
        ("Convection",           "convection/",
         "Plume + enhanced diffusion (gradient-safe sigmoid active mask)"),
        ("Shortwave penetration","shortwave_penetration.py",
         "Jerlov optical-water types"),
        ("Ice-shelf basal melt", "ice_shelf.py",
         "Jenkins 1991 / Holland-Jenkins 1999 three-equation (ISOMIP+)"),
        ("Biogeochemistry",      "biogeochemistry/",
         "Abiotic DIC + ALK carbonate · NPZD · air–sea CO₂ Schmidt-number gas exchange"),
    ]
    cw, ch = 6.00, 0.95
    gx = M
    gy = 1.78
    for i, (name, pkg, body) in enumerate(cats):
        r, c = divmod(i, 2)
        x = gx + c * (cw + 0.30)
        y = gy + r * (ch + 0.18)
        b.add(sp_rect(idg, x, y, cw, ch, PAPER, line_color=BORDER))
        b.add(sp_rect(idg, x, y, 0.07, ch, TEAL))
        b.add(sp_textbox(idg, x + 0.18, y + 0.08, cw - 0.30, 0.30,
            [Para([Run(name, size=13, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
        b.add(sp_textbox(idg, x + 0.18, y + 0.36, cw - 0.30, 0.28,
            [Para([Run(pkg, size=10, color=RUST, font=MONO_FONT)])]))
        b.add(sp_textbox(idg, x + 0.18, y + 0.62, cw - 0.30, ch - 0.65,
            [Para([Run(body, size=10.5, color=INK)], line_spacing=1.15)]))
    b.add(sp_textbox(idg, M, gy + 4 * (ch + 0.18) + 0.10, SLIDE_W_IN - 2 * M, 0.45,
        [Para([Run("Ocean test matrix: 57 / 57 PASS across lat-lon, "
                   "tripolar eORCA1, cubed-sphere C-D, MPAS Voronoi.",
                   size=12, bold=True, italic=True, color=RUST)], align="ctr")]))
    return b


def slide_land():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Land — slab to multilayer Richards + carbon",
                       "modularity · land", 15, TOTAL_SLIDES)
    lx, lw = M, 6.6
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Two complexity tiers under one interface",
                   size=16, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.20, lw, 4.6,
        bullets([
            "Slab land (land/slab_land.py): energy balance + bucket hydrology + snow + stomata (Farquhar / Ball-Berry / Medlyn / Jarvis).",
            "Multilayer land (land/multilayer_land.py): 8-layer soil, Richards equation (mixed-form Picard, Van Genuchten / Clapp-Hornberger / Brooks-Corey / Campbell / PDI / Lu).",
            "Soil thermal: Johansen diffusion · tridiagonal vertical solver.",
            "Snow: accumulation / melt with age-dependent albedo and latitude-varying veg albedo.",
            "Carbon cycle (DALEC-990): 6 pools (labile / foliage / root / wood / litter / SOM), LUE GPP + phenology.",
            "PFT-weighted parameter providers: Constant · PFT · Neural — all differentiable.",
        ], size=12.5, line_spacing=1.20, space_after_pt=5)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.70, rw, 4.95, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 1.85, rw - 0.5, 0.4,
        [Para([Run("8-layer soil column", size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    layer_x = rx + 0.6
    layer_w = rw - 1.2
    layer_h = 0.42
    top_y = 2.40
    layers = [
        ("L1", "0.00 – 0.02 m"), ("L2", "0.02 – 0.05 m"),
        ("L3", "0.05 – 0.10 m"), ("L4", "0.10 – 0.20 m"),
        ("L5", "0.20 – 0.40 m"), ("L6", "0.40 – 0.80 m"),
        ("L7", "0.80 – 1.60 m"), ("L8", "1.60 – 3.20 m"),
    ]
    fill_colors = ["E7C89A", "D3AC7F", "BC9166", "A6784E",
                   "8B5F3A", "6F492A", "53361E", "392514"]
    for i, (lab, depth) in enumerate(layers):
        y = top_y + i * (layer_h + 0.02)
        b.add(sp_rect_with_text(idg, layer_x, y, 1.0, layer_h, fill_colors[i],
            [Para([Run(lab, size=11, bold=True, color=PAPER, font=MONO_FONT)], align="ctr")],
            anchor="ctr", margin=0))
        b.add(sp_rect_with_text(idg, layer_x + 1.05, y, layer_w - 1.05, layer_h, fill_colors[i],
            [Para([Run(depth, size=11, color=PAPER)], align="ctr")],
            anchor="ctr", margin=0))
    return b


def slide_seaice():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Sea ice — thermo + EVP/mEVP rheology + ITD",
                       "modularity · ice", 16, TOTAL_SLIDES)
    lx, lw = M, 6.5
    b.add(sp_textbox(idg, lx, 1.80, lw, 4.6,
        bullets([
            "Step function ice/sea_ice.py — modes: slab · free-drift · EVP (Hunke & Dukowicz 1997) · mEVP (Bouillon 2013 / Kimmritz 2015).",
            "Rheology: ice strength, strain rates, delta-deformation, VP stress, EVP / mEVP updates (ice/rheology.py, ice/dynamics.py).",
            "Ice-thickness distribution (ITD): Lipscomb 2001 linear remapping for multi-category ice (ice/itd.py).",
            "Transport: advection + boundary masking. Snow on ice + flooding (ice/snow.py).",
            "Brine pockets + salt flux to ocean (ice/brine.py). Ridging (ice/ridging.py). Melt ponds (ice/ponds.py).",
            "Delta-Eddington shortwave + Maykut–Untersteiner albedo (ice/shortwave.py).",
            "Standalone test matrix: 15 benchmarks across thermo, dynamics, transport, ITD, integration.",
        ], size=12.5, line_spacing=1.20, space_after_pt=5)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.80, rw, 4.85, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 1.95, rw - 0.5, 0.4,
        [Para([Run("Why EVP matters for AD",
                   size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    body_lines = [
        "Sea-ice rheology in CICE-style models is solved implicitly — elliptic problems are awkward to differentiate.",
        "",
        "EVP / mEVP recast the implicit problem as a sub-cycled explicit pseudo-time integration. Every sub-step is a pure JAX operation, so the whole rheology solver is naturally inside jax.lax.scan and inside jax.grad.",
        "",
        "Trade-off: more sub-steps for the same stiffness — but each is fully GPU-friendly and AD-friendly.",
    ]
    paras = []
    for line in body_lines:
        if line == "":
            paras.append(Para([], line_spacing=1.0, space_after_pt=4))
        else:
            paras.append(Para([Run(line, size=12, color=INK)],
                              line_spacing=1.30, space_after_pt=3))
    b.add(sp_textbox(idg, rx + 0.25, 2.45, rw - 0.5, 4.10, paras))
    return b


def slide_coupler_grids():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Coupler and grids", "modularity · coupler", 17, TOTAL_SLIDES)
    lx, lw = M, 6.0
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Coupler — tile-based surface exchange",
                   size=16, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.20, lw, 2.6,
        bullets([
            "TileFractions (ocean / ice / land / lake) blended area-weighted in coupler/coupler.py",
            "AtmToSurface / SurfaceToAtm / TileResponse typed pytrees",
            "Bulk fluxes (coupler/bulk_flux.py): COARE 3.0 · Large–Yeager · fixed-z0 with AD-safe Monin–Obukhov iteration via lax.fori_loop",
            "Surface energy: zenith-dependent ocean albedo (Briegleb 1992), snow age decay, sea-ice T feedback",
            "FluxAccumulator with conservation-aware accumulation",
            "Two-layer lake model (coupler/lake/) with surface coupling",
        ], size=12, line_spacing=1.18, space_after_pt=4)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_textbox(idg, rx, 1.70, rw, 0.4,
        [Para([Run("Grids — geometry + halo + regrid",
                   size=16, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, rx, 2.20, rw, 2.6,
        bullets([
            "Cubed-sphere C-D (Putman & Lin 2007): corner-aware halo, FV3 D-grid corner fill, edge blending",
            "Lat-lon C-grid (regular · Mercator · regional), Tripolar eORCA1 with pole-fold halo descriptor",
            "Gaussian + spherical-harmonic transforms (T21 / T42 / …) — pinned to CPU on Apple Silicon",
            "MPAS Voronoi SCVT: recursive bisection + Lloyd; LSQ edge-to-cell + Thuburn kite-area TRiSK",
            "Vertical: hybrid sigma-pressure L20–L60, sinh stretching",
            "Native 4D halo: one MPI message per exchange across all levels",
        ], size=12, line_spacing=1.18, space_after_pt=4)))
    by = 4.95
    b.add(sp_rect(idg, M, by, SLIDE_W_IN - 2 * M, 0.012, BORDER))
    b.add(sp_textbox(idg, M, by + 0.10, SLIDE_W_IN - 2 * M, 0.40,
        [Para([Run("Shared operators (core/)", size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, M, by + 0.50, SLIDE_W_IN - 2 * M, 1.4,
        [Para([Run("core/operators_cdgrid.py (FV3 PPM + divergence damping) "
                   "is reused by both atmosphere and ocean cubed-sphere cores. "
                   "core/operators_fv_latlon.py / operators_fv_latlon_3d.py "
                   "serve all lat-lon FV cores. core/operators_voronoi.py "
                   "serves MPAS atmosphere + ocean. core/fv_tp_2d.py is the "
                   "canonical PPM transport step. Spectral and FC-Gram "
                   "operators live in core/spectral.py and core/fc_gram.py.",
                   size=12, color=INK)], line_spacing=1.30)]))
    return b


def slide_parallel():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Parallel runtime — MPI + SPMD + ensemble",
                       "runtime", 18, TOTAL_SLIDES)
    lx, lw = M, 6.6
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("ParallelRuntime.create() — one entry point",
                   size=16, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    modes = [
        ("serial",       "1 rank · 1 device — identity ops"),
        ("multi_device", "1 rank · N devices — JAX SPMD halos + local reductions"),
        ("mpi",          "N ranks · 1 device each — MPI halos + MPI reductions"),
        ("hybrid",       "N ranks × M devices — MPI between ranks + SPMD within rank"),
    ]
    y = 2.20
    for mode, desc in modes:
        b.add(sp_rect_with_text(idg, lx, y, 1.7, 0.42, TEAL,
            [Para([Run(mode, size=12, bold=True, color=PAPER, font=MONO_FONT)], align="ctr")],
            anchor="ctr", margin=0))
        b.add(sp_textbox(idg, lx + 1.85, y, lw - 1.95, 0.42,
            [Para([Run(desc, size=12, color=INK)])], anchor="ctr"))
        y += 0.55
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.70, rw, 4.95, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 1.85, rw - 0.5, 0.4,
        [Para([Run("AD-safe halo + reductions", size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, rx + 0.25, 2.30, rw - 0.5, 4.30,
        bullets([
            "custom_vjp wrapper around mpi4jax.sendrecv: backward swaps source / dest endpoints",
            "Differentiable reductions: only allreduce(SUM); max / min / allgather / bcast diagnostic-only",
            "Native 4D halo: one MPI message per exchange — not vmap(pad_halo)",
            "Async halo + compute overlap (parallel/async_halo.py)",
            "Voronoi RCB + METIS partitioning · VoronoiHaloExchange",
            "Ensemble parallelism: vmap over ensemble dim × scan over time + checkpoint",
            "Persistent JAX compile cache via LEGOESM_JAX_CACHE_DIR",
        ], size=11.5, line_spacing=1.20, space_after_pt=3)))
    return b


def slide_applications_intro():
    return make_divider("02", "Applications",
        "Differentiable physics tuning · AIMIP · NeuralGCM-style hybrids · "
        "SFNO S2S · SCM · RCE / RCEMIP · CRM · ocean experiments.")


def slide_amip():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "AMIP — RRTMGP + Sundqvist + SBM across grids",
                       "applications · amip", 20, TOTAL_SLIDES)
    lx, lw = M, 5.5
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("What's in the production AMIP CMIP6 deck",
                   size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.15, lw, 4.5,
        bullets([
            "Cubed-sphere C48 + lat-lon C144 production grids; Gaussian T42 + MPAS ico6 smoke-tested",
            "Radiation: RRTMGP (LW g128, SW g112) with bundled jax-rrtmgp lookup tables",
            "Clouds: Sundqvist large-scale + cloud fraction",
            "Convection: SBM (Frierson Betts-Miller), AIMIP-tunable",
            "Turbulence: Louis surface layer + bulk Ri PBL",
            "Forcing: amip_forcing — SST, sea-ice, GHG, ozone, aerosol, solar",
            "CMOR-compliant output (CF-1.8): Amon · Lmon · Omon · Oyr · Ofx · SImon · SIyr",
        ], size=12.5, line_spacing=1.18, space_after_pt=4)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    fig = first_existing(
        RESULTS / "held_suarez_gray_fv_C16_L20_10d_rerun_20260307" / "amip_snapshots.png",
        DEBUG / "spectral_hs100_final.png",
        LOGO_PATH)
    b.add_image_rect(idg, rx, 1.70, rw, 4.4, fig,
        caption="AMIP-style diagnostic stack — Held-Suarez + gray radiation rollout.")
    return b


def slide_held_suarez():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Held-Suarez aquaplanet across cores",
                       "applications · validation", 21, TOTAL_SLIDES)
    fig = first_existing(
        RESULTS / "held_suarez_matrix" / "03_spectral" / "field_snapshots.png",
        DEBUG / "spectral_hs100_final.png",
        DEBUG / "spectral_hs100.png")
    b.add_image_rect(idg, M, 1.65, SLIDE_W_IN - 2 * M, 3.6, fig,
        caption="Held-Suarez 1994 forcing — spectral T21 final state showing the developed eddy-driven jet.")
    b.add(sp_textbox(idg, M, 5.55, SLIDE_W_IN - 2 * M, 1.5,
        bullets([
            "Same forcing, three dynamical cores: FV A-grid · FV C-D grid · spectral — all reach the canonical zonal-mean climatology.",
            "Energy budget closure verified; jet latitude, eddy momentum flux, and EKE within published Held-Suarez bands.",
            "Held-Suarez is the workhorse smoke test for any new dycore change before AMIP runs.",
        ], size=12, line_spacing=1.18, space_after_pt=3)))
    return b


def slide_scm():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Single-column model — fast offline iteration",
                       "applications · scm", 22, TOTAL_SLIDES)
    lx, lw = M, 6.5
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("atmosphere/forcing/scm/scm.py · SingleColumnModel",
                   size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.15, lw, 4.55,
        bullets([
            "Dycore-free driver on a single (lat, lon) column — no horizontal advection / PGF / Coriolis.",
            "Reuses the same HydrostaticState pytree as the 3-D model, shape (face=1, x=1, y=1, level=nlev) — not a separate SCM-only state.",
            "Calls the canonical hydrostatic physics factory make_physics — every scheme that runs in 3-D runs here unchanged.",
            "Sigma vertical grid via create_sigma_coordinate; tracers (q_c, q_r, q_i, q_s) auto-materialised by physics.",
            "Three time integrators: Forward Euler · Heun RK2 · classical RK4 (TIME_INTEGRATORS registry, register_time_integrator()).",
            "Compatibility validator rejects multi-stage RK with stateful schemes (TKE / EDMF, mass-flux / Tiedtke / Bechtold, prognostic-spectral GWD, diurnal radiation) — never silently lies about order.",
            "Because state + physics factory are shared, the SCM is jax.grad-compatible end-to-end — fastest path for offline gradient-based parameter tuning with TrainablePhysicsParams.",
        ], size=12, line_spacing=1.18, space_after_pt=4)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.70, rw, 2.45, TEAL_DARK))
    b.add(sp_textbox(idg, rx + 0.30, 1.85, rw - 0.6, 0.35,
        [Para([Run("API · 50-day tropical RCE smoke test",
                   size=12, italic=True, color=PALE_INK, font=HEADER_FONT)])]))
    code_lines = [
        "scm = SingleColumnModel.create(",
        "    physics_config=...,",
        "    nlev=30, dt=300.0,",
        "    T_profile=..., q_v_profile=...,",
        "    latitude_deg=0.0,",
        "    time_integrator=\"rk4\")",
        "",
        "state, hist = scm.run(",
        "    nsteps=14400, save_every=12)",
    ]
    paras = [Para([Run(l, size=12, color=LIGHT_GREY, font=MONO_FONT)],
                  line_spacing=1.22, space_after_pt=0) for l in code_lines]
    b.add(sp_textbox(idg, rx + 0.30, 2.20, rw - 0.6, 1.85, paras))
    b.add(sp_rect(idg, rx, 4.25, rw, 2.40, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 4.35, rw - 0.5, 0.35,
        [Para([Run("Use cases", size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, rx + 0.25, 4.75, rw - 0.5, 1.85,
        bullets([
            "Parameterization integration tests",
            "Timestep & scheme stability sweeps",
            "Idealized process cases (RCE, GABLS, BOMEX)",
            "Fast iteration before 3-D AMIP runs",
            "ML-physics training-data generation",
        ], size=11.5, line_spacing=1.15, space_after_pt=2)))
    return b


def slide_rce():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "RCE — radiative-convective equilibrium",
                       "applications · rce", 23, TOTAL_SLIDES)
    lx, lw = M, 6.2
    b.add(sp_textbox(idg, lx, 1.70, lw, 4.8,
        bullets([
            "Three RCE flavors share one driver: prescribed-SST · slab-ocean · slab-land",
            "Spectral T21 + Gaussian grid · slab cubed-sphere C36 · multi-month rollouts",
            "Used to validate radiation–convection–boundary-layer closure under a clean global-mean budget",
            "Trainable physics modes plug straight into RCE: ideal sandbox for gradient-based parameter calibration",
            "Single-column model (atmosphere/forcing/scm/scm.py): same physics factory as 3D runs — for fast offline diagnostics and ML-physics training data",
        ], size=13, line_spacing=1.22, space_after_pt=4)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    fig = first_existing(
        RESULTS / "rce_slab_ocean" / "rce_snapshots.png",
        DEBUG / "spectral_hs100_eqinit.png")
    b.add_image_rect(idg, rx, 1.70, rw, 4.45, fig,
        caption="RCE-style equilibrium snapshot (slab-ocean / equilibrium-init reference).")
    return b


def slide_rcemip():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "RCEMIP — long-stability harness + convection sweep",
                       "applications · rcemip", 24, TOTAL_SLIDES)
    cards = [
        ("Hydrostatic RCE",  "run_rce.py",
         "Aquaplanet / land-planet · slab ocean (50 m / 50+200 m two-layer) or slab land · 4 grids: cubed-sphere, Gaussian, lat-lon, MPAS"),
        ("RCEMIP-I plane CRM", "run_rcemip_plane.py",
         "PlaneCompressibleEulerModel · Wing 2018 IC · radiation {gray | rrtmgp | none} · microphysics zoo · Smagorinsky LES + sponge + hyperdiff"),
        ("Stretched CRM smoke", "run_rce_smoke_stretched.py",
         "Stretched HeightCoordinate (~50 m surface) · Wing IC · tracer positivity · moist-mass fixer · CFL & CWV / MSE diagnostics"),
    ]
    cw = (SLIDE_W_IN - 2 * M - 0.4) / 3
    ch = 1.55; cy = 1.70
    for i, (name, script, body) in enumerate(cards):
        x = M + i * (cw + 0.20)
        b.add(sp_rect(idg, x, cy, cw, ch, PAPER, line_color=BORDER))
        b.add(sp_rect(idg, x, cy, cw, 0.08, TEAL))
        b.add(sp_textbox(idg, x + 0.20, cy + 0.15, cw - 0.40, 0.40,
            [Para([Run(name, size=14, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
        b.add(sp_textbox(idg, x + 0.20, cy + 0.50, cw - 0.40, 0.30,
            [Para([Run(script, size=10, color=RUST, font=MONO_FONT)])]))
        b.add(sp_textbox(idg, x + 0.20, cy + 0.80, cw - 0.40, ch - 0.85,
            [Para([Run(body, size=10.5, color=INK)], line_spacing=1.18)]))
    by = cy + ch + 0.25
    bh = SLIDE_H_IN - by - 0.55
    pw = (SLIDE_W_IN - 2 * M - 0.30) / 2
    px1, px2 = M, M + pw + 0.30
    b.add(sp_rect(idg, px1, by, pw, bh, TINT, line_color=BORDER))
    b.add(sp_rect(idg, px1, by, 0.10, bh, RUST))
    b.add(sp_textbox(idg, px1 + 0.25, by + 0.12, pw - 0.40, 0.35,
        [Para([Run("RCEMIP100 — 100 sim-day long-stability harness",
                   size=13, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, px1 + 0.25, by + 0.45, pw - 0.40, 0.30,
        [Para([Run("run_rcemip_long.py · aggregate_rcemip100.py",
                   size=10, color=RUST, font=MONO_FONT)])]))
    b.add(sp_textbox(idg, px1 + 0.25, by + 0.80, pw - 0.40, bh - 0.95,
        bullets([
            "4 non-hydrostatic dycores: plane_fd · plane_spectral · cubed_sphere · mpas",
            "Pure-dycore mass-conservation + finiteness baseline (physics deferred)",
            "Pass criteria: max|w| < 50 m/s · mass drift < 1e-6 · q_v ∈ [0, 0.030]",
            "Cross-grid aggregator emits a status · t_d · max|w| · max|θ′| · drift · wall-time table",
            "Current end-state JSONs in results/rcemip100/",
        ], size=11, line_spacing=1.18, space_after_pt=3)))
    b.add(sp_rect(idg, px2, by, pw, bh, TINT, line_color=BORDER))
    b.add(sp_rect(idg, px2, by, 0.10, bh, TEAL))
    b.add(sp_textbox(idg, px2 + 0.25, by + 0.12, pw - 0.40, 0.35,
        [Para([Run("Convection scheme sweep",
                   size=13, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, px2 + 0.25, by + 0.45, pw - 0.40, 0.30,
        [Para([Run("run_rce_convection_sweep.py",
                   size=10, color=RUST, font=MONO_FONT)])]))
    b.add(sp_textbox(idg, px2 + 0.25, by + 0.80, pw - 0.40, bh - 0.95,
        bullets([
            "Fixed-SST lat-lon FV with gray radiation + bulk BL + minimal microphysics scaffold",
            "8 schemes side-by-side: sbm · tiedtke · zhang_mcfarlane · emanuel · bechtold · kuo · kain_fritsch · mass_flux",
            "Tightened M_b_max=1e-5 (stripped-down setup lacks production shock-absorbers)",
            "Status precedence: ERROR → BLOWUP → RUNAWAY → OK",
            "Per-scheme snapshots: snapshot_{scheme}.npz + snapshot3d_{scheme}.npz",
        ], size=11, line_spacing=1.18, space_after_pt=3)))
    return b


def slide_crm():
    """RCE runs with a cloud-resolving model — non-hydrostatic CRM physics."""
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Cloud-resolving RCE — non-hydrostatic CRM",
                       "applications · crm", 25, TOTAL_SLIDES)

    # Left column: bullets
    lx, lw = M, 6.5
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Plane non-hydrostatic dycore with the full physics stack",
                   size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.15, lw, 4.55,
        bullets([
            "PlaneCompressibleEulerModel: non-hydrostatic compressible Euler on a plane (run_rcemip_plane.py).",
            "RCEMIP-I analytical IC (Wing 2018 Table A1–A2): two-segment θ(z), q_v(z) = q_sfc·exp(−z / 4 km).",
            "Stretched HeightCoordinate option (run_rce_smoke_stretched.py): ~50 m surface layer, deep CRM aloft.",
            "LES closure: Smagorinsky subgrid stresses + scalar fluxes.",
            "Stability protection: sponge layer aloft + hyperdiffusion + CFL diagnostic.",
            "Surface fluxes via bulk formulas (COARE / fixed-z₀); composer make_rcemip_physics sums radiation + microphysics tendencies (model_type=\"plane\").",
            "AD-compatible end to end: tracer positivity filter, moist-mass fixer, mean-wind removal all written as differentiable projections.",
        ], size=12, line_spacing=1.18, space_after_pt=4)))

    # Right column: vertical schematic + configurable physics matrix
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M

    # Top half: vertical column schematic
    b.add(sp_rect(idg, rx, 1.70, rw, 3.30, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 1.85, rw - 0.5, 0.35,
        [Para([Run("Vertical structure", size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    layers = [
        ("Sponge layer",        SLATE,   "Rayleigh damping toward IC"),
        ("Free troposphere",    "5F9EA0", "non-hydrostatic dynamics + microphysics + radiation"),
        ("Boundary layer",      "B85042", "LES (Smagorinsky) + bulk surface fluxes"),
        ("Stretched surface",   "8B5F3A", "~50 m layer (HeightCoordinate)"),
    ]
    ly = 2.30
    lh = 0.50
    for name, fill, note in layers:
        b.add(sp_rect_with_text(idg, rx + 0.30, ly, 2.4, lh, fill,
            [Para([Run(name, size=11, bold=True, color=PAPER)], align="ctr")],
            anchor="ctr", margin=0))
        b.add(sp_textbox(idg, rx + 2.80, ly, rw - 3.10, lh,
            [Para([Run(note, size=10.5, color=INK)])], anchor="ctr"))
        ly += lh + 0.10

    # Bottom half: configurable physics row
    b.add(sp_rect(idg, rx, 5.15, rw, 1.50, PAPER, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 5.22, rw - 0.5, 0.35,
        [Para([Run("Configurable physics", size=13, bold=True, color=RUST, font=HEADER_FONT)])]))
    cfg_y = 5.65
    cfg_h = 0.45
    cfg_items = [
        ("Radiation",   "gray · rrtmgp · none"),
        ("Microphysics","kessler · sundqvist · seifert · morrison · thompson · ml_emulator"),
    ]
    for label, opts in cfg_items:
        b.add(sp_textbox(idg, rx + 0.30, cfg_y, 1.2, cfg_h,
            [Para([Run(label, size=11, bold=True, color=TEAL_DARK, font=HEADER_FONT)])],
            anchor="ctr"))
        b.add(sp_textbox(idg, rx + 1.55, cfg_y, rw - 1.75, cfg_h,
            [Para([Run(opts, size=10, color=INK, font=MONO_FONT)])], anchor="ctr"))
        cfg_y += 0.45

    return b


def slide_aimip():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "AIMIP & differentiable physics tuning",
                       "applications · ai-physics", 26, TOTAL_SLIDES)
    lx, lw = M, 6.5
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Mode 1 · trainable classical physics",
                   size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.15, lw, 4.5,
        bullets([
            "TrainablePhysicsParams (Equinox module): 8 physics params wrapped in sigmoid range constraints",
            "Held–Suarez: τ_equator ∈ [5,10] d · τ_pole ∈ [1,3] d",
            "SBM convection: τ_c ∈ [1, 4] h · RH_ref ∈ [0.6, 0.9]",
            "Bulk fluxes: C_H, C_E ∈ [10⁻³, 5·10⁻³]",
            "Albedos: α_ice ∈ [0.4, 0.8] · α_ocean ∈ [0.03, 0.10]",
            "Scheme-specific subsets via trainable_constraints_for_scheme()",
            "AIMIPClassicalParams adds Tiedtke + Louis + McFarlane + Xu–Randall + gray-radiation tunables",
        ], size=12.5, line_spacing=1.20, space_after_pt=4)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.70, rw, 4.95, TEAL_DARK))
    b.add(sp_textbox(idg, rx + 0.30, 1.90, rw - 0.6, 0.4,
        [Para([Run("training loop · gradient flows",
                   size=13, italic=True, color=PALE_INK, font=HEADER_FONT)])]))
    code_lines = [
        "params = TrainablePhysicsParams.init(...)",
        "",
        "def loss(p, state0, target):",
        "    final = build_segment_fn(p).raw(",
        "        state0, forcing, n_steps=N)",
        "    return area_weighted_mse(",
        "        final.q_v, target.q_v)",
        "",
        "loss_val, grads = (",
        "    eqx.filter_value_and_grad(loss)",
        "    (params, state0, target))",
        "",
        "params = optax_update(params, grads)",
    ]
    paras = [Para([Run(l, size=13, color=LIGHT_GREY, font=MONO_FONT)],
                  line_spacing=1.25, space_after_pt=0) for l in code_lines]
    b.add(sp_textbox(idg, rx + 0.30, 2.30, rw - 0.6, 4.4, paras))
    return b


def slide_neural_gcm():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "NeuralGCM + SFNO hybrids",
                       "applications · ai-physics", 27, TOTAL_SLIDES)
    cw = (SLIDE_W_IN - 2 * M - 0.30) / 2
    ch = 5.0
    x1, x2 = M, M + cw + 0.30
    y = 1.70
    b.add(sp_rect(idg, x1, y, cw, ch, PAPER, line_color=BORDER))
    b.add(sp_rect(idg, x1, y, 0.10, ch, RUST))
    b.add(sp_textbox(idg, x1 + 0.25, y + 0.15, cw - 0.40, 0.45,
        [Para([Run("Mode 2 · NeuralGCM-style spectral PE",
                   size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, x1 + 0.25, y + 0.70, cw - 0.40, ch - 0.80,
        bullets([
            "Spectral PE dycore + neural physics under one segment kernel",
            "training/neural_gcm_spectral.py · train_neural_gcm()",
            "Equinox NN replaces (or augments) classical physics; conservation correctors keep mass / moisture exact",
            "Same loss / optimizer infrastructure as Mode 1 — only the parameter tree changes",
            "Gradient checkpointing through lax.scan for long rollouts",
        ], size=11.5, line_spacing=1.18, space_after_pt=3)))
    b.add(sp_rect(idg, x2, y, cw, ch, PAPER, line_color=BORDER))
    b.add(sp_rect(idg, x2, y, 0.10, ch, TEAL))
    b.add(sp_textbox(idg, x2 + 0.25, y + 0.15, cw - 0.40, 0.45,
        [Para([Run("Mode 3 · SFNO coupled to dycore",
                   size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, x2 + 0.25, y + 0.70, cw - 0.40, ch - 0.80,
        bullets([
            "SFNO (Bonev 2023 + Watt-Meyer / Clark / Henn ACE 2023) in ml/sfno.py",
            "Encoder → N SFNOBlocks (spectral conv + MLP) → decoder + big-skip residual",
            "Defaults: embed_dim=256 · n_blocks=8 · mlp_expansion=4 · per-block gradient_checkpoint",
            "Channel packing (ml/channel_packing.py): state ↔ tensor for SW / PE-3D / Ocean · WB2 pressure levels",
            "Conservation correctors post-network: dry mass / moisture / ocean volume / heat / salt",
            "Drop-in replacement for any dycore in compiled_segments — same SegmentCarry",
        ], size=11.5, line_spacing=1.18, space_after_pt=3)))
    return b


def slide_sfno_s2s():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "SFNO + slab ocean — S2S coupled rollouts",
                       "applications · s2s", 28, TOTAL_SLIDES)
    lx, lw = M, 6.6
    b.add(sp_textbox(idg, lx, 1.75, lw, 4.8,
        bullets([
            "ml/s2s/sfno_slab/: SFNO atmosphere + slab ocean — S2S campaign infrastructure (training, rollout, coupling, evaluation, ChaosBench-compatible).",
            "ml/s2s/neuralgcm_slab/: NeuralGCM atmosphere + slab ocean — neuralgcm_backend, slab_coupling, campaign, ensemble.",
            "Same SurfaceState / TileFractions interface as the full coupled model — the slab is just a simpler ocean component plugged into the existing coupler.",
            "Trained on ERA5 (era5_to_state.py · era5_loader.py) with area-weighted MSE + spectral loss; optimizer in ml/training.create_optimizer().",
            "Joint ML-physics workflow (ml/physics/): PhysicsTeacherDataset + capture_physics_teacher_snapshot from AMIP runs trains a parameterization replacement that respects conservation.",
        ], size=12.5, line_spacing=1.20, space_after_pt=5)))
    rx = lx + lw + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_rect(idg, rx, 1.75, rw, 4.85, TINT, line_color=BORDER))
    b.add(sp_textbox(idg, rx + 0.25, 1.90, rw - 0.5, 0.4,
        [Para([Run("S2S coupled step", size=14, bold=True, color=RUST, font=HEADER_FONT)])]))
    chain = [
        ("ERA5 state",            "(packed channels)"),
        ("SFNO atm step",         "(spectral conv)"),
        ("conservation correctors","(mass / moisture)"),
        ("slab ocean step",       "(SST tendency)"),
        ("tile blend",            "(coupler)"),
        ("next state",            ""),
    ]
    y = 2.50
    for i, (label, sub) in enumerate(chain):
        b.add(sp_rect(idg, rx + 0.30, y, rw - 0.6, 0.50, PAPER, line_color=BORDER))
        b.add(sp_textbox(idg, rx + 0.45, y, (rw - 0.9) * 0.55, 0.50,
            [Para([Run(label, size=12, bold=True, color=TEAL_DARK, font=HEADER_FONT)])], anchor="ctr"))
        if sub:
            b.add(sp_textbox(idg, rx + 0.45 + (rw - 0.9) * 0.55, y, (rw - 0.9) * 0.45, 0.50,
                [Para([Run(sub, size=10, color=MUTED)], align="r")], anchor="ctr"))
        if i < len(chain) - 1:
            b.add(sp_textbox(idg, rx + (rw - 0.4) / 2, y + 0.48, 0.4, 0.18,
                [Para([Run("↓", size=14, bold=True, color=RUST)], align="ctr")]))
        y += 0.66
    return b


def slide_ocean_experiments():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Ocean experiments — Drake / DINO / OMIP-2",
                       "applications · ocean", 29, TOTAL_SLIDES)
    fig = DOCS / "images" / "50yr_implicit" / "drake_zonal_u_evolution.png"
    b.add_image_rect(idg, M, 1.65, 6.6, 4.0, fig,
        caption="Drake passage zonal-u evolution over 50 yr (implicit barotropic, cubed-sphere ocean).")
    rx = M + 6.6 + 0.30
    rw = SLIDE_W_IN - rx - M
    b.add(sp_textbox(idg, rx, 1.70, rw, 0.4,
        [Para([Run("Ocean experiment suite", size=15, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, rx, 2.15, rw, 4.6,
        bullets([
            "Global overturning across all 4 grids (200-yr / 50-yr-implicit / GM-Redi / MPAS-baseline)",
            "DINO double-gyre (Hochet 2025)",
            "ACC channel · Phillips two-layer · Eady · Held–Larichev",
            "NeverWorld2-lite · ISOMIP+ (ice-shelf cavity)",
            "OMIP-2 JRA55-do multi-decadal across lat-lon / tripolar eORCA1 / cubed-sphere / MPAS",
            "Bryan THC distorted-physics centennial spin-up",
            "Veros fidelity harness — adapters for DINO, ACC, lock exchange, overflow, Eady uniform",
        ], size=12, line_spacing=1.18, space_after_pt=3)))
    return b


def slide_validation_intro():
    return make_divider("03", "Validation strategy",
        "Dycore progression suite · Williamson · DCMIP · Held-Suarez · ocean "
        "test matrix (57/57) · Veros fidelity harness · visual + budget verification.")


def slide_validation_overview():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Validation harness — what we run, when",
                       "validation", 31, TOTAL_SLIDES)
    rows = [
        ["Atmosphere test matrix",       "scripts/matrix/run_atmosphere_test_matrix.py",      "sw · hydro · nh · climate · tracer · dcmip2008 · dcmip2012 · dcmip2016 · hughes"],
        ["Dycore progression suite",     "tests/validation/run_dycore_progression_suite.py",
                                          "SW spectral → SW FV → hydro FV → spectral HS → NH FV → spectral NH"],
        ["Ocean test matrix",            "scripts/matrix/run_ocean_test_matrix.py",            "57 / 57 PASS across lat-lon, tripolar, cubed-sphere, MPAS Voronoi"],
        ["Sea ice test matrix",          "scripts/matrix/run_sea_ice_test_matrix.py",          "15 benchmarks: thermo · dynamics · transport · ITD · integration"],
        ["MPI differentiability tests",  "tests/distributed/test_mpi_differentiability.py",
                                          "AD safety of _sendrecv_vjp + allreduce(SUM), checked via 2-rank runs"],
        ["Ocean fidelity vs Veros",      "ocean/fidelity/",                              "DINO · ACC · lock exchange · overflow · Eady uniform — cross-model report"],
    ]
    cx = M
    headers = ["Suite", "Entry point", "Coverage"]
    cw = [3.0, 4.4, 4.85]
    y0 = 1.80; rh = 0.66
    b.add(sp_rect(idg, cx, y0, sum(cw), 0.42, TEAL))
    x = cx
    for i, h in enumerate(headers):
        b.add(sp_textbox(idg, x + 0.12, y0, cw[i] - 0.24, 0.42,
            [Para([Run(h, size=12, bold=True, color=PAPER)])], anchor="ctr", margin=0))
        x += cw[i]
    for ri, row in enumerate(rows):
        y = y0 + 0.42 + ri * rh
        bg = PAPER if ri % 2 == 0 else TINT
        b.add(sp_rect(idg, cx, y, sum(cw), rh, bg))
        x = cx
        for i, cell in enumerate(row):
            font = MONO_FONT if i == 1 else BODY_FONT
            size = 11 if i == 1 else 12
            run_color = TEAL_DARK if i == 0 else (RUST if i == 1 else INK)
            bold = i == 0
            b.add(sp_textbox(idg, x + 0.12, y, cw[i] - 0.24, rh,
                [Para([Run(cell, size=size, bold=bold, color=run_color, font=font)])],
                anchor="ctr", margin=0))
            x += cw[i]
    b.add(sp_textbox(idg, M, y0 + 0.42 + len(rows) * rh + 0.20, SLIDE_W_IN - 2 * M, 0.65,
        [Para([Run("Each scientific change has a narrowest-relevant matrix; "
                   "nontrivial numerical changes always run the dycore "
                   "progression suite end-to-end before merge.",
                   size=12, italic=True, color=MUTED)], align="ctr", line_spacing=1.25)]))
    return b


def slide_williamson_grid():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Williamson SW benchmarks — same case, three cores",
                       "validation · dycore", 32, TOTAL_SLIDES)
    fw = (SLIDE_W_IN - 2 * M - 0.4) / 3
    fh = 3.5; fy = 1.80
    figs = [
        (first_existing(
            RESULTS / "williamson_matrix_ssp45" / "01_latlon_spectral_williamson2" / "field_snapshots.png",
            DIAG / "w2_herr_duogrid_latlon.png",
            DIAG / "w2_h_err_dg.png"),
         "Duogrid · lat-lon"),
        (first_existing(
            RESULTS / "williamson_matrix_ssp45" / "02_latlon_fv_williamson2" / "field_snapshots.png",
            DIAG / "w2_herr_production_latlon.png"),
         "FV · lat-lon"),
        (first_existing(
            RESULTS / "williamson_matrix_ssp45" / "03_cubesphere_fv_williamson2" / "field_snapshots.png",
            DIAG / "w2_herr_production.png"),
         "FV · cubed-sphere"),
    ]
    for i, (path, lab) in enumerate(figs):
        x = M + i * (fw + 0.20)
        b.add(sp_rect(idg, x, fy, fw, fh + 0.45, PAPER, line_color=BORDER))
        b.add(sp_textbox(idg, x, fy + 0.05, fw, 0.35,
            [Para([Run(lab, size=13, bold=True, color=TEAL_DARK, font=HEADER_FONT)], align="ctr")]))
        b.add_image_rect(idg, x + 0.10, fy + 0.45, fw - 0.20, fh - 0.20, path)
    b.add(sp_textbox(idg, M, fy + fh + 0.65, SLIDE_W_IN - 2 * M, 1.30,
        bullets([
            "Williamson 2 (steady-state geostrophic flow) — must reproduce the analytical solution to round-off; same case on three cores reveals grid imprinting and edge effects.",
            "Williamson 5 (zonal flow over an isolated mountain) — sensitive to PPM limiter and divergence damping; visual inspection of v-wind is the canonical edge-artifact check.",
            "Both cases also run on cubed-sphere native projection (field_snapshots_cube_native.png) to verify halo / corner handling.",
        ], size=12, line_spacing=1.20, space_after_pt=3)))
    return b


def slide_roadmap():
    b = SlideBuild(); idg = IdGen()
    add_content_chrome(b, idg, "Where we are, where we're going",
                       "roadmap", 33, TOTAL_SLIDES)

    # Small logo accent — bottom-right above the call-to-action bar.
    b.add_image_rect(idg, SLIDE_W_IN - M - 1.4, 5.55, 1.4, 1.4, LOGO_PATH)

    lx, lw = M, 5.6
    b.add(sp_textbox(idg, lx, 1.70, lw, 0.4,
        [Para([Run("Shipped", size=18, bold=True, color=TEAL_DARK, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, lx, 2.15, lw, 4.5,
        bullets([
            "Coupled atm + ocean + ice + land + lake on 4 grid families",
            "End-to-end jax.grad through MPI distributed runs",
            "AMIP-CMIP6 deck (RRTMGP + Sundqvist + SBM + Louis)",
            "57/57 ocean test matrix · 15/15 sea-ice tests · dycore progression suite",
            "AIMIP-style trainable classical physics",
            "SFNO + NeuralGCM-style hybrid cores · S2S slab campaigns",
            "Veros ocean fidelity harness · CMOR-compliant output",
        ], size=12.5, line_spacing=1.20, space_after_pt=3)))

    rx = lx + lw + 0.30
    rw = 5.6
    b.add(sp_textbox(idg, rx, 1.70, rw, 0.4,
        [Para([Run("Next 12 months", size=18, bold=True, color=RUST, font=HEADER_FONT)])]))
    b.add(sp_textbox(idg, rx, 2.15, rw, 4.5,
        bullets([
            "Multi-year coupled AMIP+OMIP with prognostic sea ice",
            "Full incremental 4D-Var with ensemble B (gen-be)",
            "GPU strong-scaling beyond Levante baseline",
            "End-to-end gradient-based tuning against ERA5 + ARGO",
            "MPAS-Voronoi at production resolution (ico7)",
            "Public release: docs, tutorials, reference notebooks",
            "Community-contributed physics schemes via the lego interface",
        ], size=12.5, line_spacing=1.20, space_after_pt=3)))

    by = 6.55
    b.add(sp_rect_with_text(idg, M, by, SLIDE_W_IN - 2 * M, 0.55, TEAL_DARK,
        [Para([Run("Thank you  ·  Questions?   github.com/climate-federation/legoESM   ·   pg2328@columbia.edu",
                   size=16, bold=True, color=PAPER, font=HEADER_FONT)], align="ctr")],
        anchor="ctr", margin=0))
    return b


# ===========================================================================
# OOXML scaffolding
# ===========================================================================

CONTENT_TYPES_TPL = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
__SLIDE_OVERRIDES__
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

CORE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>legoESM Seminar</dc:title>
  <dc:creator>Pierre Gentine</dc:creator>
  <cp:lastModifiedBy>legoESM build script</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-05-25T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-05-25T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"""

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>legoESM build script</Application>
  <PresentationFormat>Widescreen</PresentationFormat>
  <Slides>__NSLIDES__</Slides>
</Properties>
"""

THEME_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="legoESM">
  <a:themeElements>
    <a:clrScheme name="legoESM">
      <a:dk1><a:srgbClr val="1F2937"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="0F4C5C"/></a:dk2>
      <a:lt2><a:srgbClr val="F1F5F9"/></a:lt2>
      <a:accent1><a:srgbClr val="0F4C5C"/></a:accent1>
      <a:accent2><a:srgbClr val="415A77"/></a:accent2>
      <a:accent3><a:srgbClr val="E36414"/></a:accent3>
      <a:accent4><a:srgbClr val="94A3B8"/></a:accent4>
      <a:accent5><a:srgbClr val="475569"/></a:accent5>
      <a:accent6><a:srgbClr val="CBD5E1"/></a:accent6>
      <a:hlink><a:srgbClr val="E36414"/></a:hlink>
      <a:folHlink><a:srgbClr val="415A77"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="legoESM">
      <a:majorFont><a:latin typeface="Georgia"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
      <a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Office">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
        <a:ln w="25400" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
        <a:ln w="38100" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>
"""

SLIDE_MASTER_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>
                <a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles>
    <p:titleStyle><a:lvl1pPr algn="l"><a:defRPr sz="3200" b="1"><a:solidFill><a:srgbClr val="0F4C5C"/></a:solidFill><a:latin typeface="Georgia"/></a:defRPr></a:lvl1pPr></p:titleStyle>
    <p:bodyStyle><a:lvl1pPr><a:defRPr sz="1400"><a:solidFill><a:srgbClr val="1F2937"/></a:solidFill><a:latin typeface="Calibri"/></a:defRPr></a:lvl1pPr></p:bodyStyle>
    <p:otherStyle><a:lvl1pPr><a:defRPr sz="1400"><a:solidFill><a:srgbClr val="1F2937"/></a:solidFill><a:latin typeface="Calibri"/></a:defRPr></a:lvl1pPr></p:otherStyle>
  </p:txStyles>
</p:sldMaster>
"""

SLIDE_MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""

SLIDE_LAYOUT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>
                <a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""

SLIDE_LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def presentation_xml(n_slides):
    sld_id_list = []
    for i in range(n_slides):
        sld_id_list.append(f'<p:sldId id="{256 + i}" r:id="rId{i + 2}"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                saveSubsetFonts="1">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{''.join(sld_id_list)}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="screen16x9"/>
  <p:notesSz cx="{SLIDE_H}" cy="{SLIDE_W}"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>
</p:presentation>
"""


def presentation_rels(n_slides):
    parts = ['<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" '
        'Target="slideMasters/slideMaster1.xml"/>']
    for i in range(n_slides):
        parts.append(
            f'<Relationship Id="rId{i + 2}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="slides/slide{i + 1}.xml"/>')
    parts.append(
        f'<Relationship Id="rId{n_slides + 2}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" '
        f'Target="theme/theme1.xml"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        + "\n".join(parts) + "\n</Relationships>\n")


def slide_xml(b):
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"\n'
        '       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"\n'
        '       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">\n'
        '  <p:cSld>\n'
        '    <p:spTree>\n'
        '      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>\n'
        '      <p:grpSpPr>\n'
        '        <a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>\n'
        '                <a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>\n'
        '      </p:grpSpPr>\n'
        + "".join(b.shapes_xml) +
        '    </p:spTree>\n'
        '  </p:cSld>\n'
        '  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>\n'
        '</p:sld>\n')


def slide_rels(b, slide_idx, media_for):
    rels = ['<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" '
            'Target="../slideLayouts/slideLayout1.xml"/>']
    for rId, path in zip(b.image_rids, b.image_paths):
        media_name = media_for[str(path)]
        rels.append(
            f'<Relationship Id="{rId}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="../media/{media_name}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        + "\n".join(rels) + "\n</Relationships>\n")


SLIDE_FUNCS = [
    slide_title, slide_outline, slide_motivation, slide_design_principles,
    slide_landscape, slide_architecture, slide_jax_foundations, slide_ad_story,
    slide_conservation, slide_modularity_intro, slide_atmosphere_dycores,
    slide_atmosphere_physics, slide_ocean_dycores, slide_ocean_physics,
    slide_land, slide_seaice, slide_coupler_grids, slide_parallel,
    slide_applications_intro, slide_amip, slide_held_suarez, slide_scm,
    slide_rce, slide_rcemip, slide_crm, slide_aimip, slide_neural_gcm,
    slide_sfno_s2s, slide_ocean_experiments, slide_validation_intro,
    slide_validation_overview, slide_williamson_grid, slide_roadmap,
]


def build(out):
    n = len(SLIDE_FUNCS)
    assert n == TOTAL_SLIDES, f"expected {TOTAL_SLIDES} slides, got {n}"
    slides = [fn() for fn in SLIDE_FUNCS]

    # Dedupe media: each unique source path → one file in ppt/media/
    media_for = {}
    media_payload = {}  # filename → bytes
    media_idx = 0
    for b in slides:
        for path in b.image_paths:
            key = str(path)
            if key in media_for:
                continue
            media_idx += 1
            suffix = path.suffix or ".png"
            media_for[key] = f"image_{media_idx:03d}{suffix}"

    slide_overrides = []
    for i in range(n):
        slide_overrides.append(
            f'  <Override PartName="/ppt/slides/slide{i+1}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    content_types = CONTENT_TYPES_TPL.replace(
        "__SLIDE_OVERRIDES__", "\n".join(slide_overrides))
    app_xml = APP_XML.replace("__NSLIDES__", str(n))

    with ZipFile(out, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("docProps/core.xml", CORE_XML)
        z.writestr("docProps/app.xml", app_xml)
        z.writestr("ppt/presentation.xml", presentation_xml(n))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(n))
        z.writestr("ppt/theme/theme1.xml", THEME_XML)
        z.writestr("ppt/slideMasters/slideMaster1.xml", SLIDE_MASTER_XML)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", SLIDE_MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", SLIDE_LAYOUT_XML)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", SLIDE_LAYOUT_RELS)
        for i, b in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(b))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels(b, i, media_for))
        # Write each unique media file once
        for key, media_name in media_for.items():
            with open(key, "rb") as f:
                z.writestr(f"ppt/media/{media_name}", f.read())
    return out


if __name__ == "__main__":
    out = REPO / "docs" / "assets" / "legoESM_seminar.pptx"
    p = build(out)
    size_kb = p.stat().st_size / 1024
    print(f"Wrote {p}")
    print(f"  size: {size_kb:.1f} KB")
    print(f"  slides: {TOTAL_SLIDES}")
