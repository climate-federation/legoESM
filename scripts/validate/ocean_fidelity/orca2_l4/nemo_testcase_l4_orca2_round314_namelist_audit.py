#!/usr/bin/env python3
"""Round 314: every namelist key ORCA2_OMIP_L4 resolves, against rung 10's card.

NEMO's value is ``namelist_ref`` overlaid by ``EXP00/namelist_cfg`` (and the
two SI3 files).  The parse is checked against a rung-10 record run's own
resolved dump (``output.namelist.dyn``/``.ice``: every key that run read)
and against its ``ocean.output`` prints.  The card value comes from the
built card (``build_orca2_zps_card``): its resolved config, its state seeds,
or -- for file names and two scheme constants -- the source literal the
builder executes.  Never from a comment.

Statuses: MATCH, DIFFERS, NOT-MODELLED (no card counterpart; ``why`` says
what the card does instead), INERT (NEMO's own resolved switch, named in
``why``, makes the key irrelevant; the switch value is asserted).

Fail-closed: an unclassified key, a DIFFERS row without a (a)/(b)/(c)
class, an INERT row whose switch does not hold, or a parse disagreement
with the run's dump that no rule explains all exit 1.

Searched first: ``audit_orca1_tke_card.py`` audits ORCA1 TKEConfig fields
against defaults, not namelists; the six line parsers in
``scripts/validate/ocean_fidelity/testcases`` read one line per key and
neither the dump's continuation lines nor derived-type ``sn_*`` lines.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, NamedTuple

REPO = Path(__file__).resolve().parents[4]
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
EXP00 = NEMO / "cfgs/ORCA2_OMIP_L4/EXP00"
CPP_KEYS = NEMO / "cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm"
RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
    "acquisition/orca1ice_surface_entry_every_step_a_np2")
RECIPE_SRC = REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py"
TKE_SRC = REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py"
IWM_SRC = REPO / "packages/ocean/legoesm/ocean/iwm_forcing.py"

STATUSES = ("MATCH", "DIFFERS", "NOT-MODELLED", "INERT")


class AuditError(RuntimeError):
    pass


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AuditError(msg)


# --------------------------------------------------------------------------
# Namelist parsing (source files and NEMO's resolved dump share one grammar)
# --------------------------------------------------------------------------
_KEY = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_%]*(?:\(\s*\d+\s*\))?)\s*=(.*)$")


def strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _split(text: str) -> list[str]:
    parts, cur, quote = [], [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch == ",":
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def norm(token: str):
    t = token.strip()
    if t[:1] in ("'", '"'):
        return t[1:-1].rstrip() if len(t) > 1 and t[-1] == t[0] else t[1:].rstrip()
    low = t.lower()
    if low in (".true.", "t", ".t.", "true"):
        return True
    if low in (".false.", "f", ".f.", "false"):
        return False
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        return low


def values(text: str) -> list:
    out = []
    for part in _split(text):
        m = re.fullmatch(r"(\d+)\*(.*)", part)
        if m and not part.startswith(("'", '"')):
            out.extend([norm(m.group(2))] * int(m.group(1)))
        else:
            out.append(norm(part))
    return out


def parse_namelist(path: Path, ns: str) -> dict[tuple[str, str, str], list]:
    """``(ns, group, key)`` -> normalised value tokens, in file order."""

    found: dict[tuple[str, str, str], list] = {}
    group, key, buf = None, None, ""

    def flush():
        if group and key is not None:
            vals = values(buf)
            name = (ns, group, key)
            if name in found and found[name] != vals:
                found[name] = vals            # NEMO keeps the last read
            else:
                found[name] = vals

    for raw in Path(path).read_text(errors="replace").splitlines():
        line = strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("&"):
            flush()
            group, key, buf = line[1:].split()[0].lower(), None, ""
            continue
        if line.startswith("/"):
            flush()
            group, key, buf = None, None, ""
            continue
        if group is None:
            continue
        m = _KEY.match(line)
        if m:
            flush()
            key, buf = re.sub(r"\s+", "", m.group(1)).lower(), m.group(2)
        elif key is not None:
            buf = buf.rstrip() + ("" if buf.rstrip().endswith(",") else ",") + line
    flush()
    return found


def overlay(*parsed: dict) -> dict:
    out: dict = {}
    for p in parsed:
        out.update(p)
    return out


def resolve_against_dump(dump: dict, src: dict) -> dict:
    """Map a source overlay onto the dump's keys: derived-type fields by
    position, ``x(i)`` element assignments into ``x``.  Returns
    ``(ns, group, key)`` -> tokens or ``None`` (key not in any source file)."""

    fields: dict[tuple[str, str, str], list[str]] = {}
    for ns, g, k in dump:
        if "%" in k:
            base, sub = k.split("%", 1)
            fields.setdefault((ns, g, base), [])
            if sub not in fields[(ns, g, base)]:
                fields[(ns, g, base)].append(sub)
    elems: dict[tuple[str, str, str], dict[int, object]] = {}
    for (ns, g, k), v in src.items():
        m = re.fullmatch(r"([a-z0-9_]+)\((\d+)\)", k)
        if m:
            elems.setdefault((ns, g, m.group(1)), {})[int(m.group(2))] = v[0]
    out: dict = {}
    for ns, g, k in dump:
        if "%" in k:
            base, sub = k.split("%", 1)
            vals = src.get((ns, g, base))
            pos = fields[(ns, g, base)].index(sub)
            out[(ns, g, k)] = None if vals is None or pos >= len(vals) else [vals[pos]]
        elif (ns, g, k) in src:
            out[(ns, g, k)] = src[(ns, g, k)]
        elif (ns, g, k) in elems:
            d = list(dump[(ns, g, k)])
            for i, v in elems[(ns, g, k)].items():
                d[i - 1] = v
            out[(ns, g, k)] = d
        else:
            out[(ns, g, k)] = None
    return out


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, float) and isinstance(b, float):
        return a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b))
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def same_list(src: list, dump: list) -> tuple[bool, str]:
    """Source tokens vs dump tokens.  Two explained forms: a source array
    shorter than the declared one (NEMO keeps the rest at its initial
    value), and a string the dump truncates to its declared length."""

    if len(src) > len(dump):
        return False, "source longer than dump"
    note = "partial array" if len(src) < len(dump) else ""
    for a, b in zip(src, dump):
        if same(a, b):
            continue
        if (isinstance(a, str) and isinstance(b, str) and b
                and a.lower().startswith(b.lower())):
            note = "dump truncates string to declared length"
            continue
        return False, f"{a!r} != {b!r}"
    return True, note


# --------------------------------------------------------------------------
# ocean.output cross-check
# --------------------------------------------------------------------------
def ocean_output_values(text: str, key: str) -> list:
    name = key.split("%")[0]
    if "(" in name:
        return []
    pat = re.compile(r"(?<![A-Za-z0-9_%])" + re.escape(name) + r"\s*=\s*(\S+)",
                     re.IGNORECASE)
    return [norm(m.group(1).rstrip(",):;")) for m in pat.finditer(text)]


def ocean_output_check(text: str, key: str, record: list) -> tuple[str, str]:
    if "%" in key or not record:
        return "n/a", ""
    seen = ocean_output_values(text, key)
    if not seen:
        return "absent", ""
    if any(same(s, record[0]) for s in seen):
        return "agree", ""
    return "disagree", ",".join(sorted({show(s) for s in seen}))


# --------------------------------------------------------------------------
# Card side
# --------------------------------------------------------------------------
def flatten_card(card) -> dict[str, object]:
    flat: dict[str, object] = {}

    def walk(prefix, nt):
        for f in nt._fields:
            v = getattr(nt, f)
            if hasattr(v, "_fields"):
                walk(prefix + f + ".", v)
            elif isinstance(v, (bool, str, int, float, tuple)) or v is None:
                flat[prefix + f] = v

    walk("", card.recipe.model_config)
    for f in card._fields:
        v = getattr(card, f)
        if isinstance(v, (bool, str, int, float, tuple)) or v is None:
            flat["card." + f] = v
    return flat


def _function_text(path: Path, name: str) -> str:
    text = path.read_text()
    start = text.index(f"\ndef {name}(")
    end = text.find("\ndef ", start + 1)
    return text[start:end if end > 0 else None]


class CardContext(NamedTuple):
    flat: dict
    card: object
    builder_files: tuple[str, ...]
    iwm_variables: tuple[str, ...]
    ts_variables: tuple[str, ...]
    tke_ebb: float
    builds_zstar: bool
    nn_e_resolved: int


def build_context(record: Path, ocean_output: str) -> CardContext:
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_orca2_zps_card(record)
    builder = _function_text(RECIPE_SRC, "build_orca2_zps_card")
    iwm = IWM_SRC.read_text()
    ts = _function_text(RECIPE_SRC, "build_orca2_initial_ts")
    m = re.search(r"^_NEMO_TKE_EBB\s*=\s*([0-9.eE+-]+)", TKE_SRC.read_text(), re.M)
    require(m is not None, "TKE scheme constant _NEMO_TKE_EBB not found")
    nn_e = re.search(r"in iterations nn_e\s*=\s*(\d+)", ocean_output)
    require(nn_e is not None, "ocean.output does not print the resolved nn_e")
    flat = flatten_card(card)
    st = card.recipe.initial_state
    wet = np.asarray(st.tke_avm.data) > 0
    flat["seed.avm_values"] = tuple(sorted({float(x) for x in np.asarray(st.tke_avm.data)[wet]}))
    avt = np.asarray(st.tke_avt.data)[wet]
    flat["seed.avt_max"] = float(avt.max())
    flat["seed.avt_min_over_max"] = float(avt.min() / avt.max())
    flat["mask.fmask_max"] = float(np.max(np.asarray(
        card.recipe.z_coord.nemo_een_barotropic.fmask)))
    return CardContext(
        flat=flat, card=card,
        builder_files=tuple(re.findall(r'root / "([^"]+)"', builder)),
        iwm_variables=tuple(re.findall(r'"((?:power|scale)_[a-z]+)"',
                                     re.search(r"^_POWER_VARS.*\n_SCALE_VARS.*$", iwm, re.M).group(0))),
        ts_variables=tuple(dict.fromkeys(re.findall(r'"(vo[a-z]+)"', ts))),
        tke_ebb=float(m.group(1)),
        builds_zstar="create_z_star_from_thicknesses(" in builder,
        nn_e_resolved=int(nn_e.group(1)),
    )


# --------------------------------------------------------------------------
# Rules.  First full match on "ns:group.key" wins.
# --------------------------------------------------------------------------
class Rule(NamedTuple):
    pat: str
    kind: str                       # map | off | nm | inert
    card: Callable | None = None    # ctx -> card value
    want: Callable | None = None    # (nemo_scalar, ctx) -> expected card value
    why: str = ""
    effect: str = ""
    switch: tuple = ()              # inert: (("ns:group.key", value), ...)
    cls: str = ""                   # DIFFERS: "a:<rung>" | "b:D<n>" | "c"


def C(path: str) -> Callable:
    return lambda ctx: ctx.flat[path]


def has_file(name: str) -> Callable:
    return lambda ctx: name if any(f.startswith(name) for f in ctx.builder_files) else "not opened"


def file_name(nemo: object, ctx) -> object:
    return nemo


NO_STATE = "no effect on model state (I/O, decomposition, diagnostics, control)"
REPLAY = ("replayed: the card is driven by the record's surface frames "
          "(qns, qsr, emp, sfx, taum, utau, vtau, rnf, rnf_tsc, fr_i, "
          "rCdU_ice); it computes none of this itself")
SURF = "surface fluxes (replayed from NEMO)"
ICE = "sea ice (replayed from NEMO; out of scope, Decision 50)"
RK3 = ("oce:none", "key_RK3")       # compile key, checked against CPP_KEYS


def _rules() -> list[Rule]:
    r: list[Rule] = []
    a = r.append
    # ---- sea ice: the card runs no SI3 ----
    a(Rule(r"ice:.*", "nm", why=REPLAY + "; no ice model on the card", effect=ICE))
    # ---- no model-state groups ----
    for g in ("namctl", "nammpp", "namnc4", "namtile", "namtrd", "nam_diadct", "namobs"):
        a(Rule(rf"oce:{g}\..*", "nm", why="card is a single in-memory replay; no counterpart",
               effect=NO_STATE))
    # ---- namcfg ----
    a(Rule(r"oce:namcfg\.ln_read_cfg", "map", card=has_file("ORCA_R2_zps_domcfg"),
           want=lambda n, c: "ORCA_R2_zps_domcfg" if n else "not opened",
           why="builder reads the domain file", effect="grid"))
    a(Rule(r"oce:namcfg\.cn_domcfg", "map", card=has_file("ORCA_R2_zps_domcfg"),
           want=file_name, why="builder reads the same domain file", effect="grid"))
    a(Rule(r"oce:namcfg\.ln_closea", "off", why="no closed-sea treatment on the card", effect="grid"))
    a(Rule(r"oce:namcfg\.cn_domcfg_out", "inert", switch=(("oce:namcfg.ln_write_cfg", False),),
           why="ln_write_cfg = F"))
    a(Rule(r"oce:namcfg\.ln_write_cfg", "nm", why="output only", effect=NO_STATE))
    # ---- nameos ----
    a(Rule(r"oce:nameos\.ln_eos80", "map", card=lambda c: c.flat["eos"] == "nemo_eos80",
           why="eos = 'nemo_eos80'", effect="density"))
    a(Rule(r"oce:nameos\.ln_(teos10|seos)", "map",
           card=lambda c: c.flat["eos"] in ("nemo_teos10", "nemo_seos"),
           why="card eos is EOS-80", effect="density"))
    a(Rule(r"oce:nameos\.rn_.*", "inert", switch=(("oce:nameos.ln_seos", False),),
           why="S-EOS coefficients; ln_seos = F"))
    # ---- namwad ----
    a(Rule(r"oce:namwad\.ln_wd_dl", "off", why="no wetting and drying on the card", effect="grid"))
    a(Rule(r"oce:namwad\..*", "inert", switch=(("oce:namwad.ln_wd_dl", False),), why="ln_wd_dl = F"))
    # ---- namdom ----
    a(Rule(r"oce:namdom\.rn_dt", "map", card=C("card.dt_s"), why="card time step", effect="time step"))
    a(Rule(r"oce:namdom\.rn_atfp", "inert", switch=(RK3,),
           why="Asselin filter is leapfrog-only; ORCA2_OMIP_L4 compiles key_RK3"))
    a(Rule(r"oce:namdom\.ln_c1d", "off", why="3-D card", effect="grid"))
    a(Rule(r"oce:namdom\.ln_shuman", "off", why="no Shuman filter on the card", effect="momentum"))
    a(Rule(r"oce:namdom\.ln_meshmask", "nm", why="output only", effect=NO_STATE))
    # ---- namrun ----
    a(Rule(r"oce:namrun\.nn_itend", "map", card=C("card.n_steps"),
           why="card's declared run length (the ladders replay 10 steps)",
           effect="run length only", cls="c"))
    a(Rule(r"oce:namrun\.ln_rstart", "map", card=lambda c: False,
           why="card builds its state from the deck files (rest + climatology), no restart",
           effect="initial state"))
    a(Rule(r"oce:namrun\.nn_rstctl", "inert", switch=(("oce:namrun.ln_rstart", False),),
           why="ln_rstart = F"))
    a(Rule(r"oce:namrun\.ln_1st_euler", "inert", switch=(RK3,),
           why="leapfrog start only; key_RK3"))
    a(Rule(r"oce:namrun\.ln_top", "inert", switch=(("oce:none", "no key_top"),),
           why="passive tracers need key_top, which ORCA2_OMIP_L4 does not compile"))
    a(Rule(r"oce:namrun\.(nn_date0|nn_time0|nn_leapy)", "nm",
           why="no calendar on the card: the replay hands it each step's forcing",
           effect="forcing timing (replayed)"))
    a(Rule(r"oce:namrun\..*", "nm", why="run control / output", effect=NO_STATE))
    # ---- namlbc ----
    a(Rule(r"oce:namlbc\.rn_shlat", "map", card=C("mask.fmask_max"),
           why="largest value of the card's vorticity F mask (no-slip = 2)", effect="momentum"))
    a(Rule(r"oce:namlbc\.ln_vorlat", "off", why="not built", effect="momentum"))
    # ---- nam_vvl ----
    a(Rule(r"oce:nam_vvl\.ln_vvl_zstar", "map", card=lambda c: c.builds_zstar,
           why="builder builds z-star thicknesses", effect="vertical coordinate"))
    a(Rule(r"oce:nam_vvl\.ln_vvl_(ztilde|layer|ztilde_as_zstar|zstar_at_eqtor)", "off",
           why="card is z-star", effect="vertical coordinate"))
    a(Rule(r"oce:nam_vvl\.(rn_ahe3|rn_rst_e3t|rn_lf_cutoff|rn_zdef_max)", "inert",
           switch=(("oce:nam_vvl.ln_vvl_ztilde", False),), why="z-tilde only; ln_vvl_ztilde = F"))
    a(Rule(r"oce:nam_vvl\.nn_vvl_interp", "inert", switch=(("oce:none", "key_qco"),),
           why="key_qco builds thicknesses from ratios, not vvl interpolation"))
    a(Rule(r"oce:nam_vvl\.ln_vvl_dbg", "nm", why="debug prints", effect=NO_STATE))
    # ---- namtsd ----
    a(Rule(r"oce:namtsd\.ln_tsd_init", "map", card=lambda c: True,
           why="builder reads the T/S climatology as its initial state", effect="initial state"))
    a(Rule(r"oce:namtsd\.ln_tsd_dmp", "nm", why="card has no interior T/S damping",
           effect="interior T/S damping"))
    for v, f in (("tem", "data_1m_potential_temperature_nomask"), ("sal", "data_1m_salinity_nomask")):
        a(Rule(rf"oce:namtsd\.sn_{v}%clname", "map", card=has_file(f), want=file_name,
               why="builder opens the same climatology file", effect="initial state"))
    a(Rule(r"oce:namtsd\.sn_(tem|sal)%clvar", "map", card=lambda c: c.ts_variables,
           want=lambda n, c: tuple(v for v in c.ts_variables) if n in c.ts_variables else (n,),
           why="variables the builder reads", effect="initial state"))
    a(Rule(r"oce:namtsd\.sn_(tem|sal)%.*", "map", card=lambda c: "two-record fldread weights",
           want=lambda n, c: "two-record fldread weights",
           why="builder applies fldread's two-record monthly interpolation at kt=1",
           effect="initial state"))
    a(Rule(r"oce:namtsd\.cn_dir", "nm", why="card reads from the deck root it is given",
           effect=NO_STATE))
    # ---- nam_tide ----
    a(Rule(r"oce:nam_tide\.ln_tide", "map", card=C("tidal_forcing.enabled"), effect="tides"))
    a(Rule(r"oce:nam_tide\..*", "inert", switch=(("oce:nam_tide.ln_tide", False),), why="ln_tide = F"))
    # ---- namsbc ----
    a(Rule(r"oce:namsbc\.ln_traqsr", "map",
           card=lambda c: c.flat["physics.shortwave_penetration.scheme"] != "none",
           why="card computes shortwave penetration itself", effect="shortwave penetration"))
    for k in ("ln_usr", "ln_flx", "ln_abl", "ln_cpl", "ln_mixcpl", "ln_dm2dc", "ln_apr_dyn",
              "ln_wave", "ln_ice_embd"):
        a(Rule(rf"oce:namsbc\.{k}", "off", why="not on the card", effect=SURF))
    a(Rule(r"oce:namsbc\.nn_components", "inert", switch=(("oce:namsbc.ln_cpl", False),),
           why="coupled only; ln_cpl = F"))
    a(Rule(r"oce:namsbc\..*", "nm", why=REPLAY, effect=SURF))
    a(Rule(r"oce:namsbc_(blk|ssr|fwb)\..*", "nm", why=REPLAY, effect=SURF))
    # ---- namsbc_rnf ----
    a(Rule(r"oce:namsbc_rnf\.ln_rnf_depth", "map",
           card=lambda c: c.flat["runoff_depth_spread_m"] != 0.0,
           why="runoff_depth_spread_m = 0 (Decision 126)", effect="runoff"))
    a(Rule(r"oce:namsbc_rnf\.ln_rnf_depth_ini", "off", why="no depth file on the card", effect="runoff"))
    a(Rule(r"oce:namsbc_rnf\.ln_rnf_icb", "off", why="no iceberg runoff", effect="runoff"))
    a(Rule(r"oce:namsbc_rnf\.(rn_dep_max|rn_rnf_max|nn_rnf_depth_file)", "inert",
           switch=(("oce:namsbc_rnf.ln_rnf_depth", False), ("oce:namsbc_rnf.ln_rnf_depth_ini", False)),
           why="depth spreading only; ln_rnf_depth = ln_rnf_depth_ini = F"))
    a(Rule(r"oce:namsbc_rnf\.sn_t_rnf%.*", "inert", switch=(("oce:namsbc_rnf.ln_rnf_tem", False),),
           why="ln_rnf_tem = F"))
    a(Rule(r"oce:namsbc_rnf\.sn_s_rnf%.*", "inert", switch=(("oce:namsbc_rnf.ln_rnf_sal", False),),
           why="ln_rnf_sal = F"))
    a(Rule(r"oce:namsbc_rnf\.sn_dep_rnf%.*", "inert",
           switch=(("oce:namsbc_rnf.ln_rnf_depth_ini", False),), why="ln_rnf_depth_ini = F"))
    a(Rule(r"oce:namsbc_rnf\.sn_i_rnf%.*", "inert", switch=(("oce:namsbc_rnf.ln_rnf_icb", False),),
           why="ln_rnf_icb = F"))
    a(Rule(r"oce:namsbc_rnf\.(ln_rnf_mouth|rn_hrnf|rn_avt_rnf|sn_cnf%.*)", "nm",
           why="card adds no river-mouth vertical mixing (NEMO: extra 1e-3 m2/s over the "
               "top 15 m where the river-mouth map is set)",
           effect="vertical mixing at river mouths"))
    a(Rule(r"oce:namsbc_rnf\..*", "nm", why=REPLAY, effect=SURF))
    # ---- nambdy ----
    a(Rule(r"oce:nambdy\.ln_bdy", "off", why="global card, no open boundaries", effect="grid"))
    a(Rule(r"oce:nambdy\..*", "inert", switch=(("oce:nambdy.ln_bdy", False),), why="ln_bdy = F"))
    # ---- namzdf ----
    VM = "vertical mixing"
    a(Rule(r"oce:namzdf\.ln_zdftke", "map",
           card=lambda c: c.flat["physics.vertical_mixing.scheme"] == "tke", effect=VM))
    a(Rule(r"oce:namzdf\.ln_zdf(cst|ric|gls|osm)", "map",
           card=lambda c: c.flat["physics.vertical_mixing.scheme"] != "tke",
           why="card's closure is TKE", effect=VM))
    a(Rule(r"oce:namzdf\.ln_zdf(mfc|swm)", "off", why="not on the card", effect=VM))
    a(Rule(r"oce:namzdf\.ln_zdfevd", "map",
           card=lambda c: c.flat["physics.convection.scheme"] == "enhanced_diffusion", effect=VM))
    a(Rule(r"oce:namzdf\.nn_evdm", "map",
           card=lambda c: 0.0 if c.flat["physics.convection.enhanced_diffusion.nu_conv"] == 0.0 else 1.0,
           why="0 = tracers only (card nu_conv = 0)", effect=VM))
    a(Rule(r"oce:namzdf\.rn_evd", "map", card=C("physics.convection.enhanced_diffusion.K_conv"), effect=VM))
    a(Rule(r"oce:namzdf\.ln_zdfnpc", "off", why="not on the card", effect=VM))
    a(Rule(r"oce:namzdf\.nn_npcp?", "inert", switch=(("oce:namzdf.ln_zdfnpc", False),), why="ln_zdfnpc = F"))
    a(Rule(r"oce:namzdf\.ln_zdfddm", "map", card=C("physics.vertical_mixing.ddm.enabled"), effect=VM))
    a(Rule(r"oce:namzdf\.rn_avts", "map", card=C("physics.vertical_mixing.ddm.rn_avts"), effect=VM))
    a(Rule(r"oce:namzdf\.rn_hsbfr", "map", card=C("physics.vertical_mixing.ddm.rn_hsbfr"), effect=VM))
    a(Rule(r"oce:namzdf\.ln_zdfiwm", "map", card=C("physics.vertical_mixing.iwm.enabled"), effect=VM))
    a(Rule(r"oce:namzdf\.ln_zad_aimp", "map", card=C("adaptive_implicit_vertadv"), effect="vertical advection"))
    a(Rule(r"oce:namzdf\.rn_avm0", "map", card=lambda c: max(c.flat["seed.avm_values"]),
           why="card's carried avm seed", effect=VM))
    a(Rule(r"oce:namzdf\.rn_avt0", "map", card=C("seed.avt_max"), why="card's carried avt seed", effect=VM))
    a(Rule(r"oce:namzdf\.nn_avb", "map", card=lambda c: 0.0 if len(c.flat["seed.avm_values"]) == 1 else 1.0,
           why="0 = no vertical profile (card seed uniform)", effect=VM))
    a(Rule(r"oce:namzdf\.nn_havtb", "map",
           card=lambda c: 1.0 if abs(c.flat["seed.avt_min_over_max"] - 0.1) < 1e-12 else 0.0,
           why="1 = equatorial band, 0.1 at the equator (card seed min/max)", effect=VM))
    # ---- namzdf_tke ----
    T = "physics.vertical_mixing.tke."
    for k, f in (("rn_ediff", "c_k"), ("rn_ediss", "c_eps"), ("rn_emin0", "tke_surface_min"),
                 ("rn_bshear", "bshear_floor"), ("rn_mxl0", "mxl0_min_m"), ("rn_lc", "lc_coeff"),
                 ("rn_efr", "etau_frac"), ("ln_lc", "lc")):
        a(Rule(rf"oce:namzdf_tke\.{k}", "map", card=C(T + f), effect=VM))
    a(Rule(r"oce:namzdf_tke\.rn_emin", "map", card=C(T + "tke_background"),
           want=lambda n, c: 1.0e-10,
           why="ln_zdfiwm = T forces rn_emin = 1e-10 in zdf_tke_init", effect=VM))
    a(Rule(r"oce:namzdf_tke\.rn_ebb", "map", card=lambda c: c.tke_ebb,
           why="hard-coded in the card's TKE scheme (not a card field)", effect=VM))
    a(Rule(r"oce:namzdf_tke\.nn_mxl", "map", card=lambda c: float(c.flat[T + "tke_mxl_choice"]),
           why="card's choice 3 = NEMO nn_mxl 3", effect=VM))
    a(Rule(r"oce:namzdf_tke\.ln_mxl0", "map", card=lambda c: c.flat[T + "tke_mxl_choice"] == 3,
           why="choice 3 carries the surface stress anchor", effect=VM))
    a(Rule(r"oce:namzdf_tke\.nn_mxlice", "map", card=lambda c: float(c.flat[T + "nemo_mxlice"]),
           why="needs SI3's mean ice thickness, which the card does not compute", effect=VM,
           cls="b:D135"))
    a(Rule(r"oce:namzdf_tke\.rn_mxlice", "inert", switch=(("oce:namzdf_tke.nn_mxlice", 2.0),),
           why="read only by nn_mxlice = 1; NEMO runs 2"))
    a(Rule(r"oce:namzdf_tke\.nn_pdl", "map",
           card=lambda c: 1.0 if c.flat[T + "prandtl_mode"] == "nemo_ri" else 0.0, effect=VM))
    a(Rule(r"oce:namzdf_tke\.nn_etau", "map",
           card=lambda c: 1.0 if c.flat[T + "etau_mode"] == "below_ml" else 0.0, effect=VM))
    a(Rule(r"oce:namzdf_tke\.nn_htau", "map",
           card=lambda c: 1.0 if c.flat[T + "etau_htau_mode"] == "latitude" else 0.0, effect=VM))
    a(Rule(r"oce:namzdf_tke\.nn_eice", "map", card=lambda c: float(c.flat[T + "eice"]), effect=VM))
    a(Rule(r"oce:namzdf_tke\.nn_bc_surf", "map",
           card=lambda c: 1.0 if c.flat[T + "surface_bc"] == "nemo_dirichlet" else 0.0, effect=VM))
    a(Rule(r"oce:namzdf_tke\.nn_bc_bot", "inert", switch=(("oce:namsbc.ln_wave", False),),
           why="read but never used outside wave coupling; ln_wave = F"))
    a(Rule(r"oce:namzdf_tke\.ln_mxhsw", "off", why="wave-height mixing length not on the card", effect=VM))
    # ---- namzdf_iwm ----
    a(Rule(r"oce:namzdf_iwm\.ln_mevar", "map", card=C("physics.vertical_mixing.iwm.mevar"), effect=VM))
    a(Rule(r"oce:namzdf_iwm\.ln_tsdiff", "map", card=C("physics.vertical_mixing.iwm.tsdiff"), effect=VM))
    a(Rule(r"oce:namzdf_iwm\.sn_[a-z]+%clname", "map", card=has_file("zdfiwm_forcing_orca2"),
           want=file_name, why="builder opens the same map file", effect=VM))
    a(Rule(r"oce:namzdf_iwm\.sn_[a-z]+%clvar", "map",
           card=lambda c: c.iwm_variables, want=lambda n, c: c.iwm_variables if n in c.iwm_variables else (n,),
           why="variables the card's IWM reader takes", effect=VM))
    a(Rule(r"oce:namzdf_iwm\.sn_[a-z]+%.*", "map", card=lambda c: "read once",
           want=lambda n, c: "read once",
           why="one yearly climatology record; the card reads it once at build", effect=VM))
    a(Rule(r"oce:namzdf_iwm\.cn_dir", "nm", why="card reads from the deck root it is given",
           effect=NO_STATE))
    # ---- namdrg / namdrg_bot ----
    D = "bottom_drag."
    a(Rule(r"oce:namdrg\.ln_drg_off", "map",
           card=lambda c: c.flat[D + "bottom_drag_scheme"] == "none", effect="bottom drag"))
    a(Rule(r"oce:namdrg\.ln_lin", "map",
           card=lambda c: c.flat[D + "bottom_drag_scheme"] == "nemo_linear", effect="bottom drag"))
    a(Rule(r"oce:namdrg\.ln_(non_lin|loglayer)", "map",
           card=lambda c: c.flat[D + "bottom_drag_scheme"] != "nemo_linear", effect="bottom drag"))
    a(Rule(r"oce:namdrg\.ln_drgimp", "map", card=C("zdf_drag_in_matrix"), effect="bottom drag"))
    a(Rule(r"oce:namdrg\.ln_drgice_imp", "map", card=C("implicit_ice_ocean_top_drag"),
           why="ice-ocean top drag left off until the card computes its own ice-ocean "
               "coefficient (rounds 305-306)", effect="ice-ocean drag", cls="b:D133"))
    a(Rule(r"oce:namdrg_bot\.rn_cd0", "map", card=C(D + "bottom_drag_cd0"), effect="bottom drag"))
    a(Rule(r"oce:namdrg_bot\.rn_uc0", "map", card=C(D + "bottom_drag_uc0"), effect="bottom drag"))
    a(Rule(r"oce:namdrg_bot\.(rn_cdmax|rn_ke0|rn_z0)", "inert", switch=(("oce:namdrg.ln_lin", True),),
           why="linear drag uses only rn_Cd0*rn_Uc0; ln_lin = T"))
    a(Rule(r"oce:namdrg_bot\.ln_boost", "off", why="not on the card", effect="bottom drag"))
    a(Rule(r"oce:namdrg_bot\.rn_boost", "inert", switch=(("oce:namdrg_bot.ln_boost", False),),
           why="ln_boost = F"))
    # ---- namtra_ldf ----
    G = "gm_redi."
    LT = "lateral tracer mixing"
    a(Rule(r"oce:namtra_ldf\.ln_traldf_(lap|iso)", "map",
           card=lambda c: c.flat[G + "slope_scheme"] == "nemo_iso_lap",
           why="top-level gm_redi is the copy the lat-lon step reads (physics.lateral_mixing "
               "scheme = 'none')", effect=LT))
    a(Rule(r"oce:namtra_ldf\.ln_traldf_(off|blp|lev|hor|triad)", "map",
           card=lambda c: c.flat[G + "slope_scheme"] != "nemo_iso_lap", effect=LT))
    a(Rule(r"oce:namtra_ldf\.ln_traldf_msc", "map", card=C(G + "msc_stabilize"),
           why="card keeps the full K33 term implicit (GYRE form) instead of NEMO's "
               "stabilising split", effect=LT, cls="c"))
    a(Rule(r"oce:namtra_ldf\.rn_slpmax", "map", card=C(G + "S_max"), effect=LT))
    a(Rule(r"oce:namtra_ldf\.(ln_triad_iso|ln_botmix_triad|rn_sw_triad)", "inert",
           switch=(("oce:namtra_ldf.ln_traldf_triad", False),), why="ln_traldf_triad = F"))
    a(Rule(r"oce:namtra_ldf\.nn_aht_ijk_t", "map",
           card=lambda c: 20.0 if c.flat[G + "kappa_redi_lat_scaling"] else 0.0,
           why="card: constant kappa_Redi everywhere (lat scaling off); NEMO 20 = "
               "1/2*rn_Ud*max(e1,e2) per face", effect=LT, cls="c"))
    a(Rule(r"oce:namtra_ldf\.rn_ud", "map", card=C(G + "kappa_redi_diffusive_velocity"),
           why="card carries no diffusive velocity; it uses kappa_Redi = 1000 m2/s",
           effect=LT, cls="c"))
    a(Rule(r"oce:namtra_ldf\.rn_ld", "inert", switch=(("oce:namtra_ldf.nn_aht_ijk_t", 20.0),),
           why="nn_aht_ijk_t = 20 takes the length from max(e1,e2); rn_Ld only prints aht0"))
    # ---- namtra_eiv ----
    EDDY = "eddy-induced advection / mixed-layer eddies"
    a(Rule(r"oce:namtra_eiv\.ln_ldfeiv", "map", card=lambda c: c.flat[G + "kappa_GM"] > 0.0,
           why="card kappa_GM = 0 (declared unmeasured: 'staged_gm_eiv')", effect=EDDY, cls="c"))
    a(Rule(r"oce:namtra_eiv\.ln_eke_equ", "off", why="not on the card", effect=EDDY))
    a(Rule(r"oce:namtra_eiv\..*", "nm", why="follows ln_ldfeiv: the card runs no eddy-induced velocity",
           effect=EDDY))
    # ---- namdyn_ldf ----
    LM = "lateral momentum mixing"
    a(Rule(r"oce:namdyn_ldf\.(ln_dynldf_lap|ln_dynldf_lev)", "map",
           card=lambda c: c.flat["lateral_viscosity_operator"] == "nemo_div_curl", effect=LM))
    a(Rule(r"oce:namdyn_ldf\.ln_dynldf_(off|blp|hor|iso)", "map",
           card=lambda c: c.flat["lateral_viscosity_operator"] != "nemo_div_curl", effect=LM))
    a(Rule(r"oce:namdyn_ldf\.nn_dynldf_typ", "map",
           card=lambda c: 0.0 if c.flat["lateral_viscosity_operator"] == "nemo_div_curl" else 1.0,
           why="0 = divergence-curl form", effect=LM))
    a(Rule(r"oce:namdyn_ldf\.nn_ahm_ijk_t", "map",
           card=lambda c: -30.0 if c.flat["lateral_viscosity_coefficient_source"] == "nemo_ahm_3d_file" else 0.0,
           why="card reads eddy_viscosity_3D.nc", effect=LM))
    a(Rule(r"oce:namdyn_ldf\.(rn_uv|rn_lv)", "inert", switch=(("oce:namdyn_ldf.nn_ahm_ijk_t", -30.0),),
           why="nn_ahm_ijk_t = -30 reads the coefficient from file"))
    a(Rule(r"oce:namdyn_ldf\.rn_ahm_b", "inert", switch=(("oce:namdyn_ldf.ln_dynldf_iso", False),),
           why="iso-neutral momentum background only; ln_dynldf_iso = F"))
    a(Rule(r"oce:namdyn_ldf\.(rn_csmc|rn_minfac|rn_maxfac)", "inert",
           switch=(("oce:namdyn_ldf.nn_ahm_ijk_t", -30.0),), why="Smagorinsky only; nn_ahm_ijk_t = -30"))
    # ---- namtra_qsr ----
    SW = "shortwave penetration"
    S = "physics.shortwave_penetration."
    a(Rule(r"oce:namtra_qsr\.ln_qsr_rgb", "map", card=lambda c: c.flat[S + "scheme"] == "nemo_qsr_rgb",
           effect=SW))
    a(Rule(r"oce:namtra_qsr\.ln_qsr_(2bd|5bd|bio)", "map",
           card=lambda c: c.flat[S + "scheme"] != "nemo_qsr_rgb", effect=SW))
    a(Rule(r"oce:namtra_qsr\.nn_chlprfl", "map",
           card=lambda c: 1.0 if c.flat[S + "rgb_chl_profile"] == "morel_berthon" else 0.0, effect=SW))
    a(Rule(r"oce:namtra_qsr\.rn_abs", "map", card=C(S + "rgb_ir_fraction"), effect=SW))
    a(Rule(r"oce:namtra_qsr\.rn_si0", "map", card=C(S + "rgb_ir_extinction_m"), effect=SW))
    a(Rule(r"oce:namtra_qsr\.rn_si1", "inert", switch=(("oce:namtra_qsr.ln_qsr_2bd", False),),
           why="two-band only; ln_qsr_2bd = F"))
    a(Rule(r"oce:namtra_qsr\.rn_par", "inert",
           switch=(("oce:namtra_qsr.ln_qsr_5bd", False), ("oce:namtra_qsr.ln_qsr_bio", False)),
           why="five-band / biology only"))
    a(Rule(r"oce:namtra_qsr\.(nn_chldta|sn_chl%.*|cn_dir)", "nm",
           why="the replay reads chlorophyll from the same file each step (forcing input, "
               "not a card setting)", effect=SW))
    # ---- nambbc (geothermal) ----
    a(Rule(r"oce:nambbc\.rn_geoflx_cst", "inert", switch=(("oce:nambbc.nn_geoflx", 2.0),),
           why="nn_geoflx = 2 reads the map"))
    a(Rule(r"oce:nambbc\..*", "nm", why="card's step has no geothermal heat source",
           effect="interior tracer sources (geothermal)"))
    # ---- nambbl ----
    BBL = "bottom boundary layer"
    a(Rule(r"oce:nambbl\.ln_trabbl", "map",
           card=lambda c: c.flat["card.bbl_diffusive_option"] > 0 or c.flat["card.bbl_adv_option"] > 0,
           effect=BBL))
    a(Rule(r"oce:nambbl\.nn_bbl_ldf", "map", card=lambda c: float(c.flat["card.bbl_diffusive_option"]),
           effect=BBL))
    a(Rule(r"oce:nambbl\.nn_bbl_adv", "map", card=lambda c: float(c.flat["card.bbl_adv_option"]),
           effect=BBL))
    a(Rule(r"oce:nambbl\.rn_ahtbbl", "map", card=C("card.bbl_aht_m2_s"), effect=BBL))
    a(Rule(r"oce:nambbl\.rn_gambbl", "inert", switch=(("oce:nambbl.nn_bbl_adv", 0.0),),
           why="advective BBL only; nn_bbl_adv = 0"))
    # ---- namtra_dmp ----
    a(Rule(r"oce:namtra_dmp\..*", "nm", why="card has no interior T/S damping",
           effect="interior T/S damping"))
    # ---- namtra_adv ----
    TA = "tracer advection"
    a(Rule(r"oce:namtra_adv\.ln_traadv_fct", "map",
           card=lambda c: c.flat["tracer_advection"].startswith("fct"), effect=TA))
    a(Rule(r"oce:namtra_adv\.nn_fct_[hv]", "map",
           card=lambda c: float(c.flat["tracer_advection"][3:]) if c.flat["tracer_advection"].startswith("fct") else -1.0,
           why="card 'fct2' = second order both ways", effect=TA))
    a(Rule(r"oce:namtra_adv\.nn_fct_imp", "inert", switch=(("oce:namzdf.ln_zad_aimp", False),),
           why="adaptive-implicit vertical advection only; ln_zad_aimp = F"))
    a(Rule(r"oce:namtra_adv\.ln_traadv_(off|cen|mus|ubs|qck)", "map",
           card=lambda c: not c.flat["tracer_advection"].startswith("fct"), effect=TA))
    a(Rule(r"oce:namtra_adv\.(nn_cen_h|nn_cen_v)", "inert", switch=(("oce:namtra_adv.ln_traadv_cen", False),),
           why="ln_traadv_cen = F"))
    a(Rule(r"oce:namtra_adv\.ln_mus_ups", "inert", switch=(("oce:namtra_adv.ln_traadv_mus", False),),
           why="ln_traadv_mus = F"))
    a(Rule(r"oce:namtra_adv\.nn_ubs_v", "inert", switch=(("oce:namtra_adv.ln_traadv_ubs", False),),
           why="ln_traadv_ubs = F"))
    # ---- namtra_mle ----
    a(Rule(r"oce:namtra_mle\.ln_mle", "map", card=lambda c: c.flat["physics.mle"] is not None,
           why="card physics.mle = None (not in its unmeasured list either)", effect=EDDY, cls="c"))
    a(Rule(r"oce:namtra_mle\..*", "nm", why="follows ln_mle: the card runs no mixed-layer eddies",
           effect=EDDY))
    # ---- namdyn_adv / vor / hpg / spg ----
    MO = "momentum"
    a(Rule(r"oce:namdyn_adv\.ln_dynadv_vec", "map",
           card=lambda c: c.flat["momentum_advection"] == "vector_invariant", effect=MO))
    a(Rule(r"oce:namdyn_adv\.ln_dynadv_(off|cen2|up3)", "map",
           card=lambda c: c.flat["momentum_advection"] != "vector_invariant", effect=MO))
    a(Rule(r"oce:namdyn_adv\.nn_dynkeg", "map",
           card=lambda c: 0.0 if c.flat["ke_gradient_scheme"] == "c2" else 1.0, effect=MO))
    a(Rule(r"oce:namdyn_vor\.ln_dynvor_een", "map",
           card=lambda c: c.flat["vorticity_scheme"] == "een_total", effect=MO))
    a(Rule(r"oce:namdyn_vor\.ln_dynvor_(ens|ene|ent)", "map",
           card=lambda c: not c.flat["vorticity_scheme"].startswith("een"), effect=MO))
    a(Rule(r"oce:namdyn_vor\.nn_e3f_typ", "map",
           card=lambda c: 0.0 if c.flat["een_e3f_scheme"] == "nemo_avg4" else 1.0, effect=MO))
    a(Rule(r"oce:namdyn_vor\.ln_dynvor_(mix|msk)", "off", why="not on the card", effect=MO))
    a(Rule(r"oce:namdyn_hpg\.ln_hpg_sco", "map", card=lambda c: c.flat["pgf_scheme"] == "nemo_sco", effect=MO))
    a(Rule(r"oce:namdyn_hpg\.ln_hpg_(zco|djc|prj|isf)", "map",
           card=lambda c: c.flat["pgf_scheme"] != "nemo_sco", effect=MO))
    a(Rule(r"oce:namdyn_hpg\.ln_hpg_djc_vn[hv]", "inert", switch=(("oce:namdyn_hpg.ln_hpg_djc", False),),
           why="ln_hpg_djc = F"))
    B = "barotropic."
    a(Rule(r"oce:namdyn_spg\.ln_dynspg_ts", "map",
           card=lambda c: c.flat[B + "barotropic_solver"] == "explicit_substep", effect=MO))
    a(Rule(r"oce:namdyn_spg\.ln_dynspg_exp", "map",
           card=lambda c: c.flat[B + "barotropic_solver"] != "explicit_substep", effect=MO))
    a(Rule(r"oce:namdyn_spg\.ln_bt_fw", "map",
           card=lambda c: c.flat[B + "barotropic_time_filter"] == "nemo_ab3am4",
           why="card's split-explicit program is the ln_bt_fw = T, nn_bt_flt = 3 one certified "
               "on the ORCA2 records", effect=MO))
    a(Rule(r"oce:namdyn_spg\.nn_bt_flt", "map",
           card=lambda c: 3.0 if c.flat[B + "barotropic_time_filter"] == "nemo_ab3am4" else -1.0, effect=MO))
    a(Rule(r"oce:namdyn_spg\.rn_bt_alpha", "map", card=C(B + "nemo_barotropic_filter_alpha"), effect=MO))
    a(Rule(r"oce:namdyn_spg\.ln_bt_auto", "map",
           card=lambda c: c.flat[B + "n_barotropic_substeps"] == c.nn_e_resolved,
           why="card fixes the substep count at NEMO's resolved value (ocean.output nn_e)",
           effect=MO))
    a(Rule(r"oce:namdyn_spg\.nn_e", "inert", switch=(("oce:namdyn_spg.ln_bt_auto", True),),
           why="ln_bt_auto = T recomputes it"))
    a(Rule(r"oce:namdyn_spg\.rn_bt_cmax", "nm",
           why="card does not derive the substep count; it fixes NEMO's resolved 65", effect=MO))
    # ---- namberg ----
    a(Rule(r"oce:namberg\.ln_icebergs", "map", card=C("card.icebergs_enabled"),
           why="icebergs off on the card and the record", effect="icebergs", cls="b:D7"))
    a(Rule(r"oce:namberg\..*", "nm", why="follows ln_icebergs: no iceberg model on the card",
           effect="icebergs"))
    # ---- namisf / namsto ----
    a(Rule(r"oce:namisf\.ln_isf", "off", why="no ice-shelf cavities", effect="grid"))
    a(Rule(r"oce:namisf\..*", "inert", switch=(("oce:namisf.ln_isf", False),), why="ln_isf = F"))
    a(Rule(r"oce:namsto\.ln_sto_.*", "off", why="no stochastic parameterisation", effect="stochastic"))
    a(Rule(r"oce:namsto\..*", "inert",
           switch=tuple((f"oce:namsto.ln_sto_{s}", False) for s in ("ldf", "hpg", "pstar", "trd", "eos", "trc")),
           why="every ln_sto_* = F"))
    return r


RULES = _rules()


def show(v) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return show(v[0])
        if len(set(map(repr, v))) == 1:
            return f"{len(v)}*{show(v[0])}"
        return ",".join(show(x) for x in v)
    if v is None:
        return "None"
    return str(v)


def _switch_holds(name: str, want, nemo: dict, cpp: str) -> bool:
    if name == "oce:none":
        if want == "no key_top":
            return "key_top" not in cpp.split()
        return want in cpp.split()
    ns, rest = name.split(":", 1)
    g, k = rest.split(".", 1)
    val = nemo.get((ns, g, k))
    require(val is not None, f"INERT switch {name} is not a resolved key")
    return same(val[0], want)


def classify(key: tuple[str, str, str], nemo: dict, ctx, cpp: str) -> dict:
    ns, g, k = key
    full = f"{ns}:{g}.{k}"
    val = nemo[key]
    scalar = val[0] if len(val) == 1 else val
    for rule in RULES:
        if not re.fullmatch(rule.pat, full):
            continue
        row = {"why": rule.why, "effect": rule.effect, "class": ""}
        if rule.kind == "nm":
            row.update(status="NOT-MODELLED", card="-")
        elif rule.kind == "inert":
            for name, want in rule.switch:
                require(_switch_holds(name, want, nemo, cpp),
                        f"{full}: INERT switch {name} != {want!r}")
            row.update(status="INERT", card="-")
        elif rule.kind == "off":
            require(scalar is False or scalar == 0.0 or scalar == "none",
                    f"{full}: 'off' rule but NEMO value is {scalar!r}")
            row.update(status="MATCH", card="not built (off)")
        else:
            card_v = rule.card(ctx)
            want = rule.want(scalar, ctx) if rule.want else scalar
            if isinstance(card_v, bool) or isinstance(want, bool):
                ok = isinstance(card_v, bool) and isinstance(want, bool) and card_v == want
            elif isinstance(card_v, (int, float)) and isinstance(want, (int, float)):
                ok = same(float(card_v), float(want))
            else:
                ok = same(card_v, want)
            row.update(status="MATCH" if ok else "DIFFERS", card=show(card_v))
            if not ok:
                require(bool(rule.cls), f"{full}: DIFFERS without an (a)/(b)/(c) class")
                row["class"] = rule.cls
        return row
    raise AuditError(f"UNCLASSIFIED {full}")


# --------------------------------------------------------------------------
def audit(record: Path, exp00: Path, shared: Path, ctx=None) -> dict:
    dump = {**parse_namelist(record / "output.namelist.dyn", "oce"),
            **parse_namelist(record / "output.namelist.ice", "ice")}
    rec_src = overlay(parse_namelist(record / "namelist_ref", "oce"),
                      parse_namelist(record / "namelist_cfg", "oce"),
                      parse_namelist(record / "namelist_ice_ref", "ice"),
                      parse_namelist(record / "namelist_ice_cfg", "ice"))
    exp_src = overlay(parse_namelist(shared / "namelist_ref", "oce"),
                      parse_namelist(exp00 / "namelist_cfg", "oce"),
                      parse_namelist(shared / "namelist_ice_ref", "ice"),
                      parse_namelist(exp00 / "namelist_ice_cfg", "ice"))
    rec_res = resolve_against_dump(dump, rec_src)
    exp_res = resolve_against_dump(dump, exp_src)

    selfcheck = {"equal": 0, "fortran_initial": 0, "explained": [], "unexplained": []}
    for key, dv in dump.items():
        sv = rec_res[key]
        if sv is None:
            selfcheck["fortran_initial"] += 1
            continue
        ok, note = same_list(sv, dv)
        name = f"{key[0]}:{key[1]}.{key[2]}"
        if not ok:
            selfcheck["unexplained"].append(f"{name}: {note}")
        elif note:
            selfcheck["explained"].append(f"{name}: {note}")
        else:
            selfcheck["equal"] += 1
    require(not selfcheck["unexplained"],
            "record namelists do not reproduce NEMO's dump: "
            + "; ".join(selfcheck["unexplained"][:10]))

    nemo: dict = {}
    differs_from_record = {}
    for key, dv in dump.items():
        ev, rv = exp_res[key], rec_res[key]
        if (ev is None) == (rv is None) and (ev is None or same_list(ev, rv)[0] and same_list(rv, ev)[0]):
            nemo[key] = dv
        else:
            nemo[key] = ev if ev is not None else dv
            differs_from_record[key] = True
            if ev is None:
                differs_from_record[key] = "EXP00 leaves it at namelist_ref/initial; record cfg sets it"
    oo_text = (record / "ocean.output").read_text(errors="replace")
    cpp = CPP_KEYS.read_text()
    if ctx is None:
        ctx = build_context(record, oo_text)
    rows = []
    for key in dump:
        status = classify(key, nemo, ctx, cpp)
        oo, oo_val = ocean_output_check(oo_text, key[2], dump[key])
        rows.append({
            "ns": key[0], "group": key[1], "key": key[2],
            "nemo": show(nemo[key]), "record": show(dump[key]),
            "record_differs_from_exp00": key in differs_from_record,
            "ocean_output": oo, "ocean_output_value": oo_val,
            **status,
        })
    counts = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    return {"rows": rows, "counts": counts, "selfcheck": selfcheck,
            "n_keys": len(rows),
            "record_vs_exp00_rows": sum(r["record_differs_from_exp00"] for r in rows),
            "ocean_output": {s: sum(r["ocean_output"] == s for r in rows)
                             for s in ("agree", "disagree", "absent", "n/a")}}


def table_markdown(report: dict, header: str) -> str:
    lines = [header, "",
             "| ns | group | key | NEMO (EXP00) | record | rec!=EXP00 | ocean.output | card | status | class | why |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        oo = r["ocean_output"] + (f" ({r['ocean_output_value']})" if r["ocean_output_value"] else "")
        cells = [r["ns"], r["group"], r["key"], r["nemo"], r["record"],
                 "Y" if r["record_differs_from_exp00"] else "", oo, r["card"],
                 r["status"], r["class"], r["why"]]
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", type=Path, default=RECORD)
    p.add_argument("--exp00", type=Path, default=EXP00)
    p.add_argument("--shared", type=Path, default=NEMO / "cfgs/SHARED")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--table-md", type=Path, default=None)
    a = p.parse_args()
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    try:
        report = audit(a.record, a.exp00, a.shared)
    except AuditError as exc:
        print(f"AUDIT_FAIL: {exc}")
        return 1
    report["provenance"] = {"stamp": worktree_stamp(repo=REPO), "record": str(a.record),
                            "exp00": str(a.exp00), "shared": str(a.shared)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if a.table_md:
        stamp = report["provenance"]["stamp"]
        a.table_md.write_text(table_markdown(report, (
            "# ORCA2 rung 10 namelist audit table (round 314)\n\n"
            f"Generated by `scripts/validate/ocean_fidelity/orca2_l4/"
            f"nemo_testcase_l4_orca2_round314_namelist_audit.py` at {stamp.get('sha', stamp)}; "
            f"record `{a.record.name}`; counts {report['counts']}.")))
    print(json.dumps({k: report[k] for k in ("n_keys", "counts", "record_vs_exp00_rows",
                                             "ocean_output")}))
    print("selfcheck:", {k: (v if isinstance(v, int) else len(v))
                         for k, v in report["selfcheck"].items()})
    for r in report["rows"]:
        if r["status"] == "DIFFERS":
            print(f"DIFFERS {r['ns']}:{r['group']}.{r['key']} nemo={r['nemo']} "
                  f"card={r['card']} class={r['class']}")
    print("AUDIT_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
