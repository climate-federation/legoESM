"""Which NEMO TIME LEVEL each oracle dump belongs to (#1226).

NEMO's modified-leapfrog step carries three time levels — ``Nbb`` (before),
``Nnn`` (now), ``Naa`` (after) — and a routine is frequently called with T/S at
ONE level and geometry at ANOTHER::

    CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nnn )   ! stpmlf.F90:184
    CALL eos_rab( ts(:,:,:,:,Nnn), rab_n, Nnn )   ! stpmlf.F90:185

Comparing a dump against the WRONG level silently substitutes
``|T_now - T_before|`` for "error".  That difference is largest in the
thermocline, so it does not look like noise — it looks like a beautifully
depth-structured defect.  It cost this campaign a full false diagnosis: a
7074-cell "tail", three 1.8% alpha "outliers" and a "levels 6-8 structure",
all of which evaporated at the correct level.

The mistake is easy precisely because the state LOOKS right — same field, same
shape, same units, plausible magnitude.  So the correct level is recorded here
ONCE, next to its NEMO source line, and :func:`select_ts` makes picking it the
default action rather than a thing to remember.

HARNESS glue: it selects comparison inputs and changes no answer.
"""
from __future__ import annotations

__all__ = ["TimeLevel", "time_level_for_dump", "select_ts", "register_dump"]

TimeLevel = str  # "before" | "now" | "after"

_VALID: frozenset[str] = frozenset({"before", "now", "after"})

# dump basename -> (time level of its T/S, NEMO source line that proves it).
# The GEOMETRY level is separate and usually Nnn — see the module docstring.
_DUMP_TIME_LEVEL: dict[str, tuple[TimeLevel, str]] = {
    # eos_rab / bn2 family: T/S at Nbb, geometry at Nnn.
    "dump_alpha_b.bin": ("before", "stpmlf.F90:184 eos_rab(ts(...,Nbb), rab_b, Nnn)"),
    "dump_beta_b.bin": ("before", "stpmlf.F90:184 eos_rab(ts(...,Nbb), rab_b, Nnn)"),
    "tke_dump_rn2b.bin": ("before", "rn2b = bn2(ts(...,Nbb)); zdfmxl.F90:98 integrates rn2b"),
    "eiv_dump_rn2b.bin": ("before", "same rn2b; NOTE only 35 levels — missing the deepest interface"),
    # zdf_mxl outputs are computed FROM rn2b, hence also before-level inputs.
    "dump_nmln.bin": ("before", "zdfmxl.F90:96-101, integrand is rn2b (before)"),
    "dump_hmlp.bin": ("before", "zdfmxl.F90:104, gdepw(nmln) from the rn2b integral"),
    # ldf_slp slopes are built on the before-level density field.
    "eiv_dump_uslp.bin": ("before", "ldfslp.F90 uses rn2b/rab_b"),
    "eiv_dump_vslp.bin": ("before", "ldfslp.F90 uses rn2b/rab_b"),
    "eiv_dump_wslpi.bin": ("before", "ldfslp.F90 uses rn2b/rab_b"),
    "eiv_dump_wslpj.bin": ("before", "ldfslp.F90 uses rn2b/rab_b"),
    # Asselin filter dumps are explicit about their own level.
    "atf_dump_tem_before.bin": ("before", "traatf_qco.F90, pre-filter state"),
    "atf_dump_sal_before.bin": ("before", "traatf_qco.F90, pre-filter state"),
    "atf_dump_ssh_before.bin": ("before", "dynatf_qco.F90, pre-filter state"),
    "atf_dump_uu_before.bin": ("before", "dynatf_qco.F90, pre-filter state"),
    "atf_dump_vv_before.bin": ("before", "dynatf_qco.F90, pre-filter state"),
    "atf_dump_tem_after.bin": ("after", "traatf_qco.F90, post-filter state"),
    "atf_dump_sal_after.bin": ("after", "traatf_qco.F90, post-filter state"),
    "atf_dump_ssh_after.bin": ("after", "dynatf_qco.F90, post-filter state"),
    "atf_dump_uu_after.bin": ("after", "dynatf_qco.F90, post-filter state"),
    "atf_dump_vv_after.bin": ("after", "dynatf_qco.F90, post-filter state"),
}


def register_dump(basename: str, level: TimeLevel, source: str) -> None:
    """Register a dump's time level, CITING the NEMO line that proves it.

    ``source`` is mandatory and must be non-empty: an unsourced entry is a
    guess, and a guess here is exactly the failure this module exists to stop.
    """
    if level not in _VALID:
        raise ValueError(
            f"unknown time level {level!r} for {basename!r}; "
            f"expected one of {sorted(_VALID)}")
    if not source.strip():
        raise ValueError(
            f"register_dump({basename!r}) needs a NEMO source citation "
            "(file:line showing which time level feeds it) — an unsourced "
            "entry is a guess")
    _DUMP_TIME_LEVEL[basename] = (level, source)


def time_level_for_dump(basename: str) -> TimeLevel:
    """Time level of ``basename``'s T/S; RAISES on an unregistered dump.

    Deliberately fail-CLOSED (dispatch hardening): defaulting an unknown dump
    to "now" is what produced the false diagnosis in the first place, and a
    silent default would reintroduce it for every dump added later.
    """
    basename = basename.rsplit("/", 1)[-1]
    try:
        return _DUMP_TIME_LEVEL[basename][0]
    except KeyError:
        raise ValueError(
            f"dump {basename!r} has no registered NEMO time level. Read the "
            "call site in the NEMO source, then register_dump(name, level, "
            "source) with the file:line. Do NOT assume 'now' — many routines "
            "run on Nbb T/S with Nnn geometry (stpmlf.F90:184), and comparing "
            "against the wrong level substitutes |T_now - T_before| for error."
        ) from None


def select_ts(basename: str, *, now, before, after=None):
    """Return the ``(T, S)`` pair that matches ``basename``'s time level.

    Make the right thing the default action: pass all the levels you loaded and
    let the registry choose, instead of picking one by hand at each call site.

    ``now``/``before``/``after`` are ``(T, S)`` tuples.
    """
    level = time_level_for_dump(basename)
    chosen = {"now": now, "before": before, "after": after}[level]
    if chosen is None:
        raise ValueError(
            f"dump {basename!r} needs the {level!r} time level, but no "
            f"{level!r} state was supplied to select_ts()")
    return chosen
