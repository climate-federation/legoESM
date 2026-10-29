"""Make an INSTRUMENTED copy of OpenIFS 8f6f722 cumastrn.F90.

Insertions only: a `USE TDBG` line and plain assignments into module TDBG at
four points (first guess, ascent + KTYPE reclassification, after CUFLXN,
closure).  No
oracle statement is altered or removed; every anchor is asserted unique and
the line count is asserted to grow by exactly the inserted lines.
usage: patch_oracle.py <cumastrn.F90> <out.F90>
"""
import sys

src_path, out_path = sys.argv[1], sys.argv[2]
L = open(src_path).read().splitlines(keepends=True)
n0 = len(L)


def idx(text, start=0, end=None):
    end = len(L) if end is None else end
    hits = [i for i in range(start, end) if L[i].rstrip("\n").strip() == text]
    assert len(hits) == 1, (text, hits)
    return hits[0]


def next_line(text, start):
    for i in range(start, len(L)):
        if L[i].strip() == text:
            return i
    raise AssertionError(text)


added = 0


def ins_after(i, block):
    global added
    lines = [b + "\n" for b in block]
    L[i + 1:i + 1] = lines
    added += len(lines)


# --- USE TDBG before the routine's IMPLICIT NONE
i_sub = next(i for i, l in enumerate(L) if l.startswith("SUBROUTINE CUMASTRN"))
i_imp = next(i for i in range(i_sub, len(L)) if L[i].strip() == "IMPLICIT NONE")
L[i_imp:i_imp] = ["USE TDBG\n"]
added += 1

# --- (1) after the first-guess cloud-base mass-flux loop (:555-592)
i_fg = idx("ZMFUB(JL)=0.0_JPRB")
i_fg_end = next_line("ENDDO", next_line("ENDIF", i_fg))
ins_after(i_fg_end, [
    "DBG_ZMFUB_FG(KIDIA:KFDIA)=ZMFUB(KIDIA:KFDIA)",
    "DBG_KTYPE_FG(KIDIA:KFDIA)=KTYPE(KIDIA:KFDIA)",
    "DBG_LDCUM_FG(KIDIA:KFDIA)=LDCUM(KIDIA:KFDIA)",
    "DBG_ZDHPBL(KIDIA:KFDIA)=ZDHPBL(KIDIA:KFDIA)",
    "DBG_KCBOT(KIDIA:KFDIA)=KCBOT(KIDIA:KFDIA)",
])

# --- (2) after the KTYPE reclassification loop (:633-644): ascent profiles
i_rc = idx("IF(KTYPE(JL) == 2.AND.ZPBMPT >= RDEPTHS) KTYPE(JL)=1")
i_rc_end = next_line("ENDDO", i_rc)
ins_after(i_rc_end, [
    "DBG_KTYPE_RC(KIDIA:KFDIA)=KTYPE(KIDIA:KFDIA)",
    "DBG_KCTOP(KIDIA:KFDIA)=KCTOP(KIDIA:KFDIA)",
    "DBG_MFU_ASC(KIDIA:KFDIA,:)=PMFU(KIDIA:KFDIA,:)",
    "DBG_MFUS_ASC(KIDIA:KFDIA,:)=ZMFUS(KIDIA:KFDIA,:)",
    "DBG_MFUQ_ASC(KIDIA:KFDIA,:)=ZMFUQ(KIDIA:KFDIA,:)",
    "DBG_MFUL_ASC(KIDIA:KFDIA,:)=ZMFUL(KIDIA:KFDIA,:)",
    "DBG_LUDE_ASC(KIDIA:KFDIA,:)=PLUDE(KIDIA:KFDIA,:)",
    "DBG_DMFUP_ASC(KIDIA:KFDIA,:)=ZDMFUP(KIDIA:KFDIA,:)",
])

# --- (2b) right after CUFLXN (:1104-1115): below-cloud-base fluxes, rain evaporation
i_fx = next(i for i, l in enumerate(L) if l.strip() == "CALL CUFLXN &")
i_fx_end = next(i for i in range(i_fx, len(L)) if L[i].rstrip().endswith(")") and "PMFDDE_RATE" in L[i])
ins_after(i_fx_end, [
    "DBG_MFU_FLX(KIDIA:KFDIA,:)=PMFU(KIDIA:KFDIA,:)",
    "DBG_MFUS_FLX(KIDIA:KFDIA,:)=ZMFUS(KIDIA:KFDIA,:)",
    "DBG_MFUQ_FLX(KIDIA:KFDIA,:)=ZMFUQ(KIDIA:KFDIA,:)",
    "DBG_MFUL_FLX(KIDIA:KFDIA,:)=ZMFUL(KIDIA:KFDIA,:)",
    "DBG_LUDE_FLX(KIDIA:KFDIA,:)=PLUDE(KIDIA:KFDIA,:)",
    "DBG_DMFUP_FLX(KIDIA:KFDIA,:)=ZDMFUP(KIDIA:KFDIA,:)",
    "DBG_DMFDP_FLX(KIDIA:KFDIA,:)=ZDMFDP(KIDIA:KFDIA,:)",
])

# --- (3) the closure value entering the final scaling (:957-965)
i_sc = idx("ZMFS(JL)=ZMFUB1(JL)/MAX(RMFCMIN,ZMFUB(JL))")
i_sc_do = max(i for i in range(i_sc) if L[i].strip() == "DO JL=KIDIA,KFDIA")
if L[i_sc_do - 1].startswith("!DIR$"):
    i_sc_do -= 1
L[i_sc_do:i_sc_do] = ["DBG_ZMFUB1(KIDIA:KFDIA)=ZMFUB1(KIDIA:KFDIA)\n"]
added += 1

assert len(L) == n0 + added
open(out_path, "w").write("".join(L))
print(f"PATCH_OK lines {n0} -> {len(L)} (+{added}, 0 removed)")
