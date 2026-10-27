"""Make a WRITE-ONLY instrumented copy of WRF's module_mp_thompson.F.

The copy differs from the shipped file by exactly two insertions: ``USE
thom_dump`` in the module header, and, immediately before mp_thompson's
final "apply and pass back" loop, copies of its ten tendency arrays into
``thom_dump::dmp(ii, k, :)``.  Nothing is read back, so the scheme's
results cannot change; make_fixture.py checks that the dump build's final
state is bitwise identical to the unpatched build on every column.

usage: python patch_dump.py <WRF>/phys/module_mp_thompson.F <out.F>
"""
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
use = "      USE module_mp_radar\n"
anchor = "!.. All tendencies computed, apply and pass back final values to parent.\n"
assert s.count(use) == 1 and s.count(anchor) == 1
dump = ["      if (allocated(dmp)) then\n"]
for j, v in enumerate(("tten", "qvten", "qcten", "qiten", "qrten", "qsten",
                       "qgten", "niten", "nrten", "ncten"), start=1):
    dump.append(f"         dmp(ii,kts:kte,{j}) = {v}(kts:kte)\n")
dump.append("      endif\n")
i = s.index(anchor)
i = s.rindex("!+---+", 0, i)          # insert before the banner line
s = s[:i] + "".join(dump) + s[i:]
s = s.replace(use, use + "      USE thom_dump\n")
open(out, "w").write(s)
print("PATCH_OK", out)
