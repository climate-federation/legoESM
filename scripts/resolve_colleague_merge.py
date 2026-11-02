"""Resolve the diff3 conflicts from merging the colleague's physics (b309178, 2026-10-14) into our tree.
Usage: python resolve_colleague_merge.py <file.merged> <output> <run_omip|model|jra55>"""
import sys

def parse(lines):
    out, i = [], 0          # items: str line or (ours, base, theirs)
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            o, b, t, part = [], [], [], "o"; i += 1
            while not lines[i].startswith(">>>>>>> "):
                if lines[i].startswith("||||||| "): part = "b"
                elif lines[i].startswith("======="): part = "t"
                else: {"o": o, "b": b, "t": t}[part].append(lines[i])
                i += 1
            out.append((o, b, t))
        else:
            out.append(lines[i])
        i += 1
    return out

ours = lambda o, b, t: o
theirs = lambda o, b, t: t

def ind(block):
    s = next(l for l in block if l.strip()); return s[:len(s) - len(s.lstrip())]

def heat_switch(o, b, t):
    p = ind(o); sh = lambda L: ["    " + l if l.strip() else l for l in L]
    return ([p + '# merged: LEGOESM_HEATFLUX=ours -> non-solar flux only (SW via penetration); default colleague (q_net)\n',
             p + 'if __import__("os").environ.get("LEGOESM_HEATFLUX", "colleague") == "ours":\n'] + sh(o) +
            [p + 'else:\n'] + sh(t))

def bgc_adv(o, b, t):
    old = 'if _bio is not None and not os.environ.get("LEGOESM_BGC_NO_ADV"):'
    new = 'if _bio is not None and not _is_marbl_tracers(_bio) and not os.environ.get("LEGOESM_BGC_NO_ADV"):'
    t2 = [l.replace(old, new) for l in t]
    assert any(new in l for l in t2), "guard line not found"
    return t2 + o          # colleague block (inner indent) first, then ours (outer indent)

RULES = {"run_omip": [theirs, heat_switch, heat_switch, theirs, theirs],
         "model": [bgc_adv],
         "jra55": [ours, ours, ours]}

src, dst, kind = sys.argv[1:4]
items = parse(open(src).read().splitlines(keepends=True))
conf = [x for x in items if isinstance(x, tuple)]
assert len(conf) == len(RULES[kind]), f"{src}: {len(conf)} conflicts, expected {len(RULES[kind])}"
rules = iter(RULES[kind]); res = []
for x in items:
    res.extend(next(rules)(*x) if isinstance(x, tuple) else [x])
txt = "".join(res)
assert "<<<<<<< " not in txt and ">>>>>>> " not in txt
compile(txt, dst, "exec")
open(dst, "w").write(txt)
print(f"{dst}: {len(conf)} conflicts resolved, {len(txt)} bytes")
