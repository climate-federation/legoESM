"""HLO collective census of the compiled SPMD window step (C48 kt=2 on 24
virtual CPU devices): every collective op, its shape, and the source line it
lowers from.  Run via scripts/cluster/fv3_native/tiled_m6_hlo_census.sbatch.

Expected after the 2026-09-05 C-pressure fix: exactly ONE all-reduce (the
f64[km] Courant max) and no whole-window collective-permutes; a whole-window
permute here means some phase is still slicing the sharded window axis per
face (a loop arm running inside the batched step)."""
import re, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh
from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import FV3DuoConfig, FV3DuoDynamicsModel
from legoesm.grids.factory import create_fv3_duo_grid
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
kt = int(sys.argv[2]) if len(sys.argv) > 2 else 2
grid = create_fv3_duo_grid(N)
mesh = Mesh(np.array(jax.devices()[:6 * kt * kt]).reshape(6, kt, kt), ("face", "tile_i", "tile_j"))
m = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=10, hydrostatic=True, n_split=3), step_spmd_mesh=mesh, step_windows=(kt, 11))
if len(sys.argv) > 3 and sys.argv[3] == "pack":
    m._window_comm.pack_pad_refresh = True
    print("M8-A packed pad refresh ON")
m._window_comm.firing_log = []
st = m.dcmip16_initial_state()
lay_W = m.window_layout.W
fn = m._step_fn
# the jitted callable is wrapped; find the underlying pjit'd function
inner = None
for cell in (fn.__closure__ or ()):
    if hasattr(cell.cell_contents, "lower"):
        inner = cell.cell_contents
        break
if inner is None:
    print("no jitted callable in the step closure"); sys.exit(1)
try:
    lowered = inner.lower(st["state"], st["press"], st["q"], 900.0, st["omga"], st["nh"])
except Exception as e:
    print("lower failed on", type(inner), e); sys.exit(1)
txt = lowered.compile().as_text()
ops = {}
for line in txt.splitlines():
    mm = re.search(r"= (\S+)\[.*?\] (all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all)\(", line)
    mm2 = re.search(r"(all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all)(-start|-done)?\(", line)
    if mm2:
        kind = mm2.group(1)
        shape = re.search(r"^\s*\S+ = (\S+?)\s", line)
        key = (kind, shape.group(1) if shape else "?")
        ops[key] = ops.get(key, 0) + 1
for (k, s), n in sorted(ops.items(), key=lambda kv: -kv[1]):
    print(f"{n:6d}  {k:20s} {s}")
print("total collectives:", sum(ops.values()))

print("\n== all-reduce sites (metadata) ==")
for line in txt.splitlines():
    if re.search(r"all-reduce(-start)?\(", line):
        md = re.search(r"metadata=\{[^}]*\}", line)
        shp = re.search(r"^\s*\S+ = (\S+?)\s", line)
        print(" ", shp.group(1)[:60] if shp else "?", md.group(0)[:600] if md else "(no metadata)")
print("\n== full-window collective-permutes (metadata, first 6) ==")
k = 0
for line in txt.splitlines():
    if "collective-permute" in line and f"[1,{lay_W},{lay_W},10]" in line:
        md = re.search(r"metadata=\{[^}]*\}", line)
        pairs = re.search(r"source_target_pairs=\{[^}]*\}", line)
        print(" ", (pairs.group(0)[:80] if pairs else "?"), (md.group(0)[:600] if md else "(no metadata)"))
        k += 1
        if k >= 6:
            break

print("\n== distinct source lines of ALL full-window permutes ==")
sites = {}
for line in txt.splitlines():
    if "collective-permute" in line and ("[1,49,49,10]" in line or "[1,49,50,10]" in line or "[1,50,49,10]" in line):
        md = re.search(r'source_file="([^"]*)" source_line=(\d+)', line)
        key = (md.group(1).split("/")[-1], md.group(2)) if md else ("?", "?")
        sites[key] = sites.get(key, 0) + 1
for k, v in sorted(sites.items(), key=lambda kv: -kv[1]):
    print(f"  {v:5d}  {k}")

print("\n== UNOPTIMIZED HLO: squeeze / slice ops on window-sized arrays, with source lines ==")
raw = lowered.as_text(dialect="hlo")
sites = {}
for line in raw.splitlines():
    if not re.search(r"= f64\[\d+,49,(49|50),10\]\{[^}]*\} (squeeze|slice|dynamic-slice|gather)\(", line):
        continue
    if "f64[24,49" in line.split("(")[1] if "(" in line else False:
        pass
    op = re.search(r"\} (squeeze|slice|dynamic-slice|gather)\(", line).group(1)
    md = re.search(r'source_file="([^"]*)" source_line=(\d+)', line)
    key = (op, md.group(1).split("/")[-1] if md else "?", md.group(2) if md else "?")
    sites[key] = sites.get(key, 0) + 1
for k, v in sorted(sites.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {v:5d}  {k}")

print("\n== STABLEHLO: slices along the WINDOW axis of window-sized arrays, by source location ==")
sh = lowered.as_text()
locdefs = {}
for line in sh.splitlines():
    m2 = re.match(r'#(loc\d+) = loc\((.*)\)', line.strip())
    if m2:
        locdefs[m2.group(1)] = m2.group(2)[:140]
def resolve(l, depth=0):
    d = locdefs.get(l, "")
    inner = re.findall(r'#(loc\d+)', d)
    if inner and depth < 6:
        return resolve(inner[0], depth + 1) if '"' not in d else d
    return d
sites = {}
for line in sh.splitlines():
    if "stablehlo.slice" not in line and "stablehlo.dynamic_slice" not in line:
        continue
    if not re.search(r"tensor<24x49x(49|50)x10xf64>", line):
        continue
    m3 = re.search(r"tensor<(\d+)x49x(49|50)x10xf64>\s*$|-> tensor<(\d+)x49", line)
    lim = re.search(r"limit_indices = array<i64: (\d+),", line)
    start = re.search(r"start_indices = array<i64: (\d+),", line)
    if lim and start and int(lim.group(1)) - int(start.group(1)) == 24:
        continue                       # full window axis: not a batch-axis slice
    locid = re.search(r"loc\(#(loc\d+)\)", line)
    src = resolve(locid.group(1)) if locid else "?"
    sites[src] = sites.get(src, 0) + 1
for k, v in sorted(sites.items(), key=lambda kv: -kv[1])[:20]:
    print(f"  {v:5d}  {k}")
print("\n== STABLEHLO: squeeze/reshape producing 24x49x50x10 (the u shape) with source ==")
sites = {}
for line in sh.splitlines():
    if ("stablehlo.reshape" in line or "squeeze" in line) and re.search(r"-> tensor<24x49x50x10xf64>", line):
        locid = re.search(r"loc\(#(loc\d+)\)", line)
        src = resolve(locid.group(1)) if locid else "?"
        sites[src] = sites.get(src, 0) + 1
for k, v in sorted(sites.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {v:5d}  {k}")

print("\n== RAW samples ==")
k = 0
for line in sh.splitlines():
    if "stablehlo.slice" in line and re.search(r"tensor<24x49x(49|50)x10xf64>", line):
        print(line.strip()[:300]); k += 1
        if k >= 2: break
k = 0
for line in sh.splitlines():
    if line.startswith("#loc") and "callsite" not in line:
        print(line.strip()[:200]); k += 1
        if k >= 3: break
for line in sh.splitlines():
    if line.startswith("#loc") and "callsite" in line:
        print(line.strip()[:300]); break

print("\n== STABLEHLO(debug_info): batch-axis slices of window-sized arrays -> source ==")
try:
    shd = lowered.as_text(debug_info=True)
except TypeError:
    shd = lowered.as_text(dialect="stablehlo", debug_info=True)
locdefs = {}
for line in shd.splitlines():
    m2 = re.match(r'#(loc\d*) = loc\((.*)\)\s*$', line.strip())
    if m2:
        locdefs[m2.group(1)] = m2.group(2)
def resolve(l, depth=0):
    d = locdefs.get(l, "")
    if '"' in d and ":" in d and "callsite" not in d:
        return d[:160]
    inner = re.findall(r'#(loc\d*)', d)
    for i in inner:
        r = resolve(i, depth + 1) if depth < 8 else ""
        if r and "site-packages" not in r:
            return r
    return d[:160]
sites = {}
for line in shd.splitlines():
    m3 = re.search(r"stablehlo\.slice \S+ \[(\d+):(\d+), [^\]]*\] : \(tensor<24x(49|50)x(49|50)x10xf64>\)", line)
    if not m3:
        continue
    if int(m3.group(2)) - int(m3.group(1)) == 24:
        continue
    locid = re.search(r"loc\(#(loc\d*)\)", line)
    src = resolve(locid.group(1)) if locid else "?"
    sites[src] = sites.get(src, 0) + 1
for k, v in sorted(sites.items(), key=lambda kv: -kv[1])[:20]:
    print(f"  {v:5d}  {k}")
print("\n== reshapes producing 24x49x50x10 ==")
sites = {}
for line in shd.splitlines():
    if "stablehlo.reshape" in line and re.search(r"-> tensor<24x49x50x10xf64>", line):
        locid = re.search(r"loc\(#(loc\d*)\)", line)
        src = resolve(locid.group(1)) if locid else "?"
        sites[src] = sites.get(src, 0) + 1
for k, v in sorted(sites.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {v:5d}  {k}")

# constants by byte size (per-rank compile/runtime memory that scales with
# the GLOBAL grid lives here: a full (nb, W, W, ...) literal is stored on
# every rank; 2026-09-05 the 216-rank C192 row was OOM-killed at 24.6 GB)
def nbytes(shape):
    m = re.match(r"(\w+)\[([\d,]*)\]", shape)
    if not m:
        return 0
    dt = {"f64": 8, "f32": 4, "s64": 8, "s32": 4, "u64": 8, "u32": 4, "pred": 1,
          "s8": 1, "u8": 1}.get(m.group(1), 8)
    dims = [int(x) for x in m.group(2).split(",") if x]
    return dt * int(np.prod(dims)) if dims else dt
consts = {}
for line in txt.splitlines():
    mm = re.search(r"^\s*\S+ = (\w+\[[\d,]*\])\S* constant\(", line)
    if mm:
        shp = mm.group(1)
        consts[shp] = consts.get(shp, 0) + 1
tot = sum(nbytes(s_) * n for s_, n in consts.items())
print(f"\n== constants: {sum(consts.values())} literals, {tot / 2**20:.1f} MiB total ==")
for shp, n in sorted(consts.items(), key=lambda kv: -nbytes(kv[0]) * kv[1])[:15]:
    print(f"  {nbytes(shp) * n / 2**20:8.1f} MiB  x{n:5d}  {shp}")
import resource
print(f"rss after compile: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
print("DONE census")

# firing census (M8-B scoping): firings per step by (name, kind, per-level?)
from collections import Counter
log = m._window_comm.firing_log
per = Counter()
for name, kind, shapes in log:
    lvl = "per-level" if all(len(sh) == 2 or (len(sh) == 3 and sh[2] <= 1) for sh in shapes) else "k-stacked"
    per[(name, kind, lvl, len(shapes), shapes[0])] += 1
print(f"\n== firings traced: {len(log)} (includes IC-time firings, if any) ==")
for (name, kind, lvl, n, shp), c in sorted(per.items(), key=lambda kv: -kv[1]):
    print(f"  {c:5d}  {name:10s} {str(kind):6s} {lvl:9s} arrays={n} first={shp}")
print("per-level firings:", sum(c for (nm, k, l, n, sh), c in per.items() if l == "per-level"),
      " k-stacked:", sum(c for (nm, k, l, n, sh), c in per.items() if l == "k-stacked"))
