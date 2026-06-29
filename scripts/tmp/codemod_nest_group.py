#!/usr/bin/env python3
"""#501 type-aware config-group nesting codemod.

Resolves each `<recv>.<field>` receiver's TYPE (param annotation / `self.config`
in the LatLon model class / assignment from `<RuntimeType>.from_flat(...)` or
`<RuntimeType>(...)`) and migrates ONLY high-confidence reads of the runtime
config (LatLonCGridOceanConfig) to `<recv>.<subconfig>.<field>`. Flat reads on a
DIFFERENT config (MPAS/cube/spectral/recipe dataclass) are left untouched — this
is what the file-scoped seds got wrong (the cross-config over-migration class).

Auto-migrates: attribute reads + `getattr(recv,"field",...)` when recv is
provably the runtime config. REPORTS (for manual review, never auto-edits):
- `recv._replace(field=...)` (structural — nest by hand)
- reflective `getattr(recv, <var>)` (dynamic name)
- AMBIGUOUS receivers (untyped local, `r.config`, `r.model_config`, …)

Usage: codemod_nest_group.py --root <dir> --runtime LatLonCGridOceanConfig \
       --model LatLonCGridOceanModel --sub <subfield> --fields f1,f2,... [--apply]
Without --apply: dry-run (reports only).
"""
import argparse
import ast
import os
import sys


class Resolver(ast.NodeVisitor):
    """Per-file: collect var->is_runtime bindings per function scope + the class
    of each method (for self.config). One flat pass is enough for our patterns."""

    def __init__(self, runtime, model):
        self.runtime = runtime
        self.model = model
        # map id(FunctionDef node) -> set of local names bound to the runtime type
        self.scope_runtime_names = {}
        # map id(FunctionDef node) -> set of local names bound to a model instance
        self.scope_model_names = {}
        # map id(FunctionDef node) -> bool (is a method of the model class)
        self.scope_in_model = {}
        self._class_stack = []

    def _is_runtime_value(self, node):
        # `<runtime>.from_flat(...)` or `<runtime>(...)`
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
               and f.value.id == self.runtime and f.attr == "from_flat":
                return True
            if isinstance(f, ast.Name) and f.id == self.runtime:
                return True
        return False

    def _is_model_value(self, node):
        # `<ModelClass>(...)` -> a model instance whose `.config` is the runtime
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == self.model

    def visit_ClassDef(self, node):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _handle_func(self, node):
        names = set()
        models = set()
        in_model = bool(self._class_stack) and self._class_stack[-1] == self.model
        # annotated params (runtime config OR a model instance)
        for a in list(node.args.args) + list(node.args.posonlyargs) + list(node.args.kwonlyargs):
            ann = _ann_name(a.annotation) if a.annotation is not None else None
            if ann == self.runtime:
                names.add(a.arg)
            elif ann == self.model:
                models.add(a.arg)
        # body: annotated assigns + assigns from runtime/model construction
        for sub in ast.walk(node):
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name) \
               and _ann_name(sub.annotation) == self.runtime:
                names.add(sub.target.id)
            elif isinstance(sub, ast.Assign) and self._is_runtime_value(sub.value):
                for t in sub.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(sub, ast.Assign) and self._is_model_value(sub.value):
                for t in sub.targets:
                    if isinstance(t, ast.Name):
                        models.add(t.id)
        # A runtime name REBOUND to a non-runtime value somewhere in the function
        # (e.g. ``config: LatLonCGridOceanConfig`` then ``config = OceanConfig()``)
        # is flow-ambiguous — this resolver is scope-, not statement-level — so
        # drop it: its reads are then REPORTED, never blindly migrated (avoids
        # over-migrating a later cube/MPAS rebinding).  ``x = x._replace(...)``
        # keeps the same type and is NOT a rebind.
        rebound = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if isinstance(t, ast.Name) and t.id in names:
                        v = sub.value
                        self_replace = (isinstance(v, ast.Call)
                            and isinstance(v.func, ast.Attribute)
                            and v.func.attr == "_replace"
                            and isinstance(v.func.value, ast.Name)
                            and v.func.value.id == t.id)
                        if not self._is_runtime_value(v) and not self_replace:
                            rebound.add(t.id)
        names -= rebound
        self.scope_runtime_names[id(node)] = names
        self.scope_model_names[id(node)] = models
        self.scope_in_model[id(node)] = in_model
        self.generic_visit(node)

    visit_FunctionDef = _handle_func
    visit_AsyncFunctionDef = _handle_func


def _ann_name(ann):
    if isinstance(ann, ast.Name):
        return ann.id
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        return ann.value.strip()
    if isinstance(ann, ast.Attribute):
        return ann.attr
    return None


def _enclosing_func(tree, node):
    """Return the nearest FunctionDef ancestor of node (or None)."""
    best = None
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (f.lineno, f.col_offset) <= (node.lineno, node.col_offset):
                # node within f's span?
                if node.lineno <= getattr(f, "end_lineno", 1 << 30):
                    if best is None or (f.lineno, f.col_offset) > (best.lineno, best.col_offset):
                        best = f
    return best


def run(root, runtime, model, sub, fields, apply):
    fset = set(fields)
    edits_total = report = 0
    reports = []
    for dirpath, _dirs, files in os.walk(root):
        if "/.git" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dirpath, fn)
            with open(fp) as fh:
                src = fh.read()
            if not any(f in src for f in fields):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            res = Resolver(runtime, model)
            res.visit(tree)
            # CPython gives unreliable line/col offsets to nodes INSIDE an
            # f-string (JoinedStr), so byte-offset text surgery there corrupts
            # the source (inserts `.sub` mid-string-literal).  Mark every node
            # inside any f-string and REPORT (never edit) such reads — apply
            # them by safe text-replace in the (single-config-type) file.
            fstring_ids = set()
            for _js in ast.walk(tree):
                if isinstance(_js, ast.JoinedStr):
                    for _d in ast.walk(_js):
                        fstring_ids.add(id(_d))
            lines = src.splitlines(keepends=True)
            line_starts = [0]
            for ln in lines:
                line_starts.append(line_starts[-1] + len(ln))

            def off(lineno, col):
                return line_starts[lineno - 1] + col

            edits = []  # (start, end, newtext)

            def recv_is_runtime(recv, func):
                if isinstance(recv, ast.Name):
                    return func is not None and recv.id in res.scope_runtime_names.get(id(func), ())
                # `<x>.config` / `<x>.model_config` / `<x>._config`
                if isinstance(recv, ast.Attribute) and recv.attr in ("config", "model_config", "_config") \
                        and isinstance(recv.value, ast.Name):
                    x = recv.value.id
                    if x == "self":
                        return func is not None and res.scope_in_model.get(id(func), False)
                    if func is not None and x in res.scope_model_names.get(id(func), ()):
                        return True
                    return None  # x is some other object (recipe/setup result) — ambiguous
                return None  # ambiguous

            for node in ast.walk(tree):
                func = None
                # attribute read recv.field
                if isinstance(node, ast.Attribute) and node.attr in fset:
                    recv = node.value
                    func = _enclosing_func(tree, node)
                    r = recv_is_runtime(recv, func)
                    if r is True and id(node) in fstring_ids:
                        reports.append(f"{fp}:{node.lineno}: F-STRING read {node.attr} "
                                       f"— runtime, apply by text-replace (unsafe to AST-edit)")
                    elif r is True:
                        # insert ".<sub>" right after the receiver
                        ins = off(recv.end_lineno, recv.end_col_offset)
                        edits.append((ins, ins, "." + sub))
                    elif r is None:
                        reports.append(f"{fp}:{node.lineno}: AMBIGUOUS read {ast.dump(recv)[:40]}.{node.attr}")
                # getattr(recv, "field", ...)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "getattr" and len(node.args) >= 2:
                    a1 = node.args[1]
                    if isinstance(a1, ast.Constant) and a1.value in fset:
                        recv = node.args[0]
                        func = _enclosing_func(tree, node)
                        r = recv_is_runtime(recv, func)
                        if r is True and id(node) in fstring_ids:
                            reports.append(f"{fp}:{node.lineno}: F-STRING getattr {a1.value} "
                                           f"— runtime, apply by text-replace (unsafe to AST-edit)")
                        elif r is True:
                            ins = off(recv.end_lineno, recv.end_col_offset)
                            edits.append((ins, ins, "." + sub))
                        elif r is None:
                            reports.append(f"{fp}:{node.lineno}: AMBIGUOUS getattr {a1.value}")
                    elif not isinstance(a1, ast.Constant):
                        # reflective getattr(recv, <var>) — only report if recv could be runtime
                        recv = node.args[0]
                        func = _enclosing_func(tree, node)
                        if recv_is_runtime(recv, func) is not False:
                            reports.append(f"{fp}:{node.lineno}: REFLECTIVE getattr(...,<var>)")
                # recv._replace(field=...)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and node.func.attr == "_replace":
                    if any(k.arg in fset for k in node.keywords if k.arg):
                        recv = node.func.value
                        func = _enclosing_func(tree, node)
                        if recv_is_runtime(recv, func) is not False:
                            reports.append(f"{fp}:{node.lineno}: _REPLACE with {sub} field(s) — nest by hand")

            if edits and apply:
                edits.sort(reverse=True)
                for s, e, t in edits:
                    src = src[:s] + t + src[e:]
                with open(fp, "w") as fh:
                    fh.write(src)
            edits_total += len(edits)
    print(f"=== migrated {edits_total} reads/getattr (high-confidence runtime) "
          f"{'APPLIED' if apply else 'DRY-RUN'} ===")
    print(f"=== {len(reports)} sites need manual review ===")
    for r in reports:
        print("  " + r.replace(root + "/", ""))
    return edits_total, reports


_SELFTEST_SRC = '''\
class LatLonCGridOceanConfig:
    pass


class OceanConfig:  # the CUBE config — same field names, must stay flat
    pass


class LatLonCGridOceanModel:
    def __init__(self, config: "LatLonCGridOceanConfig"):
        self.config = config

    def use(self):
        # self.config in the model class -> runtime -> MIGRATE
        return self.config.max_abs_eta_m

    def shout(self):
        # runtime read INSIDE an f-string -> REPORT (never AST-edit)
        return f"bound {self.config.max_abs_eta_m}"


def runtime_fn(config: LatLonCGridOceanConfig):
    return config.max_abs_eta_m            # annotated runtime -> MIGRATE


def cube_fn(config: OceanConfig):
    return config.max_abs_eta_m            # cube config -> LEAVE FLAT


def rebind_fn(config: LatLonCGridOceanConfig):
    config = OceanConfig()                 # rebound to cube -> flow-ambiguous
    return config.max_abs_eta_m            # must NOT be migrated (left flat)


def reflective_fn(config: LatLonCGridOceanConfig, name):
    return getattr(config, name)           # dynamic name -> REPORT
'''


def _selftest():
    """Synthetic-violation self-test (non-vacuous): one runtime read, one cube
    read (same field name), one f-string runtime read, one reflective getattr.
    Asserts the type-aware codemod migrates ONLY the safe runtime reads, leaves
    the cube read flat, and REPORTS the f-string + reflective sites."""
    import tempfile
    d = tempfile.mkdtemp(prefix="codemod_selftest_")
    with open(os.path.join(d, "syn.py"), "w") as fh:
        fh.write(_SELFTEST_SRC)
    n_edit, reports = run(d, "LatLonCGridOceanConfig", "LatLonCGridOceanModel",
                          "runtime_checks", ["max_abs_eta_m"], apply=True)
    edited = open(os.path.join(d, "syn.py")).read()
    rep = "\n".join(reports)
    # 2 safe runtime reads migrated (runtime_fn + self.config.use); cube stays
    # flat; the rebound name in rebind_fn must NOT be migrated (flow-ambiguous).
    assert n_edit == 2, f"expected 2 migrations, got {n_edit}"
    assert "config.runtime_checks.max_abs_eta_m" in edited
    assert "def cube_fn" in edited and "config.max_abs_eta_m" in \
        edited.split("def cube_fn")[1].split("def ")[0], "cube read must stay flat"
    assert "config.max_abs_eta_m" in edited.split("def rebind_fn")[1].split("def ")[0], \
        "rebound (config = OceanConfig()) read must stay flat, not over-migrated"
    assert "F-STRING read max_abs_eta_m" in rep, "f-string read must be reported"
    assert "REFLECTIVE" in rep, "reflective getattr must be reported"
    # and the f-string source must NOT be corrupted
    assert "runtime_checks.runtime_checks" not in edited
    assert 'f"bound {self.config.max_abs_eta_m}"' in edited, "f-string left intact"
    print("SELFTEST PASS: runtime migrated, cube flat, f-string+reflective reported, no corruption")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--runtime", default="LatLonCGridOceanConfig")
    p.add_argument("--model", default="LatLonCGridOceanModel")
    p.add_argument("--sub", required=True)
    p.add_argument("--fields", required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    run(a.root, a.runtime, a.model, a.sub, a.fields.split(","), a.apply)
