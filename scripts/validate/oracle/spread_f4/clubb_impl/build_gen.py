"""Build gen_spread_impl.py = committed CLUBB generator (wt_f4 main) + the reviewed
SPREAD adaptation of spread/clubb_core/gen_spread.py (spy, perturbed copies of
each recorded step call, C3 readback), sharing the committed step()'s
l_implemented-aware call."""
r2 = open('/burg/glab/users/pg2328/oracle_audit/spread/clubb_core/gen_spread.py').read()
g = open('/burg/glab/users/pg2328/wt_f4/scripts/validate/oracle/clubb_core/gen_clubb_core_fixture.py').read()
a = r2.index("# ---- SPREAD adaptation"); b = r2.index("\ndef main(")
block = r2[a:b].rstrip() + "\n\n"
s = r2.index("    import clubb_f2py\n")
e = r2.index("    clubb_f2py.f2py_advance_clubb_core = _spy\n") + len("    clubb_f2py.f2py_advance_clubb_core = _spy\n")
spy = r2[s:e]
s2 = r2.index("    def spread_step(name, st, n):")
e2 = r2.index("\n\n", r2.index('res[f"{name}/seed{sd}/s{n}/{k}"]', s2))
spread = r2[s2:e2] + "\n\n"
assert spread.count("A._advance_clubb_core_api(d)") == 1
spread = spread.replace("A._advance_clubb_core_api(d)", "_advance(d)")
g = g.replace("\ndef main(", "\n" + block + "\ndef main(", 1)
anchor = "        init_clubb_case, read_namelist)\n"
assert g.count(anchor) == 1
g = g.replace(anchor, anchor + "\n" + spy, 1)
old_step = '''        if IMPLEMENTED:
            # _advance_clubb_core_api hard-codes l_implemented=False (the SCM
            # driver); CAM passes .true.
            orig = clubb_api.advance_clubb_core
            clubb_api.advance_clubb_core = (
                lambda *a, **k: orig(*a, **dict(k, l_implemented=True)))
            try:
                A._advance_clubb_core_api(st)
            finally:
                clubb_api.advance_clubb_core = orig
        else:
            A._advance_clubb_core_api(st)
'''
assert old_step in g
adv = "    def _advance(st):   # the committed step()'s core call, shared with spread_step\n" + \
    old_step + "\n"
g = g.replace(old_step, "        _advance(st)\n", 1)
g = g.replace("    def step(st):\n", adv + "    def step(st):\n", 1)
g = g.replace("    def trajectory(name, st):\n", spread + "    def trajectory(name, st):\n", 1)
old_loop = '''        for n in range(1, N_STEPS + 1):
            step(st)
            record(name, st, n)'''
assert g.count(old_loop) == 1
g = g.replace(old_loop, '''        for n in range(1, N_STEPS + 1):
            if SPREAD_MODE == "pert":
                spread_step(name, st, n)
            step(st)
            record(name, st, n)''', 1)
open('gen_spread_impl.py', 'w').write(g)
print("built")
