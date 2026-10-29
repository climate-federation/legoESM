"""Write the dump copy of module_bl_mynnedmf.F90: identical source plus
write-only calls (mynn_dump_mod) after the sub-steps of ``mynnedmf``.
usage: python patch_dump.py <module_bl_mynnedmf.F90> <out.F90>"""
import sys

# (anchor, occurrence, dumps) -- dumps go after the statement that starts with
# the anchor (continuation lines included).
HOOKS = [
    ("    call get_pblh(kts,kte,pblh,thv1,qke1,ust,zw1,dz1,xland,kpbl)", 1,
     ["dump0('pblh', pblh)", "dump0('kpbl', real(kpbl,kind_phys))"]),
    ("    call scale_aware(dx,pblh,psig_bl,psig_shcu)", 1,
     ["dump0('psig_bl', psig_bl)", "dump0('psig_shcu', psig_shcu)"]),
    ("    call mym_condensation (kts,kte,", 1,
     ["dump0('flt', flt)", "dump0('fltv', fltv)", "dump0('flq', flq)",
      "dump0('rmol', rmol)", "dump0('pmz', pmz)", "dump0('phh', phh)",
      "dump1('vt_cond', vt1)", "dump1('vq_cond', vq1)", "dump1('sgm', sgm1)",
      "dump1('cldfra_cond', cldfra_bl1)", "dump1('qc_bl_cond', qc_bl1)",
      "dump1('qi_bl_cond', qi_bl1)"]),
    ("    !> Options for downdrafts or analytical top-down method:", 1,
     ["dump1('s_aw', s_aw1)", "dump1('s_awthl', s_awthl1)", "dump1('s_awqt', s_awqt1)",
      "dump1('s_awqv', s_awqv1)", "dump1('s_awqc', s_awqc1)", "dump1('s_awu', s_awu1)",
      "dump1('s_awv', s_awv1)", "dump1('edmf_a_mf', edmf_a1)",
      "dump1('edmf_w_mf', edmf_w1)", "dump1('edmf_qt_mf', edmf_qt1)",
      "dump1('edmf_thl_mf', edmf_thl1)", "dump1('edmf_ent_mf', edmf_ent1)",
      "dump1('edmf_qc_mf', edmf_qc1)", "dump1('tkeprod_up', TKEprod_up)",
      "dump1('cldfra_mf', cldfra_bl1)", "dump1('qc_bl_mf', qc_bl1)",
      "dump1('vt_mf', vt1)", "dump1('vq_mf', vq1)", "dump0('maxmf', maxmf)",
      "dump0('ztop_plume', ztop_plume)", "dump0('maxwidth', maxwidth)",
      "dump0('excess_h', excess_h)", "dump0('excess_q', excess_q)"]),
    ("    call mym_turbulence(", 1,
     ["dump1('thlv_tot', thlv1)", "dump1('thv_tot', thv1)", "dump1('thl_tot', thl_tot1)",
      "dump1('qc_tot', qc_tot1)", "dump1('qi_tot', qi_tot1)",
      "dump1('el_turb', el1)", "dump1('sh_turb', sh1)", "dump1('sm_turb', sm1)",
      "dump1('dfm', dfm1)", "dump1('dfh', dfh1)", "dump1('dfq', dfq1)",
      "dump1('pdk', pdk1)", "dump1('pdt', pdt1)", "dump1('pdq', pdq1)",
      "dump1('pdc', pdc1)", "dump1('tcd', tcd1)", "dump1('qcd', qcd1)"]),
    ("    call mym_predict(kts,kte,bl_mynn_closure,", 1,
     ["dump1('qke_pred', qke1)", "dump1('tsq_pred', tsq1)", "dump1('qsq_pred', qsq1)",
      "dump1('cov_pred', cov1)"]),
    ("    diss_heat1(kte) = zero", 1, ["dump1('diss_heat', diss_heat1)"]),
]
# the dumps before mym_turbulence must see the inputs, so that hook is split:
PRE = {"    call mym_turbulence(": 5}


def stmt_end(lines, i):
    while lines[i].rstrip().endswith("&"):
        i += 1
    return i


def main(src, out):
    lines = open(src).read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("  subroutine mynnedmf("))
    stop = next(i for i, l in enumerate(lines) if l.startswith("  END SUBROUTINE mynnedmf"))
    ins = []
    for anchor, occ, dumps in HOOKS:
        hits = [i for i in range(start, stop) if lines[i].startswith(anchor)]
        assert len(hits) >= occ, (anchor, hits)
        i = hits[occ - 1]
        npre = PRE.get(anchor, 0)
        if npre:
            ins.append((i, dumps[:npre]))
        e = i if anchor.startswith("    !>") else stmt_end(lines, i)
        if anchor.startswith("    !>"):
            ins.append((i, dumps))
        else:
            ins.append((e + 1, dumps[npre:]))
    for pos, dumps in sorted(ins, key=lambda t: -t[0]):
        lines[pos:pos] = ["    call " + d for d in dumps]
    # module-level use of the hook module, right after "module module_bl_mynnedmf"
    m = lines.index("module module_bl_mynnedmf")
    lines.insert(m + 1, " use mynn_dump_mod, only: dump0, dump1")
    open(out, "w").write("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
