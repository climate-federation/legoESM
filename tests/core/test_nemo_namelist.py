"""Direct test of the shared NEMO namelist reader (source files and resolved dumps)."""

from legoesm.core.nemo_namelist import norm, overlay, parse_namelist


def test_source_and_dump_grammar_and_cfg_overrides_ref(tmp_path):
    ref = tmp_path / "namelist_ref"
    ref.write_text(
        "&namsbc   ! comment\n"
        "   nn_fsbc  = 5    ! not 3\n"
        "   ln_blk   = .false.\n"
        "   cn_dir   = './a!b'\n"
        "   rn_x     = 2*1.5d-3,\n"
        "            0.5\n"
        "/\n"
    )
    cfg = tmp_path / "namelist_cfg"
    cfg.write_text("&namsbc\n   ln_blk = .true.\n/\n")
    dump = tmp_path / "output.namelist.dyn"
    dump.write_text("&NAMSBC\n NN_FSBC=5          ,\n LN_BLK=T,\n/\n")
    merged = overlay(parse_namelist(ref, "oce"), parse_namelist(cfg, "oce"))
    assert merged[("oce", "namsbc", "nn_fsbc")] == [5.0]
    assert merged[("oce", "namsbc", "ln_blk")] == [True]          # cfg replaces ref
    assert merged[("oce", "namsbc", "cn_dir")] == ["./a!b"]       # '!' inside quotes is data
    assert merged[("oce", "namsbc", "rn_x")] == [1.5e-3, 1.5e-3, 0.5]
    resolved = parse_namelist(dump, "oce")
    assert resolved[("oce", "namsbc", "nn_fsbc")] == [5.0] and resolved[("oce", "namsbc", "ln_blk")] == [True]
    assert norm("1.4000000000000000E-003") == 1.4e-3 == norm("1.4e-3")
