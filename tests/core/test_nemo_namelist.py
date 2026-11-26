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


def test_several_assignments_on_one_line_and_on_the_group_header(tmp_path):
    one_line = tmp_path / "one_line"
    one_line.write_text("&namsbc\n nn_fsbc=5, ln_blk=T,\n rn_x = 1., 2., rn_y(1)=3. rn_z=4.\n/\n")
    got = parse_namelist(one_line, "oce")
    assert got[("oce", "namsbc", "nn_fsbc")] == [5.0]
    assert got[("oce", "namsbc", "ln_blk")] == [True]
    assert got[("oce", "namsbc", "rn_x")] == [1.0, 2.0]
    assert got[("oce", "namsbc", "rn_y(1)")] == [3.0]
    assert got[("oce", "namsbc", "rn_z")] == [4.0]

    header = tmp_path / "header"
    header.write_text("&namsbc nn_fsbc=5\n ln_blk=T\n/\n&namdom rn_dt=90. /\n rn_lost=1.\n")
    got = parse_namelist(header, "oce")
    assert got == {
        ("oce", "namsbc", "nn_fsbc"): [5.0],
        ("oce", "namsbc", "ln_blk"): [True],
        ("oce", "namdom", "rn_dt"): [90.0],
    }


def test_equals_and_slash_inside_quotes_stay_values(tmp_path):
    path = tmp_path / "quoted"
    path.write_text("&nam\n cn_a='x=1, y/z', ln_b=T\n cn_c = './a/b',\n          'c=d'\n/\n")
    got = parse_namelist(path, "oce")
    assert got[("oce", "nam", "cn_a")] == ["x=1, y/z"]
    assert got[("oce", "nam", "ln_b")] == [True]
    assert got[("oce", "nam", "cn_c")] == ["./a/b", "c=d"]


def test_two_assignments_on_one_line_and_on_the_group_header(tmp_path):
    nml = tmp_path / "namelist_cfg"
    nml.write_text("&namsbc\n   nn_fsbc=5, ln_blk=T,\n/\n&namx rn_a=1.0\n/\n")
    parsed = parse_namelist(nml, "oce")
    assert parsed[("oce", "namsbc", "nn_fsbc")] == [5.0]
    assert parsed[("oce", "namsbc", "ln_blk")] == [True]
    assert parsed[("oce", "namx", "rn_a")] == [1.0]
