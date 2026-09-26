"""The rank-tagged DINO kt=1 acquisition: its patch, and its reader.

The record this builds exists because the shipped one cannot be read: NEMO's
``dyn_spg_ts`` writes every debug stream to a fixed filename under guards that
never mention the MPI rank (``dynspg_ts.F90:206``, ``:926``), and the run used
16 ranks in one directory.  Everything here is about making that impossible to
repeat SILENTLY -- each test is a synthetic violation of one of the properties
the acquisition claims.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "scripts", "validate", "ocean_fidelity",
    "dino_1226", "nemo_dino_kt1_rankdump"))
_ORACLE_SRC = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
               "MY_SRC/dynspg_ts.F90")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rankdump_patch = _load("rankdump_patch")
read_rankdump = _load("read_rankdump")


def test_run_sh_is_committed_and_executable():
    p = os.path.join(_DIR, "run.sh")
    assert os.path.exists(p)
    assert os.access(p, os.X_OK)


def _fake_nemo(tmp_path):
    """The smallest tree run.sh will accept far enough to reach its guards."""
    n = tmp_path / "nemo"
    (n / "cfgs" / "DINO" / "RUN_FROMREST_KT1").mkdir(parents=True)
    (n / "arch").mkdir()
    (n / "arch" / "arch-conda.fcm").write_text("")
    return n


@pytest.mark.parametrize("where", ["cfgs/SHARED", "cfgs/DINO/RUN_X", "src/OCE",
                                   "cfgs", ""])
def test_run_sh_refuses_to_write_the_record_inside_the_oracle(tmp_path, where):
    """The record may not land anywhere in the NEMO checkout.

    An earlier guard named only cfgs/DINO and src/, and a reviewer walked
    OUT=$NEMO/cfgs/SHARED straight through it -- run.sh would then have
    clobbered the namelist_ref every configuration includes.
    """
    n = _fake_nemo(tmp_path)
    out = str(n / where) if where else str(n)
    r = subprocess.run(
        ["bash", os.path.join(_DIR, "run.sh")], capture_output=True, text=True,
        env={**os.environ, "NEMO": str(n), "OUT": out})
    assert r.returncode == 2, r.stdout + r.stderr
    assert "REFUSING" in r.stderr, r.stderr


def test_run_sh_resolves_symlinks_before_deciding(tmp_path):
    """Canonicalisation must be LOAD-BEARING, not incidental.

    NEMO and OUT are given through DIFFERENT names for the same tree: NEMO by
    its real path, OUT through a symlink that lands inside it. A guard that
    compared the strings, or that resolved neither, would accept this.
    """
    n = _fake_nemo(tmp_path)
    link = tmp_path / "elsewhere"
    link.symlink_to(n / "cfgs")
    r = subprocess.run(
        ["bash", os.path.join(_DIR, "run.sh")], capture_output=True, text=True,
        env={**os.environ, "NEMO": str(n), "OUT": str(link / "SHARED")})
    assert r.returncode == 2 and "REFUSING" in r.stderr, r.stderr


def test_run_sh_refuses_a_relative_out_that_lands_in_the_oracle(tmp_path):
    """OUT is resolved BEFORE the guard, not after the script cd's to NEMO.

    `guard` canonicalises against the invocation directory while the record is
    created after `cd "$NEMO"`, so a relative OUT was checked in one place and
    written in another -- `OUT=cfgs/SHARED` passed and then landed inside the
    oracle.
    """
    n = _fake_nemo(tmp_path)
    r = subprocess.run(
        ["bash", os.path.join(_DIR, "run.sh")], capture_output=True, text=True,
        cwd=str(n), env={**os.environ, "NEMO": str(n), "OUT": "cfgs/SHARED"})
    assert r.returncode == 2 and "REFUSING" in r.stderr, r.stderr


@pytest.mark.parametrize("out", ["/", "/tmp", "/home", "/data"])
def test_run_sh_refuses_a_system_directory(tmp_path, out):
    n = _fake_nemo(tmp_path)
    r = subprocess.run(
        ["bash", os.path.join(_DIR, "run.sh")], capture_output=True, text=True,
        env={**os.environ, "NEMO": str(n), "OUT": out})
    assert r.returncode == 2 and "system directory" in r.stderr, r.stderr


def test_the_patch_target_inside_the_config_copy_is_ACCEPTED(tmp_path):
    """The refusals must not refuse the one path the script has to write.

    The checkout-wide guard added for `cfgs/SHARED` rejected the config copy's
    own MY_SRC, which killed the acquisition after `makenemo` had already
    created the copy -- and the leftover copy then tripped the "already
    exists" refusal on every retry.
    """
    body = open(os.path.join(_DIR, "run.sh")).read()
    guard = body[body.index("guard() {"):body.index("\nguard \"$COPY\"")]
    probe = (guard + '\nguard "$NEMO/cfgs/DINO_KT1_RANKDUMP/MY_SRC/x.F90" '
             'cfgcopy && echo ACCEPTED\n'
             'guard "$NEMO/cfgs/DINO/MY_SRC/x.F90" cfgcopy && echo LEAKED\n')
    r = subprocess.run(["bash", "-c", probe], capture_output=True, text=True,
                       env={**os.environ, "NEMO": str(tmp_path / "nemo")})
    assert "ACCEPTED" in r.stdout, r.stdout + r.stderr
    assert "LEAKED" not in r.stdout
    assert "REFUSING" in r.stderr


def test_run_sh_refuses_to_reuse_a_config_copy(tmp_path):
    n = _fake_nemo(tmp_path)
    (n / "cfgs" / "DINO_KT1_RANKDUMP").mkdir()
    r = subprocess.run(
        ["bash", os.path.join(_DIR, "run.sh")], capture_output=True, text=True,
        env={**os.environ, "NEMO": str(n), "OUT": str(tmp_path / "out")})
    assert r.returncode == 2 and "already exists" in r.stderr, r.stderr


def test_run_sh_refuses_without_an_arch_file(tmp_path):
    n = _fake_nemo(tmp_path)
    (n / "arch" / "arch-conda.fcm").unlink()
    r = subprocess.run(
        ["bash", os.path.join(_DIR, "run.sh")], capture_output=True, text=True,
        env={**os.environ, "NEMO": str(n), "OUT": str(tmp_path / "out")})
    assert r.returncode == 2 and "arch-conda.fcm" in r.stderr, r.stderr


def test_patch_refuses_to_touch_the_oracle_tree(tmp_path):
    """The whole point of the config copy: cfgs/DINO is never written."""
    victim = tmp_path / "nemo" / "cfgs" / "DINO" / "MY_SRC"
    victim.mkdir(parents=True)
    f = victim / "dynspg_ts.F90"
    f.write_text("      LOGICAL  ::   ll_spg_dump \n")
    before = f.read_text()
    with pytest.raises(SystemExit, match="REFUSING"):
        rankdump_patch.patch(str(f))
    assert f.read_text() == before, "the refusal still wrote the file"


def test_patch_refuses_to_touch_shared_src(tmp_path):
    victim = tmp_path / "nemo" / "src" / "OCE" / "DYN"
    victim.mkdir(parents=True)
    f = victim / "dynspg_ts.F90"
    f.write_text("      LOGICAL  ::   ll_spg_dump \n")
    with pytest.raises(SystemExit, match="REFUSING"):
        rankdump_patch.patch(str(f))


@pytest.mark.skipif(not os.path.exists(_ORACLE_SRC),
                    reason=f"NEMO oracle source not on this machine: "
                           f"{_ORACLE_SRC}")
def test_patch_is_additive_and_rank_tags_every_stream(tmp_path):
    copy = tmp_path / "cfgs" / "DINO_KT1_RANKDUMP" / "MY_SRC"
    copy.mkdir(parents=True)
    f = copy / "dynspg_ts.F90"
    shutil.copy(_ORACLE_SRC, f)
    original = f.read_text()
    rankdump_patch.patch(str(f))
    patched = f.read_text()

    # ADDITIVE: the ONLY lines that leave the file are the debug `OPEN`
    # filenames and the substep block the patch deliberately reframes.  Any
    # other removal would mean the patch touched physics.
    allowed = set(rankdump_patch._OLD_SUBSTEP.splitlines())
    removed = [ln for ln in original.splitlines()
               if ln not in set(patched.splitlines())]
    assert all(ln in allowed or "FILE='" in ln for ln in removed), removed
    assert removed, "the patch removed nothing at all -- it did not apply"
    # no fixed filename literal is left for a second rank to collide on
    import re as _re
    assert not _re.search(r"FILE='[A-Za-z0-9_]+\.bin'", patched)
    assert "FILE='substep_dump" not in patched
    assert "TRIM(cl_rk)" in patched
    assert "ACTION='WRITE'" in patched.split("TRIM(cl_sub)")[1][:400]
    # and it refuses to run twice
    with pytest.raises(SystemExit, match="already patched"):
        rankdump_patch.patch(str(f))


def _tile(path, *, narea, jn, jpi, jpj, nimpp, njmpp, hls, jpiglo, jpjglo,
          fill):
    """One synthetic per-(rank, substep) file in the patched Fortran's layout.

    ``jpiglo``/``jpjglo`` INCLUDE the global halo, exactly as NEMO's do.
    """
    hdr = np.array([jpi, jpj, 45, jn, narea, nimpp, njmpp, hls,
                    1 + hls, jpi - hls, 1 + hls, jpj - hls, jpiglo, jpjglo],
                   dtype=np.int32)
    body = np.full((len(read_rankdump.ARRAYS), jpj, jpi), float(fill))
    with open(path, "wb") as fh:
        hdr.tofile(fh)
        body.tofile(fh)


def test_reader_rejects_a_header_that_disagrees_with_the_length(tmp_path):
    """The raced record's signature: header says one jpj, length says another."""
    p = tmp_path / "substep_r0000_s001.bin"
    _tile(p, narea=1, jn=1, jpi=6, jpj=5, nimpp=1, njmpp=1, hls=1,
          jpiglo=8, jpjglo=6, fill=1.0)
    raw = bytearray(p.read_bytes())
    raw[4:8] = np.int32(4).tobytes()          # lie about jpj
    p.write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="raced writer"):
        read_rankdump.read_tile(str(p))


def test_reader_stitches_the_tiles_onto_the_global_domain(tmp_path):
    # 2 x 1 decomposition of a HALOLESS 8 x 6 domain at halo width 1.
    hls, nxg, nyg = 1, 8, 6
    jpiglo, jpjglo = nxg + 2 * hls, nyg + 2 * hls
    _tile(tmp_path / "substep_r0000_s001.bin", narea=1, jn=1,
          jpi=4 + 2 * hls, jpj=nyg + 2 * hls, nimpp=1, njmpp=1, hls=hls,
          jpiglo=jpiglo, jpjglo=jpjglo, fill=1.0)
    _tile(tmp_path / "substep_r0001_s001.bin", narea=2, jn=1,
          jpi=4 + 2 * hls, jpj=nyg + 2 * hls, nimpp=5, njmpp=1, hls=hls,
          jpiglo=jpiglo, jpjglo=jpjglo, fill=2.0)
    s = read_rankdump.stitch(str(tmp_path), 1)
    assert s["ranks"] == 2
    f = s["un_e"]
    assert f.shape == (nyg, nxg)
    assert not np.isnan(f).any()
    assert np.array_equal(f[:, :4], np.full((nyg, 4), 1.0))
    assert np.array_equal(f[:, 4:], np.full((nyg, 4), 2.0))


def test_reader_refuses_a_tile_map_with_a_hole(tmp_path):
    """A hole would otherwise render as a plausible field with NaNs nobody
    reduced -- worse than a crash."""
    hls, nxg, nyg = 1, 8, 6
    jpiglo, jpjglo = nxg + 2 * hls, nyg + 2 * hls
    _tile(tmp_path / "substep_r0000_s001.bin", narea=1, jn=1,
          jpi=4 + 2 * hls, jpj=nyg + 2 * hls, nimpp=1, njmpp=1, hls=hls,
          jpiglo=jpiglo, jpjglo=jpjglo, fill=1.0)
    with pytest.raises(SystemExit, match="uncovered"):
        read_rankdump.stitch(str(tmp_path), 1)


def test_reader_refuses_a_double_covered_tile_map(tmp_path):
    hls, nxg, nyg = 1, 8, 6
    jpiglo, jpjglo = nxg + 2 * hls, nyg + 2 * hls
    for r, narea in ((0, 1), (1, 2)):
        _tile(tmp_path / f"substep_r{r:04d}_s001.bin", narea=narea, jn=1,
              jpi=nxg + 2 * hls, jpj=nyg + 2 * hls, nimpp=1, njmpp=1,
              hls=hls, jpiglo=jpiglo, jpjglo=jpjglo, fill=float(narea))
    with pytest.raises(SystemExit, match="covered twice"):
        read_rankdump.stitch(str(tmp_path), 1)


_LAYOUT = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
           "RUN_FROMREST_KT1/layout.dat")

#: The REAL DINO kt=1 decomposition, read off layout.dat (16 ranks, 2 x 8,
#: nn_hls = 2, global 56 x 203 INCLUDING the halo for a 52 x 199 domain).
#: (rank0, jpi, jpj, nimpp, njmpp) -- note the ragged last j-band, jpj = 28.
_DINO_LAYOUT = [(r, 30, 29 if r < 14 else 28,
                 1 if r % 2 == 0 else 27,
                 1 + 25 * (r // 2))
                for r in range(16)]


def test_reader_stitches_the_REAL_16_rank_dino_decomposition(tmp_path):
    """The tests above are a 2-rank i-only split; this is the decomposition
    the record is actually produced on, including its ragged last j-band."""
    hls, jpiglo, jpjglo = 2, 56, 203
    for r, jpi, jpj, nimpp, njmpp in _DINO_LAYOUT:
        _tile(tmp_path / f"substep_r{r:04d}_s001.bin", narea=r + 1, jn=1,
              jpi=jpi, jpj=jpj, nimpp=nimpp, njmpp=njmpp, hls=hls,
              jpiglo=jpiglo, jpjglo=jpjglo, fill=float(r))
    s = read_rankdump.stitch(str(tmp_path), 1)
    assert s["ranks"] == 16
    f = s["un_e"]
    assert f.shape == (jpjglo - 2 * hls, jpiglo - 2 * hls) == (199, 52)
    assert not np.isnan(f).any(), "the tile map left a hole"
    # every rank owns a contiguous block, and rank r's fill must land there
    for r, jpi, jpj, nimpp, njmpp in _DINO_LAYOUT:
        ni, nj = jpi - 2 * hls, jpj - 2 * hls
        blk = f[njmpp - 1:njmpp - 1 + nj, nimpp - 1:nimpp - 1 + ni]
        assert np.array_equal(blk, np.full((nj, ni), float(r))), \
            f"rank {r} landed in the wrong place"


@pytest.mark.skipif(not os.path.exists(_LAYOUT),
                    reason=f"NEMO record layout not on this machine: {_LAYOUT}")
def test_the_layout_constant_matches_the_oracles_own_layout_dat():
    """The decomposition above is a CLAIM about the record; read the file."""
    lines = open(_LAYOUT).read().splitlines()
    hdr = next(i for i, ln in enumerate(lines)
               if ln.split()[:6] == ["rank", "ii", "ij", "jpi", "jpj", "nimpp"])
    rows = []
    for ln in lines[hdr + 1:]:
        p = ln.split()
        if len(p) < 7 or not p[0].lstrip("-").isdigit():
            break
        rows.append(tuple(int(x) for x in (p[0], p[3], p[4], p[5], p[6])))
    assert len(rows) == 16, rows
    assert rows == _DINO_LAYOUT, (rows[:4], _DINO_LAYOUT[:4])


def test_header_field_list_matches_the_patch():
    """The reader's field list and the Fortran the patch writes are ONE
    contract; drift between them silently shifts every array."""
    assert len(read_rankdump.HEADER_FIELDS) == 14
    written = [f.strip() for f in rankdump_patch._HEADER.split(",")]
    assert written == list(read_rankdump.HEADER_FIELDS)
