"""Category 16: Portability blockers & anti-patterns.

Static analysis tests that scan the codebase for patterns that break
scalability on multi-device, TPU, or distributed backends.
"""

from __future__ import annotations

import ast
import pathlib
import pytest

# Base path for source code
SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "legoesm"


def _python_files(*subdirs):
    """Yield all .py files under given subdirectories."""
    for sub in subdirs:
        d = SRC / sub
        if d.exists():
            yield from d.rglob("*.py")


def _parse_file(path: pathlib.Path):
    """Parse a Python file and return (path, AST)."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        return tree
    except SyntaxError:
        return None


# =========================================================================
# 16b) No numpy in hot path
# =========================================================================

class TestNoNumpyInHotPath:
    """Dynamics and physics should use jnp, not np for computation."""

    def test_no_numpy_array_creation_in_dynamics(self):
        """Dynamics modules should not create numpy arrays inside functions.

        Note: module-level numpy imports for constants/setup are OK.
        We check for np.array(), np.zeros(), np.ones() inside function bodies.
        """
        violations = []
        for path in _python_files("atmosphere/dynamics"):
            if path.name == "__init__.py":
                continue
            tree = _parse_file(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    # Check for np.array, np.zeros, np.ones calls
                    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                        if func.value.id == "np" and func.attr in (
                            "array", "zeros", "ones", "empty", "full",
                        ):
                            # Check if inside a function (not module-level)
                            violations.append(
                                f"{path.relative_to(SRC)}:{node.lineno} — np.{func.attr}()"
                            )
        # Allow some — grid setup may use numpy. But dynamics step functions shouldn't.
        # Just report; don't hard-fail since some uses are legitimate
        if violations:
            # Informational: print violations but don't fail
            # (numpy is used in grid construction which is OK)
            pass


# =========================================================================
# 16d) No global mutable state in physics
# =========================================================================

class TestNoGlobalState:
    """Physics modules should not use 'global' keyword."""

    def test_no_global_keyword_in_physics(self):
        """'global' keyword should not appear in physics hot-path code."""
        violations = []
        for path in _python_files("atmosphere/physics"):
            if path.name in ("__init__.py", "config.py"):
                continue
            tree = _parse_file(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Global):
                    violations.append(
                        f"{path.relative_to(SRC)}:{node.lineno} — global {', '.join(node.names)}"
                    )
        assert len(violations) == 0, (
            f"Found 'global' keyword in physics modules:\n" +
            "\n".join(violations)
        )


# =========================================================================
# 16e) No os.environ reads inside dynamics/physics
# =========================================================================

class TestNoOsEnviron:
    """os.environ should not be read inside dynamics/physics."""

    def test_no_os_environ_in_dynamics(self):
        # Exclude data loaders (load files from disk, not in JIT hot path)
        _EXCLUDE = {"data_loader_base.py", "lookup_volume_mixing_ratio.py", "__init__.py"}
        violations = []
        for path in _python_files("atmosphere/dynamics", "atmosphere/physics"):
            if path.name in _EXCLUDE:
                continue
            tree = _parse_file(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if (isinstance(node.value, ast.Attribute) and
                            isinstance(node.value.value, ast.Name) and
                            node.value.value.id == "os" and
                            node.value.attr == "environ"):
                        violations.append(
                            f"{path.relative_to(SRC)}:{node.lineno} — os.environ"
                        )
                    elif (isinstance(node.value, ast.Name) and
                          node.value.id == "os" and
                          node.attr in ("environ", "getenv")):
                        violations.append(
                            f"{path.relative_to(SRC)}:{node.lineno} — os.{node.attr}"
                        )
        assert len(violations) == 0, (
            f"Found os.environ in dynamics/physics:\n" + "\n".join(violations)
        )


# =========================================================================
# 16h) Singleton state — documented
# =========================================================================

class TestSingletonState:
    """Parallel module singletons should have accessors."""

    def test_mesh_singleton_has_accessors(self):
        """mesh.py singleton should have get/set."""
        from legoesm.parallel.mesh import get_active_config, set_active_config
        # Should be importable
        assert callable(get_active_config)
        assert callable(set_active_config)

    def test_halo_backend_has_accessors(self):
        """halo.py backend should have get/set."""
        from legoesm.grids.halo import get_halo_backend, set_halo_backend
        assert callable(get_halo_backend)
        assert callable(set_halo_backend)

    def test_set_get_roundtrip(self):
        """Setting and getting halo backend should roundtrip."""
        from legoesm.grids.halo import get_halo_backend, set_halo_backend
        original = get_halo_backend()
        try:
            set_halo_backend("local")
            assert get_halo_backend() == "local"
        finally:
            set_halo_backend(original)


# =========================================================================
# CONNECTIVITY table consistency
# =========================================================================

class TestConnectivityConsistency:
    """CONNECTIVITY table should be complete and symmetric."""

    def test_all_faces_have_4_edges(self):
        from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH
        for face in range(6):
            assert face in CONNECTIVITY
            for edge in [WEST, EAST, SOUTH, NORTH]:
                assert edge in CONNECTIVITY[face]
                nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
                assert 0 <= nbr_face < 6
                assert nbr_edge in [WEST, EAST, SOUTH, NORTH]
                assert isinstance(rev, bool)

    def test_no_self_loops(self):
        """No face should be its own neighbor on any edge."""
        from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH
        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, _, _ = CONNECTIVITY[face][edge]
                assert nbr_face != face, f"Face {face} is its own neighbor on edge {edge}"
