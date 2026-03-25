"""Tests that Voronoi/MPAS mesh construction requires float64.

Verifies the fail-fast guard: ``create_voronoi_mesh`` must raise
a clear ``RuntimeError`` when JAX x64 is disabled, and produce
float64 geometry arrays when x64 is enabled.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Helper: run a snippet in a *fresh* process so JAX x64 state is clean
# ---------------------------------------------------------------------------

def _run_snippet(code: str, *, env_extra: dict[str, str] | None = None):
    """Execute *code* in a subprocess, return (returncode, stdout, stderr)."""
    import os
    env = os.environ.copy()
    # Ensure x64 is NOT inherited unless we explicitly set it
    env.pop("JAX_ENABLE_X64", None)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVoronoiPrecisionGuard:
    """Verify that create_voronoi_mesh enforces x64."""

    def test_raises_without_x64(self):
        """Must raise RuntimeError when JAX x64 is disabled."""
        code = """\
        from legoesm.grids.voronoi import create_voronoi_mesh
        create_voronoi_mesh(1, lloyd_iterations=2)
        """
        rc, stdout, stderr = _run_snippet(code)
        assert rc != 0, (
            "create_voronoi_mesh should have failed without x64, "
            f"but exited 0.\nstdout: {stdout}\nstderr: {stderr}"
        )
        assert "requires float64" in stderr, (
            f"Error message should mention 'requires float64'.\nstderr: {stderr}"
        )

    def test_succeeds_with_x64(self):
        """Must succeed and return float64 geometry with x64 enabled."""
        code = """\
        from legoesm.grids.voronoi import create_voronoi_mesh
        import jax
        jax.config.update("jax_enable_x64", True)
        m = create_voronoi_mesh(1, lloyd_iterations=2)
        assert m.areaCell.dtype.name == "float64", f"areaCell dtype: {m.areaCell.dtype}"
        assert m.dvEdge.dtype.name == "float64", f"dvEdge dtype: {m.dvEdge.dtype}"
        assert m.weightsOnEdge.dtype.name == "float64", f"weightsOnEdge dtype: {m.weightsOnEdge.dtype}"
        print("OK")
        """
        rc, stdout, stderr = _run_snippet(
            code, env_extra={"JAX_ENABLE_X64": "1"},
        )
        assert rc == 0, (
            f"create_voronoi_mesh failed with x64 enabled.\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )
        assert "OK" in stdout

    def test_env_var_enables_x64(self):
        """JAX_ENABLE_X64=1 env var should be sufficient."""
        code = """\
        from legoesm.grids.voronoi import create_voronoi_mesh
        m = create_voronoi_mesh(1, lloyd_iterations=2)
        print(m.areaCell.dtype)
        """
        rc, stdout, stderr = _run_snippet(
            code, env_extra={"JAX_ENABLE_X64": "1"},
        )
        assert rc == 0, f"Failed with JAX_ENABLE_X64=1.\nstderr: {stderr}"
        assert "float64" in stdout


class TestRequireX64Utility:
    """Unit tests for the require_x64 helper itself."""

    def test_passes_when_x64_enabled(self):
        """No error when x64 is already on."""
        import jax
        jax.config.update("jax_enable_x64", True)
        from legoesm.runtime.backend import require_x64
        # Should not raise
        require_x64("test component")

    def test_error_message_includes_component_name(self):
        """Error message should include the component name."""
        code = """\
        from legoesm.runtime.backend import require_x64
        try:
            require_x64("MyComponent")
        except RuntimeError as e:
            print(str(e))
        """
        rc, stdout, stderr = _run_snippet(code)
        assert "MyComponent" in stdout or "MyComponent" in stderr
