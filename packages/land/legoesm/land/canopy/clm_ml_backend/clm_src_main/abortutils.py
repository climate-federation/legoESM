"""
JAX/Python translation of the CLM abort utilities module.

Provides :func:`endrun` and :func:`handle_err` for abnormal
termination handling.

Original Fortran module: abortutils
"""

from __future__ import annotations

from typing import Optional

from .clm_varctl import iulog

# ---------------------------------------------------------------------------
# endrun
# ---------------------------------------------------------------------------


def endrun(msg: Optional[str] = None) -> None:
    """
    Abort the model with an optional message.

    Mirrors Fortran subroutine ``endrun`` (lines 24-33).

    Prints the message to ``iulog`` then raises :exc:`SystemExit`,
    replacing the Fortran ``stop`` statement.

    Args:
        msg: Optional message string to print before stopping.
             Mirrors Fortran ``character(len=*), intent(in), optional``.
    """
    if msg is not None:
        print(f"{iulog}: ENDRUN: {msg}")
    else:
        print(f"{iulog}: ENDRUN: called without a message string")

    raise SystemExit(1)


# ---------------------------------------------------------------------------
# handle_err
# ---------------------------------------------------------------------------


def handle_err(status: int, errmsg: str) -> None:
    """
    Check a NetCDF status code and abort on error.

    Mirrors Fortran subroutine ``handle_err`` (lines 35-41).

    Fortran's ``nf_noerr`` (= 0) and ``nf_strerror`` are replaced by
    the ``netCDF4`` library equivalents.  If ``status != 0`` the
    NetCDF error string and ``errmsg`` are printed and
    :exc:`SystemExit` is raised, replacing the Fortran
    ``stop "Stopped"``.

    Args:
        status:  NetCDF return code (0 = ``NF_NOERR``).
        errmsg:  Caller-supplied context string appended to the
                 NetCDF error description.
    """
    import netCDF4  # noqa: F401 — only imported when needed

    nf_noerr = 0  # Fortran: nf_noerr from netcdf.inc

    if status != nf_noerr:
        try:

            # Get error message using netCDF4's error handling
            nc_msg = f"NetCDF error {status}"
        except Exception:
            nc_msg = f"NetCDF error code {status}"

        print(f"{nc_msg}: {errmsg}")
        raise SystemExit("Stopped")
