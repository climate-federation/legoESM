"""
JAX translation of histFileMod Fortran module.

Methods for CLM history file handling. Provides stub registration
functions for adding 1-D and 2-D fields to the master history field
list. In the original CLM these routines register metadata and data
pointers for netCDF output; in the JAX translation they serve as
no-op stubs preserving the full public interface.

Original Fortran module: histFileMod
Fortran lines 1-90
"""

from typing import Optional

from jax import Array

# ---------------------------------------------------------------------------
# Public: register a 1-D single-level history field
# ---------------------------------------------------------------------------


def hist_addfld1d(
    fname: str,
    units: str,
    avgflag: str,
    long_name: str,
    type1d_out: Optional[str] = None,  # Output type (from data type)
    ptr_gcell: Optional[Array] = None,  # Pointer to gridcell array
    ptr_lunit: Optional[Array] = None,  # Pointer to landunit array
    ptr_col: Optional[Array] = None,  # Pointer to column array
    ptr_patch: Optional[Array] = None,  # Pointer to patch array
    ptr_lnd: Optional[Array] = None,  # Pointer to lnd array
    ptr_atm: Optional[Array] = None,  # Pointer to atm array
    p2c_scale_type: Optional[str] = None,  # Scale type: pfts -> column
    c2l_scale_type: Optional[str] = None,  # Scale type: columns -> landunits
    l2g_scale_type: Optional[str] = None,  # Scale type: landunits -> gridcells
    set_lake: Optional[float] = None,  # Value to set lakes to
    set_nolake: Optional[float] = None,  # Value to set non-lakes to
    set_urb: Optional[float] = None,  # Value to set urban to
    set_nourb: Optional[float] = None,  # Value to set non-urban to
    set_noglc: Optional[float] = None,  # Value to set non-glacier to
    set_spec: Optional[float] = None,  # Value to set special to
    default: Optional[str] = None,  # 'inactive' to exclude from primary tape
) -> None:
    """
    Register a 1-D single-level field on the master history field list.

    Mirrors Fortran subroutine ``hist_addfld1d`` (lines 22-56).
    The Fortran implementation body is empty (a stub); this translation
    preserves the full public interface signature while providing a
    no-op body.

    Args:
        fname: Field name.
            Fortran: ``character(len=*), intent(in) :: fname``.
        units: Units of the field.
            Fortran: ``character(len=*), intent(in) :: units``.
        avgflag: Time averaging flag.
            Fortran: ``character(len=*), intent(in) :: avgflag``.
        long_name: Long name of the field.
            Fortran: ``character(len=*), intent(in) :: long_name``.
        type1d_out: Output type derived from the data type.
            Fortran: ``character(len=*), optional, intent(in)``.
        ptr_gcell: Data array at the gridcell level.
            Fortran: ``real(r8), optional, pointer :: ptr_gcell(:)``.
        ptr_lunit: Data array at the landunit level.
            Fortran: ``real(r8), optional, pointer :: ptr_lunit(:)``.
        ptr_col: Data array at the column level.
            Fortran: ``real(r8), optional, pointer :: ptr_col(:)``.
        ptr_patch: Data array at the patch level.
            Fortran: ``real(r8), optional, pointer :: ptr_patch(:)``.
        ptr_lnd: Data array at the land level.
            Fortran: ``real(r8), optional, pointer :: ptr_lnd(:)``.
        ptr_atm: Data array at the atmosphere level.
            Fortran: ``real(r8), optional, pointer :: ptr_atm(:)``.
        p2c_scale_type: Scale type for subgrid averaging of PFTs to
            columns. Fortran: ``character(len=*), optional, intent(in)``.
        c2l_scale_type: Scale type for subgrid averaging of columns to
            landunits. Fortran: ``character(len=*), optional, intent(in)``.
        l2g_scale_type: Scale type for subgrid averaging of landunits
            to gridcells. Fortran: ``character(len=*), optional, intent(in)``.
        set_lake: Value to assign to lake points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_nolake: Value to assign to non-lake points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_urb: Value to assign to urban points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_nourb: Value to assign to non-urban points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_noglc: Value to assign to non-glacier points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_spec: Value to assign to special points.
            Fortran: ``real(r8), optional, intent(in)``.
        default: If ``'inactive'``, field will not appear on the
            primary history tape.
            Fortran: ``character(len=*), optional, intent(in)``.
    """


# ---------------------------------------------------------------------------
# Public: register a 2-D multi-level history field
# ---------------------------------------------------------------------------


def hist_addfld2d(
    fname: str,
    type2d: str,
    units: str,
    avgflag: str,
    long_name: str,
    type1d_out: Optional[str] = None,  # Output type (from data type)
    ptr_gcell: Optional[Array] = None,  # Pointer to gridcell array
    ptr_lunit: Optional[Array] = None,  # Pointer to landunit array
    ptr_col: Optional[Array] = None,  # Pointer to column array
    ptr_patch: Optional[Array] = None,  # Pointer to patch array
    ptr_lnd: Optional[Array] = None,  # Pointer to lnd array
    ptr_atm: Optional[Array] = None,  # Pointer to atm array
    p2c_scale_type: Optional[str] = None,  # Scale type: pfts -> column
    c2l_scale_type: Optional[str] = None,  # Scale type: columns -> landunits
    l2g_scale_type: Optional[str] = None,  # Scale type: landunits -> gridcells
    set_lake: Optional[float] = None,  # Value to set lakes to
    set_nolake: Optional[float] = None,  # Value to set non-lakes to
    set_urb: Optional[float] = None,  # Value to set urban to
    set_nourb: Optional[float] = None,  # Value to set non-urban to
    set_spec: Optional[float] = None,  # Value to set special to
    no_snow_behavior: Optional[int] = None,  # Special handling for multi-layer snow fields
    default: Optional[str] = None,  # 'inactive' to exclude from primary tape
) -> None:
    """
    Register a 2-D multi-level field on the master history field list.

    Mirrors Fortran subroutine ``hist_addfld2d`` (lines 58-88).
    The Fortran implementation body is empty (a stub); this translation
    preserves the full public interface signature while providing a
    no-op body.

    Args:
        fname: Field name.
            Fortran: ``character(len=*), intent(in) :: fname``.
        type2d: 2-D output type (e.g. number of levels).
            Fortran: ``character(len=*), intent(in) :: type2d``.
        units: Units of the field.
            Fortran: ``character(len=*), intent(in) :: units``.
        avgflag: Time averaging flag.
            Fortran: ``character(len=*), intent(in) :: avgflag``.
        long_name: Long name of the field.
            Fortran: ``character(len=*), intent(in) :: long_name``.
        type1d_out: Output type derived from the data type.
            Fortran: ``character(len=*), optional, intent(in)``.
        ptr_gcell: 2-D data array at the gridcell level.
            Fortran: ``real(r8), optional, pointer :: ptr_gcell(:,:)``.
        ptr_lunit: 2-D data array at the landunit level.
            Fortran: ``real(r8), optional, pointer :: ptr_lunit(:,:)``.
        ptr_col: 2-D data array at the column level.
            Fortran: ``real(r8), optional, pointer :: ptr_col(:,:)``.
        ptr_patch: 2-D data array at the patch level.
            Fortran: ``real(r8), optional, pointer :: ptr_patch(:,:)``.
        ptr_lnd: 2-D data array at the land level.
            Fortran: ``real(r8), optional, pointer :: ptr_lnd(:,:)``.
        ptr_atm: 2-D data array at the atmosphere level.
            Fortran: ``real(r8), optional, pointer :: ptr_atm(:,:)``.
        p2c_scale_type: Scale type for subgrid averaging of PFTs to
            columns. Fortran: ``character(len=*), optional, intent(in)``.
        c2l_scale_type: Scale type for subgrid averaging of columns to
            landunits. Fortran: ``character(len=*), optional, intent(in)``.
        l2g_scale_type: Scale type for subgrid averaging of landunits
            to gridcells. Fortran: ``character(len=*), optional, intent(in)``.
        set_lake: Value to assign to lake points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_nolake: Value to assign to non-lake points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_urb: Value to assign to urban points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_nourb: Value to assign to non-urban points.
            Fortran: ``real(r8), optional, intent(in)``.
        set_spec: Value to assign to special points.
            Fortran: ``real(r8), optional, intent(in)``.
        no_snow_behavior: If provided, enables special handling for
            multi-layer snow fields.
            Fortran: ``integer, optional, intent(in)``.
        default: If ``'inactive'``, field will not appear on the
            primary history tape.
            Fortran: ``character(len=*), optional, intent(in)``.
    """
