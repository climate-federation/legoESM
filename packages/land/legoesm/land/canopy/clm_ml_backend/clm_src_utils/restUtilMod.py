"""
JAX translation of restUtilMod module.

This module provides generic routines and types for use with restart files.
Translated from Fortran source: restUtilMod.F90 (lines 1-93)

Note: This module provides I/O interfaces for restart files. Since JAX computations
are pure and JIT-compiled, actual I/O operations must be handled outside the
computational graph. These functions serve as interface stubs that would be
implemented in host Python code managing the JAX kernels.

Original Fortran module structure:
- Generic interface 'restartvar' for 1D and 2D array handling
- NetCDF-based restart file I/O operations
- Support for various data types and attributes

In JAX implementation:
- I/O operations are stubs (cannot be JIT-compiled)
- Actual file operations handled via host callbacks or pre/post-processing
- Type-safe interfaces using NamedTuples and Protocols
"""

from typing import NamedTuple, Optional, Protocol, Tuple, Union, runtime_checkable

import jax.numpy as jnp
from jax import Array

# =============================================================================
# Type Definitions
# =============================================================================

# Type alias for double precision (r8 in Fortran, line 7)
r8 = jnp.float64


@runtime_checkable
class RestartVarProtocol(Protocol):
    """
    Protocol defining the interface for restart variable operations.

    Corresponds to the Fortran interface 'restartvar' (lines 16-19).
    This protocol allows for multiple implementations handling different
    array dimensions (1D, 2D, etc.).
    """

    def __call__(self, varname: str, data: Array, **kwargs) -> Array:
        """
        Generic restart variable handler.

        Args:
            varname: Name of the variable in restart file
            data: Array data (1D, 2D, etc.)
            **kwargs: Additional arguments for restart operations

        Returns:
            Processed array data
        """
        ...


class RestartVar1DResult(NamedTuple):
    """Result from restartvar_1d operation.

    Attributes:
        data: The 1D data array
        readvar: Whether the variable was successfully read
    """

    data: Array
    readvar: bool


class RestartVar2DResult(NamedTuple):
    """Result from restartvar_2d operation.

    Attributes:
        data: The 2D data array
        readvar: Whether the variable was successfully read
    """

    data: Array
    readvar: bool


# =============================================================================
# Core Functions
# =============================================================================


def restartvar_1d(
    ncid: int,
    flag: str,
    varname: str,
    xtype: int,
    dim1name: str,
    dim2name: str,
    switchdim: bool,
    long_name: str,
    units: str,
    interpinic_flag: str,
    data: Array,
    readvar: bool,
    comment: Optional[str] = None,
    flag_meanings: Optional[Tuple[str, ...]] = None,
    missing_value: Optional[float] = None,
    fill_value: Optional[float] = None,
    imissing_value: Optional[int] = None,
    ifill_value: Optional[int] = None,
    flag_values: Optional[Tuple[int, ...]] = None,
    nvalid_range: Optional[Tuple[int, int]] = None,
) -> RestartVar1DResult:
    """Handle reading/writing 1D restart variables to/from NetCDF files.

    This is a stub implementation for the restart I/O interface. In JAX-based
    implementations, restart I/O is typically handled outside the JIT-compiled
    computational core, using standard Python I/O libraries.

    Translated from Fortran source lines 26-57 in restUtilMod.F90.

    Args:
        ncid: NetCDF file identifier
        flag: Operation flag ('read' or 'write')
        varname: Variable name (or colon-delimited list)
        xtype: NetCDF data type
        dim1name: First dimension name
        dim2name: Second dimension name
        switchdim: Whether to switch dimensions
        long_name: Long descriptive name for variable
        units: Units string for variable
        interpinic_flag: Flag for interpolation using interpinic
        data: 1D array of data to write or buffer for reading
        readvar: Whether variable was successfully read
        comment: Optional comment attribute
        flag_meanings: Optional flag meanings attribute
        missing_value: Optional missing value for real data
        fill_value: Optional fill value for real data
        imissing_value: Optional missing value for integer data
        ifill_value: Optional fill value for integer data
        flag_values: Optional flag values for integer data
        nvalid_range: Optional valid range [min, max] for integer data

    Returns:
        RestartVar1DResult containing:
            - data: The data array (unchanged in stub)
            - readvar: Updated read status flag

    Note:
        This is a stub that returns the input data unchanged. Actual I/O
        operations should be implemented in the host Python code that manages
        the JAX computational kernels.
    """
    # Validate input dimensions
    assert data.ndim == 1, f"Expected 1D array, got {data.ndim}D"

    # Stub implementation - actual I/O happens outside JAX
    # In a real implementation, this would interface with NetCDF libraries
    # but those operations are not JIT-compatible

    if flag.lower() == "read":
        # For read operations, return the data as-is (would be populated by actual I/O)
        return RestartVar1DResult(data=data, readvar=True)
    elif flag.lower() == "write":
        # For write operations, data is written (side effect handled externally)
        return RestartVar1DResult(data=data, readvar=readvar)
    else:
        raise ValueError(f"Invalid flag: {flag}. Must be 'read' or 'write'")


def restartvar_2d(
    ncid: int,
    flag: str,
    varname: str,
    xtype: int,
    dim1name: str,
    dim2name: str,
    switchdim: bool,
    long_name: str,
    units: str,
    interpinic_flag: str,
    data: Array,
    readvar: bool,
    comment: Optional[str] = None,
    flag_meanings: Optional[Tuple[str, ...]] = None,
    missing_value: Optional[float] = None,
    fill_value: Optional[float] = None,
    imissing_value: Optional[int] = None,
    ifill_value: Optional[int] = None,
    flag_values: Optional[Tuple[int, ...]] = None,
    nvalid_range: Optional[Tuple[int, int]] = None,
) -> RestartVar2DResult:
    """Handle reading/writing 2D restart variables to/from NetCDF files.

    This function provides an interface for restart I/O operations on 2D arrays.
    Since this is an I/O operation, it cannot be JIT-compiled and should be
    handled outside JAX's computational graph.

    Fortran source: restUtilMod.F90, lines 60-91

    Args:
        ncid: NetCDF file descriptor/id
        flag: Operation flag - 'read' or 'write'
        varname: Variable name (or colon-delimited list)
        xtype: NetCDF data type identifier
        dim1name: First dimension name
        dim2name: Second dimension name
        switchdim: Whether to switch dimension order
        long_name: Long descriptive name for variable
        units: Units string for variable
        interpinic_flag: Flag for interpolation using interpinic
        data: 2D data array to read into or write from, shape (dim1, dim2)
        readvar: Input flag indicating if variable should be read
        comment: Optional attribute comment
        flag_meanings: Optional tuple of flag meaning strings
        missing_value: Optional missing value indicator (real)
        fill_value: Optional fill value (real)
        imissing_value: Optional missing value indicator (integer)
        ifill_value: Optional fill value (integer)
        flag_values: Optional tuple of flag values (integer)
        nvalid_range: Optional valid range as (min, max) tuple

    Returns:
        RestartVar2DResult containing:
            - data: The data array (potentially modified if reading)
            - readvar: Whether variable was successfully read

    Note:
        This is a placeholder for I/O operations. In a real implementation,
        this would interface with NetCDF libraries outside of JAX's JIT
        compilation. The actual I/O should be performed on the host before
        or after JIT-compiled computations.
    """
    # Validate input dimensions
    assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"

    # This is a stub implementation since actual I/O cannot be JIT-compiled
    # In practice, this would be handled via:
    # 1. Host callbacks for I/O operations
    # 2. Pre-loading data before JIT compilation
    # 3. Post-processing results after JIT compilation

    if flag.lower() == "read":
        # For read operations, return the data as-is (would be populated by actual I/O)
        return RestartVar2DResult(data=data, readvar=True)
    elif flag.lower() == "write":
        # For write operations, data is written (side effect handled externally)
        return RestartVar2DResult(data=data, readvar=readvar)
    else:
        raise ValueError(f"Invalid flag: {flag}. Must be 'read' or 'write'")


def restartvar(
    varname: str,
    xtype: int,
    dim1name: str,
    dim2name: Optional[str] = None,
    switchdim: Optional[int] = None,
    lowerb2: Optional[int] = None,
    upperb2: Optional[int] = None,
    data: Optional[Union[Array, None]] = None,
    readvar: bool = False,
    comment: Optional[str] = None,
    flag: Optional[str] = None,
    **kwargs,
) -> Union[RestartVar1DResult, RestartVar2DResult]:
    """Generic interface for restart variable I/O operations.

    This function dispatches to appropriate 1D or 2D handling based on the
    presence of dim2name parameter. It mimics the Fortran interface behavior
    where restartvar can handle both 1D and 2D arrays.

    Fortran source: restUtilMod.F90, lines 16-20 (interface definition)

    Args:
        varname: Name of the variable in restart file
        xtype: NetCDF external data type
        dim1name: Name of first dimension
        dim2name: Name of second dimension (None for 1D arrays)
        switchdim: Dimension to switch (for 2D arrays)
        lowerb2: Lower bound of second dimension
        upperb2: Upper bound of second dimension
        data: Input/output data array (1D or 2D)
        readvar: True for read operation, False for write
        comment: Variable comment/description
        flag: Control flag for special handling
        **kwargs: Additional arguments passed to specific implementations

    Returns:
        RestartVar1DResult or RestartVar2DResult depending on dimensionality

    Note:
        Fortran source lines 16-20 define the generic interface.
        This implementation provides runtime dispatch based on array dimensions.
    """
    if data is None:
        raise ValueError("data array must be provided")

    # Dispatch based on whether this is 1D or 2D
    if dim2name is None or data.ndim == 1:
        # Call 1D version
        return restartvar_1d(
            ncid=kwargs.get("ncid", 0),
            flag=flag or "read",
            varname=varname,
            xtype=xtype,
            dim1name=dim1name,
            dim2name=dim2name or "",
            switchdim=bool(switchdim),
            long_name=kwargs.get("long_name", ""),
            units=kwargs.get("units", ""),
            interpinic_flag=kwargs.get("interpinic_flag", ""),
            data=data,
            readvar=readvar,
            comment=comment,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["ncid", "long_name", "units", "interpinic_flag"]
            },
        )
    else:
        # Call 2D version
        return restartvar_2d(
            ncid=kwargs.get("ncid", 0),
            flag=flag or "read",
            varname=varname,
            xtype=xtype,
            dim1name=dim1name,
            dim2name=dim2name,
            switchdim=bool(switchdim),
            long_name=kwargs.get("long_name", ""),
            units=kwargs.get("units", ""),
            interpinic_flag=kwargs.get("interpinic_flag", ""),
            data=data,
            readvar=readvar,
            comment=comment,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["ncid", "long_name", "units", "interpinic_flag"]
            },
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "r8",
    "RestartVarProtocol",
    "RestartVar1DResult",
    "RestartVar2DResult",
    "restartvar",
    "restartvar_1d",
    "restartvar_2d",
]
