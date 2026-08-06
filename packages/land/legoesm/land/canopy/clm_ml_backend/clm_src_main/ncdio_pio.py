"""
NetCDF I/O module for CLM using Python interfaces.

This module provides generic interfaces to write and read fields to/from
netcdf files for CLM, replacing the Fortran NetCDF PIO functionality with
Python-based NetCDF operations using xarray and netCDF4.

Translation of CLM-ml_v1/clm_src_main/ncdio_pio.F90 to Python/JAX.
"""

import logging

# NetCDF handling libraries
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

try:
    # Suppress specific warnings during netCDF4/xarray import
    # These libraries may generate deprecation warnings from their dependencies
    # that are not relevant to CLM users
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="netCDF4")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
        import netCDF4 as nc4
        import xarray as xr
    NETCDF_AVAILABLE = True
    XarrayDataset = xr.Dataset
    NetCDF4Dataset = nc4.Dataset
except (ImportError, AttributeError):
    # Provide fallback for environments without netCDF or with NumPy compatibility issues
    nc4 = None
    xr = None
    NETCDF_AVAILABLE = False
    # Create dummy types for annotations
    XarrayDataset = type(None)
    NetCDF4Dataset = type(None)

# Import related modules
try:
    from .abortutils import CLMError
except ImportError:
    CLMError = RuntimeError

# Set up logger
logger = logging.getLogger(__name__)


class NCDDataType(Enum):
    """NetCDF data type enumeration."""

    DOUBLE = "double"
    FLOAT = "float"
    INT = "int"
    LONG = "long"


class FileMode(Enum):
    """File access mode enumeration."""

    READ = "r"
    WRITE = "w"
    APPEND = "a"
    READ_WRITE = "r+"


# NetCDF data type constants (for Fortran compatibility)
ncd_double = NCDDataType.DOUBLE
ncd_int = NCDDataType.INT


@dataclass
class file_desc_t:
    """
    NetCDF file descriptor type.

    This class encapsulates NetCDF file operations and maintains
    compatibility with the original Fortran file_desc_t type.

    Attributes:
        ncid: NetCDF file ID or file handle
        filepath: Path to the NetCDF file
        mode: File access mode
        is_open: Whether the file is currently open
        dataset: xarray Dataset for high-level operations
        nc_file: netCDF4 Dataset for low-level operations
        metadata: Additional file metadata
    """

    ncid: Optional[Union[int, str]] = None
    filepath: Optional[Path] = None
    mode: FileMode = FileMode.READ
    is_open: bool = False
    dataset: Optional[XarrayDataset] = None
    nc_file: Optional[NetCDF4Dataset] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize file descriptor."""
        if isinstance(self.filepath, str):
            self.filepath = Path(self.filepath)

    def is_valid(self) -> bool:
        """Check if file descriptor is valid."""
        if not self.is_open:
            return True  # Closed state is valid

        # If marked as open, must have at least one valid handle
        return (self.dataset is not None) or (self.nc_file is not None)

    def get_info(self) -> Dict[str, Any]:
        """Get file descriptor information."""
        info = {
            "filepath": str(self.filepath) if self.filepath else None,
            "mode": self.mode.value,
            "is_open": self.is_open,
            "has_xarray": self.dataset is not None,
            "has_netcdf4": self.nc_file is not None,
            "is_valid": self.is_valid(),
        }

        if self.is_open and self.nc_file:
            try:
                info.update(
                    {
                        "dimensions": list(self.nc_file.dimensions.keys()),
                        "variables": list(self.nc_file.variables.keys()),
                        "num_dimensions": len(self.nc_file.dimensions),
                        "num_variables": len(self.nc_file.variables),
                    }
                )
            except Exception:
                pass

        return info


@dataclass
class NetCDFIOManager:
    """
    Manager for NetCDF I/O operations.

    This class provides centralized management of NetCDF file operations,
    error handling, and performance optimization.

    Attributes:
        open_files: Dictionary of open file descriptors
        default_mode: Default file access mode
        enable_caching: Whether to cache dataset operations
        cache: Simple cache for frequently accessed data
        error_handling: Error handling strategy
    """

    open_files: Dict[str, file_desc_t] = field(default_factory=dict)
    default_mode: FileMode = FileMode.READ
    enable_caching: bool = True
    cache: Dict[str, Any] = field(default_factory=dict)
    error_handling: str = "strict"  # "strict", "warn", "ignore"

    def get_open_file_count(self) -> int:
        """Get number of currently open files."""
        return sum(1 for f in self.open_files.values() if f.is_open)

    def close_all_files(self) -> None:
        """Close all open files."""
        for file_desc in self.open_files.values():
            if file_desc.is_open:
                try:
                    ncd_pio_closefile(file_desc)
                except Exception as e:
                    logger.warning(f"Failed to close file {file_desc.filepath}: {e}")

        self.open_files.clear()
        logger.info("All NetCDF files closed")


# Global NetCDF I/O manager
_netcdf_manager = NetCDFIOManager()


def ncd_pio_openfile(
    ncid: file_desc_t, fname: str, mode: Optional[Union[str, FileMode]] = None
) -> None:
    """
    Open a NetCDF PIO file.

    Args:
        ncid: NetCDF file descriptor (modified in place)
        fname: Input filename to open
        mode: File access mode (optional, defaults to read-only)

    Raises:
        CLMError: If file cannot be opened or NetCDF is not available
        FileNotFoundError: If file does not exist in read mode
    """
    global _netcdf_manager

    if not NETCDF_AVAILABLE:
        raise CLMError("NetCDF libraries not available (install netCDF4 and xarray)")

    try:
        logger.debug(f"Opening NetCDF file: {fname}")

        # Determine file mode
        if mode is None:
            file_mode = FileMode.READ
        elif isinstance(mode, str):
            mode_map = {"r": FileMode.READ, "w": FileMode.WRITE, "a": FileMode.APPEND}
            file_mode = mode_map.get(mode, FileMode.READ)
        else:
            file_mode = mode

        # Check file exists for read mode
        filepath = Path(fname)
        if file_mode == FileMode.READ and not filepath.exists():
            raise FileNotFoundError(f"NetCDF file not found: {fname}")

        # Open with netCDF4 for low-level operations
        nc_file = nc4.Dataset(fname, file_mode.value)

        # Also open with xarray for high-level operations (read mode only)
        dataset = None
        if file_mode == FileMode.READ:
            try:
                dataset = xr.open_dataset(fname)
            except Exception as e:
                logger.warning(f"Could not open with xarray: {e}")

        # Update file descriptor
        ncid.ncid = nc_file.filepath()
        ncid.filepath = filepath
        ncid.mode = file_mode
        ncid.is_open = True
        ncid.nc_file = nc_file
        ncid.dataset = dataset
        ncid.metadata.update(
            {
                "opened_at": Path(fname).stat().st_mtime if filepath.exists() else None,
                "file_size": filepath.stat().st_size if filepath.exists() else 0,
            }
        )

        # Register with manager
        _netcdf_manager.open_files[fname] = ncid

        logger.debug(f"Successfully opened NetCDF file: {fname} (mode: {file_mode.value})")

    except Exception as e:
        error_msg = f"Failed to open NetCDF file '{fname}': {e}"
        logger.error(error_msg)
        raise CLMError(error_msg) from e


def ncd_pio_closefile(ncid: file_desc_t) -> None:
    """
    Close a NetCDF PIO file.

    Args:
        ncid: NetCDF file descriptor

    Raises:
        CLMError: If file cannot be closed properly
    """
    global _netcdf_manager

    try:
        if not ncid.is_open:
            logger.debug("File already closed")
            return

        logger.debug(f"Closing NetCDF file: {ncid.filepath}")

        # Close xarray dataset
        if ncid.dataset is not None:
            ncid.dataset.close()
            ncid.dataset = None

        # Close netCDF4 dataset
        if ncid.nc_file is not None:
            ncid.nc_file.close()
            ncid.nc_file = None

        # Update file descriptor
        ncid.is_open = False
        ncid.ncid = None

        # Remove from manager
        if ncid.filepath and str(ncid.filepath) in _netcdf_manager.open_files:
            del _netcdf_manager.open_files[str(ncid.filepath)]

        logger.debug(f"Successfully closed NetCDF file: {ncid.filepath}")

    except Exception as e:
        error_msg = f"Failed to close NetCDF file: {e}"
        logger.error(error_msg)
        raise CLMError(error_msg) from e


def ncd_inqdid(ncid: file_desc_t, name: str) -> int:
    """
    Inquire on a dimension id.

    Args:
        ncid: NetCDF file descriptor
        name: Dimension name

    Returns:
        Dimension id (index in dimensions list)

    Raises:
        CLMError: If dimension is not found
    """
    try:
        if not ncid.is_open or ncid.nc_file is None:
            raise CLMError("File not properly opened")

        logger.debug(f"Inquiring dimension id for: {name}")

        # Get dimension info
        if name not in ncid.nc_file.dimensions:
            available_dims = list(ncid.nc_file.dimensions.keys())
            raise CLMError(f"Dimension '{name}' not found. Available: {available_dims}")

        # Return dimension "id" (we use the position in the dimensions dict)
        dim_names = list(ncid.nc_file.dimensions.keys())
        dimid = dim_names.index(name)

        logger.debug(f"Found dimension '{name}' with id {dimid}")
        return dimid

    except Exception as e:
        error_msg = f"Failed to inquire dimension id for '{name}': {e}"
        logger.error(error_msg)

        raise CLMError(error_msg) from e


def ncd_inqdlen(ncid: file_desc_t, dimid: Union[int, str]) -> int:
    """
    Get the length of the given dimension.

    Args:
        ncid: NetCDF file descriptor
        dimid: Dimension id or name

    Returns:
        Dimension length

    Raises:
        CLMError: If dimension is not found
    """
    try:
        if not ncid.is_open or ncid.nc_file is None:
            raise CLMError("File not properly opened")

        logger.debug(f"Inquiring dimension length for: {dimid}")

        # Convert dimid to dimension name if needed
        if isinstance(dimid, int):
            dim_names = list(ncid.nc_file.dimensions.keys())
            if dimid >= len(dim_names):
                raise CLMError(f"Dimension id {dimid} out of range (max: {len(dim_names)-1})")
            dim_name = dim_names[dimid]
        else:
            dim_name = dimid

        # Get dimension length
        if dim_name not in ncid.nc_file.dimensions:
            available_dims = list(ncid.nc_file.dimensions.keys())
            raise CLMError(f"Dimension '{dim_name}' not found. Available: {available_dims}")

        dimlen = len(ncid.nc_file.dimensions[dim_name])

        logger.debug(f"Dimension '{dim_name}' has length {dimlen}")
        return dimlen

    except Exception as e:
        error_msg = f"Failed to inquire dimension length for '{dimid}': {e}"
        logger.error(error_msg)

        raise CLMError(error_msg) from e


def ncd_defvar(*args, **kwargs) -> None:
    """
    Define variables (dummy routine for compatibility).

    This function is a placeholder to maintain compatibility with
    the original Fortran interface. Variable definition should be
    handled during file creation.
    """
    logger.debug("ncd_defvar called (dummy routine)")
    pass


def ncd_inqvdlen(*args, **kwargs) -> None:
    """
    Inquire variable dimension length (dummy routine for compatibility).

    This function is a placeholder to maintain compatibility with
    the original Fortran interface. Use ncd_inqdlen instead.
    """
    logger.debug("ncd_inqvdlen called (dummy routine)")
    pass


def ncd_io_1d(
    varname: str,
    data: jnp.ndarray,
    flag: str,
    ncid: file_desc_t,
    readvar: Optional[bool] = None,
    nt: Optional[int] = None,
    posNOTonfile: Optional[bool] = None,
) -> Tuple[jnp.ndarray, bool]:
    """
    NetCDF I/O of a 1D variable.

    Args:
        varname: Variable name
        data: Data array (input for write, modified for read)
        flag: 'read' or 'write'
        ncid: NetCDF file descriptor
        readvar: Was variable read successfully (output)
        nt: Time sample index (optional)
        posNOTonfile: Position is NOT on this file (optional)

    Returns:
        Tuple of (data_array, success_flag)

    Raises:
        CLMError: If I/O operation fails
    """
    try:
        if not ncid.is_open or ncid.nc_file is None:
            raise CLMError("File not properly opened")

        logger.debug(f"NetCDF I/O 1D: {varname}, flag: {flag}")

        flag_lower = flag.lower().strip()

        if flag_lower == "read":
            return _read_variable_1d(ncid, varname, data, nt)
        elif flag_lower == "write":
            return _write_variable_1d(ncid, varname, data, nt)
        else:
            raise CLMError(f"Invalid flag '{flag}'. Must be 'read' or 'write'")

    except Exception as e:
        error_msg = f"NetCDF I/O 1D failed for variable '{varname}': {e}"
        logger.error(error_msg)
        raise CLMError(error_msg) from e


def ncd_io_2d(
    varname: str,
    data: jnp.ndarray,
    flag: str,
    ncid: file_desc_t,
    readvar: Optional[bool] = None,
    nt: Optional[int] = None,
    posNOTonfile: Optional[bool] = None,
) -> Tuple[jnp.ndarray, bool]:
    """
    NetCDF I/O of a 2D variable.

    Args:
        varname: Variable name
        data: Data array (input for write, modified for read)
        flag: 'read' or 'write'
        ncid: NetCDF file descriptor
        readvar: Was variable read successfully (output)
        nt: Time sample index (optional)
        posNOTonfile: Position is NOT on this file (optional)

    Returns:
        Tuple of (data_array, success_flag)

    Raises:
        CLMError: If I/O operation fails
    """
    try:
        if not ncid.is_open or ncid.nc_file is None:
            raise CLMError("File not properly opened")

        logger.debug(f"NetCDF I/O 2D: {varname}, flag: {flag}")

        flag_lower = flag.lower().strip()

        if flag_lower == "read":
            return _read_variable_2d(ncid, varname, data, nt)
        elif flag_lower == "write":
            return _write_variable_2d(ncid, varname, data, nt)
        else:
            raise CLMError(f"Invalid flag '{flag}'. Must be 'read' or 'write'")

    except Exception as e:
        error_msg = f"NetCDF I/O 2D failed for variable '{varname}': {e}"
        logger.error(error_msg)
        raise CLMError(error_msg) from e


def _read_variable_1d(
    ncid: file_desc_t, varname: str, data: jnp.ndarray, nt: Optional[int]
) -> Tuple[jnp.ndarray, bool]:
    """
    Read a 1D variable from NetCDF file.

    Args:
        ncid: NetCDF file descriptor
        varname: Variable name
        data: Data array template (for shape/type info)
        nt: Time index (optional)

    Returns:
        Tuple of (data_array, success_flag)
    """
    try:
        # Check if variable exists
        if varname not in ncid.nc_file.variables:
            available_vars = list(ncid.nc_file.variables.keys())
            logger.warning(f"Variable '{varname}' not found. Available: {available_vars}")
            return data, False

        # Get variable
        var = ncid.nc_file.variables[varname]

        # Read data with time index if specified
        if nt is not None and "time" in var.dimensions:
            # Assume time is the first dimension
            read_data = var[nt, :]
        else:
            read_data = var[:]

        # Convert to JAX array
        jax_data = jnp.array(read_data)

        # Ensure 1D
        if jax_data.ndim > 1:
            jax_data = jax_data.flatten()

        logger.debug(f"Successfully read 1D variable '{varname}' with shape {jax_data.shape}")
        return jax_data, True

    except Exception as e:
        logger.error(f"Failed to read 1D variable '{varname}': {e}")
        return data, False


def _read_variable_2d(
    ncid: file_desc_t, varname: str, data: jnp.ndarray, nt: Optional[int]
) -> Tuple[jnp.ndarray, bool]:
    """
    Read a 2D variable from NetCDF file.

    Args:
        ncid: NetCDF file descriptor
        varname: Variable name
        data: Data array template (for shape/type info)
        nt: Time index (optional)

    Returns:
        Tuple of (data_array, success_flag)
    """
    try:
        # Check if variable exists
        if varname not in ncid.nc_file.variables:
            available_vars = list(ncid.nc_file.variables.keys())
            logger.warning(f"Variable '{varname}' not found. Available: {available_vars}")
            return data, False

        # Get variable
        var = ncid.nc_file.variables[varname]

        # Read data with time index if specified
        if nt is not None and "time" in var.dimensions:
            # Assume time is the first dimension
            read_data = var[nt, :, :]
        else:
            read_data = var[:, :]

        # Convert to JAX array
        jax_data = jnp.array(read_data)

        # Ensure 2D
        if jax_data.ndim == 1:
            # Reshape based on original data shape if available
            if data.ndim == 2:
                jax_data = jax_data.reshape(data.shape)
            else:
                jax_data = jax_data.reshape(-1, 1)
        elif jax_data.ndim > 2:
            # Flatten extra dimensions
            jax_data = jax_data.reshape(jax_data.shape[0], -1)

        logger.debug(f"Successfully read 2D variable '{varname}' with shape {jax_data.shape}")
        return jax_data, True

    except Exception as e:
        logger.error(f"Failed to read 2D variable '{varname}': {e}")
        return data, False


def _write_variable_1d(
    ncid: file_desc_t, varname: str, data: jnp.ndarray, nt: Optional[int]
) -> Tuple[jnp.ndarray, bool]:
    """
    Write a 1D variable to NetCDF file.

    Args:
        ncid: NetCDF file descriptor
        varname: Variable name
        data: Data array to write
        nt: Time index (optional)

    Returns:
        Tuple of (data_array, success_flag)
    """
    try:
        # Convert JAX array to numpy for netCDF writing
        np_data = np.array(data)

        # Create variable if it doesn't exist
        if varname not in ncid.nc_file.variables:
            # Create dimensions if needed
            dim_name = f"{varname}_dim"
            if dim_name not in ncid.nc_file.dimensions:
                ncid.nc_file.createDimension(dim_name, len(np_data))

            # Create variable
            var = ncid.nc_file.createVariable(varname, "f8", (dim_name,))
        else:
            var = ncid.nc_file.variables[varname]

        # Write data
        if nt is not None and "time" in var.dimensions:
            var[nt, :] = np_data
        else:
            var[:] = np_data

        # Sync to disk
        ncid.nc_file.sync()

        logger.debug(f"Successfully wrote 1D variable '{varname}' with shape {np_data.shape}")
        return data, True

    except Exception as e:
        logger.error(f"Failed to write 1D variable '{varname}': {e}")
        return data, False


def _write_variable_2d(
    ncid: file_desc_t, varname: str, data: jnp.ndarray, nt: Optional[int]
) -> Tuple[jnp.ndarray, bool]:
    """
    Write a 2D variable to NetCDF file.

    Args:
        ncid: NetCDF file descriptor
        varname: Variable name
        data: Data array to write
        nt: Time index (optional)

    Returns:
        Tuple of (data_array, success_flag)
    """
    try:
        # Convert JAX array to numpy for netCDF writing
        np_data = np.array(data)

        # Create variable if it doesn't exist
        if varname not in ncid.nc_file.variables:
            # Create dimensions with common names if they don't exist
            # Use 'time' for first dimension and 'space' for second if not already defined
            dims = list(ncid.nc_file.dimensions.keys())

            if len(dims) >= 2:
                # Reuse existing dimensions if sizes match
                dim1_name = dims[0]
                dim2_name = dims[1] if len(dims) > 1 else f"{varname}_dim2"
            else:
                # Create new dimensions with standard names
                dim1_name = "time" if "time" not in dims else f"{varname}_dim1"
                dim2_name = "space" if "space" not in dims else f"{varname}_dim2"

            if dim1_name not in ncid.nc_file.dimensions:
                ncid.nc_file.createDimension(dim1_name, np_data.shape[0])
            if dim2_name not in ncid.nc_file.dimensions:
                ncid.nc_file.createDimension(dim2_name, np_data.shape[1])

            # Create variable
            var = ncid.nc_file.createVariable(varname, "f8", (dim1_name, dim2_name))
        else:
            var = ncid.nc_file.variables[varname]

        # Write data
        if nt is not None and "time" in var.dimensions:
            var[nt, :, :] = np_data
        else:
            var[:, :] = np_data

        # Sync to disk
        ncid.nc_file.sync()

        logger.debug(f"Successfully wrote 2D variable '{varname}' with shape {np_data.shape}")
        return data, True

    except Exception as e:
        logger.error(f"Failed to write 2D variable '{varname}': {e}")
        return data, False


# Generic ncd_io interface (maintains Fortran compatibility)
def ncd_io(
    varname: str,
    data: jnp.ndarray,
    flag: str,
    ncid: file_desc_t,
    readvar: Optional[bool] = None,
    nt: Optional[int] = None,
    posNOTonfile: Optional[bool] = None,
) -> Tuple[jnp.ndarray, bool]:
    """
    Generic NetCDF I/O interface.

    This function automatically dispatches to 1D or 2D I/O based on data dimensionality.

    Args:
        varname: Variable name
        data: Data array
        flag: 'read' or 'write'
        ncid: NetCDF file descriptor
        readvar: Was variable read successfully (output)
        nt: Time sample index (optional)
        posNOTonfile: Position is NOT on this file (optional)

    Returns:
        Tuple of (data_array, success_flag)
    """
    if data.ndim == 1:
        return ncd_io_1d(varname, data, flag, ncid, readvar, nt, posNOTonfile)
    elif data.ndim == 2:
        return ncd_io_2d(varname, data, flag, ncid, readvar, nt, posNOTonfile)
    else:
        raise CLMError(f"Unsupported data dimensionality: {data.ndim}. Only 1D and 2D supported.")


# Utility functions
def get_netcdf_manager() -> NetCDFIOManager:
    """Get the global NetCDF I/O manager."""
    return _netcdf_manager


def reset_netcdf_manager() -> None:
    """Reset the NetCDF I/O manager."""
    global _netcdf_manager
    _netcdf_manager.close_all_files()
    _netcdf_manager = NetCDFIOManager()


def create_simple_netcdf_file(
    filename: str, variables: Dict[str, jnp.ndarray], dimensions: Optional[Dict[str, int]] = None
) -> file_desc_t:
    """
    Create a simple NetCDF file with specified variables.

    Args:
        filename: Output filename
        variables: Dictionary of variable_name -> data_array
        dimensions: Optional dimension specifications

    Returns:
        File descriptor for the created file
    """
    try:
        # Create file
        ncid = file_desc_t()
        ncd_pio_openfile(ncid, filename, FileMode.WRITE)

        # Write variables
        for varname, data in variables.items():
            _, success = ncd_io(varname, data, "write", ncid)
            if not success:
                logger.warning(f"Failed to write variable '{varname}'")

        logger.info(f"Created simple NetCDF file: {filename} with {len(variables)} variables")
        return ncid

    except Exception as e:
        logger.error(f"Failed to create simple NetCDF file: {e}")
        raise CLMError(f"NetCDF file creation failed: {e}") from e


def print_netcdf_summary(ncid: file_desc_t) -> None:
    """
    Print a summary of NetCDF file contents.

    Args:
        ncid: NetCDF file descriptor
    """
    if not ncid.is_open or ncid.nc_file is None:
        print("File is not open")
        return

    info = ncid.get_info()

    print(f"\n=== NetCDF File Summary: {ncid.filepath} ===")
    print(f"Mode: {info['mode']}")
    print(f"Dimensions: {info.get('num_dimensions', 0)}")
    print(f"Variables: {info.get('num_variables', 0)}")

    if "dimensions" in info:
        print("\nDimensions:")
        for dim_name in info["dimensions"]:
            dim_len = ncd_inqdlen(ncid, dim_name)
            print(f"  {dim_name}: {dim_len}")

    if "variables" in info:
        print("\nVariables:")
        for var_name in info["variables"][:10]:  # Show first 10
            var = ncid.nc_file.variables[var_name]
            print(f"  {var_name}: {var.shape} ({var.dtype})")

        if len(info["variables"]) > 10:
            print(f"  ... and {len(info['variables']) - 10} more variables")

    print("=" * 50)


# Export interface
__all__ = [
    "file_desc_t",
    "ncd_pio_openfile",
    "ncd_pio_closefile",
    "ncd_inqdid",
    "ncd_inqdlen",
    "ncd_defvar",
    "ncd_inqvdlen",
    "ncd_io",
    "ncd_io_1d",
    "ncd_io_2d",
    "ncd_double",
    "ncd_int",
    "NCDDataType",
    "FileMode",
    "NetCDFIOManager",
    "get_netcdf_manager",
    "reset_netcdf_manager",
    "create_simple_netcdf_file",
    "print_netcdf_summary",
]
