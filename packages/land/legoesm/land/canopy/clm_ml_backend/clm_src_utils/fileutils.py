"""File utilities module for CLM-JAX.

This module provides file I/O utility functions translated from Fortran.
Original source: fileutils.F90, lines 1-175

Note: In JAX/Python context, many file operations are handled differently.
File I/O operations involve side effects and are not directly translatable
to JAX's pure functional paradigm. These functions use standard Python I/O
and should be called outside JIT-compiled computation graphs.

Translation Notes:
- get_filename: Pure string manipulation function
- getfil: File location and validation with Python pathlib
- opnfil: File opening with Python native I/O
- relavu: File closing and resource management
- getavu: Not translated (Fortran unit management not needed in Python)

Original Fortran dependencies:
- abortutils (endrun): Replaced with return values and exceptions
- clm_varctl (iulog): Replaced with Python logging
- shr_file_mod: Replaced with Python file handling
"""

import logging
import os
from typing import Literal, Optional, TextIO, Tuple

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

# File format types (for opnfil)
UNFORMATTED_FORMATS = ("u", "U")
FORMATTED_FORMATS = ("f", "F")

# =============================================================================
# PURE UTILITY FUNCTIONS
# =============================================================================


def get_filename(fulpath: str) -> str:
    """Returns filename given full pathname.

    Extracts the filename from a full path by finding the last '/' separator.
    Corresponds to Fortran lines 27-48.

    Args:
        fulpath: Full pathname string

    Returns:
        Filename portion of the path (everything after the last '/')

    Examples:
        >>> get_filename("/path/to/file.txt")
        'file.txt'
        >>> get_filename("file.txt")
        'file.txt'
        >>> get_filename("/path/to/dir/")
        ''

    Note:
        This is a pure Python function (not JIT-compiled) as it operates on
        Python strings which are not JAX arrays. The logic preserves the
        exact behavior of the Fortran implementation (lines 38-46).
    """
    # Fortran line 38: klen = len_trim(fulpath)
    fulpath_trimmed = fulpath.rstrip()
    klen = len(fulpath_trimmed)

    # Fortran lines 39-42: Find last occurrence of '/'
    # do i = klen, 1, -1
    #    if (fulpath(i:i) == '/') go to 10
    # end do
    # i = 0
    # In Fortran, i=0 means fulpath(1:klen) = whole string (1-based indexing)
    # In Python, we need i=-1 so that [i+1:] = [0:] = whole string
    i = -1
    for idx in range(klen - 1, -1, -1):
        if fulpath_trimmed[idx] == "/":
            i = idx
            break

    # Fortran line 43: get_filename = fulpath(i+1:klen)
    # Note: Fortran uses 1-based indexing, Python uses 0-based
    # Fortran i+1:klen maps to Python i+1:klen (since klen is already length)
    result = fulpath_trimmed[i + 1 : klen]

    return result


# =============================================================================
# FILE I/O FUNCTIONS (Side-effectful, not JIT-compatible)
# =============================================================================


def getfil(fulpath: str, iflag: int = 0) -> Tuple[str, bool]:
    """Obtain local copy of file.

    First checks current working directory, then checks full pathname on disk.

    Original Fortran source: fileutils.F90, lines 50-104

    Args:
        fulpath: Archival or permanent disk full pathname
        iflag: 0 => abort if file not found, 1 => do not abort

    Returns:
        Tuple containing:
            - locfn: Output local file name
            - success: True if file was found, False otherwise

    Note:
        This is a pure Python function (not JIT-compiled) as it performs
        file system I/O operations which are inherently side-effectful.
        The original Fortran code calls endrun() which we replace with
        returning a success flag for better composability.

    Example:
        >>> locfn, success = getfil("/data/input.nc")
        >>> if success:
        ...     # Process file
        ...     pass
    """
    # Line 72-73: Get local file name from full name
    locfn = os.path.basename(fulpath)

    # Line 74-79: Check if local filename has zero length
    if len(locfn.strip()) == 0:
        logger.error("(GETFIL): local filename has zero length")
        if iflag == 0:
            return "", False
        else:
            return "", False
    else:
        logger.info(f"(GETFIL): attempting to find local file {locfn.strip()}")

    # Line 81-87: First check if file is in current working directory
    if os.path.exists(locfn):
        logger.info(f"(GETFIL): using {locfn.strip()} in current working directory")
        return locfn, True

    # Line 89-90: Second check for full pathname on disk
    locfn = fulpath

    # Line 92-103: Check if full path exists
    if os.path.exists(fulpath):
        logger.info(f"(GETFIL): using {fulpath.strip()}")
        return locfn, True
    else:
        logger.error(f"(GETFIL): failed getting file from full path: {fulpath}")
        if iflag == 0:
            # In original Fortran, this calls endrun()
            # We raise an exception to match expected abort behavior
            raise FileNotFoundError(f"(GETFIL): file not found: {fulpath}")
        else:
            # iflag=1: do not abort, return the path with failure status
            return fulpath, False


def opnfil(locfn: str, iun: int, form: Literal["u", "U", "f", "F"]) -> Optional[object]:
    """Open file in unformatted or formatted form.

    Translated from Fortran subroutine opnfil (lines 107-141).

    In Python/JAX context, this function provides file opening with error
    handling similar to the Fortran version. Note that Python doesn't have
    the same concept of Fortran unit numbers, but we maintain the interface
    for compatibility.

    Args:
        locfn: Local filename to open. Must be non-empty string.
        iun: Fortran unit number (maintained for interface compatibility,
             but not used in Python file handling).
        form: File format specifier:
              'u' or 'U' = unformatted (binary mode in Python)
              'f' or 'F' = formatted (text mode in Python)

    Returns:
        File handle object if successful, None otherwise.

    Raises:
        SystemExit: If file cannot be opened or filename is empty.

    Note:
        - Original Fortran lines 107-141
        - In Fortran, files are opened on specific unit numbers
        - In Python, we return file handles directly
        - Binary mode ('rb'/'wb') corresponds to unformatted
        - Text mode ('r'/'w') corresponds to formatted

    Example:
        >>> fh = opnfil("data.bin", 10, 'u')
        >>> # Read binary data
        >>> fh.close()
    """
    # Line 123: Check for zero-length filename
    if len(locfn.strip()) == 0:
        logger.error("(OPNFIL): local filename has zero length")
        raise SystemExit("OPNFIL: Empty filename")

    # Lines 125-129: Determine format type
    if form in UNFORMATTED_FORMATS:
        # Unformatted = binary mode
        mode = "rb+"
    else:
        # Formatted = text mode
        mode = "r+"

    # Line 130: Open file with error handling
    try:
        # Create file if it doesn't exist (status='unknown' behavior)
        if not os.path.exists(locfn):
            # Create empty file first
            open(locfn, "w").close()

        # Open in requested mode
        file_handle = open(locfn, mode)

        # Lines 134-136: Success message
        logger.info(f"(OPNFIL): Successfully opened file {locfn.strip()} " f"on unit= {iun}")

        return file_handle

    except IOError as e:
        # Lines 131-133: Error handling
        logger.error(
            f"(OPNFIL): failed to open file {locfn.strip()} " f"on unit {iun} ierr={e.errno}"
        )
        raise SystemExit(f"OPNFIL: Failed to open {locfn}")


def relavu(iunit: Optional[TextIO]) -> None:
    """Close and release file unit no longer in use.

    Translated from Fortran subroutine relavu (lines 158-173).

    In the original Fortran code, this function:
    1. Closes the Fortran unit
    2. Calls shr_file_freeUnit to release the unit number

    In Python/JAX context:
    - File handles are managed by Python's context managers
    - No explicit unit number management is needed
    - This function simply closes the file if it's open

    Args:
        iunit: File handle (Python file object) to close. If None, no action taken.

    Note:
        This is a compatibility function. In modern Python code, prefer using
        context managers (with statements) for file handling.

    Example:
        >>> f = open('data.txt', 'r')
        >>> # ... use file ...
        >>> relavu(f)

        Preferred modern approach:
        >>> with open('data.txt', 'r') as f:
        ...     # ... use file ...
        ...     pass  # Automatically closed
    """
    # Lines 171-172 from original Fortran
    if iunit is not None and not iunit.closed:
        iunit.close()

    # Line 172: call shr_file_freeUnit(iunit)
    # In Python, file handles are automatically managed by garbage collection
    # No explicit unit number freeing is needed


# =============================================================================
# TYPE ALIASES
# =============================================================================

FileHandle = object  # Generic file handle type for type hints


# =============================================================================
# NOTES ON UNTRANSLATED FUNCTIONS
# =============================================================================

# The following Fortran function was not translated as it's not needed in Python:
#
# - getavu: Gets next available Fortran unit number (lines 144-155)
#   In Python, file handles are managed directly without unit numbers.
#   Use open() directly or context managers (with statements).
#
# In a JAX-based implementation, file I/O should be handled outside
# the JIT-compiled computation graph, typically in the host Python code
# that calls the JAX functions. These operations use standard Python
# file I/O (open, close, etc.) or libraries like h5py, netCDF4, xarray, etc.
