"""
Subgrid structure initialization module for CLM.

This module provides lower-level routines for initializing the subgrid structure,
including patch-level array management and subgrid hierarchy setup.

Translation of CLM-ml_v1/clm_src_main/initSubgridMod.F90 to Python/JAX.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

# Import related modules (these would need to be implemented)
try:
    from .PatchType import patch
except ImportError:
    # Provide fallback implementation for testing
    @dataclass
    class PatchData:
        """Fallback patch data structure."""

        column: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))
        gridcell: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))
        itype: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))

        def resize(self, new_size: int) -> None:
            """Resize arrays to accommodate new patches."""
            current_size = len(self.column)
            if new_size > current_size:
                # Extend arrays
                pad_size = new_size - current_size
                self.column = jnp.pad(self.column, (0, pad_size), constant_values=-1)
                self.gridcell = jnp.pad(self.gridcell, (0, pad_size), constant_values=-1)
                self.itype = jnp.pad(self.itype, (0, pad_size), constant_values=-1)

    # Create global patch instance
    patch = PatchData()

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class PatchData:
    """
    Simple dataclass to hold patch data arrays.

    Attributes:
        column: Array of column indices for each patch
        gridcell: Array of gridcell indices for each patch
        itype: Array of patch types (PFTs)
    """

    column: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))
    gridcell: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))
    itype: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))

    def resize(self, new_size: int):
        """Resize arrays to new size."""
        self.column = jnp.zeros(new_size, dtype=int)
        self.gridcell = jnp.zeros(new_size, dtype=int)
        self.itype = jnp.zeros(new_size, dtype=int)


@dataclass
class SubgridStructure:
    """
    Manages subgrid structure state and operations.

    This class tracks the hierarchical relationship between gridcells,
    columns, and patches in the CLM subgrid structure.

    Attributes:
        max_patches: Maximum number of patches allowed
        current_patches: Current number of active patches
        patch_columns: Array mapping patches to columns
        patch_gridcells: Array mapping patches to gridcells
        patch_types: Array of patch types (PFTs)
        active_mask: Boolean mask for active patches
        metadata: Additional structure metadata
    """

    max_patches: int = 1000
    current_patches: int = 0
    patch_columns: jnp.ndarray = field(default_factory=lambda: jnp.zeros(1000, dtype=int))
    patch_gridcells: jnp.ndarray = field(default_factory=lambda: jnp.zeros(1000, dtype=int))
    patch_types: jnp.ndarray = field(default_factory=lambda: jnp.zeros(1000, dtype=int))
    active_mask: jnp.ndarray = field(default_factory=lambda: jnp.zeros(1000, dtype=bool))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize arrays to correct size."""
        if len(self.patch_columns) != self.max_patches:
            self.patch_columns = jnp.zeros(self.max_patches, dtype=int)
            self.patch_gridcells = jnp.zeros(self.max_patches, dtype=int)
            self.patch_types = jnp.zeros(self.max_patches, dtype=int)
            self.active_mask = jnp.zeros(self.max_patches, dtype=bool)

    def is_valid(self) -> bool:
        """Check if subgrid structure is valid."""
        # Check array sizes
        if not all(
            len(arr) == self.max_patches
            for arr in [
                self.patch_columns,
                self.patch_gridcells,
                self.patch_types,
                self.active_mask,
            ]
        ):
            return False

        # Check current patches count
        if self.current_patches < 0 or self.current_patches > self.max_patches:
            return False

        # Check active mask consistency
        active_count = jnp.sum(self.active_mask)
        if active_count != self.current_patches:
            return False

        return True

    def get_active_patches(self) -> jnp.ndarray:
        """Get indices of active patches."""
        return jnp.where(self.active_mask)[0]

    def get_patch_info(self, patch_idx: int) -> Dict[str, Any]:
        """Get information about a specific patch."""
        if patch_idx < 0 or patch_idx >= self.max_patches:
            raise ValueError(f"Invalid patch index: {patch_idx}")

        return {
            "patch_index": patch_idx,
            "column": int(self.patch_columns[patch_idx]),
            "gridcell": int(self.patch_gridcells[patch_idx]),
            "patch_type": int(self.patch_types[patch_idx]),
            "is_active": bool(self.active_mask[patch_idx]),
        }

    def resize(self, new_max_patches: int) -> "SubgridStructure":
        """Create a new SubgridStructure with different maximum patch count."""
        if new_max_patches < self.current_patches:
            raise ValueError("Cannot resize to smaller than current patch count")

        new_structure = SubgridStructure(max_patches=new_max_patches)

        # Copy existing data
        copy_size = min(self.max_patches, new_max_patches)
        new_structure.patch_columns = new_structure.patch_columns.at[:copy_size].set(
            self.patch_columns[:copy_size]
        )
        new_structure.patch_gridcells = new_structure.patch_gridcells.at[:copy_size].set(
            self.patch_gridcells[:copy_size]
        )
        new_structure.patch_types = new_structure.patch_types.at[:copy_size].set(
            self.patch_types[:copy_size]
        )
        new_structure.active_mask = new_structure.active_mask.at[:copy_size].set(
            self.active_mask[:copy_size]
        )

        new_structure.current_patches = self.current_patches
        new_structure.metadata = self.metadata.copy()

        return new_structure


# Global subgrid structure
_subgrid_structure = SubgridStructure()


def add_patch(pi: int, ptype: int) -> int:
    """
    Add an entry in the patch-level arrays.

    pi gives the index of the last patch added; the new patch is added at pi+1;
    and the pi argument is incremented accordingly. The input value of pi is the
    index of last patch added, and the output value is the index of the
    newly-added patch.

    This implementation processes a single-patch configuration (one grid cell with
    one column and one patch), and the subgrid patch structure is set accordingly.
    This is the standard configuration for tower site simulations.

    Args:
        pi: patch index (input: index of last patch added, output: index of newly-added patch)
        ptype: patch type (PFT)

    Returns:
        New patch index (pi + 1)

    Raises:
        ValueError: If patch type is invalid or maximum patches exceeded
        RuntimeError: If patch addition fails
    """
    global _subgrid_structure, patch

    try:
        logger.debug(f"Adding patch at index {pi + 1} with type {ptype}")

        # Validate inputs
        if ptype < 0:
            raise ValueError(f"Invalid patch type: {ptype}")

        # Calculate new patch index
        new_pi = pi + 1

        # Check if we need to resize arrays
        if new_pi >= _subgrid_structure.max_patches:
            logger.info(f"Resizing subgrid structure to accommodate patch {new_pi}")
            new_size = max(_subgrid_structure.max_patches * 2, new_pi + 100)
            _subgrid_structure = _subgrid_structure.resize(new_size)

        # Update global patch structure (Fortran compatibility)
        # Ensure patch arrays are large enough
        if hasattr(patch, "resize") and len(patch.column) <= new_pi:
            patch.resize(new_pi + 100)

        # Set patch properties (0-based indexing — Python/JAX arrays start at 0)
        if hasattr(patch, "column"):
            patch.column = patch.column.at[new_pi].set(0)  # Column 0 (0-based)
            patch.gridcell = patch.gridcell.at[new_pi].set(0)  # Gridcell 0 (0-based)
            patch.itype = patch.itype.at[new_pi].set(ptype)

        # Update subgrid structure
        _subgrid_structure.patch_columns = _subgrid_structure.patch_columns.at[new_pi].set(1)
        _subgrid_structure.patch_gridcells = _subgrid_structure.patch_gridcells.at[new_pi].set(1)
        _subgrid_structure.patch_types = _subgrid_structure.patch_types.at[new_pi].set(ptype)
        _subgrid_structure.active_mask = _subgrid_structure.active_mask.at[new_pi].set(True)
        _subgrid_structure.current_patches += 1

        # Update metadata
        _subgrid_structure.metadata.update(
            {
                "last_added_patch": new_pi,
                "last_added_type": ptype,
                "total_patches_added": _subgrid_structure.current_patches,
            }
        )

        logger.debug(f"Successfully added patch {new_pi} with type {ptype}")

        return new_pi

    except Exception as e:
        logger.error(f"Failed to add patch: {e}")
        raise RuntimeError(f"Patch addition failed: {e}") from e


def validate_patch_hierarchy(
    patch_columns: jnp.ndarray,
    patch_gridcells: jnp.ndarray,
    patch_types: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> bool:
    """
    Validate the patch hierarchy structure.

    Args:
        patch_columns: Array of patch-to-column mappings
        patch_gridcells: Array of patch-to-gridcell mappings
        patch_types: Array of patch types
        active_mask: Boolean mask for active patches

    Returns:
        True if hierarchy is valid, False otherwise
    """
    # Check for positive indices where active
    valid_columns = jnp.all(jnp.where(active_mask, patch_columns > 0, True))
    valid_gridcells = jnp.all(jnp.where(active_mask, patch_gridcells > 0, True))
    valid_types = jnp.all(jnp.where(active_mask, patch_types >= 0, True))

    # Convert JAX boolean to Python bool
    result = valid_columns & valid_gridcells & valid_types
    return bool(result)


@jax.jit
def get_patch_statistics(patch_types: jnp.ndarray, active_mask: jnp.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for active patches.

    Args:
        patch_types: Array of patch types
        active_mask: Boolean mask for active patches

    Returns:
        Dictionary with patch statistics (values are JAX scalars)
    """
    # Use jnp.where instead of boolean indexing for JIT compatibility
    # Assign -1 to inactive patches, then calculate stats only on valid (>= 0) values
    active_types = jnp.where(active_mask, patch_types, -1)

    # Count valid types (>= 0)
    valid_mask = active_types >= 0
    num_valid = jnp.sum(valid_mask)

    # Use jnp.where to handle empty case without branching
    # If no valid types, return 0 for all stats
    min_val = jnp.where(num_valid > 0, jnp.min(jnp.where(valid_mask, active_types, jnp.inf)), 0.0)
    max_val = jnp.where(num_valid > 0, jnp.max(jnp.where(valid_mask, active_types, -jnp.inf)), 0.0)
    mean_val = jnp.where(
        num_valid > 0, jnp.sum(jnp.where(valid_mask, active_types, 0)) / num_valid, 0.0
    )

    # Return JAX scalars directly (don't use float())
    return {
        "num_active_patches": jnp.float64(num_valid),
        "min_patch_type": jnp.float64(min_val),
        "max_patch_type": jnp.float64(max_val),
        "mean_patch_type": jnp.float64(mean_val),
    }


def get_subgrid_structure() -> SubgridStructure:
    """
    Get the current subgrid structure.

    Returns:
        Copy of current subgrid structure
    """
    global _subgrid_structure
    return _subgrid_structure


def reset_subgrid_structure(max_patches: int = 1000) -> None:
    """
    Reset the subgrid structure.

    Args:
        max_patches: Maximum number of patches for new structure
    """
    global _subgrid_structure, patch
    _subgrid_structure = SubgridStructure(max_patches=max_patches)

    # Reset global patch structure if available
    if hasattr(patch, "resize"):
        # First reset the patch indices to allow any resize
        patch.begp = 0
        patch.endp = -1
        patch.num_patches = 0
        # Now resize
        patch.resize(max_patches)
        patch.column = jnp.zeros(max_patches, dtype=int)
        patch.gridcell = jnp.zeros(max_patches, dtype=int)
        patch.itype = jnp.zeros(max_patches, dtype=int)

    logger.info(f"Subgrid structure reset with max_patches={max_patches}")


def create_simple_subgrid(
    patch_types: List[int],
    column_assignments: Optional[List[int]] = None,
    gridcell_assignments: Optional[List[int]] = None,
) -> SubgridStructure:
    """
    Create a simple subgrid structure for testing.

    Args:
        patch_types: List of patch types to create
        column_assignments: Optional column assignments (defaults to 1 for all)
        gridcell_assignments: Optional gridcell assignments (defaults to 1 for all)

    Returns:
        SubgridStructure with specified patches

    Raises:
        ValueError: If input arrays have mismatched lengths
    """
    num_patches = len(patch_types)

    if column_assignments is None:
        column_assignments = [1] * num_patches
    if gridcell_assignments is None:
        gridcell_assignments = [1] * num_patches

    if len(column_assignments) != num_patches or len(gridcell_assignments) != num_patches:
        raise ValueError("All input arrays must have the same length")

    # Create structure with enough space
    structure = SubgridStructure(max_patches=max(num_patches + 10, 100))

    # Add patches
    for i in range(num_patches):
        structure.patch_columns = structure.patch_columns.at[i].set(column_assignments[i])
        structure.patch_gridcells = structure.patch_gridcells.at[i].set(gridcell_assignments[i])
        structure.patch_types = structure.patch_types.at[i].set(patch_types[i])
        structure.active_mask = structure.active_mask.at[i].set(True)

    structure.current_patches = num_patches
    structure.metadata["creation_type"] = "simple_subgrid"

    return structure


def add_multiple_patches(patch_info: List[Tuple[int, int, int]]) -> List[int]:
    """
    Add multiple patches in sequence.

    Args:
        patch_info: List of tuples (column, gridcell, ptype) for each patch

    Returns:
        List of patch indices for added patches

    Raises:
        ValueError: If patch info is invalid
        RuntimeError: If any patch addition fails
    """
    patch_indices = []
    current_pi = _subgrid_structure.current_patches - 1  # Last added patch index

    try:
        for column, gridcell, ptype in patch_info:
            # Validate inputs
            if column <= 0 or gridcell <= 0 or ptype < 0:
                raise ValueError(
                    f"Invalid patch info: column={column}, gridcell={gridcell}, ptype={ptype}"
                )

            # Add patch (this updates current_pi)
            current_pi = add_patch(current_pi, ptype)

            # Update column and gridcell if different from default (1)
            if column != 1:
                _subgrid_structure.patch_columns = _subgrid_structure.patch_columns.at[
                    current_pi
                ].set(column)
                if hasattr(patch, "column"):
                    patch.column = patch.column.at[current_pi].set(column)

            if gridcell != 1:
                _subgrid_structure.patch_gridcells = _subgrid_structure.patch_gridcells.at[
                    current_pi
                ].set(gridcell)
                if hasattr(patch, "gridcell"):
                    patch.gridcell = patch.gridcell.at[current_pi].set(gridcell)

            patch_indices.append(current_pi)

        logger.info(f"Successfully added {len(patch_info)} patches")
        return patch_indices

    except Exception as e:
        logger.error(f"Failed to add multiple patches: {e}")
        raise RuntimeError(f"Multiple patch addition failed: {e}") from e


def print_subgrid_summary() -> None:
    """
    Print a summary of the current subgrid structure.
    """
    global _subgrid_structure

    print("\n=== Subgrid Structure Summary ===")
    print(f"Max patches: {_subgrid_structure.max_patches}")
    print(f"Current patches: {_subgrid_structure.current_patches}")
    print(f"Valid structure: {_subgrid_structure.is_valid()}")

    if _subgrid_structure.current_patches > 0:
        active_indices = _subgrid_structure.get_active_patches()
        print(f"Active patch indices: {active_indices[:min(10, len(active_indices))]}...")

        # Get statistics
        stats = get_patch_statistics(_subgrid_structure.patch_types, _subgrid_structure.active_mask)
        print("Patch Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")

    if _subgrid_structure.metadata:
        print("\nMetadata:")
        for key, value in _subgrid_structure.metadata.items():
            print(f"  {key}: {value}")

    print("=" * 35)


def validate_subgrid_consistency() -> Tuple[bool, str]:
    """
    Validate subgrid structure consistency.

    Returns:
        Tuple of (is_valid, error_message)
    """
    global _subgrid_structure

    # Basic structure validation
    if not _subgrid_structure.is_valid():
        return False, "Subgrid structure basic validation failed"

    # Hierarchy validation
    hierarchy_valid = validate_patch_hierarchy(
        _subgrid_structure.patch_columns,
        _subgrid_structure.patch_gridcells,
        _subgrid_structure.patch_types,
        _subgrid_structure.active_mask,
    )

    if not hierarchy_valid:
        return False, "Patch hierarchy validation failed"

    # Check for duplicate active patches at same location
    if _subgrid_structure.current_patches > 1:
        active_mask = _subgrid_structure.active_mask
        active_columns = _subgrid_structure.patch_columns[active_mask]
        active_gridcells = _subgrid_structure.patch_gridcells[active_mask]

        # Create location tuples and check for duplicates
        locations = list(zip(active_columns, active_gridcells))
        if len(locations) != len(set(locations)):
            return False, "Duplicate patches found at same gridcell/column location"

    return True, "Subgrid structure is valid and consistent"


# Factory functions for common subgrid patterns
def create_single_patch_subgrid(ptype: int = 1) -> None:
    """
    Create a simple single-patch subgrid.

    Args:
        ptype: Patch type for the single patch
    """
    reset_subgrid_structure()
    add_patch(-1, ptype)  # Start with pi=-1, so new patch gets index 0
    logger.info(f"Created single patch subgrid with type {ptype}")


def create_multi_column_subgrid(patch_configs: List[Tuple[int, int]]) -> None:
    """
    Create a multi-column subgrid structure.

    Args:
        patch_configs: List of (column, ptype) tuples
    """
    reset_subgrid_structure()

    patch_info = [(col, 1, ptype) for col, ptype in patch_configs]  # All in gridcell 1
    add_multiple_patches(patch_info)

    logger.info(f"Created multi-column subgrid with {len(patch_configs)} patches")


# Export interface
__all__ = [
    "add_patch",
    "SubgridStructure",
    "PatchData",
    "get_subgrid_structure",
    "reset_subgrid_structure",
    "create_simple_subgrid",
    "add_multiple_patches",
    "validate_patch_hierarchy",
    "get_patch_statistics",
    "print_subgrid_summary",
    "validate_subgrid_consistency",
    "create_single_patch_subgrid",
    "create_multi_column_subgrid",
]
