"""
Patch data type module for CLM.

This module defines the patch data structure and Plant Functional Type (PFT)
classifications for the CLM vegetation model. Patches represent the finest
spatial resolution in the CLM hierarchy: gridcell → column → patch.

Translation of CLM-ml_v1/clm_src_main/PatchType.F90 to Python/JAX.
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp

# Import related modules
try:
    from .clm_varcon import ispval
    from .clm_varcon import spval as nan
except ImportError:
    # Provide fallback values
    ispval = -9999
    nan = -999.0

# Set up logger
logger = logging.getLogger(__name__)


class PFTType(IntEnum):
    """
    Plant Functional Type (PFT) enumeration.

    Defines all supported vegetation types with their integer codes.
    Values correspond to the original Fortran patch type definitions.
    """

    # Non-vegetated
    NOT_VEGETATED = 0

    # Trees - Needleleaf
    NEEDLELEAF_EVERGREEN_TEMPERATE_TREE = 1
    NEEDLELEAF_EVERGREEN_BOREAL_TREE = 2
    NEEDLELEAF_DECIDUOUS_BOREAL_TREE = 3

    # Trees - Broadleaf Evergreen
    BROADLEAF_EVERGREEN_TROPICAL_TREE = 4
    BROADLEAF_EVERGREEN_TEMPERATE_TREE = 5

    # Trees - Broadleaf Deciduous
    BROADLEAF_DECIDUOUS_TROPICAL_TREE = 6
    BROADLEAF_DECIDUOUS_TEMPERATE_TREE = 7
    BROADLEAF_DECIDUOUS_BOREAL_TREE = 8

    # Shrubs
    BROADLEAF_EVERGREEN_SHRUB = 9
    BROADLEAF_DECIDUOUS_TEMPERATE_SHRUB = 10
    BROADLEAF_DECIDUOUS_BOREAL_SHRUB = 11

    # Grasses
    C3_ARCTIC_GRASS = 12
    C3_NON_ARCTIC_GRASS = 13
    C4_GRASS = 14

    # Generic Crops
    C3_CROP = 15
    C3_IRRIGATED = 16

    # Corn
    TEMPERATE_CORN = 17
    IRRIGATED_TEMPERATE_CORN = 18

    # Wheat
    SPRING_WHEAT = 19
    IRRIGATED_SPRING_WHEAT = 20
    WINTER_WHEAT = 21
    IRRIGATED_WINTER_WHEAT = 22

    # Soybean
    TEMPERATE_SOYBEAN = 23
    IRRIGATED_TEMPERATE_SOYBEAN = 24

    # Barley
    BARLEY = 25
    IRRIGATED_BARLEY = 26
    WINTER_BARLEY = 27
    IRRIGATED_WINTER_BARLEY = 28

    # Rye
    RYE = 29
    IRRIGATED_RYE = 30
    WINTER_RYE = 31
    IRRIGATED_WINTER_RYE = 32

    # Other crops
    CASSAVA = 33
    IRRIGATED_CASSAVA = 34
    CITRUS = 35
    IRRIGATED_CITRUS = 36
    COCOA = 37
    IRRIGATED_COCOA = 38
    COFFEE = 39
    IRRIGATED_COFFEE = 40
    COTTON = 41
    IRRIGATED_COTTON = 42
    DATEPALM = 43
    IRRIGATED_DATEPALM = 44
    FODDERGRASS = 45
    IRRIGATED_FODDERGRASS = 46
    GRAPES = 47
    IRRIGATED_GRAPES = 48
    GROUNDNUTS = 49
    IRRIGATED_GROUNDNUTS = 50
    MILLET = 51
    IRRIGATED_MILLET = 52
    OILPALM = 53
    IRRIGATED_OILPALM = 54
    POTATOES = 55
    IRRIGATED_POTATOES = 56
    PULSES = 57
    IRRIGATED_PULSES = 58
    RAPESEED = 59
    IRRIGATED_RAPESEED = 60
    RICE = 61
    IRRIGATED_RICE = 62
    SORGHUM = 63
    IRRIGATED_SORGHUM = 64
    SUGARBEET = 65
    IRRIGATED_SUGARBEET = 66
    SUGARCANE = 67
    IRRIGATED_SUGARCANE = 68
    SUNFLOWER = 69
    IRRIGATED_SUNFLOWER = 70
    MISCANTHUS = 71
    IRRIGATED_MISCANTHUS = 72
    SWITCHGRASS = 73
    IRRIGATED_SWITCHGRASS = 74
    TROPICAL_CORN = 75
    IRRIGATED_TROPICAL_CORN = 76
    TROPICAL_SOYBEAN = 77
    IRRIGATED_TROPICAL_SOYBEAN = 78


class VegetationCategory(IntEnum):
    """Broader vegetation categories for classification."""

    NOT_VEGETATED = 0
    TREE = 1
    SHRUB = 2
    GRASS = 3
    CROP = 4


class PhotosynthesisType(IntEnum):
    """Photosynthesis pathway types."""

    NONE = 0  # Non-vegetated
    C3 = 3
    C4 = 4


class PhenologyType(IntEnum):
    """Phenology types."""

    NOT_VEGETATED = 0
    EVERGREEN = 1
    DECIDUOUS = 2


@dataclass
class patch_type:
    """
    Patch data type representing the finest spatial scale in CLM.

    This class manages patch-level data including vegetation types and
    hierarchical relationships to columns and gridcells.

    Attributes:
        column: Column indices for patch hierarchy mapping
        gridcell: Gridcell indices for patch hierarchy mapping
        itype: Vegetation type (PFT) for each patch
        begp: Beginning patch index
        endp: Ending patch index
        max_patches: Maximum number of patches supported
        metadata: Additional patch metadata
    """

    column: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))
    gridcell: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))
    itype: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=int))
    begp: int = 0
    endp: int = 0
    max_patches: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize patch arrays if not provided."""
        # Only initialize arrays if they are truly empty and max_patches > 0
        # Leave empty arrays as-is if max_patches is not being used for allocation
        pass

    def Init(self, begp: int, endp: int) -> None:
        """
        Initialize module data structure.

        Args:
            begp: Beginning patch index
            endp: Ending patch index

        Raises:
            ValueError: If patch indices are invalid
        """
        try:
            logger.debug(f"Initializing patch structure: begp={begp}, endp={endp}")

            if begp < 0 or endp < begp:
                raise ValueError(f"Invalid patch indices: begp={begp}, endp={endp}")

            # Store indices
            self.begp = begp
            self.endp = endp

            # Calculate required size
            num_patches = endp - begp + 1

            # Resize arrays if needed
            if num_patches > len(self.column):
                new_size = max(num_patches, self.max_patches)
                self.column = jnp.full(new_size, ispval, dtype=int)
                self.gridcell = jnp.full(new_size, ispval, dtype=int)
                self.itype = jnp.full(new_size, ispval, dtype=int)
            else:
                # Reset existing arrays
                self.column = self.column.at[:].set(ispval)
                self.gridcell = self.gridcell.at[:].set(ispval)
                self.itype = self.itype.at[:].set(ispval)

            # Update metadata
            # Record initialization metadata
            import time

            self.metadata.update(
                {
                    "initialized": True,
                    "begp": begp,
                    "endp": endp,
                    "num_patches": num_patches,
                    "initialization_time": time.time(),  # Unix timestamp of initialization
                }
            )

            logger.debug(f"Patch structure initialized successfully for {num_patches} patches")

        except Exception as e:
            logger.error(f"Failed to initialize patch structure: {e}")
            raise ValueError(f"Patch initialization failed: {e}") from e

    def is_valid(self) -> bool:
        """Check if patch structure is valid."""
        try:
            # Check basic structure
            if len(self.column) != len(self.gridcell) or len(self.column) != len(self.itype):
                return False

            # Check indices
            if self.begp < 0 or self.endp < self.begp:
                return False

            # Check array bounds
            num_patches = self.endp - self.begp + 1
            if num_patches > len(self.column):
                return False

            return True

        except Exception:
            return False

    def get_patch_info(self, patch_idx: int) -> Dict[str, Any]:
        """
        Get information about a specific patch.

        Args:
            patch_idx: Patch index (direct array index into column/gridcell/itype arrays)

        Returns:
            Dictionary with patch information

        Raises:
            ValueError: If patch index is invalid
        """
        if patch_idx < self.begp or patch_idx > self.endp:
            raise ValueError(f"Patch index {patch_idx} out of range [{self.begp}, {self.endp}]")

        # Use direct array indexing - patch arrays are indexed directly by patch_idx
        array_idx = patch_idx

        if array_idx < 0 or array_idx >= len(self.column):
            raise ValueError(f"Patch index {patch_idx} exceeds array bounds")

        pft = int(self.itype[array_idx])

        return {
            "patch_index": patch_idx,
            "array_index": array_idx,
            "column": int(self.column[array_idx]),
            "gridcell": int(self.gridcell[array_idx]),
            "pft_code": pft,
            "pft_name": get_pft_name(pft),
            "vegetation_category": get_vegetation_category(pft),
            "photosynthesis_type": get_photosynthesis_type(pft),
            "phenology_type": get_phenology_type(pft),
            "is_vegetated": is_vegetated(pft),
            "is_tree": is_tree(pft),
            "is_crop": is_crop(pft),
            "is_irrigated": is_irrigated(pft),
        }

    def get_active_patches(self) -> jnp.ndarray:
        """Get indices of active (valid) patches."""
        # For patches with sparse distribution, check from begp to endp
        if self.begp > 0 or self.endp >= self.begp:
            # Check the range from begp to endp+1
            start_idx = self.begp
            end_idx = self.endp + 1
            active_mask = self.itype[start_idx:end_idx] != ispval
            return jnp.where(active_mask)[0] + self.begp
        # For empty patches
        return jnp.array([], dtype=int)

    def resize(self, new_max_patches: int) -> None:
        """
        Resize patch arrays.

        Args:
            new_max_patches: New maximum number of patches
        """
        if new_max_patches < (self.endp - self.begp + 1):
            raise ValueError("Cannot resize to smaller than current patch count")

        # Create new arrays
        new_column = jnp.full(new_max_patches, ispval, dtype=int)
        new_gridcell = jnp.full(new_max_patches, ispval, dtype=int)
        new_itype = jnp.full(new_max_patches, ispval, dtype=int)

        # Copy existing data
        copy_size = min(len(self.column), new_max_patches)
        new_column = new_column.at[:copy_size].set(self.column[:copy_size])
        new_gridcell = new_gridcell.at[:copy_size].set(self.gridcell[:copy_size])
        new_itype = new_itype.at[:copy_size].set(self.itype[:copy_size])

        # Update arrays
        self.column = new_column
        self.gridcell = new_gridcell
        self.itype = new_itype
        self.max_patches = new_max_patches

        logger.debug(f"Patch arrays resized to {new_max_patches}")


# Global patch instance (for Fortran compatibility)
patch = patch_type()


# PFT classification functions
def get_pft_name(pft_code: int) -> str:
    """
    Get the name of a PFT from its code.

    Args:
        pft_code: PFT integer code

    Returns:
        PFT name string
    """
    try:
        pft = PFTType(pft_code)
        return pft.name.lower().replace("_", " ")
    except ValueError:
        return f"unknown_pft_{pft_code}"


def get_vegetation_category(pft_code: int) -> VegetationCategory:
    """
    Get the broad vegetation category for a PFT.

    Args:
        pft_code: PFT integer code

    Returns:
        VegetationCategory enum value
    """
    if pft_code == 0:
        return VegetationCategory.NOT_VEGETATED
    elif 1 <= pft_code <= 8:
        return VegetationCategory.TREE
    elif 9 <= pft_code <= 11:
        return VegetationCategory.SHRUB
    elif 12 <= pft_code <= 14:
        return VegetationCategory.GRASS
    elif pft_code >= 15:
        return VegetationCategory.CROP
    else:
        return VegetationCategory.NOT_VEGETATED


def get_photosynthesis_type(pft_code: int) -> PhotosynthesisType:
    """
    Get the photosynthesis type for a PFT.

    Args:
        pft_code: PFT integer code

    Returns:
        PhotosynthesisType enum value
    """
    if pft_code == 0:
        return PhotosynthesisType.NONE
    elif pft_code == 14:  # C4 grass
        return PhotosynthesisType.C4
    elif pft_code in [17, 18, 75, 76]:  # Corn (C4)
        return PhotosynthesisType.C4
    elif pft_code in [63, 64]:  # Sorghum (C4)
        return PhotosynthesisType.C4
    elif pft_code in [67, 68]:  # Sugarcane (C4)
        return PhotosynthesisType.C4
    elif pft_code in [71, 72, 73, 74]:  # Miscanthus, Switchgrass (C4)
        return PhotosynthesisType.C4
    else:
        return PhotosynthesisType.C3


def get_phenology_type(pft_code: int) -> PhenologyType:
    """
    Get the phenology type for a PFT.

    Args:
        pft_code: PFT integer code

    Returns:
        PhenologyType enum value
    """
    if pft_code == 0:
        return PhenologyType.NOT_VEGETATED

    # Evergreen types
    evergreen_pfts = [1, 2, 4, 5, 9]  # Needleleaf evergreen, broadleaf evergreen
    if pft_code in evergreen_pfts:
        return PhenologyType.EVERGREEN

    # Deciduous types
    deciduous_pfts = [3, 6, 7, 8, 10, 11]  # Deciduous trees and shrubs
    if pft_code in deciduous_pfts:
        return PhenologyType.DECIDUOUS

    # Grasses and crops are typically seasonal (deciduous-like)
    return PhenologyType.DECIDUOUS


def is_vegetated(pft_code: int) -> bool:
    """Check if PFT represents vegetated land."""
    return pft_code > 0


def is_tree(pft_code: int) -> bool:
    """Check if PFT represents a tree."""
    return 1 <= pft_code <= 8


def is_shrub(pft_code: int) -> bool:
    """Check if PFT represents a shrub."""
    return 9 <= pft_code <= 11


def is_grass(pft_code: int) -> bool:
    """Check if PFT represents grass."""
    return 12 <= pft_code <= 14


def is_crop(pft_code: int) -> bool:
    """Check if PFT represents a crop."""
    return pft_code >= 15


def is_irrigated(pft_code: int) -> bool:
    """Check if PFT represents an irrigated crop."""
    # Irrigated crops have even numbers starting from 16
    return pft_code >= 16 and (pft_code % 2 == 0)


def is_c3_plant(pft_code: int) -> bool:
    """Check if PFT uses C3 photosynthesis."""
    return get_photosynthesis_type(pft_code) == PhotosynthesisType.C3


def is_c4_plant(pft_code: int) -> bool:
    """Check if PFT uses C4 photosynthesis."""
    return get_photosynthesis_type(pft_code) == PhotosynthesisType.C4


def get_pft_statistics(patch_instance: patch_type) -> Dict[str, Any]:
    """
    Calculate statistics for PFTs in a patch instance.

    Args:
        patch_instance: Patch type instance

    Returns:
        Dictionary with PFT statistics
    """
    active_patches = patch_instance.get_active_patches()

    if len(active_patches) == 0:
        return {
            "num_patches": 0,
            "num_vegetated": 0,
            "num_trees": 0,
            "num_crops": 0,
            "num_c3": 0,
            "num_c4": 0,
            "unique_pfts": [],
        }

    # Get PFT codes for active patches
    pft_codes = []
    for patch_idx in active_patches:
        # Use direct array indexing - patch arrays are indexed directly by patch_idx
        array_idx = patch_idx
        if 0 <= array_idx < len(patch_instance.itype):
            pft_codes.append(int(patch_instance.itype[array_idx]))

    pft_codes = jnp.array(pft_codes)

    # Calculate statistics
    num_vegetated = jnp.sum(pft_codes > 0)
    num_trees = jnp.sum((pft_codes >= 1) & (pft_codes <= 8))
    num_crops = jnp.sum(pft_codes >= 15)

    # Count C3/C4
    num_c3 = sum(1 for pft in pft_codes if is_c3_plant(pft))
    num_c4 = sum(1 for pft in pft_codes if is_c4_plant(pft))

    return {
        "num_patches": len(active_patches),
        "num_vegetated": int(num_vegetated),
        "num_trees": int(num_trees),
        "num_crops": int(num_crops),
        "num_c3": num_c3,
        "num_c4": num_c4,
        "unique_pfts": list(jnp.unique(pft_codes)),
    }


# Utility functions
def create_patch_instance(begp: int = 0, endp: int = 0, max_patches: int = 1000) -> patch_type:
    """
    Create a new patch instance.

    Args:
        begp: Beginning patch index
        endp: Ending patch index
        max_patches: Maximum number of patches

    Returns:
        New patch_type instance
    """
    instance = patch_type(max_patches=max_patches)
    if endp > 0:
        instance.Init(begp, endp)
    return instance


def reset_global_patch() -> None:
    """Reset the global patch instance."""
    global patch
    patch = patch_type()
    logger.info("Global patch instance reset")


def validate_patch_structure(patch_instance: patch_type) -> Tuple[bool, str]:
    """
    Validate patch structure consistency.

    Args:
        patch_instance: Patch instance to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not patch_instance.is_valid():
        return False, "Basic patch structure validation failed"

    # Check for valid PFT codes
    active_patches = patch_instance.get_active_patches()
    for patch_idx in active_patches:
        try:
            patch_info = patch_instance.get_patch_info(patch_idx)
            pft_code = patch_info["pft_code"]

            # Check PFT code range
            if pft_code < 0 or pft_code > 78:
                return False, f"Invalid PFT code {pft_code} at patch {patch_idx}"

        except Exception as e:
            return False, f"Error validating patch {patch_idx}: {e}"

    return True, "Patch structure is valid"


def print_patch_summary(patch_instance: patch_type) -> None:
    """
    Print a summary of patch structure and PFT distribution.

    Args:
        patch_instance: Patch instance to summarize
    """
    print("\n=== Patch Structure Summary ===")
    print(f"Patch range: [{patch_instance.begp}, {patch_instance.endp}]")
    print(f"Max patches: {patch_instance.max_patches}")
    print(f"Valid structure: {patch_instance.is_valid()}")

    # Get statistics
    stats = get_pft_statistics(patch_instance)
    print("\nPFT Statistics:")
    for key, value in stats.items():
        if key != "unique_pfts":
            print(f"  {key}: {value}")

    if stats["unique_pfts"]:
        print(f"  Unique PFTs: {stats['unique_pfts'][:10]}...")  # Show first 10

    # Show some example patches
    active_patches = patch_instance.get_active_patches()
    if len(active_patches) > 0:
        print("\nExample patches:")
        for i, patch_idx in enumerate(active_patches[:3]):  # Show first 3
            try:
                info = patch_instance.get_patch_info(patch_idx)
                print(f"  Patch {patch_idx}: {info['pft_name']} (PFT {info['pft_code']})")
            except Exception as e:
                print(f"  Patch {patch_idx}: Error - {e}")

    if patch_instance.metadata:
        print("\nMetadata:")
        for key, value in patch_instance.metadata.items():
            print(f"  {key}: {value}")

    print("=" * 35)


# Factory functions
def create_single_pft_patches(pft_code: int, num_patches: int = 1) -> patch_type:
    """
    Create patches with a single PFT type.

    Args:
        pft_code: PFT code to assign to all patches
        num_patches: Number of patches to create

    Returns:
        Patch instance with specified PFT
    """
    patch_instance = create_patch_instance(0, num_patches - 1, num_patches)

    # Set all patches to the specified PFT
    for i in range(num_patches):
        patch_instance.itype = patch_instance.itype.at[i].set(pft_code)
        patch_instance.column = patch_instance.column.at[i].set(1)  # Default column
        patch_instance.gridcell = patch_instance.gridcell.at[i].set(1)  # Default gridcell

    return patch_instance


def create_mixed_vegetation_patches(pft_codes: List[int]) -> patch_type:
    """
    Create patches with mixed vegetation types.

    Args:
        pft_codes: List of PFT codes

    Returns:
        Patch instance with mixed PFTs
    """
    num_patches = len(pft_codes)
    patch_instance = create_patch_instance(0, num_patches - 1, num_patches)

    # Set PFT for each patch
    for i, pft_code in enumerate(pft_codes):
        patch_instance.itype = patch_instance.itype.at[i].set(pft_code)
        patch_instance.column = patch_instance.column.at[i].set(1)  # Default column
        patch_instance.gridcell = patch_instance.gridcell.at[i].set(1)  # Default gridcell

    return patch_instance


# Export interface
__all__ = [
    "patch_type",
    "patch",  # Global instance
    "PFTType",
    "VegetationCategory",
    "PhotosynthesisType",
    "PhenologyType",
    "get_pft_name",
    "get_vegetation_category",
    "get_photosynthesis_type",
    "get_phenology_type",
    "is_vegetated",
    "is_tree",
    "is_shrub",
    "is_grass",
    "is_crop",
    "is_irrigated",
    "is_c3_plant",
    "is_c4_plant",
    "get_pft_statistics",
    "create_patch_instance",
    "reset_global_patch",
    "validate_patch_structure",
    "print_patch_summary",
    "create_single_pft_patches",
    "create_mixed_vegetation_patches",
]
