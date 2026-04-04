# legoESM Ocean Boundary Conditions Analysis Summary

## How We Got Here

### **Initial Questions**
1. **Grid types in legoESM**: Found 4 supported grids (cubed-sphere, lat-lon, MPAS, spectral)
2. **Grid design philosophy**: Discovered flexible coupling - components CAN use same grid (shared operators) OR different grids (via regridding)
3. **Ocean model type**: Confirmed finite volume (not finite difference) using PPM transport
4. **Boundary conditions**: Identified current approach and potential issues

### **Key Discovery: Boundary Treatment Inconsistency**

**Current legoESM approach** (similar to many ocean models):
- **Post-hoc masking**: Compute pressure gradients first, then multiply by land mask
- **Neumann fill workaround**: Fill land cells with neighbor values to avoid sharp gradients
- **Conservation fixes**: Apply global corrections with masking

**MITgcm approach** (mathematically superior):
- **Baked-in masking**: Land boundaries never contribute to flux calculations
- **hFac system**: Fractional face areas (hFacW, hFacS) multiply every flux
- **Rest state preservation**: Zero flux through land boundaries by construction

### **Identified Problems**
Claude Chat highlighted failure modes in legoESM's approach:
1. **Corner cell inconsistencies** at diagonal coastlines
2. **Metric amplification** of tiny gradients near poles (1/cos²(lat) factor)
3. **Ordering dependencies** between Neumann fill and conservation fixes
4. **Not mathematically rigorous** - patches a fundamental issue rather than solving it

---

## Current Status

### **legoESM Ocean Architecture**
- **Finite volume model** with PPM reconstruction
- **Multiple grid support**: cubed-sphere, lat-lon, MPAS, spectral
- **Free surface** with split-explicit barotropic/baroclinic stepping
- **Land masking**: Binary mask (1=ocean, 0=land) applied post-computation
- **Boundary treatment**: Neumann fill + post-hoc masking

### **Evidence of Issues**
- Small but persistent eta drift in rest state tests (~1e-6 m bands in lat-lon)
- Current approach works "reasonably well" but not mathematically exact
- Potential for amplification in high-resolution or long-term runs

---

## Proposed Solution: MITgcm-Style Refactor

### **Core Concept**
Replace **post-hoc masking** with **baked-in masking** using MITgcm's hFac approach.

### **Implementation Plan**

#### **Phase 1: Infrastructure**
1. **Add hFac arrays to grid structures**:
   ```python
   class LatLonOceanGrid:
       hFacC: jnp.ndarray  # Cell center mask (n_lat, n_lon)
       hFacW: jnp.ndarray  # West face mask (n_lat, n_lon+1)  
       hFacS: jnp.ndarray  # South face mask (n_lat+1, n_lon)
   ```

2. **Create hFac computation from land mask**:
   ```python
   def compute_hfac_from_mask(land_mask):
       # Face masks = 1 only if BOTH adjacent cells are ocean
       hFacW = jnp.minimum(land_mask_west_neighbors)
       hFacS = jnp.minimum(land_mask_south_neighbors)
   ```

#### **Phase 2: Operator Refactor**
1. **Modify flux computations** in `latlon_operators.py`:
   - Multiply every flux by appropriate hFac during computation
   - Remove all post-hoc masking steps

2. **Update pressure gradient** and momentum operators:
   - Bake hFac into face area calculations
   - Ensure zero flux through land boundaries by construction

3. **Extend to other grids**:
   - Cubed-sphere: Adapt hFac to face-based structure
   - MPAS: Use edge connectivity for hFac computation
   - Spectral: May need different approach due to global nature

#### **Phase 3: Cleanup**
1. **Remove Neumann fill** (`_neumann_fill_latlon`)
2. **Remove post-hoc masking** throughout ocean code:
   - `conservation.py`: `* mask` operations
   - All dynamics modules: masking after gradient computation
3. **Simplify conservation fixers** - no masking needed

#### **Phase 4: Validation**
1. **Rest state tests**: Should achieve machine precision (no drift)
2. **Conservation verification**: Volume, heat, salt exactly conserved
3. **Cross-grid comparison**: Ensure hFac approach works on all grids
4. **Performance benchmarking**: Should be faster (no Neumann fill iterations)

### **Benefits**
- **Mathematically rigorous**: Zero spurious pressure forces by construction
- **JAX-friendly**: Pure array operations, fully differentiable
- **Performance**: Eliminates iterative Neumann fill
- **Maintainability**: Single source of truth for boundary treatment
- **Exact rest states**: Machine precision preservation

### **Risks/Considerations**
- **Significant refactor**: Touches core operators across all ocean grids
- **Testing burden**: Must verify all existing ocean tests still pass
- **Spectral grid challenges**: Global nature may need special handling
- **Backward compatibility**: May change numerical results slightly

---

## Next Steps

1. **Start with lat-lon grid**: Simplest case, well-defined face structure
2. **Implement hFac infrastructure**: Grid structures and computation functions
3. **Refactor one operator at a time**: Begin with flux_divergence_latlon
4. **Validate incrementally**: Test rest state preservation after each change
5. **Extend to other grids**: Once lat-lon approach is proven
6. **Performance comparison**: Benchmark against current Neumann fill approach

This refactor would bring legoESM's ocean boundary treatment up to **MITgcm's mathematical rigor** while maintaining **JAX compatibility** and **multi-grid flexibility**.

---

## Background Context

### **Grid Types in legoESM**
- **Cubed-sphere**: Gnomonic equidistant projection, 6 faces, no polar singularities
- **Lat-lon**: Traditional regular grid, periodic longitude, wall boundaries at poles
- **MPAS**: Unstructured Voronoi mesh, variable resolution capability
- **Spectral**: Spherical harmonics, Gaussian quadrature points, global coupling

### **Comparison with Other Models**
- **NeuralGCM**: Uses Gaussian grid with spherical harmonics (different from legoESM's multi-grid approach)
- **Traditional ESMs (CESM)**: Forced to use different grids for atmosphere/ocean, requiring complex coupling
- **legoESM advantage**: Grid flexibility - components can use same or different grids as needed

### **Design Philosophy**
legoESM emphasizes:
- **End-to-end differentiability** (`jax.grad` compatibility)
- **Conservation** (mass, energy, momentum)
- **Multi-grid flexibility** (research-friendly)
- **JAX-native implementation** (JIT, autodiff, parallelism)

The proposed hFac refactor aligns with these principles by providing mathematically sound boundary treatment that's fully differentiable and efficient in JAX.