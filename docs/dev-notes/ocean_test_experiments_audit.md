# Ocean Test Experiments Audit List

## Complete List of Ocean Test Experiments

### 1. **rest_state** 
- **Description**: Rest-state adjustment (stability check)
- **Purpose**: Test numerical stability when ocean should remain at rest
- **Status**: ✅ Reviewed, FIXED, and VALIDATED
- **Comments**: 
  - **Initial conditions**: T: 20°C→2°C exponential profile, S: 35 PSU uniform, u=v=eta=0
  - **Domain**: Global with land at |lat|>80°, uniform 5500m depth
  - **EOS**: Wright (1997) - creates density stratification from T profile
  - **Issue identified**: Mixing parameterizations active (K_h=1000, K_v=0.0001 m²/s) will cause drift
  - **10-day test results (BEFORE fix)**:
    - **cubed_sphere**: eta drift=8.40e-11, T drift=2.77e-04
    - **latlon**: eta drift=1.58e-09, T drift=2.26e-04  
    - **mpas**: eta drift=3.62e-09, T drift=2.64e-04
    - **spectral**: eta drift=6.38e+03, T drift=2.51e-04 ⚠️ (using spectral coefficients!)
  - **MAJOR BUG FOUND**: Spectral grid was using spectral coefficient units instead of physical units
  - **FIX APPLIED**: Modified `_make_scalar_fn()` to use `sh_synthesis()` to convert spectral coefficients to physical fields
  - **10-day test results (AFTER fix)**:
    - **cubed_sphere**: eta drift=8.40e-11, T drift=2.77e-04
    - **latlon**: eta drift=1.58e-09, T drift=2.26e-04  
    - **mpas**: eta drift=3.62e-09, T drift=2.64e-04
    - **spectral**: eta drift=4.80e-13, T drift=2.51e-04 ✅ (now in physical units!)
  - **Analysis**: T drift ~2.5e-04 consistent across all grids (expected from mixing). **Spectral grid now shows EXCELLENT eta drift** - actually the best of all grids!
  - **Cross-script validation (2026-04-03)**: Verified that current and old scripts produce identical numerical results, with visual differences due to simulation duration (initialization transients vs long-term cube face artifacts)
  - **Final validation**: 1-day runs show eta drift=1.63e-11 (cubed_sphere), 6.85e-11 (latlon), 2.31e-10 (mpas), 1.39e-13 (spectral) - all excellent
  - **Verdict**: ✅ Test working correctly for all grids. Cross-grid comparison scientifically meaningful. Old script removed to avoid confusion.

### 2. **barotropic_wave**
- **Description**: Gaussian SSH perturbation propagation  
- **Purpose**: Test barotropic gravity wave propagation
- **Status**: ✅ REFACTORED and VALIDATED
- **Comments**: 
  - **Modular implementation**: `src/legoesm/ocean/experiments/barotropic_wave.py`
  - **Configuration**: BarotropicWaveConfig with 1m SSH amplitude, 10° width, centered at (180°E, 0°N)
  - **Validation**: Wave amplitude preservation, conservation metrics, L2 analysis
  - **Grid support**: All grids (cubed_sphere, latlon, mpas, spectral)
  - **Test results**: PASS across all grids with expected wave propagation

### 3. **wind_gyre**
- **Description**: Wind-driven double-gyre circulation
- **Purpose**: Test wind forcing and circulation development  
- **Status**: ✅ REFACTORED and VALIDATED
- **Comments**:
  - **Modular implementation**: `src/legoesm/ocean/experiments/wind_gyre.py`
  - **Configuration**: WindGyreConfig with dedicated gyre initialization
  - **Validation**: Circulation development, SSH drift, speed metrics
  - **Grid support**: cubed_sphere, latlon, mpas (spectral not implemented)
  - **Test results**: PASS with proper gyre circulation development

### 4. **baroclinic** 
- **Description**: Meridional temperature front relaxation
- **Purpose**: Test baroclinic adjustment processes
- **Status**: ✅ REFACTORED and VALIDATED
- **Comments**:
  - **Modular implementation**: `src/legoesm/ocean/experiments/baroclinic.py`
  - **Configuration**: BaroclinicConfig with ±5°C meridional temperature gradient
  - **Validation**: Temperature drift, circulation development, adjustment metrics
  - **Grid support**: All grids (cubed_sphere, latlon, mpas, spectral)
  - **Test results**: PASS with expected thermal wind adjustment

### 5. **phillips_two_layer**
- **Description**: Phillips 2-layer baroclinic instability
- **Purpose**: Test baroclinic instability dynamics
- **Reference**: Phillips (1954) model
- **Status**: ✅ REFACTORED and VALIDATED
- **Comments**:
  - **Modular implementation**: `src/legoesm/ocean/experiments/phillips_two_layer.py`  
  - **Configuration**: PhillipsTwoLayerConfig with 2-level, 3500m depth, zonal jet + relaxation
  - **Validation**: Temperature evolution, instability growth, SSH development
  - **Grid support**: All grids (cubed_sphere, latlon, mpas, spectral)
  - **Special features**: Temperature relaxation forcing, momentum damping

### 6. **inertia_gravity_wave**
- **Description**: Inertia-gravity (Poincare) wave propagation
- **Purpose**: Test wave dynamics with rotation and analytical validation
- **Reference**: Bishnu et al. (2024)
- **Status**: ✅ REFACTORED and VALIDATED
- **Comments**:
  - **Modular implementation**: `src/legoesm/ocean/experiments/inertia_gravity_wave.py`
  - **Configuration**: InertiaGravityWaveConfig with 1000m equivalent depth, analytical solution comparison
  - **Validation**: L2 error vs analytical solution, amplitude preservation, dispersion properties
  - **Grid support**: All grids (cubed_sphere, latlon, mpas, spectral)
  - **Test results**: PASS with L2 errors and proper wave propagation

### 7. **lock_exchange** 
- **Description**: Density-driven gravity current (RPE diagnostic)
- **Purpose**: Test numerical mixing in density currents
- **Reference**: NEMO; Petersen et al. (2015)
- **Status**: ✅ REFACTORED and VALIDATED
- **Comments**:
  - **Modular implementation**: `src/legoesm/ocean/experiments/lock_exchange.py`
  - **Configuration**: LockExchangeConfig with 500m depth, 20 levels, temperature front at 0°
  - **Validation**: Reference Potential Energy evolution, mixing metrics, conservation
  - **Grid support**: cubed_sphere, latlon, mpas (spectral limited due to RPE complexity)
  - **Special features**: High vertical resolution, RPE diagnostics for mixing quantification

### 8. **overflow**
- **Description**: Dense water descending a bathymetric slope
- **Purpose**: Test dense water overflow dynamics with variable bathymetry
- **Reference**: NEMO; Petersen et al. (2015)  
- **Status**: ✅ REFACTORED and VALIDATED
- **Comments**:
  - **Modular implementation**: `src/legoesm/ocean/experiments/overflow.py`
  - **Configuration**: OverflowConfig with latitude-dependent temperature/bathymetry structure
  - **Validation**: PE evolution, overflow development, temperature stability
  - **Grid support**: cubed_sphere, latlon, mpas (spectral limited due to bathymetry)
  - **Special features**: Variable bathymetry (500m shelf → 2000m basin), 20 vertical levels

### 9. **stommel_gyre_tracer**
- **Description**: Passive tracer in wind-driven Stommel gyre
- **Purpose**: Test tracer transport and conservation in realistic circulation
- **Reference**: Hecht et al. (2000)
- **Status**: ✅ REFACTORED and VALIDATED
- **Comments**:
  - **Modular implementation**: `src/legoesm/ocean/experiments/stommel_gyre_tracer.py`
  - **Configuration**: StommelGyreTracerConfig with gyre circulation + Gaussian salinity blob
  - **Validation**: Tracer integral conservation, overshoot/undershoot detection, transport metrics  
  - **Grid support**: cubed_sphere, latlon, mpas (spectral not implemented due to complexity)
  - **Special features**: Passive tracer transport, conservation critical validation

## Major Improvements Made

### 1. **Fixed Spectral Grid Physical Units Bug** ✅
- **Problem**: Spectral grid reported drift in spectral coefficient units instead of physical units
- **Fix**: Modified `_make_scalar_fn()` to use `sh_synthesis()` for spectral-to-physical conversion  
- **Result**: All grids now report consistent physical units for meaningful cross-grid comparison

### 2. **Added Consistent Color Scales for Cross-Grid Comparison** ✅
- **Problem**: Each grid's plots used individual auto-scaled colormaps, making visual comparison impossible
- **Fix**: Added `FIELD_RANGES` constants with scientifically reasonable ranges for each test case
- **Result**: All grid plots now use identical color scales, enabling direct visual comparison of field magnitudes

### 3. **Complete Modular Refactoring of Ocean Experiments** ✅
- **Problem**: All ocean experiments embedded in massive 2800+ line monolithic script
- **Solution**: Comprehensive architectural refactoring into self-contained experiment modules
- **Files Created**: 9 experiment modules in `src/legoesm/ocean/experiments/`
- **Benefits**: 
  - **Maintainability**: Each experiment ~400 lines vs embedded in massive file
  - **Documentation**: Comprehensive scientific context, references, validation criteria
  - **Configurability**: Typed configuration classes with physical defaults
  - **Testability**: Easy to unit test, import for analysis workflows
  - **Reusability**: Experiments accessible as importable modules
  - **Consistency**: Uniform interfaces across all experiments
- **Backwards Compatibility**: Existing test matrix script unchanged and fully functional
- **Validation**: All experiments tested and pass with new modular architecture

## Approach
- Go through experiments one by one
- Update status and add comments as we review each one
- Make code edits as needed during the review process
- Focus on configuration correctness rather than underlying physics implementation

## Legend
- ⏳ Needs review
- 🔍 Under review
- ✅ Reviewed (working correctly)
- ⚠️ Issues found (needs fixes)
- ❌ Major problems (broken)