# Ocean Validation Framework Improvement Plan

## Executive Summary

Based on analysis of legoESM's validation frameworks across components, the ocean validation is currently the least sophisticated compared to atmosphere and land models. This document outlines a comprehensive plan to enhance ocean test case validation by adopting best practices from the more mature atmosphere validation framework.

## Current State Analysis

### Component Comparison

| Component | **Ocean** | **Atmosphere** | **Land** |
|-----------|-----------|---------------|----------|
| **Analytical solutions** | ❌ Only IGW wave | ✅ Williamson TC2 exact solution | ❌ Complex processes |
| **Error norms** | Basic L2 for IGW only | ✅ L1, L2, L∞ comprehensive | ❌ Constraint-based |
| **Physical constraints** | Basic conservation | ✅ + cube-edge artifacts | ✅ Physical bounds, budgets |
| **Long-term validation** | ❌ Short 5-day tests | ✅ 1200-day Held-Suarez | ✅ 15-180 day realistic |
| **Literature benchmarks** | ✅ NEMO references | ✅ Well-established | ✅ Process understanding |
| **Cross-grid comparison** | Basic | ✅ Comprehensive | Limited |
| **Visual validation** | ❌ None | ✅ Cube-edge artifact detection | ❌ Field bounds |
| **Error diagnostics** | Basic blowup detection | ✅ 1353-line comprehensive suite | ✅ Physics-based constraints |

### Ocean Validation Limitations

#### **Current Test Cases** (9 cases, 4 grids)
1. `rest_state` - Basic stability
2. `barotropic_wave` - Wave propagation  
3. `wind_gyre` - Wind-driven circulation
4. `baroclinic` - Thermal front dynamics
5. `phillips_two_layer` - Baroclinic instability
6. `inertia_gravity_wave` - Wave dynamics (has analytical solution)
7. `lock_exchange` - Density currents (NEMO benchmark)
8. `overflow` - Dense water flow (NEMO benchmark)
9. `stommel_gyre_tracer` - Tracer transport

#### **Current Evaluation Criteria** (`scripts/matrix/run_ocean_test_matrix.py`)
- **Blowup detection**: `|η| < threshold` (100-200m)
- **Conservation drift**: Basic relative change calculations
- **Finite field validation**: `jnp.isfinite()` checks
- **Test-specific metrics**: RPE for mixing tests, L2 error for IGW

#### **Key Gaps**
1. **No comprehensive error norms** (only L2 for IGW test)
2. **No visual artifact detection** (unlike atmosphere cube-edge analysis)
3. **No long-term validation** (longest test is 5 days)
4. **Limited analytical benchmarks** (only IGW has exact solution)
5. **No process-specific constraints** (unlike land bounds checking)
6. **No cross-grid error analysis** 

## Improvement Plan

### Phase 1: Framework Enhancement (Learning from Atmosphere)

#### **1.1 Analytical Solution Development**
**Goal**: Expand beyond IGW to additional exact solutions

**Actions**:
- **Rest State Test**: Develop analytical zero-tendency validation
  - All tendencies should be machine precision zero
  - Implement comprehensive tendency analysis
  - Add drift rate quantification
  
- **Barotropic Wave Test**: Create analytical benchmark
  - Develop shallow water analytical solution for wave propagation
  - Implement exact dispersion relation validation
  - Add phase speed and amplitude error tracking

- **Geostrophic Balance**: Add analytical geostrophic adjustment test
  - Initial imbalance should relax to geostrophic equilibrium
  - Exact solution available for linear case

**Deliverables**:
- `analytical_solutions.py` module with exact solution functions
- Enhanced test cases with analytical comparison
- Error norm computation framework

#### **1.2 Comprehensive Error Analysis Framework**
**Goal**: Implement Williamson-style error diagnostics

**Model after**: `tests/williamson_diagnostic.py` (662 lines)

**Actions**:
- **Multi-norm error analysis**: L1, L2, L∞ norms for all test cases
- **Grid convergence studies**: Systematic resolution refinement
- **Cross-discretization comparison**: Error analysis across 4 grid types
- **Temporal error tracking**: Error evolution over integration period

**Implementation**:
```python
# New file: diagnostics/ocean_error_analysis.py
class OceanErrorAnalyzer:
    def compute_error_norms(self, computed, analytical):
        """Compute L1, L2, L∞ error norms."""
        
    def grid_convergence_analysis(self, resolutions, errors):
        """Analyze spatial convergence rates."""
        
    def cross_grid_comparison(self, solutions_dict):
        """Compare solutions across grid types."""
        
    def temporal_error_evolution(self, error_time_series):
        """Track how errors evolve in time."""
```

#### **1.3 Visual Artifact Detection**
**Goal**: Implement cube-edge and grid artifact detection

**Model after**: `williamson_diagnostic.py` cube-edge analysis

**Actions**:
- **Cubed-sphere edge artifacts**: Boundary vs interior error ratios
- **Grid imprinting detection**: Spectral analysis of grid-scale noise
- **Spurious mixing visualization**: RPE spatial distribution analysis
- **Circulation pattern validation**: Streamfunction and vorticity analysis

**Implementation**:
```python
# Enhancement to existing framework
def cube_edge_artifact_analysis(error_field, grid):
    """Compare max |error| at face boundaries vs interiors."""
    
def grid_imprinting_detection(field, grid):
    """Detect grid-scale artifacts via spectral analysis."""
    
def circulation_pattern_validation(u, v, grid, expected_pattern):
    """Validate circulation structure against expected patterns."""
```

### Phase 2: Test Case Enhancement

#### **2.1 Individual Test Case Improvements**

##### **Rest State Test**
**Current**: Basic drift monitoring
**Enhanced**:
- **Analytical validation**: All tendencies = 0 (machine precision)
- **Process isolation**: Test momentum, mass, heat tendencies separately  
- **Sensitivity analysis**: Response to small perturbations
- **Long-term stability**: Extended integration (30+ days)

##### **Inertia-Gravity Wave Test** 
**Current**: Basic L2 error vs analytical solution
**Enhanced**:
- **Comprehensive error norms**: L1, L2, L∞ analysis
- **Dispersion relation validation**: ω vs k relationship
- **Phase speed accuracy**: Group velocity tracking
- **Grid convergence study**: Resolution refinement analysis

##### **Lock Exchange Test**
**Current**: RPE drift monitoring
**Enhanced**:
- **Physical constraint validation**: Density stratification preservation
- **Mixing efficiency metrics**: Compare against DNS/LES benchmarks
- **Front propagation analysis**: Speed and structure validation
- **Sensitivity to numerical parameters**: Grid resolution, time step, diffusion

##### **Overflow Test**
**Current**: RPE tracking
**Enhanced**:
- **Plume descent validation**: Compare descent rate against theory
- **Entrainment analysis**: Mixing with ambient water
- **Bathymetric slope sensitivity**: Various topographic configurations
- **Realistic parameter ranges**: Use observational constraints

##### **Wind Gyre Test**
**Current**: Basic circulation development
**Enhanced**:
- **Western boundary current analysis**: Jet structure and separation
- **Streamfunction validation**: Compare against Stommel/Munk solutions  
- **Recirculation efficiency**: Energy transfer from wind to gyre
- **Steady-state convergence**: Time to equilibrium analysis

##### **Barotropic/Baroclinic Tests**
**Current**: Qualitative evolution
**Enhanced**:
- **Analytical benchmarks**: Develop exact solutions where possible
- **Instability analysis**: Growth rates, wavelength selection
- **Energy cascade analysis**: Kinetic/potential energy evolution
- **Modal decomposition**: Baroclinic mode structure validation

#### **2.2 New Test Cases**

##### **Equatorial Wave Dynamics**
- **Kelvin waves**: Exact analytical solutions available
- **Rossby waves**: Dispersion relation validation
- **Mixed Rossby-gravity waves**: Phase speed analysis

##### **Coastal Upwelling**
- **Ekman dynamics**: Analytical cross-shore structure
- **Upwelling efficiency**: Compare against theory
- **Stratification effects**: Realistic density profiles

##### **Tidal Dynamics**
- **Amphidromic systems**: Well-known analytical solutions
- **Internal tides**: Generation and propagation
- **Energy dissipation**: Bottom friction validation

### Phase 3: Advanced Validation Framework

#### **3.1 Long-term Validation**
**Goal**: Extended simulations with realistic forcing

**Model after**: Atmosphere Held-Suarez (1200 days), Land stability tests (15-180 days)

**Actions**:
- **Seasonal cycles**: 1+ year integrations with realistic forcing
- **Climate drift analysis**: Long-term conservation and stability  
- **Regime transitions**: Response to changing forcing
- **Statistical validation**: Mean state and variability metrics

#### **3.2 Process-Specific Constraints**
**Goal**: Physics-based validation criteria

**Model after**: Land model bounds checking (`test_land_stability.py`)

**Implementation**:
```python
# New validation constraints
class OceanPhysicsValidator:
    def validate_density_bounds(self, rho, T, S):
        """Ensure realistic density ranges."""
        
    def validate_mixing_efficiency(self, RPE_initial, RPE_final):
        """Check mixing against physical limits."""
        
    def validate_circulation_strength(self, streamfunction, wind_stress):
        """Validate gyre circulation vs forcing."""
        
    def validate_energy_cascades(self, ke_spectrum, pe_spectrum):
        """Check energy cascade direction and rates."""
```

#### **3.3 Observational Validation Framework**
**Goal**: Comparison against real ocean observations

**Actions**:
- **Mean state validation**: Temperature, salinity climatologies
- **Variability metrics**: Eddy kinetic energy, mixed layer depth
- **Regional comparisons**: Specific ocean basins and processes
- **Scaling relationships**: Energy transfer rates, mixing coefficients

### Phase 4: Implementation Timeline

#### **Month 1-2: Framework Development**
- [ ] Implement analytical solution library
- [ ] Develop comprehensive error analysis framework  
- [ ] Create visual artifact detection tools
- [ ] Design process-specific constraint validators

#### **Month 3-4: Test Case Enhancement**
- [ ] Enhance existing 9 test cases with new framework
- [ ] Add analytical benchmarks where possible
- [ ] Implement long-term validation protocols
- [ ] Develop cross-grid comparison tools

#### **Month 5-6: Advanced Features**
- [ ] Add new test cases (equatorial waves, coastal upwelling, tides)
- [ ] Implement observational validation framework
- [ ] Create automated reporting and visualization
- [ ] Performance optimization and documentation

#### **Month 7-8: Validation and Documentation**
- [ ] Comprehensive testing of new framework
- [ ] Validation against literature benchmarks
- [ ] User documentation and tutorials
- [ ] Integration with existing CI/CD pipeline

## Success Metrics

### **Quantitative Targets**
1. **Error Analysis**: L1, L2, L∞ norms for all test cases
2. **Grid Convergence**: Document convergence rates for all discretizations
3. **Long-term Stability**: 1+ year integrations with <1% climate drift  
4. **Cross-grid Consistency**: <10% difference between grid types for same physics
5. **Literature Benchmarks**: Match published results within error bars

### **Qualitative Improvements**  
1. **Physical Realism**: Robust constraint validation
2. **Visual Validation**: Comprehensive artifact detection
3. **Process Understanding**: Clear physics-based success criteria
4. **User Experience**: Clear pass/fail determination
5. **Scientific Credibility**: Publication-quality validation suite

## Resource Requirements

### **Development Effort**
- **Senior Ocean Modeler**: 6-8 months full-time
- **Numerical Methods Expert**: 2-3 months consulting
- **Visualization Specialist**: 1-2 months part-time

### **Computational Resources**
- **Development Testing**: Moderate (existing CI resources)
- **Grid Convergence Studies**: High (multiple resolution runs)
- **Long-term Validation**: Very High (1+ year integrations)

### **Literature Research**
- **Analytical Solutions**: Ocean dynamics textbooks, papers
- **Observational Data**: WOCE, Argo, satellite climatologies  
- **Benchmark Studies**: NEMO, MOM6, MITGCM validation papers

## Conclusion

This plan will transform legoESM's ocean validation from basic numerical checks to a comprehensive, physics-based validation framework comparable to the atmosphere component. The enhanced framework will provide:

1. **Rigorous error quantification** through analytical solutions
2. **Robust physical constraint validation** ensuring realistic behavior
3. **Long-term stability assessment** for climate applications  
4. **Cross-discretization verification** ensuring numerical robustness
5. **Literature-quality benchmarking** enabling publication and adoption

The systematic approach, based on proven atmosphere validation methods, will establish legoESM's ocean component as a scientifically credible and thoroughly validated modeling framework.