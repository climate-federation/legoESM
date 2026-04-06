# Cross-Grid Comparison Plots Enhancement Plan

**Status**: ✅ **COMPLETED** (2026-04-03)  
**Implementation**: Successfully integrated into ocean test matrix  
**Documentation**: See `ocean_test_matrix_changelog.md` for full details

## Problem
Currently, comparing results across different grid types requires manually opening multiple folders and comparing individual plots. This is inefficient and makes it hard to spot cross-grid differences.

## Proposed Solution: Test-Level Comparison Plots

### Directory Structure Enhancement
```
results/[output_name]/
├── summary.json
├── summary.txt
└── [test_case]/                           ← Add comparison plots here
    ├── comparison_timeseries.png          ← NEW: All grids time series overlay
    ├── comparison_snapshots_eta.png       ← NEW: Final eta snapshot, 4-panel
    ├── comparison_snapshots_SST.png       ← NEW: Final SST snapshot, 4-panel  
    ├── comparison_conservation.png        ← NEW: Conservation drift comparison
    ├── comparison_summary.txt             ← NEW: Cross-grid metrics table
    └── [grid_type]/                       ← Existing individual results
        └── [resolution]/
            ├── results.txt
            ├── (all existing files...)
```

## Implementation Plan

### Phase 1: Collection Infrastructure
Create functions to:
1. **Collect results across grids**: Gather time series, final snapshots, metrics
2. **Validate data consistency**: Ensure comparable time points and fields
3. **Handle missing grids**: Skip grids that failed or weren't run

### Phase 2: Cross-Grid Comparison Plots

#### **1. Time Series Comparison (`comparison_timeseries.png`)**
```python
def _create_comparison_timeseries(test_case_dir, grid_results):
    """4-panel plot showing key time series for all grids overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Panel 1: Mean eta evolution
    for grid, data in grid_results.items():
        axes[0,0].plot(data['times'], data['mean_eta'], label=grid)
    axes[0,0].set_ylabel('Mean η (m)')
    axes[0,0].legend()
    
    # Panel 2: Mean temperature evolution  
    # Panel 3: Max speed evolution (if available)
    # Panel 4: Conservation metrics
```

#### **2. Final Snapshot Comparison (`comparison_snapshots_eta.png`)**
```python  
def _create_comparison_snapshots(test_case_dir, grid_results, field='eta'):
    """4-panel plot showing final snapshot of field across all grids."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Use consistent colormap from FIELD_RANGES
    test_case = extract_test_case_name(test_case_dir.name)
    vmin, vmax = FIELD_RANGES[test_case][field]
    
    for i, (grid, data) in enumerate(grid_results.items()):
        ax = axes.flat[i]
        final_field = data['final_snapshots'][field]  # Final time snapshot
        im = ax.imshow(final_field, vmin=vmin, vmax=vmax, ...)
        ax.set_title(f'{grid} - Final {field}')
        
    # Single shared colorbar
    fig.colorbar(im, ax=axes, ...)
```

#### **3. Conservation Comparison (`comparison_conservation.png`)**
```python
def _create_comparison_conservation(test_case_dir, grid_results):
    """3-panel plot comparing volume, heat, salt conservation across grids."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for grid, data in grid_results.items():
        # Volume conservation
        axes[0].plot(data['times'], data['vol_rel'], label=grid)
        
        # Heat conservation  
        axes[1].plot(data['times'], data['heat_rel'], label=grid)
        
        # Salt conservation
        axes[2].plot(data['times'], data['salt_rel'], label=grid)
```

### Phase 3: Summary Table
#### **Cross-Grid Metrics Table (`comparison_summary.txt`)**
```
Rest State Test - Cross-Grid Comparison
===========================================
Grid            η Drift      T Drift     Max Speed    Wall Time    Status
---------------------------------------------------------------------------  
cubed_sphere   1.04e-12     6.99e-06       0.000       4.9s       PASS
latlon         1.06e-12     2.22e-06       0.000       2.1s       PASS  
mpas           2.84e-11     2.62e-06       0.000       0.8s       PASS
spectral       1.04e-14     2.49e-06       0.000       1.8s       PASS
---------------------------------------------------------------------------
Best η:        spectral     Range: 1.04e-14 to 2.84e-11
Best T:        latlon       Range: 2.22e-06 to 6.99e-06  
Fastest:       mpas         Range: 0.8s to 4.9s
```

### Phase 4: Integration Points

#### **Modify Main Test Runner**
```python
def run_test_matrix():
    # ... existing code ...
    
    # After all grids complete for a test case
    for test_case in completed_tests:
        test_case_dir = output_base / test_case
        grid_results = collect_grid_results(test_case_dir)
        
        if len(grid_results) > 1:  # Only if multiple grids ran
            create_cross_grid_comparisons(test_case_dir, grid_results)
```

#### **Data Collection Function**
```python  
def collect_grid_results(test_case_dir):
    """Collect results from all grids that completed for this test case."""
    grid_results = {}
    
    for grid_dir in test_case_dir.iterdir():
        if grid_dir.is_dir():
            # Load timeseries CSV
            csv_file = grid_dir / resolution / "mean_timeseries.csv"
            # Load final snapshot NPZ
            npz_file = grid_dir / resolution / "snapshots_latlon.npz"
            # Load results metadata
            results_file = grid_dir / resolution / "results.txt"
            
            if all(f.exists() for f in [csv_file, npz_file, results_file]):
                grid_results[grid_dir.name] = {
                    'timeseries': pd.read_csv(csv_file),
                    'snapshots': np.load(npz_file), 
                    'metadata': parse_results_txt(results_file)
                }
                
    return grid_results
```

## Benefits

### **Immediate Visual Comparison**
- See all grid results at once instead of opening 4+ folders
- Spot outlier grids immediately  
- Assess cross-grid convergence visually

### **Quantitative Assessment** 
- Direct numerical comparison in summary table
- Identify best/worst performing grids per metric
- Track wall time performance differences

### **Publication Ready**
- High-quality comparison figures for papers
- Consistent color scales and formatting
- Professional multi-panel layouts

### **Debugging Efficiency**
- Quickly identify which grids have problems
- Visual detection of numerical issues
- Easy assessment of test case success

## Implementation Timeline
1. **Week 1**: Basic data collection infrastructure
2. **Week 2**: Time series comparison plots  
3. **Week 3**: Snapshot comparison plots
4. **Week 4**: Conservation plots and summary tables
5. **Week 5**: Integration testing and refinement

This enhancement will make the ocean test matrix much more user-friendly and effective for validation work!