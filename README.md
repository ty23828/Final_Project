# FWI Super-Resolution Enhancement Project

A comprehensive machine learning pipeline for enhancing Fire Weather Index (FWI) spatial resolution from 25km to 1km using advanced downsampling-upsampling strategies and ensemble modeling techniques.

## Project Overview

This project implements a novel approach to enhance the spatial resolution of Fire Weather Index data from ERA5 reanalysis (25km) to high-resolution predictions (1km) using machine learning models. The system uses a downsampling-upsampling training strategy to learn realistic resolution enhancement patterns.

### Key Features
- **Resolution Enhancement**: 25km to 1km FWI predictions
- **Multiple ML Models**: Random Forest, Gradient Boosting, Neural Networks, Ridge Regression
- **Comprehensive Analysis**: Feature importance, statistical validation, vulnerability assessment
- **Real-world Applications**: Fire risk mapping, case studies, vulnerability analysis

## Project Structure

```
Final_code/
├── Core Analysis Modules
│   ├── EDA.py                    # Exploratory Data Analysis & Preprocessing
│   ├── Model.py                  # Super-resolution model training & prediction
│   ├── Importance.py             # Feature importance analysis with downscaling
│   └── Statistical.py            # Comprehensive statistical analysis
│
├── Validation & Testing
│   ├── Validation.py            # Multi-model validation framework
│   ├── Vail_1km_resample.py     # 1km resampling validation
│   └── Resample_1km.py          # Basic 1km resampling methods
│
├── Analysis & Visualization
│   ├── Case.py                  # Specific fire event case studies
│   ├── Vulnerability.py         # Fire vulnerability assessment
│   ├── Year_compare.py          # Annual performance comparison
│   ├── Daily_compare.py         # Daily performance analysis
│   └── Different_resolution.py  # Multi-resolution comparison
│
├── CMIP6 Climate Models
│   └── experiment/CMIP6_modelled_data/
│       ├── enhanced_ml_fwi_prediction.py
│       ├── compare_predictions.py
│       └── convert.py
│
└── Generated Results & Reports
    ├── *.png                    # Visualization outputs
    ├── *.csv                    # Processed datasets
    └── *_report.txt             # Analysis reports
```

## Quick Start

### 1. Data Preparation & EDA
```bash
# Step 1: Run exploratory data analysis and preprocessing
python EDA.py
```
**Outputs:**
- `merged_25km_data.csv` - Preprocessed 25km resolution data
- `training_pairs.csv` - Training data for super-resolution
- `correlation_*.png` - Feature correlation visualizations

### 2. Super-Resolution Model Training
```bash
# Step 2: Train super-resolution models
python Model.py
```
**Outputs:**
- `fwi_1km_super_resolution.csv` - Enhanced 1km predictions
- `super_resolution_model_performance.csv` - Model metrics
- `super_resolution_analysis_report_*.txt` - Comprehensive analysis

### 3. Feature Importance Analysis
```bash
# Step 3: Analyze feature importance with proper downscaling
python Importance.py
```
**Outputs:**
- `downscaled_feature_importance_*.png` - Importance visualizations
- `downscaled_feature_importance_report_*.txt` - Detailed analysis

### 4. Model Validation
```bash
# Step 4: Validate predictions against reference data
python Validation.py
```
**Outputs:**
- `model_comparison_summary.csv` - Performance comparison
- `model_comparison_plots.png` - Validation visualizations

## Core Methodology

### Super-Resolution Strategy
1. **Downsampling Training**: Learn patterns from 25km to 50km to 25km
2. **Feature Engineering**: Extract spatial, meteorological, and gradient features
3. **Ensemble Modeling**: Train multiple ML models (RF, GB, NN, Ridge)
4. **Resolution Enhancement**: Apply learned patterns to 25km to 1km

### Model Architecture
```python
# Example usage
from Model import FWISuperResolutionModel

# Initialize system
sr_system = FWISuperResolutionModel()

# Load and preprocess data
sr_system.load_era5_data()
sr_system.preprocess_and_merge()

# Create training data
sr_system.create_downsampled_training_data('50km')

# Train models
sr_system.train_super_resolution_models()

# Generate 1km predictions
sr_system.apply_super_resolution_25km_to_1km()
```

## Analysis Modules

### Feature Importance Analysis
```bash
python Importance.py
```
- **Downscaling Strategy**: Meteovars 25km to 50km, FWI stays 25km
- **Multiple Models**: RF, GB, NN, Ridge regression
- **Consensus Ranking**: Cross-model feature importance

### Statistical Analysis
```bash
python Statistical.py
```
- **Distribution Analysis**: 25km vs 1km FWI distributions
- **Extreme Performance**: Model performance on high FWI values
- **Bias Assessment**: Systematic bias patterns

### Case Studies
```bash
python Case.py
```
- **June 2017 Fires**: Specific fire event analysis
- **October 2017**: High FWI day counting
- **Spatial Visualization**: Fire location mapping

### Vulnerability Assessment
```bash
python Vulnerability.py
```
- **Risk Mapping**: Combined FWI and exposure analysis
- **LitPop Integration**: Population and asset exposure
- **Mainland Portugal**: Focused vulnerability analysis

## Required Data Files

### ERA5 Reanalysis Data
```
experiment/ERA5_reanalysis_fwi/
└── era5_fwi_2017_portugal_3decimal.csv

experiment/ERA5_reanalysis_atmospheric_parameters/
├── era5_daily_max_temp_2017_portugal.csv
└── era5_daily_mean_2017_combined_3decimal.csv
```

### Exposure Data
```
LitPop_pc_30arcsec_PRT.csv  # Population and asset exposure data
```

## Key Results

### Model Performance
- **Best Model**: Gradient Boosting (R² = 0.985)
- **Resolution Enhancement**: 25km to 1km (1.5x spatial density increase)
- **Feature Count**: 50+ meteorological and spatial features

### Generated Insights
- **Extreme Events**: 1km resolution captures 1.1x more extreme FWI events
- **Spatial Patterns**: Enhanced detail in fire risk distribution
- **Validation**: Strong correlation with reference data (R² > 0.8)

## Output Files

### Core Predictions
- `fwi_1km_super_resolution.csv` - Main 1km predictions
- `fwi_1km_predictions_*.csv` - Model-specific predictions

### Analysis Results
- `*_feature_importance_*.png` - Feature importance plots
- `*_performance_*.png` - Model performance visualizations
- `*_validation_*.png` - Validation plots
- `*_vulnerability_*.png` - Risk assessment maps

### Reports
- `super_resolution_analysis_report_*.txt` - Comprehensive analysis
- `fwi_analysis_report.txt` - Statistical analysis summary
- `downscaled_feature_importance_report_*.txt` - Feature analysis

## Dependencies

```python
# Core ML/Data Science
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
cartopy>=0.20.0

# Geospatial
xarray>=0.19.0
netCDF4>=1.5.0
```

## Usage Examples

### Run Complete Pipeline
```bash
# Full pipeline execution
python EDA.py && python Model.py && python Validation.py
```

### Individual Analysis
```bash
# Feature importance only
python Importance.py

# Statistical analysis only
python Statistical.py

# Vulnerability assessment only
python Vulnerability.py
```

### Custom Configuration
```python
# Custom model training
from Model import FWISuperResolutionModel

sr_system = FWISuperResolutionModel()
sr_system.load_era5_data()
sr_system.preprocess_and_merge()

# Use 75km downsampling instead of 50km
sr_system.create_downsampled_training_data('75km')
sr_system.train_super_resolution_models()
```

## Research Applications

### Fire Management
- **High-resolution risk mapping**
- **Early warning systems**
- **Resource allocation optimization**

### Climate Research
- **Downscaling validation**
- **Extreme event analysis**
- **Model intercomparison**

### Operational Use
- **Daily fire weather forecasting**
- **Seasonal risk assessment**
- **Emergency response planning**

## Related Publications

This codebase supports research in:
- Fire weather index enhancement
- Machine learning downscaling
- Extreme event modeling
- Spatial resolution enhancement

## Contributing

1. **Data Processing**: Enhance preprocessing pipeline
2. **Model Development**: Add new ML models
3. **Validation Methods**: Improve validation techniques
4. **Visualization**: Create new analysis plots

## License

This project is part of academic research. Please cite appropriately if used in publications.

---


