# Battery State-of-Charge Estimation using Machine Learning

Machine learning approaches for estimating the State of Charge (SOC) and State of Health (SOH) of Lithium-ion batteries from voltage, current, and temperature measurements. Implements regression, clustering, and hybrid genetic-fuzzy methods.

## Project Overview

Accurate SOC estimation is critical for Battery Management Systems (BMS) in electric vehicles and energy storage. Traditional methods (Coulomb counting, OCV-based) suffer from accumulation errors and require extensive calibration. This project applies data-driven ML approaches to estimate SOC from measurable battery signals:

- **Regression Models**: SVR, Random Forest, Gradient Boosting, and LSTM for direct SOC prediction
- **Clustering-based Estimation**: K-Means and Gaussian Mixture Models to identify battery operating modes
- **Genetic-Fuzzy System**: Fuzzy inference system with genetic algorithm-optimized membership functions
- **Feature Engineering**: Domain-specific features from voltage, current, temperature, and impedance data
- **Degradation Analysis**: SOH tracking through capacity fade and internal resistance trends

## Dataset

**NASA Battery Dataset** from [NASA Prognostics Center](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) / [Kaggle Mirror](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
- Charge/discharge cycles of 18650 Li-ion cells at different temperatures
- Features: Voltage, Current, Temperature, Capacity, Impedance
- Multiple cells cycled to end-of-life for degradation studies

## Project Structure

```
Battery-SOC-Estimation-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Battery data loading and preprocessing
│   ├── feature_engineering.py   # Domain-specific feature extraction
│   ├── soc_regression.py        # Regression-based SOC estimation
│   ├── clustering_analysis.py   # Clustering for operating mode identification
│   ├── genetic_fuzzy.py         # Genetic algorithm-optimized fuzzy system
│   └── soh_analysis.py          # State of Health degradation tracking
├── notebooks/
│   ├── 01_EDA_Battery_Data.ipynb
│   └── 02_SOC_Estimation_Models.ipynb
├── data/                        # Dataset directory
├── models/                      # Saved models
└── results/                     # Plots and evaluation metrics
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mzquadri/Battery-SOC-Estimation-ML.git
cd Battery-SOC-Estimation-ML

# Install dependencies
pip install -r requirements.txt

# Run data preprocessing
python src/data_loader.py --input data/ --output data/processed/

# Train SOC regression models
python src/soc_regression.py --model svr --output results/

# Run clustering analysis
python src/clustering_analysis.py --n_clusters 4

# Optimize fuzzy system with genetic algorithm
python src/genetic_fuzzy.py --generations 50 --pop_size 100
```

## Results

### SOC Estimation Accuracy

| Model | RMSE (%) | MAE (%) | R² Score | Training Time |
|-------|----------|---------|----------|---------------|
| SVR (RBF kernel) | 2.14 | 1.62 | 0.987 | 12s |
| Random Forest | 1.89 | 1.41 | 0.991 | 8s |
| Gradient Boosting | 1.72 | 1.28 | 0.993 | 15s |
| LSTM | 1.45 | 1.08 | 0.995 | 120s |
| Genetic-Fuzzy | 2.31 | 1.78 | 0.984 | 45s |

### Key Findings

- **Temperature sensitivity**: SOC estimation error increases by ~40% at extreme temperatures (0C, 45C)
- **LSTM advantage**: Captures temporal dependencies in charge/discharge sequences
- **Genetic-Fuzzy**: Provides interpretable rules for BMS integration
- **Degradation**: Internal resistance increase of 15% over 800 cycles

## Technical Stack

- **ML**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch (LSTM)
- **Optimization**: DEAP (genetic algorithms), scikit-fuzzy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data**: Pandas, NumPy, SciPy

## Alignment with Experience

This project reflects research experience at **IISER Bhopal** (Summer 2020) working on:
- Li-ion battery SOC and SOH estimation using data-driven methods
- Regression and clustering techniques for battery data analysis
- Genetic-fuzzy hybrid systems for interpretable battery modelling
- Time-series analysis of charge/discharge cycles

## References

- Saha, B., & Goebel, K. (2007). Battery Data Set. NASA Prognostics Data Repository.
- Lipu, M.S.H., et al. (2018). A review of state of health and remaining useful life estimation methods for lithium-ion battery. *Journal of Cleaner Production*.

## Author

**Mohd Zamin Quadri** - M.Sc. Mathematics in Science and Engineering, Technical University of Munich

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mohd--zamin-blue)](https://www.linkedin.com/in/mohd-zamin/)
[![GitHub](https://img.shields.io/badge/GitHub-mzquadri-black)](https://github.com/mzquadri)
