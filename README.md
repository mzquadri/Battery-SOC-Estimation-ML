# Battery State-of-Charge Estimation with Machine Learning

A lithium-ion battery does not carry a fuel gauge. Its state of charge has to be inferred from voltage, current, temperature, and cycling history, and every one of those signals drifts as the battery ages. This repository experiments with machine-learning ways to make that estimate, and to say something useful about how healthy the battery still is. It brings together regression models, clustering, engineered cycle features, and a genetic-fuzzy prototype in one testable codebase, so that the individual ideas can be compared rather than treated as a black box.

> **Research prototype:** This repository does not include a battery dataset, trained weights, or tracked evaluation outputs. It must not be used to operate a battery-management system or to make safety decisions. The scripts can run against authorized source data or the included deterministic synthetic demonstration generator.

## Included Methods

- SOC regression with SVR, Random Forest, XGBoost, LightGBM, and an LSTM implementation
- Cycle-aware feature engineering from voltage, current, temperature, and time
- K-Means and Gaussian-mixture clustering for exploratory operating-regime analysis
- A genetic-optimized fuzzy SOC estimator
- Capacity-fade, SOH, resistance, and remaining-useful-life exploratory analyses

The figure below shows how the pieces fit together:

![Battery state of charge estimation pipeline](docs/diagrams/pipeline.svg)

## Data and Evaluation Scope

The code accepts compatible NASA Battery Dataset MATLAB files or CSV exports. Obtain and use source data in accordance with its terms; no external data is redistributed here. `src.data_loader.generate_synthetic_battery_data` provides generated discharge cycles only for pipeline development.

Earlier score tables and degradation claims are intentionally not presented here because this repository has no versioned source split, run configuration, model artifact, or metric report to substantiate them. A meaningful benchmark should record the cell identifiers, source-data version, preprocessing parameters, temporal split, random seed, dependency versions, and evaluation artifacts. Synthetic-data results demonstrate code execution only, not battery-estimation performance.

## Project Structure

```text
.
├── src/
│   ├── data_loader.py          # NASA-compatible loading, cleaning, synthetic demo data
│   ├── feature_engineering.py  # Signal and cycle features
│   ├── soc_regression.py       # SOC regression models and metrics
│   ├── clustering_analysis.py  # K-Means and GMM analysis
│   ├── genetic_fuzzy.py        # Genetic-fuzzy SOC prototype
│   └── soh_analysis.py         # Capacity/SOH degradation analysis
├── notebooks/                  # Exploratory notebooks
├── data/                       # Local data only; ignored by Git
├── models/                     # Local model artifacts only; ignored by Git
├── results/                    # Local run outputs only; ignored by Git
└── scripts/
    ├── check_repository.py
    └── smoke_test.py
```

## Quick Start

```bash
git clone https://github.com/mzquadri/Battery-SOC-Estimation-ML.git
cd Battery-SOC-Estimation-ML
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt

# Deterministic generated-data pipeline for development only
python -m src.data_loader --synthetic --output data/processed
python -m src.feature_engineering --input data/processed/battery_processed.csv --output data/processed/battery_features.csv
python -m src.soc_regression --model rf --input data/processed/battery_features.csv --output results

# Repository integrity and core-function smoke checks
python scripts/check_repository.py
python scripts/smoke_test.py
```

Run modules with `python -m src.<module>` rather than `python src/<module>.py`; the analysis modules use package-relative imports. The generated data and outputs above are ignored by Git. Review all changes to preprocessing and generated results before treating a run as comparable to another experiment.

For external data, point `--input` to a compatible MAT file, a CSV file, or a directory containing `battery_data.csv` or `B*.csv` exports. Create features before running a model. Use a temporal cell/cycle split and verify that no cells or cycles leak between train and test sets.

## Dependencies

Core preprocessing and baseline models use NumPy, pandas, SciPy, and scikit-learn. XGBoost, LightGBM, PyTorch, DEAP, and scikit-fuzzy support optional methods; install dependencies from `requirements.txt` before using those paths.

## References

- Saha, B., & Goebel, K. (2007). *Battery Data Set*. NASA Prognostics Data Repository.
- Lipu, M. S. H., et al. (2018). A review of state of health and remaining useful life estimation methods for lithium-ion battery. *Journal of Cleaner Production*.

## License

This project is released under the [MIT License](LICENSE).
