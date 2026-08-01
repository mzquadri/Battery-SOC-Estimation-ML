"""Check that the documented source files are present and syntactically valid."""

from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "src/data_loader.py",
    "src/feature_engineering.py",
    "src/soc_regression.py",
    "src/clustering_analysis.py",
    "src/genetic_fuzzy.py",
    "src/soh_analysis.py",
    "notebooks/01_EDA_Battery_Data.ipynb",
    "notebooks/02_SOC_Estimation_Models.ipynb",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    for source in (ROOT / "src").glob("*.py"):
        py_compile.compile(source, doraise=True)

    print(f"Repository check passed: {len(REQUIRED_FILES)} required files available.")


if __name__ == "__main__":
    main()
