"""
Clustering Analysis for Battery Operating Mode Identification.

Uses unsupervised learning to identify distinct operating regimes
(charge phases, discharge phases, rest periods, degradation states).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering models
# ---------------------------------------------------------------------------


def kmeans_clustering(
    X: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> dict:
    """
    K-Means clustering for operating mode identification.

    Returns cluster labels and evaluation metrics.
    """
    logger.info("Running K-Means (k=%d)...", n_clusters)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X)

    # Evaluation
    silhouette = silhouette_score(X, labels) if n_clusters > 1 else 0
    calinski = calinski_harabasz_score(X, labels) if n_clusters > 1 else 0
    inertia = kmeans.inertia_

    logger.info(
        "K-Means: Silhouette=%.3f, Calinski-Harabasz=%.1f, Inertia=%.1f",
        silhouette,
        calinski,
        inertia,
    )

    return {
        "model": kmeans,
        "labels": labels,
        "centroids": kmeans.cluster_centers_,
        "n_clusters": n_clusters,
        "silhouette": round(silhouette, 4),
        "calinski_harabasz": round(calinski, 2),
        "inertia": round(inertia, 2),
    }


def gmm_clustering(
    X: np.ndarray,
    n_components: int = 4,
    covariance_type: str = "full",
    random_state: int = 42,
) -> dict:
    """
    Gaussian Mixture Model clustering.

    Allows soft assignment (probability of belonging to each cluster).
    """
    logger.info("Running GMM (k=%d, cov=%s)...", n_components, covariance_type)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=5,
        max_iter=200,
    )
    labels = gmm.fit_predict(X)
    probabilities = gmm.predict_proba(X)

    silhouette = silhouette_score(X, labels) if n_components > 1 else 0

    logger.info(
        "GMM: Silhouette=%.3f, BIC=%.1f, AIC=%.1f", silhouette, gmm.bic(X), gmm.aic(X)
    )

    return {
        "model": gmm,
        "labels": labels,
        "probabilities": probabilities,
        "means": gmm.means_,
        "n_components": n_components,
        "silhouette": round(silhouette, 4),
        "bic": round(gmm.bic(X), 2),
        "aic": round(gmm.aic(X), 2),
    }


# ---------------------------------------------------------------------------
# Optimal cluster selection
# ---------------------------------------------------------------------------


def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 10),
    method: str = "kmeans",
) -> dict:
    """
    Find optimal number of clusters using elbow method and silhouette analysis.

    Returns dict with scores per k value.
    """
    logger.info("Searching optimal k in range %s...", list(k_range))

    results = []
    for k in k_range:
        if method == "kmeans":
            res = kmeans_clustering(X, n_clusters=k)
            results.append(
                {
                    "k": k,
                    "silhouette": res["silhouette"],
                    "calinski_harabasz": res["calinski_harabasz"],
                    "inertia": res["inertia"],
                }
            )
        else:
            res = gmm_clustering(X, n_components=k)
            results.append(
                {
                    "k": k,
                    "silhouette": res["silhouette"],
                    "bic": res["bic"],
                    "aic": res["aic"],
                }
            )

    results_df = pd.DataFrame(results)

    # Best k by silhouette
    best_k = results_df.loc[results_df["silhouette"].idxmax(), "k"]
    logger.info("Optimal k by silhouette: %d", best_k)

    return {
        "results": results_df,
        "best_k": int(best_k),
    }


# ---------------------------------------------------------------------------
# Cluster interpretation
# ---------------------------------------------------------------------------


def interpret_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Characterize each cluster by its feature statistics.

    Returns summary DataFrame with mean/std per cluster per feature.
    """
    df = df.copy()
    df["cluster"] = labels

    summary = df.groupby("cluster")[feature_cols].agg(["mean", "std", "count"])

    # Flatten multi-level columns
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()

    return summary


def label_operating_modes(
    cluster_summary: pd.DataFrame,
    voltage_col: str = "voltage_mean",
    current_col: str = "current_mean",
) -> dict:
    """
    Automatically label clusters as operating modes based on feature patterns.

    Heuristic mapping:
        - High current, decreasing voltage -> Active Discharge
        - Negative current, increasing voltage -> Charging
        - Near-zero current -> Rest/Idle
        - High temperature -> Stress Mode
    """
    labels = {}

    for _, row in cluster_summary.iterrows():
        cluster_id = row["cluster"]
        avg_voltage = row.get(voltage_col, 0)
        avg_current = row.get(current_col, 0)

        if avg_current > 1.5:
            labels[cluster_id] = "Active Discharge (High Load)"
        elif avg_current > 0.5:
            labels[cluster_id] = "Moderate Discharge"
        elif avg_current < -0.5:
            labels[cluster_id] = "Charging"
        elif abs(avg_current) < 0.1:
            labels[cluster_id] = "Rest/Idle"
        else:
            labels[cluster_id] = f"Transition (I={avg_current:.2f}A)"

    return labels


# ---------------------------------------------------------------------------
# SOC-based clustering (for stratified analysis)
# ---------------------------------------------------------------------------


def cluster_by_soc_region(
    df: pd.DataFrame,
    soc_col: str = "soc",
    n_regions: int = 5,
) -> pd.DataFrame:
    """
    Segment data by SOC regions and analyze feature distributions per region.

    Useful for understanding model performance across different SOC levels.
    """
    df = df.copy()

    boundaries = np.linspace(0, 1, n_regions + 1)
    region_labels = [
        f"SOC {boundaries[i]:.0%}-{boundaries[i + 1]:.0%}" for i in range(n_regions)
    ]

    df["soc_region"] = pd.cut(
        df[soc_col],
        bins=boundaries,
        labels=region_labels,
        include_lowest=True,
    )

    region_stats = (
        df.groupby("soc_region")
        .agg(
            count=("soc", "count"),
            avg_voltage=("voltage", "mean")
            if "voltage" in df.columns
            else ("soc", "count"),
            avg_current=("current", "mean")
            if "current" in df.columns
            else ("soc", "count"),
            avg_temp=("temperature", "mean")
            if "temperature" in df.columns
            else ("soc", "count"),
        )
        .reset_index()
    )

    logger.info("SOC region analysis:\n%s", region_stats.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_clustering_analysis(
    df: pd.DataFrame,
    feature_cols: list,
    n_clusters: int = 4,
) -> dict:
    """Run complete clustering analysis pipeline."""
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)

    # K-Means
    kmeans_result = kmeans_clustering(X, n_clusters=n_clusters)

    # GMM
    gmm_result = gmm_clustering(X, n_components=n_clusters)

    # Find optimal k
    optimal = find_optimal_k(X, k_range=range(2, min(8, len(df) // 100)))

    # Interpret clusters
    summary = interpret_clusters(df, kmeans_result["labels"], feature_cols)

    return {
        "kmeans": kmeans_result,
        "gmm": gmm_result,
        "optimal_k": optimal,
        "cluster_summary": summary,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Battery Clustering Analysis")
    parser.add_argument(
        "--input", type=str, default="data/processed/battery_features.csv"
    )
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    from .feature_engineering import get_feature_columns

    df = pd.read_csv(args.input)
    feature_cols = get_feature_columns(df)

    results = run_clustering_analysis(df, feature_cols, n_clusters=args.n_clusters)

    print(f"\nK-Means Silhouette: {results['kmeans']['silhouette']}")
    print(f"GMM Silhouette: {results['gmm']['silhouette']}")
    print(f"Optimal k: {results['optimal_k']['best_k']}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    results["cluster_summary"].to_csv(f"{args.output}/cluster_summary.csv", index=False)
