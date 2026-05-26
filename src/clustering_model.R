"""
clustering_model.py
-------------------
K-Means clustering pipeline for customer segmentation.
Includes optimal K selection, model fitting, segment labelling,
and visualisation utilities.

Author: Shalmalee Sharma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Colour palette for segments
SEGMENT_COLOURS = {
    "High-Value Loyal":   "#2ECC71",
    "New Customers":      "#3498DB",
    "At-Risk Customers":  "#E74C3C",
    "Low-Value Customers":"#95A5A6",
}


def find_optimal_k(rfm_scaled: pd.DataFrame, k_range: range = range(2, 11)) -> int:
    """
    Use Elbow Method + Silhouette Score to determine optimal number of clusters.

    Args:
        rfm_scaled: Scaled RFM DataFrame
        k_range: Range of K values to test

    Returns:
        Optimal K value
    """
    features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values
    inertias, silhouette_scores = [], []

    logger.info("Finding optimal K...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))
        logger.info(f"  K={k} | Inertia: {km.inertia_:.0f} | Silhouette: {silhouette_score(features, labels):.3f}")

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimal K Selection", fontsize=14, fontweight="bold")

    axes[0].plot(k_range, inertias, "o-", color="#3498DB", linewidth=2, markersize=8)
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, silhouette_scores, "o-", color="#2ECC71", linewidth=2, markersize=8)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/optimal_k_selection.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Saved optimal_k_selection.png")

    optimal_k = k_range[np.argmax(silhouette_scores)]
    logger.info(f"Optimal K = {optimal_k} (highest silhouette score)")
    return optimal_k


def fit_kmeans(rfm_scaled: pd.DataFrame, k: int) -> tuple[KMeans, np.ndarray]:
    """
    Fit K-Means model with optimal K.

    Returns:
        Tuple of (fitted KMeans model, cluster labels)
    """
    features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)
    logger.info(f"K-Means fitted with K={k}")
    logger.info(f"Cluster distribution: {pd.Series(labels).value_counts().to_dict()}")
    return km, labels


def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable segment names based on RFM cluster centroids.
    Segments are ranked by Monetary value (desc) and Recency (asc).
    """
    cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # Rank clusters: lower recency = more recent; higher monetary = more valuable
    cluster_summary["Score"] = (
        cluster_summary["Monetary"].rank(ascending=True) +
        cluster_summary["Frequency"].rank(ascending=True) -
        cluster_summary["Recency"].rank(ascending=True)
    )
    ranked = cluster_summary["Score"].rank(ascending=False).astype(int)

    segment_map = {
        1: "High-Value Loyal",
        2: "New Customers",
        3: "At-Risk Customers",
        4: "Low-Value Customers",
    }
    rfm["Segment"] = rfm["Cluster"].map(ranked).map(segment_map)
    logger.info(f"Segment distribution:\n{rfm['Segment'].value_counts()}")
    return rfm


def plot_rfm_distributions(rfm: pd.DataFrame):
    """Plot RFM distributions coloured by segment."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("RFM Distributions by Customer Segment", fontsize=14, fontweight="bold")

    metrics = ["Recency", "Frequency", "Monetary"]
    for ax, metric in zip(axes, metrics):
        for segment, colour in SEGMENT_COLOURS.items():
            subset = rfm[rfm["Segment"] == segment][metric]
            ax.hist(subset, bins=30, alpha=0.6, color=colour, label=segment)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    handles = [mpatches.Patch(color=c, label=s) for s, c in SEGMENT_COLOURS.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.12, 0.9))
    plt.tight_layout()
    plt.savefig("../outputs/rfm_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Saved rfm_distributions.png")


def plot_segment_summary(rfm: pd.DataFrame):
    """Bar chart of average RFM values per segment."""
    summary = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Average RFM Values by Segment", fontsize=14, fontweight="bold")

    colours = [SEGMENT_COLOURS.get(s, "#BDC3C7") for s in summary.index]

    for ax, metric in zip(axes, ["Recency", "Frequency", "Monetary"]):
        bars = ax.bar(summary.index, summary[metric], color=colours, edgecolor="white", linewidth=0.8)
        ax.set_title(f"Avg {metric}")
        ax.set_ylabel(metric)
        ax.set_xticklabels(summary.index, rotation=25, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("../outputs/segment_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Saved segment_summary.png")


def plot_scatter_3d(rfm: pd.DataFrame):
    """3D scatter plot of RFM space coloured by segment."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for segment, colour in SEGMENT_COLOURS.items():
        subset = rfm[rfm["Segment"] == segment]
        ax.scatter(subset["Recency"], subset["Frequency"], subset["Monetary"],
                   c=colour, label=segment, alpha=0.6, s=20)

    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Monetary ($)")
    ax.set_title("Customer Segments in RFM Space", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.1, 1.1))

    plt.tight_layout()
    plt.savefig("../outputs/rfm_3d_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Saved rfm_3d_scatter.png")


def generate_segment_report(rfm: pd.DataFrame) -> pd.DataFrame:
    """Generate a business-ready summary report per segment."""
    report = rfm.groupby("Segment").agg(
        Customer_Count =("CustomerID",  "count"),
        Avg_Recency    =("Recency",     "mean"),
        Avg_Frequency  =("Frequency",   "mean"),
        Avg_Monetary   =("Monetary",    "mean"),
        Total_Revenue  =("Monetary",    "sum"),
    ).round(1)

    report["Revenue_Share_%"] = (report["Total_Revenue"] / report["Total_Revenue"].sum() * 100).round(1)
    report = report.sort_values("Total_Revenue", ascending=False)

    logger.info(f"\n{'='*60}\nSEGMENT REPORT\n{'='*60}\n{report.to_string()}")
    return report


if __name__ == "__main__":
    print("Import this module and use individual functions in the notebook.")