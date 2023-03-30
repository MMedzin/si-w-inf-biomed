from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from utils import (
    plot_clustering,
    save_clusters,
    load_clusters,
    prepare_seq_list_for_clustering,
)
import logging
from datetime import datetime
from argparse import ArgumentParser

START_TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H%M")
EXPERIMENT_NAME = f"Experiment_{START_TIMESTAMP}"

logging.basicConfig(
    filename=f"{EXPERIMENT_NAME}.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

DATA_DIR = Path("../data/P2/HNRNPA2B1")
FSHAPE_DIR = DATA_DIR / "hnrnpa2b1_binding_sites_fshape"
EXPECTED_PATTERN_FILE = DATA_DIR / "hnrnpa2b1_expected_pattern.txt"
PLOTS_DIR = Path("./plots") / EXPERIMENT_NAME
PLOTS_DIR.mkdir(exist_ok=True)
CLUSTERING_DIR = Path("./clusters")
SEED = 23

KMEANS_KWARGS = dict(
    init="k-means++",
    n_init="auto",
    random_state=SEED,
)
KMEANS_K_VALUES = range(3, 20)

DBSCAN_KWARGS = dict()
DBSCAN_EPS_VALUES = []
DBSCAN_MIN_SAMPLES_VALUES = []


def load_expected_pattern() -> pd.DataFrame:
    return pd.read_csv(EXPECTED_PATTERN_FILE, delimiter="\t", header=None)


def process_fshape_files(length: int) -> list[pd.DataFrame]:
    window_sizes = [length, length + 1, length + 2]
    possible_seqs = {l: [] for l in window_sizes}
    for file in FSHAPE_DIR.iterdir():
        seq_df = pd.read_csv(file, delimiter="\t", header=None)
        for window_size in window_sizes:
            for i in range(0, len(seq_df) - window_size):
                subseq = seq_df.iloc[i : i + window_size]
                if (subseq.iloc[:, 0] > 1.0).any():
                    possible_seqs[window_size].append(subseq)
    return possible_seqs


def tune_kmeans_k(
    X: pd.DataFrame, plot_path: Optional[Path], interactive: bool = True
) -> Optional[KMeans]:
    sse = []
    silhouette_scores = []
    for k in KMEANS_K_VALUES:
        sse.append(KMeans(n_clusters=k, **KMEANS_KWARGS).fit(X).inertia_)
        labels = KMeans(n_clusters=k, **KMEANS_KWARGS).fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    plt.suptitle("KMeans performance")

    ax[0].plot(KMEANS_K_VALUES, sse)
    ax[0].set_xlabel("k values")
    ax[0].set_ylabel("Sum of Squared Errors")

    ax[1].plot(KMEANS_K_VALUES, silhouette_scores)
    ax[1].set_xlabel("k values")
    ax[1].set_ylabel("Silhouette score")

    plt.tight_layout()

    if plot_path is not None:
        fig.savefig(plot_path)

    if interactive:
        plt.show()
        selected_k = int(input("Select k value: "))
        while selected_k not in KMEANS_K_VALUES:
            selected_k = int(input(f"Select k value (from {list(KMEANS_K_VALUES)}): "))
        logging.info("KMeans k selected: %d", selected_k)
        return KMeans(n_clusters=selected_k, **KMEANS_KWARGS)
    return None


def tune_dbscan(
    X: pd.DataFrame, eps_plot_path: Optional[Path], interactive: bool = True
) -> Optional[DBSCAN]:
    # calculate minimum samples
    min_samples = X.shape[1] * 2
    logging.info("DBSCAN minimum samples value calculated as: %d", min_samples)

    # tune epsilon
    k = X.shape[1] * 2 - 1
    nns = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nns.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, k - 1]

    plt.plot(distances)
    plt.title("Distance Curve (of KNN)")
    plt.ylabel("Distance threshold (epsilon)")

    if eps_plot_path is not None:
        plt.savefig(eps_plot_path)

    if interactive:
        plt.show()

        eps = float(input("Select epsilon value: "))
        while not (distances.min() <= eps <= distances.max()):
            eps = float(
                input(
                    f"Select epsilon value ({distances.min()} <= eps <= {distances.max()}): "
                )
            )
        logging.info("Selected DBSCAN epsilon value: %f", eps)
        return DBSCAN(eps=eps, min_samples=min_samples)
    else:
        return None


def main(clusters_path: Optional[Path] = None) -> None:
    expected_pattern = load_expected_pattern()
    pattern_length = expected_pattern.shape[0]

    possible_seqs = process_fshape_files(pattern_length)

    if clusters_path is not None and clusters_path.exists():
        logging.info("Loading clusters from file %s", str(clusters_path))
        seq_clusters = load_clusters(clusters_path)
        logging.info("Clusters loaded succesfully")
        print(seq_clusters)
    else:
        seq_clusters = {}
        for seq_len, seqs in possible_seqs.items():
            logging.info("Running clustering for sequences of length %d...", seq_len)
            X_seq = prepare_seq_list_for_clustering(seqs)
            tuned_kmeans = tune_kmeans_k(
                X_seq,
                plot_path=PLOTS_DIR / f"KMeans_tune_plot_len_{seq_len}.png",
                interactive=True,
            )
            kmeans_clusters = pd.Series(
                tuned_kmeans.fit_predict(X_seq), index=X_seq.index
            )
            kmeans_clusters_counts = kmeans_clusters.value_counts()
            plot_kmeans_clusters = kmeans_clusters.map(
                lambda x: f"{x} ({kmeans_clusters_counts[x]})"
            )
            plot_kmeans_clusters.name = "Cluster (size)"

            tuned_dbscan = tune_dbscan(
                X_seq,
                eps_plot_path=PLOTS_DIR / f"DBSCAN_epsilon_tune_plot_len_{seq_len}.png",
                interactive=True,
            )
            dbscan_clusters = pd.Series(
                tuned_dbscan.fit_predict(X_seq), index=X_seq.index
            )
            dbscan_clusters_counts = dbscan_clusters.value_counts()
            plot_dbscan_clusters = dbscan_clusters.map(
                lambda x: f"{x} ({dbscan_clusters_counts[x]})"
            )
            plot_dbscan_clusters.name = "Cluster (size)"

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plot_clustering(X_seq, plot_kmeans_clusters, ax=ax[0])
            ax[0].set_title("KMeans clustering")
            plot_clustering(X_seq, plot_dbscan_clusters, ax=ax[1])
            ax[1].set_title("DBSCAN clustering")
            fig.savefig(PLOTS_DIR / f"Clustering_UMAP_visualization_{seq_len}.png")
            fig.show()

            selected_clustering = input(
                "Select clustering ([k]means|[d]bscan): "
            ).lower()
            while selected_clustering not in ["k", "kmeans", "d", "dbscan"]:
                selected_clustering = input(
                    "Select clustering ([k]means|[d]bscan): "
                ).lower()
            seq_clusters[seq_len] = (
                kmeans_clusters if selected_clustering[0] == "k" else dbscan_clusters
            )
            logging.info("Selected clustering: %s", selected_clustering)
            logging.info("Clustering done")
        save_clusters(seq_clusters, CLUSTERING_DIR / f"Clusters_{START_TIMESTAMP}.json")

    sequences_dfs = {
        seq_len: prepare_seq_list_for_clustering(seqs, col_num=1)
        for seq_len, seqs in possible_seqs.items()
    }

    # TODO use STUMPY to find consensus sequence from 3 biggest clusters


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--clusters_path", required=False, default=None, type=Path)
    kwargs = dict(argparser.parse_args()._get_kwargs())
    main(**kwargs)
