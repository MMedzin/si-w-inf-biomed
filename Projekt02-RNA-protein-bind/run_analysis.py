import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from pattern import Pattern
from utils import (
    load_clusters,
    plot_clustering,
    prepare_seq_list_for_clustering,
    save_clusters,
)

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
SEARCH_DIR = DATA_DIR / "hnrnpa2b1_search_fshape"
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

CLUSTER_COL = "cluster"
REACT_COL = "vector"
SEQ_COL = "sequence"


def get_sequence_str(df: pd.DataFrame, str_col: int = 1) -> str:
    return "".join(df.iloc[:, str_col].values)


def get_reactive_vector(df: pd.DataFrame, reactive_col: int = 0) -> list[float]:
    return df.loc[:, reactive_col].values.tolist()


def load_expected_pattern() -> Pattern:
    pattern_df = pd.read_csv(EXPECTED_PATTERN_FILE, delimiter="\t", header=None)
    return Pattern(
        sequence=get_sequence_str(pattern_df),
        fSHAPE=get_reactive_vector(pattern_df),
        file=str(EXPECTED_PATTERN_FILE.name),
        start=pattern_df.index.min(),
        end=pattern_df.index.max(),
    )


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


def search_for_patterns(motifs: dict) -> pd.DataFrame:
    """Go get those patterns in the SEARCH_DIR"""

    matching_patterns = []

    for file in tqdm(SEARCH_DIR.iterdir()):
        seq_df = pd.read_csv(file, delimiter="\t", header=None)

        for window_size, patterns_to_match in motifs.items():
            for i in range(0, len(seq_df) - window_size):
                subseq = seq_df.iloc[i : i + window_size]
                if not (subseq.iloc[:, 0] > 1.0).any():
                    continue

                subseq_str = get_sequence_str(subseq)
                for ptm in patterns_to_match:
                    assert isinstance(
                        ptm, Pattern
                    ), f"Expected Pattern but got {type(ptm)}"
                    if not ptm.matches_sequence(subseq_str):
                        continue

                    new_pattern = Pattern(
                        sequence=subseq_str,
                        fSHAPE=get_reactive_vector(subseq),
                        file=str(file.name),
                        start=i,
                        end=i + window_size,
                    )
                    new_pattern.associate(ptm)
                    matching_patterns.append(new_pattern.to_dict())

    return pd.DataFrame(matching_patterns)


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

    possible_seqs = process_fshape_files(expected_pattern.seq_len)

    if clusters_path is not None and clusters_path.exists():
        logging.info("Loading clusters from file %s", str(clusters_path))
        seq_clusters = load_clusters(clusters_path)
        logging.info("Clusters loaded succesfully")
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

    logging.info("Aggregating cluster info, reactiveness vectors and DNA sequences")
    aggregated_data = dict()
    for pattern_len in seq_clusters.keys():
        # creating a new df - one row for one sequence
        df = seq_clusters[pattern_len].rename(CLUSTER_COL).to_frame()
        df[REACT_COL] = [get_reactive_vector(seq) for seq in possible_seqs[pattern_len]]
        df[SEQ_COL] = [get_sequence_str(seq) for seq in possible_seqs[pattern_len]]

        # limit the data to only clusters with at least 3 members
        cluster_sizes = df[CLUSTER_COL].value_counts()
        cluster_sizes = cluster_sizes.loc[cluster_sizes > 3]
        if len(cluster_sizes) > 3:
            # limit to the 3 biggest clusters
            cluster_sizes = cluster_sizes.iloc[:3]

        top_clusters = cluster_sizes.index.tolist()
        df = df.loc[df[CLUSTER_COL].isin(top_clusters)]
        aggregated_data[pattern_len] = df.copy()

    logging.info("Looking for motifs with STUMPY")
    motifs = dict()
    for motif_len in tqdm(seq_clusters.keys()):
        consensus_patterns = []
        agg_df = aggregated_data[motif_len]

        for cluster_i in agg_df[CLUSTER_COL].unique():
            one_cluster_df = agg_df.loc[agg_df[CLUSTER_COL] == cluster_i]
            Rs = one_cluster_df[REACT_COL].values
            Ss = one_cluster_df[SEQ_COL].values

            radius, motif_idx, _ = stumpy.ostinato(Rs, motif_len)
            if radius == np.inf:
                logging.warning(
                    "Motif? What's that? Using sequence #%s... (motif_len: %s, cluster_i: %s)",
                    motif_idx,
                    motif_len,
                    cluster_i,
                )
            consensus_patterns.append(
                Pattern(sequence=Ss[motif_idx], fSHAPE=Rs[motif_idx])
            )

        motifs[motif_len] = consensus_patterns

    # add our OG pattern to the mix
    motifs[expected_pattern.seq_len].append(expected_pattern)

    logging.info("Searching for matching patterns in longer files from %s", SEARCH_DIR)
    report = search_for_patterns(motifs)
    report = report.dropna().sort_values(by="aS", ascending=True)
    report.to_csv(f"result_{START_TIMESTAMP}.csv", index=False)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--clusters_path", required=False, default=None, type=Path)
    kwargs = dict(argparser.parse_args()._get_kwargs())
    main(**kwargs)
