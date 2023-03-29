from typing import Optional
import numpy as np
import pandas as pd
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json


def metric(seq_a: np.array, seq_b: np.array) -> float:
    ...


def plot_clustering(
    X: pd.DataFrame,
    clusters: pd.Series,
    show: bool = True,
    **scatterplot_kwargs,
):
    umap_embedding = pd.DataFrame(
        UMAP(n_components=2, densmap=True).fit_transform(X), index=X.index
    )
    sns.scatterplot(data=umap_embedding, x=0, y=1, hue=clusters, **scatterplot_kwargs)


def prepare_seq_list_for_clustering(
    seqs: list[pd.DataFrame], col_num: int = 0
) -> pd.DataFrame:
    return (
        pd.concat(
            [seq.iloc[:, col_num].to_frame().reset_index(drop=True) for seq in seqs],
            axis=1,
        )
        .transpose()
        .reset_index(drop=True)
        .fillna(0)
    )


def save_clusters(seq_clusters: dict[int, pd.Series], path: Path) -> None:
    saving_dict = {
        seq_len: clusters.to_dict() for seq_len, clusters in seq_clusters.items()
    }
    with open(path, "w") as file:
        json.dump(saving_dict, file)


def load_clusters(path: Path) -> dict[int, pd.Series]:
    with open(path, "r") as file:
        loaded_dict = json.load(file)
    return {
        int(seq_len): pd.Series(clusters_dict)
        for seq_len, clusters_dict in loaded_dict.items()
    }
