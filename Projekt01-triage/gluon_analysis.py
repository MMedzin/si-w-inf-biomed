from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

DATA_DIR = Path("../data/P1")
AP_DATA = DATA_DIR / "ap_pro_data.xls"
AE_DATA = DATA_DIR / "ae_retro_data.xlsx"
HP_DATA = DATA_DIR / "hp_retro_data.xls"
SP_DATA = DATA_DIR / "sp_retro_data.xls"

RESULTS_DIR = Path("results")

SEED = 42
N_SPLITS = 10
N_REPEATS = 5
CV_SCHEME = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
)


def load_ap_data() -> tuple[pd.DataFrame, str]:
    ap_data = pd.read_excel(
        AP_DATA,
        sheet_name="Physicians - Discretized",
        header=2,
        index_col=1,
    ).drop(columns=["VisitID", "Observer"])
    label_col = "Triage"
    return ap_data, label_col


def load_ae_data() -> tuple[pd.DataFrame, str]:
    ae_data = pd.read_excel(
        AE_DATA,
        sheet_name="Discretized",
    ).drop(columns=["DECISION_OFFSET"])
    label_col = "CORR_CATEGORY"
    return ae_data, label_col


def load_hp_data() -> tuple[pd.DataFrame, str]:
    hp_data = pd.read_excel(
        HP_DATA,
        sheet_name="Discretized Data (Final)",
        header=2,
        index_col=0,
    )
    label_col = "TRIAGE"
    return hp_data, label_col


def load_sp_data() -> tuple[pd.DataFrame, str]:
    sp_data = pd.read_excel(
        SP_DATA,
        sheet_name="Discretized Data (Final)",
        header=2,
        index_col=0,
    )
    label_col = "TRIAGE"
    return sp_data, label_col


def preprocess_missing_values(data: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Method replacing `?`, which represent missing values in analysed datasets witn `None` and
    removing samples without a label.
    """
    df = data.applymap(lambda x: x.strip() if isinstance(x, str) else x).replace(
        "?", None
    )
    return df.loc[~df.loc[:, label_col].isna()]


def main() -> None:
    data_loaders = {
        "AP": load_ap_data,
        "AE": load_ae_data,
        "HP": load_hp_data,
        "SP": load_sp_data,
    }

    for name, loader in tqdm(data_loaders.items()):
        data, label_col = loader()
        data = preprocess_missing_values(data, label_col)
        data_train, data_test = train_test_split(data, test_size=0.3, random_state=SEED)
        for i, (train_index, valid_index) in tqdm(
            enumerate(
                CV_SCHEME.split(
                    data_train.drop(columns=label_col), data_train.loc[:, label_col]
                )
            ),
            total=N_REPEATS * N_SPLITS,
            leave=False,
        ):
            train_data = TabularDataset(data_train.iloc[train_index])
            validation_data = TabularDataset(data_train.iloc[valid_index])
            predictor = TabularPredictor(
                label=label_col, eval_metric="roc_auc_ovo_macro"
            ).fit(train_data=train_data)
            # predictions = predictor.predict(validation_data)
            leaderboard = predictor.leaderboard(validation_data)
            leaderboard["cv_iter"] = i
            leaderboard.to_csv(
                RESULTS_DIR / f"{name}_leaderboard.csv", header=(i == 0), mode="a"
            )

        results = pd.read_csv(RESULTS_DIR / f"{name}_leaderboard.csv", index_col=0)
        results = results.reset_index(drop=True)
        results.to_csv(RESULTS_DIR / f"{name}_leaderboard.csv")

        summary_results = (
            results.loc[:, ["model", "score_test", "score_val"]]
            .groupby("model")
            .describe()
        )
        summary_results.columns = summary_results.columns.map("_".join)
        summary_results.to_csv(RESULTS_DIR / f"{name}_leaderboard_summary.csv")
        best_model = summary_results.score_test_mean.sort_values(ascending=False).index[
            0
        ]

        results.loc[:, ["model", "score_test"]].rename(
            columns={"score_test": "roc_auc_ovo_macro_test"}
        ).boxplot(column="roc_auc_ovo_macro_test", by="model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{name}_models_score_test_boxplot.png")

        predictor = TabularPredictor(
            label=label_col, eval_metric="roc_auc_ovo_macro"
        ).fit(train_data=data_train)
        predictor.set_model_best(best_model)
        predictor.refit_full()
        pred_proba = predictor.predict_proba(data_test)
        pred = predictor.predict(data_test)

        holdout_results = pd.Series(
            {
                "model": best_model,
                "params": predictor.info()["model_info"][best_model]["hyperparameters"],
                "roc_auc": roc_auc_score(
                    data_test.loc[:, label_col], pred_proba, multi_class="ovo"
                ),
                "accuracy": accuracy_score(data_test.loc[:, label_col], pred),
                "recall": recall_score(
                    data_test.loc[:, label_col], pred, average="macro"
                ),
                "precision": precision_score(
                    data_test.loc[:, label_col], pred, average="macro"
                ),
            },
            name="Holdout results",
        )
        holdout_results.to_csv(RESULTS_DIR / f"{name}_best_model_holdout_results.csv")


if __name__ == "__main__":
    main()
