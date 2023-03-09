from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

DATA_DIR = Path("../data/P1")
AP_DATA = DATA_DIR / "ap_pro_data.xls"
SEED = 42
N_SPLITS = 10
N_REPEATS = 5
CV_SCHEME = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
)


def main() -> None:
    ap_data = pd.read_excel(
        AP_DATA,
        sheet_name="Physicians - Discretized",
        header=2,
        index_col=1,
    ).drop(columns=["VisitID", "Observer"])
    label_col = "Triage"

    ap_train, ap_test = train_test_split(ap_data, test_size=0.3, random_state=SEED)
    for i, (train_index, valid_index) in tqdm(
        enumerate(
            CV_SCHEME.split(
                ap_train.drop(columns=label_col), ap_train.loc[:, label_col]
            )
        ),
        total=N_REPEATS * N_SPLITS,
    ):
        train_data = TabularDataset(ap_train.iloc[train_index])
        validation_data = TabularDataset(ap_train.iloc[valid_index])
        predictor = TabularPredictor(
            label=label_col, eval_metric="roc_auc_ovo_macro"
        ).fit(train_data=train_data)
        # predictions = predictor.predict(validation_data)
        leaderboard = predictor.leaderboard(validation_data)
        leaderboard["cv_iter"] = i
        leaderboard.to_csv("AP_Leaderboard.csv", header=(i == 0), mode="a")

    results = pd.read_csv("AP_Leaderboard.csv", index_col=0)
    results = results.reset_index(drop=True)
    results.to_csv("AP_Leaderboard.csv")

    summary_results = (
        results.loc[:, ["model", "score_test", "score_val"]].groupby("model").describe()
    )
    summary_results.columns = summary_results.columns.map("_".join)
    summary_results.to_csv("AP_Leaderboard_summary.csv")
    best_model = summary_results.score_test_mean.sort_values(ascending=False).index[0]

    results.loc[:, ["model", "score_test"]].rename(
        columns={"score_test": "roc_auc_ovo_macro_test"}
    ).boxplot(column="roc_auc_ovo_macro_test", by="model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("Models_score_test_boxplot.png")

    predictor = TabularPredictor(label=label_col, eval_metric="roc_auc_ovo_macro").fit(
        train_data=ap_train
    )
    predictor.set_model_best(best_model)
    predictor.refit_full()
    pred_proba = predictor.predict_proba(ap_test)
    pred = predictor.predict(ap_test)

    holdout_results = pd.Series(
        {
            "model": best_model,
            "params": predictor.info()["model_info"][best_model]["hyperparameters"],
            "roc_auc": roc_auc_score(
                ap_test.loc[:, label_col], pred_proba, multi_class="ovo"
            ),
            "accuracy": accuracy_score(ap_test.loc[:, label_col], pred),
            "recall": recall_score(ap_test.loc[:, label_col], pred, average="macro"),
            "precision": precision_score(
                ap_test.loc[:, label_col], pred, average="macro"
            ),
        },
        name="Holdout results",
    )
    holdout_results.to_csv("Best_model_holdout_results.csv")


if __name__ == "__main__":
    main()
