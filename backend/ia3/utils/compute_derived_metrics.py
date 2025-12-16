import numpy as np
import pandas as pd

EPS = 1e-8


def compute_derived_metrics_ia3(predicted_meta: dict) -> dict:
    """
    IA3 version of derived metric computation.
    Computes stability, gap, and robustness metrics for IA3 based on
    the aggregated experiment CSV structure.

    Metrics:
    - Stability: std / mean (with EPS added for stability)
    - Gap: max - min
    - Robustness score
    """

    df = pd.DataFrame([predicted_meta])

    df["loss_stability"] = df["train_loss_last_std"] / (df["train_loss_last_mean"] + EPS)
    df["loss_slope_stability"] = df["loss_slope_std"] / (df["loss_slope_mean"].abs() + EPS)
    df["loss_best_worst_gap"] = df["train_loss_last_max"] - df["train_loss_last_min"]

    df["grad_stability"] = df["gradient_norm_mean_std"] / (df["gradient_norm_mean_mean"] + EPS)
    df["grad_ratio"] = df["gradient_norm_mean_max"] / (df["gradient_norm_mean_min"] + EPS)

    df["eval_stability"] = df["eval_loss_std"] / (df["eval_loss_mean"] + EPS)
    df["eval_best_worst_gap"] = df["eval_loss_max"] - df["eval_loss_min"]

    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bert_score", "exact_match", "quality_score"]
    for metric in metrics:
        df[f"{metric}_stability"] = df[f"{metric}_std"] / (df[f"{metric}_mean"].abs() + EPS)
        df[f"{metric}_gap"] = df[f"{metric}_max"] - df[f"{metric}_min"]

    df["overfit_mean"] = df["overfit_flag_mean"]
    df["overfit_variance"] = df["overfit_flag_max"] - df["overfit_flag_min"]

    df["loss_diff_train_eval"] = df["train_loss_last_mean"] - df["eval_loss_mean"]

    df["efficiency_stability"] = df["training_efficiency_std"] / (df["training_efficiency_mean"] + EPS)
    df["efficiency_gap"] = df["training_efficiency_max"] - df["training_efficiency_min"]

    df["robustness_score"] = (
                                     df["loss_stability"] +
                                     df["grad_stability"] +
                                     df["eval_stability"] +
                                     df["rouge1_stability"] +
                                     df["quality_score_stability"] +
                                     df["efficiency_stability"]
                             ) / 6.0

    return df.to_dict(orient="records")[0]
