import numpy as np
import pandas as pd

EPS = 1e-8


def compute_derived_metrics(meta: dict) -> dict:
    """
    Compute derived stability, gap, and robustness metrics
    for Prefix Tuning, aligned with LoRA training dynamics.
    """

    df = pd.DataFrame([meta])

    df["loss_stability"] = (
        df["train_loss_last_std"] /
        (df["train_loss_last_mean"].abs() + EPS)
    )

    df["loss_slope"] = df["loss_slope_mean"]

    df["loss_slope_stability"] = (
        df["loss_slope_std"] /
        (df["loss_slope_mean"].abs() + EPS)
    )

    df["loss_best_worst_gap"] = (
        df["train_loss_last_max"] -
        df["train_loss_last_min"]
    )

    df["gradient_norm"] = df["gradient_norm_mean_mean"]

    df["grad_stability"] = (
        df["gradient_norm_mean_std"] /
        (df["gradient_norm_mean_mean"].abs() + EPS)
    )

    df["grad_ratio"] = (
        df["gradient_norm_mean_max"] /
        (df["gradient_norm_mean_min"] + EPS)
    )

    df["eval_stability"] = (
        df["eval_loss_std"] /
        (df["eval_loss_mean"].abs() + EPS)
    )

    df["eval_best_worst_gap"] = (
        df["eval_loss_max"] -
        df["eval_loss_min"]
    )

    for metric in [
        "rouge1", "rouge2", "rougeL",
        "bleu", "bert_score",
        "exact_match", "quality_score"
    ]:
        df[f"{metric}_stability"] = (
            df[f"{metric}_std"] /
            (df[f"{metric}_mean"].abs() + EPS)
        )
        df[f"{metric}_gap"] = (
            df[f"{metric}_max"] -
            df[f"{metric}_min"]
        )

    df["training_speed"] = df["training_efficiency_mean"]

    df["training_speed_stability"] = (
        df["training_efficiency_std"] /
        (df["training_efficiency_mean"].abs() + EPS)
    )

    df["loss_diff_train_eval"] = (
        df["train_loss_last_mean"] -
        df["eval_loss_mean"]
    )

    df["robustness_score"] = (
        df["loss_stability"] +
        df["loss_slope_stability"] +
        df["grad_stability"] +
        df["eval_stability"] +
        df["training_speed_stability"]
    ) / 5.0

    return df.to_dict(orient="records")[0]
