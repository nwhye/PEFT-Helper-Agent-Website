import numpy as np
import pandas as pd

EPS = 1e-8

def compute_derived_metrics(predicted_meta: dict) -> dict:
    """
    Compute derived metrics such as stability, gaps, robustness score from predicted numeric metrics.
    Guarantees no NaNs in stability metrics by adding a small EPS to denominators.
    """

    df = pd.DataFrame([predicted_meta])

    # 1. Loss-based stability
    df["loss_stability"] = df["train_loss_last_std"] / (df["train_loss_last_mean"] + EPS)
    df["loss_slope_stability"] = df["loss_slope_std"] / (df["loss_slope_mean"].abs() + EPS)
    df["loss_best_worst_gap"] = df["train_loss_last_max"] - df["train_loss_last_min"]

    # 2. Gradient stability
    df["grad_stability"] = df["gradient_norm_mean_std"] / (df["gradient_norm_mean_mean"] + EPS)
    df["grad_ratio"] = df["gradient_norm_mean_max"] / (df["gradient_norm_mean_min"] + EPS)

    # 3. Eval loss stability
    df["eval_stability"] = df["eval_loss_std"] / (df["eval_loss_mean"] + EPS)
    df["eval_best_worst_gap"] = df["eval_loss_max"] - df["eval_loss_min"]

    # 4. Performance metric stability (ROUGE, BLEU, BERT, exact_match, quality_score)
    for metric in ["rouge1", "rouge2", "rougeL", "bleu", "bert_score", "exact_match", "quality_score"]:
        df[f"{metric}_stability"] = df[f"{metric}_std"] / (df[f"{metric}_mean"].abs() + EPS)
        df[f"{metric}_gap"] = df[f"{metric}_max"] - df[f"{metric}_min"]

    # 5. Overfitting behavior
    df["overfit_mean"] = df["overfit_flag_mean"]
    df["overfit_variance"] = df["overfit_flag_max"] - df["overfit_flag_min"]
    df["loss_diff_train_eval"] = df["train_loss_last_mean"] - df["eval_loss_mean"]

    # 6. Training efficiency
    df["efficiency_stability"] = df["training_efficiency_std"] / (df["training_efficiency_mean"] + EPS)
    df["efficiency_gap"] = df["training_efficiency_max"] - df["training_efficiency_min"]

    # 7. Global robustness score
    df["robustness_score"] = (
        df["loss_stability"] +
        df["grad_stability"] +
        df["eval_stability"] +
        df["rouge1_stability"] +
        df["quality_score_stability"] +
        df["efficiency_stability"]
    )

    derived_meta = df.to_dict(orient="records")[0]

    return derived_meta
