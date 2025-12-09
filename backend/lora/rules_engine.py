import json


def generate_full_recommendations(meta, predicted_values, thresholds_path="utils/recommendation_thresholds.json"):
    """
    Generate hyperparameter recommendations based on predicted metrics and dynamic thresholds.
    meta: dict of model metrics
    predicted_values: tuple(overfit, efficiency, gen_gap)
    thresholds_path: path to the JSON file with precomputed thresholds
    """

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    overfit, efficiency, gen_gap = predicted_values
    recs = []

    loss_stab = meta.get("loss_stability", 0)
    grad_norm = meta.get("gradient_norm_mean_mean", 0)
    grad_ratio = meta.get("grad_ratio", 1)
    eval_loss = meta.get("eval_loss_mean", 0)
    quality = meta.get("quality_score_mean", 0)
    batch_size = meta.get("batch_size", 4)
    lora_r = meta.get("lora_r", 4)
    lora_alpha = meta.get("lora_alpha", 8)
    lora_dropout = meta.get("lora_dropout", 0.05)

    def get_threshold(key, fallback):
        return thresholds.get(key, fallback)

    # Overfitting / Regularization

    if overfit > get_threshold("high_overfit_q75", 0.1):
        recs.append("Increase lora_dropout or reduce lora_r to reduce overfitting")

    if gen_gap > get_threshold("loss_best_worst_gap_q75", 0.35):
        recs.append("Reduce learning_rate or simplify target_modules to improve generalization")

    if loss_stab > get_threshold("loss_stability_q75", 0.08):
        recs.append("Reduce learning_rate or increase lora_dropout for more stable training")

    if meta.get("loss_slope_stability", 0) > get_threshold("loss_slope_stability_q75", 3):
        recs.append("Increase batch_size or decrease learning_rate to stabilize gradients")

    if meta.get("quality_score_gap", 0) > get_threshold("quality_score_gap_q75", 0.015):
        recs.append("Reduce lora_r or adjust target_modules to improve consistency")

    # Training Efficiency

    if efficiency < get_threshold("training_efficiency_mean_q25", 15000):
        recs.append("Increase learning_rate or batch_size to improve training speed")

    if efficiency < get_threshold("training_efficiency_mean_q50", 25000) and loss_stab > get_threshold("loss_stability_q75", 0.08):
        recs.append("Reduce lora_r or lora_alpha to improve efficiency")

    if efficiency > get_threshold("training_efficiency_mean_q75", 40000) and eval_loss > get_threshold("eval_loss_mean_q75", 2.7):
        recs.append("Reduce target_modules or lora_alpha for more controlled training")

    # Gradient / Stability

    if grad_ratio > get_threshold("grad_ratio_q75", 1.15):
        recs.append("Reduce lora_r or simplify target_modules to control gradient imbalance")

    if grad_norm > get_threshold("gradient_norm_mean_mean_q75", 0.22):
        recs.append("Lower lora_alpha or learning_rate to reduce gradient magnitude")

    # Epoch suggestions
    if eval_loss > get_threshold("eval_loss_mean_q75", 2.7):
        recs.append("Increase epoch count for better convergence")

    if efficiency > get_threshold("training_efficiency_mean_q75", 35000):
        recs.append("Reduce epoch count or increase batch_size to shorten training time")

    # LoRA specific adjustments

    if overfit > get_threshold("high_overfit_q75", 0.15) and lora_r > 8:
        recs.append("Reduce lora_r to prevent overfitting")

    if lora_alpha > 12 and grad_norm > get_threshold("gradient_norm_mean_mean_q75", 0.22):
        recs.append("Reduce lora_alpha to improve stability")

    if lora_dropout < 0.05 and overfit > get_threshold("high_overfit_q75", 0.1):
        recs.append("Increase lora_dropout to reduce overfitting")

    # Target Modules

    if quality < get_threshold("quality_score_mean_q25", 0.24):
        recs.append("Adjust target_modules to improve overall output quality")

    if gen_gap > get_threshold("loss_best_worst_gap_q75", 0.35):
        recs.append("Use fewer target_modules to reduce generalization gap")

    # Learning rate / batch size

    if loss_stab > get_threshold("loss_stability_q75", 0.08) and gen_gap > get_threshold("loss_best_worst_gap_q75", 0.35):
        recs.append("Reduce learning_rate to prevent unstable training")

    if efficiency < get_threshold("training_efficiency_mean_q25", 20000) and grad_norm < get_threshold("gradient_norm_mean_mean_q25", 0.15):
        recs.append("Increase learning_rate for better convergence speed")

    if batch_size < 4 and efficiency < get_threshold("training_efficiency_mean_q50", 30000):
        recs.append("Increase batch_size to improve stability")

    if not recs:
        recs.append("Configuration appears stable and efficient; no immediate changes needed")

    return recs
