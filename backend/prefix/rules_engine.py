import json


def generate_prefix_recommendations(meta, predicted_values, thresholds_path="utils/recommendation_thresholds.json"):
    """
    Generate hyperparameter recommendations for prefix-tuning based on predicted metrics and dynamic thresholds.

    meta: dict of model metrics
    predicted_values: tuple(overfit_mean, training_efficiency_mean, loss_diff_train_eval)
    thresholds_path: path to JSON file with precomputed thresholds
    """

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    overfit, efficiency, gen_gap = predicted_values
    recs = []

    loss_stab = meta.get("loss_stability", 0)
    loss_slope_stab = meta.get("loss_slope_stability", 0)
    quality_gap = meta.get("quality_score_gap", 0)
    grad_norm = meta.get("gradient_norm_mean_mean", 0)
    grad_ratio = meta.get("grad_ratio", 1)
    eval_loss = meta.get("eval_loss_mean", 0)
    quality = meta.get("quality_score_mean", 0)
    batch_size = meta.get("batch_size", 4)
    prefix_len = meta.get("prefix_length", "all")
    layer_scope = meta.get("layer_scope", None)
    prefix_hidden = meta.get("prefix_hidden", 128)

    def get_threshold(key, fallback):
        return thresholds.get(key, fallback)

    # Overfitting / Stability Rules
    if overfit > get_threshold("overfit_flag_mean_q75", 0.1):
        recs.append("Reduce prefix_length or layer_tuned (scope) to reduce overfitting")
    if gen_gap > get_threshold("loss_best_worst_gap_q75", 0.35):
        recs.append("Reduce learning_rate or simplify prefix design (Prefix length, Layers tuned,Prefix Hidden) to improve generalization")
    if loss_stab > get_threshold("loss_stability_q75", 0.1):
        recs.append("Reduce learning_rate or increase prefix_hidden for more stable training")
    if loss_slope_stab > get_threshold("loss_slope_stability_q75", 4.0):
        recs.append("Increase batch_size or decrease learning_rate to stabilize gradients")
    if quality_gap > get_threshold("quality_score_gap_q75", 0.025):
        recs.append("Adjust prefix_length or layer_tuned (scope) to improve output consistency")

    # Training Efficiency Rules
    if efficiency < get_threshold("training_efficiency_mean_q25", 5000):
        recs.append("Increase learning_rate or batch_size to improve training speed")
    if efficiency < get_threshold("training_efficiency_mean_q50", 10000) and loss_stab > get_threshold(
            "loss_stability_q75", 0.1):
        recs.append("Reduce prefix_hidden or layer_tuned (scope) to improve efficiency")
    if efficiency > get_threshold("training_efficiency_mean_q75", 30000) and eval_loss > get_threshold(
            "eval_loss_mean_q75", 2.5):
        recs.append("Reduce prefix_length or layer_tuned (scope) for more controlled training")

    # Gradient / Norm Rules
    if grad_ratio > get_threshold("grad_ratio_q75", 1.15):
        recs.append("Reduce prefix_hidden or layer_tuned (scope) to control gradient imbalance")
    if grad_norm > get_threshold("gradient_norm_mean_mean_q75", 0.22):
        recs.append("Lower learning_rate or prefix_hidden to reduce gradient magnitude")

    # Epoch / Convergence Rules
    if eval_loss > get_threshold("eval_loss_mean_q75", 2.7):
        recs.append("Increase number of epochs for better convergence")
    if efficiency > get_threshold("training_efficiency_mean_q75", 35000):
        recs.append("Reduce epochs or increase batch_size to shorten training time")

    # Prefix Tuning Specific Adjustments
    if overfit > get_threshold("overfit_flag_mean_q75", 0.15) and prefix_len != "all":
        recs.append("Use full prefix ('all') to stabilize training and reduce overfitting")
    if layer_scope and layer_scope > 128 and grad_norm > get_threshold("gradient_norm_mean_mean_q75", 0.22):
        recs.append("Reduce layer_tuned (scope) to improve stability")
    if prefix_hidden < 128 and overfit > get_threshold("overfit_flag_mean_q75", 0.1):
        recs.append("Increase prefix_hidden to reduce overfitting")

    # Output Quality Rules
    if quality < get_threshold("quality_score_mean_q25", 0.24):
        recs.append("Increase prefix_hidden or adjust prefix_length to improve output quality")
    if gen_gap > get_threshold("loss_best_worst_gap_q75", 0.35):
        recs.append("Simplify prefix scope (reduce layer_tuned (scope)) to reduce generalization gap")

    # Learning rate / batch size rules
    if loss_stab > get_threshold("loss_stability_q75", 0.1) and gen_gap > get_threshold("loss_best_worst_gap_q75",
                                                                                        0.35):
        recs.append("Reduce learning_rate to prevent unstable training")
    if efficiency < get_threshold("training_efficiency_mean_q25", 5000) and grad_norm < get_threshold(
            "gradient_norm_mean_mean_q25", 0.15):
        recs.append("Increase learning_rate for faster convergence")
    if batch_size < 4 and efficiency < get_threshold("training_efficiency_mean_q50", 10000):
        recs.append("Increase batch_size to improve stability")

    if not recs:
        recs.append("Configuration appears stable and efficient; no immediate changes needed")

    return recs
