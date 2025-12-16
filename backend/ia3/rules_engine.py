import json


def _state_from_thresholds(value, q25, q50, q75):
    if value <= q25:
        return "green"
    if value <= q50:
        return "yellow"
    return "red"


def generate_ia3_recommendations(meta: dict, thresholds_path="utils/ia3_recommendation_thresholds.json"):
    with open(thresholds_path, "r") as f:
        T = json.load(f)

    recs = []

    loss_stab = meta.get("loss_stability", 0.0)
    loss_slope_stab = meta.get("loss_slope_stability", 0.0)
    grad_stab = meta.get("grad_stability", 0.0)
    efficiency = meta.get("training_efficiency_mean", 0.0)
    efficiency_gap = meta.get("efficiency_gap", 0.0)
    overfit = meta.get("overfit_flag_mean", 0.0)
    robustness = meta.get("robustness_score", 0.0)
    eval_loss = meta.get("eval_loss_mean", 0.0)
    quality = meta.get("quality_score_mean", 0.0)
    quality_gap = meta.get("quality_score_gap", 0.0)
    loss_gap = meta.get("loss_best_worst_gap", 0.0)

    batch_size = int(meta.get("batch_size", 4))
    epoch = int(meta.get("epoch", 1))
    layers_tuned = int(meta.get("layers_tuned", 0))

    tm_cols = [c for c in meta.keys() if c.startswith("tm_")]
    active_modules = sum([meta.get(c, 0) for c in tm_cols])

    loss_state = _state_from_thresholds(loss_stab, T["loss_stability_q25"], T["loss_stability_q50"],
                                        T["loss_stability_q75"])
    slope_state = _state_from_thresholds(loss_slope_stab, T["loss_slope_stability_q25"], T["loss_slope_stability_q50"],
                                         T["loss_slope_stability_q75"])
    grad_state = _state_from_thresholds(grad_stab, T["grad_stability_q25"], T["grad_stability_q50"],
                                        T["grad_stability_q75"])
    efficiency_state = _state_from_thresholds(efficiency, T["training_efficiency_mean_q25"],
                                              T["training_efficiency_mean_q50"], T["training_efficiency_mean_q75"])

    is_training_healthy = loss_state == "green" and slope_state == "green" and grad_state == "green"

    if loss_state == "red":
        recs.append("Reduce learning rate — IA3 loss is unstable.")
    if slope_state == "red":
        recs.append("Increase batch size or reduce learning rate to stabilize IA3 training dynamics.")
    if grad_state == "red":
        recs.append("Reduce number of IA3 layers or restrict target modules to attention-only.")

    if overfit > T["overfit_flag_mean_q75"]:
        recs.append("Reduce IA3 layers tuned to mitigate overfitting.")

    if loss_gap > T["loss_best_worst_gap_q75"]:
        recs.append("Reduce learning rate or IA3 layer scope to improve generalization.")

    if efficiency < T["training_efficiency_mean_q25"]:
        recs.append("Increase learning rate or batch size — IA3 should train efficiently.")

    if efficiency_gap > T["efficiency_gap_q75"]:
        recs.append("Reduce IA3 layers or epochs — efficiency varies too much across runs.")

    if efficiency > T["training_efficiency_mean_q75"] and eval_loss > T["eval_loss_mean_q50"]:
        recs.append("Reduce IA3 layers — training is fast but not converging well.")

    if quality < T["quality_score_mean_q75"]:
        recs.append("Expand IA3 to additional attention modules (q/k/v) to improve model capacity.")

    if quality_gap > T["quality_score_gap_q75"] and active_modules > 2:
        recs.append("Restrict IA3 target modules to reduce output inconsistency.")

    if eval_loss > T["eval_loss_mean_q50"] and epoch < 3:
        recs.append("Increase epochs — IA3 often requires more steps to converge.")

    if efficiency_state == "red" and batch_size < 8:
        recs.append("Increase batch size for more stable IA3 optimization.")

    if efficiency_state == "green" and epoch > 1:
        recs.append("(Optional) Reduce epochs to save compute — IA3 is already converging well.")

    if robustness < T["robustness_score_q75"]:
        recs.append("Overall robustness is low — reduce IA3 layer count and simplify target modules.")

    if robustness > T["robustness_score_q90"] and quality < T["quality_score_mean_q75"]:
        recs.append("(Optional) Increase IA3 layers or target modules to improve expressiveness.")

    if not recs:
        recs.append("Configuration appears stable, efficient, and robust — no IA3 adjustments needed.")

    return recs
