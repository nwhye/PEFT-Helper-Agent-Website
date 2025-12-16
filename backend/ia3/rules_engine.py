import json
import math


def training_efficiency_state(v):
    if v > 20000:
        return "green"
    if v > 10000:
        return "yellow"
    return "red"


def loss_slope_state(v):
    if v < 0.0005:
        return "green"
    if v < 0.001:
        return "yellow"
    return "red"


def gradient_norm_state(v):
    if v < 3.4:
        return "green"
    if v < 4.3:
        return "yellow"
    return "red"


def generate_ia3_recommendations(
    meta,
    predicted_values,
    thresholds_path="utils/ia3_recommendation_thresholds.json",
):
    """
    Generate IA3 hyperparameter suggestions based on predicted metrics:
    - training_efficiency
    - loss_slope
    - gradient_norm

    Fully aligned with frontend GREEN / YELLOW / RED logic.
    """

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    training_efficiency = predicted_values.get("training_speed", 0.0)
    loss_slope = predicted_values.get("loss_slope", 0.0)
    gradient_norm = predicted_values.get("gradient_norm", 0.0)

    batch_size = int(meta.get("batch_size", 4))
    epoch = int(meta.get("epoch", 1))
    layers_tuned = int(meta.get("layers_tuned", 0))
    target_modules = meta.get("target_modules", ["q", "k", "v"])
    learning_rate = float(meta.get("learning_rate", 1e-6))

    num_modules = len(target_modules)

    recs = []

    efficiency_state = training_efficiency_state(training_efficiency)
    slope_state = loss_slope_state(loss_slope)
    grad_state = gradient_norm_state(gradient_norm)

    is_training_healthy = (
        efficiency_state == "green"
        and slope_state == "green"
        and grad_state == "green"
    )


    if efficiency_state == "red":
        if batch_size >= 16:
            recs.append("Consider decreasing batch size to improve IA3 training efficiency.")
        elif batch_size < 16 or learning_rate < 1e-6:
            recs.append("Increase batch size or slightly increase learning rate to improve IA3 efficiency.")

    if slope_state == "red":
        if batch_size >= 8 or target_modules == ["q", "k", "v", "o"]:
            recs.append("Consider decreasing batch size or set of target modules to improve IA3 stability.")
        elif epoch < 3:
            recs.append("Increase epochs or reduce learning rate for more stable IA3 training.")

    if grad_state == "red":
        if batch_size >= 8 or epoch > 64:
            recs.append("Consider decreasing batch size or epoch count to reduce IA3 gradient spikes.")
        elif layers_tuned > 1:
            recs.append("Reduce number of IA3 layers to stabilize gradients.")


    if efficiency_state == "yellow":
        if batch_size >= 8:
            recs.append("Consider decreasing batch size slightly to improve IA3 efficiency.")
        elif batch_size < 8:
            recs.append("Consider very slightly increasing batch size to improve IA3 efficiency.")

    if slope_state == "yellow":
        if batch_size >= 16:
            recs.append("Consider decreasing batch size to improve IA3 stability.")
        elif epoch < 12:
            recs.append("Slightly increase epochs to improve IA3 loss stability.")

    if grad_state == "yellow":
        if batch_size >= 16 or epoch > 32:
            recs.append("Consider slightly decreasing batch size or epoch count to improve gradients.")
        elif layers_tuned > 2:
            recs.append("Consider reducing IA3 layers for extra gradient stability.")
        else:
            recs.append("Consider slightly increasing batch size or reduce target modules to improve gradient stability.")


    if efficiency_state == "green":
        if epoch > 1:
            recs.append("(Optional) Reduce epochs if you want to improve training efficiency further.")

    if grad_state == "red" or slope_state == "red":
        if layers_tuned > 1:
            recs.append("Reduce IA3 layers to stabilize training gradients or loss.")

    if is_training_healthy:
        if layers_tuned < 8:
            recs.append("(Optional) Consider increasing IA3 layers to improve model capacity.")

    if grad_state == "red" and slope_state == "red":
        recs.append(
            "Both gradient spikes and unstable loss detected; consider increasing batch size and reducing epochs."
        )


    if not recs:
        recs.append(
            "Configuration appears stable and efficient; no immediate IA3 changes needed."
        )

    return recs
