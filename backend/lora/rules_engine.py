import json
import math


def training_speed_state(v):
    if v > 60000:
        return "green"
    if v > 30000:
        return "yellow"
    return "red"


def loss_slope_state(v):
    v = abs(v)
    if v < 0.0005:
        return "green"
    if v < 0.001:
        return "yellow"
    return "red"


def gradient_norm_state(v):
    if v < 0.2:
        return "green"
    if v < 0.3:
        return "yellow"
    return "red"


def generate_full_recommendations(
    meta,
    predicted_values,
    thresholds_path="utils/recommendation_thresholds.json"
):
    """
    Generate hyperparameter suggestions based on predicted metrics:
    - training_speed
    - loss_slope
    - gradient_norm

    Fully aligned with frontend GREEN / YELLOW / RED logic.
    """

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    training_speed = predicted_values.get("training_speed", 0.0)
    loss_slope = predicted_values.get("loss_slope", 0.0)
    gradient_norm = predicted_values.get("gradient_norm", 0.0)

    batch_size = int(meta.get("batch_size", 4))
    epoch = int(meta.get("epoch", 1))
    lora_r = int(meta.get("lora_r", 4))
    lora_alpha = float(meta.get("lora_alpha", 8))
    lora_dropout = float(meta.get("lora_dropout", 0.05))
    learning_rate = float(meta.get("learning_rate", 1e-6))
    target_modules = meta.get("target_modules", ["q", "v"])

    print(batch_size, epoch, lora_r, lora_alpha, lora_dropout, learning_rate, target_modules)

    num_modules = len(target_modules)

    recs = []
    ##suggested_params = set()

    speed_state = training_speed_state(training_speed)
    slope_state = loss_slope_state(loss_slope)
    grad_state = gradient_norm_state(gradient_norm)

    is_training_healthy = (
        speed_state == "green"
        and slope_state == "green"
        and grad_state == "green"
    )

    if speed_state == "red":
        if batch_size >= 15:
            recs.append(f"Consider decreasing batch size to improve training speed..")
        elif batch_size < 15 or learning_rate < 1e-6:
            recs.append("Increase batch size or slightly increase learning rate to improve training speed.")

    if slope_state == "red":
        if batch_size >= 7:
            recs.append(f"Consider decreasing batch size to improve stability.")
        elif lora_dropout < 0.1 or epoch < 3:
            recs.append("Increase LoRA dropout, epochs, or reduce learning rate for more stable training.")

    if grad_state == "red":
        if batch_size >= 7 or epoch > 64:
            recs.append(f"Consider decreasing batch size or epoch count to improve gradient spikes.")
        elif lora_r > 1 or lora_alpha > 8:
            recs.append("Reduce LoRA rank or LoRA alpha to reduce gradient spikes.")

    if speed_state == "yellow":
        if batch_size >= 7:
            recs.append(f"Consider decreasing batch size to improve training speed..")
        elif batch_size < 7:
            recs.append("Consider very slightly increasing batch size to improve training speed.")

    if slope_state == "yellow":
        if batch_size >= 16:
            recs.append(f"Consider decreasing batch size to improve stability.")
        elif epoch < 12:
            recs.append("Slightly increase epochs to improve training stability.")

    if grad_state == "yellow":
        if batch_size >= 16 or epoch > 32:
            recs.append(f"Consider slightly decreasing batch size or epoch count to improve gradients.")
        elif lora_alpha > 12:
            recs.append("Consider lowering LoRA alpha for extra gradient stability.")

    if speed_state == "green":
        if epoch > 1:
            recs.append("(Optional) Reduce epochs if you want to improve training speed even further.")

    quality_gap = meta.get("quality_score_gap", 0.0)

    if (
        not is_training_healthy
        and quality_gap > thresholds.get("quality_score_gap_q75", 0.02)
        and num_modules > 2
    ):
        recs.append("Reduce target modules to improve training stability.")

    if (
        is_training_healthy
        and quality_gap < thresholds.get("quality_score_gap_q25", 0.011)
        and num_modules < 2
    ):
        recs.append("Consider expanding target modules to improve model capacity.")

    if grad_state == "red" or slope_state == "red":
        if lora_r > 1:
            recs.append("Reduce LoRA rank to stabilize training gradients or loss.")

    if is_training_healthy and quality_gap < thresholds.get("quality_score_gap_q25", 0.011):
        if lora_r < 16:  # upper limit example
            recs.append("(Optional) Consider increasing LoRA rank to improve model capacity.")

    if grad_state == "red" or slope_state == "red":
        if lora_alpha > 8:
            recs.append("Reduce LoRA alpha to prevent gradient spikes or unstable loss.")

    if is_training_healthy and slope_state != "red" and quality_gap < thresholds.get("quality_score_gap_q25", 0.011):
        if lora_alpha < 12:
            recs.append("(Optional) Consider increasing LoRA alpha to improve model learning capacity.")

    if slope_state == "red" or slope_state == "yellow":
        if lora_dropout < 0.1:
            recs.append("Increase LoRA dropout to stabilize training.")

    if is_training_healthy and slope_state == "green":
        if lora_dropout > 0.05:
            recs.append("(Optional) Consider slightly reducing LoRA dropout for faster convergence.")

    if grad_state == "red" and slope_state == "red":
        recs.append(
            "Both gradient spikes and unstable loss detected; consider reducing LoRA rank and alpha, and increasing dropout.")

    if not recs:
        recs.append(
            "Configuration appears stable and efficient; no immediate changes needed."
        )

    return recs
