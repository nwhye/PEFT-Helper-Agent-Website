import json


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


def generate_prefix_recommendations(meta, predicted_values, thresholds_path="utils/recommendation_thresholds.json"):
    """
    Generate hyperparameter recommendations for Prefix-tuning based on predicted metrics:
    - training_speed
    - loss_slope
    - gradient_norm

    meta: dict of user-provided config and metrics
    predicted_values: dict containing {"training_speed", "loss_slope", "gradient_norm"}
    thresholds_path: JSON file with thresholds for dynamic rules
    """

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    training_speed = predicted_values.get("training_speed", 0.0)
    loss_slope = predicted_values.get("loss_slope", 0.0)
    gradient_norm = predicted_values.get("gradient_norm", 0.0)

    batch_size = int(meta.get("batch_size", 4))
    epoch = int(meta.get("epoch", 1))
    prefix_length = meta.get("prefix_length", "all")
    prefix_hidden = int(meta.get("prefix_hidden", 128))
    prefix_dropout = float(meta.get("prefix_dropout", 0.05))
    learning_rate = float(meta.get("learning_rate", 1e-6))
    layers_tuned = meta.get("layers_tuned", 1)
    target_modules = meta.get("target_modules", ["q", "v"])

    num_modules = len(target_modules)
    recs = []

    speed_state = training_speed_state(training_speed)
    slope_state = loss_slope_state(loss_slope)
    grad_state = gradient_norm_state(gradient_norm)
    is_training_healthy = speed_state == "green" and slope_state == "green" and grad_state == "green"

    if speed_state == "red":
        if batch_size >= 15:
            recs.append("Consider decreasing batch size to improve training speed.")
        else:
            recs.append("Increase batch size or slightly increase learning rate to improve training speed.")
    elif speed_state == "yellow":
        if batch_size < 7:
            recs.append("Consider slightly increasing batch size to improve training speed.")
        else:
            recs.append("Consider slightly decreasing batch size to improve training speed.")

    if slope_state == "red":
        if batch_size >= 7:
            recs.append("Consider decreasing batch size to improve stability.")
        else:
            recs.append("Increase prefix_dropout, epochs, or reduce learning rate for more stable training.")
    elif slope_state == "yellow":
        if epoch < 12:
            recs.append("Slightly increase epochs to improve training stability.")

    if grad_state == "red":
        if batch_size >= 7 or epoch > 64:
            recs.append("Consider decreasing batch size or epoch count to reduce gradient spikes.")
        elif prefix_hidden > 128:
            recs.append("Reduce prefix_hidden to reduce gradient magnitude.")
    elif grad_state == "yellow":
        if batch_size >= 16 or epoch > 32:
            recs.append("Consider slightly decreasing batch size or epoch count to improve gradient stability.")
        elif prefix_hidden > 128:
            recs.append("Consider slightly reducing prefix_hidden for gradient stability.")

    if prefix_length != "all" and grad_state == "red":
        recs.append("Use full prefix ('all') to stabilize training and reduce gradient spikes.")
    if layers_tuned > 1 and grad_state == "red":
        recs.append("Reduce number of layers tuned to improve stability.")

    quality_gap = meta.get("quality_score_gap", 0.0)
    if not is_training_healthy and quality_gap > thresholds.get("quality_score_gap_q75", 0.02) and num_modules > 2:
        recs.append("Reduce target modules to improve training stability.")
    if is_training_healthy and quality_gap < thresholds.get("quality_score_gap_q25", 0.011) and num_modules < 2:
        recs.append("Consider expanding target modules to improve model capacity.")

    if not recs:
        recs.append("Configuration appears stable and efficient; no immediate changes needed.")

    return recs
