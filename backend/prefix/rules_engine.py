import json


def training_speed_state(v):
    if v > 25000:   # roughly q75
        return "green"
    if v > 8000:    # roughly q50
        return "yellow"
    return "red"


def loss_slope_state(v):
    if v < 0.0005:   # q50 is ~0.000486
        return "green"
    if v < 0.0015:   # covers q25 to q75 range
        return "yellow"
    return "red"


def gradient_norm_state(v):
    if v < 0.35:   # below q50
        return "green"
    if v < 0.43:   # q75
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
    target_modules = meta.get("target_modules", ["q", "v"])

    num_modules = len(target_modules)
    recs = []

    speed_state = training_speed_state(training_speed)
    slope_state = loss_slope_state(loss_slope)
    grad_state = gradient_norm_state(gradient_norm)
    is_training_healthy = speed_state == "green" and slope_state == "green" and grad_state == "green"

    if speed_state == "red":
        if batch_size >= 16:
            recs.append("Consider decreasing batch size to improve training speed.")
        else:
            recs.append("Increase batch size or slightly increase learning rate to improve training speed.")
            if learning_rate < 1e-6:
                recs.append("Increase learning rate slightly to accelerate training.")
            if prefix_dropout > 0.05:
                recs.append("Consider reducing prefix_dropout to improve training speed.")
    elif speed_state == "yellow":
        if batch_size < 8:
            recs.append("Consider slightly increasing batch size to improve training speed.")
        else:
            recs.append("Consider slightly decreasing batch size to improve training speed.")
            if learning_rate < 1e-6:
                recs.append("Slightly increase learning rate to optimize training speed.")
            if prefix_dropout > 0.05:
                recs.append("Slightly reduce prefix_dropout to improve training speed.")

    if slope_state == "red":
        if batch_size >= 8:
            recs.append("Consider decreasing batch size to improve stability.")
        else:
            recs.append("Increase prefix_hidden, epochs, or reduce learning rate for more stable training.")
        if prefix_hidden < 128:
            recs.append("Increase prefix_hidden to improve training stability and reduce loss slope.")
        if learning_rate > 1e-6:
            recs.append("Reduce learning rate slightly to stabilize loss slope.")
        if prefix_dropout < 0.05:
            recs.append("Slightly increase prefix_dropout to stabilize training.")
    elif slope_state == "yellow":
        if epoch < 12:
            recs.append("Slightly increase epochs to improve training stability.")
        if prefix_dropout < 0.05:
            recs.append("Slightly increase prefix_dropout to smooth training.")
        if learning_rate > 1e-6:
            recs.append("Slightly reduce learning rate to improve stability.")

    if grad_state == "red":
        if batch_size >= 8 or epoch > 64:
            recs.append("Consider decreasing batch size or epoch count to reduce gradient spikes.")
        if prefix_hidden > 128:
            recs.append("Reduce prefix_hidden to reduce gradient magnitude.")
        if prefix_hidden < 128:
            recs.append("Increase prefix_hidden to improve gradient handling and stability.")
        if learning_rate > 1e-6:
            recs.append("Reduce learning rate to control gradient spikes.")
        if prefix_dropout < 0.05:
            recs.append("Increase prefix_dropout slightly to stabilize gradients.")
    elif grad_state == "yellow":
        if batch_size >= 16 or epoch > 32:
            recs.append("Consider slightly decreasing batch size or epoch count to improve gradient stability.")
        elif prefix_hidden > 128:
            recs.append("Consider slightly reducing prefix_hidden for gradient stability.")
        elif prefix_hidden < 128:
            recs.append("Consider slightly increasing prefix_hidden for better gradient control.")
        if learning_rate > 1e-6:
            recs.append("Slightly reduce learning rate for better gradient stability.")
        if prefix_dropout < 0.05:
            recs.append("Slightly increase prefix_dropout for smoother gradients.")

    if prefix_length < 32 and grad_state in ["red", "yellow"]:
        recs.append("Increase prefix length to stabilize training and reduce gradient issues.")

    if not recs:
        recs.append("Configuration appears stable and efficient; no immediate changes needed.")

    return recs
