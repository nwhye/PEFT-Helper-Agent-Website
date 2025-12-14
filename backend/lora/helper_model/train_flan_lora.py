import os
import csv
from copy import deepcopy

import torch
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig

MODEL_NAME = "google/flan-t5-base"
OUTPUT_FILE = "output/flan_lora_grid_with_seed.csv"
MAX_SAMPLES = 500

LEARNING_RATES = [1e-6, 3e-6, 1e-5]
BATCH_SIZES = [4, 8]
EPOCHS = [1, 3]

LORA_R = [4, 8, 16]
LORA_ALPHA = [8, 16]
LORA_DROPOUT = [0.05, 0.1]
TARGET_MODULES = [["q","v"], ["q","v","o"]]
LAYERS_TUNED = ["all", "encoder_last_3", "decoder_last_3"]

fieldnames = [
    "model_name", "peft", "lora_r", "lora_alpha", "lora_dropout",
    "target_modules", "layers_tuned",
    "learning_rate", "batch_size", "epoch", "train_loss_first", "train_loss_last",
    "loss_slope", "gradient_norm_mean", "learning_rate_final", "eval_loss",
    "rouge1", "rouge2", "rougeL", "bleu", "bert_score",
    "exact_match", "quality_score", "overfit_flag", "training_efficiency", "seed",
]

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
write_header = not os.path.exists(OUTPUT_FILE)
csv_file = open(OUTPUT_FILE, "a", newline="")
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
if write_header:
    writer.writeheader()


dataset = load_dataset("tatsu-lab/alpaca")["train"].shuffle(seed=67).select(range(MAX_SAMPLES))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess(examples):
    inputs = [f"Instruction: {i}" for i in examples["instruction"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = [[(t if t != tokenizer.pad_token_id else -100) for t in seq] for seq in labels["input_ids"]]
    return model_inputs


tokenized = dataset.map(preprocess, batched=True)
tokenized = tokenized.train_test_split(test_size=0.2)
tokenized.set_format("torch")


rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
try:
    bert = evaluate.load("bertscore")
    use_bert = True
except Exception:
    print("BERTScore not installed. Skipping.")
    use_bert = False


class LossGradTracker(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.grad_norms = []
        self.lrs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "grad_norm" in logs:       # <- grab gradient norm directly from HF
            self.grad_norms.append(logs["grad_norm"])

    def on_optimizer_step(self, args, state, control, optimizer, **kwargs):
        self.lrs.append(optimizer.param_groups[0]["lr"])
        return control


def compute_exact_match(preds, refs):
    if len(preds) == 0:
        return 0.0
    return sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip()) / len(preds)


for r in LORA_R:
    for alpha in LORA_ALPHA:
        for dropout in LORA_DROPOUT:
            for modules in TARGET_MODULES:
                for layers in LAYERS_TUNED:
                    for lr in LEARNING_RATES:
                        for bs in BATCH_SIZES:
                            for epoch in EPOCHS:

                                print(f"\n=== LoRA run | r={r}, alpha={alpha}, dropout={dropout}, modules={modules}, layers={layers}, lr={lr}, bs={bs}, epochs={epoch} ===")

                                model = AutoModelForSeq2SeqLM.from_pretrained(
                                    MODEL_NAME,
                                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                                )
                                model.config.use_cache = False

                                config = LoraConfig(
                                    r=r,
                                    lora_alpha=alpha,
                                    lora_dropout=dropout,
                                    bias="none",
                                    target_modules=modules,
                                    task_type="SEQ_2_SEQ_LM",
                                )
                                model = get_peft_model(model, config)
                                model.config.use_cache = False

                                args = TrainingArguments(
                                    output_dir="output/tmp",
                                    eval_strategy="steps",
                                    learning_rate=lr,
                                    lr_scheduler_type="constant",
                                    warmup_steps=0,
                                    per_device_train_batch_size=bs,
                                    per_device_eval_batch_size=bs,
                                    num_train_epochs=epoch,
                                    save_strategy="no",
                                    report_to="none",
                                    logging_steps=1,
                                )

                                tracker = LossGradTracker()

                                trainer = Trainer(
                                    model=model,
                                    args=args,
                                    train_dataset=tokenized["train"],
                                    eval_dataset=tokenized["test"],
                                    tokenizer=tokenizer,
                                    data_collator=collator,
                                    callbacks=[tracker]
                                )

                                print("\n=== DEBUG: LR scheduler ===")
                                print(trainer.lr_scheduler)
                                print("======================================\n")

                                try:
                                    trainer.train()
                                    eval_metrics = trainer.evaluate()
                                except Exception as e:
                                    print(f"⚠ Training failed: {e}")
                                    continue

                                device = "cuda" if torch.cuda.is_available() else "cpu"
                                model.to(device)
                                model.eval()

                                loader = DataLoader(tokenized["test"], batch_size=bs)
                                preds, refs = [], []

                                for batch in loader:
                                    input_ids = batch["input_ids"].to(device)
                                    attn = batch["attention_mask"].to(device)
                                    with torch.no_grad():
                                        out = model.generate(
                                            input_ids=input_ids,
                                            attention_mask=attn,
                                            max_new_tokens=128
                                        )
                                    preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

                                    labels = batch["labels"].clone()
                                    labels[labels == -100] = tokenizer.pad_token_id
                                    refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

                                rouge_res = rouge.compute(predictions=preds, references=refs)
                                bleu_res = bleu.compute(predictions=preds, references=refs)

                                if use_bert:
                                    try:
                                        bert_res = bert.compute(predictions=preds, references=refs, lang="en")
                                        bert_score = float(sum(bert_res["f1"]) / len(bert_res["f1"]))
                                    except Exception:
                                        bert_score = None
                                else:
                                    bert_score = None


                                train_loss_first = tracker.train_losses[0] if tracker.train_losses else None
                                train_loss_last = tracker.train_losses[-1] if tracker.train_losses else None

                                if tracker.train_losses and len(tracker.train_losses) > 2:
                                    loss_slope = (train_loss_first - train_loss_last) / len(tracker.train_losses)
                                else:
                                    loss_slope = None

                                gradient_norm_mean = (
                                    sum(tracker.grad_norms) / len(tracker.grad_norms)
                                    if tracker.grad_norms else None
                                )

                                learning_rate_final = tracker.lrs[-1] if tracker.lrs else lr

                                exact_match = compute_exact_match(preds, refs)

                                quality_score = (
                                    rouge_res["rougeL"] * 0.4 +
                                    rouge_res["rouge1"] * 0.2 +
                                    bleu_res["bleu"] * 0.2 +
                                    (bert_score if bert_score else 0) * 0.2
                                )

                                overfit_flag = int(
                                    train_loss_last is not None and
                                    eval_metrics.get("eval_loss", 0) > (train_loss_last * 1.3)
                                )

                                training_efficiency = (
                                    quality_score / (epoch * bs * lr)
                                    if lr > 0 else None
                                )

                                row = {
                                    "model_name": MODEL_NAME,
                                    "peft": "lora",
                                    "lora_r": r,
                                    "lora_alpha": alpha,
                                    "lora_dropout": dropout,
                                    "target_modules": str(modules),
                                    "layers_tuned": layers,
                                    "learning_rate": lr,
                                    "batch_size": bs,
                                    "epoch": epoch,
                                    "train_loss_first": train_loss_first,
                                    "train_loss_last": train_loss_last,
                                    "loss_slope": loss_slope,
                                    "gradient_norm_mean": gradient_norm_mean,
                                    "learning_rate_final": learning_rate_final,
                                    "eval_loss": eval_metrics.get("eval_loss"),
                                    "rouge1": rouge_res["rouge1"],
                                    "rouge2": rouge_res["rouge2"],
                                    "rougeL": rouge_res["rougeL"],
                                    "bleu": bleu_res["bleu"],
                                    "bert_score": bert_score,
                                    "exact_match": exact_match,
                                    "quality_score": quality_score,
                                    "overfit_flag": overfit_flag,
                                    "training_efficiency": training_efficiency,
                                    "seed": 67,
                                }

                                writer.writerow(row)
                                csv_file.flush()
                                print("✓ Row written:", row)

csv_file.close()
print(f"\nFinished. CSV saved to: {OUTPUT_FILE}")
