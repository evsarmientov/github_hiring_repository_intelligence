import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.utils.class_weight import compute_class_weight
from src.utils import logger, ensure_dir, LABEL_MAP, LABEL_NAMES, save_json

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
OUTPUT_DIR = "models/trained_models"


class RepoDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_splits(splits_dir: str = "data/splits") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(f"{splits_dir}/train.csv")
    val = pd.read_csv(f"{splits_dir}/val.csv")
    test = pd.read_csv(f"{splits_dir}/test.csv")
    return train, val, test


def encode_labels(df: pd.DataFrame) -> list[int]:
    return [LABEL_MAP.get(str(l).strip(), 5) for l in df["label"]]


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


def compute_metrics_fn(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    output_dir: str = OUTPUT_DIR,
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    num_labels = len(LABEL_MAP)
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    train_texts = train_df["bert_input"].astype(str).tolist()
    val_texts = val_df["bert_input"].astype(str).tolist()
    train_labels = encode_labels(train_df)
    val_labels = encode_labels(val_df)

    logger.info(f"Train: {len(train_texts)} | Val: {len(val_texts)}")
    logger.info(f"Label distribution (train): {pd.Series(train_labels).value_counts().to_dict()}")

    train_dataset = RepoDataset(train_texts, train_labels, tokenizer)
    val_dataset = RepoDataset(val_texts, val_labels, tokenizer)

    class_weights = compute_class_weights(train_labels, num_labels)
    logger.info(f"Class weights: {class_weights.tolist()}")

    ensure_dir(output_dir)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting fine-tuning...")
    train_result = trainer.train()

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    history = trainer.state.log_history
    save_json(history, f"{output_dir}/training_history.json")

    label2id = {k: str(v) for k, v in LABEL_MAP.items()}
    id2label = {str(v): k for k, v in LABEL_MAP.items()}
    save_json({"label2id": label2id, "id2label": id2label, "model_name": model_name}, f"{output_dir}/label_config.json")

    logger.info(f"Training complete. Best val F1: {train_result.metrics}")
    return trainer, tokenizer


def predict(texts: list[str], model_dir: str = OUTPUT_DIR) -> tuple[list[str], list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_ids = np.argmax(probs, axis=-1)
    pred_labels = [LABEL_NAMES.get(int(i), "low_value") for i in pred_ids]
    confidences = [float(probs[i, pred_ids[i]]) for i in range(len(texts))]
    return pred_labels, confidences
