import torch
import pandas as pd
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from datasets import Dataset, ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================
# 設定
# =========================

MODEL_NAME = "cl-tohoku/bert-base-japanese"
MODEL_SAVE_PATH = "./model"
DATA_PATH = "./data/sentiment_data.csv"

labels = ["ネガティブ", "ニュートラル", "ポジティブ"]

# =========================
# データ読み込み
# =========================

df = pd.read_csv(DATA_PATH)

dataset = Dataset.from_pandas(df).cast_column(
    "label",
    ClassLabel(num_classes=3, names=labels)
)

# =========================
# トークナイザー
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize_function)

# 学習用 / テスト用に分割
dataset = dataset.train_test_split(test_size=0.2)

# =========================
# 評価関数
# =========================

def compute_metrics(eval_pred):
    logits, labels_ids = eval_pred

    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_ids,
        preds,
        average="weighted"
    )

    acc = accuracy_score(labels_ids, preds)

    return {
        "accuracy": acc,
        "f1": f1
    }

# =========================
# モデル読み込み
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# =========================
# 学習設定
# =========================

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_steps=50
)

# =========================
# Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# =========================
# 学習
# =========================

trainer.train()

# =========================
# モデル保存
# =========================

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("モデル保存完了")

# =========================
# 推論用関数
# =========================

# 保存したモデルを再読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    pred = torch.argmax(logits, dim=1).item()

    return labels[pred]

# =========================
# テスト
# =========================

if __name__ == "__main__":

    test_texts = [
        "このホテルは最高！",
        "普通だった",
        "最悪でもう行きたくない"
    ]

    for text in test_texts:
        result = predict(text)
        print(f"入力: {text}")
        print(f"予測: {result}")
        print("-" * 30)