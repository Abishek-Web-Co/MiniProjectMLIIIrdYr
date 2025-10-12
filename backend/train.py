import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# --- 1. Configuration ---
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./emotion-detection-model"
TRAIN_DATA_PATH = "train.txt"
VALIDATION_DATA_PATH = "val.txt"
NUM_EPOCHS = 3

# --- 2. Load and Prepare the Dataset ---
print("Loading and preparing datasets...")

id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label2id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
NUM_LABELS = len(id2label)

df_train = pd.read_csv(TRAIN_DATA_PATH, sep=';', header=None, names=['text', 'label'])
df_val = pd.read_csv(VALIDATION_DATA_PATH, sep=';', header=None, names=['text', 'label'])

df_train['label'] = df_train['label'].map(label2id)
df_val['label'] = df_val['label'].map(label2id)

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)

# --- 3. Tokenization ---
print(f"Loading tokenizer for model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("Tokenizing datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# --- 4. Load the Pre-trained Model ---
print(f"Loading pre-trained model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

# --- 5. Define Evaluation Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

# --- 6. Configure Training ---
print("Configuring training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# --- 7. Create and Run the Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("Starting model training...")
trainer.train()
print("Training complete.")

# --- 8. Save the Final Model ---
print(f"Saving the best model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
print("Model saved successfully!")