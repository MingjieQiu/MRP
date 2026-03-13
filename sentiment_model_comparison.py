import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# RNN Models
# ----------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h[-1])


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

# ----------------------------
# Load IMDb Dataset (small subset)
# ----------------------------
dataset = load_dataset("imdb")
dataset["train"] = dataset["train"].select(range(500))
dataset["test"] = dataset["test"].select(range(250))
print(f"Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")

# ----------------------------
# Tokenizer for RNN
# ----------------------------
tokenizer_rnn = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_rnn(batch):
    return tokenizer_rnn(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset_rnn = dataset.map(tokenize_rnn, batched=True)
dataset_rnn.set_format(type="torch", columns=["input_ids","label"])

train_loader = DataLoader(dataset_rnn["train"], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset_rnn["test"], batch_size=32)

# ----------------------------
# RNN Training / Evaluation
# ----------------------------
def train_rnn(model, epochs=3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Training {model.__class__.__name__}"):
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

def eval_rnn(model):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device)
            y = batch["label"]
            output = model(x)
            p = torch.argmax(output, dim=1).cpu()
            preds.extend(p.numpy())
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)

# ----------------------------
# Transformer Evaluation (DistilBERT only)
# ----------------------------
def run_transformer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    encoded = dataset.map(tokenize, batched=True)
    encoded.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )

    # Accuracy metric function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        eval_strategy="steps",
        save_strategy="no",
        logging_steps=10,
        max_steps=10,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics['eval_accuracy']

# ----------------------------
# Run All Models
# ----------------------------
results = {}
vocab_size = tokenizer_rnn.vocab_size

# RNN models
rnn_models = {"GRU": GRUClassifier(vocab_size), "BiLSTM": BiLSTMClassifier(vocab_size)}
for name, model in rnn_models.items():
    print(f"\nTraining {name}")
    train_rnn(model, epochs=1)  # 1 epoch for demo
    acc = eval_rnn(model)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Transformer model
transformer_models = {"DistilBERT": "distilbert-base-uncased"}
for name, model_name in transformer_models.items():
    print(f"\nTraining {name}")
    acc = run_transformer(model_name)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# ----------------------------
# Save CSV and Plot
# ----------------------------
df = pd.DataFrame(list(results.items()), columns=["Model","Accuracy"])
df.to_csv("results.csv", index=False)
print("Accuracy results saved to results.csv")

plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color=['skyblue','lightgreen','orange'])
plt.ylabel("Accuracy")
plt.title("Sentiment Model Comparison on IMDb")
plt.ylim(0,1)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.savefig("accuracy_comparison.png", bbox_inches="tight")
plt.show()
print("Bar chart saved to accuracy_comparison.png")