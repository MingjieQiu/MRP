import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Load IMDb Dataset (subset for local run)
# ----------------------------
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"][:2000]  # 2000 samples
train_labels = dataset["train"]["label"][:2000]
test_texts = dataset["test"]["text"][:500]     # 500 samples
test_labels = dataset["test"]["label"][:500]

# ----------------------------
# Simple integer tokenizer for RNNs
# ----------------------------
def build_vocab(texts, max_vocab=5000):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {"<PAD>":0, "<UNK>":1}
    for i, word in enumerate(counter.most_common(max_vocab-2), start=2):
        vocab[word[0]] = i
    return vocab

vocab = build_vocab(train_texts)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

def encode(text, vocab, max_len=128):
    tokens = [vocab.get(w, 1) for w in text.lower().split()]
    if len(tokens) < max_len:
        tokens += [0]*(max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

class RNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.encodings = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

train_dataset_rnn = RNNDataset(train_texts, train_labels, vocab)
test_dataset_rnn = RNNDataset(test_texts, test_labels, vocab)
train_loader = DataLoader(train_dataset_rnn, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset_rnn, batch_size=32)

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
        self.fc = nn.Linear(hidden_dim*2, 2)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

# ----------------------------
# RNN Training/Evaluation
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
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

def eval_rnn(model):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device)
            y = batch["label"]
            out = model(x)
            p = torch.argmax(out, dim=1).cpu()
            preds.extend(p.numpy())
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)

# ----------------------------
# Transformer (DistilBERT)
# ----------------------------
transformer_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def run_transformer(model_name):
    # Tokenize
    def tokenize(batch):
        return transformer_tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    train_enc = transformer_tokenizer(train_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    test_enc = transformer_tokenizer(test_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    train_dataset = torch.utils.data.TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(test_enc["input_ids"], test_enc["attention_mask"], torch.tensor(test_labels))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics["accuracy"]

# ----------------------------
# Run All Models
# ----------------------------
results = {}
# RNNs
rnn_models = {"GRU": GRUClassifier(vocab_size), "BiLSTM": BiLSTMClassifier(vocab_size)}
for name, model in rnn_models.items():
    print(f"\nTraining {name}")
    train_rnn(model, epochs=3)
    acc = eval_rnn(model)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Transformer
results["DistilBERT"] = run_transformer("distilbert-base-uncased")
print(f"DistilBERT Accuracy: {results['DistilBERT']:.4f}")

# ----------------------------
# Save CSV and Plot
# ----------------------------
df = pd.DataFrame(list(results.items()), columns=["Model","Accuracy"])
df.to_csv("results.csv", index=False)
print("Saved results.csv")

plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color=['skyblue','lightgreen','orange'])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Sentiment Model Comparison on IMDb")
for i,v in enumerate(results.values()):
    plt.text(i, v+0.01, f"{v:.2f}", ha='center')
plt.savefig("accuracy_comparison.png", bbox_inches="tight")
plt.show()
print("Saved accuracy_comparison.png")