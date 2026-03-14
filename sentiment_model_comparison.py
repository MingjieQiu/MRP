import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import Counter
import time
import numpy as np

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Load IMDb Dataset
# ----------------------------
dataset = load_dataset("imdb")
random.seed(42)
train_indices = random.sample(range(len(dataset["train"])), 5000)
test_indices = random.sample(range(len(dataset["test"])), 1000)

train_texts = [dataset["train"][i]["text"] for i in train_indices]
train_labels = [dataset["train"][i]["label"] for i in train_indices]
test_texts = [dataset["test"][i]["text"] for i in test_indices]
test_labels = [dataset["test"][i]["label"] for i in test_indices]

# ----------------------------
# Simple integer tokenizer for RNNs
# ----------------------------
def build_vocab(texts, max_vocab=10000):
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
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=600):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        h = self.dropout(h[-1])
        return self.fc(h)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=600):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# ----------------------------
# RNN Training/Evaluation
# ----------------------------
def train_rnn(model, epochs=7):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training {model.__class__.__name__}"):
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def eval_rnn(model):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            out = model(x)
            p = torch.argmax(out, dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return accuracy_score(labels, preds)

# ----------------------------
# Transformer (DistilBERT)
# ----------------------------
transformer_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def run_transformer(model_name):
    train_encodings = transformer_tokenizer(train_texts, padding=True, truncation=True, max_length=128)
    test_encodings = transformer_tokenizer(test_texts, padding=True, truncation=True, max_length=128)

    train_dataset = HFDataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels
    })
    test_dataset = HFDataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels
    })

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
        eval_strategy="epoch",
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
    return metrics['eval_accuracy']

# ----------------------------
# Run All Models with Timing
# ----------------------------
results = {}
model_details = {}

rnn_models = {"GRU": GRUClassifier(vocab_size), "BiLSTM": BiLSTMClassifier(vocab_size)}
for name, model in rnn_models.items():
    print(f"\nTraining {name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Measure training time
    start_time = time.time()
    train_rnn(model, epochs=3)
    training_time = time.time() - start_time
    
    # Evaluate model
    acc = eval_rnn(model)
    
    # Store results
    results[name] = acc
    model_details[name] = {
        'accuracy': acc,
        'training_time': training_time,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    print(f"{name} Accuracy: {acc:.4f}, Training Time: {training_time:.2f}s")

# Commented out DistilBERT training to focus on RNN models
# results["DistilBERT"] = run_transformer("distilbert-base-uncased")
# print(f"DistilBERT Accuracy: {results['DistilBERT']:.4f}")

# ----------------------------
# Comprehensive Results and Visualizations
# ----------------------------

# Create detailed results DataFrame
detailed_results = []
for name, details in model_details.items():
    detailed_results.append({
        'Model': name,
        'Accuracy': details['accuracy'],
        'Training_Time_s': details['training_time'],
        'Total_Params': details['total_params'],
        'Trainable_Params': details['trainable_params'],
        'Params_Millions': details['total_params'] / 1_000_000
    })

df_detailed = pd.DataFrame(detailed_results)
df_detailed.to_csv("detailed_results.csv", index=False)
print("Saved detailed_results.csv")
print(df_detailed)

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Accuracy Comparison
axes[0, 0].bar(results.keys(), results.values(), color=['skyblue','lightgreen'])
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_title("Model Accuracy Comparison")
for i,v in enumerate(results.values()):
    axes[0, 0].text(i, v+0.01, f"{v:.3f}", ha='center')

# 2. Training Time Comparison
times = [model_details[name]['training_time'] for name in results.keys()]
axes[0, 1].bar(results.keys(), times, color=['skyblue','lightgreen'])
axes[0, 1].set_ylabel("Training Time (seconds)")
axes[0, 1].set_title("Training Time Comparison")
for i,v in enumerate(times):
    axes[0, 1].text(i, v+1, f"{v:.1f}s", ha='center')

# 3. Model Complexity (Parameters)
params = [model_details[name]['total_params'] for name in results.keys()]
param_labels = [f"{p/1_000_000:.1f}M" for p in params]
axes[1, 0].bar(results.keys(), params, color=['skyblue','lightgreen'])
axes[1, 0].set_ylabel("Total Parameters")
axes[1, 0].set_title("Model Complexity")
for i,v in enumerate(params):
    axes[1, 0].text(i, v+50000, f"{v/1_000_000:.1f}M", ha='center')

# 4. Efficiency: Accuracy vs Training Time
axes[1, 1].scatter(times, [model_details[name]['accuracy'] for name in results.keys()], 
                  s=[p/1000 for p in params], alpha=0.7, c=['skyblue','lightgreen'])
axes[1, 1].set_xlabel("Training Time (seconds)")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_title("Efficiency: Accuracy vs Training Time")
for i, name in enumerate(results.keys()):
    axes[1, 1].annotate(name, (times[i], model_details[name]['accuracy']), 
                      xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig("comprehensive_comparison.png", bbox_inches="tight", dpi=300)
plt.show()
print("Saved comprehensive_comparison.png")

# Print summary
print("\n" + "="*50)
print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
print("="*50)
for name, details in model_details.items():
    print(f"\n{name} Model:")
    print(f"  Accuracy: {details['accuracy']:.4f}")
    print(f"  Training Time: {details['training_time']:.2f} seconds")
    print(f"  Total Parameters: {details['total_params']:,}")
    print(f"  Trainable Parameters: {details['trainable_params']:,}")
    print(f"  Efficiency: {details['accuracy']/details['training_time']:.6f} accuracy/sec")
