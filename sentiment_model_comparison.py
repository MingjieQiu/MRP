import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import Counter
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load IMDb Dataset
dataset = load_dataset("imdb")
random.seed(42)
train_indices = random.sample(range(len(dataset["train"])), 5000)
test_indices = random.sample(range(len(dataset["test"])), 1000)

train_texts = [dataset["train"][i]["text"] for i in train_indices]
train_labels = [dataset["train"][i]["label"] for i in train_indices]
test_texts = [dataset["test"][i]["text"] for i in test_indices]
test_labels = [dataset["test"][i]["label"] for i in test_indices]


# Simple integer tokenizer for RNNs
def build_vocab(texts: list[str], max_vocab: int = 10000) -> dict[str, int]:
    """
    Build vocabulary from text corpus.
    Args:
        texts (list): List of text strings
        max_vocab (int): Maximum vocabulary size
    Returns:
        dict: Vocabulary mapping words to indices
    """
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(counter.most_common(max_vocab - 2), start=2):
        vocab[word[0]] = i
    return vocab


vocab = build_vocab(train_texts)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")


def encode(text: str, vocab: dict[str, int], max_len: int = 128) -> list[int]:
    """
    Convert text to integer sequence.   
    Args:
        text (str): Input text
        vocab (dict): Vocabulary mapping
        max_len (int): Maximum sequence length   
    Returns:
        list: Encoded token sequence
    """
    tokens = [vocab.get(w, 1) for w in text.lower().split()]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


class RNNDataset(Dataset):
    """
    PyTorch Dataset for RNN models with integer tokenized text.
    Args:
        texts (list): List of text strings
        labels (list): List of corresponding labels
        vocab (dict): Vocabulary mapping words to indices
        max_len (int): Maximum sequence length
    """

    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int = 128):
        self.encodings = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


train_dataset_rnn = RNNDataset(train_texts, train_labels, vocab)
test_dataset_rnn = RNNDataset(test_texts, test_labels, vocab)
train_loader = DataLoader(train_dataset_rnn, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset_rnn, batch_size=32)


# RNN Models
class GRUClassifier(nn.Module):
    """
    GRU-based text classifier for sentiment analysis.
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Embedding dimension
        hidden_dim (int): Hidden layer dimension
    """

    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 600):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        output, h = self.gru(x)
        # Use last hidden state from both directions
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM-based text classifier for sentiment analysis.   
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Embedding dimension
        hidden_dim (int): Hidden layer dimension
    """

    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 600):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        output, (h, _) = self.lstm(x)
        # Use last hidden state from both directions
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


# RNN Training/Evaluation
def train_rnn(model: nn.Module, epochs: int = 3) -> None:
    """
    Train RNN model for sentiment classification.
    Args:
        model (nn.Module): RNN model to train
        epochs (int): Number of training epochs 
    """
    model.to(device)
    # Use AdamW for better convergence and cosine annealing for learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Training {model.__class__.__name__}"):
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Track training accuracy
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")


def eval_rnn(model: nn.Module) -> float:
    """
    Evaluate RNN model on test dataset.
    Args:
        model (nn.Module): Trained RNN model
    Returns:
        float: Accuracy score
    """
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


# Transformer (DistilBERT)
transformer_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def run_transformer(model_name: str) -> float:
    """
    Train and evaluate transformer model (DistilBERT) for sentiment classification.
    Args:
        model_name (str): Pretrained model name 
    Returns:
        float: Accuracy score
    """
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


# Run All Models
results = {}
model_details = {}

# Train RNN models
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

# Train DistilBERT model
print(f"\nTraining DistilBERT")
start_time = time.time()

# Load DistilBERT model to count parameters
distilbert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
distilbert_params = sum(p.numel() for p in distilbert_model.parameters())
distilbert_trainable_params = sum(p.numel() for p in distilbert_model.parameters() if p.requires_grad)

distilbert_acc = run_transformer("distilbert-base-uncased")
distilbert_time = time.time() - start_time

results["DistilBERT"] = distilbert_acc
model_details["DistilBERT"] = {
    'accuracy': distilbert_acc,
    'training_time': distilbert_time,
    'total_params': distilbert_params,
    'trainable_params': distilbert_trainable_params
}

print(
    f"DistilBERT Accuracy: {distilbert_acc:.4f}, Training Time: {distilbert_time:.2f}s, Parameters: {distilbert_params:,}")

# Comprehensive Results and Visualizations
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
df_detailed.to_csv("results/detailed_results.csv", index=False)
print("Saved results/detailed_results.csv")
print(df_detailed)

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Accuracy Comparison
axes[0, 0].bar(results.keys(), results.values(), color=['skyblue', 'lightgreen', 'orange'])
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_title("Model Accuracy Comparison")
for i, v in enumerate(results.values()):
    axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha='center')

# 2. Training Time Comparison
times = [model_details[name]['training_time'] for name in results.keys()]
axes[0, 1].bar(results.keys(), times, color=['skyblue', 'lightgreen', 'orange'])
axes[0, 1].set_ylabel("Training Time (seconds)")
axes[0, 1].set_title("Training Time Comparison")
for i, v in enumerate(times):
    axes[0, 1].text(i, v + 1, f"{v:.1f}s", ha='center')

# 3. Model Complexity (Parameters)
params = [model_details[name]['total_params'] for name in results.keys()]
param_labels = [f"{p / 1_000_000:.1f}M" for p in params]
axes[1, 0].bar(results.keys(), params, color=['skyblue', 'lightgreen', 'orange'])
axes[1, 0].set_ylabel("Total Parameters")
axes[1, 0].set_title("Model Complexity")
for i, v in enumerate(params):
    axes[1, 0].text(i, v + 50000, f"{v / 1_000_000:.1f}M", ha='center')

# 4. Efficiency: Accuracy vs Training Time
axes[1, 1].scatter(times, [model_details[name]['accuracy'] for name in results.keys()],
                   s=[p / 1000 for p in params], alpha=0.7, c=['skyblue', 'lightgreen', 'orange'])
axes[1, 1].set_xlabel("Training Time (seconds)")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_title("Efficiency: Accuracy vs Training Time")
for i, name in enumerate(results.keys()):
    axes[1, 1].annotate(name, (times[i], model_details[name]['accuracy']),
                        xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig("results/comprehensive_comparison.png", bbox_inches="tight", dpi=300)
plt.show()
print("Saved results/comprehensive_comparison.png")

# Print summary
print("\n" + "=" * 50)
print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
print("=" * 50)
for name, details in model_details.items():
    print(f"\n{name} Model:")
    print(f"  Accuracy: {details['accuracy']:.4f}")
    print(f"  Training Time: {details['training_time']:.2f} seconds")
    print(f"  Total Parameters: {details['total_params']:,}")
    print(f"  Trainable Parameters: {details['trainable_params']:,}")
    print(f"  Efficiency: {details['accuracy'] / details['training_time']:.6f} accuracy/sec")
