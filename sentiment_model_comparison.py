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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imdb_dataset():
    """Load and prepare IMDb dataset with fixed random seed."""
    dataset = load_dataset("imdb")
    random.seed(42)  # Set random seed for reproducibility
    train_indices = random.sample(range(len(dataset["train"])), 5000)
    test_indices = random.sample(range(len(dataset["test"])), 1000)

    train_texts = [dataset["train"][i]["text"] for i in train_indices]
    train_labels = [dataset["train"][i]["label"] for i in train_indices]
    test_texts = [dataset["test"][i]["text"] for i in test_indices]
    test_labels = [dataset["test"][i]["label"] for i in test_indices]

    return train_texts, train_labels, test_texts, test_labels


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
        counter.update(t.lower().split())  # Convert to lowercase and split into words
    vocab = {"<PAD>": 0, "<UNK>": 1}  # Special tokens for padding and unknown words
    for i, word in enumerate(counter.most_common(max_vocab - 2), start=2):  # Reserve indices 0 and 1
        vocab[word[0]] = i
    return vocab


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
    tokens = [vocab.get(w, 1) for w in text.lower().split()]  # Convert words to indices, use <UNK> for unknown
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))  # Pad with <PAD> tokens
    else:
        tokens = tokens[:max_len]  # Truncate if too long
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
        return {"input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),  # Convert to tensor
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


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
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Word embedding layer
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3,
                          bidirectional=True)  # Bidirectional GRU
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc1 = nn.Linear(hidden_dim * 2, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 2)  # Output layer for binary classification
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # Convert token indices to embeddings
        output, h = self.gru(x)  # Process through GRU
        # Use last hidden state from both directions (forward and backward)
        h = torch.cat((h[-2], h[-1]), dim=1)  # Concatenate final forward and backward hidden states
        h = self.dropout(h)  # Apply dropout
        h = self.relu(self.fc1(h))  # Apply ReLU activation
        h = self.dropout(h)  # Apply dropout again
        return self.fc2(h)  # Return logits for classification


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
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Word embedding layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3,
                            bidirectional=True)  # Bidirectional LSTM
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc1 = nn.Linear(hidden_dim * 2, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 2)  # Output layer for binary classification
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # Convert token indices to embeddings
        output, (h, _) = self.lstm(x)  # Process through LSTM, ignore cell state
        # Use last hidden state from both directions (forward and backward)
        h = torch.cat((h[-2], h[-1]), dim=1)  # Concatenate final forward and backward hidden states
        h = self.dropout(h)  # Apply dropout
        h = self.relu(self.fc1(h))  # Apply ReLU activation
        h = self.dropout(h)  # Apply dropout again
        return self.fc2(h)  # Return logits for classification


def train_rnn(model: nn.Module, train_loader: DataLoader, epochs: int = 8) -> list[float]:
    """
    Train RNN model for sentiment classification.
    Args:
        model (nn.Module): RNN model to train
        train_loader (DataLoader): Training data loader
        epochs (int): Number of training epochs 
    Returns:
        list: Training accuracy for each epoch
    """
    model.to(device)
    # Use AdamW for better convergence and cosine annealing for learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW optimizer with weight decay
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Cosine learning rate scheduler

    model.train()
    epoch_accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Training {model.__class__.__name__}"):
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()  # Clear gradients
            out = model(x)  # Forward pass
            loss = criterion(out, y)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            total_loss += loss.item()  # Accumulate loss

            # Track training accuracy
            preds = torch.argmax(out, dim=1)  # Get predicted class
            correct += (preds == y).sum().item()  # Count correct predictions
            total += y.size(0)  # Count total samples

        avg_loss = total_loss / len(train_loader)  # Average loss per batch
        train_acc = correct / total  # Training accuracy
        epoch_accuracies.append(train_acc)  # Store epoch accuracy
        scheduler.step()  # Update learning rate

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

    return epoch_accuracies


def eval_rnn(model: nn.Module, test_loader: DataLoader) -> float:
    """
    Evaluate RNN model on test dataset.
    Args:
        model (nn.Module): Trained RNN model
        test_loader (DataLoader): Test data loader
    Returns:
        float: Accuracy score
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch in test_loader:
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            out = model(x)  # Forward pass
            p = torch.argmax(out, dim=1)  # Get predicted class
            preds.extend(p.cpu().numpy())  # Store predictions
            labels.extend(y.cpu().numpy())  # Store true labels
    return accuracy_score(labels, preds)


def run_transformer(model_name: str, train_texts: list[str], train_labels: list[int],
                    test_texts: list[str], test_labels: list[int]) -> tuple[float, list[float]]:
    """
    Train and evaluate transformer model (DistilBERT) for sentiment classification.
    Args:
        model_name (str): Pretrained model name 
        train_texts (list): Training text data
        train_labels (list): Training labels
        test_texts (list): Test text data
        test_labels (list): Test labels
    Returns:
        tuple: Final accuracy score and list of training accuracies per epoch
    """
    transformer_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings = transformer_tokenizer(train_texts, padding=True, truncation=True, max_length=128)
    test_encodings = transformer_tokenizer(test_texts, padding=True, truncation=True, max_length=128)

    train_dataset = HFDataset.from_dict({
        "input_ids": train_encodings["input_ids"],  # Tokenized input IDs
        "attention_mask": train_encodings["attention_mask"],  # Attention masks
        "labels": train_labels  # Training labels
    })
    test_dataset = HFDataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels
    })

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Load pretrained model

    training_accuracies = []

    def compute_metrics(eval_pred):
        logits, labels = eval_pred  # Extract logits and labels
        preds = logits.argmax(axis=1)  # Get predicted class
        accuracy = accuracy_score(labels, preds)

        # Store training accuracy for this epoch
        if hasattr(trainer, 'state') and trainer.state.epoch is not None:
            if trainer.state.epoch <= trainer.args.num_train_epochs:
                training_accuracies.append(accuracy)

        return {"accuracy": accuracy}

    args = TrainingArguments(
        output_dir="./results",  # Output directory
        per_device_train_batch_size=8,  # Training batch size
        per_device_eval_batch_size=8,  # Evaluation batch size
        num_train_epochs=8,  # Number of training epochs
        eval_strategy="epoch",  # Evaluate at each epoch
        save_strategy="no",  # Don't save checkpoints
        logging_steps=50,  # Log every 50 steps
        disable_tqdm=False  # Show progress bars
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

    expected_epochs = args.num_train_epochs
    if len(training_accuracies) > expected_epochs:
        training_accuracies = training_accuracies[:expected_epochs]

    return metrics['eval_accuracy'], training_accuracies


def save_epoch_accuracies(epoch_accuracies: dict, max_epochs: int) -> None:
    """Save epoch accuracy data to CSV file."""
    print("\nSaving epoch accuracy data...")

    # Create DataFrame with epoch numbers as index
    df_epochs = pd.DataFrame({'Epoch': range(1, max_epochs + 1)})

    # Add accuracy data for each model
    for model_name in ['GRU', 'BiLSTM', 'DistilBERT']:
        if model_name in epoch_accuracies:
            # Pad with None if model has fewer epochs than max_epochs
            accuracies = epoch_accuracies[model_name] + [None] * (max_epochs - len(epoch_accuracies[model_name]))
            df_epochs[model_name] = accuracies

    df_epochs.to_csv("results/epoch_accuracies.csv", index=False)
    print("Saved results/epoch_accuracies.csv")


def create_comprehensive_visualizations(model_details: dict) -> None:
    """Create comprehensive model comparison visualizations."""
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
    results = {name: details['accuracy'] for name, details in model_details.items()}
    axes[0, 0].bar(results.keys(), results.values(), color=['skyblue', 'lightgreen', 'orange'])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Model Accuracy Comparison")
    for i, v in enumerate(results.values()):
        axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha='center')

    # 2. Training Time Comparison
    times = [details['training_time'] for details in model_details.values()]
    axes[0, 1].bar(results.keys(), times, color=['skyblue', 'lightgreen', 'orange'])
    axes[0, 1].set_ylabel("Training Time (seconds)")
    axes[0, 1].set_title("Training Time Comparison")
    for i, v in enumerate(times):
        axes[0, 1].text(i, v + 1, f"{v:.1f}s", ha='center')

    # 3. Model Complexity (Parameters)
    params = [details['total_params'] for details in model_details.values()]
    param_labels = [f"{p / 1_000_000:.1f}M" for p in params]
    axes[1, 0].bar(results.keys(), params, color=['skyblue', 'lightgreen', 'orange'])
    axes[1, 0].set_ylabel("Total Parameters")
    axes[1, 0].set_title("Model Complexity")
    for i, v in enumerate(params):
        axes[1, 0].text(i, v + 50000, f"{v / 1_000_000:.1f}M", ha='center')

    # 4. Efficiency: Accuracy vs Training Time
    axes[1, 1].scatter(times, list(results.values()),
                       s=[p / 1000 for p in params], alpha=0.7, c=['skyblue', 'lightgreen', 'orange'])
    axes[1, 1].set_xlabel("Training Time (seconds)")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Efficiency: Accuracy vs Training Time")
    for i, name in enumerate(results.keys()):
        axes[1, 1].annotate(name, (times[i], results[name]), xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig("results/comprehensive_comparison.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("Saved results/comprehensive_comparison.png")


def create_epoch_accuracy_plot(epoch_accuracies: dict, max_epochs: int) -> None:
    """Create combined accuracy vs epochs plot."""
    print("Creating combined accuracy vs epochs plot...")
    plt.figure(figsize=(12, 8))

    for model_name in ['GRU', 'BiLSTM', 'DistilBERT']:
        if model_name in epoch_accuracies:
            epochs = range(1, len(epoch_accuracies[model_name]) + 1)
            plt.plot(epochs, epoch_accuracies[model_name], marker='o', linewidth=2, label=model_name)

    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy vs Epochs for All Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max_epochs)
    plt.ylim(0, 1)
    plt.yticks([i / 10 for i in range(11)])
    plt.xticks(range(1, max_epochs + 1))
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_epochs.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("Saved results/accuracy_vs_epochs.png")


def print_model_summary(model_details: dict) -> None:
    """Print comprehensive model comparison summary."""
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


def main():
    """Main execution function."""
    print("Using device:", device)

    # Step 1: Load and prepare data
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()
    vocab = build_vocab(train_texts)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Step 2: Create datasets and data loaders
    train_dataset_rnn = RNNDataset(train_texts, train_labels, vocab)
    test_dataset_rnn = RNNDataset(test_texts, test_labels, vocab)
    train_loader = DataLoader(train_dataset_rnn, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset_rnn, batch_size=32)

    # Step 3: Train all models
    results = {}
    model_details = {}
    epoch_accuracies = {}

    # Train RNN models
    rnn_models = {"GRU": GRUClassifier(vocab_size), "BiLSTM": BiLSTMClassifier(vocab_size)}
    for name, model in rnn_models.items():
        print(f"\nTraining {name}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Measure training time
        start_time = time.time()
        epoch_acc = train_rnn(model, train_loader, epochs=8)
        training_time = time.time() - start_time

        # Evaluate model
        acc = eval_rnn(model, test_loader)

        # Store results
        results[name] = acc
        model_details[name] = {
            'accuracy': acc,
            'training_time': training_time,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
        epoch_accuracies[name] = epoch_acc

        print(f"{name} Accuracy: {acc:.4f}, Training Time: {training_time:.2f}s")

    # Train DistilBERT model
    print(f"\nTraining DistilBERT")
    start_time = time.time()

    # Load DistilBERT model to count parameters
    distilbert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    distilbert_params = sum(p.numel() for p in distilbert_model.parameters())
    distilbert_trainable_params = sum(p.numel() for p in distilbert_model.parameters() if p.requires_grad)

    distilbert_acc, distilbert_epoch_acc = run_transformer("distilbert-base-uncased", train_texts, train_labels,
                                                           test_texts, test_labels)
    distilbert_time = time.time() - start_time

    results["DistilBERT"] = distilbert_acc
    model_details["DistilBERT"] = {
        'accuracy': distilbert_acc,
        'training_time': distilbert_time,
        'total_params': distilbert_params,
        'trainable_params': distilbert_trainable_params
    }
    epoch_accuracies["DistilBERT"] = distilbert_epoch_acc

    print(
        f"DistilBERT Accuracy: {distilbert_acc:.4f}, Training Time: {distilbert_time:.2f}s, Parameters: {distilbert_params:,}")

    # Save epoch accuracy data
    max_epochs = max(len(acc_list) for acc_list in epoch_accuracies.values())
    save_epoch_accuracies(epoch_accuracies, max_epochs)

    # Step 4: Create visualizations
    create_comprehensive_visualizations(model_details)
    create_epoch_accuracy_plot(epoch_accuracies, max_epochs)

    # Print summary
    print_model_summary(model_details)


if __name__ == "__main__":
    main()
