# MRP: Mini Research Problem

MSCS2201-2: Artificial Intelligence @ Sofia University

# Sentiment Model Comparison

## Project Goal

This project aims to compare the performance of different neural network architectures for sentiment analysis on movie reviews. The goal is to evaluate how traditional recurrent neural networks (RNNs) compare to modern transformer-based models in terms of accuracy, training efficiency, and model complexity.

## Dataset Description

The project uses the **IMDb Large Movie Review Dataset**, which contains:
- 50,000 movie reviews labeled as positive or negative
- 25,000 reviews for training and 25,000 for testing
- Each review is labeled with a binary sentiment score (0 = negative, 1 = positive)
- Reviews vary in length from a few sentences to several paragraphs

For this comparison, we use a subset of 5,000 training reviews and 1,000 test reviews to ensure reasonable training times while maintaining statistical significance.

## Models Compared

This project compares the following neural network architectures:

- **GRU (Gated Recurrent Unit)**: A type of recurrent neural network optimized for sequence processing
- **BiLSTM (Bidirectional LSTM)**: Long Short-Term Memory network that processes sequences in both directions
- **DistilBERT**: A smaller, faster version of BERT that maintains high performance

The project trains, evaluates, and visualizes the accuracy of each model to provide insights into the trade-offs between model complexity, training time, and predictive performance.

---

## Requirements

Python 3.12 recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python sentiment_model_comparison.py
```

## Results

### Model Performance Summary

| Model | Accuracy | Training Time (s) | Total Parameters | Trainable Parameters | Parameters (Millions) |
|-------|----------|-------------------|------------------|---------------------|----------------------|
| GRU | 0.709 | 394.12 | 13,042,370 | 13,042,370 | 13.04 |
| BiLSTM | 0.642 | 509.67 | 16,287,170 | 16,287,170 | 16.29 |
| DistilBERT | 0.861 | 793.84 | 66,955,010 | 66,955,010 | 66.96 |


### Training Progress
Accuracy of each epoch

| Epoch | GRU | BiLSTM | DistilBERT |
|-------|-----|--------|------------|
| 1 | 0.652 | 0.618 | - |
| 2 | 0.687 | 0.634 | - |
| 3 | 0.701 | 0.642 | - |
| 4 | 0.708 | 0.645 | - |
| 5 | 0.709 | 0.642 | - |

### Visualization

The project generates a comprehensive comparison chart showing:
- Model accuracy comparison
- Training time comparison  
- Model complexity (parameter count)
- Efficiency (accuracy vs training time)

![Model Comparison Results](results/comprehensive_comparison.png)

#### Training Progress Visualization
![Training Accuracy vs Epochs](results/accuracy_vs_epochs.png)

## Key Findings

---

1. **DistilBERT** achieved the highest accuracy (86.1%) but required the longest training time
2. **GRU** provided the best balance of accuracy (70.9%) and training efficiency
3. **BiLSTM** had the lowest accuracy (64.2%) among the tested models
4. Transformer models (DistilBERT) significantly outperformed RNN models in accuracy
5. RNN models were faster to train but less accurate than transformer models
