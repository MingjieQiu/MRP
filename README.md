# MRP: Mini Research Problem

MSCS2201-2: Artificial Intelligence @ Sofia University

Assignment #7.1: Attack MNIST Recognition

# Sentiment Model Comparison

This project compares 5 models for sentiment analysis on the IMDb dataset:

- GRU (Gated Recurrent Unit)  
- BiLSTM (Bidirectional LSTM)  
- DistilBERT (Transformer)  
- RoBERTa (Transformer)  
- DeBERTa (Transformer)  

It trains, evaluates, and visualizes the accuracy of each model.

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

### Detailed Results

The complete results are available in the CSV file: [results/detailed_results.csv](results/detailed_results.csv)

### Visualization

The project generates a comprehensive comparison chart showing:
- Model accuracy comparison
- Training time comparison  
- Model complexity (parameter count)
- Efficiency (accuracy vs training time)

![Model Comparison Results](results/comprehensive_comparison.png)

### Key Findings

1. **DistilBERT** achieved the highest accuracy (86.1%) but required the longest training time
2. **GRU** provided the best balance of accuracy (70.9%) and training efficiency
3. **BiLSTM** had the lowest accuracy (64.2%) among the tested models
4. Transformer models (DistilBERT) significantly outperformed RNN models in accuracy
5. RNN models were faster to train but less accurate than transformer models
