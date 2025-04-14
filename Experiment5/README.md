# Sequence-to-Sequence English to Spanish Translation

This project implements a deep learning-based sequence-to-sequence (seq2seq) model for English to Spanish machine translation using LSTM networks, with and without attention mechanisms.

## 🚀 Objective

To build and compare the performance of two architectures for neural machine translation:

1. LSTM Encoder-Decoder (Vanilla seq2seq)
2. LSTM Encoder-Decoder with Attention:
   - Bahdanau (Additive) Attention
   - Luong (Multiplicative) Attention

## 📁 Dataset

The dataset contains English-Spanish sentence pairs in plain text format (one pair per line, separated by a tab).

📌 **Dataset Link**: [Kaggle - Spanish Translation Dataset](https://www.kaggle.com/datasets/ayushchaudhary2411/spanish)

**Example:**

If the dataset is too large to process efficiently, sample a subset (e.g., 10,000 pairs) and split into:

- 80% Training
- 10% Validation
- 10% Testing

## 🛠️ Preprocessing

- Tokenization
- Lowercasing
- Padding
- Train/Validation/Test split

## 🧠 Models

### 1. LSTM Encoder-Decoder (No Attention)

- Basic seq2seq model with LSTM encoder and decoder
- Word embeddings (learned or pre-trained like GloVe)
- Teacher forcing used during training
- Evaluation using BLEU score

### 2. LSTM Encoder-Decoder with Attention

- **Bahdanau Attention** (Additive)
- **Luong Attention** (Multiplicative)
- Attention weight visualization for selected translations
- BLEU score comparison with non-attention model

## 📊 Evaluation

- Primary Metric: **BLEU Score**
- Qualitative analysis through attention visualization

## 🌐 (Optional) Deployment

You may deploy the final model via a simple web application using Flask or Streamlit to accept English input and return Spanish translation.

---



