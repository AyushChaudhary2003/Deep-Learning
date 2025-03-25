# Text Generation using RNN and LSTM

This project explores text generation using **Recurrent Neural Networks (RNNs)** with **One-Hot Encoding** and **Trainable Word Embeddings**. The model is trained on a dataset of **100 poems** to predict the next word in a sequence.

## Implementation
- **One-Hot Encoding:** Converts words into one-hot vectors and trains an RNN.
- **Trainable Embeddings:** Uses an embedding layer to learn word representations.
- **Comparison:** Evaluates training time, loss, and text quality between both methods.

## Usage
1. Install dependencies:  
   ```bash
   pip install torch torchvision numpy pandas nltk
   ```
2. Train the model:  
   ```bash
   python train_model.py
   ```
3. Generate text:  
   ```bash
   python generate_text.py
   ```

## Author
- **[Your Name]**

