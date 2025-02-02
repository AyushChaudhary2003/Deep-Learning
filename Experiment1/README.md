# MNIST Digit Classification with NumPy

This project demonstrates how to build and train a simple neural network from scratch using only **NumPy** to classify handwritten digits from the **MNIST dataset**. The MNIST dataset consists of 28x28 grayscale images of digits (0–9), and the goal is to correctly classify these images into their respective digit labels.

The implementation avoids using high-level deep learning frameworks like TensorFlow or PyTorch, focusing instead on fundamental concepts such as **forward/backward propagation**, **activation functions**, and **gradient descent**.

---

## Key Steps

### 1. Dataset Preparation
- The MNIST dataset is loaded from IDX files, which contain images and labels.
- Each 28x28 image is flattened into a 784-dimensional vector.
- Pixel values are normalized to the range `[0, 1]` by dividing by 255.
- Labels are one-hot encoded to represent each digit as a 10-dimensional vector.

### 2. Neural Network Architecture
A simple 2-layer neural network is implemented:
- **Input Layer**: 784 neurons (one for each pixel in the flattened image).
- **Hidden Layer**: 128 neurons with ReLU activation.
- **Output Layer**: 10 neurons with Softmax activation (one for each digit class).

### 3. Forward Propagation
- Input data is passed through the network to compute predictions.
- The hidden layer applies the **ReLU activation function** to introduce non-linearity.
- The output layer uses the **Softmax activation function** to produce probabilities for each digit class.

### 4. Loss Calculation
- The **cross-entropy loss** is computed to measure the difference between the predicted probabilities and the true labels.
- This loss is minimized during training to improve the model's accuracy.

### 5. Backward Propagation
- Gradients of the loss concerning the weights and biases are computed using the chain rule.
- These gradients are used to update the weights and biases via **gradient descent**.

### 6. Training
- The model is trained using **mini-batch gradient descent**:
  - The dataset is divided into small batches.
  - For each batch, forward and backward propagation are performed.
  - Weights and biases are updated iteratively over multiple epochs.
- Training progress is monitored by tracking the loss after each epoch.

### 7. Evaluation
- After training, the model's performance is evaluated on the test dataset.
- Predictions are made by selecting the class with the highest probability from the Softmax output.
- Accuracy is calculated as the percentage of correctly classified images.

---
## Repository Link
🔗 [https://github.com/AyushChaudhary2003/Deep-Learning/tree/main/Experiment1](https://github.com/AyushChaudhary2003/Deep-Learning/tree/main/Experiment1)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AyushChaudhary2003/Deep-Learning.git
   cd Deep-Learning/Experiment1
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy
   ```

3. Run the script to train and evaluate the model:
   ```bash
   python train.py
   ```

---

## Results
- The model achieves an accuracy of **X%** on the test dataset.
- Training loss and accuracy are logged for each epoch to monitor progress.

---

## Dependencies
- Python 3.x
- NumPy

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- The MNIST dataset is sourced from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).
- This implementation is inspired by foundational concepts in deep learning and neural networks.

---

Feel free to contribute or report issues! 🚀
