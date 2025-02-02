# Neural Network Classification with NumPy

This project demonstrates how to train a neural network from scratch using **NumPy** to classify two different types of datasets: a **linearly separable dataset** (circles) and a **non-linearly separable dataset** (moons). The implementation avoids using high-level deep learning frameworks like TensorFlow or PyTorch, focusing instead on fundamental concepts such as **forward/backward propagation**, **activation functions**, and **gradient descent**.

---

## Key Steps

### 1. **Linearly Separable Dataset (Circles)**
- **Dataset**: A synthetic dataset generated using `sklearn.datasets.make_circles`.
- **Model Architecture**:
  - **Input Layer**: 2 neurons (one for each feature).
  - **Output Layer**: 1 neuron with Sigmoid activation (for binary classification).
- **Training**:
  - The model is trained without a hidden layer since the dataset is linearly separable.
  - Forward propagation computes predictions using the Sigmoid activation function.
  - Backward propagation updates weights and biases using gradient descent.
  - Binary cross-entropy loss is used to measure the difference between predictions and true labels.

### 2. **Non-Linearly Separable Dataset (Moons)**
- **Dataset**: A synthetic dataset generated using `sklearn.datasets.make_moons`.
- **Model Architecture**:
  - **Input Layer**: 2 neurons (one for each feature).
  - **Hidden Layer**: 4 neurons with ReLU activation to introduce non-linearity.
  - **Output Layer**: 1 neuron with Sigmoid activation (for binary classification).
- **Training**:
  - The model is trained with a hidden layer to handle the non-linear decision boundary.
  - Forward propagation computes predictions using ReLU for the hidden layer and Sigmoid for the output layer.
  - Backward propagation updates weights and biases using gradient descent.
  - Binary cross-entropy loss is used to measure the difference between predictions and true labels.

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/numpy-neural-network.git
   cd numpy-neural-network
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy scikit-learn matplotlib
   ```

3. Run the scripts to train and evaluate the models:
   - For the **circles dataset** (without hidden layer):
     ```bash
     python train_circles.py
     ```
   - For the **moons dataset** (with hidden layer):
     ```bash
     python train_moons.py
     ```

---

## Results
- **Circles Dataset**:
  - The model achieves high accuracy without a hidden layer, demonstrating its ability to classify linearly separable data.
- **Moons Dataset**:
  - The model achieves high accuracy with a hidden layer, demonstrating its ability to classify non-linearly separable data using ReLU activation.

---

## Dependencies
- Python 3.x
- NumPy
- Scikit-learn (for dataset generation)
- Matplotlib (for visualization)

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- The synthetic datasets are generated using `sklearn.datasets`.
- This implementation is inspired by foundational concepts in deep learning and neural networks.

---

Feel free to contribute or report issues! 🚀
