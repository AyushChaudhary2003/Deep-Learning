# Neural Network Classification with NumPy

This repository demonstrates how to build and train neural networks from scratch using **NumPy** to classify two types of datasets:
1. **Linearly Separable Dataset (Circles)**: A simple neural network without a hidden layer.
2. **Non-Linearly Separable Dataset (Moons)**: A neural network with a hidden layer and ReLU activation.

The implementation avoids high-level deep learning frameworks like TensorFlow or PyTorch, focusing instead on core concepts such as **forward/backward propagation**, **activation functions**, and **gradient descent**.

---

## Features
- **Circle Dataset**:
  - Linearly separable dataset generated using `sklearn.datasets.make_circles`.
  - Neural network with **no hidden layer** and **Sigmoid activation** for binary classification.
- **Moon Dataset**:
  - Non-linearly separable dataset generated using `sklearn.datasets.make_moons`.
  - Neural network with a **hidden layer** and **ReLU activation** for binary classification.
- **Training**:
  - Mini-batch gradient descent for optimization.
  - Binary cross-entropy loss for both datasets.
- **Visualization**:
  - Decision boundaries and training progress are visualized using `matplotlib`.

---

## Repository Link
🔗 [https://github.com/AyushChaudhary2003/Deep-Learning/tree/main/Experiment2](https://github.com/AyushChaudhary2003/Deep-Learning/tree/main/Experiment2)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AyushChaudhary2003/Deep-Learning.git
   cd Deep-Learning/Experiment2
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy scikit-learn matplotlib
   ```

---

## Usage
- To train the model on the **circles dataset** (no hidden layer):
  ```bash
  python train_circles.py
  ```
- To train the model on the **moons dataset** (with hidden layer):
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

## Acknowledgments
- The synthetic datasets are generated using `sklearn.datasets`.
- This implementation is inspired by foundational concepts in deep learning and neural networks.

---

Feel free to contribute or report issues! 🚀
