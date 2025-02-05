# Neural Network for Circle and Moon Datasets

This project consists of two separate Python implementations demonstrating the use of neural networks on synthetic datasets: a "circle" dataset and a "moon" dataset. The goal is to build and evaluate simple neural network models, including one with a custom backpropagation implementation and another using a pre-built multi-layer perceptron (MLP) classifier.

### Requirements

- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these dependencies with:

```bash
pip install numpy matplotlib scikit-learn
```

### Code Overview

#### 1. **Neural Network on Circle Dataset (MLP Classifier)**

This part uses the `make_circles` function from `sklearn.datasets` to generate a 2D circular dataset. The neural network is trained using the Multi-layer Perceptron (MLP) classifier from `sklearn.neural_network` without any hidden layers (i.e., using a linear activation function). The accuracy of the model is evaluated on a test set, and the decision boundary is visualized.

Key Steps:
- Generate synthetic circular data.
- Split the data into training and testing sets.
- Train an MLP model without hidden layers.
- Plot the decision boundary.

#### 2. **Neural Network on Moon Dataset (Custom Implementation)**

This section demonstrates the construction and training of a neural network on the "moon" dataset using custom code. The neural network has one hidden layer and employs backpropagation with the sigmoid activation function. The model is trained using gradient descent, and the loss is calculated with binary cross-entropy. After training, the model's accuracy on a test set is computed, and the decision boundary is visualized.

Key Steps:
- Generate synthetic moon-shaped data.
- Split the data into training and testing sets.
- Standardize features using `StandardScaler`.
- Implement the forward pass and backpropagation.
- Train the model using gradient descent.
- Plot the decision boundary.

### Results

- **Circle Dataset (MLP Classifier)**: The accuracy of the model without hidden layers is printed and the decision boundary is visualized.
- **Moon Dataset (Custom Neural Network)**: After training, the model's test accuracy is printed, and the decision boundary is visualized.

### Usage

1. Run both scripts separately to see how neural networks perform on different synthetic datasets.
2. Modify the hyperparameters (e.g., number of hidden units, learning rate, etc.) to explore their effect on model performance.
3. Visualize the decision boundaries to understand the decision-making process of the trained models.

---

Let me know if you'd like to make any changes or additions!
