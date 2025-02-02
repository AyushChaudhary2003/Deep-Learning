# Neural Network Decision Boundaries on the "Circles" Dataset

## Overview
This project explores how neural networks classify non-linearly separable data using the `make_circles` dataset. It compares a simple perceptron (no hidden layers) with a deeper model using a hidden layer and ReLU activation.

## Dataset
- The dataset consists of two concentric circles, making it non-linearly separable.
- Noise is added to introduce variation.
- The dataset is split into 80% training and 20% testing.

## Models
### Model 1: No Hidden Layer (Linear Classifier)
- Uses `MLPClassifier` with no hidden layers (`hidden_layer_sizes=()`).
- Uses an identity activation function (linear classifier).
- Expected to have low accuracy due to the dataset's non-linearity.

### Model 2: Single Hidden Layer with ReLU Activation
- Uses `MLPClassifier` with one hidden layer (`hidden_layer_sizes=(5,)`).
- Applies ReLU activation to introduce non-linearity.
- Expected to perform significantly better than Model 1.

## Visualization
- Decision boundaries for both models are plotted.
- Highlights the effectiveness of using a hidden layer in classification.

## Installation & Execution
1. Clone the repository:
   ```sh
   git clone https://github.com/AyushChaudhary2003/Deep-Learning.git
   cd Deep-Learning/Experiment2
   ```
2. Install dependencies:
   ```sh
   pip install numpy matplotlib scikit-learn
   ```
3. Run the script:
   ```sh
   python experiment2.py
   ```

## Results
- The simple perceptron struggles with classification due to its linear nature.
- The model with a hidden layer successfully captures the circular pattern.
- Accuracy and decision boundary plots highlight the improvements.

## Repository
For full code and details, visit the GitHub repository:  
[GitHub Repository](https://github.com/AyushChaudhary2003/Deep-Learning/tree/main/Experiment2)
