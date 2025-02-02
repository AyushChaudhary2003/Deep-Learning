## Overview
This repository contains deep learning models and associated resources. It includes implementations of various architectures, training scripts, and datasets used for experiments.

## Objective
The objective of this lab is to implement Convolutional Neural Networks (CNNs) to classify images in the Cats vs. Dogs dataset and the CIFAR-10 dataset. The aim is to explore different configurations and analyze their impact on performance by experimenting with:
- 3 Activation Functions: ReLU, Tanh, Leaky ReLU
- 3 Weight Initialization Techniques: Xavier Initialization, Kaiming Initialization, Random Initialization
- 3 Optimizers: SGD, Adam, RMSprop

Additionally, the best CNN model for both datasets will be compared with a pre-trained ResNet-18 model.

## Features
- Pretrained models for quick inference
- Custom model training scripts
- Dataset preprocessing utilities
- Performance evaluation tools

## Dataset
Cats vs. Dogs Dataset: [Kaggle - Dogs vs. Cats](https://www.kaggle.com/datasets/ayushchaudhary2411/dogs-vs-cats)
- Pretrained models for quick inference
- Custom model training scripts
- Dataset preprocessing utilities
- Performance evaluation tools

## Installation
Repository Link: [Deep Learning - Experiment 3](https://github.com/AyushChaudhary2003/Deep-Learning/tree/main/Experiment3)
To set up the repository, clone it and install the required dependencies:
```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
```

## Usage
To train a model, run the following command:
```bash
python train.py --config configs/default.yaml
```
For inference, use:
```bash
python infer.py --model checkpoint.pth --input sample.jpg
```
