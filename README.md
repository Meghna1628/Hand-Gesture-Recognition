# Hand Gesture Recognition using Convolutional Neural Networks (CNN)

## Overview
This project implements a **Hand Gesture Recognition System** using a Convolutional Neural Network (CNN). The system recognizes hand gestures from images based on the **Sign Language MNIST dataset**, leveraging PyTorch for model building and training. This includes data preprocessing, model architecture definition, training, evaluation, and visualization of results.

---

## Features
- **Custom CNN Architecture**: A deep learning model with convolutional and fully connected layers.
- **Dataset Splitting**: Train-test split for model validation.
- **Training and Evaluation**: Metrics include accuracy and loss for both training and test data.
- **Visualization**: Epoch-wise accuracy and loss plots for model performance evaluation.
- **Model Summary**: Visualizes the CNN architecture using `torchsummary` and `torchviz`.

---

## Prerequisites
The project requires the following:
- Python 3.8+
- Libraries:
  - PyTorch
  - NumPy
  - pandas
  - scikit-learn
  - Matplotlib
  - `torchsummary` and `torchviz` for model visualization

Install the required dependencies using:
```bash
pip install torch torchvision scikit-learn matplotlib pandas torchsummary torchviz
```

---

## Dataset
The **Sign Language MNIST** dataset is used for training and testing. The dataset includes:
- **Images**: 28x28 grayscale images of hand gestures.
- **Labels**: 26 classes representing the alphabets A-Z.

  You can download the dataset from Kaggle's [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?select=sign_mnist_train).
---

## Results
- **Training Accuracy**: Achieved high accuracy with consistent improvement across epochs.
- **Test Accuracy**: Generalized well to unseen data with comparable accuracy.
- **Visualization**:
  - Accuracy and loss plots.
  - CNN model architecture diagram (`model.png`).

---

