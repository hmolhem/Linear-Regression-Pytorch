# Salary Prediction using PyTorch Linear Regression

This project implements a **Simple Linear Regression** model using the **PyTorch** framework. The goal is to predict an individual's salary based on their years of professional experience, serving as a foundational step toward understanding Deep Learning and Convolutional Neural Networks (CNNs).

---

## 🚀 Project Overview
In this project, we treat Linear Regression as a single-layer neural network with one input and one output. This approach introduces the core components of the PyTorch workflow:
* **Tensors**: Handling data in $n$-dimensional arrays.
* **Autograd**: Automatic computation of gradients.
* **Optimization**: Updating weights using Stochastic Gradient Descent (SGD).



## 📊 Dataset
The dataset contains historical data of years of experience vs. salary.
* **Source**: [YBI Foundation - Salary Data](https://github.com/YBI-Foundation/Dataset/raw/main/Salary%20Data.csv)
* **Input (X)**: Years of Experience
* **Output (y)**: Salary

## 🛠️ Environment Setup (Conda)
To ensure all dependencies are correctly managed, follow these steps to set up your Conda environment:

```bash
# Create a new conda environment
conda create -n LSR python=3.11.14

# Activate the environment
conda activate LSR

# Install PyTorch (CPU version or CUDA if available)
conda install pytorch torchvision torchaudio -c pytorch

# Install additional libraries
conda install pandas matplotlib scikit-learn