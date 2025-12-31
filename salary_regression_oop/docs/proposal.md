# Project Proposal: Object-Oriented Linear Regression from Scratch

## 1. Introduction
This project aims to implement a univariate Linear Regression model to predict employee salaries based on years of experience. Unlike standard implementations that rely on high-level libraries like Scikit-Learn for the model logic, this project will construct the algorithm from scratch using NumPy. The primary goal is to bridge the gap between mathematical theory and software engineering by organizing the solution into a modular Object-Oriented architecture.

## 2. Objectives
* **Technical:** Implement the core components of machine learning (Model, Loss, Optimizer, Training Loop) as distinct Python classes.
* **Educational:** Deepen understanding of Gradient Descent mechanics and Python OOP principles.
* **Practical:** successfully predict salary values on a test dataset.

## 3. System Architecture
The project will be structured into five interacting classes, mirroring standard ML frameworks:

1.  **`DataHandler`**
    * **Role:** ETL (Extract, Transform, Load).
    * **Responsibility:** Fetches data from the remote source, handles local caching in the `data/` folder, performs train/test splits, and visualizes raw data.
2.  **`Model`**
    * **Role:** The Predictor.
    * **Responsibility:** Stores parameters ($\theta_0, \theta_1$) and implements the forward pass equation: $\hat{y} = \theta_1 x + \theta_0$.
3.  **`Loss`**
    * **Role:** The Critic.
    * **Responsibility:** Calculates Mean Squared Error (MSE) and computes gradients (derivatives) to guide learning.
4.  **`Optimizer`**
    * **Role:** The Teacher.
    * **Responsibility:** Updates model parameters using the calculated gradients and a defined learning rate.
5.  **`Trainer`**
    * **Role:** The Orchestrator.
    * **Responsibility:** Manages the training loop, connecting the Model, Loss, and Optimizer to iteratively improve predictions.
6.  **`Evaluator`**
    * **Role:** The Assessor.
    * **Responsibility:** Evaluates model performance on the test dataset and visualizes predicted vs. actual salaries.


## 4. Data Source
* **Dataset:** Salary Data (Years of Experience vs. Salary)
* **Source:** [YBI Foundation GitHub Repository](https://github.com/YBI-Foundation/Dataset/raw/main/Salary%20Data.csv)
* **Type:** Simple Regression (1 feature, 1 target)

## 5. Tools & Technologies
* **Language:** Python 3.x
* **IDE:** Jupyter Notebook / VSCode
* **Scikit-Learn:** Only for data splitting (train/test)
* **Core Math:** NumPy
* **Data Handling:** Pandas
* **Visualization:** Matplotlib / Seaborn