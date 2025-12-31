import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self, style="seaborn-v0_8-muted"):
        """Initialize style settings for all plots."""
        plt.style.use(style)
        sns.set_theme(style="whitegrid")

    def plot_scatter(self, data, x_label, y_label, title=None, color='blue', show_reg=False):
        """Enhanced scatter plot with optional regression trend line."""
        plt.figure(figsize=(10, 6))
        if show_reg:
            sns.regplot(data=data, x=x_label, y=y_label, 
                        scatter_kws={'alpha':0.5, 'color':color}, 
                        line_kws={'color':'red'})
        else:
            plt.scatter(data[x_label], data[y_label], color=color, alpha=0.5)
            
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title if title else f'{y_label} vs {x_label}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def plot_regression_results(self, y_true, y_pred, title="Actual vs Predicted"):
        """Scatter plot for regression performance with an identity (1:1) line."""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
        
        # Perfect prediction line
        max_val = max(np.max(y_true), np.max(y_pred))
        min_val = min(np.min(y_true), np.min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_residuals(self, y_true, y_pred):
        """Plots errors to check for non-linear patterns (ideally a random cloud)."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 5))
        plt.scatter(y_pred, residuals, alpha=0.5, color='orange')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (Errors)')
        plt.title('Residual Plot: Error Analysis')
        plt.show()

    def plot_loss(self, train_loss, val_loss=None, title="Model Loss Over Epochs"):
        """Plots training and validation loss for neural networks."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
        if val_loss:
            plt.plot(val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_distribution(self, data, column_name):
        """Visualizes feature distribution (Histogram + KDE)."""
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column_name], kde=True, color='purple')
        plt.title(f'Distribution of {column_name}')
        plt.show()

    def plot_correlation(self, df):
        """Generates a heatmap for the correlation matrix (lower triangle only)."""
        plt.figure(figsize=(12, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def plot_feature_importance(self, feature_names, importances):
        """Horizontal bar chart for model feature importance."""
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance')
        plt.show()
