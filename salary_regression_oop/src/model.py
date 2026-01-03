import numpy as np
import pandas as pd

class Model:
    def __init__(self, theta0=0.0, theta1=0.0):
        self.theta0 = theta0
        self.theta1 = theta1

    def linear_estimator(self, df, x_label, y_label):
        """
        Args:
            df: pandas DataFrame
            x_label: feature column name
            y_label: target column name for output
        Returns:
            pandas DataFrame with [x_label, y_label]
        """
        x_array = df[x_label].values
        predictions = self.theta0 + self.theta1 * x_array
        
        # Return DataFrame with both input and predictions
        result_df = pd.DataFrame({
            x_label: x_array, 
            y_label: predictions
        }, index=df.index)
        
        return result_df
    
