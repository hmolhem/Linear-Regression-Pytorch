
class DataHandler:
    def __init__(self, file_path):
        import pandas as pd
        self.file_path = file_path # Path to the CSV file
        self.data = pd.read_csv(self.file_path) # Load data into a DataFrame
        self.original_data = self.data.copy() # Keep a copy of the original data
        self.numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist() # Numeric columns
        

    def load_data(self):
        import pandas as pd
        data = pd.read_csv(self.file_path)
        return data
    
    def get_labels(self):
        return tuple(self.data.columns.tolist())
    
    def get_data(self, data, x_label, y_label):
        x = data[x_label].values
        y = data[y_label].values
        return x, y
    
    def split_data(self, train_size=0.7, seed=42):
        """
        Splits the data into training and testing sets.
        train_size: float (0.0 to 1.0)
        seed: int for reproducibility
        """
        import numpy as np
        # 1. Shuffle the data indices
        np.random.seed(seed)
        # Create a shuffled array of indices based on the length of the dataframe
        shuffled_indices = np.random.permutation(len(self.data))
        
        # 2. Determine the split point
        split_index = int(len(self.data) * train_size)
        
        # 3. Slice indices
        train_indices = shuffled_indices[:split_index]
        test_indices = shuffled_indices[split_index:]
        
        # 4. Create the sets using .iloc
        train_set = self.data.iloc[train_indices]
        test_set = self.data.iloc[test_indices]
        
        return train_set, test_set

    
    def min_max_normalize(self):
        """Formula: (x - min) / (max - min) -> Range [0, 1]"""
        df = self.data[self.numeric_cols] # DataFrame with only numeric columns
        self.data[self.numeric_cols] = (df - df.min()) / (df.max() - df.min()) # Normalize
        return self.data
    
    def inverse_min_max(self, normalized_df=None):
        """Formula: x_orig = x_norm * (max - min) + min"""
        if 'min' not in self.params:
            raise ValueError("No Min-Max parameters found. Normalize first!")
            
        target_df = normalized_df if normalized_df is not None else self.data
        # Apply the reverse math
        reversed_df = target_df[self.numeric_cols] * (self.params['max'] - self.params['min']) + self.params['min']
        return reversed_df

    def z_score_standardize(self):
        """Formula: (x - mean) / std -> Mean=0, Std=1"""
        df = self.data[self.numeric_cols] # DataFrame with only numeric columns
        self.data[self.numeric_cols] = (df - df.mean()) / df.std() # Standardize
        return self.data
    
    def inverse_z_score(self, standardized_df=None):
        """Formula: x_orig = (x_std * std) + mean"""
        if 'mean' not in self.params:
            raise ValueError("No Z-Score parameters found. Standardize first!")
            
        target_df = standardized_df if standardized_df is not None else self.data
        
        reversed_df = (target_df[self.numeric_cols] * self.params['std']) + self.params['mean']
        return reversed_df

    def robust_scale(self):
        """Formula: (x - median) / (Q3 - Q1)"""
        df = self.data[self.numeric_cols]
        
        # Calculate and store parameters before scaling
        self.params['median'] = df.median() # Median
        q1 = df.quantile(0.25) # First Quartile
        q3 = df.quantile(0.75) # Third Quartile
        self.params['iqr'] = q3 - q1 # Interquartile Range
        
        # Apply Robust Scaling
        # We use the stored params to ensure consistency
        self.data[self.numeric_cols] = (df - self.params['median']) / self.params['iqr']
        return self.data
    
    
    def inverse_robust_scale(self, target_df=None):
        """Formula: x_orig = (x_scaled * IQR) + median"""
        if 'median' not in self.params:
            raise ValueError("No Robust Scale parameters found. Scale the data first!")
            
        # Use provided dataframe (like model predictions) or the internal data
        df = target_df[self.numeric_cols] if target_df is not None else self.data[self.numeric_cols]
        
        # Reverse the math: multiply by IQR then add median
        reversed_df = (df * self.params['iqr']) + self.params['median']
        return reversed_df
    

    def mean_normalize(self):
        """Formula: (x - mean) / (max - min)"""
        df = self.data[self.numeric_cols]
        # Store parameters before normalizing
        self.params['mean'] = df.mean()
        self.params['max_min_diff'] = df.max() - df.min()
        
        self.data[self.numeric_cols] = (df - self.params['mean']) / self.params['max_min_diff']
        return self.data
    
    def inverse_mean_normalize(self, target_df=None):
        """Formula: x_orig = (x_norm * (max - min)) + mean"""
        if 'mean' not in self.params:
            raise ValueError("No Mean Norm parameters found. Normalize first!")
            
        df = target_df[self.numeric_cols] if target_df is not None else self.data[self.numeric_cols]
        
        # Reverse the math
        reversed_df = (df * self.params['max_min_diff']) + self.params['mean']
        return reversed_df

    def unit_vector_normalize(self):
        """Formula: x / sqrt(sum(x^2))"""
        import numpy as np
        df_numeric = self.data[self.numeric_cols]
        # Calculate and store the L2 norm for each row
        l2_norm = np.sqrt((df_numeric**2).sum(axis=1))
        self.params['l2_norm'] = l2_norm
        
        # Divide each row by its norm
        self.data[self.numeric_cols] = df_numeric.divide(l2_norm, axis=0)
        return self.data
    
    def inverse_unit_vector(self, target_df=None):
        """Formula: x_orig = x_norm * original_l2_norm"""
        if 'l2_norm' not in self.params:
            raise ValueError("No L2 Norm parameters found. Normalize first!")
            
        df = target_df[self.numeric_cols] if target_df is not None else self.data[self.numeric_cols]
        
        # Multiply each row by its stored original norm
        reversed_df = df.multiply(self.params['l2_norm'], axis=0)
        return reversed_df

    def reset_data(self):
        """Restore original values."""
        self.data = self.original_data.copy() # Restore original data
        return self.data
    