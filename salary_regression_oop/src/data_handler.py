
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        import pandas as pd
        data = pd.read_csv(self.file_path)
        return data
    
    def get_data(self, data, x_label, y_label):
        x = data[x_label].values
        y = data[y_label].values
        return x, y
    
    def split_data(self, train_data_size=0.7):
        from sklearn.model_selection import train_test_split
        data = self.load_data()
        train_set, test_set = train_test_split(data, test_size=1 - train_data_size, random_state=42)
        return train_set, test_set
    
    
        
    def plot(self, data, x_label, y_label, title=None, color=None):
        import matplotlib.pyplot as plt
        plt.scatter(data[x_label], data[y_label], color='blue' if color is None else color, alpha=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        else:
            plt.title(f'{y_label} vs {x_label}')
        plt.show()