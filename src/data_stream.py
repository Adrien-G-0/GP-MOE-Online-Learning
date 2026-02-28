import numpy as np

class DataStreamer:
    """
    Simulates the sequential arrival of observations (x_i, y_i) one by one.
    Also handles standardization of the data (zero mean, unit variance).
    """
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): Full dataset features, shape (N, D)
            y (np.ndarray): Full dataset targets, shape (N,)
        """
        self.X_raw = np.array(X)
        self.y_raw = np.array(y)
        self.N, self.D = self.X_raw.shape
        self.current_index = 0
        
        # Standardization parameters
        self.X_mean = np.zeros(self.D)
        self.X_std = np.ones(self.D)
        self.y_mean = 0.0
        self.y_std = 1.0
        
        self.is_standardized = False

    def standardize(self):
        """Standardize the entire dataset (zero mean, unit variance)."""
        self.X_mean = np.mean(self.X_raw, axis=0)
        self.X_std = np.std(self.X_raw, axis=0)
        # Handle zero standard deviation
        self.X_std[self.X_std == 0] = 1.0
        
        self.y_mean = np.mean(self.y_raw)
        self.y_std = np.std(self.y_raw)
        if self.y_std == 0:
            self.y_std = 1.0
            
        self.X_stdized = (self.X_raw - self.X_mean) / self.X_std
        self.y_stdized = (self.y_raw - self.y_mean) / self.y_std
        self.is_standardized = True

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next observation (x_i, y_i)."""
        if self.current_index >= self.N:
            raise StopIteration
            
        if self.is_standardized:
            x_i = self.X_stdized[self.current_index]
            y_i = self.y_stdized[self.current_index]
        else:
            x_i = self.X_raw[self.current_index]
            y_i = self.y_raw[self.current_index]
            
        self.current_index += 1
        return x_i, y_i
        
    def reset(self):
        """Resets the streamer to the beginning."""
        self.current_index = 0
