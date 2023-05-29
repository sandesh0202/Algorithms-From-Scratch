import numpy as np
from collections import Counter

def euclidean(x1,x2):
    distance = np.sqrt(sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        pred = [self._predict(x) for x in X]
        
    def _predict(self, x):
        distances = [euclidean(x, x_train) for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k] 
        k_neighbours_labels = [self.y_train[i] for i in k_indices]
        
        prediction = Counter(k_neighbours_labels).most_common()
        return prediction[0][0]