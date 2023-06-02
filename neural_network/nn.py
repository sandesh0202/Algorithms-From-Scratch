import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class CategoricalEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        ypred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = ypred_clipped[range(samples), y_true]
            
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(ypred_clipped*y_true, axis=1)
            
        negative_log_likelihoods = np.mean(correct_confidences)
        return negative_log_likelihoods          
    
class Activattion_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output
    
class Activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims= True))
        probability = exp_values / np.sum(exp_values, axis=1, keepdims= True)
        self.output = probability
    
layer1 = Layer_Dense(2, 3)
layer2 = Layer_Dense(3, 3)
activation1 = Activattion_ReLU()
activation2 = Activation_softmax()

layer1.forward(X)
#print(layer1.output[:5])
activation1.forward(layer1.output)
#print(activation1.output[:5])
layer2.forward(activation1.output)
activation2.forward(layer2.output)
#print(activation2.output[:5])

loss_function = CategoricalEntropyLoss()
loss = loss_function.calculate(activation2.output, y)

print(loss) 