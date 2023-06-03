import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Initialize the neural network framework
nnfs.init()

# Generate spiral data for training
X, y = spiral_data(100, 3)

# Define a class for a dense layer in the neural network
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases randomly
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        # Perform forward propagation and calculate the output
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

# Define a class for calculating loss
class Loss:
    def calculate(self, output, y):
        # Calculate the loss between the predicted output and the true values
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Define a subclass for categorical cross-entropy loss
class CategoricalEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        # Apply softmax function to the predicted output and calculate the loss
        samples = len(y_pred)
        ypred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = ypred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(ypred_clipped*y_true, axis=1)
            
        negative_log_likelihoods = np.mean(correct_confidences)
        return negative_log_likelihoods

# Define a class for the ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        # Apply ReLU activation function element-wise to the inputs
        self.output = np.maximum(0, inputs)
        return self.output

# Define a class for the softmax activation function
class Activation_softmax:
    def forward(self, inputs):
        # Apply softmax activation function to the inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Create instances of the defined classes
layer1 = Layer_Dense(2, 3)
layer2 = Layer_Dense(3, 3)
activation1 = Activation_ReLU()
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