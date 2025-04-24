import numpy as np
import pandas as pd
from matplotlib import image
import os

# Function that goes through all subpaths in a path directory and reads all the images there and returns a dataframe
def load_images(path):
    column_names = [f'pixel{i}' for i in range(1, 28 * 28 + 1)]
    column_names.append('label')
    images_df = pd.DataFrame(columns=column_names)

    for subpath in os.listdir(path):
        images = []
        full_path = os.path.join(path, subpath)
        for img in os.listdir(full_path):
            img_data = image.imread(os.path.join(full_path, img))
            images.append(img_data.flatten())
        
        subpath_df = pd.DataFrame(images, columns=column_names[:-1])
        subpath_df['label'] = subpath
        
        # Drop empty or all-NA columns before concatenating
        subpath_df = subpath_df.dropna(axis=1, how='all')
        
        images_df = pd.concat([images_df, subpath_df], ignore_index=True)
    
    return images_df

train_set = load_images("Train")
test_set = load_images("Test")

X_train_original = train_set.drop(columns='label')
y_train_original = train_set['label']

X_test_original = test_set.drop(columns='label')
y_test_original = test_set['label']

X_train = np.array(X_train_original)
y_train = np.array(y_train_original)

X_test = np.array(X_test_original)
y_test = np.array(y_test_original)

# OneHotEncode labels
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Reshape the input to (n_samples, 28*28) instead of (n_samples, 1, 28*28)
X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

# Activation functions and their derivatives
def relu(X, derivative=False):
    if derivative:
        return np.where(X > 0, 1, 0)
    return np.maximum(0, X)

def softmax(X):
    e_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # For numerical stability
    return e_X / np.sum(e_X, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred, derivative=False):
    if derivative:
        return y_pred - y_true  # derivative of cross-entropy
    else:
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / m  # Added small value to avoid log(0)

# Layer class definition
class Layer:
    def __init__(self, input_size, output_size, activator_type):
        self.weights = np.random.randn(input_size, output_size) * 0.01  # Initialize weights with small random values
        self.bias = np.zeros((1, output_size))  # Initialize biases to zeros
        self.activator_type = activator_type

    def forward(self, input_data):
        self.input_data = input_data
        self.predictions = np.dot(input_data, self.weights) + self.bias  # Linear combination
        if self.activator_type == 'relu':
            self.activated_predictions = relu(self.predictions)  # Apply ReLU activation function
        elif self.activator_type == 'softmax':
            self.activated_predictions = softmax(self.predictions)  # Apply Softmax for output layer
        return self.activated_predictions

    def backward(self, dA, learning_rate):
        # Backpropagation: Calculate gradients and update weights and biases
        
        if self.activator_type == 'relu':
            # Derivative of ReLU
            dZ = dA * relu(self.predictions, derivative=True)
        elif self.activator_type == 'softmax':
            # Derivative of Softmax (cross-entropy loss already has derivative)
            dZ = dA  # For softmax with cross-entropy, dA is directly the gradient of the loss

        # Gradient of loss w.r.t weights and biases
        m = self.input_data.shape[0]  # Number of examples
        dW = np.dot(self.input_data.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        
        # Return the gradient for the previous layer (dA for the previous layer)
        dA_prev = np.dot(dZ, self.weights.T)
        return dA_prev

# Neural Network class definition
class NeuralNetwork:
    def __init__(self, layer_sizes, activator_type):
        # Create layers
        self.layers = []
        for i in range(1, len(layer_sizes)):
            activator = 'relu' if i < len(layer_sizes) - 1 else 'softmax'  # Use ReLU for hidden layers and Softmax for output
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], activator))

    def forward(self, X):
        # Perform forward pass through all layers
        self.cache = []  # To store activations of each layer
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
            self.cache.append(activation)
        return activation

    def backward(self, X, y_true, learning_rate):
        # Perform backpropagation through all layers
        
        # Compute loss gradient (derivative of Cross-Entropy loss)
        dA = cross_entropy_loss(y_true, self.cache[-1], derivative=True)
        
        # Backpropagate through each layer (in reverse order)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        # Train the network for a fixed number of epochs
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Compute the loss
            loss = cross_entropy_loss(y, predictions)
            
            # Backward pass (update weights and biases)
            self.backward(X, y, learning_rate)
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss}")

# Network configuration: 3 layers, with 28*28 input features, 100 hidden neurons, and 10 output neurons (for MNIST)
layer_sizes = [28*28, 100, 5]  # [input size, hidden layer size, output size]
activator_type = 'relu'  # ReLU for hidden layers, Softmax for output

# Create the neural network
nn = NeuralNetwork(layer_sizes, activator_type)

# Train the network for 1000 epochs with a learning rate of 0.1
epochs = 1000
learning_rate = (0.136+0.15)/2
nn.train(X_train, y_train, epochs, learning_rate)
