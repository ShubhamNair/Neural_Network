import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


class ActivationFunction:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return z > 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = ActivationFunction.sigmoid(z)
        return s * (1 - s)


class Layer:
    def __init__(self, size_input, size_output, activation_function):
        self.W = np.random.randn(size_input, size_output) * np.sqrt(2. / size_input)
        self.b = np.zeros((1, size_output))
        self.activation = activation_function
        self.A = None
        self.Z = None
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        self.Z = np.dot(A_prev, self.W) + self.b
        if self.activation == 'relu':
            self.A = ActivationFunction.relu(self.Z)
        elif self.activation == 'sigmoid':
            self.A = ActivationFunction.sigmoid(self.Z)
        return self.A

    def backward(self, dA, A_prev, lambda_reg=0.1, m=1):
        if self.activation == 'relu':
            dZ = dA * ActivationFunction.relu_derivative(self.Z)
        elif self.activation == 'sigmoid':
            dZ = dA * ActivationFunction.sigmoid_derivative(self.Z)

        self.dW = np.dot(A_prev.T, dZ) / m + (lambda_reg / m) * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, self.W.T)

        return dA_prev

    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class NeuralNetwork:
    def __init__(self, layer_dims, activation_functions):
        self.layers = []
        for i in range(1, len(layer_dims)):
            self.layers.append(Layer(layer_dims[i - 1], layer_dims[i], activation_functions[i - 1]))

    def forward_propagation(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_cost(self, AL, Y, lambda_reg=0.1):
        m = Y.shape[0]

        # Clip AL to avoid numerical issues with np.log
        AL_clipped = np.clip(AL, 1e-10, 1 - 1e-10)

        # Compute cross-entropy cost with clipped AL
        cross_entropy_cost = -np.sum(Y * np.log(AL_clipped) + (1 - Y) * np.log(1 - AL_clipped)) / m

        # Compute L2 regularization cost
        L2_cost = sum(np.sum(np.square(layer.W)) + np.sum(np.square(layer.b)) for layer in self.layers)
        L2_cost = (lambda_reg / (2 * m)) * L2_cost

        # Total cost
        cost = cross_entropy_cost + L2_cost

        return np.squeeze(cost)

    def backward_propagation(self, AL, Y, X, lambda_reg=0.1):
        m = Y.shape[0]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA = dAL

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = X if i == 0 else self.layers[i - 1].A
            dA = layer.backward(dA, A_prev, lambda_reg, m)

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def train(self, X, Y, learning_rate, num_iterations, print_cost):
        for i in range(num_iterations):
            AL = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y, lambda_reg=0.1)
            self.backward_propagation(AL, Y, X, lambda_reg=0.1)
            self.update_parameters(learning_rate)
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost:.6f}")


# Load the MNIST dataset directly from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Filter out all digits except 0 and 1
filter_index = np.where((y_train == 0) | (y_train == 1))
x_train, y_train = x_train[filter_index], y_train[filter_index]

filter_index_test = np.where((y_test == 0) | (y_test == 1))
x_test, y_test = x_test[filter_index_test], y_test[filter_index_test]

# Preprocess the data
x_train = x_train.reshape(-1, 784) / 255.0  # Normalize and flatten the images
x_test = x_test.reshape(-1, 784) / 255.0

# Define network architecture
layer_dims = [784, 10, 8, 8, 4, 1]  # Input size adjusted for 28x28 MNIST images
activation_functions = ['relu', 'relu', 'relu', 'relu', 'sigmoid']

# Initialize the neural network
nn = NeuralNetwork(layer_dims, activation_functions)

# Train the neural network
# nn.train(x_train, y_train, learning_rate=0.1, num_iterations=2500, print_cost=True)
nn.train(x_train, y_train, learning_rate=0.001, num_iterations=2500, print_cost=True)

# Perform a forward pass on the test set
AL = nn.forward_propagation(x_test)
predictions = AL > 0.5  # Convert probabilities to binary predictions

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
