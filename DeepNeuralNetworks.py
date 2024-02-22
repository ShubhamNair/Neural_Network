import numpy as np

class Parameters:
    def __init__(self):
        self.weights = None
        self.bias = None

    def set_bias(self, bias):
        self.bias = bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias


class Neuron:
    def __init__(self, input_size):
        self.weights = None
        self.bias = None
        self.input_size = input_size
        self.aggregate_signal = None
        self.activation = None
        self.output = None
        self.delta = None

    def neuron(self, inputs):
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be set before forward pass")

        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Inputs shape ({inputs.shape}) is incompatible with weights shape ({self.weights.shape})")

        self.aggregate_signal = np.sum(np.dot(inputs, self.weights.T) + self.bias)
        self.activation = self.layer.activation(self.aggregate_signal)
        self.output = self.activation

    def update_weights(self, learning_rate, delta):
        self.weights += learning_rate * delta * self.input
        self.bias += learning_rate * delta


class ActivationFunctions:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Activation:
    def __init__(self, type):
        self.type = type

    def __call__(self, inputs):
        if self.type == "linear":
            return ActivationFunctions.linear(inputs)
        elif self.type == "relu":
            return ActivationFunctions.relu(inputs)
        elif self.type == "sigmoid":
            return ActivationFunctions.sigmoid(inputs)
        elif self.type == "tanh":
            return ActivationFunctions.tanh(inputs)
        elif self.type == "softmax":
            return ActivationFunctions.softmax(inputs)
        else:
            raise ValueError(f"Invalid activation function type: {self.type}")


class Loss:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred))


class Layer:
    def __init__(self, neurons, parameters, activation_type, regularization=None, dropout_rate=None):
        self.neurons = neurons
        self.parameters = parameters
        self.weights = self.parameters.get_weights()
        self.bias = self.parameters.get_bias()
        self.activation_type = activation_type
        self.activation = Activation(activation_type)
        self.neurons_layer = len(neurons)
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def forward(self, inputs, training=True):
        outputs = []
        for neuron in self.neurons:
            neuron.weights = np.random.rand(self.neurons_layer)
            neuron.bias = self.bias
            neuron.layer = self
            neuron.neuron(inputs)
            outputs.append(neuron.output)
        outputs = np.array(outputs)
        
        if training and self.dropout_rate is not None:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=outputs.shape) / (1 - self.dropout_rate)
            outputs *= self.dropout_mask
        
        return outputs

    def backward(self, inputs, deltas, learning_rate):
        next_layer = self.layer_below
        next_deltas = next_layer.deltas
        for i, neuron in enumerate(self.neurons):
            delta = deltas[i]
            activation_derivative = neuron.activation_derivative()
            error = np.dot(delta, next_deltas) * activation_derivative
            if self.regularization == 'l1':
                error += learning_rate * np.sign(neuron.weights)
            elif self.regularization == 'l2':
                error += learning_rate * neuron.weights
            neuron.update_weights(learning_rate, error)


class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []

    def add_layer(self, num_neurons, activation_type, regularization=None, dropout_rate=None):
        parameters = Parameters()
        parameters.set_bias(np.random.rand(num_neurons))
        layer = Layer([Neuron(self.input_size) for _ in range(num_neurons)], parameters, activation_type, regularization, dropout_rate)
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-2].layer_below = layer

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def calculate_loss(self, predictions, targets, loss_type):
        if loss_type == "mean_squared_error":
            return Loss.mean_squared_error(targets, predictions)
        elif loss_type == "binary_cross_entropy":
            return Loss.binary_cross_entropy(targets, predictions)
        elif loss_type == "categorical_cross_entropy":
            return Loss.categorical_cross_entropy(targets, predictions)
        else:
            raise ValueError(f"Invalid loss function type: {loss_type}")

    def train(self, X_train, y_train, X_val, y_val, learning_rate, epochs, loss_type):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X_train)

            # Calculate training loss
            training_loss = self.calculate_loss(predictions, y_train, loss_type)

            # Validation forward pass
            val_predictions = self.forward(X_val)

            # Calculate validation loss
            validation_loss = self.calculate_loss(val_predictions, y_val, loss_type)

            # Backpropagation
            self.layers[-1].deltas = predictions - y_train
            for layer in reversed(self.layers[:-1]):
                layer.backward(X_train, learning_rate)

            # Print epoch information
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Training Loss = {training_loss}, Validation Loss = {validation_loss}")


# Prototype of training, validation, and testing sets
input_size = 2
X_train = np.random.randint(2, size=(100, input_size))
y_train = np.random.randint(2, size=(100, 1))

X_val = np.random.randint(2, size=(20, input_size))
y_val = np.random.randint(2, size=(20, 1))

X_test = np.random.randint(2, size=(20, input_size))
y_test = np.random.randint(2, size=(20, 1))

# Create and train the neural network
dnn = NeuralNetwork(input_size)
dnn.add_layer(3, "relu", regularization='l2', dropout_rate=0.2)
dnn.add_layer(2, "sigmoid", regularization='l2', dropout_rate=0.2)
dnn.add_layer(1, "sigmoid", regularization='l2', dropout_rate=0.2)

learning_rate = 0.01
epochs = 1000
loss_type = "mean_squared_error"

dnn.train(X_train, y_train, X_val, y_val, learning_rate, epochs, loss_type)

# Test the neural network
predictions = dnn.forward(X_test)
test_loss = dnn.calculate_loss(predictions, y_test, loss_type)
print(f"Test Loss: {test_loss}")
