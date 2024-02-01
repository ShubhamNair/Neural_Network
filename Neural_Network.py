import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sigmoid_x = Activation.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def forward(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        self.output = Activation.sigmoid(weighted_sum)
        return self.output

    def backward(self, d_error):
        d_output = d_error * Activation.sigmoid_derivative(self.output)
        self.d_weights = np.outer(d_output, self.inputs)
        self.d_bias = np.sum(d_output)
        self.d_inputs = np.dot(self.weights, d_output)

class Layer:
    def __init__(self, input_size, num_neurons):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.outputs

    def backward(self, d_error):
        self.d_inputs = np.array([neuron.backward(d_error[i]) for i, neuron in enumerate(self.neurons)])

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(input_size, hidden_size)
        self.output_layer = Layer(hidden_size, output_size)

    def predict(self, inputs):
        hidden_layer_output = self.hidden_layer.forward(inputs)
        output_layer_output = self.output_layer.forward(hidden_layer_output)
        return output_layer_output

    def backward(self, d_error):
        self.output_layer.backward(d_error)
        self.hidden_layer.backward(self.output_layer.d_inputs)

class LossFunction:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

class GradDescent:
    @staticmethod
    def update(params, d_params, learning_rate):
        for param, d_param in zip(params, d_params):
            param -= learning_rate * d_param

class Training:
    @staticmethod
    def train(model, X_train, y_train, epochs, loss_fn, optimizer):
        for epoch in range(epochs):
            model.forward(X_train)
            loss = loss_fn.mean_squared_error(y_train, model.output_layer.outputs)
            d_loss = loss_fn.mean_squared_error_derivative(y_train, model.output_layer.outputs)
            model.backward(d_loss)

            optimizer.update(model.hidden_layer.neurons[0].weights, model.hidden_layer.neurons[0].d_weights, optimizer.learning_rate)
            optimizer.update(model.hidden_layer.neurons[0].bias, model.hidden_layer.neurons[0].d_bias, optimizer.learning_rate)
            optimizer.update(model.output_layer.neurons[0].weights, model.output_layer.neurons[0].d_weights, optimizer.learning_rate)
            optimizer.update(model.output_layer.neurons[0].bias, model.output_layer.neurons[0].d_bias, optimizer.learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
input_size = 2
hidden_size = 4
output_size = 1
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])
model = Model(input_size, hidden_size, output_size)
optimizer = GradDescent()
optimizer.learning_rate = 0.1
trainer = Training()
trainer.train(model, X_train, y_train, epochs=1000, loss_fn=LossFunction, optimizer=optimizer)
