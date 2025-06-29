from ctypes import ArgumentError
from enum import Enum
from typing import List
import numpy as np
from numpy.typing import NDArray

from src.activation_function import ActivationFunction

class PerceptronOptimizer(Enum):
    GRADIENT_DESCENT = 0
    MOMENTUM = 1
    ADAM = 2

    @classmethod
    def from_string(cls, name: str):
        return cls[name.upper()]

class NeuralNet:
    def __init__(
    self,
    input_count: int,
    hidden_layers: List[int], 
    activation_func: ActivationFunction, 
    optimizer: PerceptronOptimizer = PerceptronOptimizer.ADAM,
    beta_func: float = 1,             
    random_weight_initialize: bool = True
):
        """
        input_count: the amount of input arguments
        hidden_layers: list of the amount of neurons per layer (including output layer)
        activation_func: activation function used in all layers
        """

        if (len(hidden_layers) == 0):
            raise ArgumentError("One layer must be specified")

        self.hidden_layers = hidden_layers.copy()
        self.activation_func = activation_func
        self.weights: List[NDArray[np.float64]] = [] # NDArray can be vector or matrix, in this case matrix
        self.beta_func = beta_func
        self.data_error = 1

        prev_neuron_count = input_count
        for neurons in hidden_layers:
            if random_weight_initialize:
                weight_matrix = np.random.uniform(-1, 1, (neurons, prev_neuron_count + 1)) # +1 for bias
            else:
                weight_matrix = np.zeros((neurons, prev_neuron_count + 1))
            self.weights.append(weight_matrix)

            prev_neuron_count = neurons

        if optimizer == PerceptronOptimizer.GRADIENT_DESCENT:
            self.update_weights_func = self._gradient_descent
        elif optimizer == PerceptronOptimizer.MOMENTUM:
            self.update_weights_func = self._momentum
            self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]
            self.alpha: float = 0.9  
        else: # it will default to adam
            self.update_weights_func = self._adam_step
            self.t = 0  
            self.m = [np.zeros_like(w) for w in self.weights] 
            self.v = [np.zeros_like(w) for w in self.weights] 
            self.beta1: float = 0.9
            self.beta2: float = 0.999
            self.epsilon: float = 1e-8

    def forward_pass(self, input_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        input_values: vector of input data (without bias)
        """
        self.values: List[NDArray[np.float64]] = [input_values] # the input values and neuron values of each layer, size equals to amount of layers + 1
        self.sums: List[NDArray[np.float64]] = [] # the weighted sums without the activation, size equals to amount of layers
        current_values = input_values
        for weight_matrix in self.weights:
            current_values = np.insert(current_values, 0, 1.0)  # add bias
            z = np.dot(weight_matrix, current_values)
            self.sums.append(z)
            current_values: NDArray[np.float64] = self.activation_func.func(z, self.beta_func)
            self.values.append(current_values)
        return current_values  
    
    def update_weights(
        self,
        input_values: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.1
    ):
        self.update_weights_func(input_values, expected_output, learning_rate)
        


    def _gradient_descent (  
        self,
        input_values: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.1
    ):
        final_output = self.forward_pass(input_values)

        # Deltas for output layer
        self.data_error = 0
        error = expected_output - final_output
        self.data_error = np.sum(error**2)
        deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)

        # Backpropagate deltas and update weights
        for i in reversed(range(len(self.weights))):
            # Prepare values with bias
            values = np.insert(self.values[i], 0, 1.0)
            # If deltas is 3x1 and values 4x1, np.outer will transverse values to 1x4, to get the resulting matrix of size 3x4
            self.weights[i] += learning_rate * np.outer(deltas, values) 

            if i > 0: # if we are not on the last layer, we should calculate delta

                # Remove bias weights from current layer, they don't have associated delta
                weights_wo_bias = self.weights[i][:, 1:]

                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)

    def _momentum(
    self,
    input_values: NDArray[np.float64],
    expected_output: NDArray[np.float64],
    learning_rate: float = 0.1,
):
        final_output = self.forward_pass(input_values)

        self.data_error = 0
        error = expected_output - final_output
        self.data_error = np.sum(error**2)
        deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)

        for i in reversed(range(len(self.weights))):
            values = np.insert(self.values[i], 0, 1.0)  # agregar bias
            gradient = np.outer(deltas, values)

            # Momentum update: Δw(t+1) = -η * grad + α * Δw(t)
            delta_w = learning_rate * gradient + self.alpha * self.prev_weight_updates[i]
            self.weights[i] += delta_w
            self.prev_weight_updates[i] = delta_w  

            if i > 0:
                weights_wo_bias = self.weights[i][:, 1:]
                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)

    def _adam_step(
        self,
        input_values: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.001
    ):
        self.t += 1
        final_output = self.forward_pass(input_values)

        # Output layer delta
        self.data_error = 0
        error = expected_output - final_output
        self.data_error = np.sum(error ** 2)
        deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)

        grads = []

        for i in reversed(range(len(self.weights))):
            values = np.insert(self.values[i], 0, 1.0)
            grad = - np.outer(deltas, values)
            grads.insert(0, grad)

            if i > 0:
                weights_wo_bias = self.weights[i][:, 1:]
                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)

        # Update weights using Adam
        for i in range(len(self.weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class BasicAutoEncoder:
    def __init__(
        self,
        dataset: NDArray[np.float64],
        hidden_encoder_layers_to_latent_space: List[int],
        activation_func: ActivationFunction,
        learn_rate: float = 0.1,
        min_error: float = 0.1,
        max_epochs: int = 10000
    ):
        """
        dataset: matrix NxM (N rows of data, M variables)
        hidden_encoder_layers_to_latent_space: not including input layer, list of neurons per layer in the encoder (last is the latent space)
        """
        if dataset.shape[0] == 0 or dataset.shape[1] == 0:
            raise ValueError("Dataset must be a non-empty 2D array")

        self.dataset = dataset
        input_count = dataset.shape[1]

        self.layers = hidden_encoder_layers_to_latent_space.copy()
        for decoder_layer in hidden_encoder_layers_to_latent_space[:-1][::-1]:
            self.layers.append(decoder_layer)
        self.layers.append(input_count)

        self.neural_net = NeuralNet(input_count, self.layers, activation_func)

        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.min_error = min_error
        self.error = 100
        self.current_epoch = 0

    def has_next(self) -> bool:
        return bool(self.error > self.min_error and self.current_epoch < self.max_epochs)

    def next_epoch(self):
        if not self.has_next():
            raise Exception("Solution was already found or max epochs were reached")

        self.error = 0
        self.current_epoch += 1

        for i in range(self.dataset.shape[0]):
            row = self.dataset[i]
            self.neural_net.update_weights(row, row, self.learn_rate)
            self.error += self.neural_net.data_error

        self.error /= 2
    
    def get_current_latent_space(self) -> NDArray[np.float64]:
        """
        Returns the latent representation (vector in the bottleneck layer)
        based on the most recent forward pass.
        Raises an exception if no forward pass was done yet.
        """
        if not hasattr(self.neural_net, "values") or len(self.neural_net.values) == 0:
            raise RuntimeError("No forward pass has been performed yet.")

        latent_layer_index = len(self.neural_net.hidden_layers) // 2
        return self.neural_net.values[latent_layer_index]


    def try_current_epoch(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        inputs: single vector
        returns: prediction for this vector
        """
        return self.neural_net.forward_pass(inputs)

    def try_testing_set(self, testing_set: NDArray[np.float64]) -> List[NDArray[np.float64]]:
        """
        testing_set: matrix NxM
        returns: list with all the predictions of each row
        """
        predictions = []
        for i in range(testing_set.shape[0]):
            row = testing_set[i]
            prediction = self.try_current_epoch(row)
            predictions.append(prediction)
        return predictions
