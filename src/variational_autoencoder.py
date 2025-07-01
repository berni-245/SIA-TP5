from ast import Tuple
from ctypes import ArgumentError
from enum import Enum
from math import sqrt
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
        final_output: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.1
    ):
        self.update_weights_func(final_output, expected_output, learning_rate)

    def _gradient_descent (  
        self,
        final_output: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.1
    ):
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
        final_output: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.1,
    ):
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
        final_output: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.001
    ):
        self.t += 1
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

class VariationalAutoEncoder:
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
        self.latent_space_dim = hidden_encoder_layers_to_latent_space[-1]

        self.encoder_layers = hidden_encoder_layers_to_latent_space.copy()
        self.encoder_layers[-1] = 2 * self.latent_space_dim # the latent space must generate 2 for each z
        self.encoder = NeuralNet(input_count, self.encoder_layers, activation_func)
        
        self.decoder_layers = hidden_encoder_layers_to_latent_space.copy()[:-1][::-1] # remove the latent_space and reverse the array
        self.decoder_layers.append(input_count) # add to the end the input count
        self.decoder = NeuralNet(self.latent_space_dim, self.decoder_layers, activation_func)

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

            output = self._forward_pass(row)
            recon_error, kl_divergence, total_loss = self._loss_function(row, output)
            self.error += total_loss

            # Backprop: decoder
            self.decoder.update_weights(output, row, self.learn_rate)

            # Backprop: encoder (KL + grad de decoder hacia encoder)
            # Calcular derivada del loss w.r.t. z
            # Nota: derivada del MSE (1/2)||x-x'||^2 respecto a z
            decoder_weights_0 = self.decoder.weights[0][:, 1:]  # sin bias
            delta_z = np.dot(
                decoder_weights_0.T,
                recon_error * self.decoder.activation_func.deriv(self.decoder.sums[0], self.decoder.beta_func)
            )

            # KL divergence grad
            d_kl_d_mu = self.mu_array
            d_kl_d_logvar = 0.5 * (np.exp(self.log_var_array) - 1)

            # Propagamos total hacia atrás (z depende de mu/logvar)
            dz_dmu = 1
            dz_dlogvar = 0.5 * np.exp(0.5 * self.log_var_array) * np.random.normal(size=self.log_var_array.shape)

            grad_mu = delta_z * dz_dmu + d_kl_d_mu
            grad_logvar = delta_z * dz_dlogvar + d_kl_d_logvar

            # Target ficticio para encoder: volver a mu + logvar modificados por grad
            pseudo_target = np.concatenate([self.mu_array - self.learn_rate * grad_mu,
                                            self.log_var_array - self.learn_rate * grad_logvar])

            self.encoder.update_weights(np.concatenate(
                [self.mu_array, self.log_var_array]
            ), pseudo_target, self.learn_rate)

        self.error /= self.dataset.shape[0]


    def _sampling(self):
        std = np.exp(0.5 * self.log_var_array)
        eps = np.random.normal(size=std.shape)
        return self.mu_array + std * eps
    
    def _forward_pass(self, input_data: NDArray):
        mu_and_log_var_arrays = self.encoder.forward_pass(input_data) # dim of 2 * self.latent_space_dim
        self.mu_array = mu_and_log_var_arrays[:self.latent_space_dim] # the first half are the mus
        self.log_var_array = mu_and_log_var_arrays[self.latent_space_dim:] # the second half are the log_var
        self.z_values = self._sampling()        
        output = self.decoder.forward_pass(self.z_values)
        return output
    
    def _loss_function(self, input_data: NDArray, output: NDArray, lam: float = 1):
        reconstruction_error = sqrt(np.sum((input_data - output)**2))
        kl_divergence = -0.5 * np.sum(1 + self.log_var_array - self.mu_array**2 - np.exp(self.log_var_array))
        total_loss = reconstruction_error + lam * kl_divergence
        return (reconstruction_error, kl_divergence, total_loss)

    
    def get_current_latent_space(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the latent representation (vector in the bottleneck layer)
        after doing a forward pass with the given values
        """
        self._forward_pass(input)

        return self.z_values

    def decode_from_latent_space_representation(self, latent_space_representation: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the output decoded from the latent_space_representation given
        """
        return self.decoder.forward_pass(latent_space_representation)

    def try_current_epoch(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        inputs: single vector
        returns: prediction for this vector
        """
        return self._forward_pass(inputs)

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
