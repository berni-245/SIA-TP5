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
        activate_output: bool = True
    ):
        if (len(hidden_layers) == 0):
            raise ArgumentError("One layer must be specified")

        self.hidden_layers = hidden_layers.copy()
        self.activation_func = activation_func
        self.weights: List[NDArray[np.float64]] = []
        self.beta_func = beta_func
        self.data_error = 1
        self.activate_output = activate_output

        prev_neuron_count = input_count
        for neurons in hidden_layers:
            scale = np.sqrt(2.0 / (prev_neuron_count + 1 + neurons))
            weight_matrix = np.random.normal(0, scale, (neurons, prev_neuron_count + 1))
            self.weights.append(weight_matrix)
            prev_neuron_count = neurons

        if optimizer == PerceptronOptimizer.GRADIENT_DESCENT:
            self.update_weights_func = self._gradient_descent
        elif optimizer == PerceptronOptimizer.MOMENTUM:
            self.update_weights_func = self._momentum
            self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]
            self.alpha: float = 0.9
        else:
            self.update_weights_func = self._adam_step
            self.t = 0
            self.m = [np.zeros_like(w) for w in self.weights]
            self.v = [np.zeros_like(w) for w in self.weights]
            self.beta1: float = 0.9
            self.beta2: float = 0.999
            self.epsilon: float = 1e-8

    def forward_pass(self, input_values: NDArray[np.float64]) -> NDArray[np.float64]:
        self.values: List[NDArray[np.float64]] = [input_values]
        self.sums: List[NDArray[np.float64]] = []
        current_values = input_values
        for i, weight_matrix in enumerate(self.weights):
            current_values = np.insert(current_values, 0, 1.0)
            z = np.dot(weight_matrix, current_values)
            self.sums.append(z)
            if i == len(self.weights) - 1 and not self.activate_output:
                current_values = z
            else:
                current_values = self.activation_func.func(z, self.beta_func)
            self.values.append(current_values)
        return current_values

    def update_weights(self, final_output, expected_output, learning_rate=0.1, grad_output=None):
        if grad_output is not None and self.activate_output:
            grad_output *= self.activation_func.deriv(self.sums[-1], self.beta_func)
        return self.update_weights_func(final_output, expected_output, learning_rate, grad_output)

    def _gradient_descent(self, final_output, expected_output, learning_rate=0.1, grad_output = None):
        if grad_output is not None:
            deltas = grad_output
        else:
            error = expected_output - final_output
            self.data_error = np.sum(error**2)
            deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)
        for i in reversed(range(len(self.weights))):
            values = np.insert(self.values[i], 0, 1.0)
            self.weights[i] += learning_rate * np.outer(deltas, values)
            if i > 0:
                weights_wo_bias = self.weights[i][:, 1:]
                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)
        return np.dot(self.weights[0][:, 1:].T, deltas)  # ∂L/∂input


    def _momentum(self, final_output, expected_output, learning_rate=0.1, grad_output=None):
        if grad_output is not None:
            deltas = grad_output
        else:
            error = expected_output - final_output
            self.data_error = np.sum(error**2)
            deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)
        for i in reversed(range(len(self.weights))):
            values = np.insert(self.values[i], 0, 1.0)
            gradient = np.outer(deltas, values)
            delta_w = learning_rate * gradient + self.alpha * self.prev_weight_updates[i]
            self.weights[i] += delta_w
            self.prev_weight_updates[i] = delta_w
            if i > 0:
                weights_wo_bias = self.weights[i][:, 1:]
                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)
        return np.dot(self.weights[0][:, 1:].T, deltas)  # ∂L/∂input

    def _adam_step(self, final_output, expected_output, learning_rate=0.001, grad_output=None):
        self.t += 1
        if grad_output is not None:
            deltas = grad_output
        else:
            error = expected_output - final_output
            self.data_error = np.sum(error**2)
            deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)
        grads = []
        for i in reversed(range(len(self.weights))):
            values = np.insert(self.values[i], 0, 1.0)
            grad = - np.outer(deltas, values)
            grads.insert(0, grad)
            if i > 0:
                weights_wo_bias = self.weights[i][:, 1:]
                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)
        for i in range(len(self.weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return np.dot(self.weights[0][:, 1:].T, deltas)  # ∂L/∂input

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
        if dataset.shape[0] == 0 or dataset.shape[1] == 0:
            raise ValueError("Dataset must be a non-empty 2D array")

        self.dataset = dataset
        input_dim = dataset.shape[1]
        self.latent_dim = hidden_encoder_layers_to_latent_space[-1]

        encoder_layers = hidden_encoder_layers_to_latent_space.copy()
        encoder_layers[-1] = 2 * self.latent_dim  # output = mu + logvar

        self.encoder = NeuralNet(
            input_count=input_dim,
            hidden_layers=encoder_layers,
            activation_func=activation_func,
            optimizer=PerceptronOptimizer.ADAM,
            beta_func=1,
            activate_output=False
        )

        decoder_layers = hidden_encoder_layers_to_latent_space[:-1][::-1]
        decoder_layers.append(input_dim)

        self.decoder = NeuralNet(
            input_count=self.latent_dim,
            hidden_layers=decoder_layers,
            activation_func=activation_func,
            optimizer=PerceptronOptimizer.ADAM,
            beta_func=1,
        )

        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.min_error = min_error
        self.error = float("inf")
        self.current_epoch = 0

    def has_next(self) -> bool:
        return self.error > self.min_error and self.current_epoch < self.max_epochs

    def next_epoch(self):
        if not self.has_next():
            raise RuntimeError("Training already converged or max epochs reached")

        self.current_epoch += 1
        total_loss = 0.0

        for i in range(self.dataset.shape[0]):
            x = self.dataset[i]

            # --------- FORWARD ---------
            mu_logvar = self.encoder.forward_pass(x)
            mu = mu_logvar[:self.latent_dim]
            logvar = np.clip(mu_logvar[self.latent_dim:], -10, 10)

            std = np.exp(0.5 * logvar)
            eps = np.random.normal(size=std.shape)
            z = mu + std * eps

            x_hat = self.decoder.forward_pass(z)

            # --------- LOSS ------------
            recon_loss = np.sum((x - x_hat) ** 2)
            kl_div = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))
            loss = recon_loss + kl_div
            total_loss += loss

            # --------- BACKPROP DECODER ----------
            grad_z = self.decoder.update_weights(x_hat, x, self.learn_rate)

            # --------- BACKPROP ENCODER ----------
            # Derivadas por la reparametrización
            dz_dmu = 1
            dz_dlogvar = 0.5 * np.exp(0.5 * logvar) * eps

            # Derivadas del KL respecto a mu y logvar
            dkl_dmu = mu
            dkl_dlogvar = 0.5 * (np.exp(logvar) - 1)

            # Gradiente total (∂L/∂mu y ∂L/∂logvar)
            grad_mu = grad_z * dz_dmu + dkl_dmu
            grad_logvar = grad_z * dz_dlogvar + dkl_dlogvar
            grad_encoder_output = np.concatenate([grad_mu, grad_logvar])

            # Propagamos el gradiente real hacia el encoder
            self.encoder.update_weights(mu_logvar, None, self.learn_rate, grad_output=grad_encoder_output)
            # print(f"Epoch {self.current_epoch} - Avg loss: {total_loss / self.dataset.shape[0]:.6f}")
            # print(f"Grad z norm: {np.linalg.norm(grad_z):.6f}")
            # print(f"Grad encoder output norm: {np.linalg.norm(grad_encoder_output):.6f}")
            # print(f"Mu mean: {np.mean(mu):.6f} Logvar mean: {np.mean(logvar):.6f}")


        self.error = total_loss / self.dataset.shape[0]


    def try_current_epoch(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        mu_logvar = self.encoder.forward_pass(input)
        mu = mu_logvar[:self.latent_dim]
        logvar = mu_logvar[self.latent_dim:]
        std = np.exp(0.5 * logvar)
        eps = np.random.normal(size=std.shape)
        z = mu + std * eps
        return self.decoder.forward_pass(z)

    def try_testing_set(self, test_data: NDArray[np.float64]) -> List[NDArray[np.float64]]:
        return [self.try_current_epoch(row) for row in test_data]

    def get_current_latent_space(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        mu_logvar = self.encoder.forward_pass(input)
        mu = mu_logvar[:self.latent_dim]
        logvar = mu_logvar[self.latent_dim:]
        std = np.exp(0.5 * logvar)
        eps = np.random.normal(size=std.shape)
        return mu + std * eps

    def decode_from_latent_space_representation(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.decoder.forward_pass(z)
