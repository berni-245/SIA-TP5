import copy

import numpy as np

from src.activation_function import ActivationFunction
from src.perceptron_optimizer import SGD, Optimizer


class VariationalAutoencoder:
    def __init__(self, dataset, input_dim, latent_dim, hidden_layers, activation_func, optimizer: Optimizer,
                 epochs=1000, batch_size=None, beta=0.001, min_error=0.1):
        # Modelos
        self.latent_dim = latent_dim
        self.encoder = Autoencoder(
            layers=[input_dim] + hidden_layers + [latent_dim * 2],
            activation_func=activation_func,
            optimizer=optimizer,
            activate_output=False,
        )
        self.decoder = Autoencoder(
            layers=[latent_dim] + hidden_layers[::-1] + [input_dim],
            activation_func=activation_func,
            optimizer=copy.deepcopy(optimizer),
        )

        self.dataset = np.array([sample.flatten() for sample in dataset])
        self.n_samples = self.dataset.shape[0]
        self.batch_size = batch_size or self.n_samples
        self.max_epochs = epochs
        self.beta = beta
        self.min_error = min_error
        
        self.epoch = 0
        self.loss_history = []

    def feed_forward(self, x):
        encoder_activations = self.encoder.forward(x)
        encoded = encoder_activations[-1]

        limit = self.latent_dim
        mu = encoded[:, :limit]
        log_var = np.clip(encoded[:, limit:], -10, 10)
        z = self.reparametrize(mu, log_var)
        decoder_activations = self.decoder.forward(z)

        return mu, log_var, z, encoder_activations, decoder_activations

    def reparametrize(self, mu, log_var):
        std = np.exp(0.5 * log_var)
        eps = np.random.normal(size=std.shape)
        return mu + std * eps

    def loss_fn(self, x, x_hat, mu, log_var, beta=1.0):
        bce = -np.mean(x * np.log(x_hat + 1e-8) + (1 - x) * np.log(1 - x_hat + 1e-8))
        kl = -0.5 * np.mean(np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1))
        return bce + beta * kl

    def has_next(self):
        if self.epoch >= self.max_epochs:
            return False
        if self.loss_history and self.loss_history[-1][1] <= self.min_error:
            return False
        return True

    def next_epoch(self):
        idx = np.random.permutation(self.n_samples)
        total_loss = 0
        current_beta = self.beta * min(1.0, (self.epoch + 1) / (self.max_epochs * 0.5))

        for i in range(0, self.n_samples, self.batch_size):
            left = i
            right = left + self.batch_size
            batch_idx = idx[left:right]
            batch_x = self.dataset[batch_idx]

            mu, log_var, z, encoder_activations, decoder_activations = self.feed_forward(batch_x)
            encoded = encoder_activations[-1]
            reconstructed = decoder_activations[-1]

            loss = self.loss_fn(batch_x, reconstructed, mu, log_var, beta=current_beta)
            total_loss += loss * len(batch_x)

            grad_z = self.decoder.back_propagate(batch_x, decoder_activations)

            grad_mu_bce = grad_z
            grad_log_var_bce = grad_z * 0.5 * (z - mu)

            grad_mu_kl = mu
            grad_log_var_kl = 0.5 * (np.exp(log_var) - 1)

            grad_mu = grad_mu_bce + current_beta * grad_mu_kl
            grad_log_var = grad_log_var_bce + current_beta * grad_log_var_kl
            encoder_grad = np.concatenate([grad_mu, grad_log_var], axis=1)

            encoder_target = encoded - encoder_grad
            self.encoder.back_propagate(encoder_target, encoder_activations)

        avg_loss = total_loss / self.n_samples
        self.epoch += 1
        self.loss_history.append((self.epoch, avg_loss))
        return avg_loss


    def decode(self, z):
        decoder_activations = self.decoder.forward(z)
        return decoder_activations[-1]


class Autoencoder:
    def __init__(self, layers, activation_func: ActivationFunction, optimizer: Optimizer, activate_output=True):
        self.layers = layers
        self.func = activation_func
        self.optimizer = optimizer
        self.weights = []
        self.activate_output = activate_output

        for i in range(len(layers) - 1):
            neurons = layers[i + 1]
            inputs = layers[i] + 1  # +1 for bias
            scale = np.sqrt(2.0 / (inputs + neurons))
            self.weights.append(np.random.normal(0, scale, (neurons, inputs)))

        self.optimizer.initialize(self.weights)

    def add_bias(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        bias = np.ones((x.shape[0], 1))
        return np.concatenate((bias, x), axis=1)

    def forward(self, x):
        input = np.array(x)
        if input.ndim == 1:
            input = input[np.newaxis, :]
        activations = [input]

        for i, weight_matrix in enumerate(self.weights):
            input_with_bias = self.add_bias(input)
            h = input_with_bias @ weight_matrix.T  
            if i == len(self.weights) - 1 and not self.activate_output:
                output = h
            else:
                output = self.func.func(h, 1)
            activations.append(output)
            input = output

        return activations

    def back_propagate(self, x, activations):
        x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        batch_size = x.shape[0]
        deltas = [None] * len(self.weights)
        output = activations[-1]
        deltas[-1] = output - x

        if self.activate_output:
            derivs = self.func.deriv_from_out(output, 1)  # Vectorizado: deriv_from_out debe aceptar arrays
            deltas[-1] *= derivs

        for i in reversed(range(len(deltas) - 1)):
            j = i + 1
            layer_output = activations[j]
            next_delta = deltas[j]
            next_weights = self.weights[j]

            hidden_error = next_delta @ next_weights[:, 1:]  # ignorar bias weights
            derivs = self.func.deriv_from_out(layer_output, 1)
            deltas[i] = hidden_error * derivs

        grad_wrt_input = deltas[0] @ self.weights[0][:, 1:]

        # Actualizar pesos con cálculo vectorizado
        for i in range(len(self.weights)):
            layer_input = self.add_bias(activations[i])  # (batch_size, n_features + 1)
            weight_gradients = deltas[i].T @ layer_input  # type: ignore
            weight_gradients /= batch_size

            self.optimizer.update(i, self.weights[i], weight_gradients)

        return grad_wrt_input
