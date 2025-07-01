import copy

import numpy as np

from src.activation_function import ActivationFunction
from src.perceptron_optimizer import SGD, Optimizer


class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim, hidden_layers, activation_func, optimizer: Optimizer):
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

    def reparametrize(self, mu, log_var):
        std = np.exp(0.5 * log_var)
        eps = np.random.normal(size=std.shape)
        return mu + std * eps

    def forward(self, x):
        encoder_activations = self.encoder.forward(x)
        encoded = encoder_activations[-1]

        limit = self.latent_dim
        mu = encoded[:, :limit]
        log_var = np.clip(encoded[:, limit:], -10, 10)
        z = self.reparametrize(mu, log_var)
        decoder_activations = self.decoder.forward(z)

        return mu, log_var, z, encoder_activations, decoder_activations

    def binary_cross_entropy(self, x, x_hat):
        return -np.mean(x * np.log(x_hat + 1e-8) + (1 - x) * np.log(1 - x_hat + 1e-8))

    def kl_divergence(self, mu, log_var):
        return -0.5 * np.mean(np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1))

    def loss_fn(self, x, x_hat, mu, log_var, beta=1.0):
        bce = self.binary_cross_entropy(x, x_hat)
        kl = self.kl_divergence(mu, log_var)
        return bce + beta * kl

    def train(self, x, epochs=1000, batch_size=None, beta=0.001):
        x = np.array([sample.flatten() for sample in x])
        n_samples = x.shape[0]
        batch_size = batch_size or n_samples

        loss_history = []
        samples_processed = 0

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            total_loss = 0

            # KL Annealing
            current_beta = beta * min(1.0, (epoch + 1) / (epochs * 0.5))

            for i in range(0, n_samples, batch_size):
                left = i
                right = left + batch_size
                batch_idx = idx[left:right]
                batch_x = x[batch_idx]

                mu, log_var, z, encoder_activations, decoder_activations = self.forward(batch_x)
                encoded = encoder_activations[-1]
                reconstructed = decoder_activations[-1]

                loss = self.loss_fn(batch_x, reconstructed, mu, log_var, beta=current_beta)
                total_loss += loss

                samples_processed += len(batch_x)
                loss_history.append((samples_processed, loss))

                grad_z = self.decoder.backward(batch_x, decoder_activations)

                grad_mu_bce = grad_z
                grad_log_var_bce = grad_z * 0.5 * (z - mu)

                grad_mu_kl = mu
                grad_log_var_kl = 0.5 * (np.exp(log_var) - 1)

                grad_mu = grad_mu_bce + current_beta * grad_mu_kl
                grad_log_var = grad_log_var_bce + current_beta * grad_log_var_kl
                encoder_grad = np.concatenate([grad_mu, grad_log_var], axis=1)

                encoder_target = encoded - encoder_grad
                self.encoder.backward(encoder_target, encoder_activations)

            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / (n_samples / batch_size)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

        return loss_history

    def decode(self, z):
        decoder_activations = self.decoder.forward(z)
        return decoder_activations[-1]


class Autoencoder:
    def __init__(self, layers, activation_func: ActivationFunction, optimizer: Optimizer, activate_output=True):
        self.layers = layers
        self.func = activation_func
        self.optimizer = optimizer
        self.weights = []
        self.biases = []
        self.activate_output = activate_output

        # Inicialización Xavier/Glorot
        for i in range(len(layers) - 1):
            neurons = layers[i + 1]
            inputs = layers[i] + 1  # +1 for bias
            scale = np.sqrt(2.0 / (inputs + neurons))
            self.weights.append(np.random.normal(0, scale, (neurons, inputs)))

        self.optimizer.initialize(self.weights)

    def forward(self, x):
        input = np.array(x)
        activations = [input]

        for i, weight_matrix in enumerate(self.weights):
            bias = np.ones((input.shape[0], 1))
            input_with_bias = np.concatenate((bias, input), axis=1)
            h = np.dot(input_with_bias, weight_matrix.T)
            if i == len(self.weights) - 1 and not self.activate_output:
                output = h
            else:
                output = np.array([self.func.func(h_i, 1) for h_i in h])
            activations.append(output)
            input = output

        return activations

    def backward(self, x, activations):
        deltas = [None] * len(self.weights)
        output = activations[-1]
        deltas[-1] = output - np.array(x)

        # Delta de la capa de salida
        if self.activate_output:
            deltas[-1] *= np.array([self.func.deriv_from_out(o, 1) for o in output])

        # Delta de las capas ocultas
        for i in reversed(range(len(deltas) - 1)):
            j = i + 1
            layer_output = activations[j]
            next_delta = deltas[j]
            next_weights = self.weights[j]

            # Calcular delta evitando bias
            hidden_error = np.dot(next_delta, next_weights[:, 1:]) # type: ignore
            deltas[i] = hidden_error * np.array([self.func.deriv_from_out(o, 1) for o in layer_output])

        grad_wrt_input = np.dot(deltas[0], self.weights[0][:, 1:]) # type: ignore

        # Actualizar pesos
        for i in range(len(self.weights)):
            batch_size = x.shape[0]
            weight_gradients = np.zeros_like(self.weights[i])

            for b in range(batch_size):
                layer_input = np.concatenate(([1], activations[i][b]))
                weight_gradients += np.outer(deltas[i][b], layer_input) # type: ignore

            weight_gradients /= batch_size
            self.optimizer.update(i, self.weights[i], weight_gradients)

        return grad_wrt_input
