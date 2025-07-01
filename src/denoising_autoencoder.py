import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

from src.activation_function import ActivationFunction
from src.basic_autoencoder import BasicAutoEncoder
from src.utils import add_salt_and_pepper_noise

class DenoisingAutoencoder(BasicAutoEncoder):
    def __init__(
        self,
        dataset: NDArray[np.float64],
        hidden_encoder_layers_to_latent_space: List[int],
        activation_func: ActivationFunction,
        learn_rate: float = 0.1,
        min_error: float = 0.1,
        max_epochs: int = 10000,
        noise_level: Tuple[float, float] = (0.1, 0.3)
    ):
        """
        noise_level: min and max noise levels (percentage). During training a random noise_level in this
                     interval will be chosen.
        """
        super().__init__(
            dataset,
            hidden_encoder_layers_to_latent_space,
            activation_func,
            learn_rate,
            min_error,
            max_epochs
        )
        if not 0 <= noise_level[0] <= 1 or not 0 <= noise_level[1] <= 1:
            raise ValueError("Noise levels must be between 0 and 1")
        self.noise_level = noise_level

    def next_epoch(self):
        if not self.has_next():
            raise Exception("Solution was already found or max epochs were reached")

        self.error = 0
        self.current_epoch += 1

        for i in range(self.dataset.shape[0]):
            input = self.dataset[i]
            noise_level = np.random.uniform(self.noise_level[0], self.noise_level[1])
            noisy_input = add_salt_and_pepper_noise(input, noise_level)
            self.neural_net.update_weights(noisy_input, input, self.learn_rate)
            self.error += self.neural_net.data_error

        self.error /= self.dataset.shape[0]
