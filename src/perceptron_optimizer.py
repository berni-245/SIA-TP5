from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def initialize(self, weights):
        pass

    @abstractmethod
    def update_weights(self, i, weights, weight_gradients):
        pass

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, layers=None):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.simple = False
        self.m = None
        self.v = None

        if layers is not None and len(layers) <= 3 and max(layers) <= 10:
            self.beta1 = 0.8
            self.beta2 = 0.9
            self.epsilon = 1e-6
            self.learning_rate *= 0.1
            self.simple = True

    def initialize(self, weights):
        self.m = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.v = [np.zeros_like(w, dtype=np.float32) for w in weights]

    def update_weights(self, i, weights, weight_gradients):
        self.t += 1

        np.multiply(self.v[i], self.beta2, out=self.v[i]) # type: ignore
        np.add(self.v[i], (1 - self.beta2) * np.square(weight_gradients), out=self.v[i]) # type: ignore

        np.multiply(self.m[i], self.beta1, out=self.m[i]) # type: ignore
        np.add(self.m[i], (1 - self.beta1) * weight_gradients, out=self.m[i]) # type: ignore

        v_hat = self.v[i] / (1 - self.beta2**self.t) # type: ignore
        m_hat = self.m[i] / (1 - self.beta1**self.t) # type: ignore
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        if self.simple:
            update = np.clip(update, -0.1, 0.1)

        np.subtract(weights, update, out=weights)

class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def initialize(self, weights):
        pass

    def update_weights(self, i, weights, weight_gradients):
        np.subtract(weights, self.learning_rate * weight_gradients, out=weights)


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def initialize(self, weights):
        self.velocity = [np.zeros_like(w, dtype=np.float32) for w in weights]

    def update_weights(self, i, weights, weight_gradients):
        np.multiply(self.velocity[i], self.momentum, out=self.velocity[i]) # type: ignore
        np.add(self.velocity[i], self.learning_rate * weight_gradients, out=self.velocity[i]) # type: ignore
        np.subtract(weights, self.velocity[i], out=weights) # type: ignore

def get_optimizer(name, **kwargs):
    if name == "gradient":
        return GradientDescent(**kwargs)
    elif name == "momentum":
        return Momentum(**kwargs)
    elif name == "adam":
        return Adam(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")