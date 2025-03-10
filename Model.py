import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm
import wandb

# Activation functions and their derivatives

class Softmax:
    def compute(self, x):
        x_stable = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def derivative(self, x):
        s = self.compute(x)
        return s * (1 - s)

class Sigmoid:
    def compute(self, x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def derivative(self, x):
        sig = self.compute(x)
        return sig * (1 - sig)
    
class Tanh:
    def compute(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1.0 - np.tanh(x) ** 2

class ReLU:
    def compute(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        grad = np.ones(x.shape)
        grad[x <= 0] = 0
        return grad

# Initializers

class RandomNormalInitializer:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def initialize(self, n_in, n_out):
        weights = np.random.normal(self.mean, self.std, (n_in, n_out))
        biases = np.random.normal(self.mean, self.std, (n_out, 1))
        return weights, biases
    
class XavierUniformInitializer:
    def initialize(self, n_in, n_out):
        limit = np.sqrt(6.0 / (n_in + n_out))
        weights = np.random.uniform(low=-limit, high=limit, size=(n_in, n_out))
        biases = np.zeros((n_out, 1))
        return weights, biases

class HeNormalInitializer:
    def initialize(self, n_in, n_out):
        std = np.sqrt(2.0 / n_in)
        weights = np.random.normal(0, std, (n_in, n_out))
        biases = np.zeros((n_out, 1))
        return weights, biases

# Data scaler

class MinMaxScaler:
    def fit(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)

    def transform(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val + 1e-8)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Layers

class InputLayer:
    def __init__(self, data):
        self.name = "InputLayer"
        self.data = data
        self.size = data.shape[0]
        self.layer_type = "Input"

    def __repr__(self):
        return f"{self.layer_type} - Size: {self.size}"

class DenseLayer:
    def __init__(self, units, activation, name, final=False):
        self.name = name
        self.units = units
        self.activation = activation
        self.activation_name = type(activation).__name__
        self.layer_type = "DenseLayer"
        self.weights = None
        self.biases = None
        self.input_dim = None
        
    def __repr__(self):
        return f"{self.layer_type} - Units: {self.units}, Activation: {self.activation_name}"

# Loss and Encoder

class CrossEntropyLoss:
    def compute_loss(self, targets, predictions):
        return -np.sum(targets * np.log(predictions + 1e-8))

    def gradient(self, targets, predictions):
        return predictions - targets

class OneHotEncoder:
    def fit(self, labels, num_classes):
        self.labels = labels
        self.num_classes = num_classes

    def transform(self, labels):
        onehot = np.zeros((self.num_classes, labels.size))
        for idx, label in enumerate(labels):
            onehot[label, idx] = 1
        return onehot
    
    def fit_transform(self, labels, num_classes):
        self.fit(labels, num_classes)
        return self.transform(labels)
    
    def inverse_transform(self, onehot):
        return np.argmax(onehot, axis=0)

# Optimizers

class BasicOptimizer:
    def __init__(self, eta=0.01):
        self.lr = eta
        self.update_val = 0

    def set_parameters(self, params):
        if "learning_rate" in params:
            self.lr = params["learning_rate"]

    def compute_update(self, grad):
        grad = np.clip(grad, -1, 1)
        self.update_val = self.lr * grad
        return self.update_val
    
class MomentumOptimizer:
    def __init__(self, eta=0.01, momentum=0.9):
        self.lr = eta
        self.momentum = momentum
        self.velocity = 0

    def set_parameters(self, params):
        if "learning_rate" in params:
            self.lr = params["learning_rate"]
        if "momentum" in params:
            self.momentum = params["momentum"]

    def compute_update(self, grad):
        self.velocity = self.momentum * self.velocity + self.lr * grad
        return self.velocity

class NesterovOptimizer:
    def __init__(self, eta=1e-3, momentum=0.9):
        self.lr = eta
        self.momentum = momentum
        self.velocity = 0

    def set_parameters(self, params):
        if "learning_rate" in params:
            self.lr = params["learning_rate"]
        if "momentum" in params:
            self.momentum = params["momentum"]

    def compute_update(self, grad):
        prev_velocity = self.velocity
        self.velocity = self.momentum * self.velocity - self.lr * grad
        update = -self.momentum * prev_velocity + (1 + self.momentum) * self.velocity
        return update

class RMSPropOptimizer:
    def __init__(self, beta=0.9, eta=1e-3, epsilon=1e-7):
        self.beta = beta
        self.lr = eta
        self.epsilon = epsilon
        self.cache = 0

    def set_parameters(self, params):
        if "learning_rate" in params:
            self.lr = params["learning_rate"]
        if "beta" in params:
            self.beta = params["beta"]
        if "epsilon" in params:
            self.epsilon = params["epsilon"]

    def compute_update(self, grad):
        self.cache = self.beta * self.cache + (1 - self.beta) * (grad**2)
        return (self.lr / np.sqrt(self.cache + self.epsilon)) * grad

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, lr=1e-2, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.iteration = 1

    def set_parameters(self, params):
        if "learning_rate" in params:
            self.lr = params["learning_rate"]
        if "beta1" in params:
            self.beta1 = params["beta1"]
        if "beta2" in params:
            self.beta2 = params["beta2"]
        if "epsilon" in params:
            self.epsilon = params["epsilon"]

    def compute_update(self, grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.iteration)
        v_hat = self.v / (1 - self.beta2**self.iteration)
        self.iteration += 1
        return (self.lr / np.sqrt(v_hat + self.epsilon)) * m_hat

class NadamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, lr=1e-3, epsilon=1e-7):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.iteration = 1

    def set_parameters(self, params):
        if "learning_rate" in params:
            self.lr = params["learning_rate"]
        if "beta1" in params:
            self.beta1 = params["beta1"]
        if "beta2" in params:
            self.beta2 = params["beta2"]
        if "epsilon" in params:
            self.epsilon = params["epsilon"]

    def compute_update(self, grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.iteration)
        v_hat = self.v / (1 - self.beta2**self.iteration)
        update_term = self.beta1 * m_hat + (1 - self.beta1 / (1 - self.beta1**self.iteration)) * grad
        self.iteration += 1
        return (self.lr / np.sqrt(v_hat + self.epsilon)) * update_term

# Mapping optimizer names to instances
optimizer_mapping = {
    "SGD": BasicOptimizer(),
    "Momentum": MomentumOptimizer(),
    "Nesterov": NesterovOptimizer(),
    "RMSProp": RMSPropOptimizer(),
    "Adam": AdamOptimizer(),
    "Nadam": NadamOptimizer()
}

# Neural Network Class

class NeuralNet:
    def __init__(self, layers, batch_size, optimizer_name, init_method, epochs, targets, loss_type,
                 X_val=None, targets_val=None, use_wandb=False, optimizer_params=None):
        self.layers = layers
        self.batch_size = batch_size
        self.init_method = init_method
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.targets = targets
        self.num_batches = math.ceil(targets.shape[1] / batch_size)
        self.loss_type = loss_type
        self.use_wandb = use_wandb
        self.loss = CrossEntropyLoss()
        if X_val is not None and targets_val is not None:
            self.X_val = X_val
            self.targets_val = targets_val
        else:
            self.X_val = None
            self.targets_val = None
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self._initialize_parameters(optimizer_params)

    def _initialize_parameters(self, optimizer_params):
        prev_units = self.layers[0].size
        for layer in self.layers[1:]:
            layer.input_dim = prev_units
            prev_units = layer.units
            layer.weight_optimizer = deepcopy(optimizer_mapping[self.optimizer_name])
            layer.bias_optimizer = deepcopy(optimizer_mapping[self.optimizer_name])
            if optimizer_params:
                layer.weight_optimizer.set_parameters(optimizer_params)
                layer.bias_optimizer.set_parameters(optimizer_params)
            if isinstance(layer.activation, ReLU):
                layer.weights, layer.biases = HeNormalInitializer().initialize(layer.input_dim, layer.units)
            else:
                if self.init_method == "Random":
                    layer.weights, layer.biases = RandomNormalInitializer().initialize(layer.input_dim, layer.units)
                else:
                    layer.weights, layer.biases = XavierUniformInitializer().initialize(layer.input_dim, layer.units)

    def forward_batch(self, X):
        activations = [X]
        linear_outputs = [None]
        for layer in self.layers[1:]:
            z = np.dot(layer.weights.T, activations[-1]) + layer.biases
            linear_outputs.append(z)
            a = layer.activation.compute(z)
            activations.append(a)
        if self.loss_type == "CrossEntropy":
            softmax = Softmax()
            activations[-1] = softmax.compute(activations[-1])
        return activations, linear_outputs

    def backward_pass(self):
        loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        lr_history = []

        for epoch in tqdm(range(self.epochs)):
            perm = np.random.permutation(self.targets.shape[1])
            X_train = self.layers[0].data[:, perm]
            Y_train = self.targets[:, perm]
            num_batches = math.ceil(Y_train.shape[1] / self.batch_size)
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                X_batch = X_train[:, start:end]
                Y_batch = Y_train[:, start:end]

                activations, linear_outputs = self.forward_batch(X_batch)
                grad_out = self.loss.gradient(Y_batch, activations[-1])
                grad_linear = grad_out

                # Apply weight decay to gradients if specified
                weight_decay = self.optimizer_params.get("weight_decay", 0)
                grad_w = np.dot(activations[-2], grad_linear.T) + weight_decay * self.layers[-1].weights
                grad_b = np.sum(grad_linear, axis=1, keepdims=True)
                update = self.layers[-1].weight_optimizer.compute_update(grad_w)
                self.layers[-1].weights -= update
                self.layers[-1].biases -= self.layers[-1].bias_optimizer.compute_update(grad_b)

                for idx in range(len(self.layers) - 2, 0, -1):
                    grad_from_next = np.dot(self.layers[idx+1].weights, grad_linear)
                    activation_deriv = self.layers[idx].activation.derivative(linear_outputs[idx])
                    grad_linear = grad_from_next * activation_deriv
                    grad_w = np.dot(activations[idx-1], grad_linear.T) + weight_decay * self.layers[idx].weights
                    grad_b = np.sum(grad_linear, axis=1, keepdims=True)
                    self.layers[idx].weights -= self.layers[idx].weight_optimizer.compute_update(grad_w)
                    self.layers[idx].biases -= self.layers[idx].bias_optimizer.compute_update(grad_b)

            train_acts, _ = self.forward_batch(self.layers[0].data)
            current_loss = self.loss.compute_loss(self.targets, train_acts[-1])
            loss_history.append(current_loss)
            encoder = OneHotEncoder()
            train_pred = encoder.inverse_transform(train_acts[-1])
            train_actual = encoder.inverse_transform(self.targets)
            train_acc = np.sum(train_pred == train_actual)
            train_acc_history.append(train_acc)

            if self.X_val is not None and self.targets_val is not None:
                val_acts, _ = self.forward_batch(self.X_val)
                val_loss = self.loss.compute_loss(self.targets_val, val_acts[-1])
                val_loss_history.append(val_loss)
                val_pred = encoder.inverse_transform(val_acts[-1])
                val_actual = encoder.inverse_transform(self.targets_val)
                val_acc = np.sum(val_pred == val_actual)
                val_acc_history.append(val_acc)
            else:
                val_loss_history.append(None)
                val_acc_history.append(None)

            lr_history.append(self.layers[-1].weight_optimizer.lr)

            if self.use_wandb and wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": current_loss / self.targets.shape[1],
                    "train_accuracy": train_acc / self.targets.shape[1],
                    "val_loss": (val_loss / self.targets_val.shape[1]) if self.targets_val is not None else None,
                    "val_accuracy": (val_acc / self.targets_val.shape[1]) if self.targets_val is not None else None
                })

        return {
            "train_loss": loss_history,
            "val_loss": val_loss_history,
            "train_accuracy": train_acc_history,
            "val_accuracy": val_acc_history,
            "learning_rate": lr_history
        }

    def evaluate(self, X_test, targets_test):
        activations, _ = self.forward_batch(X_test)
        test_loss = self.loss.compute_loss(targets_test, activations[-1])
        encoder = OneHotEncoder()
        predicted = encoder.inverse_transform(activations[-1])
        actual = encoder.inverse_transform(targets_test)
        accuracy = np.sum(predicted == actual)
        return accuracy, test_loss, predicted
