import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm
import wandb

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
        # Numerically stable Sigmoid to avoid overflow
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
        grad = np.ones_like(x)
        grad[x <= 0] = 0
        return grad

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
    
class MinMaxScaler:
    def fit(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)

    def transform(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val  + 1e-8)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class InputLayer:
    def __init__(self, data):
        self.name = "InputLayer"
        self.data = data
        self.size = data.shape[0]
        self.output = data
        self.layer_type = "Input"
        self.val_data = None
        self.test_data = None

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
        self.linear_output = None
        self.output = None
        self.val_linear = None
        self.val_output = None
        self.test_linear = None
        self.test_output = None

    def __repr__(self):
        return f"{self.layer_type} - Units: {self.units}, Activation: {self.activation_name}"

class CrossEntropyLoss:
    def compute_loss(self, targets, predictions):
        return -np.sum(targets * np.log(predictions + 1e-8))

    def gradient(self, targets, predictions):
        return -targets / (predictions + 1e-8)

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

class BasicOptimizer:
    def __init__(self, eta=0.01):
        self.lr = eta
        self.update_val = 0

    def set_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def compute_update(self, grad):
        grad = np.clip(grad, -1, 1)
        self.update_val = self.lr * grad
        return self.update_val

optimizer_mapping = {"Basic": BasicOptimizer()}

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
        if targets_val is not None:
            self.X_val = X_val
            self.layers[0].val_data = X_val
            self.targets_val = targets_val
        self._initialize_parameters(optimizer_params)

    def _initialize_parameters(self, optimizer_params):
        prev_units = self.layers[0].size
        for layer in self.layers[1:]:
            layer.input_dim = prev_units
            weight_shape = (prev_units, layer.units)
            prev_units = layer.units
            layer.weight_optimizer = deepcopy(optimizer_mapping[self.optimizer_name])
            layer.bias_optimizer = deepcopy(optimizer_mapping[self.optimizer_name])
            if optimizer_params:
                layer.weight_optimizer.set_parameters(optimizer_params)
                layer.bias_optimizer.set_parameters(optimizer_params)
            if self.init_method == "RandomNormal":
                layer.weights, layer.biases = RandomNormalInitializer().initialize(weight_shape[0], weight_shape[1])
            else:
                layer.weights, layer.biases = XavierUniformInitializer().initialize(weight_shape[0], weight_shape[1])

    def forward_pass(self):
        for idx in range(1, len(self.layers)):
            prev_output = self.layers[idx-1].output
            self.layers[idx].linear_output = np.dot(prev_output.T, self.layers[idx].weights).T - self.layers[idx].biases
            self.layers[idx].output = self.layers[idx].activation.compute(self.layers[idx].linear_output)
        if hasattr(self.layers[0], 'val_data') and self.layers[0].val_data is not None:
            for idx in range(1, len(self.layers)):
                prev_val = self.layers[0].val_data if idx == 1 else self.layers[idx-1].val_output
                self.layers[idx].val_linear = np.dot(prev_val.T, self.layers[idx].weights).T - self.layers[idx].biases
                self.layers[idx].val_output = self.layers[idx].activation.compute(self.layers[idx].val_linear)
        if self.loss_type == "CrossEntropy":
            softmax = Softmax()
            self.layers[-1].output = softmax.compute(self.layers[-1].output)
            if self.layers[-1].val_output is not None:
                self.layers[-1].val_output = softmax.compute(self.layers[-1].val_output)

    def evaluate(self, X_test, targets_test):
        self.layers[0].test_data = X_test  # Set test_data for InputLayer
        for idx in range(1, len(self.layers)):
            # Use test_data if previous layer is InputLayer, else use test_output
            prev_test = self.layers[idx-1].test_data if idx == 1 else self.layers[idx-1].test_output
            self.layers[idx].test_linear = np.dot(prev_test.T, self.layers[idx].weights).T - self.layers[idx].biases
            self.layers[idx].test_output = self.layers[idx].activation.compute(self.layers[idx].test_linear)
        if self.loss_type == "CrossEntropy":
            softmax = Softmax()
            self.layers[-1].test_output = softmax.compute(self.layers[-1].test_output)
        test_loss = self.loss.compute_loss(targets_test, self.layers[-1].test_output)
        encoder = OneHotEncoder()
        predicted = encoder.inverse_transform(self.layers[-1].test_output)
        actual = encoder.inverse_transform(targets_test)
        accuracy = np.sum(predicted == actual)
        return accuracy, test_loss, predicted

    def backward_pass(self):
        loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        lr_history = []

        for epoch in tqdm(range(self.epochs)):
            # Run forward pass to compute outputs
            self.forward_pass()

            for batch in range(self.num_batches):
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                batch_targets = self.targets[:, start:end]
                batch_predictions = self.layers[-1].output[:, start:end]

                # Compute gradients for the output layer
                grad_out = self.loss.gradient(batch_targets, batch_predictions)
                grad_linear = grad_out

                prev_activation = self.layers[-2].output[:, start:end]
                grad_w = np.dot(prev_activation, grad_linear.T)
                grad_b = -np.sum(grad_linear, axis=1, keepdims=True)

                self.layers[-1].weights -= self.layers[-1].weight_optimizer.compute_update(grad_w)
                self.layers[-1].biases -= self.layers[-1].bias_optimizer.compute_update(grad_b)

                # Backpropagate through hidden layers
                for idx in range(len(self.layers) - 2, 0, -1):
                    grad_from_next = np.dot(self.layers[idx+1].weights, grad_linear)
                    activation_deriv = self.layers[idx].activation.derivative(self.layers[idx].linear_output[:, start:end])
                    grad_linear = grad_from_next * activation_deriv

                    prev_activation = self.layers[idx-1].output[:, start:end]
                    grad_w = np.dot(prev_activation, grad_linear.T)
                    grad_b = -np.sum(grad_linear, axis=1, keepdims=True)

                    self.layers[idx].weights -= self.layers[idx].weight_optimizer.compute_update(grad_w)
                    self.layers[idx].biases -= self.layers[idx].bias_optimizer.compute_update(grad_b)

                # Update outputs after batch update
                self.forward_pass()

            lr_history.append(self.layers[-1].weight_optimizer.lr)
            current_loss = self.loss.compute_loss(self.targets, self.layers[-1].output)
            loss_history.append(current_loss)
            train_acc, val_acc = self._compute_accuracy(validation=True)
            train_acc_history.append(train_acc)
            val_loss = self.loss.compute_loss(self.targets_val, self.layers[-1].val_output)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            if self.use_wandb and wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": current_loss / self.targets.shape[1],
                    "train_accuracy": train_acc / self.targets.shape[1],
                    "val_loss": val_loss / self.targets_val.shape[1],
                    "val_accuracy": val_acc / self.targets_val.shape[1]
                })

        return {
            "train_loss": loss_history,
            "val_loss": val_loss_history,
            "train_accuracy": train_acc_history,
            "val_accuracy": val_acc_history,
            "learning_rate": lr_history
        }



    def _compute_accuracy(self, validation=False):
        encoder = OneHotEncoder()
        train_pred = encoder.inverse_transform(self.layers[-1].output)
        train_actual = encoder.inverse_transform(self.targets)
        train_acc = np.sum(train_pred == train_actual)
        if validation:
            val_pred = encoder.inverse_transform(self.layers[-1].val_output)
            val_actual = encoder.inverse_transform(self.targets_val)
            val_acc = np.sum(val_pred == val_actual)
            return train_acc, val_acc
        return train_acc