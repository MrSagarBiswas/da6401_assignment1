import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm
import wandb

class Softmax:
    def compute(self, x):
        x_stable = x - np.max(x, axis=0, keepdims=True) #To avoid overflow
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def derivative(self, x):
        s = self.compute(x)
        jacobian = np.diagflat(s) - np.outer(s, s)
        return jacobian

class RandomNormalInitializer:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def initialize(self, n_in, n_out):
        weights = np.random.normal(loc=self.mean, scale=self.std, size=(n_in, n_out))
        biases = np.random.normal(loc=self.mean, scale=self.std, size=(n_out, 1))
        return weights, biases

class InputLayer:
    def __init__(self, data):
        self.name = "InputLayer"
        self.data = data
        self.size = data.shape[0]
        self.output = data
        self.layer_type = "Input"
        # Optional attributes for validation and test data:
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
        # These attributes will be set during network initialization:
        self.weights = None
        self.biases = None
        self.input_dim = None
        self.linear_output = None
        self.output = None
        # For validation and testing forward passes:
        self.val_linear = None
        self.val_output = None
        self.test_linear = None
        self.test_output = None

    def __repr__(self):
        return f"{self.layer_type} - Units: {self.units}, Activation: {self.activation_name}"

class CrossEntropyLoss:
    def compute_loss(self, targets, predictions):
        # Add a tiny constant to avoid log(0)
        loss = -np.sum(targets * np.log(predictions + 1e-8))
        return loss

    def gradient(self, targets, predictions):
        return -targets / (predictions + 1e-8)

class OneHotEncoder:
    def fit(self, labels, num_classes):
        self.labels = labels
        self.num_classes = num_classes

    def transform(self):
        onehot = np.zeros((self.num_classes, self.labels.size))
        for idx, label in enumerate(self.labels):
            onehot[label, idx] = 1
        return onehot

    def fit_transform(self, labels, num_classes):
        self.fit(labels, num_classes)
        return self.transform()

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
        self.update_val = self.lr * grad
        return self.update_val

optimizer_mapping = {
    "Basic": BasicOptimizer()
}

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
            # Attach fresh copies of the chosen optimizer for weights and biases
            layer.weight_optimizer = deepcopy(optimizer_mapping[self.optimizer_name])
            layer.bias_optimizer = deepcopy(optimizer_mapping[self.optimizer_name])
            if optimizer_params:
                layer.weight_optimizer.set_parameters(optimizer_params)
                layer.bias_optimizer.set_parameters(optimizer_params)
            # Initialize weights and biases
            if self.init_method == "XavierUniform":
                init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
                layer.weights = np.array(init(shape=weight_shape))
                layer.biases = np.zeros((layer.units, 1))
            else:
                layer.weights, layer.biases = RandomNormalInitializer().initialize(prev_units, layer.units)

    def forward_pass(self):
        # Forward pass for training data
        for idx in range(1, len(self.layers)):
            prev_output = self.layers[idx-1].output
            # Compute linear transformation: (W^T * input - bias)
            self.layers[idx].linear_output = np.dot(prev_output.T, self.layers[idx].weights).T - self.layers[idx].biases
            self.layers[idx].output = self.layers[idx].activation.compute(self.layers[idx].linear_output)
        # Compute forward pass for validation data if available
        if hasattr(self.layers[0], 'val_data') and self.layers[0].val_data is not None:
            for idx in range(1, len(self.layers)):
                if idx == 1:
                    prev_val = self.layers[0].val_data
                else:
                    prev_val = self.layers[idx-1].val_output
                self.layers[idx].val_linear = np.dot(prev_val.T, self.layers[idx].weights).T - self.layers[idx].biases
                self.layers[idx].val_output = self.layers[idx].activation.compute(self.layers[idx].val_linear)
        # If using cross-entropy, apply softmax to final outputs
        if self.loss_type == "CrossEntropy":
            softmax = Softmax()
            self.layers[-1].output = softmax.compute(self.layers[-1].output)
            if hasattr(self.layers[-1], 'val_output') and self.layers[-1].val_output is not None:
                self.layers[-1].val_output = softmax.compute(self.layers[-1].val_output)

    def evaluate(self, X_test, targets_test):
        self.layers[0].test_data = X_test
        for idx in range(1, len(self.layers)):
            self.layers[idx].test_linear = np.dot(self.layers[idx-1].test_data.T, self.layers[idx].weights).T - self.layers[idx].biases
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

            # Process mini-batches
            for batch in range(self.num_batches):
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                batch_targets = self.targets[:, start:end]
                batch_predictions = self.layers[-1].output[:, start:end]

                # Compute gradient at the output layer
                grad_out = self.loss.gradient(batch_targets, batch_predictions)
                grad_linear = grad_out * self.layers[-1].activation.derivative(self.layers[-1].linear_output[:, start:end])
                
                # Compute gradients for weights and biases (for the final layer)
                prev_activation = self.layers[-2].output[:, start:end]
                grad_w = np.dot(grad_linear, prev_activation.T)
                grad_b = -np.sum(grad_linear, axis=1, keepdims=True)

                # Update final layer parameters
                update_w = self.layers[-1].weight_optimizer.compute_update(grad_w)
                update_b = self.layers[-1].bias_optimizer.compute_update(grad_b)
                self.layers[-1].weights -= update_w
                self.layers[-1].biases -= update_b

                # Backpropagate through hidden layers
                for idx in range(len(self.layers)-2, 0, -1):
                    grad_from_next = np.dot(self.layers[idx+1].weights.T, grad_linear)
                    grad_linear = grad_from_next * self.layers[idx].activation.derivative(self.layers[idx].linear_output[:, start:end])
                    prev_activation = self.layers[idx-1].output[:, start:end]
                    grad_w = np.dot(grad_linear, prev_activation.T)
                    grad_b = -np.sum(grad_linear, axis=1, keepdims=True)
                    update_w = self.layers[idx].weight_optimizer.compute_update(grad_w)
                    update_b = self.layers[idx].bias_optimizer.compute_update(grad_b)
                    self.layers[idx].weights -= update_w
                    self.layers[idx].biases -= update_b

                # Refresh the forward pass after updating parameters
                self.forward_pass()

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
