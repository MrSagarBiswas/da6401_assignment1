import numpy as np
import math
import wandb

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_act(x):
    return (2 / (1 + np.exp(-2 * x))) - 1

def relu_act(x):
    return np.where(x >= 0, x, 0)

def softmax_fn(x):
    exp_vals = np.exp(x)
    return exp_vals / np.sum(exp_vals, axis=0)

def print_and_record(epoch, train_loss, valid_loss, train_acc, valid_acc):
    print(f"Epoch {epoch + 1}: train_loss = {train_loss:.2f}, valid_loss = {valid_loss:.2f}, train_accuracy = {train_acc:.2f}, val_accuracy = {valid_acc:.2f}")
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': valid_loss,
        'val_accuracy': valid_acc
    })

class NeuralNetwork:
    def __init__(self, input_size=784, num_classes=10, num_hidden=1, hidden_units=4,
                 init_method="Random", activation="Sigmoid", loss_fn="cross_entropy",
                 epochs=1, batch_size=4, optimizer="sgd", lr=0.1, momentum=0.9,
                 beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.005):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.total_layers = num_hidden + 1
        self.hidden_units = hidden_units
        self.init_method = init_method
        self.activation = activation
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.weights = [None] * (self.total_layers + 1)
        self.biases = [None] * (self.total_layers + 1)
        self.pre_activations = [None] * (self.total_layers + 1)
        self.activations = [None] * (self.total_layers + 1)
        self.grad_pre = [None] * (self.total_layers + 1)
        self.grad_biases = [None] * (self.total_layers + 1)
        self.grad_weights = [None] * (self.total_layers + 1)
        self.velocity_w = [None] * (self.total_layers + 1)
        self.velocity_b = [None] * (self.total_layers + 1)
        self.lookahead_w = [None] * (self.total_layers + 1)
        self.lookahead_b = [None] * (self.total_layers + 1)
        self.rms_w = [None] * (self.total_layers + 1)
        self.rms_b = [None] * (self.total_layers + 1)
        self.adam_mw = [None] * (self.total_layers + 1)
        self.adam_mb = [None] * (self.total_layers + 1)
        self.adam_vw = [None] * (self.total_layers + 1)
        self.adam_vb = [None] * (self.total_layers + 1)

        self._init_params()

    def _init_params(self):
        if self.num_hidden == 0:
            if self.activation == "ReLU":
                self.weights[1] = np.random.randn(self.num_classes, self.input_size) * np.sqrt(2.0 / self.input_size)
            elif self.init_method == "Random":
                self.weights[1] = np.random.randn(self.num_classes, self.input_size)
            elif self.init_method == "Xavier":
                self.weights[1] = np.random.randn(self.num_classes, self.input_size) * np.sqrt(1.0 / self.input_size)
            else:
                raise ValueError("Unknown initialization method.")
            self.biases[1] = np.zeros((self.num_classes, 1))
            return

        if self.init_method == "Random":
            self.weights[1] = np.random.randn(self.hidden_units, self.input_size)
            for layer in range(2, self.total_layers):
                self.weights[layer] = np.random.randn(self.hidden_units, self.hidden_units)
            self.weights[self.total_layers] = np.random.randn(self.num_classes, self.hidden_units)
        elif self.init_method == "Xavier":
            self.weights[1] = np.random.randn(self.hidden_units, self.input_size) * np.sqrt(1.0 / self.input_size)
            for layer in range(2, self.total_layers):
                self.weights[layer] = np.random.randn(self.hidden_units, self.hidden_units) * np.sqrt(1.0 / self.hidden_units)
            self.weights[self.total_layers] = np.random.randn(self.num_classes, self.hidden_units) * np.sqrt(1.0 / self.hidden_units)
        else:
            raise ValueError("Unknown initialization method.")

        for layer in range(1, self.total_layers):
            self.biases[layer] = np.zeros((self.hidden_units, 1))
        self.biases[self.total_layers] = np.zeros((self.num_classes, 1))

    def forward_pass(self, X):
        self.activations[0] = X
        for layer in range(1, self.total_layers):
            self.pre_activations[layer] = self.biases[layer] + np.dot(self.weights[layer], self.activations[layer - 1])
            if self.activation == "Sigmoid":
                self.activations[layer] = sigmoid(self.pre_activations[layer])
            elif self.activation == "Tanh":
                self.activations[layer] = tanh_act(self.pre_activations[layer])
            elif self.activation == "ReLU":
                self.activations[layer] = relu_act(self.pre_activations[layer])
        self.pre_activations[self.total_layers] = self.biases[self.total_layers] + np.dot(self.weights[self.total_layers], self.activations[self.total_layers - 1])
        self.activations[self.total_layers] = softmax_fn(self.pre_activations[self.total_layers])

    def backward_pass(self, Y):
        if self.loss_fn == "cross_entropy":
            self.grad_pre[self.total_layers] = self.activations[self.total_layers] - Y
        elif self.loss_fn == "mean_squared_error":
            self.grad_pre[self.total_layers] = (self.activations[self.total_layers] - Y) * (self.activations[self.total_layers] * (1 - self.activations[self.total_layers]))
        self.grad_biases[self.total_layers] = np.sum(self.grad_pre[self.total_layers], axis=1, keepdims=True)
        self.grad_weights[self.total_layers] = np.dot(self.grad_pre[self.total_layers], self.activations[self.total_layers - 1].T) + self.weight_decay * self.weights[self.total_layers]
        for layer in range(self.total_layers - 1, 0, -1):
            delta = np.dot(self.weights[layer + 1].T, self.grad_pre[layer + 1])
            if self.activation == "Sigmoid":
                deriv = self.activations[layer] * (1 - self.activations[layer])
            elif self.activation == "Tanh":
                deriv = 1 - self.activations[layer] ** 2
            elif self.activation == "ReLU":
                deriv = np.where(self.pre_activations[layer] > 0, 1, 0)
            self.grad_pre[layer] = delta * deriv
            self.grad_biases[layer] = np.sum(self.grad_pre[layer], axis=1, keepdims=True)
            self.grad_weights[layer] = np.dot(self.grad_pre[layer], self.activations[layer - 1].T) + self.weight_decay * self.weights[layer]

    def predict(self, X):
        temp_pre = [None] * (self.total_layers + 1)
        temp_act = [None] * (self.total_layers + 1)
        temp_act[0] = X
        for layer in range(1, self.total_layers):
            temp_pre[layer] = self.biases[layer] + np.dot(self.weights[layer], temp_act[layer - 1])
            if self.activation == "Sigmoid":
                temp_act[layer] = sigmoid(temp_pre[layer])
            elif self.activation == "Tanh":
                temp_act[layer] = tanh_act(temp_pre[layer])
            elif self.activation == "ReLU":
                temp_act[layer] = relu_act(temp_pre[layer])
        temp_pre[self.total_layers] = self.biases[self.total_layers] + np.dot(self.weights[self.total_layers], temp_act[self.total_layers - 1])
        temp_act[self.total_layers] = softmax_fn(temp_pre[self.total_layers])
        return temp_act[self.total_layers].T

    def compute_loss(self, preds, Y):
        loss = 0.0
        N = Y.shape[0]
        if self.loss_fn == "cross_entropy":
            for i in range(N):
                loss += math.log(preds[i][Y[i].argmax()] + 1e-9)
            loss = -loss / N
        elif self.loss_fn == "mean_squared_error":
            loss = np.sum((Y - preds) ** 2) / N
        return loss

    def accuracy(self, preds, Y):
        total = Y.shape[0]
        correct = 0
        for i in range(total):
            if Y[i].argmax() == preds[i].argmax():
                correct += 1
        return correct / total

    def train_sgd(self, X, Y, X_val, Y_val):
        epoch = 0
        num_samples = X.shape[0]
        while epoch < self.epochs:
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                self.forward_pass(X[start:end].T)
                self.backward_pass(Y[start:end].T)
                for layer in range(1, self.total_layers + 1):
                    self.weights[layer] -= self.lr * self.grad_weights[layer]
                    self.biases[layer] -= self.lr * self.grad_biases[layer]
            train_preds = self.predict(X.T)
            train_loss = self.compute_loss(train_preds, Y)
            train_acc = self.accuracy(train_preds, Y)
            val_preds = self.predict(X_val.T)
            val_loss = self.compute_loss(val_preds, Y_val)
            val_acc = self.accuracy(val_preds, Y_val)
            print_and_record(epoch, train_loss, val_loss, train_acc, val_acc)
            epoch += 1

    def fit(self, X_train, Y_train, X_val, Y_val):
        if self.optimizer == "sgd":
            self.train_sgd(X_train, Y_train, X_val, Y_val)