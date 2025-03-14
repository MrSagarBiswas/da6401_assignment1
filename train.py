import argparse
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
import Model
import wandb
import datetime

def main():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    # Command-line arguments with best configuration as defaults
    parser.add_argument('-wp', '--wandb_project', default='fashion-mnist-classification', help='Weights & Biases project name')
    parser.add_argument('-we', '--wandb_entity', default='mrsagarbiswas-iit-madras', help='Weights & Biases entity')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
    parser.add_argument('-o', '--optimizer', default='momentum', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('-beta', '--beta', type=float, default=0.9, help='Beta for RMSprop')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 for Adam/Nadam')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 for Adam/Nadam')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, help='Epsilon for optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('-w_i', '--weight_init', default='Xavier', choices=['random', 'Xavier'], help='Weight initialization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('-a', '--activation', default='tanh', choices=['sigmoid', 'tanh', 'ReLU'], help='Activation function')
    args = parser.parse_args()

    # Activation function mapping
    activation_map = {'sigmoid': 'Sigmoid', 'tanh': 'Tanh', 'relu': 'ReLU'}
    activation = activation_map.get(args.activation.lower(), 'Sigmoid')

    if args.wandb_project:
        run_name = f'hl{args.num_layers}_hs_{args.hidden_size}_bs_{args.batch_size}_ac_{activation}_opt_{args.optimizer}_lr_{args.learning_rate}'
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={**vars(args), "activation": activation},
            name=run_name
        )
        is_wandb = True
    else:
        is_wandb = False

    # Load dataset
    if args.dataset == 'mnist':
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    else:
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Split train into 90% train, 10% validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=69)

    # Preprocess data
    def preprocess(X, y):
        X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        y = np.eye(10)[y]
        return X, y

    X_train, y_train = preprocess(X_train, y_train)
    X_val, y_val = preprocess(X_val, y_val)
    X_test, y_test = preprocess(X_test, y_test)

    # Initialize model
    model = Model.NeuralNetwork(
        input_size=784, num_classes=10, num_hidden=args.num_layers, hidden_units=args.hidden_size,
        init_method=args.weight_init, activation=activation, loss_fn=args.loss, epochs=args.epochs,
        batch_size=args.batch_size, optimizer=args.optimizer, lr=args.learning_rate, momentum=args.momentum,
        beta=args.beta, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, weight_decay=args.weight_decay,
        iswandb=is_wandb
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    test_preds = model.predict(X_test.T)
    test_loss = model.compute_loss(test_preds, y_test)
    test_acc = model.accuracy(test_preds, y_test)

    # Evaluate on validation set
    val_preds = model.predict(X_val.T)
    val_loss = model.compute_loss(val_preds, y_val)
    val_acc = model.accuracy(val_preds, y_val)

    # Log final metrics to wandb
    if is_wandb:
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "created": datetime.datetime.now().isoformat()
        })

    # Print final metrics
    print(f"\nFinal Validation => Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    print(f"Test Set => Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()