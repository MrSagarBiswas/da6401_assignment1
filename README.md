# Fashion MNIST Classification using a Custom Neural Network

This repository contains a custom implementation of a fully connected neural network built from scratch (using NumPy) to classify images from the MNIST and Fashion MNIST datasets. The network supports multiple activation functions, loss functions, and optimizers. It also integrates with Weights & Biases (WandB) for logging metrics.

> **Note:**  
> - The project code is modularized into separate files:
>   - **Model.py**: Contains the `NeuralNetwork` class implementation.
>   - **train.py**: The main training script with command-line hyperparameter configuration.
>   - **Jupyter Notebooks**: Specific notebooks for experiments and questions (e.g., Questions 1, 4, 7, 8, 10).

---

## Project Structure

- **Model.py**  
  - Implements the `NeuralNetwork` class.
  - Contains activation functions: sigmoid, tanh, ReLU, and softmax.
  - Implements forward and backward propagation routines.
  - Includes multiple optimizer update rules (SGD, Momentum, RMSprop, Adam, Nadam).
  - Provides helper functions for logging (via WandB), loss computation, and accuracy calculation.

- **train.py**  
  - Parses command-line arguments for hyperparameter settings.
  - Loads and preprocesses the dataset (MNIST or Fashion MNIST).
  - Splits the data into training (90%), validation (10%), and test sets.
  - Instantiates and trains the `NeuralNetwork` model.
  - Evaluates the model on the validation and test sets.
  - Logs final metrics (loss and accuracy) to WandB (if configured).

- **Jupyter Notebooks**  
  - Several notebooks are provided for specific experiments and questions (e.g., Questions 1, 4, 7, 8, 10) with additional plots and detailed analysis.

---

## Dependencies

Ensure you have the following installed:
- Python 3.x
- NumPy
- TensorFlow (for accessing Keras datasets)
- scikit-learn
- WandB (Weights & Biases)
- (Optional) Jupyter Notebook

Install the required packages via:
```bash
pip install numpy tensorflow scikit-learn wandb
```

## Training the Model

To train the model, run the `train.py` script. It accepts several command-line arguments to customize training.

### Usage


### Common Options

- `-d, --dataset`: Choose between `mnist` and `fashion_mnist` (default: `fashion_mnist`).
- `-e, --epochs`: Number of training epochs (default: `10`).
- `-b, --batch_size`: Batch size for training (default: `64`).
- `-l, --loss`: Loss function to use (`cross_entropy` or `mean_squared_error`; default: `cross_entropy`).
- `-o, --optimizer`: Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, or `nadam`; default: `momentum`).
- `-lr, --learning_rate`: Learning rate (default: `0.0001`).
- `-nhl, --num_layers`: Number of hidden layers (default: `4`).
- `-sz, --hidden_size`: Number of neurons per hidden layer (default: `128`).
- `-a, --activation`: Activation function (`sigmoid`, `tanh`, or `ReLU`; default: `tanh`).
- `-wp, --wandb_project`: Weights & Biases project name.
- `-we, --wandb_entity`: Weights & Biases entity.

### Examples

1. Train on Fashion MNIST using Adam for 20 epochs:

```bash
python train.py -d fashion_mnist -e 20 -o adam
```

2. Train on MNIST using SGD for 15 epochs with a batch size of 32 and a learning rate of 0.001:

```bash
python train.py -d mnist -e 15 -b 32 -o sgd -lr 0.001
```

3. Train on MNIST using RMSprop for 30 epochs, a batch size of 16, a learning rate of 0.0005, and ReLU activation:

```bash
python train.py -d mnist -e 30 -b 16 -o rmsprop -lr 0.0005 -a ReLU
```

4. Train on Fashion MNIST using Nadam for 10 epochs with weight decay 0.5 and a hidden size of 64:

```bash
python train.py -d fashion_mnist -e 10 -o nadam -w_d 0.5 -sz 64
```

## Evaluation

After training, the model is evaluated on both the validation and test sets. The script computes the loss (using the selected loss function) and the accuracy of the model. Final metrics are printed to the console and, if configured, logged to Weights & Biases.

## Weights & Biases Integration

To enable logging with Weights & Biases, provide your project name and entity when running the script:

```bash
python train.py -wp your_project_name -we your_entity
```

Ensure you have a Weights & Biases account and have installed the wandb library.

## Customization

You can customize the neural network architecture and training parameters either by modifying the command-line arguments in `train.py` or by directly editing `Model.py`.

## Conclusion

This repository provides a flexible framework for training and evaluating a custom neural network. Experiment with different optimizers, activation functions, and network configurations. Contributions and suggestions are welcome!

Happy training!

