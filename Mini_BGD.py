import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Načtení dat z .csv
def load_data(file_path):
    data = pd.read_csv(file_path)
    Xy = data.values  
    return Xy

# Příklad použití
Xy= load_data("data2.csv")

# Přidáme sloupec jedniček pro bias termín
Xy = np.c_[np.ones((Xy.shape[0], 1)), Xy]

def create_random_batch(Xy):
    m = Xy.shape[0]
    rnd_index = np.arange(m)
    np.random.shuffle(rnd_index)
    Xy_new = Xy[rnd_index]
    return Xy_new


def create_mini_batches(Xy_new, batch_size):
    m = len(Xy_new)
    num_batches = int(np.ceil(m / batch_size))  # Zaokrouhlení nahoru na nejbližší celé číslo
    mini_batches = []

    # Vytvoření mini-batchů
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, m)  # Pokud bychom překročili konec dat
        mini_batch = Xy_new[start_idx:end_idx]
        mini_batches.append(mini_batch)

    return mini_batches

# Inicializace parametrů theta
def initialize_parameters(n):
    return np.zeros((n, 1))

# Definice modelu lineární regrese
def linear_regression_model(X, theta):
    return np.dot(X, theta)

def mean_squared_error(y, y_predict):
    m = len(y)
    mse = 1 / m * np.sum((y - y_predict) ** 2)
    return mse

def relative_absolute_error(y, y_predict):
    y_mean = np.mean(y)
    #rae = abs(np.sum(y - y_predict)) / abs(np.sum(y - y_mean))
    rae = np.sum(np.abs(y - y_predict)) / np.sum(np.abs(y - y_mean))
    return rae

# Definice gradient descent pro minibatch:
def minibatch_gradient_descent(mini_batches, theta, alpha, iters, eps):
    for batch in mini_batches:
        for i in range(iters):
            X_batch = batch[:, :-1]
            y_batch = batch[:, -1].reshape(-1, 1)
            predict_batch = linear_regression_model(X_batch, theta)
            residuals_batch = predict_batch - y_batch
            gradient = (1 / len(batch)) * np.dot(X_batch.T, residuals_batch)
            theta -= alpha * gradient
    mse = mean_squared_error(y_batch, predict_batch)
    rae = relative_absolute_error(y_batch, predict_batch)
    return theta, mse, rae



Xy_random = create_random_batch(Xy)
mini_batches = create_mini_batches(Xy_random,100)
theta = initialize_parameters(mini_batches[0].shape[1]-1) # počet proměnných  = počet parametrů
theta, mse, rae = minibatch_gradient_descent(mini_batches, theta, 0.001, 10000, 0.1)

print(f"Mean Squared Error: {mse}\nRelative Absolute Error: {rae}\nParameters: {theta}")

