import numpy as np
import pandas as pd

# Načtení dat z .csv
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # všechny sloupce kromě posledního
    y = data.iloc[:, -1].values   # poslední sloupec
    return X, y

# Příklad použití
X, y = load_data("data2.csv")

# Přidáme sloupec jedniček pro bias termín
X = np.c_[np.ones((X.shape[0], 1)), X]

# Inicializace parametrů theta
def initialize_parameters(n):
    return np.zeros((n, 1))

theta = initialize_parameters(X.shape[1]) # počet proměnných  = počet parametrů

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

def relative_absolute_error(y, y_predict):
    y_mean = np.mean(y)
    rae = np.sum(np.abs(y - y_predict)) / np.sum(np.abs(y - y_mean))
    return rae

def stochastic_gradient_descent(X, y, theta, alpha, iters, eps):
    for i in range(iters):
        idx = np.random.randint(0, X.shape[0])  # Index jednoho náhodného řádku
        predict = linear_regression_model(X[idx], theta)
        residual = predict - y[idx]
        gradient = X[idx].reshape(-1, 1) * residual
        theta -= alpha * gradient

        if abs(residual) <= eps:
            print(f"converted after {i +1} iterations")
            break
        
    # Výpočet chyby
    y_predict = linear_regression_model(X, theta)
    mse = mean_squared_error(y, y_predict)
    rae = relative_absolute_error(y, y_predict)
        

    return theta, mse, rae

# Spuštění SGD
theta, mse, rae = stochastic_gradient_descent(X, y.reshape(-1,1), theta, alpha=0.1, iters=10000, eps = 0.00001)
print(f"Mean Squared Error: {mse}\nRelative Absolute Error: {rae}\nParameters: {theta}")
