import numpy as np
import pandas as pd

# Načtení dat z CSV souboru
def load_data(file_path):
    """
    Načte data z CSV souboru.
    
    Parametry:
    - file_path: cesta k CSV souboru
    
    Návratové hodnoty:
    - X: matice vstupních dat
    - y: vektor cílových hodnot
    """
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # všechny sloupce kromě posledního
    y = data.iloc[:, -1].values   # poslední sloupec
    return X, y

# Příklad použití
X, y = load_data("data.csv")

# Přidáme sloupec jedniček pro bias termín
X = np.c_[np.ones((X.shape[0], 1)), X]

# Inicializace parametrů theta
def initialize_parameters(n):
    """
    Inicializuje parametry theta na nuly.
    
    Parametry:
    - n: počet parametrů včetně bias termínu
    
    Návratová hodnota:
    - theta: inicializovaný vektor parametrů
    """
    return np.zeros((n, 1))

theta = initialize_parameters(X.shape[1])

# Definice modelu lineární regrese
def linear_regression_model(X, theta):
    """
    Lineární regresní model.
    
    Parametry:
    - X: matice vstupních dat
    - theta: vektor parametrů modelu
    
    Návratová hodnota:
    - predikce: vektor predikcí modelu
    """
    return np.dot(X, theta)

# Definice účelové funkce
def cost_function(X, y, theta):
    """
    Vypočte hodnotu účelové funkce (Mean Squared Error) pro dané parametry.
    
    Parametry:
    - X: matice vstupních dat
    - y: vektor cílových hodnot
    - theta: vektor parametrů modelu
    
    Návratová hodnota:
    - cost: hodnota účelové funkce
    """
    m = len(y)
    predictions = linear_regression_model(X, theta)
    sq_errors = (predictions - y) ** 2
    cost = 1 / (2 * m) * np.sum(sq_errors)
    return cost

# Algoritmus Batch Gradient Descent
def batch_gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Algoritmus Batch Gradient Descent pro optimalizaci parametrů modelu.
    
    Parametry:
    - X: matice vstupních dat
    - y: vektor cílových hodnot
    - theta: počáteční vektor parametrů modelu
    - alpha: rychlost učení (learning rate)
    - num_iterations: počet iterací algoritmu
    
    Návratová hodnota:
    - theta: optimalizovaný vektor parametrů
    - costs: hodnoty nákladové funkce v průběhu iterací
    """
    m = len(y)
    costs = []
    for _ in range(num_iterations):
        predictions = linear_regression_model(X, theta)
        errors = predictions - y
        gradient = (1 / m) * np.dot(X.T, errors)
        theta -= alpha * gradient
        cost = cost_function(X, y, theta)
        costs.append(cost)
    return theta, costs


# Nastavení hyperparametrů
alpha = 0.01  # rychlost učení
num_iterations = 10000  # počet iterací

# Trénování modelu
optimal_theta, costs = batch_gradient_descent(X, y.reshape(-1, 1), theta, alpha, num_iterations)

# Výpis optimalních parametrů
print("Optimal parameters:")
print(optimal_theta)
print(costs[-1])

# # Vykreslení průběhu nákladové funkce
import matplotlib.pyplot as plt
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()


from sklearn.linear_model import LinearRegression
import numpy as np

# Vytvoření instance modelu lineární regrese
model = LinearRegression(fit_intercept=True)

# Trénování modelu pomocí algoritmu Batch Gradient Descent
model.fit(X, y)

# Výpis optimalních parametrů
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)










