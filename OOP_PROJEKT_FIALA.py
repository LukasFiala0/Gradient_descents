import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

class LinearRegression_mine:
    """Custom implementation of Linear Regression with gradient descent."""
    
    def __init__(self, alpha: float, iters: int, eps: float):
        """
        Initialize LinearRegression_mine object.

        Parameters:
            alpha (float): Learning rate.
            iters (int): Maximum number of iterations.
            eps (float): Convergence threshold for early stopping.
        """
        self.alpha = alpha
        self.iters = iters
        self.eps = eps
        self.mse_history = []

    def loadData(self, filename: str):
        """
        Load data from a CSV file.

        Parameters:
            filename (str): Name of the CSV file containing the data.

        Returns:
            numpy.ndarray: Containing features (X) and target variable (y) arrays and column of bias.
        """
        data = pd.read_csv(filename)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        X = np.c_[np.ones((X.shape[0], 1)), X] #bias term
        return X, y.reshape(-1, 1)
    
    def initializeParameters(self, n: int):
        """
        Initialize parameters theta.

        Parameters:
            n (int): Number of features including the intercept.

        Returns:
            numpy.ndarray: Initialized parameters.
        """
        return np.zeros((n, 1))
    
    def linearRegressionModel(self, X):
        """
        Calculate the linear regression model prediction.

        Parameters:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted values.
        """
        return np.dot(X, self.theta)
    
    def meanSquaredError(self, y, y_predict):
        """
        Calculate Mean Squared Error (MSE).

        Parameters:
            y (numpy.ndarray): True target values.
            y_predict (numpy.ndarray): Predicted values.

        Returns:
            float: Mean Squared Error.
        """
        m = len(y)
        mse = 1 / m * np.sum((y - y_predict) ** 2)
        return mse

    def relativeAbsoluteError(self, y, y_predict):
        """
        Calculate Relative Absolute Error (RAE).

        Parameters:
            y (numpy.ndarray): True target values.
            y_predict (numpy.ndarray): Predicted values.

        Returns:
            float: Relative Absolute Error.
        """
        y_mean = np.mean(y)
        rae = np.sum(np.abs(y - y_predict)) / np.sum(np.abs(y - y_mean))
        return rae
    
    def fit_BGD(self, X, y):
        """
        Fit the model using Batch Gradient Descent.

        Parameters:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target variable.

        Returns:
            None
        """
        self.theta = self.initializeParameters(X.shape[1])
        m = len(y)
        for i in range(self.iters):
            predict_loop = self.linearRegressionModel(X)
            residuals = predict_loop - y
            gradient = (1 / m) * np.dot(X.T, residuals)
            self.theta -= self.alpha * gradient
            mse = self.meanSquaredError(y, predict_loop)
            self.mse_history.append(mse)
            if mse <= self.eps:
                print(f"Converged after {i+1} iterations")
                break

    def fit_SGD(self, X, y):
        """
        Fit the model using Stochastic Gradient Descent.

        Parameters:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target variable.

        Returns:
            None
        """
        self.theta = self.initializeParameters(X.shape[1])
        m = len(y)
        for i in range(self.iters):
            idx = np.random.randint(0, X.shape[0])
            predict = self.linearRegressionModel(X[idx])
            residual = predict - y[idx]
            gradient = X[idx].reshape(-1, 1) * residual
            self.theta -= self.alpha * gradient
            mse = self.meanSquaredError(y, predict)
            self.mse_history.append(mse) 
            if mse <= self.eps:
                print(f"Converged after {i+1} iterations")
                break

    def fit_MBGD(self, X, y, batch_size):
        """
        Fit the model using Mini-batch Gradient Descent.

        Parameters:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target variable.
            batch_size (int): Size of mini-batch.

        Returns:
            None
        """
        self.theta = self.initializeParameters(X.shape[1])
        m = len(y)
        num_batches = m // batch_size
        for i in range(self.iters):
            # Shuffle the data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx = (j + 1) * batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                predict_batch = self.linearRegressionModel(X_batch)
                residuals_batch = predict_batch - y_batch
                gradient = (1 / batch_size) * np.dot(X_batch.T, residuals_batch)
                self.theta -= self.alpha * gradient
            mse = self.meanSquaredError(y, self.linearRegressionModel(X))  
            self.mse_history.append(mse)
            if mse <= self.eps:
                print(f"Converged after {i+1} iterations")
                break

    def predict(self, X):
        """
        Predict target variable.

        Parameters:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted values.
        """
        return self.linearRegressionModel(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.

        Parameters:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): True target values.

        Returns:
            tuple: Mean Squared Error (MSE), Relative Absolute Error (RAE), Predicted values.
        """
        y_predict = self.predict(X)
        mse = self.meanSquaredError(y, y_predict)
        rae = self.relativeAbsoluteError(y, y_predict)
        return mse, rae, y_predict
    
    def predictDiagram(self, y, y_predict):
        """
        Plot predicted vs. true values.

        Parameters:
            y (numpy.ndarray): True target values.
            y_predict (numpy.ndarray): Predicted values.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_predict, color='blue', marker='o', edgecolors='black', alpha=0.6)
        plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
        plt.title(f'Skutečné hodnoty vs. Předpovězené hodnoty, alpha={self.alpha}')
        plt.xlabel('Skutečné hodnoty')
        plt.ylabel('Předpovězené hodnoty')
        plt.grid(True)
        plt.show()

    def graphMSE(self, name):
        """
        Plot MSE history.

        Parameters:
            name (str): Name of the plot.

        Returns:
            None
        """
        plt.figure(figsize=(13,8))
        iterations = np.arange(1, len(self.mse_history) + 1)
        plt.plot(iterations, self.mse_history, color="black")
        plt.title(f'Závislost MSE na počtu iterací, {name}')
        plt.xlabel('Počet iterací')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()



# Příklad použití
model1 = LinearRegression_mine(0.001, 100000, 0.01)
X1, y1 = model1.loadData("data5.csv")
model1.fit_BGD(X1, y1)
mse1, rae1, y_predict1 = model1.evaluate(X1, y1)
print(f"BATCH-GRADIENT-DESCENT\nMean Squared Error: {mse1}\nRelative Absolute Error: {rae1}\nParameters: {model1.theta}")
model1.predictDiagram(y1, y_predict1)
model1.graphMSE("Batch gradient descent")

    
model2 = LinearRegression_mine(0.001, 100000, 0.1)
X2, y2 = model2.loadData("data5.csv")
model2.fit_SGD(X2, y2)
mse2, rae2, y_predict2 = model2.evaluate(X2, y2)
print(f"STOCHASTIC-GRADIENT-DESCENT\nMean Squared Error: {mse2}\nRelative Absolute Error: {rae2}\nParameters: {model2.theta}")
model2.predictDiagram(y2, y_predict2)
model2.graphMSE("Stochastic gradient descent")

model3 = LinearRegression_mine(0.001, 100000, 0.01)
X3, y3 = model3.loadData("data5.csv")
model3.fit_MBGD(X3, y3, batch_size=32)
mse3, rae3, y_predict3 = model3.evaluate(X3, y3)
print(f"MINI-BATCH-GRADIENT-DESCENT\nMean Squared Error: {mse3}\nRelative Absolute Error: {rae3}\nParameters: {model3.theta}")
model3.predictDiagram(y3, y_predict3)
model3.graphMSE("Mini batch gradient descent")

# Porovnání s knihovnou:
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # všechny sloupce kromě posledního
    y = data.iloc[:, -1].values   # poslední sloupec
    return X, y

X, y = load_data("data5.csv")
model_ref = LinearRegression(fit_intercept=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_ref.fit(X_train, y_train)
y_pred = model_ref.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rae = mean_absolute_error(y_test, y_pred)
print(f"VYUŽITÍ KNIHOVNY:\nMean Squared Error: {mse}\nRelative Absolute Error: {rae}")
print("Parametry regrese:")
print("Intercept:", model_ref.intercept_)
print("Koeficienty:", model_ref.coef_)