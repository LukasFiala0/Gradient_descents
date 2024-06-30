import numpy as np
import pandas as pd

# Nastavení náhodného generátoru
np.random.seed(42)

# Počet vzorků
num_samples = 10000

# Generování první sady dat
X1 = np.random.rand(num_samples, 3)  # Generování hodnot x1, x2, x3
y1 = 2*X1[:, 0] + 3*X1[:, 1] + 4*X1[:, 2] + np.random.normal(0, 0.1, num_samples)  # Generování hodnot y s lineárním modelem
data1 = np.column_stack((X1, y1))  # Sloučení X a y do jednoho pole

# Generování druhé sady dat
X2 = np.random.rand(num_samples, 3)
y2 = -1*X2[:, 0] + 5*X2[:, 1] - 2*X2[:, 2] + np.random.normal(0, 0.1, num_samples)
data2 = np.column_stack((X2, y2))

# Generování třetí sady dat
X3 = np.random.rand(num_samples, 3)
y3 = 3*X3[:, 0] - 2*X3[:, 1] + 6*X3[:, 2] + np.random.normal(0, 0.1, num_samples)
data3 = np.column_stack((X3, y3))

# Vytvoření DataFrame z dat
df1 = pd.DataFrame(data1, columns=['x1', 'x2', 'x3', 'y'])
df2 = pd.DataFrame(data2, columns=['x1', 'x2', 'x3', 'y'])
df3 = pd.DataFrame(data3, columns=['x1', 'x2', 'x3', 'y'])

# Uložení dat do CSV souborů
df1.to_csv('data3.csv', index=False)
df2.to_csv('data4.csv', index=False)
df3.to_csv('data5.csv', index=False)


