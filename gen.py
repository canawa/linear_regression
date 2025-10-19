import numpy as np
import pandas as pd

# Размер датасета
n_samples = 100000  # 100 тысяч строк
n_features = 5      # 5 признаков

# Генерация признаков
X = np.random.rand(n_samples, n_features) * 100  # случайные числа от 0 до 100

# Задаём "истинные" коэффициенты для линейной модели
true_coeffs = np.array([3.5, -2.2, 7.1, 0.5, -1.7])
bias = 10

# Генерация целевой переменной с шумом
y = X @ true_coeffs + bias + np.random.normal(0, 10, size=n_samples)

# Создание DataFrame
columns = [f'feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=columns)
df['target'] = y

# Сохраняем в CSV (если нужно)
df.to_csv('linear_regression_dataset.csv', index=False)

print(df.head())