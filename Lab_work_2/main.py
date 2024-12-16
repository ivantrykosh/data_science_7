"""
 - Створення синтетичних даних про ціну індексу S&P 500 на біржі NASDAQ за рік на основі
лінії тренду реальних даних, шуму та аномальних вимірів за нормальним розподілом.
 - Очищення даних від аномальних вимірів з допомогою методу ковзного вікна.
 - Визначення якості моделі з допомогою R² та оптимізація моделі за допомогою зміни параметрів розподілу шуму
 - Статистичне навчання поліноміальної моделі за МНК та прогнозування ціни індексу на наступні пів року
 - Рекурентне згладжування α - β фільтром та усунення "розбіжності" фільтра з допомогою коригування вихідного значення
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.metrics import r2_score


# Зчитування даних у датафрейм та початкове перетворення
df = pd.read_csv('output.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)

# Переведемо дату в кількість днів для лінійної регресії
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Обчислення коефіцієнтів лінійної регресії y = α + β * x
coefficients = np.polyfit(df['Days'], df['Close/Last'], 1)
beta = coefficients[0]
alpha = coefficients[1]

# Обчислення значень для лінії тренду
df['Trend'] = beta * df['Days'] + alpha

# Побудова графіка реальних даних та лінії тренду
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Close/Last', data=df, label='Реальні дані')
sns.lineplot(x='Date', y='Trend', data=df, label='Лінія тренду', color='red')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.title('Динаміка зміни ціни індексу S&P 500 з часом')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.show()

# Додаємо шум (нормальний розподіл з середнім 0 і стандартним відхиленням, близьким до реального)
np.random.seed(42)
std_dev = np.std(df['Close/Last'] - df['Trend']) * 0.75
noise = np.random.normal(loc=0, scale=std_dev, size=len(df))

# Створюємо синтетичні дані, додаючи шум до тренду
df['Synthetic'] = df['Trend'] + noise

# Додаємо аномалії (2% від усіх даних)
num_anomalies = int(df.shape[0] * 0.02)
anomaly_magnitude = 5 * std_dev
anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)
df.loc[anomaly_indices, 'Synthetic'] += np.random.normal(loc=0, scale=anomaly_magnitude, size=num_anomalies)

# Побудова графіка синтетичних даних і лінії тренду
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Trend', data=df, label='Лінія тренду', color='red')
sns.lineplot(x='Date', y='Synthetic', data=df, label='Синтетичні дані (з шумом та аномаліями)')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.title('Лінія тренду і синтетичні дані з шумом та аномаліями')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.legend()
plt.show()

# Описова статистика для реальних даних
real_statistics = df['Close/Last'].describe()
print("Статистика для реальних даних:")
print(real_statistics)

# Описова статистика для синтетичних даних
synthetic_statistics = df['Synthetic'].describe()
print("\nСтатистика для синтетичних даних:")
print(synthetic_statistics)


def remove_anomalies(data_with_anomalies, window_size):
    """ Видалення аномальних вимірів з допомогою алгоритму ковзного вікна """
    number_of_data = len(data_with_anomalies)
    iterations = math.ceil(number_of_data - window_size) + 1
    medians = np.zeros(number_of_data)
    for i in range(iterations):
        window = np.zeros(window_size)
        for j in range(window_size):
            window[j] = data_with_anomalies[i + j]
        medians[i + window_size - 1] = np.median(window)
    cleared_data = np.zeros(number_of_data)
    for i in range(number_of_data):
        cleared_data[i] = medians[i]
    for i in range(window_size):
        cleared_data[i] = data_with_anomalies[i]
    return cleared_data


# Виявлення та очищення аномалій
df['Synthetic'] = remove_anomalies(df['Synthetic'], 4)

# Побудова графіка синтетичних даних і лінії тренду
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Trend', data=df, label='Лінія тренду', color='red')
sns.lineplot(x='Date', y='Close/Last', data=df, label='Реальні дані')
sns.lineplot(x='Date', y='Synthetic', data=df, label='Синтетичні дані')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.title('Лінія тренду і очищені синтетичні дані')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.legend()
plt.show()

# Обчислення R²
r2 = r2_score(df['Close/Last'], df['Synthetic'])
print(f'\nКоефіцієнт детермінації R² для синтетичних даних = {r2}')  # 0.925 при 1 * std; 0.928 при 0.75*std;

# --------------------------- Вимога 1.5 ---------------------------

# Визначаємо дані для навчання моделі
X = np.arange(len(df))  # Для спрощення візьмемо індекси замість дат
y = df['Synthetic']

# Вибираємо ступінь поліноміальної моделі
degree = 3
# Знаходимо коефіцієнти поліноміальної моделі
coeffs = np.polyfit(X, y, degree)
# Створюємо поліноміальну функцію на основі коефіцієнтів
poly_model = np.poly1d(coeffs)
# Прогнозування для навчальних даних
y_pred = poly_model(X)

# Оцінка моделі за допомогою R²
r2 = r2_score(y, y_pred)
print(f'\nКоефіцієнт детермінації R² для навченої поліноміальної моделі = {r2}')

# Побудова графіка результатів
plt.figure(figsize=(10, 6))
sns.lineplot(x=X, y=df['Synthetic'], label='Синтетичні дані')
plt.plot(X, y_pred, label="Поліноміальна регресія", color='red')
plt.title('Поліноміальна регресія')
plt.xlabel('Індекс')
plt.ylabel('Значення')
plt.legend()
plt.show()

# --------------------------- Вимога 1.6 ---------------------------

# Прогноз на 0,5 інтервалу спостереження
n_new = int(0.5 * len(X))  # Кількість нових точок для прогнозу
X_future = np.arange(len(X), len(X) + n_new)  # Новий інтервал для прогнозу

# Прогноз на нові дані
y_future_pred = poly_model(X_future)

# Побудова графіка для прогнозу
plt.figure(figsize=(10, 6))
sns.lineplot(x=X, y=df['Synthetic'], label='Синтетичні дані')
plt.plot(X, y_pred, label="Поліноміальна регресія (Навчання)", color='red')
plt.plot(X_future, y_future_pred, label="Прогноз", color='green')
plt.title("Поліноміальна регресія з екстраполяцією")
plt.xlabel("Індекс")
plt.ylabel("Значення")
plt.legend()
plt.show()


# --------------------------- Вимога 2.5 ---------------------------

def alpha_beta_filter(data):
    """ α - β фільтр """
    number_of_data = len(data)
    in_values = np.array(data, dtype=float).reshape(number_of_data, 1)
    out_values = np.zeros((number_of_data, 1))
    interval = 1

    speed = (in_values[1, 0] - in_values[0, 0]) / interval
    extra = in_values[0, 0] + speed
    alpha = 2 * (2 * 1 - 1) / (1 * (1 + 1))
    beta = 6 / (1 * (1 + 1))

    out_values[0, 0] = in_values[0, 0]
    for i in range(1, number_of_data):
        out_values[i, 0] = extra + alpha * (in_values[i, 0] - extra)
        speed = speed + (beta / interval) * (in_values[i, 0] - extra)
        extra = out_values[i, 0] + speed

        alpha = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))
    return out_values


def alpha_beta_filter_with_optimization(data):
    """
    α - β фільтр з оптимізацією для подолання явища розбіжності фільтра.
    Оптимізація: якщо похибка між прогнозованим та реальним значенням перевищує встановлений поріг,
    то прогнозоване значення коригується із заданим коефіцієнтом корекції.
    """
    number_of_data = len(data)
    in_values = np.array(data, dtype=float).reshape(number_of_data, 1)
    out_values = np.zeros((number_of_data, 1))
    interval = 1
    threshold = 50
    correction_factor = 0.5

    speed = (in_values[1, 0] - in_values[0, 0]) / interval
    extra = in_values[0, 0] + speed
    alpha = 2 * (2 * 1 - 1) / (1 * (1 + 1))
    beta = 6 / (1 * (1 + 1))

    out_values[0, 0] = in_values[0, 0]

    for i in range(1, number_of_data):
        error = in_values[i, 0] - extra
        if abs(error) > threshold:
            extra += correction_factor * error

        out_values[i, 0] = extra + alpha * error
        speed = speed + (beta / interval) * error
        extra = out_values[i, 0] + speed

        alpha = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))

    return out_values


# Застосування фільтра alpha-beta
abf = alpha_beta_filter(df['Synthetic'])
abf_optimized = alpha_beta_filter_with_optimization(df['Synthetic'])

# Візуалізація результатів
plt.plot(df['Synthetic'], label='Синтетичні дані', color='blue')
plt.plot(abf, label='Alpha - beta', color='green')
plt.plot(abf_optimized, label='Optimized alpha - beta', color='red')
plt.legend()
plt.show()
