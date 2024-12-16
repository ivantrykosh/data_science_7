"""
Оцінка динаміки тренду ціни протягом рік індексу S&P 500 на біржі NASDAQ.
Створення синтетичних даних за цей період на основі лінії тренду реальних даних та шуму за нормальним розподілом.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
std_dev = np.std(df['Close/Last'] - df['Trend'])
noise = np.random.normal(loc=0, scale=std_dev, size=len(df))

# Створюємо синтетичні дані, додаючи шум до тренду
df['Synthetic'] = df['Trend'] + noise

# Побудова графіка реальних, синтетичних даних і лінії тренду
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Close/Last', data=df, label='Реальні дані')
sns.lineplot(x='Date', y='Trend', data=df, label='Лінія тренду', color='red')
sns.lineplot(x='Date', y='Synthetic', data=df, label='Синтетичні дані (з шумом)', linestyle='--', color='green')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.title('Реальні дані індексу S&P 500, лінія тренду і синтетичні дані з шумом')
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
