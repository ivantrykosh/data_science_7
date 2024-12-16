"""
Кластеризація даних за рік про значення індексу S&P 500 методом k-means
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Зчитуємо дані та обираємо стовпці 2 (індекс дня) та 1 (значення) як ознаки
data = pd.read_csv('s&p500.csv', index_col=0)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date', ascending=True).reset_index(drop=True)
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
X = data.iloc[:, [2, 1]].values
print(X)

# Визначимо оптимальну кількість кластерів з допомогою методу Elbow
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
# Оптимальна кількість кластерів - 4
plt.plot(range(1, 11), inertia)
plt.title('Метод Elbow')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція')
plt.show()

# Виконуємо кластеризацію
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Виконуємо візуалізацію
for i in range(4):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, label=f'Кластер {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, label='Центри кластерів')
plt.title('Кластеризація даних про індекс S&P 500')
plt.xlabel('Індекс дня')
plt.ylabel('Значення індексу S&P 500')
plt.legend()
plt.show()
