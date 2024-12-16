import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Завантаження даних
file_name = 'Data_Set_10.xls'
data = pd.read_excel(file_name, sheet_name='Orders')

# Перевірка структури даних
print(data.head())
data.info()

# Візуалізація даних
# Загальний тренд продажів
data_grouped = data.groupby('Order Date')['Sales'].sum()
plt.figure(figsize=(10, 6))
plt.plot(data_grouped.index, data_grouped.values, label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Продажі за кожне півріччя
# Створення колонки для півріччя
data['Half-Year'] = data['Order Date'].dt.year.astype(str) + '-' + data['Order Date'].dt.month.apply(lambda x: 'H1' if x <= 6 else 'H2')

# Групування даних за півріччями та обчислення сумарних продажів
half_year_sales = data.groupby('Half-Year')['Sales'].sum().sort_index()

# Побудова стовпчикової діаграми
plt.figure(figsize=(10, 6))
half_year_sales.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Sales by Half-Year')
plt.xlabel('Half-Year')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Розрахунок продажів
total_sales = data['Sales'].sum()
print(f'Загальний обсяг продажів: {total_sales}')

# Розрахунок прибутку
total_profit = data['Profit'].sum()
print(f'Загальний прибуток: {total_profit}')

# Статистичні характеристики вибірки
print('Статистичні характеристики для продажів:')
print(data['Sales'].describe())
print('Статистичні характеристики для прибутку:')
print(data['Profit'].describe())


# Поліноміальна регресія 9-го порядку для згладжування
# Беремо дані по датах і продажах
sales_data = data_grouped.reset_index()
sales_data['Days'] = (sales_data['Order Date'] - sales_data['Order Date'].min()).dt.days
X = sales_data[['Days']]
y = sales_data['Sales']

# Регресійна модель
poly = PolynomialFeatures(degree=9)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Додавання згладжених продажів до даних
sales_data['Sales Smoothed'] = poly_model.predict(X_poly)

# Візуалізація згладжування
plt.figure(figsize=(10, 6))
plt.plot(sales_data['Order Date'], sales_data['Sales'], label='Original Sales')
plt.plot(sales_data['Order Date'], sales_data['Sales Smoothed'], label='Smoothed Sales', color='red')
plt.title('Sales with Polynomial Smoothing')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# Обчислення RMSE для оцінки моделі
rmse = mean_squared_error(sales_data['Sales'], sales_data['Sales Smoothed'], squared=False)
print('RMSE: ', rmse)


# Прогнозування на пів року
future_days = np.arange(sales_data['Days'].max() + 1, sales_data['Days'].max() + 181).reshape(-1, 1)
future_days_poly = poly.transform(future_days)
future_sales = poly_model.predict(future_days_poly)

# Візуалізація прогнозу
plt.figure(figsize=(10, 6))
plt.plot(sales_data['Order Date'], sales_data['Sales'], label='Original Sales')
plt.plot(sales_data['Order Date'], sales_data['Sales Smoothed'], label='Smoothed Sales', color='red')
future_dates = pd.date_range(sales_data['Order Date'].max(), periods=180, freq='D')
plt.plot(future_dates, future_sales, label='Forecasted Sales (180 Days)', linestyle='--', color='green')
plt.title('Sales with 6-Month Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# Продажі за кожне півріччя + прогноз
future_dates = [data['Order Date'].max() + pd.DateOffset(days=i) for i in range(1, 181)]
forecast_df = pd.DataFrame({
    'Order Date': future_dates,
    'Sales': future_sales
})
data_ext = pd.concat([data, forecast_df], ignore_index=True)

# Створення колонки для півріччя
data_ext['Half-Year'] = data_ext['Order Date'].dt.year.astype(str) + '-' + data_ext['Order Date'].dt.month.apply(lambda x: 'H1' if x <= 6 else 'H2')

# Групування даних за півріччями та обчислення сумарних продажів
half_year_sales = data_ext.groupby('Half-Year')['Sales'].sum().sort_index()

# Побудова стовпчикової діаграми
plt.figure(figsize=(10, 6))
half_year_sales.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Sales by Half-Year')
plt.xlabel('Half-Year')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
