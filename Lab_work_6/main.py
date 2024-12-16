import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. Завантаження та підготовка даних
df = pd.read_csv('output.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
prices = df['Close/Last'].values.reshape(-1, 1)

# Нормалізація даних
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)


# Створення вхідних та вихідних даних
def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


look_back = 7  # Використовуємо сім попередніх значень для передбачення наступного
X, y = create_dataset(prices_scaled, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Перетворення для LSTM


# 2. Побудова штучної нейронної мережі
def build_model():
    model = Sequential([
        LSTM(20, input_shape=(look_back, 1), return_sequences=True),
        LSTM(10, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 3. Навчання та прогнозування
epochs_list = [10, 50, 100, 200, 500, 1000]
results = {}

for epochs in epochs_list:
    model = build_model()
    model.fit(X, y, epochs=epochs, batch_size=10, verbose=0)

    # Прогнозування
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)  # Повернення до вихідного масштабу
    real_values = scaler.inverse_transform(y.reshape(-1, 1))

    # Оцінка точності
    mse = mean_squared_error(real_values, predictions)
    results[epochs] = {
        "mse": mse,
        "predictions": predictions.flatten()
    }

    # Візуалізація прогнозів
    plt.figure()
    plt.plot(df['Date'][look_back:], real_values, label="Actual")
    plt.plot(df['Date'][look_back:], predictions, label=f"Predicted (epochs={epochs})")
    plt.legend()
    plt.title(f"Forecast with {epochs} Epochs")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4. Аналіз залежності точності від кількості епох
epochs = list(results.keys())
mse_values = [results[ep]["mse"] for ep in epochs]

plt.figure()
plt.plot(epochs, mse_values, marker='o')
plt.title("MSE vs Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Mean Squared Error")
plt.grid()
plt.show()
