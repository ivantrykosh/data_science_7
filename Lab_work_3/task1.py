"""
Завдання:
    Розробити програмний скрипт, що реалізує багатокритеріальне оцінювання ефективності
    позашляховиків різних виробників. Формування показників та критеріїв ефективності,
    синтез багатокритеріальної оптимізаційної моделі здійснити самостійно.

Обрані критерії:
    * Ціна (USD) - мінімізований, вага 1
    * Споживання пального (літрів на 100 км) - мінімізований, вага 1
    * Максимальна швидкість (км/год) - максимізований, вага 1
    * Комфорт (оцінка від 1 до 10) - максимізований, вага 1
    * Надійність (оцінка від 1 до 10) - максимізований, вага 1
    * Прохідність (оцінка від 1 до 10) - максимізований, вага 1
"""


import pandas as pd
import numpy as np

# Для повного виводу даних у консоль
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def read_data(file_name):
    """ Зчитування даних з .xls файлу та їх перетворення до типу float """
    sample_data = pd.read_excel(file_name, index_col=0)
    for i in range(0, len(sample_data.columns) - 2):
        sample_data[sample_data.columns[i]] = sample_data[sample_data.columns[i]].replace(',', '.', regex=True).astype(float)
    return sample_data


def model(file_name):
    """ Оцінювання ефективності позашляховиків """

    # Вхідні дані
    line_column_matrix = read_data(file_name)
    print(line_column_matrix)
    scores = np.zeros(line_column_matrix.shape[1] - 2)

    # Критерії та відповідні коефіцієнти
    factors = [factor for factor in line_column_matrix.values]

    # Нормалізовані фактори
    norm_factors = [factor[:-2] for factor in factors]

    # Нормована вага кожного фактору
    g_norm = sum([factor[-1] for factor in factors])
    gs = [factor[-1] / g_norm for factor in factors]

    # Знаходження суми факторів (для нормалізації)
    sum_factors = [0 for _ in range(line_column_matrix.shape[0])]
    for i in range(len(factors)):
        if factors[i][-2]:  # Максимізований фактор
            sum_factors[i] = (1 / factors[i][:-2]).sum()
        else:  # Мінімізований фактор
            sum_factors[i] = factors[i][:-2].sum()

    # Нормалізація факторів
    for i in range(line_column_matrix.shape[1] - 2):
        for j in range(len(norm_factors)):
            if factors[j][-2]:  # Максимізований фактор
                norm_factors[j] = (1 / factors[j][:-2]) / sum_factors[j]
            else:  # Мінімізований фактор
                norm_factors[j] = factors[j][:-2] / sum_factors[j]

    # Розрахунок оцінки (score)
    for i in range(len(scores)):
        scores[i] = 0
        for j in range(len(gs)):
            scores[i] += gs[j] * (1 - norm_factors[j][i])**(-1)

    # Знаходження оптимального рішення (позашляховика)
    min_value = float('inf')
    optimal = 0
    for i in range(len(scores)):
        if min_value > scores[i]:
            min_value = scores[i]
            optimal = i
    print('\nІнтегрована оцінка (score):\n', scores)
    print('Номер оптимального позашляховика: ', optimal)
    return


if __name__ == '__main__':
    model(file_name='data.xls')
