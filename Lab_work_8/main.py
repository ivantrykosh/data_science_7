from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ----------------------------------- ПІДГОТОВКА ВХІДНИХ ДАНИХ -----------------------------------
# Зчитування файлу даних
d_sample_data = pd.read_excel('sample_data.xlsx', parse_dates=['birth_date'])
print('\nd_sample_data\n', d_sample_data)
title_d_sample_data = d_sample_data.columns

# Аналіз структури вхідних даних
print('\nInfo:')
d_sample_data.info()

# Зчитування файлу з інформацією про стовпці даних
d_data_description = pd.read_excel('data_description.xlsx')
print('\nd_data_description\n', d_data_description)

# Первинне формування скорингової таблиці
d_segment_data_description_client_bank = d_data_description[
    (d_data_description.Place_of_definition == 'Вказує позичальник')
    | (d_data_description.Place_of_definition == 'параметри, повязані з виданим продуктом')
]
d_segment_data_description_client_bank.index = range(0, len(d_segment_data_description_client_bank))
print('\nd_segment_data_description_client_bank\n', d_segment_data_description_client_bank)

# Очищення даних
# Аналіз перетину скорингових індикаторів та сегменту вхідних даних
b = d_segment_data_description_client_bank['Field_in_data']
flag_b = set(b).issubset(d_sample_data.columns)
print('\nСегмент columns за співпадінням:', flag_b)

matches = [field for field in b if field in d_sample_data.columns]
num_matches = len(matches)
print('Кількість співпадінь:', num_matches)

indices_matches = [i for i, field in enumerate(b) if field in d_sample_data.columns]
print('Індекси співпадінь:', indices_matches)

# Формування DataFrame даних з урахуванням відсутніх індикаторів скорингової таблиці
d_segment_data_description_client_bank_True = d_segment_data_description_client_bank.iloc[indices_matches]
d_segment_data_description_client_bank_True.index = range(0, len(d_segment_data_description_client_bank_True))
print('\nDataFrame співпадінь:\n', d_segment_data_description_client_bank_True)

# Очищення скорингової таблиці від пропусків (видалення стовпців з пропусками)
b = d_segment_data_description_client_bank_True['Field_in_data']
d_segment_sample_data_client_bank = d_sample_data[b]
print('\nПропуски даних сегменту DataFrame:\n', d_segment_sample_data_client_bank.isnull().sum())

d_segment_data_description_cleaning = d_segment_data_description_client_bank_True.dropna(axis=1)
d_segment_data_description_cleaning.index = range(0, len(d_segment_data_description_cleaning))
d_segment_data_description_cleaning.to_excel('d_segment_data_description_cleaning.xlsx')

# Очищення вхідних даних та збереження скорингової таблиці
d_segment_sample_cleaning = d_segment_sample_data_client_bank.drop(
    columns=['fact_addr_start_date', 'position_id', 'employment_date',
             'has_prior_employment', 'prior_employment_start_date',
             'prior_employment_end_date', 'income_frequency_other'])
d_segment_sample_cleaning.index = range(0, len(d_segment_sample_cleaning))
d_segment_sample_cleaning.to_excel('d_segment_sample_cleaning.xlsx')
print('\nПеревірка наявності пропусків у даних:\n', d_segment_sample_cleaning.isnull().sum())
print('\nСкорингова карта\n', d_segment_sample_cleaning)
print('\nІндикатори скорингу\n', d_segment_data_description_cleaning)


# ----------------------------------- ФОРМУВАННЯ СКОРИНГОВОЇ МОДЕЛІ -----------------------------------
# Завантаження даних
data = pd.read_excel('d_segment_sample_cleaning.xlsx', sheet_name='Sheet1')

# Підготовка даних. Формування стовпця age
today = datetime.now()
data['age'] = data['birth_date'].apply(
    lambda date: today.year - date.year - ((today.month, today.day) < (date.month, date.day))
)
data.drop(columns=['birth_date'], inplace=True)
print('\nData info:')
data.info()

X = data

# Масштабування даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Визначення динамічних ваг (PCA), або ж компонентів
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
data['PCA1'], data['PCA2'], data['PCA3'] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

# Доля варіації, пояснена кожною компонентою
explained_variance_ratio = pca.explained_variance_ratio_
print('\nДоля варіації, пояснена кожною компонентою:\n', explained_variance_ratio)

# Кластеризація для ризикових груп (буде дві групи ризику - одній буде надано кредит, а іншій - ні)
kmeans = KMeans(n_clusters=2, random_state=42)
data['Risk_Group'] = kmeans.fit_predict(X_pca)

# Збереження результатів
data.to_excel('scoring_results.xlsx', index=False)

# Аналіз середніх значень деяких характеристик для кожної групи ризику
group_analysis = data.groupby('Risk_Group')[['loan_amount', 'loan_days', 'age', 'monthly_income']].mean()
print(group_analysis)

# Візуалізація кластеризації
# Створення 3D-графіка
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Розподіл точок за трьома компонентами
scatter = ax.scatter(data['PCA1'], data['PCA2'], data['PCA3'], c=data['Risk_Group'], cmap='viridis', s=60)

# Додавання легенди
legend1 = ax.legend(*scatter.legend_elements(), title='Risk Group', fontsize=10)
ax.add_artist(legend1)

# Налаштування осей
ax.set_title('3D-візуалізація кластерів за головними компонентами', fontsize=16)
ax.set_xlabel('Головна компонента 1 (PCA1)', fontsize=12)
ax.set_ylabel('Головна компонента 2 (PCA2)', fontsize=12)
ax.set_zlabel('Головна компонента 3 (PCA3)', fontsize=12)

# Показ графіка
plt.tight_layout()
plt.show()

# Виявлення шахрайства з допомогою Isolation Forest
# Ініціалізація Isolation Forest
iso_forest = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
data['Fraud_Score'] = iso_forest.fit_predict(X_pca)

# Аналіз результатів (1 - шайхраство не виявлено, -1 - шахрайство виявлено)
print('\n', data['Fraud_Score'].value_counts())
# Збереження результатів
data.to_excel('scoring_and_fraud_results.xlsx', index=False)

# Візуалізація результатів
# Створення 3D-графіка
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Розподіл точок за трьома компонентами
scatter = ax.scatter(data['PCA1'], data['PCA2'], data['PCA3'], c=data['Fraud_Score'], cmap='viridis', s=60)

# Додавання легенди
legend1 = ax.legend(*scatter.legend_elements(), title='Fraud score (1 - no fraud, -1 - fraud)', fontsize=10)
ax.add_artist(legend1)

# Налаштування осей
ax.set_title('3D-візуалізація результатів визначення шахрайства', fontsize=16)
ax.set_xlabel('Головна компонента 1 (PCA1)', fontsize=12)
ax.set_ylabel('Головна компонента 2 (PCA2)', fontsize=12)
ax.set_zlabel('Головна компонента 3 (PCA3)', fontsize=12)

# Показ графіка
plt.tight_layout()
plt.show()
