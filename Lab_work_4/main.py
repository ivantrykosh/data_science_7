"""
Узагальнення інформації про меню кав'ярень Львова:
    - знаходження середньої ціни напоїв або страв у меню кожної кав'ярні
    - знаходження середньої ціни напоїв або страв у меню усіх кав'ярень
    - розподіл цін напоїв та страв по категоріях Низька, Середня, Висока по всіх кав'ярнях
    - кількість напоїв та страв у меню кожної кав'ярні
    - порівняння ціни напоїв та страв, які є у кожній із кав'ярень
"""

import pandas as pd
from matplotlib import pyplot as plt

# Зчитування даних про меню кав'ярень
df_site1 = pd.read_csv("site_1.csv")
df_site2 = pd.read_csv("site_2.csv")
df_site3 = pd.read_csv("site_3.csv")

# Назви кав'ярень
site_1_name = "Львівська копальня кави"
site_2_name = "Вірменка"
site_3_name = "Світ кави"


# Обчислення середньої ціни у меню кожної кав'ярні
site_means = {
    site_1_name: df_site1["Price"].mean(),
    site_2_name: df_site2["Price"].mean(),
    site_3_name: df_site3["Price"].mean()
}
print("Середня ціна у кожній кав'ярні (грн): ", site_means)

# Побудова стовпчастої діаграми
plt.figure(figsize=(8, 6))
plt.bar(list(site_means.keys()), list(site_means.values()), color=["blue", "orange", "green"])
plt.xlabel("Кав'ярня")
plt.ylabel("Середня ціна (грн)")
plt.title("Середня ціна у меню різних кав'ярень")
plt.show()


# Обчислення середньої ціни у меню у всіх кав'ярнях
# Дані про ціни з трьох кав'ярень
prices = pd.concat([df_site1["Price"], df_site2["Price"], df_site3["Price"]])
print("\nСередня ціна по трьох кав'ярнях (грн): ", prices.mean())


# Визначення розподілу цін напоїв та страв по категоріях Низька, Середня, Висока по всіх кав'ярнях
def price_category(price):
    """ Визначення категорії за значенням ціни """
    if price < 100:
        return "Низька"
    elif 100 <= price <= 200:
        return "Середня"
    else:
        return "Висока"


# Визначаємо категорії
prices_category = prices.apply(price_category)
print("\nРозподіл по категоріях:\n", prices_category.value_counts())
# Побудова гістограми
plt.figure(figsize=(8, 6))
prices_category.value_counts()\
    .plot(kind="bar", color=["blue", "orange", "green"])
plt.xlabel("Категорія")
plt.ylabel("Кількість напоїв та страв у меню")
plt.title("Розподіл напоїв та страв по категоріях")
plt.xticks(rotation=0)
plt.show()


# Знаходження кількості напоїв та страв у кожному меню та у всіх разом
# Підрахунок кількості напоїв та страв у кожному меню
count_site1 = df_site1.shape[0]
count_site2 = df_site2.shape[0]
count_site3 = df_site3.shape[0]

# Загальна кількість напоїв та страв у всіх меню
total_count = count_site1 + count_site2 + count_site3

print()
print(f"Кількість напоїв та страв у {site_1_name}: {count_site1}")
print(f"Кількість напоїв та страв у {site_2_name}: {count_site2}")
print(f"Кількість напоїв та страв у {site_3_name}: {count_site3}")
print(f"Загальна кількість напоїв та страв у меню: {total_count}")


# Порівняння цін на напої та страви, які є у кожній з кав'ярень
# Множини назв напоїв та страв у кожному меню
site1_items = set(df_site1["Name"])
site2_items = set(df_site2["Name"])
site3_items = set(df_site3["Name"])

# Знаходимо спільні товари у всіх трьох меню
common_items = site1_items & site2_items & site3_items

# Фільтруємо кожне меню для отримання лише спільних напоїв та страв
df_site1_common = df_site1[df_site1["Name"].isin(common_items)]
df_site2_common = df_site2[df_site2["Name"].isin(common_items)]
df_site3_common = df_site3[df_site3["Name"].isin(common_items)]

# Об'єднуємо для порівняння
comparison_df = pd.merge(
    df_site1_common[["Name", "Price"]], df_site2_common[["Name", "Price"]],
    on="Name", suffixes=(f" {site_1_name}", f" {site_2_name}")
)
comparison_df = pd.merge(comparison_df, df_site3_common[["Name", "Price"]], on="Name")
comparison_df.rename(columns={"Price": f"Price {site_3_name}"}, inplace=True)

print("\nПорівняння ціни на спільні напої та страви у всіх трьох меню:")
print(comparison_df)
