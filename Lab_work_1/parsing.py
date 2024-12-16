"""
Отримання даних про річну зміну ціни індексу S&P 500 з біржі NASDAQ за допомогою парсингу
"""

from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By

# Ініціалізація веб драйвера відповідно до браузера
driver = webdriver.Chrome()

# Отримання сторінки за URL
url = "https://www.nasdaq.com/market-activity/index/spx/historical?page=1&rows_per_page=1000&timeline=y1"
driver.get(url)

# Очікування, поки сторінка завантажиться
driver.implicitly_wait(10)

# Сторінка нестатична, тому таблицю з даними треба отримати за допомогою скрипта
shadow_host = driver.find_element(By.CSS_SELECTOR, "nsdq-table")
shadow_root = driver.execute_script("return arguments[0].shadowRoot", shadow_host)

# Знаходження відповідної таблиці
table = shadow_root.find_element(By.CSS_SELECTOR, "div.simple-table-template.table")

# Витягування заголовків стовпців таблиці
headers = [header.text for header in table.find_elements(By.CSS_SELECTOR, "div.table-header-cell")]
# Витягування рядків таблиці та значень у комірках
rows = []
for row in table.find_elements(By.CSS_SELECTOR, "div.table-row"):
    cells = [cell.text for cell in row.find_elements(By.CSS_SELECTOR, "div.table-cell")]
    rows.append(cells)

# Створення датафрейму та очищення та перетворення даних до потрібного типу
df = pd.DataFrame(rows, columns=headers)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['Close/Last'] = df['Close/Last'].replace(',', '', regex=True).astype(float)
df = df[['Date', 'Close/Last']]
print(df)
print(df.info())
df.to_csv("output.csv")

# Завершення роботи драйвера
driver.quit()
