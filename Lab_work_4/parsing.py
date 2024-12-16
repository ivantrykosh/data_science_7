"""
Парсинг меню із сайтів кав'ярень Львова
URL сайтів:
    - https://kopalnia.choiceqr.com/section:menyu-v-kav-yarnya
    - https://virmenka.com.ua/uk/menu/
    - https://svitkavy.com/uk/cafes/menu/katedralna/
"""

from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd

driver = webdriver.Chrome()


def fetch_html(url):
    """ Отримання HTML коду сторінки за URL адресою """
    driver.get(url)
    driver.implicitly_wait(2)
    response = driver.page_source
    return response


def parse_site_1():
    """ Парсинг першого сайту """
    url = "https://kopalnia.choiceqr.com/section:menyu-v-kav-yarnya"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    items = soup.select(".styles_menu-item-left__HxAVf")
    data = []
    for item in items:
        name = item.select_one(".styles_menu-item-title__Mnuv_").get_text(strip=True)
        price = (item.select_one(".styles_menu-item-price__G8nZ_ .styles_discount__EE8JM").get_text(strip=True).split(' '))[0]
        data.append({"Name": name, "Price": price})
    return data


def parse_site_2():
    """ Парсинг другого сайту """
    url = "https://virmenka.com.ua/uk/menu/"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    items = soup.select(".menu-section__item")
    data = []
    for item in items:
        name = item.select_one(".menu-item__title").get_text(strip=True)
        price = item.select_one(".menu-item__price").get_text(strip=True).split(' ')[0]
        data.append({"Name": name, "Price": price})
    return data


def parse_site_3():
    """ Парсинг третього сайту """
    url = "https://svitkavy.com/uk/cafes/menu/katedralna/"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    items = soup.select("div.cafe-menu ul")
    names = soup.select("div.cafe-menu h2")[1:]
    data = []
    for item_index in range(len(items)):
        lis = items[item_index].select("li")
        for li in lis:
            name = li.select_one(".item").contents[0].get_text(strip=True)
            if name[0].islower():
                name = names[item_index].get_text(strip=True) + " " + name
            price = li.select_one(".price").get_text(strip=True).split('/')[0]
            data.append({"Name": name, "Price": price})
    return data


# Парсинг першого меню
menu = parse_site_1()
print(menu)
# Збереження першого меню
df = pd.DataFrame(menu)
df.to_csv('site_1.csv', index=False, encoding='utf-8-sig')

# Парсинг другого меню
menu = parse_site_2()
print(menu)
# Збереження другого меню
df = pd.DataFrame(menu)
df.to_csv('site_2.csv', index=False, encoding='utf-8-sig')

# Парсинг третього меню
menu = parse_site_3()
print(menu)
# Збереження третього меню
df = pd.DataFrame(menu)
df.to_csv('site_3.csv', index=False, encoding='utf-8-sig')

driver.quit()
