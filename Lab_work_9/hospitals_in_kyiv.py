from geopy.geocoders import Nominatim
import osmnx as ox
import time
import matplotlib.pyplot as plt
import folium
from geopy.exc import GeocoderTimedOut

# Початкові параметри
city = "Kyiv, Ukraine"
tags = {"amenity": ["hospital", "clinic"]}  # Медичні заклади

# Завантаження даних
data = ox.features_from_place(city, tags)
data = data.reset_index()

# Вибір основної інформації
selected_columns = ['id', 'name', 'amenity', 'geometry']
data_cleaned = data[selected_columns]

# Додавання координат із геометрії
data_cleaned['latitude'] = data_cleaned['geometry'].apply(lambda geom: geom.y if geom.geom_type == 'Point' else None)
data_cleaned['longitude'] = data_cleaned['geometry'].apply(lambda geom: geom.x if geom.geom_type == 'Point' else None)

# Перевірка записів із відсутніми або некоректними координатами та іменами і їх видалення
invalid_coords = data_cleaned[(data_cleaned['latitude'].isna()) | (data_cleaned['longitude'].isna())]
data_cleaned = data_cleaned.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
data_cleaned = data_cleaned.loc[~data_cleaned['name'].isna()]

print(data_cleaned[['name', 'amenity', 'geometry']])


# Підрахунок кількості закладів за типом
counts = data_cleaned['amenity'].value_counts()

print("Розподіл медичних закладів за типами:")
print(counts)

# Побудова графіка
counts.plot(kind='bar', color='skyblue')
plt.title("Розподіл медичних закладів за типами у Києві")
plt.xlabel("Тип закладу")
plt.ylabel("Кількість")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Підрахунок кількості закладів за районами Києва
locator = Nominatim(user_agent="medical_map", timeout=5)


# Функція для визначення району
def get_borough(lat, lon):
    try:
        location_ = locator.reverse((lat, lon), exactly_one=True)
        if location_:
            address = location_.raw['address']
            return address['borough']
        return "Unknown"
    except GeocoderTimedOut:
        return "Timeout"


# Додавання районів до кожного запису
boroughs = []
for _, row in data_cleaned.iterrows():
    borough = get_borough(row['latitude'], row['longitude'])
    boroughs.append(borough)
    time.sleep(0.5)  # Щоб не перевищити ліміт запитів

data_cleaned['borough'] = boroughs

# Перевірка результату
print(data_cleaned[['name', 'latitude', 'longitude', 'borough']].head())

# Групування за районами та підрахунок кількості
borough_counts = data_cleaned['borough'].value_counts()

# Вивід результатів
print("Кількість медичних закладів за районами Києва:")
print(borough_counts)

# Побудова графіка
borough_counts.plot(kind='bar', color='green')
plt.title("Кількість медичних закладів за районами Києва")
plt.xlabel("Район")
plt.ylabel("Кількість")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Відображення даних на карті
location = locator.geocode(city)
center_coords = [location.latitude, location.longitude]

# Створення карти
map_kyiv = folium.Map(location=center_coords, zoom_start=12)

# Додавання точок
for _, row in data_cleaned.iterrows():
    coords = (row['latitude'], row['longitude'])
    name = f"{row['name']} ({row['amenity']})"
    folium.Marker(
        location=coords,
        popup=name,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(map_kyiv)

# Збереження карти та її відкриття у браузері
map_kyiv.save("kyiv_medical_facilities.html")
map_kyiv.show_in_browser()
