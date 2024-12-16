from geopy.geocoders import Nominatim
import osmnx as ox
import folium

# Початкові параметри
city = "Lviv, Ukraine"
tags = {"amenity": ["cafe"]}  # Кафе

# Завантаження даних
data = ox.features_from_place(city, tags)
data = data.reset_index()

# Вибір основної інформації
selected_columns = ['id', 'name', 'geometry']
data_cleaned = data[selected_columns]
# Очищення даних
data_cleaned = data_cleaned.loc[data_cleaned['geometry'].geom_type == 'Point']
data_cleaned = data_cleaned.loc[~data_cleaned['name'].isna()]

print(data_cleaned)


# Відображення даних на карті
locator = Nominatim(user_agent="cafes_map")
location = locator.geocode(city)
center_coords = [location.latitude, location.longitude]

# Створення карти
map_lviv = folium.Map(location=center_coords, zoom_start=14)

# Додавання точок
for _, row in data_cleaned.iterrows():
    coords = (row['geometry'].y, row['geometry'].x)
    folium.Marker(
        location=coords,
        popup=row['name'],
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(map_lviv)

# Збереження карти та її відкриття у браузері
map_lviv.save("lviv_cafes.html")
map_lviv.show_in_browser()
