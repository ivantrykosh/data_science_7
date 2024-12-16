"""
Кластеризація за кольором водойм на зображенні з використанням методу k-means
https://www.bing.com/maps?cp=51.524207%7E23.858193&lvl=12.3&style=a
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(image):
    """ Показ зображення """
    plt.imshow(image)
    plt.show()


def image_read(file_image):
    """ Зчитування зображення з файлу """
    image = cv2.imread(file_image)
    image_in_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_in_rgb


def image_processing(image):
    """ Обробка зображення """
    # Застосовуємо Гаусівське розмиття
    blurred_image = cv2.GaussianBlur(image, (5, 5), 5)
    # Виділяємо темні області на зображенні
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2HSV)
    _, _, v_channel = cv2.split(hsv_image)
    _, binary_image = cv2.threshold(v_channel, 40, 255, cv2.THRESH_BINARY_INV)
    # Перетворюємо маски в трьох канальне зображення
    binary_mask_3channel = cv2.merge([binary_image, binary_image, binary_image])
    # Застосовуємо маску до основного зображення
    result = cv2.bitwise_and(blurred_image, binary_mask_3channel)
    # Застосовуємо згладжування
    result = cv2.medianBlur(result, 9)
    return result


def save_result(file_name, image_in_rgb):
    """ Збереження результату у файл """
    image_in_bgr = cv2.cvtColor(image_in_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, image_in_bgr)


def clustering(image):
    """ Кластеризація з допомогою k-means """
    # Перетворюємо зображення у двовимірний масив
    two_dimension = image.reshape((-1, 3))
    two_dimension = np.float32(two_dimension)
    # Критерії зупинки для алгоритму k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Кількість кластерів та спроб
    k = 2
    attempts = 10
    # Застосовуємо алгоритм k-means
    ret, label, center = cv2.kmeans(two_dimension, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    # Отримуємо результати та відновлюємо форму зображення
    res = center[label.flatten()]
    edged_img = res.reshape(image.shape)
    return edged_img


image_entrance = image_read('lakes.jpg')
show_image(image_entrance)
image_exit = image_processing(image_entrance)
show_image(image_exit)
img_clustered = clustering(image_exit)
show_image(img_clustered)
save_result('lakes_result.jpg', img_clustered)
