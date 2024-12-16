"""
Підрахунок кількості об'єктів на зображенні
"""

import cv2
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
    # Збільшення контрасту
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v_eq = cv2.equalizeHist(v)
    hsv_eq_image = cv2.merge([h, s, v_eq])
    contrasted_image = cv2.cvtColor(hsv_eq_image, cv2.COLOR_HSV2RGB)
    # Застосування Гаусівського розмиття
    blurred = cv2.GaussianBlur(contrasted_image, (9, 9), 10)
    # Виділення світлих областей
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    _, _, v_channel = cv2.split(hsv_image)
    _, binary_image = cv2.threshold(v_channel, 180, 255, cv2.THRESH_BINARY_INV)
    # Перетворення маски в трьох канальне зображення
    binary_mask_3channel = cv2.merge([binary_image, binary_image, binary_image])
    # Застосування маски до основного зображення
    image_with_mask = cv2.bitwise_and(blurred, binary_mask_3channel)
    # Застосування згладжування
    image_with_mask = cv2.medianBlur(image_with_mask, 7)
    show_image(image_with_mask)

    # Визначення кутів на зображенні та морфологічне закриття
    edged = cv2.Canny(image_with_mask, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    return closed


def image_contours(image):
    """ Знаходження контурів """
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def save_result(file_name, image_in_rgb):
    """ Збереження результату у файл """
    image_in_bgr = cv2.cvtColor(image_in_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, image_in_bgr)


def image_recognition(image, contours):
    """ Розпізнавання об'єктів """
    total_melons = 0
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        if cv2.contourArea(contour) > 5000:
            if 3 <= len(approx) <= 5:
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
                total_melons += 1
    return image, total_melons


image_entrance = image_read("watermelons.jpg")
show_image(image_entrance)
image_exit = image_processing(image_entrance)
show_image(image_exit)
image_contours = image_contours(image_exit)
result, total_figures = image_recognition(image_entrance, image_contours)
print(f"Знайдено {total_figures} об'єктів")
show_image(result)
save_result("watermelons_recognition.jpg", result)
