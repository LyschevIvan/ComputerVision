import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def erosion_opencv(image):
    """
    Применяет эрозию к изображению используя OpenCV
    """
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def erosion_native(image):
    """
    Применяет эрозию к изображению используя нативную реализацию
    """
    # Создаем выходное изображение
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.uint8)
    
    # Проходим по каждому пикселю изображения (кроме границ)
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Получаем окно 3x3 вокруг текущего пикселя
            window = image[i-1:i+2, j-1:j+2]
            # Применяем операцию эрозии (минимум в окне)
            output[i, j] = np.min(window)
    
    return output

def compare_methods(image_path):
    """
    Сравнивает время выполнения и результаты обоих методов
    """
    # Загружаем изображение и конвертируем в градации серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Применяем пороговую обработку для получения бинарного изображения
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Замеряем время для OpenCV реализации
    start_time = time.time()
    opencv_result = erosion_opencv(binary)
    opencv_time = time.time() - start_time
    
    # Замеряем время для нативной реализации
    start_time = time.time()
    native_result = erosion_native(binary)
    native_time = time.time() - start_time
    
    # Отображаем результаты
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(binary, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(opencv_result, cmap='gray')
    plt.title(f'OpenCV\n{opencv_time:.4f} сек')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(native_result, cmap='gray')
    plt.title(f'Нативная реализация\n{native_time:.4f} сек')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return opencv_time, native_time

if __name__ == "__main__":
    # Путь к тестовому изображению
    image_path = "test_image.jpg"  # Замените на путь к вашему изображению
    
    try:
        opencv_time, native_time = compare_methods(image_path)
        print(f"\nВремя выполнения:")
        print(f"OpenCV: {opencv_time:.4f} секунд")
        print(f"Нативная реализация: {native_time:.4f} секунд")
        print(f"Разница в скорости: {native_time/opencv_time:.2f}x")
    except Exception as e:
        print(f"Ошибка при обработке изображения: {str(e)}")
