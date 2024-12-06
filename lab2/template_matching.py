import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageMatcher:
    def __init__(self):
        # Инициализируем SIFT детектор
        self.sift = cv2.SIFT_create()

    def template_matching(self, image, template):
        """
        Метод поиска шаблона на изображении с помощью template matching
        """
        # Конвертируем изображения в оттенки серого
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Получаем размеры шаблона
        h, w = template_gray.shape

        # Применяем template matching
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Находим позицию максимального значения
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Определяем верхний левый и нижний правый углы прямоугольника
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        return [(top_left, bottom_right)], max_val

    def sift_matching(self, image, template):
        """
        Метод поиска объекта на изображении с помощью SIFT
        """
        # Находим ключевые точки и дескрипторы
        kp1, des1 = self.sift.detectAndCompute(template, None)
        kp2, des2 = self.sift.detectAndCompute(image, None)

        # Создаем объект для сопоставления особых точек
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Применяем фильтр Лоу для отбора лучших совпадений
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 4:
            # Получаем координаты соответствующих точек
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            # Находим матрицу гомографии
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Получаем углы шаблона
                h, w = template.shape[:2]
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)

                # Преобразуем координаты углов
                dst = cv2.perspectiveTransform(pts, H)

                # Преобразуем в список точек
                corners = [(tuple(map(int, corner[0]))) for corner in dst]
                return [corners], len(good_matches) / len(matches)

        return [], 0

    def draw_matches(self, image, matches, method="template"):
        """
        Отрисовка результатов поиска
        """
        result = image.copy()

        if method == "template":
            for top_left, bottom_right in matches:
                cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 2)
        else:
            for corners in matches:
                # Рисуем четырехугольник
                for i in range(4):
                    cv2.line(result, corners[i], corners[(i + 1) % 4], (0, 255, 0), 2)

        return result


def process_image_template_pairs(image_template_pairs):
    """
    Функция для тестирования обоих методов
    """
    # Загружаем изображения
    # image = cv2.imread(image_path)
    # template = cv2.imread(template_path)

    # if image is None or template is None:
    #     raise ValueError("Не удалось загрузить изображения")

    matcher = ImageMatcher()
    results = []

    for idx, (image_path, template_path) in enumerate(image_template_pairs):
        print(f"Processing pair {idx + 1}: {image_path}, {template_path}")
        image = cv2.imread(image_path)
        template = cv2.imread(template_path)

        if image is None or template is None:
            print(f"Failed to load: {image_path} or {template_path}")
            continue

        # Perform template matching
        template_matches, template_score = matcher.template_matching(image, template)
        template_result = matcher.draw_matches(image, template_matches, "template")

        # Perform SIFT matching
        sift_matches, sift_score = matcher.sift_matching(image, template)
        sift_result = matcher.draw_matches(image, sift_matches, "sift")

        # Create a new figure for each image-template pair
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Image-Template Pair {idx + 1}", fontsize=16)

        # Display the template
        axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Template")
        axes[0].axis("off")

        # Display original image
        axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Input Image")
        axes[1].axis("off")

        # Display template matching result
        axes[2].imshow(cv2.cvtColor(template_result, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Template Matching\nScore: {template_score:.2f}")
        axes[2].axis("off")

        # Display SIFT matching result
        axes[3].imshow(cv2.cvtColor(sift_result, cv2.COLOR_BGR2RGB))
        axes[3].set_title(f"SIFT Matching\nScore: {sift_score:.2f}")
        axes[3].axis("off")

        # Show the plot
        plt.tight_layout()
        plt.show()

        # print("image:", os.path.basename(image_path), ' temp match:', temp_score, ' sift:', sift_score)

    # plt.tight_layout()
    # plt.show()

    # # Применяем template matching
    # template_matches, template_score = matcher.template_matching(image, template)
    # template_result = matcher.draw_matches(image, template_matches, "template")

    # # Применяем SIFT
    # sift_matches, sift_score = matcher.sift_matching(image, template)
    # sift_result = matcher.draw_matches(image, sift_matches, "sift")

    # # Отображаем результаты
    # plt.figure(figsize=(15, 5))

    # plt.subplot(131)
    # plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    # plt.title('Шаблон')
    # plt.axis('off')

    # plt.subplot(132)
    # plt.imshow(cv2.cvtColor(template_result, cv2.COLOR_BGR2RGB))
    # plt.title(f'Template Matching\nScore: {template_score:.2f}')
    # plt.axis('off')

    # plt.subplot(133)
    # plt.imshow(cv2.cvtColor(sift_result, cv2.COLOR_BGR2RGB))
    # plt.title(f'SIFT Matching\nScore: {sift_score:.2f}')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    # return template_score, sift_score


if __name__ == "__main__":
    # Пример использования
    # image_path = "image.jpg"  # Путь к исходному изображению
    # template_path = "template.jpg"  # Путь к шаблону

    image_template_pairs = [
        ("image1.jpg", "template1.jpg"),
        ("image2.jpg", "template2.png"),
        ("image3.jpeg", "template3.png"),
        ("image4.jpg", "template4.png"),
        ("image9.jpg", "template4.png"),
        ("image5.jpg", "template5.png"),
        ("image6.jpeg", "template6.png"),
        ("image7.webp", "template7.png"),
        ("image10.jpg", "template10.png"),
        ("image11.jpeg", "template10.png"),
    ]

    try:
        process_image_template_pairs(image_template_pairs)
        # print(f"\nРезультаты:")
        # print(f"Template Matching score: {template_score:.4f}")
        # print(f"SIFT Matching score: {sift_score:.4f}")
    except Exception as e:
        print(f"Ошибка при обработке изображений: {str(e)}")
