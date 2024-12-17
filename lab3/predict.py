import torch
import torchvision.transforms as transforms
from PIL import Image
from train import create_model
import json

# Загрузка классов
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def load_model(model_path):
    model = create_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return predicted.item(), probabilities[0]

def main(image_path, model_path='flower_classifier.pth'):
    # Загрузка модели
    model = load_model(model_path)
    
    # Обработка изображения
    image_tensor = process_image(image_path)
    
    # Получение предсказания
    predicted_idx, probabilities = predict(model, image_tensor)
    
    # Вывод результатов
    print(f'Предсказанный класс: {class_names[predicted_idx]}')
    print('\nВероятности по классам:')
    for i, prob in enumerate(probabilities):
        print(f'{class_names[i]}: {prob.item():.2%}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Использование: python predict.py <путь_к_изображению>')
    else:
        main(sys.argv[1])
