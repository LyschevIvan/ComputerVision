import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 5  # количество классов цветов

def create_model():
    # Загрузка предобученной ResNet-18
    model = models.resnet18(pretrained=True)
    
    # Замораживаем веса базовой модели
    for param in model.parameters():
        param.requires_grad = False
    
    # Заменяем последний слой на новый для нашей задачи
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        # Обучение
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Эпоха {epoch+1}/{NUM_EPOCHS}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Эпоха [{epoch+1}/{NUM_EPOCHS}]')
        print(f'Ошибка при обучении: {epoch_loss:.4f}, Точность обучения: {epoch_acc:.2f}%')
        print(f'Ошибка валидации: {val_loss:.4f}, Точность валидации: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, save_dir='results'):
    # Создаем директорию для результатов, если её нет
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Ошибка обучения')
    plt.plot(val_losses, label='Ошибка валидации')
    plt.title('Ошибка vs. Эпоха')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Точность обучения')
    plt.plot(val_accuracies, label='Точность валидации')
    plt.title('Точность vs. Эпоха')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png')
    plt.close()

    # Сохраняем значения в CSV
    df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies
    })
    df.to_csv(f'{save_dir}/training_history.csv', index=False)

def evaluate_model(model, test_loader, criterion, save_dir='results'):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_names = test_loader.dataset.dataset.classes if hasattr(test_loader.dataset, 'dataset') else test_loader.dataset.classes
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Тестирование'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Вычисляем метрики
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    # Создаем confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()
    
    # Создаем отчет о классификации
    report = classification_report(all_labels, all_preds,
                                 target_names=class_names,
                                 output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{save_dir}/classification_report.csv')
    
    # Сохраняем общие результаты
    with open(f'{save_dir}/test_results.txt', 'w') as f:
        f.write(f'Ошибка на тестовой выборке: {test_loss:.4f}\n')
        f.write(f'Точность на тестовой выборке: {test_accuracy:.2f}%\n')
    
    return test_loss, test_accuracy

def main():
    # Проверяем наличие датасета
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("Ошибка: Директория 'dataset' не найдена!")
        print("Пожалуйста, создайте директорию 'dataset' и поместите в неё папки с изображениями цветов")
        print("Структура директорий должна быть следующей:")
        print("dataset/")
        print("    daisy/")
        print("    dandelion/")
        print("    rose/")
        print("    sunflower/")
        print("    tulip/")
        return
    
    # Преобразования для изображений
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка датасета
    full_dataset = ImageFolder(root='dataset', transform=transform)
    
    # Разделение на обучающую, валидационную и тестовую выборки
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Создание модели и настройка обучения
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # Обучение модели
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer)
    
    # Сохранение результатов обучения
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Оценка модели на тестовой выборке
    print("\nОценка модели на тестовой выборке...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(f"Ошибка на тестовой выборке: {test_loss:.4f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.2f}%")
    
    # Сохранение модели
    torch.save(model.state_dict(), 'results/flower_classifier.pth')
    print("\nМодель и результаты сохранены в директории 'results'")

if __name__ == '__main__':
    main()
