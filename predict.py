#!/usr/bin/env python3
import os
import sys
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import joblib

# ------------------------------
# Конфигурация
N_FRAMES = 8
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------
# Сортировка имён файлов
def numerical_sort_key(filename):
    nums = re.findall(r'\d+', filename)
    return int(''.join(nums)) if nums else filename

# ------------------------------
# Загрузка модели-экстрактора
extractor = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
extractor.classifier[-1] = nn.Identity()
extractor.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------
# ИДЕНТИЧНАЯ ЛОГИКА АГРЕГАЦИИ (как в dataset-prepare)
def extract_features_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(files) < N_FRAMES:
        raise ValueError(f"Мало кадров: {len(files)}")
    files.sort(key=numerical_sort_key)
    
    # Равномерная выборка
    idx = np.linspace(0, len(files)-1, N_FRAMES, dtype=int)
    selected = [files[i] for i in idx]
    
    feats = []
    with torch.no_grad():
        for fname in selected:
            img = cv2.cvtColor(cv2.imread(os.path.join(folder_path, fname)), cv2.COLOR_BGR2RGB)
            t = transform(img).unsqueeze(0).to(DEVICE)
            feats.append(extractor(t).cpu().numpy().flatten())
            
    feats = np.array(feats) # (8, 768)
    
    # Агрегация (mean + std + delta)
    mean_f = np.mean(feats, axis=0)
    std_f = np.std(feats, axis=0)
    delta_f = np.mean(np.abs(np.diff(feats, axis=0)), axis=0)
    
    # Вектор (2304,)
    return np.concatenate([mean_f, std_f, delta_f]).reshape(1, -1)

# ------------------------------
# Загрузка моделей
scaler = joblib.load('scaler.pkl')
classifier = joblib.load('classifier.pkl')
idx_to_label = {0: 'inaction', 1: 'move', 2: 'work'}

def main():
    if len(sys.argv) < 2: sys.exit(1)
    folder = sys.argv[1]
    
    X = extract_features_from_folder(folder)
    X_scaled = scaler.transform(X)
    pred_idx = classifier.predict(X_scaled)[0]
    print(idx_to_label[pred_idx])

if __name__ == '__main__':
    main()