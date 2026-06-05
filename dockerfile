FROM python:3.11-slim

# Установка системных зависимостей для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модели
COPY predict.py .
COPY scaler.pkl .
COPY classifier.pkl .

# Предварительно загружаем веса ConvNeXt (будут закэшированы в образе)
RUN python -c "from torchvision import models; models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)"

# Точка входа
ENTRYPOINT ["python", "predict.py"]