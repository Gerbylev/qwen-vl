# Qwen2.5-VL Binary Classifier

Модифицированная версия Qwen2.5-VL для бинарной классификации (классы 0 и 1) вместо генерации текста.

## Описание

Этот проект адаптирует архитектуру Qwen2.5-VL для задач бинарной классификации:
- Загружает предобученные веса от оригинальной модели Qwen2.5-VL
- Заменяет языковую голову (lm_head) на простой классификатор 
- Позволяет проводить inference с изображением и текстом
- Поддерживает дообучение только классификатора или всей модели

## Файлы проекта

- `qwen2_5_vl_classifier.py` - Основной класс модели и тренера
- `test_classifier.py` - Тесты функциональности модели
- `example_usage.py` - Примеры использования модели
- `README.md` - Данная документация

## Установка зависимостей

```bash
pip install torch torchvision transformers Pillow requests numpy
```

## Быстрый старт

### 1. Инициализация модели

```python
from qwen2_5_vl_classifier import Qwen2_5_VLForBinaryClassification

# Создание модели
model = Qwen2_5_VLForBinaryClassification(
    model_name="Qwen/Qwen2-VL-2B-Instruct"
)

# Заморозка базовой модели (рекомендуется для начала)
model.freeze_base_model()
```

### 2. Предсказание

```python
from PIL import Image

# Загрузка изображения
image = Image.open("path/to/your/image.jpg")
text = "Ваш вопрос или описание"

# Предсказание класса (0 или 1)
predicted_class = model.predict(image, text, return_probs=False)
print(f"Предсказанный класс: {predicted_class}")

# Получение вероятностей
probabilities = model.predict(image, text, return_probs=True)
print(f"Вероятности: [Класс 0: {probabilities[0]:.3f}, Класс 1: {probabilities[1]:.3f}]")
```

### 3. Обучение

```python
from qwen2_5_vl_classifier import Qwen2_5_VLTrainer

# Создание тренера
trainer = Qwen2_5_VLTrainer(model, learning_rate=1e-4)

# Подготовка батча (пример структуры)
batch = {
    'input_ids': torch.tensor(...),      # Токенизированный текст
    'attention_mask': torch.tensor(...), # Маска внимания  
    'pixel_values': torch.tensor(...),   # Пиксели изображений
    'labels': torch.tensor([0, 1, 0])    # Бинарные метки
}

# Один шаг обучения
loss = trainer.train_step(batch)
print(f"Потери: {loss}")
```

### 4. Сохранение и загрузка весов

```python
# Сохранение весов классификатора
model.save_classifier_weights("my_classifier.pth")

# Загрузка обученных весов
model.load_pretrained_weights("my_classifier.pth")
```

## Архитектура модели

```
Qwen2.5-VL Base Model (заморожена)
        ↓
   Hidden States
   (размерность: hidden_size)
        ↓
   Классификатор:
   Linear(hidden_size → 512)
        ↓
      ReLU + Dropout(0.1)
        ↓
   Linear(512 → 2)  # 2 класса: 0 и 1
```

## Особенности реализации

### Заморозка/разморозка модели

```python
# Заморозить базовую модель (обучается только классификатор)
model.freeze_base_model()

# Разморозить всю модель
model.unfreeze_base_model()
```

### Получение представлений

Модель использует средние скрытые состояния по всем токенам (с учетом маски внимания) для классификации, что более эффективно чем использование только CLS токена.

## Тестирование

Запустите тесты для проверки корректности работы:

```bash
python test_classifier.py
```

Тесты проверяют:
- ✅ Инициализацию модели
- ✅ Предсказания на реальных изображениях
- ✅ Сохранение/загрузку весов
- ✅ Заморозку/разморозку параметров
- ✅ Создание демо датасета

## Примеры использования

Запустите полный пример:

```bash
python example_usage.py
```

## Рекомендации по обучению

### Этап 1: Обучение классификатора

1. Заморозьте базовую модель:
   ```python
   model.freeze_base_model()
   ```

2. Обучите только классификатор с высоким learning rate (1e-3 - 1e-4)

3. Сохраните веса:
   ```python
   model.save_classifier_weights("stage1_weights.pth")
   ```

### Этап 2: Fine-tuning всей модели (опционально)

1. Загрузите обученные веса классификатора
2. Разморозьте базовую модель:
   ```python
   model.unfreeze_base_model()
   ```

3. Обучите всю модель с низким learning rate (1e-5 - 1e-6)

## Структура данных для обучения

```python
training_examples = [
    {
        "image": Image.open("positive_example.jpg"),
        "text": "Описание или вопрос",
        "label": 1  # Положительный класс
    },
    {
        "image": Image.open("negative_example.jpg"),
        "text": "Описание или вопрос", 
        "label": 0  # Отрицательный класс
    }
]
```

## Технические детали

- **Базовая модель**: Qwen2.5-VL (любая размерность)
- **Предобученные веса**: Автоматически загружаются из HuggingFace
- **Классификатор**: 2-слойная нейронная сеть
- **Функция потерь**: CrossEntropyLoss
- **Поддержка**: GPU/CPU, mixed precision

## Решение проблем

### Ошибки памяти
- Уменьшите batch size
- Используйте gradient checkpointing
- Заморозьте базовую модель

### Плохое качество
- Увеличьте количество эпох обучения
- Попробуйте разные learning rates
- Используйте data augmentation
- Разморозьте базовую модель на втором этапе

### Ошибки загрузки модели
- Проверьте доступ к интернету для загрузки из HuggingFace
- Убедитесь в корректности имени модели
- Проверьте версии transformers и torch

## Лицензия

Код предоставляется "как есть" в образовательных целях. Используйте в соответствии с лицензиями оригинальных моделей Qwen.