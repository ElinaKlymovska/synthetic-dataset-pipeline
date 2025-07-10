# GenImg - AI Character Dataset Generator

🎨 Модульна система для генерації різноманітних датасетів зображень персонажів для тренування LoRA з використанням AI генерації зображень.

## ✨ Основні можливості

- 🤖 **Підтримка декількох генераторів**: Replicate API, локальний Stable Diffusion
- 🎯 **Інтелектуальна генерація**: Автоматично створює різноманітні варіації персонажів
- 📊 **Аналіз якості**: Оцінка придатності зображень для LoRA тренування
- 💾 **Управління метаданими**: Структурована система збереження метаданих
- 📤 **Експорт даних**: JSON, CSV формати для інтеграції
- 🔧 **CLI інтерфейс**: Зручна командна строка з українською локалізацією
- 🏗️ **Модульна архітектура**: Легко розширювана система компонентів

## 🚀 Швидкий старт

### Встановлення

```bash
# Клонування репозиторію
git clone <repo-url>
cd GenImg

# Встановлення залежностей
pip install -r requirements.txt

# Налаштування змінних середовища
export REPLICATE_API_TOKEN="your_token_here"
```

### Базове використання

```bash
# Перевірка середовища
python main.py env-check

# Генерація датасету
python main.py generate --reference data/input/character.jpg --count 15

# Оцінка вартості
python main.py estimate --count 20

# Аналіз результатів
python main.py analyze --dataset

# Експорт даних
python main.py export --format json
```

### Використання як Python модуль

```python
import asyncio
from src import GenImgApp

async def main():
    async with GenImgApp() as app:
        results = await app.generate_character_dataset(
            reference_image="path/to/character.jpg",
            count=10,
            output_dir="data/output"
        )
        print(f"Generated {results['successful']} images")

asyncio.run(main())
```

## 📁 Структура проекту

```
GenImg/
├── src/                    # Основний код
│   ├── core/              # Основні компоненти
│   │   ├── interfaces.py  # Абстракції та протоколи
│   │   ├── config.py      # Система конфігурації
│   │   └── services.py    # Основні сервіси
│   ├── generators/        # Генератори зображень
│   │   ├── replicate.py   # Replicate API генератор
│   │   ├── local.py       # Локальний генератор
│   │   └── batch.py       # Пакетна обробка
│   ├── cli/              # Командний інтерфейс
│   │   ├── manager.py     # CLI менеджер
│   │   └── commands.py    # Реалізація команд
│   ├── utils/            # Утиліти
│   │   ├── image.py       # Робота з зображеннями
│   │   ├── file.py        # Файлові операції
│   │   └── data.py        # Експорт та аналіз
│   └── app.py            # Головний додаток
├── tools/                # Інструменти та скрипти
├── data/                 # Дані проекту
├── config/               # Конфігурації
├── docs/                 # Документація
└── main.py              # Точка входу
```

## 🛠️ CLI команди

### Генерація зображень
```bash
python main.py generate \
  --reference data/input/character.jpg \
  --count 15 \
  --output-dir data/output \
  --model sdxl \
  --guidance-scale 7.5
```

### Аналіз та статистика
```bash
# Аналіз датасету
python main.py analyze --dataset --detailed

# Аналіз конкретного зображення
python main.py analyze --image path/to/image.jpg

# Валідація
python main.py validate --image path/to/image.jpg
```

### Експорт та управління
```bash
# Експорт в JSON
python main.py export --format json --output dataset.json

# Список моделей
python main.py models --list

# Інформація про модель
python main.py models --info sdxl
```

## ⚙️ Конфігурація

Створіть файл `config/app_config.yaml`:

```yaml
generation:
  target_count: 15
  width: 768
  height: 768
  steps: 25
  guidance_scale: 7.5
  strength: 0.8

output:
  output_dir: "data/output"
  metadata_dir: "data/metadata"
  create_archive: true

models:
  sdxl:
    model_id: "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
    cost_per_image: 0.075
    max_batch_size: 10

api:
  timeout: 300
  retry_attempts: 3
  retry_delay: 5
```

## 🔧 Розширення системи

### Додавання нового генератора

```python
from src.core import ImageGenerator

class CustomGenerator(ImageGenerator):
    async def generate_image(self, reference_image, config, output_path):
        # Ваша логіка генерації
        pass
```

### Додавання нової CLI команди

```python
def custom_command(self, args):
    """Handle custom command."""
    # Ваша логіка команди
    pass
```

## 📊 Метадані та аналіз

Система автоматично зберігає детальні метадані про кожне згенероване зображення:

- Параметри генерації
- Інформація про модель
- Статистика якості
- Придатність для LoRA тренування
- Класифікація поз та одягу

## 🤝 Внесок у проект

1. Форкніть репозиторій
2. Створіть feature гілку
3. Внесіть зміни
4. Додайте тести
5. Створіть Pull Request

## 📄 Ліцензія

MIT License - деталі в файлі LICENSE.

## 🆘 Підтримка

- GitHub Issues для повідомлення про помилки
- Документація в папці `docs/`
- Приклади використання в `examples/`

---

🎨 **GenImg** - створюйте високоякісні датасети для LoRA тренування з легкістю! 