# 📊 MCP - Система Управління Метаданими

## Що таке MCP?

**MCP (Metadata Collection Protocol)** - це система збору та управління метаданими для згенерованих зображень у проєкті GenImg. Вона створена спеціально для підготовки високоякісних датасетів для тренування LoRA моделей.

## 🚀 Основні можливості

- **📋 Автоматичне збереження метаданих** - після кожної генерації
- **🔍 Аналіз якості зображень** - оцінка придатності для тренування
- **🏷️ Автоматичне тегування** - розпізнавання поз, одягу, стилів
- **📊 Аналіз датасету** - статистика та рекомендації
- **🧹 Очищення дублікатів** - виявлення та видалення
- **📦 Експорт даних** - JSON та CSV формати
- **💾 Backup системи** - автоматичне резервування

## 📁 Структура файлів

```
GenImg/
├── mcp/                          # JSON метадані для кожного зображення
│   ├── image_a1b2c3d4.json     # Метадані згенерованого зображення
│   └── image_e5f6g7h8.json     # ...
├── config/
│   └── mcp_config.yaml          # Конфігурація MCP системи
├── scripts/
│   ├── mcp_json_saver.py        # Основний клас для роботи з метаданими
│   └── mcp_manager.py           # CLI інструмент управління
└── prompts/
    └── prompts.txt              # Структуровані промпти для генерації
```

## 🔧 Налаштування

### 1. Конфігурація

Основні налаштування в `config/mcp_config.yaml`:

```yaml
# Директорії
directories:
  mcp_output: "mcp"              # Папка для JSON файлів
  image_output: "data/output"    # Папка для зображень

# Оцінка якості
quality_assessment:
  enabled: true
  identity_score_weight: 0.4     # Збереження ідентичності (40%)
  resolution_score_weight: 0.3   # Роздільна здатність (30%)
  file_size_score_weight: 0.2    # Розмір файлу (20%)
  format_score_weight: 0.1       # Формат файлу (10%)

# Критерії для тренування
training_suitability:
  min_quality_score: 0.7         # Мінімальна якість
  min_identity_score: 0.8        # Мінімальна схожість
```

### 2. Використання в коді

```python
from scripts.mcp_json_saver import MCPMetadataManager

# Ініціалізація
manager = MCPMetadataManager()

# Створення метаданих
metadata = manager.generate_mcp_entry(
    source_image="data/input/character_reference.jpg",
    generated_image_path="data/output/character_001.png",
    prompt_text="portrait of beautiful woman, elegant dress",
    negative_prompt="blurry, low quality",
    identity_similarity_score=0.92,
    generation_cost=0.075,
    generation_time=45.2
)

# Збереження
manager.save_mcp_json(metadata)
```

## 🛠️ CLI Команди

### Основні команди

```bash
# Аналіз датасету
python scripts/mcp_manager.py analyze

# Перевірка валідності
python scripts/mcp_manager.py validate

# Показати статистику
python scripts/mcp_manager.py summary

# Експорт в JSON
python scripts/mcp_manager.py export dataset.json --format json

# Експорт в CSV
python scripts/mcp_manager.py export dataset.csv --format csv

# Очищення дублікатів
python scripts/mcp_manager.py cleanup

# Створення backup
python scripts/mcp_manager.py backup
```

### Детальні приклади

```bash
# Детальний аналіз з рекомендаціями
python scripts/mcp_manager.py analyze --detailed

# Детальна статистика
python scripts/mcp_manager.py summary --detailed

# Експорт з кастомною назвою
python scripts/mcp_manager.py export "exports/my_dataset_$(date +%Y%m%d).json"
```

## 📊 Структура метаданих

Кожен JSON файл містить:

```json
{
  "image_id": "uuid-строка",
  "timestamp": "2024-01-07T14:30:00",
  "project_name": "GenImg_Dataset",
  
  "source_image": "data/input/character_reference.jpg",
  "generated_image": "data/output/character_001.png",
  "image_hash": "md5-хеш-файлу",
  
  "generation_params": {
    "prompt_text": "portrait of beautiful woman...",
    "negative_prompt": "blurry, low quality...",
    "denoising_strength": 0.4,
    "cfg_scale": 7.5,
    "steps": 30
  },
  
  "lora_training": {
    "suitable_for_training": true,
    "quality_score": 0.85,
    "training_tags": ["front_view", "portrait", "photorealistic"],
    "pose_category": "standing",
    "outfit_category": "dress"
  },
  
  "status": {
    "generation_status": "success",
    "generation_cost": 0.075,
    "generation_time_seconds": 45.2
  }
}
```

## 📈 Аналіз якості

### Критерії оцінки

1. **Збереження ідентичності (40%)** - наскільки зображення схоже на оригінал
2. **Роздільна здатність (30%)** - якість та розмір зображення  
3. **Розмір файлу (20%)** - оптимальний розмір (0.5-5MB)
4. **Формат файлу (10%)** - перевага PNG формату

### Автоматичне тегування

Система автоматично розпізнає:

- **Пози**: `standing`, `sitting`, `lying`, `walking`, `dancing`
- **Одяг**: `dress`, `suit`, `swimwear`, `lingerie`, `casual`
- **Кути**: `front_view`, `side_view`, `back_view`
- **Стилі**: `photorealistic`, `anime_style`, `artistic`

## 🔍 Аналіз датасету

### Приклад виводу

```
📊 MCP DATASET SUMMARY
=======================================
📁 Total images: 15
✅ Successful: 14
❌ Failed: 1
🎯 Training suitable: 12
⭐ Average quality: 0.82

📐 Pose distribution:
  standing: 8
  sitting: 4
  lying: 2

👗 Outfit distribution:
  dress: 6
  suit: 3
  casual: 4

💡 Recommendations:
  • Add more walking poses
  • Consider generating swimwear variations
  • Improve lighting consistency
```

## 🚨 Найкращі практики

### ✅ Рекомендації

1. **Збереження метаданих** - завжди зберігайте після генерації
2. **Регулярний аналіз** - перевіряйте якість датасету
3. **Backup метаданих** - робіть резервні копії
4. **Очищення дублікатів** - регулярно видаляйте дублікати
5. **Контроль якості** - підтримуйте мінімум 0.7 якості

### ❌ Чого уникати

1. Не видаляйте JSON файли без backup
2. Не змінюйте структуру метаданих вручну
3. Не ігноруйте попередження валідації
4. Не накопичуйте дублікати

## 🔧 Налаштування інтеграції

### Автоматичне збереження

```python
# В ReplicateGenerator
from scripts.mcp_json_saver import MCPMetadataManager

class ReplicateGenerator:
    def __init__(self):
        self.mcp_manager = MCPMetadataManager()
    
    def generate_with_metadata(self, ...):
        # Генерація зображення
        image_path = self.generate_image(...)
        
        # Автоматичне збереження метаданих
        metadata = self.mcp_manager.generate_mcp_entry(
            source_image=reference_path,
            generated_image_path=image_path,
            prompt_text=prompt,
            # ... інші параметри
        )
        self.mcp_manager.save_mcp_json(metadata)
```

## 📦 Експорт для LoRA тренування

### JSON формат (повний)

```bash
python scripts/mcp_manager.py export lora_dataset.json
```

Створює повний датасет з усіма метаданими для комплексного аналізу.

### CSV формат (спрощений)

```bash
python scripts/mcp_manager.py export lora_dataset.csv --format csv
```

Створює спрощену таблицю з основними параметрами для швидкого аналізу.

## 🚀 Швидкий старт

1. **Налаштуйте конфігурацію**:
   ```bash
   cp config/mcp_config.yaml.example config/mcp_config.yaml
   ```

2. **Згенеруйте зображення з метаданими**:
   ```bash
   python scripts/replicate_generate.py --count 15
   ```

3. **Перевірте результати**:
   ```bash
   python scripts/mcp_manager.py summary
   ```

4. **Експортуйте датасет**:
   ```bash
   python scripts/mcp_manager.py export my_lora_dataset.json
   ```

## 🤝 Підтримка

При виникненні проблем:

1. Перевірте `logs/mcp_metadata.log`
2. Запустіть валідацію: `python scripts/mcp_manager.py validate`
3. Перевірте конфігурацію в `config/mcp_config.yaml`

---

**Готово для професійного LoRA тренування!** 🎉 