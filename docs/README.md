# 🎨 Character Dataset Generator

Генератор різноманітних зображень персонажа з одного референтного зображення за допомогою Replicate.com API.

## ✨ Особливості

- 🚀 **Простий API** - не потрібно налаштовувати сервери
- 💰 **Pay-per-use** - платите тільки за генерації (~$0.075/зображення)
- 🎯 **Висока якість** - використовує Stable Diffusion XL
- ⚡ **Швидко** - результат за 30-60 секунд
- 🎨 **Різноманітність** - 10+ стилів та фонів

## 🚀 Швидкий старт

### 1. Встановлення

```bash
# Клонування репозиторію
git clone <your-repo>
cd GenImg

# Встановлення залежностей
pip install -r requirements.txt
```

### 2. Налаштування API

```bash
# Отримайте токен на https://replicate.com/account/api-tokens
export REPLICATE_API_TOKEN="r8_your_token_here"
```

### 3. Генерація

```bash
# Перевірка API
python -m scripts.replicate_generate --check-api

# Оцінка вартості
python -m scripts.replicate_generate --estimate-cost --count 15

# Генерація зображень
python -m scripts.replicate_generate --reference data/input/character_reference.jpg --count 15
```

## 📁 Структура проєкту

```
GenImg/
├── config/
│   └── replicate_config.yaml    # Конфігурація API та параметрів
├── src/
│   ├── replicate_generator.py   # Основний генератор
│   └── utils.py                 # Допоміжні функції
├── scripts/
│   ├── replicate_generate.py    # CLI скрипт
│   └── validate_dataset.py      # Валідація результатів
├── data/
│   ├── input/                   # Референтні зображення
│   └── output/                  # Згенеровані зображення
└── requirements.txt             # Python залежності
```

## ⚙️ Конфігурація

Основні параметри в `config/replicate_config.yaml`:

```yaml
generation:
  target_count: 15
  img2img:
    denoising_strength: 0.6    # Сила зміни (0.3-0.8)
    cfg_scale: 8.0             # Дотримання промпту
    steps: 30                  # Кроки генерації
    width: 1024                # Ширина
    height: 1024               # Висота
```

## 💰 Вартість

- **SDXL**: ~$0.075/зображення
- **15 зображень**: ~$1.12
- **50 зображень**: ~$3.75

## 🎯 Приклади використання

### Базова генерація
```bash
python -m scripts.replicate_generate --reference my_character.jpg --count 10
```

### З детальним логуванням
```bash
python -m scripts.replicate_generate --verbose --count 5
```

### Кастомна директорія виводу
```bash
python -m scripts.replicate_generate --output-dir my_output --count 15
```

## 📊 Результати

Після генерації ви отримаєте:
- `data/output/character_001.png` - `character_015.png`
- `data/output/metadata.json` - метадані з промптами
- `character_dataset.zip` - архів для зручності

## 🔧 Розв'язання проблем

### Помилка "API token not set"
```bash
export REPLICATE_API_TOKEN="r8_your_token_here"
```

### Повільна генерація
- Це нормально, кожне зображення займає 30-60 секунд
- API має ліміт 10 запитів/хвилину

### Помилка "File too large"
- Зменшіть розмір референтного зображення до 1024x1024

## 📚 Документація

Детальний гід: [REPLICATE_SETUP_GUIDE.md](REPLICATE_SETUP_GUIDE.md)

## 🤝 Підтримка

- [Replicate Documentation](https://replicate.com/docs)
- [API Tokens](https://replicate.com/account/api-tokens)
- [Pricing](https://replicate.com/pricing)

---

**Готово для LoRA тренування!** 🎉 