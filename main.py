#!/usr/bin/env python3
"""
🎨 GenImg - AI Character Dataset Generator

Головна точка входу для системи генерації датасетів персонажів.
Підтримує як CLI режим, так і використання як Python модуль.

Використання:
    python main.py [command] [options]
    python main.py env-check
    python main.py generate --reference image.jpg --count 15
    python main.py analyze --dataset

Для довідки:
    python main.py --help
"""

import sys
import os
from pathlib import Path

# Додаємо поточну директорію до шляху для імпортів
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from src.cli import CLIManager
except ImportError as e:
    print(f"❌ Помилка імпорту: {e}")
    print("Переконайтеся, що всі залежності встановлені:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Головна функція для CLI використання."""
    print("🎨 GenImg - AI Character Dataset Generator")
    print("=" * 50)
    
    # Перевірка Python версії
    if sys.version_info < (3, 8):
        print("❌ Потрібна версія Python 3.8 або вища")
        print(f"Поточна версія: {sys.version}")
        sys.exit(1)
    
    # Показуємо довідку якщо немає аргументів
    if len(sys.argv) == 1:
        print("\n🆘 Використання:")
        print("  python main.py [command] [options]")
        print("\n📖 Доступні команди:")
        print("  env-check      - Перевірка середовища")
        print("  generate       - Генерація зображень")
        print("  analyze        - Аналіз датасету")
        print("  estimate       - Оцінка вартості")
        print("  export         - Експорт даних")
        print("  models         - Управління моделями")
        print("  validate       - Валідація")
        print("\n💡 Для детальної довідки:")
        print("  python main.py --help")
        print("  python main.py [command] --help")
        print()
        return 0
    
    try:
        # Створюємо та запускаємо CLI менеджер
        cli = CLIManager()
        return cli.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Операцію перервано користувачем")
        return 1
    except Exception as e:
        print(f"\n❌ Критична помилка: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 