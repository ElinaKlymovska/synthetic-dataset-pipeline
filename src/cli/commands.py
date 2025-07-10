"""
CLI Commands implementation for the image generation system.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..core import GenerationConfig
from ..utils import validate_image_file, log_system_info, export_dataset_json, export_dataset_csv


class Commands:
    """CLI Commands implementation."""
    
    def __init__(self):
        self.app = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def set_app(self, app):
        """Set the application instance."""
        self.app = app
    
    def generate(self, args):
        """Handle generate command."""
        try:
            # Validate reference image
            validation = validate_image_file(args.reference)
            if not validation["valid"]:
                self.logger.error(f"❌ Невалідне референтне зображення:")
                for error in validation["errors"]:
                    self.logger.error(f"  • {error}")
                sys.exit(1)
            
            # Show image info
            info = validation["info"]
            self.logger.info(f"📸 Референтне зображення: {args.reference}")
            self.logger.info(f"  📐 Розмір: {info['width']}x{info['height']}")
            self.logger.info(f"  📄 Формат: {info['format']}")
            self.logger.info(f"  💾 Розмір: {info['file_size_mb']} MB")
            
            # Estimate cost
            cost_info = self.app.estimate_cost(args.count, args.model)
            self.logger.info(f"\n💰 Оцінка вартості:")
            self.logger.info(f"  🔢 Зображень: {args.count}")
            self.logger.info(f"  💵 Загальна вартість: ${cost_info['total_cost']:.2f}")
            
            # Confirm generation
            if not args.no_confirm:
                confirm = input(f"\n🤔 Продовжити генерацію {args.count} зображень? (y/N): ")
                if confirm.lower() not in ["y", "yes", "так", "т"]:
                    self.logger.info("🛑 Генерацію скасовано")
                    return
            
            # Set up generation config
            config_overrides = {}
            if args.model:
                config_overrides["model_name"] = args.model
            if args.denoising_strength:
                config_overrides["strength"] = args.denoising_strength
            if args.guidance_scale:
                config_overrides["guidance_scale"] = args.guidance_scale
            if args.steps:
                config_overrides["steps"] = args.steps
            if args.width:
                config_overrides["width"] = args.width
            if args.height:
                config_overrides["height"] = args.height
            if args.seed:
                config_overrides["seed"] = args.seed
            
            # Generate images
            self.logger.info(f"\n🚀 Розпочинаємо генерацію...")
            
            results = self.app.generate_character_variations(
                reference_image=args.reference,
                count=args.count,
                output_dir=args.output_dir,
                config_overrides=config_overrides
            )
            
            # Show results
            successful = len([r for r in results if r.success])
            failed = len(results) - successful
            
            self.logger.info(f"\n✅ Генерацію завершено!")
            self.logger.info(f"  🎯 Успішно: {successful}/{args.count}")
            self.logger.info(f"  💰 Витрачено: ${sum(r.cost or 0 for r in results):.2f}")
            
            if failed > 0:
                self.logger.warning(f"⚠️ {failed} зображень не вдалося згенерувати")
                
        except Exception as e:
            self.logger.error(f"❌ Помилка генерації: {e}")
            sys.exit(1)
    
    def estimate(self, args):
        """Handle estimate command."""
        try:
            cost_info = self.app.estimate_cost(args.count, args.model)
            
            self.logger.info("💰 Оцінка вартості:")
            self.logger.info(f"  🔢 Кількість зображень: {cost_info['count']}")
            self.logger.info(f"  🤖 Модель: {cost_info['model']}")
            self.logger.info(f"  💵 Вартість за зображення: ${cost_info['cost_per_image']:.3f}")
            self.logger.info(f"  💸 Загальна вартість: ${cost_info['total_cost']:.2f} {cost_info['currency']}")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка оцінки: {e}")
            sys.exit(1)
    
    def analyze(self, args):
        """Handle analyze command."""
        if args.dataset:
            try:
                summary = self.app.get_dataset_summary()
                
                self.logger.info("📊 Аналіз датасету:")
                self.logger.info(f"  📁 Всього зображень: {summary['total_images']}")
                self.logger.info(f"  ✅ Успішних: {summary['successful_generations']}")
                self.logger.info(f"  ❌ Невдалих: {summary['failed_generations']}")
                self.logger.info(f"  🎯 Придатних для тренування: {summary['training_suitable']}")
                self.logger.info(f"  ⭐ Середня якість: {summary['average_quality_score']:.2f}")
                self.logger.info(f"  💰 Загальна вартість: ${summary['total_cost']:.2f}")
                
                if args.detailed:
                    self.logger.info("\n🎭 Розподіл по позах:")
                    for pose, count in summary['pose_distribution'].items():
                        self.logger.info(f"  {pose}: {count}")
                    
                    self.logger.info("\n👗 Розподіл по одягу:")
                    for outfit, count in summary['outfit_distribution'].items():
                        self.logger.info(f"  {outfit}: {count}")
                
            except Exception as e:
                self.logger.error(f"❌ Помилка аналізу: {e}")
                sys.exit(1)
        
        elif args.image:
            try:
                validation = validate_image_file(args.image)
                
                self.logger.info(f"🔍 Аналіз зображення: {args.image}")
                self.logger.info(f"  ✅ Валідне: {'Так' if validation['valid'] else 'Ні'}")
                
                info = validation['info']
                self.logger.info(f"  📐 Розмір: {info['width']}x{info['height']}")
                self.logger.info(f"  📄 Формат: {info['format']}")
                self.logger.info(f"  💾 Розмір файлу: {info['file_size_mb']} MB")
                self.logger.info(f"  📊 Мегапікселі: {info['megapixels']}")
                
                if validation['errors']:
                    self.logger.warning("❌ Проблеми:")
                    for issue in validation['errors']:
                        self.logger.warning(f"  • {issue}")
                
                if validation['warnings']:
                    self.logger.warning("⚠️ Попередження:")
                    for warning in validation['warnings']:
                        self.logger.warning(f"  • {warning}")
                
            except Exception as e:
                self.logger.error(f"❌ Помилка аналізу зображення: {e}")
                sys.exit(1)
        
        else:
            self.logger.error("❌ Вкажіть --dataset або --image для аналізу")
            sys.exit(1)
    
    def export(self, args):
        """Handle export command."""
        try:
            if not args.output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.output = f"exports/dataset_{timestamp}.{args.format}"
            
            export_path = self.app.export_dataset(args.output, args.format)
            self.logger.info(f"📤 Дані експортовано: {export_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка експорту: {e}")
            sys.exit(1)
    
    def models(self, args):
        """Handle models command."""
        try:
            if args.list:
                models = self.app.get_available_models()
                self.logger.info("🤖 Доступні моделі:")
                for model in models:
                    self.logger.info(f"  • {model}")
            
            elif args.info:
                model_info = self.app.get_model_info(args.info)
                if model_info:
                    self.logger.info(f"🤖 Інформація про модель '{args.info}':")
                    self.logger.info(f"  📝 Назва: {model_info['name']}")
                    self.logger.info(f"  🆔 ID: {model_info['model_id']}")
                    self.logger.info(f"  💰 Вартість: ${model_info['cost_per_generation']:.3f}")
                    self.logger.info(f"  📦 Розмір батчу: {model_info['max_batch_size']}")
                else:
                    self.logger.error(f"❌ Модель '{args.info}' не знайдено")
                    sys.exit(1)
            
            else:
                self.logger.error("❌ Вкажіть --list або --info")
                sys.exit(1)
                
        except Exception as e:
            self.logger.error(f"❌ Помилка: {e}")
            sys.exit(1)
    
    def validate(self, args):
        """Handle validate command."""
        if args.image:
            try:
                validation = validate_image_file(args.image)
                
                if validation['valid']:
                    self.logger.info(f"✅ Зображення '{args.image}' валідне")
                else:
                    self.logger.error(f"❌ Зображення '{args.image}' невалідне:")
                    for issue in validation['errors']:
                        self.logger.error(f"  • {issue}")
                    sys.exit(1)
                
            except Exception as e:
                self.logger.error(f"❌ Помилка валідації: {e}")
                sys.exit(1)
        
        elif args.config:
            try:
                if self.app.config.validate():
                    self.logger.info("✅ Конфігурація валідна")
                else:
                    self.logger.error("❌ Конфігурація невалідна")
                    sys.exit(1)
                    
            except Exception as e:
                self.logger.error(f"❌ Помилка валідації конфігурації: {e}")
                sys.exit(1)
        
        else:
            self.logger.error("❌ Вкажіть --image або --config")
            sys.exit(1)
    
    def env_check(self, args):
        """Handle env-check command."""
        from ..utils.system import check_system_requirements, validate_api_credentials
        
        system_check = check_system_requirements()
        api_check = validate_api_credentials()
        
        self.logger.info("🔍 Перевірка середовища:")
        
        if system_check["meets_requirements"]:
            self.logger.info("✅ Середовище готове до роботи")
        else:
            self.logger.error("❌ Проблеми з середовищем:")
            for rec in system_check["recommendations"]:
                self.logger.error(f"  • {rec}")
        
        # API перевірки
        if all(api_check.values()):
            self.logger.info("✅ API креденціали налаштовані")
        else:
            self.logger.warning("⚠️ Попередження про API:")
            for api, valid in api_check.items():
                status = "✅" if valid else "❌"
                self.logger.warning(f"  {status} {api}")
        
        if args.detailed:
            # Системна інформація
            sys_info = log_system_info(self.logger)
            
            self.logger.info("\n💻 Детальна системна інформація:")
            self.logger.info(f"  Платформа: {sys_info['platform']}")
            self.logger.info(f"  Python: {sys_info['python_version']}")
            self.logger.info(f"  CPU: {sys_info['cpu_count']} ядер")
            self.logger.info(f"  RAM: {sys_info['available_memory_gb']:.1f}/{sys_info['memory_gb']:.1f} GB")
            self.logger.info(f"  Диск: {sys_info['disk_space_gb']:.1f} GB вільно")
            
            # Перевірки системних вимог
            self.logger.info("\n🔧 Перевірки системних вимог:")
            for check_name, check_result in system_check["checks"].items():
                if isinstance(check_result, dict) and "passed" in check_result:
                    status = "✅" if check_result["passed"] else "❌"
                    current = check_result.get("current", "N/A")
                    required = check_result.get("required", "N/A")
                    self.logger.info(f"  {status} {check_name}: {current} (required: {required})")
                else:
                    self.logger.info(f"  ⚠️ {check_name}: {check_result}") 