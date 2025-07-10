"""
CLI Manager for the image generation system.
"""

import sys
import argparse
import logging
from typing import Optional, List
from ..core import AppConfig
from ..app import create_app, GenImgApp
from ..utils import setup_logging, check_system_requirements, validate_api_credentials
from .commands import Commands


class CLIManager:
    """Main CLI manager with improved user experience."""
    
    def __init__(self):
        self.app: Optional[GenImgApp] = None
        self.logger = setup_logging()
        self.commands = Commands()
        
    def run(self, args: Optional[List[str]] = None):
        """Main entry point for CLI."""
        try:
            parser = self._create_parser()
            parsed_args = parser.parse_args(args)
            
            # Setup logging level
            if parsed_args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            elif parsed_args.quiet:
                logging.getLogger().setLevel(logging.WARNING)
            
            # Validate environment
            if parsed_args.command != "env-check" and not parsed_args.skip_env_check:
                self._check_environment()
            
            # Initialize app if needed
            if parsed_args.command not in ["env-check", "help"]:
                self.app = create_app(
                    config_path=parsed_args.config,
                    generator_type=parsed_args.generator
                )
                # Передаємо app до команд
                self.commands.set_app(self.app)
            
            # Execute command
            self._execute_command(parsed_args)
            
        except KeyboardInterrupt:
            self.logger.info("❌ Перервано користувачем")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"❌ Помилка: {e}")
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser."""
        parser = argparse.ArgumentParser(
            description="🎨 GenImg - Генератор зображень персонажів",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Приклади використання:
  %(prog)s generate --reference photo.jpg --count 10
  %(prog)s estimate --count 15 --model sdxl
  %(prog)s analyze --dataset
  %(prog)s export --format json
            """
        )
        
        # Global options
        parser.add_argument("--config", "-c", help="Шлях до файлу конфігурації")
        parser.add_argument("--generator", "-g", default="replicate", 
                          choices=["replicate", "local"], help="Тип генератора")
        parser.add_argument("--verbose", "-v", action="store_true", help="Детальний вивід")
        parser.add_argument("--quiet", "-q", action="store_true", help="Тихий режим")
        parser.add_argument("--skip-env-check", action="store_true", 
                          help="Пропустити перевірку середовища")
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Доступні команди")
        
        # Generate command
        generate_parser = subparsers.add_parser("generate", aliases=["gen"], 
                                              help="Генерувати зображення")
        generate_parser.add_argument("--reference", "-r", required=True,
                                   help="Шлях до референтного зображення")
        generate_parser.add_argument("--count", "-n", type=int, default=15,
                                   help="Кількість зображень для генерації")
        generate_parser.add_argument("--output-dir", "-o", help="Директорія для збереження")
        generate_parser.add_argument("--model", "-m", help="Модель для генерації")
        generate_parser.add_argument("--denoising-strength", type=float, 
                                   help="Сила denoising (0.1-1.0)")
        generate_parser.add_argument("--guidance-scale", type=float,
                                   help="Guidance scale (1.0-20.0)")
        generate_parser.add_argument("--steps", type=int, help="Кількість кроків")
        generate_parser.add_argument("--width", type=int, help="Ширина зображення")
        generate_parser.add_argument("--height", type=int, help="Висота зображення")
        generate_parser.add_argument("--seed", type=int, help="Seed для генерації")
        generate_parser.add_argument("--no-confirm", action="store_true",
                                   help="Не запитувати підтвердження")
        
        # Estimate command
        estimate_parser = subparsers.add_parser("estimate", aliases=["cost"],
                                              help="Оцінити вартість генерації")
        estimate_parser.add_argument("--count", "-n", type=int, required=True,
                                   help="Кількість зображень")
        estimate_parser.add_argument("--model", "-m", help="Модель для оцінки")
        
        # Analyze command
        analyze_parser = subparsers.add_parser("analyze", aliases=["stats"],
                                             help="Аналіз датасету")
        analyze_parser.add_argument("--dataset", action="store_true",
                                  help="Показати статистику датасету")
        analyze_parser.add_argument("--image", help="Проаналізувати конкретне зображення")
        analyze_parser.add_argument("--detailed", action="store_true",
                                  help="Детальний аналіз")
        
        # Export command
        export_parser = subparsers.add_parser("export", help="Експорт даних")
        export_parser.add_argument("--format", "-f", choices=["json", "csv"], 
                                 default="json", help="Формат експорту")
        export_parser.add_argument("--output", "-o", help="Файл для збереження")
        
        # Models command
        models_parser = subparsers.add_parser("models", help="Управління моделями")
        models_parser.add_argument("--list", action="store_true", help="Список моделей")
        models_parser.add_argument("--info", help="Інформація про модель")
        
        # Validate command
        validate_parser = subparsers.add_parser("validate", help="Валідація")
        validate_parser.add_argument("--image", help="Перевірити зображення")
        validate_parser.add_argument("--config", action="store_true", 
                                   help="Перевірити конфігурацію")
        
        # Environment check command
        env_parser = subparsers.add_parser("env-check", help="Перевірка середовища")
        env_parser.add_argument("--detailed", action="store_true",
                              help="Детальна інформація")
        
        return parser
    
    def _check_environment(self):
        """Check environment and show warnings if needed."""
        # Check system requirements
        system_check = check_system_requirements()
        api_check = validate_api_credentials()
        
        issues = []
        warnings = []
        
        # Process system check results
        if not system_check.get("meets_requirements", True):
            issues.extend(system_check.get("recommendations", []))
        
        # Process API check results
        if not all(api_check.values()):
            for api, valid in api_check.items():
                if not valid:
                    warnings.append(f"API credentials not configured: {api}")
        
        if issues:
            self.logger.error("❌ Проблеми з середовищем:")
            for issue in issues:
                self.logger.error(f"  • {issue}")
            sys.exit(1)
        
        if warnings:
            self.logger.warning("⚠️ Попередження:")
            for warning in warnings:
                self.logger.warning(f"  • {warning}")
    
    def _execute_command(self, args):
        """Execute the selected command."""
        if args.command in ["generate", "gen"]:
            self.commands.generate(args)
        elif args.command in ["estimate", "cost"]:
            self.commands.estimate(args)
        elif args.command in ["analyze", "stats"]:
            self.commands.analyze(args)
        elif args.command == "export":
            self.commands.export(args)
        elif args.command == "models":
            self.commands.models(args)
        elif args.command == "validate":
            self.commands.validate(args)
        elif args.command == "env-check":
            self.commands.env_check(args)
        else:
            self.logger.error("❌ Невідома команда. Використайте --help для довідки.")
            sys.exit(1) 