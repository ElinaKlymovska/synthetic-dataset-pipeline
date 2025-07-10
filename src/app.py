"""
Головний application клас для системи генерації датасету персонажів.
Інтегрує всі компоненти з dependency injection.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from .core import (
    ServiceRegistry, GenerationConfig, GenerationResult,
    AppConfig, load_config_with_env_overrides,
    EnhancedMCPMetadataManager, DefaultPromptGenerator, ReplicateCostEstimator
)
from .generators import GeneratorFactory, BatchProcessor, ProgressTracker, create_default_configs
from .utils import (
    setup_logging, ensure_directory, validate_image_file, 
    check_system_requirements, validate_api_credentials,
    calculate_dataset_stats, export_dataset_json, export_dataset_csv,
    create_archive
)


class GenImgApp:
    """Головний application клас для генерації датасету персонажів."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Ініціалізуємо конфігурацію
        self.config_path = config_path or "config/app_config.yaml"
        self.config = self._load_config()
        
        # Ініціалізуємо логування
        self.logger = setup_logging(
            log_file="logs/genimg.log",
            level="INFO"
        )
        
        # Створюємо сервісний реєстр
        self.registry = ServiceRegistry()
        
        # Прапорець ініціалізації
        self._initialized = False
        
        self.logger.info(f"GenImgApp created with config: {self.config_path}")
    
    async def initialize(self):
        """Ініціалізує всі сервіси та компоненти."""
        if self._initialized:
            return
        
        self.logger.info("Initializing GenImgApp...")
        
        # Перевіряємо системні вимоги
        await self._check_system_requirements()
        
        # Реєструємо сервіси
        await self._register_services()
        
        # Створюємо необхідні директорії
        self._create_directories()
        
        self._initialized = True
        self.logger.info("GenImgApp initialized successfully")
    
    async def cleanup(self):
        """Очищає ресурси."""
        self.logger.info("Cleaning up GenImgApp...")
        
        # Тут можна додати cleanup логіку для сервісів
        
        self._initialized = False
        self.logger.info("GenImgApp cleanup completed")
    
    @asynccontextmanager
    async def context(self):
        """Context manager для автоматичної ініціалізації та cleanup."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()
    
    async def generate_character_dataset(
        self,
        reference_image: str,
        count: Optional[int] = None,
        output_dir: Optional[str] = None,
        generator_type: str = "replicate",
        custom_prompts: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Головна функція для генерації датасету персонажа."""
        
        if not self._initialized:
            await self.initialize()
        
        # Налаштування параметрів
        count = count or self.config.generation.target_count
        output_dir = output_dir or self.config.output.output_dir
        
        self.logger.info(f"Starting character dataset generation:")
        self.logger.info(f"  Reference image: {reference_image}")
        self.logger.info(f"  Count: {count}")
        self.logger.info(f"  Output dir: {output_dir}")
        self.logger.info(f"  Generator: {generator_type}")
        
        # Валідуємо референтне зображення
        validation = validate_image_file(reference_image)
        if not validation["valid"]:
            raise ValueError(f"Invalid reference image: {validation['errors']}")
        
        # Створюємо директорію виводу
        ensure_directory(output_dir)
        
        # Отримуємо сервіси
        generator = self._get_generator(generator_type)
        prompt_generator = self.registry.get("prompt_generator")
        cost_estimator = self.registry.get("cost_estimator")
        
        # Оцінюємо вартість
        estimated_cost = generator.estimate_cost(count)
        self.logger.info(f"Estimated cost: ${estimated_cost:.2f}")
        
        # Генеруємо промпти
        if custom_prompts:
            prompts = custom_prompts[:count]
            # Доповнюємо якщо потрібно
            while len(prompts) < count:
                additional_prompts = prompt_generator.generate_prompts(count - len(prompts))
                prompts.extend(additional_prompts)
        else:
            prompts = prompt_generator.generate_prompts(count)
        
        # Створюємо конфігурації
        configs = self._create_generation_configs(prompts)
        
        # Налаштовуємо прогрес трекер
        progress_tracker = ProgressTracker()
        progress_tracker.start(count)
        
        if progress_callback:
            def track_progress(tracker):
                progress_callback(tracker.get_progress_info())
            progress_tracker.add_callback(track_progress)
        
        # Створюємо batch processor
        batch_processor = BatchProcessor(generator, max_concurrent=3)
        
        # Генеруємо зображення
        def batch_progress_callback(current, total, result):
            progress_tracker.update(result)
        
        results = await batch_processor.process_batch_with_queue(
            reference_image=reference_image,
            configs=configs,
            output_dir=output_dir,
            progress_callback=batch_progress_callback
        )
        
        # Збираємо статистику
        successful = sum(1 for r in results if r.success)
        failed = count - successful
        total_cost = sum(r.cost or 0 for r in results if r.cost)
        
        # Створюємо архів якщо потрібно
        archive_path = None
        if self.config.output.create_archive and successful > 0:
            archive_path = await self._create_dataset_archive(output_dir)
        
        # Підсумок
        summary = {
            "total_requested": count,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / count * 100) if count > 0 else 0,
            "total_cost": total_cost,
            "output_directory": output_dir,
            "archive_path": archive_path,
            "results": results,
            "progress_info": progress_tracker.get_progress_info()
        }
        
        self.logger.info(f"Dataset generation completed: {successful}/{count} successful")
        return summary
    
    async def analyze_dataset(self, dataset_dir: Optional[str] = None) -> Dict[str, Any]:
        """Аналізує існуючий датасет."""
        if not self._initialized:
            await self.initialize()
        
        dataset_dir = dataset_dir or self.config.output.metadata_dir
        
        self.logger.info(f"Analyzing dataset in: {dataset_dir}")
        
        # Отримуємо менеджер метаданих
        metadata_manager = self.registry.get("metadata_manager")
        
        # Збираємо статистику
        dataset_summary = metadata_manager.get_dataset_summary()
        detailed_stats = calculate_dataset_stats(dataset_dir)
        
        analysis = {
            "dataset_summary": dataset_summary,
            "detailed_statistics": detailed_stats,
            "analysis_timestamp": detailed_stats.get("timestamp")
        }
        
        self.logger.info("Dataset analysis completed")
        return analysis
    
    async def export_dataset(
        self,
        format_type: str = "json",
        output_path: Optional[str] = None,
        dataset_dir: Optional[str] = None
    ) -> str:
        """Експортує датасет у вказаному форматі."""
        if not self._initialized:
            await self.initialize()
        
        dataset_dir = dataset_dir or self.config.output.metadata_dir
        
        if not output_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"exports/dataset_export_{timestamp}.{format_type}"
        
        self.logger.info(f"Exporting dataset to {format_type} format: {output_path}")
        
        if format_type.lower() == "json":
            export_dataset_json(dataset_dir, output_path)
        elif format_type.lower() == "csv":
            export_dataset_csv(dataset_dir, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        self.logger.info(f"Dataset exported successfully: {output_path}")
        return output_path
    
    def estimate_cost(
        self, 
        count: int, 
        generator_type: str = "replicate"
    ) -> Dict[str, Any]:
        """Оцінює вартість генерації."""
        generator = self._get_generator(generator_type)
        cost_estimator = self.registry.get("cost_estimator")
        
        base_cost = generator.estimate_cost(count)
        
        # Детальна оцінка від cost estimator
        detailed_cost = cost_estimator.estimate_generation_cost(
            "sdxl", count, {
                "width": self.config.generation.width,
                "height": self.config.generation.height,
                "steps": self.config.generation.steps
            }
        )
        
        return {
            "count": count,
            "base_cost": base_cost,
            "detailed_cost": detailed_cost,
            "currency": "USD",
            "estimated_time_minutes": count * 1.5,  # Приблизно 1.5 хв на зображення
            "generator_type": generator_type
        }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Повертає список доступних моделей."""
        return [
            {
                "name": name,
                "model_id": model.model_id,
                "description": model.description,
                "cost_per_image": model.cost_per_image,
                "max_resolution": model.max_resolution,
                "supported_features": model.supported_features
            }
            for name, model in self.config.models.items()
        ]
    
    def validate_environment(self) -> Dict[str, Any]:
        """Перевіряє робоче середовище."""
        system_check = check_system_requirements()
        api_check = validate_api_credentials()
        
        return {
            "system_requirements": system_check,
            "api_credentials": api_check,
            "all_valid": (
                system_check["meets_requirements"] and 
                all(api_check.values())
            )
        }
    
    def _load_config(self) -> AppConfig:
        """Завантажує конфігурацію."""
        try:
            return load_config_with_env_overrides(self.config_path)
        except FileNotFoundError:
            # Створюємо дефолтну конфігурацію
            from .config import ConfigManager
            manager = ConfigManager()
            return manager.create_default_config_file(self.config_path)
    
    async def _check_system_requirements(self):
        """Перевіряє системні вимоги."""
        validation = self.validate_environment()
        
        if not validation["all_valid"]:
            warnings = []
            
            if not validation["system_requirements"]["meets_requirements"]:
                warnings.extend(validation["system_requirements"]["recommendations"])
            
            if not all(validation["api_credentials"].values()):
                warnings.append("API credentials not properly configured")
            
            for warning in warnings:
                self.logger.warning(warning)
    
    async def _register_services(self):
        """Реєструє всі сервіси в registry."""
        # Metadata manager
        metadata_manager = EnhancedMCPMetadataManager(
            mcp_dir=self.config.output.metadata_dir,
            output_dir=self.config.output.output_dir
        )
        self.registry.register("metadata_manager", metadata_manager)
        
        # Prompt generator
        prompt_generator = DefaultPromptGenerator(self.config)
        self.registry.register("prompt_generator", prompt_generator)
        
        # Cost estimator
        cost_estimator = ReplicateCostEstimator(self.config)
        self.registry.register("cost_estimator", cost_estimator)
        
        self.logger.info("All services registered successfully")
    
    def _create_directories(self):
        """Створює необхідні директорії."""
        directories = [
            self.config.output.output_dir,
            self.config.output.metadata_dir,
            "logs",
            "exports"
        ]
        
        for directory in directories:
            ensure_directory(directory)
    
    def _get_generator(self, generator_type: str):
        """Отримує генератор вказаного типу."""
        return GeneratorFactory.create_generator(
            generator_type, self.config, self.registry
        )
    
    def _create_generation_configs(self, prompts: List[str]) -> List[GenerationConfig]:
        """Створює конфігурації генерації з промптів."""
        configs = []
        
        for prompt in prompts:
            config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self.config.prompts.negative_prompt,
                width=self.config.generation.width,
                height=self.config.generation.height,
                steps=self.config.generation.steps,
                guidance_scale=self.config.generation.guidance_scale,
                strength=self.config.generation.strength,
                seed=self.config.generation.seed,
                model_name="sdxl"
            )
            configs.append(config)
        
        return configs
    
    async def _create_dataset_archive(self, output_dir: str) -> str:
        """Створює архів датасету."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"character_dataset_{timestamp}.zip"
        archive_path = f"exports/{archive_name}"
        
        # Створюємо архів в окремому потоці
        loop = asyncio.get_event_loop()
        archive_full_path = await loop.run_in_executor(
            None, create_archive, output_dir, archive_path
        )
        
        self.logger.info(f"Dataset archive created: {archive_full_path}")
        return archive_full_path


# =============================================================================
# Convenience functions
# =============================================================================

async def generate_character_dataset(
    reference_image: str,
    count: int = 15,
    output_dir: str = "data/output",
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Зручна функція для швидкої генерації датасету."""
    
    async with GenImgApp(config_path).context() as app:
        return await app.generate_character_dataset(
            reference_image=reference_image,
            count=count,
            output_dir=output_dir
        )


async def analyze_existing_dataset(
    dataset_dir: str = "mcp",
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Зручна функція для аналізу існуючого датасету."""
    
    async with GenImgApp(config_path).context() as app:
        return await app.analyze_dataset(dataset_dir)


def estimate_generation_cost(
    count: int,
    generator_type: str = "replicate",
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Зручна функція для оцінки вартості."""
    
    app = GenImgApp(config_path)
    return app.estimate_cost(count, generator_type)


def validate_setup(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Зручна функція для валідації налаштувань."""
    
    app = GenImgApp(config_path)
    return app.validate_environment()


# =============================================================================
# CLI Helper functions
# =============================================================================

def print_generation_summary(summary: Dict[str, Any]):
    """Друкує підсумок генерації в консоль."""
    print("\n" + "="*60)
    print("🎨 ПІДСУМОК ГЕНЕРАЦІЇ ДАТАСЕТУ")
    print("="*60)
    
    print(f"📊 Загальна статистика:")
    print(f"   • Запитано: {summary['total_requested']}")
    print(f"   • Успішно: {summary['successful']}")
    print(f"   • Невдало: {summary['failed']}")
    print(f"   • Успішність: {summary['success_rate']:.1f}%")
    
    print(f"\n💰 Вартість:")
    print(f"   • Загальна: ${summary['total_cost']:.2f}")
    print(f"   • За зображення: ${summary['total_cost']/summary['successful']:.3f}" if summary['successful'] > 0 else "   • За зображення: $0.000")
    
    print(f"\n📁 Файли:")
    print(f"   • Директорія: {summary['output_directory']}")
    if summary['archive_path']:
        print(f"   • Архів: {summary['archive_path']}")
    
    progress = summary['progress_info']
    print(f"\n⏱️ Час виконання:")
    print(f"   • Загальний: {progress['elapsed_time_seconds']:.1f}с")
    print(f"   • Середній на зображення: {progress['elapsed_time_seconds']/progress['completed']:.1f}с" if progress['completed'] > 0 else "   • Середній на зображення: 0.0с")
    
    print("\n✅ Генерація завершена!")
    print("="*60)


def print_analysis_summary(analysis: Dict[str, Any]):
    """Друкує підсумок аналізу датасету."""
    summary = analysis["dataset_summary"]
    
    print("\n" + "="*60)
    print("📊 АНАЛІЗ ДАТАСЕТУ")
    print("="*60)
    
    overview = summary["dataset_overview"]
    print(f"📈 Загальний огляд:")
    print(f"   • Всього зображень: {overview['total_images']}")
    print(f"   • Успішні генерації: {overview['successful_generations']}")
    print(f"   • Невдалі генерації: {overview['failed_generations']}")
    print(f"   • Успішність: {overview['success_rate_percent']:.1f}%")
    
    quality = summary["quality_analysis"]
    print(f"\n🎯 Якість:")
    print(f"   • Середня оцінка: {quality['average_quality_score']:.2f}")
    print(f"   • Придатних для LoRA: {quality['lora_suitable_images']}")
    print(f"   • Готовність LoRA: {quality['lora_suitability_rate']:.1f}%")
    
    cost = summary["cost_analysis"]
    print(f"\n💰 Вартість:")
    print(f"   • Загальна: ${cost['total_cost_usd']:.2f}")
    print(f"   • Середня за зображення: ${cost['average_cost_per_image']:.3f}")
    
    diversity = summary["diversity_analysis"]
    print(f"\n🎭 Різноманітність:")
    print(f"   • Оцінка різноманітності: {diversity['diversity_score']:.2f}")
    print(f"   • Поз: {len(diversity['pose_distribution'])}")
    print(f"   • Одягу: {len(diversity['outfit_distribution'])}")
    
    print("\n📋 Рекомендації для LoRA тренування:")
    recommendations = summary["lora_training_recommendations"]
    if "training_advice" in recommendations:
        for advice in recommendations["training_advice"]:
            print(f"   • {advice}")
    
    print("\n✅ Аналіз завершено!")
    print("="*60)


# =============================================================================
# App Factory Functions
# =============================================================================

def create_app(config_path: Optional[str] = None, generator_type: str = "replicate") -> GenImgApp:
    """Factory function to create and initialize GenImgApp instance."""
    app = GenImgApp(config_path=config_path)
    return app


def create_app_with_dependencies(config_path: Optional[str] = None) -> GenImgApp:
    """Factory function to create GenImgApp with all dependencies pre-configured."""
    app = GenImgApp(config_path=config_path)
    return app 