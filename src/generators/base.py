"""
Базові класи та фабрика для генераторів зображень.
"""

from typing import List, Dict, Any
from ..core import ImageGenerator, GenerationConfig, AppConfig, ServiceRegistry


class GeneratorFactory:
    """Фабрика для створення генераторів зображень."""

    @staticmethod
    def create_generator(
        generator_type: str, 
        config: AppConfig, 
        registry: ServiceRegistry
    ) -> ImageGenerator:
        """Створює генератор вказаного типу."""
        
        if generator_type == "replicate":
            from .replicate import ReplicateImageGenerator
            return ReplicateImageGenerator(config, registry)
        
        elif generator_type == "local":
            from .local import LocalStableDiffusionGenerator
            return LocalStableDiffusionGenerator(config, registry)
        
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

    @staticmethod
    def get_available_generators() -> List[str]:
        """Повертає список доступних генераторів."""
        return ["replicate", "local"]

    @staticmethod
    def get_recommended_generator() -> str:
        """Повертає рекомендований генератор за замовчуванням."""
        # Replicate більш стабільний та не вимагає локального GPU
        return "replicate"


def create_default_configs(count: int, prompts: List[str]) -> List[GenerationConfig]:
    """Створює конфігурації генерації за замовчуванням."""
    
    configs = []
    
    for i in range(count):
        # Циклічно використовуємо промпти
        prompt = prompts[i % len(prompts)] if prompts else f"Professional character variation {i+1}"
        
        config = GenerationConfig(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, deformed",
            width=768,
            height=768,
            steps=25,
            guidance_scale=7.5,
            strength=0.8,
            seed=None,  # Випадковий seed
            model_name="sdxl"
        )
        
        configs.append(config)
    
    return configs 