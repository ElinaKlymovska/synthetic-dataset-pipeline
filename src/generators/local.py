"""
Локальний генератор зображень з використанням Stable Diffusion.
"""

import logging
from typing import List
from ..core import ImageGenerator, GenerationConfig, GenerationResult, AppConfig, ServiceRegistry


class LocalStableDiffusionGenerator(ImageGenerator):
    """Локальний генератор для Stable Diffusion (заготовка)."""
    
    def __init__(self, config: AppConfig, registry: ServiceRegistry):
        self.config = config
        self.registry = registry
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.warning("LocalStableDiffusionGenerator is not implemented yet")
    
    async def generate_image(
        self, 
        reference_image: str,
        config: GenerationConfig,
        output_path: str
    ) -> GenerationResult:
        """Генерує одне зображення локально."""
        
        # TODO: Реалізувати локальну генерацію
        error_msg = "Local generation not implemented yet. Use 'replicate' generator instead."
        self.logger.error(error_msg)
        
        return GenerationResult(
            success=False,
            error_message=error_msg
        )
    
    async def generate_batch(
        self,
        reference_image: str, 
        configs: List[GenerationConfig],
        output_dir: str
    ) -> List[GenerationResult]:
        """Генерує пакет зображень локально."""
        
        # TODO: Реалізувати локальну пакетну генерацію
        results = []
        
        for config in configs:
            result = await self.generate_image(reference_image, config, "")
            results.append(result)
        
        return results
    
    def estimate_cost(self, count: int) -> float:
        """Оцінює вартість генерації (для локального генератора = 0)."""
        return 0.0
    
    def validate_config(self, config: GenerationConfig) -> bool:
        """Валідує конфігурацію генерації."""
        # Базові перевірки
        return config.prompt and len(config.prompt.strip()) > 0 