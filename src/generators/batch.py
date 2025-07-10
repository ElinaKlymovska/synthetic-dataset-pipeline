"""
Пакетна обробка генерації зображень з підтримкою конкурентності.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Callable

from ..core import ImageGenerator, GenerationConfig, GenerationResult


class BatchProcessor:
    """Обробник пакетної генерації з управлінням чергою та конкурентністю."""
    
    def __init__(self, generator: ImageGenerator, max_concurrent: int = 3):
        self.generator = generator
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def process_batch_with_queue(
        self,
        reference_image: str,
        configs: List[GenerationConfig],
        output_dir: str,
        progress_callback: Optional[Callable] = None
    ) -> List[GenerationResult]:
        """Обробляє пакет зображень з управлінням чергою."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting batch processing: {len(configs)} images, max concurrent: {self.max_concurrent}")
        
        # Створюємо семафор для обмеження конкурентності
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [None] * len(configs)  # Зберігаємо порядок результатів
        
        async def generate_single(i: int, config: GenerationConfig) -> GenerationResult:
            """Генерує одне зображення з семафором."""
            async with semaphore:
                output_filename = f"generated_{i+1:03d}.png"
                output_path = str(Path(output_dir) / output_filename)
                
                self.logger.debug(f"Starting generation {i+1}/{len(configs)}: {output_filename}")
                
                result = await self.generator.generate_image(
                    reference_image, config, output_path
                )
                
                # Викликаємо callback якщо є
                if progress_callback:
                    progress_callback(i+1, len(configs), result)
                
                self.logger.debug(f"Completed generation {i+1}/{len(configs)}: {'✅' if result.success else '❌'}")
                
                return result
        
        # Створюємо та запускаємо всі задачі
        tasks = [
            generate_single(i, config) 
            for i, config in enumerate(configs)
        ]
        
        # Виконуємо всі задачі конкурентно
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обробляємо результати та винятки
        for i, result in enumerate(completed_results):
            if isinstance(result, Exception):
                self.logger.error(f"Generation {i+1} failed with exception: {result}")
                results[i] = GenerationResult(
                    success=False,
                    error_message=f"Exception: {str(result)}"
                )
            else:
                results[i] = result
        
        successful = sum(1 for r in results if r and r.success)
        self.logger.info(f"Batch processing completed: {successful}/{len(configs)} successful")
        
        return results 