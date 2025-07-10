"""
Генератор зображень через Replicate API.
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

import replicate
from PIL import Image

from ..core import ImageGenerator, GenerationConfig, GenerationResult, AppConfig, ServiceRegistry
from ..utils import download_image, get_image_info, validate_image_file


class ReplicateImageGenerator(ImageGenerator):
    """Рефакторований генератор для Replicate API з dependency injection."""
    
    def __init__(self, config: AppConfig, registry: ServiceRegistry):
        self.config = config
        self.registry = registry
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Налаштовуємо API
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable not set. "
                "Get your token from https://replicate.com/account/api-tokens"
            )
        
        replicate.api_token = self.api_token
        
        # Отримуємо сервіси з registry
        self.metadata_manager = registry.get("metadata_manager")
        self.cost_estimator = registry.get("cost_estimator")
        
        self.logger.info(f"ReplicateImageGenerator initialized with {len(self.config.models)} models")
    
    async def generate_image(
        self, 
        reference_image: str,
        config: GenerationConfig,
        output_path: str
    ) -> GenerationResult:
        """Генерує одне зображення."""
        start_time = time.time()
        
        try:
            # Валідуємо вхідні дані
            if not self.validate_config(config):
                return GenerationResult(
                    success=False,
                    error_message="Invalid generation configuration"
                )
            
            # Валідуємо референтне зображення
            validation = validate_image_file(reference_image)
            if not validation["valid"]:
                return GenerationResult(
                    success=False,
                    error_message=f"Invalid reference image: {validation['errors']}"
                )
            
            # Отримуємо конфігурацію моделі
            model_config = self._get_model_config(config.model_name)
            if not model_config:
                return GenerationResult(
                    success=False,
                    error_message=f"Unknown model: {config.model_name}"
                )
            
            self.logger.info(f"Generating image with model: {model_config.model_id}")
            
            # Підготовуємо параметри для API
            api_params = self._prepare_api_params(reference_image, config)
            
            # Викликаємо Replicate API
            image_url = await self._call_replicate_api(model_config.model_id, api_params)
            
            # Завантажуємо згенероване зображення
            download_success = download_image(image_url, output_path)
            if not download_success:
                return GenerationResult(
                    success=False,
                    error_message="Failed to download generated image"
                )
            
            generation_time = time.time() - start_time
            
            # Створюємо метадані
            metadata = self._create_generation_metadata(
                reference_image, output_path, config, generation_time, model_config
            )
            
            # Зберігаємо метадані
            if self.metadata_manager:
                self.metadata_manager.save_metadata(output_path, metadata)
            
            self.logger.info(f"Successfully generated image: {output_path}")
            
            return GenerationResult(
                success=True,
                image_path=output_path,
                metadata=metadata,
                generation_time=generation_time,
                cost=model_config.cost_per_image
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Generation failed: {str(e)}"
            self.logger.error(error_msg)
            
            return GenerationResult(
                success=False,
                error_message=error_msg,
                generation_time=generation_time
            )
    
    async def generate_batch(
        self,
        reference_image: str, 
        configs: List[GenerationConfig],
        output_dir: str
    ) -> List[GenerationResult]:
        """Генерує пакет зображень."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting batch generation of {len(configs)} images")
        
        results = []
        
        for i, config in enumerate(configs):
            output_filename = f"generated_{i+1:03d}.png"
            output_path = str(Path(output_dir) / output_filename)
            
            # Генеруємо зображення
            result = await self.generate_image(reference_image, config, output_path)
            results.append(result)
            
            # Невелика затримка між запитами для дотримання rate limits
            if i < len(configs) - 1:
                await asyncio.sleep(self.config.api.retry_delay)
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Batch generation completed: {successful}/{len(configs)} successful")
        
        return results
    
    def estimate_cost(self, count: int) -> float:
        """Оцінює вартість генерації."""
        if self.cost_estimator:
            # Використовуємо параметри за замовчуванням
            default_params = {
                "width": self.config.generation.width,
                "height": self.config.generation.height,
                "steps": self.config.generation.steps
            }
            return self.cost_estimator.estimate_generation_cost(
                "sdxl", count, default_params
            )
        else:
            # Fallback обчислення
            return count * 0.075
    
    def validate_config(self, config: GenerationConfig) -> bool:
        """Валідує конфігурацію генерації."""
        try:
            # Базові перевірки
            if not config.prompt or not config.prompt.strip():
                return False
            
            if config.width <= 0 or config.height <= 0:
                return False
            
            if config.steps <= 0 or config.steps > 100:
                return False
            
            if config.guidance_scale <= 0 or config.guidance_scale > 20:
                return False
            
            if not (0 <= config.strength <= 1):
                return False
            
            # Перевіряємо чи модель існує
            if config.model_name not in self.config.models:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_model_config(self, model_name: str):
        """Отримує конфігурацію моделі."""
        return self.config.models.get(model_name)
    
    def _prepare_api_params(self, reference_image: str, config: GenerationConfig) -> Dict[str, Any]:
        """Підготовує параметри для API запиту."""
        
        # Читаємо reference image
        with open(reference_image, "rb") as f:
            image_data = f.read()
        
        params = {
            "image": image_data,
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "width": config.width,
            "height": config.height,
            "num_inference_steps": config.steps,
            "guidance_scale": config.guidance_scale,
            "strength": config.strength,
        }
        
        if config.seed is not None:
            params["seed"] = config.seed
        
        return params
    
    async def _call_replicate_api(self, model_id: str, params: Dict[str, Any]) -> str:
        """Викликає Replicate API та повертає URL згенерованого зображення."""
        
        try:
            # Викликаємо API асинхронно
            output = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: replicate.run(model_id, input=params)
            )
            
            # Обробляємо результат
            if isinstance(output, list) and len(output) > 0:
                return output[0]
            elif isinstance(output, str):
                return output
            else:
                raise ValueError(f"Unexpected output format: {type(output)}")
                
        except Exception as e:
            raise RuntimeError(f"Replicate API call failed: {str(e)}")
    
    def _create_generation_metadata(
        self,
        reference_image: str,
        output_path: str,
        config: GenerationConfig,
        generation_time: float,
        model_config
    ) -> Dict[str, Any]:
        """Створює метадані для згенерованого зображення."""
        
        # Отримуємо інформацію про зображення
        image_info = get_image_info(output_path)
        reference_info = get_image_info(reference_image)
        
        metadata = {
            "generation_info": {
                "model_id": model_config.model_id,
                "model_name": config.model_name,
                "prompt": config.prompt,
                "negative_prompt": config.negative_prompt,
                "width": config.width,
                "height": config.height,
                "steps": config.steps,
                "guidance_scale": config.guidance_scale,
                "strength": config.strength,
                "seed": config.seed,
                "generation_time": generation_time,
                "cost": model_config.cost_per_image
            },
            "reference_image": {
                "path": reference_image,
                "width": reference_info.get("width"),
                "height": reference_info.get("height"),
                "format": reference_info.get("format"),
                "size_mb": reference_info.get("size_mb")
            },
            "output_image": {
                "path": output_path,
                "width": image_info.get("width"),
                "height": image_info.get("height"),
                "format": image_info.get("format"),
                "size_mb": image_info.get("size_mb")
            },
            "timestamp": time.time(),
            "generator": "ReplicateImageGenerator"
        }
        
        return metadata 