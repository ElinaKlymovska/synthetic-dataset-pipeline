"""
Абстракції та інтерфейси для системи генерації датасету персонажів.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from pathlib import Path
from datetime import datetime


@dataclass
class GenerationConfig:
    """Конфігурація для генерації зображень."""
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    strength: float = 0.6  # для img2img
    model_name: str = "sdxl"
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class GenerationResult:
    """Результат генерації зображення."""
    success: bool
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    cost: Optional[float] = None


class ImageGenerator(ABC):
    """Абстрактний базовий клас для генераторів зображень."""
    
    @abstractmethod
    async def generate_image(
        self, 
        reference_image: str,
        config: GenerationConfig,
        output_path: str
    ) -> GenerationResult:
        """Генерує одне зображення."""
        pass
    
    @abstractmethod
    async def generate_batch(
        self,
        reference_image: str, 
        configs: List[GenerationConfig],
        output_dir: str
    ) -> List[GenerationResult]:
        """Генерує пакет зображень."""
        pass
    
    @abstractmethod
    def estimate_cost(self, count: int) -> float:
        """Оцінює вартість генерації."""
        pass
    
    @abstractmethod
    def validate_config(self, config: GenerationConfig) -> bool:
        """Валідує конфігурацію генерації."""
        pass


@runtime_checkable
class MetadataManager(Protocol):
    """Протокол для управління метаданими."""
    
    def save_metadata(
        self, 
        image_path: str, 
        metadata: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Зберігає метадані для зображення."""
        ...
    
    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Завантажує метадані."""
        ...
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Повертає статистику датасету."""
        ...


@runtime_checkable
class ConfigurationManager(Protocol):
    """Протокол для управління конфігурацією."""
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Завантажує конфігурацію."""
        ...
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Зберігає конфігурацію."""
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Валідує конфігурацію."""
        ...
    
    def get_default_config(self) -> Dict[str, Any]:
        """Повертає дефолтну конфігурацію."""
        ...


@dataclass
class ServiceRegistry:
    """Реєстр сервісів для Dependency Injection."""
    _services: Dict[str, Any] = field(default_factory=dict)
    
    def register(self, name: str, service: Any) -> None:
        """Реєструє сервіс."""
        self._services[name] = service
    
    def get(self, name: str) -> Any:
        """Отримує сервіс."""
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered")
        return self._services[name]
    
    def has(self, name: str) -> bool:
        """Перевіряє чи існує сервіс."""
        return name in self._services


@runtime_checkable  
class PromptGenerator(Protocol):
    """Протокол для генерації промптів."""
    
    def generate_prompts(self, count: int, style: str = "default") -> List[str]:
        """Генерує список промптів."""
        ...
    
    def generate_prompt_pairs(self, count: int) -> List[tuple[str, str]]:
        """Генерує пари (prompt, negative_prompt)."""
        ...


@runtime_checkable
class QualityAnalyzer(Protocol):
    """Протокол для аналізу якості зображень."""
    
    def analyze_image(self, image_path: str) -> Dict[str, float]:
        """Аналізує якість зображення."""
        ...
    
    def calculate_identity_similarity(
        self, 
        reference_path: str, 
        generated_path: str
    ) -> float:
        """Обчислює схожість з референсним зображенням."""
        ...


@runtime_checkable
class CostEstimator(Protocol):
    """Протокол для оцінки вартості."""
    
    def estimate_generation_cost(
        self, 
        model_name: str, 
        image_count: int,
        params: Dict[str, Any]
    ) -> float:
        """Оцінює вартість генерації."""
        ...
    
    def get_model_pricing(self, model_name: str) -> Dict[str, float]:
        """Повертає ціни для моделі."""
        ...


class BaseService(ABC):
    """Базовий клас для всіх сервісів."""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Ініціалізує сервіс."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Очищає ресурси сервісу."""
        pass
    
    @property
    def initialized(self) -> bool:
        """Перевіряє чи ініціалізований сервіс."""
        return self._initialized


@dataclass
class DatasetInfo:
    """Інформація про датасет."""
    total_images: int
    successful_generations: int
    failed_generations: int
    total_cost: float
    average_generation_time: float
    quality_scores: List[float]
    created_at: datetime
    updated_at: datetime
    metadata_path: str
    images_dir: str 