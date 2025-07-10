"""
Уніфікована система управління конфігурацією для генерації датасету персонажів.
"""

import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .interfaces import ConfigurationManager


@dataclass
class ModelConfig:
    """Конфігурація моделі."""
    name: str
    model_id: str
    description: str = ""
    cost_per_image: float = 0.075
    max_resolution: int = 1024
    supported_features: List[str] = field(default_factory=list)


@dataclass  
class GenerationSettings:
    """Налаштування генерації."""
    target_count: int = 15
    batch_size: int = 1
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.6
    seed: Optional[int] = None
    timeout: int = 300


@dataclass
class PromptSettings:
    """Налаштування промптів."""
    base_character: str = "young woman, detailed face, high quality, photorealistic"
    pose_variations: List[str] = field(default_factory=lambda: [
        "full body front view, standing pose, looking at camera",
        "full body side view, standing pose, profile shot",
        "full body back view, standing pose, elegant posture"
    ])
    outfit_variations: List[str] = field(default_factory=lambda: [
        "elegant dress, formal wear",
        "casual outfit, everyday wear", 
        "business suit, professional attire"
    ])
    style_variations: List[str] = field(default_factory=lambda: [
        "realistic, photographic, 8k, professional portrait"
    ])
    background_variations: List[str] = field(default_factory=lambda: [
        "studio background, professional lighting",
        "neutral background, soft lighting"
    ])
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly, bad anatomy"


@dataclass
class OutputSettings:
    """Налаштування виводу."""
    format: str = "png"
    quality: int = 95
    output_dir: str = "data/output"
    metadata_dir: str = "mcp"
    create_archive: bool = True
    save_metadata: bool = True


@dataclass
class APISettings:
    """Налаштування API."""
    base_url: str = "https://api.replicate.com/v1"
    timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5
    rate_limit: int = 10  # requests per minute


@dataclass
class AppConfig:
    """Головна конфігурація додатка."""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    prompts: PromptSettings = field(default_factory=PromptSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    api: APISettings = field(default_factory=APISettings)
    project_name: str = "GenImg_Dataset"
    version: str = "2.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Ініціалізація після створення."""
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()


class ConfigManager:
    """Менеджер конфігурації з підтримкою валідації та різних форматів."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, AppConfig] = {}
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Завантажує конфігурацію з файлу."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Визначаємо формат за розширенням
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            return self._load_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            return self._load_json(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Зберігає конфігурацію у файл."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Оновлюємо час модифікації
        if isinstance(config, dict):
            config['updated_at'] = datetime.now().isoformat()
        
        # Визначаємо формат за розширенням
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            self._save_yaml(config, config_path)
        elif config_path.suffix.lower() == '.json':
            self._save_json(config, config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def load_app_config(self, config_path: str) -> AppConfig:
        """Завантажує конфігурацію як об'єкт AppConfig."""
        if config_path in self._cache:
            return self._cache[config_path]
        
        config_dict = self.load_config(config_path)
        app_config = self._dict_to_app_config(config_dict)
        
        # Валідуємо конфігурацію
        if not self.validate_config(asdict(app_config)):
            raise ValueError("Invalid configuration")
        
        self._cache[config_path] = app_config
        return app_config
    
    def save_app_config(self, config: AppConfig, config_path: str) -> None:
        """Зберігає AppConfig у файл."""
        config_dict = asdict(config)
        # Конвертуємо datetime об'єкти в строки
        config_dict = self._serialize_datetime_fields(config_dict)
        self.save_config(config_dict, config_path)
        self._cache[config_path] = config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Валідує конфігурацію."""
        try:
            # Базова валідація структури
            required_sections = ['generation', 'output', 'api']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            
            # Валідація значень
            gen_config = config.get('generation', {})
            if gen_config.get('target_count', 0) <= 0:
                raise ValueError("target_count must be positive")
            
            if gen_config.get('width', 0) <= 0 or gen_config.get('height', 0) <= 0:
                raise ValueError("width and height must be positive")
            
            api_config = config.get('api', {})
            if not api_config.get('base_url'):
                raise ValueError("API base_url is required")
            
            return True
            
        except Exception as e:
            print(f"Config validation error: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """Повертає дефолтну конфігурацію."""
        default_config = AppConfig()
        
        # Додаємо дефолтні моделі
        default_config.models = {
            "sdxl": ModelConfig(
                name="SDXL",
                model_id="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                description="Stable Diffusion XL for high-quality image generation",
                cost_per_image=0.075,
                supported_features=["img2img", "inpainting", "controlnet"]
            ),
            "realistic_vision": ModelConfig(
                name="Realistic Vision",
                model_id="cjwbw/realistic-vision-v5:ac732df83cea7fff18b5b7c5c43cb241f2e31d7c1a1e2c4b5c4b5c4b5c4b5c4b5c",
                description="Realistic Vision V5 for photorealistic results",
                cost_per_image=0.08,
                supported_features=["img2img"]
            )
        }
        
        return asdict(default_config)
    
    def create_default_config_file(self, config_path: str) -> AppConfig:
        """Створює дефолтний файл конфігурації."""
        default_config_dict = self.get_default_config()
        self.save_config(default_config_dict, config_path)
        return self._dict_to_app_config(default_config_dict)
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Зливає дві конфігурації."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_env_overrides(self) -> Dict[str, Any]:
        """Отримує перевизначення з змінних оточення."""
        overrides = {}
        
        # API налаштування
        if os.getenv('REPLICATE_API_TOKEN'):
            overrides.setdefault('api', {})['token'] = os.getenv('REPLICATE_API_TOKEN')
        
        if os.getenv('GENIMG_OUTPUT_DIR'):
            overrides.setdefault('output', {})['output_dir'] = os.getenv('GENIMG_OUTPUT_DIR')
        
        if os.getenv('GENIMG_MODEL'):
            overrides.setdefault('generation', {})['default_model'] = os.getenv('GENIMG_MODEL')
        
        return overrides
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Завантажує YAML файл."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _save_yaml(self, config: Dict[str, Any], file_path: Path) -> None:
        """Зберігає YAML файл."""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Завантажує JSON файл."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_json(self, config: Dict[str, Any], file_path: Path) -> None:
        """Зберігає JSON файл."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _dict_to_app_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Конвертує словник в AppConfig."""
        # Обробляємо моделі
        models = {}
        if 'models' in config_dict:
            for name, model_data in config_dict['models'].items():
                if isinstance(model_data, dict):
                    models[name] = ModelConfig(**model_data)
                else:
                    models[name] = model_data
        
        # Створюємо об'єкти для секцій
        kwargs = {
            'models': models,
            'generation': GenerationSettings(**config_dict.get('generation', {})),
            'prompts': PromptSettings(**config_dict.get('prompts', {})),
            'output': OutputSettings(**config_dict.get('output', {})),
            'api': APISettings(**config_dict.get('api', {})),
        }
        
        # Додаємо інші поля
        for field_name in ['project_name', 'version', 'created_at', 'updated_at']:
            if field_name in config_dict:
                kwargs[field_name] = config_dict[field_name]
        
        # Обробляємо datetime поля
        for dt_field in ['created_at', 'updated_at']:
            if dt_field in kwargs and isinstance(kwargs[dt_field], str):
                kwargs[dt_field] = datetime.fromisoformat(kwargs[dt_field])
        
        return AppConfig(**kwargs)
    
    def _serialize_datetime_fields(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Серіалізує datetime поля в строки."""
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        return convert_datetime(config_dict)


def get_config_manager() -> ConfigManager:
    """Фабричний метод для отримання менеджера конфігурації."""
    return ConfigManager()


def load_config_with_env_overrides(config_path: str) -> AppConfig:
    """Завантажує конфігурацію з врахуванням змінних оточення."""
    manager = get_config_manager()
    
    # Завантажуємо базову конфігурацію
    if Path(config_path).exists():
        base_config = manager.load_config(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        base_config = manager.get_default_config()
    
    # Застосовуємо перевизначення з оточення
    env_overrides = manager.get_env_overrides()
    if env_overrides:
        base_config = manager.merge_configs(base_config, env_overrides)
    
    return manager._dict_to_app_config(base_config) 