"""
Сервісний слой для системи генерації датасету персонажів.
"""

import os
import json
import hashlib
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

from .interfaces import (
    MetadataManager, PromptGenerator, CostEstimator, QualityAnalyzer,
    ServiceRegistry, BaseService, DatasetInfo
)
from .config import AppConfig, ModelConfig


class EnhancedMCPMetadataManager:
    """Покращений менеджер метаданих для LoRA тренування."""
    
    def __init__(self, mcp_dir: str = "mcp", output_dir: str = "data/output"):
        self.mcp_dir = Path(mcp_dir)
        self.output_dir = Path(output_dir)
        self.mcp_dir.mkdir(exist_ok=True)
        
    def save_metadata(
        self, 
        image_path: str, 
        metadata: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Зберігає метадані для зображення."""
        if filename is None:
            # Генеруємо filename на основі хешу зображення
            image_hash = self._calculate_image_hash(image_path)
            filename = f"image_{image_hash[:8]}.json"
        
        metadata_path = self.mcp_dir / filename
        
        # Додаємо системну інформацію
        enhanced_metadata = {
            **metadata,
            "saved_at": datetime.now().isoformat(),
            "metadata_version": "2.1",
            "lora_training_optimized": True
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
        
        return str(metadata_path)
    
    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Завантажує метадані."""
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Повертає детальну статистику датасету для LoRA тренування."""
        metadata_files = list(self.mcp_dir.glob("*.json"))
        
        if not metadata_files:
            return {
                "total_images": 0,
                "error": "No metadata files found"
            }
        
        total_images = 0
        successful_generations = 0
        failed_generations = 0
        total_cost = 0.0
        generation_times = []
        quality_scores = []
        pose_categories = {}
        outfit_categories = {}
        model_usage = {}
        
        for metadata_file in metadata_files:
            try:
                metadata = self.load_metadata(metadata_file)
                total_images += 1
                
                # Статус генерації
                status = metadata.get('status', {})
                if status.get('generation_status') == 'success':
                    successful_generations += 1
                else:
                    failed_generations += 1
                
                # Вартість
                cost = status.get('generation_cost', 0)
                if cost:
                    total_cost += cost
                
                # Час генерації
                gen_time = status.get('generation_time_seconds', 0)
                if gen_time:
                    generation_times.append(gen_time)
                
                # Якість
                lora_info = metadata.get('lora_training', {})
                quality = lora_info.get('quality_score', 0)
                if quality:
                    quality_scores.append(quality)
                
                # Категорії поз та одягу
                pose_cat = lora_info.get('pose_category', 'unknown')
                outfit_cat = lora_info.get('outfit_category', 'unknown')
                
                pose_categories[pose_cat] = pose_categories.get(pose_cat, 0) + 1
                outfit_categories[outfit_cat] = outfit_categories.get(outfit_cat, 0) + 1
                
                # Використання моделей
                model = metadata.get('generation_params', {}).get('model_version', 'unknown')
                model_usage[model] = model_usage.get(model, 0) + 1
                
            except Exception as e:
                print(f"Error processing {metadata_file}: {e}")
                continue
        
        # Обчислюємо статистики
        avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        success_rate = (successful_generations / total_images * 100) if total_images > 0 else 0
        
        # Аналіз для LoRA тренування
        lora_suitable_count = sum(1 for score in quality_scores if score > 0.7)
        diversity_score = self._calculate_diversity_score(pose_categories, outfit_categories)
        
        return {
            "dataset_overview": {
                "total_images": total_images,
                "successful_generations": successful_generations,
                "failed_generations": failed_generations,
                "success_rate_percent": round(success_rate, 1)
            },
            "cost_analysis": {
                "total_cost_usd": round(total_cost, 2),
                "average_cost_per_image": round(total_cost / total_images, 3) if total_images > 0 else 0
            },
            "performance_metrics": {
                "average_generation_time_seconds": round(avg_generation_time, 1),
                "total_generation_time_minutes": round(sum(generation_times) / 60, 1)
            },
            "quality_analysis": {
                "average_quality_score": round(avg_quality_score, 2),
                "lora_suitable_images": lora_suitable_count,
                "lora_suitability_rate": round(lora_suitable_count / total_images * 100, 1) if total_images > 0 else 0
            },
            "diversity_analysis": {
                "pose_distribution": pose_categories,
                "outfit_distribution": outfit_categories,
                "diversity_score": round(diversity_score, 2),
                "model_usage": model_usage
            },
            "lora_training_recommendations": self._generate_lora_recommendations(
                quality_scores, pose_categories, outfit_categories, total_images
            ),
            "generated_at": datetime.now().isoformat()
        }
    
    def create_enhanced_metadata(
        self,
        source_image: str,
        generated_image_path: str,
        prompt_text: str,
        generation_params: Dict[str, Any],
        quality_analysis: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Створює розширені метадані оптимізовані для LoRA тренування."""
        
        # Базова інформація
        image_id = kwargs.get('image_id', f"img_{int(time.time())}")
        
        # Аналіз якості зображення
        image_info = self._get_image_info(generated_image_path)
        image_hash = self._calculate_image_hash(generated_image_path)
        
        # LoRA-специфічний аналіз
        lora_analysis = self._analyze_for_lora_training(
            prompt_text, image_info, quality_analysis or {}
        )
        
        metadata = {
            "core_info": {
                "image_id": image_id,
                "timestamp": datetime.now().isoformat(),
                "project_name": kwargs.get('project_name', 'GenImg_Dataset'),
                "source_image": source_image,
                "generated_image": generated_image_path,
                "image_hash": image_hash
            },
            "generation_params": {
                "prompt_text": prompt_text,
                "negative_prompt": generation_params.get('negative_prompt', ''),
                "model_version": generation_params.get('model_version', 'unknown'),
                "steps": generation_params.get('steps', 30),
                "guidance_scale": generation_params.get('guidance_scale', 7.5),
                "strength": generation_params.get('strength', 0.6),
                "seed": generation_params.get('seed'),
                "resolution": f"{image_info.get('width', 0)}x{image_info.get('height', 0)}"
            },
            "image_properties": image_info,
            "lora_training": lora_analysis,
            "status": {
                "generation_status": kwargs.get('status', 'success'),
                "generation_time_seconds": kwargs.get('generation_time', 0),
                "generation_cost": kwargs.get('generation_cost', 0),
                "error_message": kwargs.get('error_message')
            },
            "metadata_version": "2.1"
        }
        
        return metadata
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Обчислює хеш зображення."""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return f"no_hash_{int(time.time())}"
    
    def _get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Отримує інформацію про зображення."""
        try:
            with Image.open(image_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "file_size_bytes": os.path.getsize(image_path),
                    "aspect_ratio": round(img.width / img.height, 2)
                }
        except Exception:
            return {}
    
    def _analyze_for_lora_training(
        self, 
        prompt: str, 
        image_info: Dict[str, Any],
        quality_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Аналізує зображення для LoRA тренування."""
        
        # Категоризація промпту
        pose_category = self._categorize_pose(prompt)
        outfit_category = self._categorize_outfit(prompt)
        
        # Обчислення якості
        quality_score = self._calculate_lora_quality_score(image_info, quality_analysis)
        
        # Витягування тегів для тренування
        training_tags = self._extract_training_tags(prompt)
        
        return {
            "suitable_for_training": quality_score > 0.6,
            "quality_score": quality_score,
            "pose_category": pose_category,
            "outfit_category": outfit_category,
            "training_tags": training_tags,
            "recommended_trigger_words": self._generate_trigger_words(prompt),
            "training_weight": self._calculate_training_weight(quality_score, pose_category, outfit_category)
        }
    
    def _categorize_pose(self, prompt: str) -> str:
        """Категоризує позу з промпту."""
        prompt_lower = prompt.lower()
        
        if "full body" in prompt_lower:
            if "front view" in prompt_lower:
                return "full_body_front"
            elif "back view" in prompt_lower:
                return "full_body_back"
            elif "side view" in prompt_lower:
                return "full_body_side"
            else:
                return "full_body_general"
        elif "portrait" in prompt_lower or "close up" in prompt_lower:
            return "portrait"
        elif "sitting" in prompt_lower:
            return "sitting"
        elif "lying" in prompt_lower:
            return "lying"
        else:
            return "standing"
    
    def _categorize_outfit(self, prompt: str) -> str:
        """Категоризує одяг з промпту."""
        prompt_lower = prompt.lower()
        
        if "dress" in prompt_lower:
            return "dress"
        elif "suit" in prompt_lower:
            return "formal"
        elif "bikini" in prompt_lower or "swimwear" in prompt_lower:
            return "swimwear"
        elif "lingerie" in prompt_lower:
            return "lingerie"
        elif "casual" in prompt_lower:
            return "casual"
        elif "cosplay" in prompt_lower:
            return "cosplay"
        else:
            return "general"
    
    def _extract_training_tags(self, prompt: str) -> List[str]:
        """Витягує теги для LoRA тренування."""
        tags = []
        prompt_lower = prompt.lower()
        
        # Базові теги
        if "woman" in prompt_lower or "girl" in prompt_lower:
            tags.append("woman")
        if "photorealistic" in prompt_lower:
            tags.append("photorealistic")
        if "portrait" in prompt_lower:
            tags.append("portrait")
        if "full body" in prompt_lower:
            tags.append("full_body")
            
        # Теги одягу
        clothing_keywords = ["dress", "suit", "bikini", "lingerie", "casual"]
        for keyword in clothing_keywords:
            if keyword in prompt_lower:
                tags.append(keyword)
        
        return list(set(tags))
    
    def _generate_trigger_words(self, prompt: str) -> List[str]:
        """Генерує рекомендовані trigger words для LoRA."""
        # Базові trigger words
        triggers = ["ohwx woman", "ohwx girl", "ohwx person"]
        
        # Додаємо специфічні на основі промпту
        if "portrait" in prompt.lower():
            triggers.append("ohwx portrait")
        if "full body" in prompt.lower():
            triggers.append("ohwx full body")
            
        return triggers
    
    def _calculate_lora_quality_score(
        self, 
        image_info: Dict[str, Any],
        quality_analysis: Dict[str, Any]
    ) -> float:
        """Обчислює якість для LoRA тренування."""
        score = 0.0
        
        # Розмір зображення (25%)
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        if width >= 1024 and height >= 1024:
            score += 0.25
        elif width >= 512 and height >= 512:
            score += 0.15
        
        # Аспектний коефіцієнт (15%)
        aspect_ratio = image_info.get('aspect_ratio', 0)
        if 0.8 <= aspect_ratio <= 1.2:  # Близько до квадрата
            score += 0.15
        
        # Формат файлу (10%)
        if image_info.get('format') == 'PNG':
            score += 0.10
        
        # Розмір файлу (розумний баланс) (10%)
        file_size_mb = image_info.get('file_size_bytes', 0) / (1024 * 1024)
        if 0.5 <= file_size_mb <= 10:
            score += 0.10
        
        # Якісний аналіз (якщо наданий) (40%)
        identity_score = quality_analysis.get('identity_similarity', 0)
        if identity_score > 0:
            score += identity_score * 0.40
        else:
            score += 0.20  # базовий бонус якщо аналіз недоступний
        
        return min(score, 1.0)
    
    def _calculate_training_weight(self, quality_score: float, pose_category: str, outfit_category: str) -> float:
        """Обчислює рекомендовану вагу для тренування."""
        base_weight = quality_score
        
        # Коригуємо на основі рідкості категорії
        pose_weights = {
            "portrait": 1.2,
            "full_body_front": 1.0,
            "full_body_side": 1.1,
            "full_body_back": 1.2,
            "sitting": 1.1,
            "lying": 1.3
        }
        
        outfit_weights = {
            "formal": 1.0,
            "casual": 1.0,
            "dress": 1.1,
            "swimwear": 1.3,
            "cosplay": 1.4
        }
        
        pose_multiplier = pose_weights.get(pose_category, 1.0)
        outfit_multiplier = outfit_weights.get(outfit_category, 1.0)
        
        return min(base_weight * pose_multiplier * outfit_multiplier, 2.0)
    
    def _calculate_diversity_score(
        self, 
        pose_categories: Dict[str, int],
        outfit_categories: Dict[str, int]
    ) -> float:
        """Обчислює показник різноманітності датасету."""
        if not pose_categories or not outfit_categories:
            return 0.0
        
        # Ентропія для поз
        total_poses = sum(pose_categories.values())
        pose_entropy = 0.0
        for count in pose_categories.values():
            if count > 0:
                p = count / total_poses
                pose_entropy -= p * (p.log2() if hasattr(p, 'log2') else 0)
        
        # Ентропія для одягу
        total_outfits = sum(outfit_categories.values())
        outfit_entropy = 0.0
        for count in outfit_categories.values():
            if count > 0:
                p = count / total_outfits
                outfit_entropy -= p * (p.log2() if hasattr(p, 'log2') else 0)
        
        # Нормалізуємо до [0, 1]
        max_pose_entropy = (len(pose_categories).bit_length() if len(pose_categories) > 1 else 1)
        max_outfit_entropy = (len(outfit_categories).bit_length() if len(outfit_categories) > 1 else 1)
        
        normalized_pose = pose_entropy / max_pose_entropy if max_pose_entropy > 0 else 0
        normalized_outfit = outfit_entropy / max_outfit_entropy if max_outfit_entropy > 0 else 0
        
        return (normalized_pose + normalized_outfit) / 2
    
    def _generate_lora_recommendations(
        self,
        quality_scores: List[float],
        pose_categories: Dict[str, int],
        outfit_categories: Dict[str, int],
        total_images: int
    ) -> Dict[str, Any]:
        """Генерує рекомендації для LoRA тренування."""
        if not quality_scores:
            return {"error": "No quality data available"}
        
        high_quality_count = sum(1 for score in quality_scores if score > 0.8)
        medium_quality_count = sum(1 for score in quality_scores if 0.6 < score <= 0.8)
        
        recommendations = {
            "dataset_readiness": "ready" if high_quality_count >= 10 else "needs_improvement",
            "recommended_training_steps": min(max(total_images * 100, 1000), 3000),
            "suggested_learning_rate": 1e-4 if high_quality_count > 20 else 1e-5,
            "batch_size_recommendation": min(max(total_images // 10, 1), 4),
            "quality_distribution": {
                "high_quality": high_quality_count,
                "medium_quality": medium_quality_count,
                "low_quality": total_images - high_quality_count - medium_quality_count
            }
        }
        
        # Додаємо конкретні поради
        advice = []
        if high_quality_count < 10:
            advice.append("Generate more high-quality images (score > 0.8)")
        if len(pose_categories) < 3:
            advice.append("Add more pose variations")
        if len(outfit_categories) < 3:
            advice.append("Add more outfit variations")
        
        recommendations["training_advice"] = advice
        
        return recommendations


class DefaultPromptGenerator:
    """Генератор промптів для різноманітного датасету."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.prompts_config = config.prompts
    
    def generate_prompts(self, count: int, style: str = "default") -> List[str]:
        """Генерує список промптів."""
        prompts = []
        
        for i in range(count):
            pose = self._get_pose_variation(i)
            outfit = self._get_outfit_variation(i)
            style_elem = self._get_style_variation(i)
            background = self._get_background_variation(i)
            
            prompt = f"{self.prompts_config.base_character}, {pose}, {outfit}, {style_elem}, {background}"
            prompts.append(prompt)
        
        return prompts
    
    def generate_prompt_pairs(self, count: int) -> List[Tuple[str, str]]:
        """Генерує пари (prompt, negative_prompt)."""
        prompts = self.generate_prompts(count)
        return [(prompt, self.prompts_config.negative_prompt) for prompt in prompts]
    
    def _get_pose_variation(self, index: int) -> str:
        """Отримує варіацію пози."""
        variations = self.prompts_config.pose_variations
        return variations[index % len(variations)]
    
    def _get_outfit_variation(self, index: int) -> str:
        """Отримує варіацію одягу."""
        variations = self.prompts_config.outfit_variations
        return variations[index % len(variations)]
    
    def _get_style_variation(self, index: int) -> str:
        """Отримує варіацію стилю."""
        variations = self.prompts_config.style_variations
        return variations[index % len(variations)]
    
    def _get_background_variation(self, index: int) -> str:
        """Отримує варіацію фону."""
        variations = self.prompts_config.background_variations
        return variations[index % len(variations)]


class ReplicateCostEstimator:
    """Оцінювач вартості для Replicate API."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.models = config.models
    
    def estimate_generation_cost(
        self, 
        model_name: str, 
        image_count: int,
        params: Dict[str, Any]
    ) -> float:
        """Оцінює вартість генерації."""
        if model_name not in self.models:
            # Дефолтна ціна якщо модель невідома
            cost_per_image = 0.075
        else:
            model = self.models[model_name]
            cost_per_image = model.cost_per_image
        
        # Коригуємо ціну на основі параметрів
        resolution_multiplier = self._get_resolution_multiplier(params)
        steps_multiplier = self._get_steps_multiplier(params)
        
        total_cost = image_count * cost_per_image * resolution_multiplier * steps_multiplier
        
        return round(total_cost, 3)
    
    def get_model_pricing(self, model_name: str) -> Dict[str, float]:
        """Повертає ціни для моделі."""
        if model_name not in self.models:
            return {"base_cost": 0.075, "currency": "USD"}
        
        model = self.models[model_name]
        return {
            "base_cost": model.cost_per_image,
            "currency": "USD",
            "max_resolution": model.max_resolution
        }
    
    def _get_resolution_multiplier(self, params: Dict[str, Any]) -> float:
        """Обчислює множник для роздільної здатності."""
        width = params.get('width', 1024)
        height = params.get('height', 1024)
        total_pixels = width * height
        
        # Базова роздільність 1024x1024
        base_pixels = 1024 * 1024
        
        return total_pixels / base_pixels
    
    def _get_steps_multiplier(self, params: Dict[str, Any]) -> float:
        """Обчислює множник для кількості кроків."""
        steps = params.get('steps', 30)
        base_steps = 30
        
        return steps / base_steps 