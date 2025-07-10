"""
Утиліти для роботи з даними, конфігураціями та експортом.
"""

import json
import yaml
import csv
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime

from .file import ensure_directory


# =============================================================================
# Конфігурація
# =============================================================================

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Завантажує YAML конфігурацію."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Зберігає конфігурацію в YAML формат."""
    ensure_directory(Path(config_path).parent)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)


def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Завантажує JSON конфігурацію."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Зберігає конфігурацію в JSON формат."""
    ensure_directory(Path(config_path).parent)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Рекурсивно зливає дві конфігурації."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


# =============================================================================
# Експорт даних
# =============================================================================

def export_dataset_json(
    metadata_dir: Union[str, Path], 
    output_path: Union[str, Path],
    include_images: bool = False
) -> Dict[str, Any]:
    """Експортує датасет в JSON формат."""
    metadata_dir = Path(metadata_dir)
    output_path = Path(output_path)
    
    ensure_directory(output_path.parent)
    
    dataset = {
        "export_info": {
            "created_at": datetime.now().isoformat(),
            "version": "2.0",
            "include_images": include_images
        },
        "images": []
    }
    
    # Збираємо всі метадані
    for metadata_file in metadata_dir.glob("*.json"):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Додаємо до датасету
            dataset["images"].append(metadata)
            
        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")
    
    # Статистика
    dataset["statistics"] = {
        "total_images": len(dataset["images"]),
        "successful_generations": len([img for img in dataset["images"] 
                                      if img.get("status", {}).get("generation_status") == "success"]),
    }
    
    # Зберігаємо
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    return dataset


def export_dataset_csv(metadata_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Експортує датасет в CSV формат."""
    metadata_dir = Path(metadata_dir)
    output_path = Path(output_path)
    
    ensure_directory(output_path.parent)
    
    # Визначаємо колонки
    fieldnames = [
        "image_id", "timestamp", "source_image", "generated_image",
        "prompt_text", "negative_prompt", "model_version",
        "steps", "guidance_scale", "strength", "resolution",
        "quality_score", "pose_category", "outfit_category",
        "generation_cost", "generation_time", "file_size_mb"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                core_info = metadata.get("core_info", {})
                gen_params = metadata.get("generation_params", {})
                lora_info = metadata.get("lora_training", {})
                status = metadata.get("status", {})
                image_props = metadata.get("image_properties", {})
                
                row = {
                    "image_id": core_info.get("image_id", ""),
                    "timestamp": core_info.get("timestamp", ""),
                    "source_image": core_info.get("source_image", ""),
                    "generated_image": core_info.get("generated_image", ""),
                    "prompt_text": gen_params.get("prompt_text", ""),
                    "negative_prompt": gen_params.get("negative_prompt", ""),
                    "model_version": gen_params.get("model_version", ""),
                    "steps": gen_params.get("steps", ""),
                    "guidance_scale": gen_params.get("guidance_scale", ""),
                    "strength": gen_params.get("strength", ""),
                    "resolution": gen_params.get("resolution", ""),
                    "quality_score": lora_info.get("quality_score", ""),
                    "pose_category": lora_info.get("pose_category", ""),
                    "outfit_category": lora_info.get("outfit_category", ""),
                    "generation_cost": status.get("generation_cost", ""),
                    "generation_time": status.get("generation_time_seconds", ""),
                    "file_size_mb": round(image_props.get("file_size_bytes", 0) / (1024*1024), 2)
                }
                
                writer.writerow(row)
                
            except Exception as e:
                print(f"Error processing {metadata_file}: {e}")


# =============================================================================
# Статистика та аналіз
# =============================================================================

def calculate_dataset_stats(metadata_dir: Union[str, Path]) -> Dict[str, Any]:
    """Обчислює статистику датасету."""
    metadata_dir = Path(metadata_dir)
    
    stats = {
        "total_files": 0,
        "valid_metadata": 0,
        "generation_success_rate": 0.0,
        "average_quality_score": 0.0,
        "total_cost": 0.0,
        "average_generation_time": 0.0,
        "pose_distribution": {},
        "outfit_distribution": {},
        "model_usage": {},
        "resolution_distribution": {},
        "file_size_stats": {
            "min_mb": float('inf'),
            "max_mb": 0.0,
            "avg_mb": 0.0
        }
    }
    
    valid_metadatas = []
    
    for metadata_file in metadata_dir.glob("*.json"):
        stats["total_files"] += 1
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            valid_metadatas.append(metadata)
            stats["valid_metadata"] += 1
            
        except Exception:
            continue
    
    if not valid_metadatas:
        return stats
    
    # Аналізуємо валідні метадані
    successful_generations = 0
    quality_scores = []
    costs = []
    generation_times = []
    file_sizes = []
    
    for metadata in valid_metadatas:
        # Успішність генерації
        status = metadata.get("status", {})
        if status.get("generation_status") == "success":
            successful_generations += 1
        
        # Якість
        lora_info = metadata.get("lora_training", {})
        quality = lora_info.get("quality_score", 0)
        if quality > 0:
            quality_scores.append(quality)
        
        # Вартість
        cost = status.get("generation_cost", 0)
        if cost > 0:
            costs.append(cost)
        
        # Час генерації
        gen_time = status.get("generation_time_seconds", 0)
        if gen_time > 0:
            generation_times.append(gen_time)
        
        # Розподіл поз і одягу
        pose = lora_info.get("pose_category", "unknown")
        outfit = lora_info.get("outfit_category", "unknown")
        
        stats["pose_distribution"][pose] = stats["pose_distribution"].get(pose, 0) + 1
        stats["outfit_distribution"][outfit] = stats["outfit_distribution"].get(outfit, 0) + 1
        
        # Модель
        gen_params = metadata.get("generation_params", {})
        model = gen_params.get("model_version", "unknown")
        stats["model_usage"][model] = stats["model_usage"].get(model, 0) + 1
        
        # Роздільна здатність
        resolution = gen_params.get("resolution", "unknown")
        stats["resolution_distribution"][resolution] = stats["resolution_distribution"].get(resolution, 0) + 1
        
        # Розмір файлу
        image_props = metadata.get("image_properties", {})
        file_size_mb = image_props.get("file_size_bytes", 0) / (1024 * 1024)
        if file_size_mb > 0:
            file_sizes.append(file_size_mb)
    
    # Обчислюємо фінальні статистики
    stats["generation_success_rate"] = (successful_generations / len(valid_metadatas)) * 100
    stats["average_quality_score"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    stats["total_cost"] = sum(costs)
    stats["average_generation_time"] = sum(generation_times) / len(generation_times) if generation_times else 0
    
    if file_sizes:
        stats["file_size_stats"] = {
            "min_mb": round(min(file_sizes), 2),
            "max_mb": round(max(file_sizes), 2),
            "avg_mb": round(sum(file_sizes) / len(file_sizes), 2)
        }
    
    return stats


def generate_quality_report(stats: Dict[str, Any]) -> str:
    """Генерує текстовий звіт про якість датасету."""
    report_lines = [
        "🎯 ЗВІТ ПРО ЯКІСТЬ ДАТАСЕТУ",
        "=" * 50,
        "",
        f"📊 Загальна статистика:",
        f"   • Всього файлів: {stats['total_files']}",
        f"   • Валідних метаданих: {stats['valid_metadata']}",
        f"   • Успішність генерації: {stats['generation_success_rate']:.1f}%",
        f"   • Середня якість: {stats['average_quality_score']:.2f}",
        f"   • Загальна вартість: ${stats['total_cost']:.2f}",
        f"   • Середній час генерації: {stats['average_generation_time']:.1f}с",
        "",
        f"📁 Розмір файлів:",
        f"   • Мін: {stats['file_size_stats']['min_mb']:.2f} MB",
        f"   • Макс: {stats['file_size_stats']['max_mb']:.2f} MB", 
        f"   • Середній: {stats['file_size_stats']['avg_mb']:.2f} MB",
        "",
        f"🤸 Розподіл поз:",
    ]
    
    for pose, count in sorted(stats['pose_distribution'].items()):
        percentage = (count / stats['valid_metadata']) * 100
        report_lines.append(f"   • {pose}: {count} ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        f"👗 Розподіл одягу:",
    ])
    
    for outfit, count in sorted(stats['outfit_distribution'].items()):
        percentage = (count / stats['valid_metadata']) * 100
        report_lines.append(f"   • {outfit}: {count} ({percentage:.1f}%)")
    
    return "\n".join(report_lines) 