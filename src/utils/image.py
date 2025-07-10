"""
Утиліти для роботи з зображеннями.
"""

import os
import cv2
import hashlib
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from urllib.parse import urlparse
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter

from .file import ensure_directory, get_file_size_mb


def load_image(image_path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Завантажує зображення з можливістю зміни розміру."""
    image_path = str(image_path)
    
    # Спробуємо завантажити з PIL спочатку
    try:
        with Image.open(image_path) as img:
            # Конвертуємо в RGB якщо потрібно
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Змінюємо розмір якщо потрібно
            if target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Конвертуємо в numpy array
            return np.array(img)
    except Exception:
        # Fallback до OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # OpenCV завантажує в BGR, конвертуємо в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if target_size:
            image = cv2.resize(image, target_size)
        
        return image


def save_image(
    image: Union[np.ndarray, Image.Image], 
    output_path: Union[str, Path], 
    quality: int = 95
) -> None:
    """Зберігає зображення з вказаною якістю."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    
    if isinstance(image, np.ndarray):
        # Конвертуємо numpy array в PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Зберігаємо з відповідними параметрами
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        pil_image.save(output_path, 'JPEG', quality=quality, optimize=True)
    elif output_path.suffix.lower() == '.png':
        pil_image.save(output_path, 'PNG', optimize=True)
    else:
        pil_image.save(output_path)


def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Отримує детальну інформацію про зображення."""
    try:
        with Image.open(image_path) as img:
            info = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "file_size_bytes": os.path.getsize(image_path),
                "file_size_mb": get_file_size_mb(image_path),
                "aspect_ratio": round(img.width / img.height, 3),
                "megapixels": round((img.width * img.height) / 1_000_000, 2)
            }
            
            # Додаткова інформація якщо доступна
            if hasattr(img, '_getexif') and img._getexif():
                info["has_exif"] = True
            else:
                info["has_exif"] = False
            
            return info
    except Exception as e:
        return {"error": str(e)}


def calculate_image_hash(image_path: Union[str, Path], algorithm: str = "md5") -> str:
    """Обчислює хеш зображення."""
    try:
        hash_func = getattr(hashlib, algorithm)()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception:
        return f"no_hash_{int(datetime.now().timestamp())}"


def resize_image_smart(
    image: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int],
    method: str = "lanczos"
) -> Image.Image:
    """Розумна зміна розміру зображення зі збереженням якості."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Визначаємо метод ресемплінгу
    resample_methods = {
        "nearest": Image.Resampling.NEAREST,
        "lanczos": Image.Resampling.LANCZOS,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC
    }
    
    resample = resample_methods.get(method.lower(), Image.Resampling.LANCZOS)
    
    # Зберігаємо пропорції
    original_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]
    
    if original_ratio > target_ratio:
        # Зображення ширше за цільове співвідношення
        new_width = target_size[0]
        new_height = int(target_size[0] / original_ratio)
    else:
        # Зображення вище за цільове співвідношення
        new_height = target_size[1]
        new_width = int(target_size[1] * original_ratio)
    
    # Змінюємо розмір
    resized = image.resize((new_width, new_height), resample)
    
    # Створюємо фінальне зображення з цільовим розміром
    final_image = Image.new('RGB', target_size, (255, 255, 255))
    
    # Центруємо зображення
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    final_image.paste(resized, (x_offset, y_offset))
    
    return final_image


def enhance_image_quality(image: Image.Image, enhance_factor: float = 1.2) -> Image.Image:
    """Покращує якість зображення."""
    # Підвищуємо чіткість
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(enhance_factor)
    
    # Покращуємо контраст
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    return image


def download_image(url: str, output_path: Union[str, Path], timeout: int = 30) -> bool:
    """Завантажує зображення з URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        output_path = Path(output_path)
        ensure_directory(output_path.parent)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False


def validate_url(url: str) -> bool:
    """Перевіряє чи є URL валідним."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_image_file(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Валідує файл зображення."""
    image_path = Path(image_path)
    
    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    # Перевіряємо існування файлу
    if not image_path.exists():
        result["errors"].append("File does not exist")
        return result
    
    # Перевіряємо розширення
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if image_path.suffix.lower() not in valid_extensions:
        result["warnings"].append(f"Unusual file extension: {image_path.suffix}")
    
    # Спробуємо відкрити як зображення
    try:
        info = get_image_info(image_path)
        if "error" in info:
            result["errors"].append(f"Cannot read image: {info['error']}")
            return result
        
        result["info"] = info
        
        # Перевіряємо розмір
        if info["width"] < 256 or info["height"] < 256:
            result["warnings"].append("Image resolution is quite low (< 256px)")
        
        if info["file_size_mb"] > 20:
            result["warnings"].append("Large file size (> 20MB)")
        
        # Перевіряємо пропорції
        if info["aspect_ratio"] < 0.5 or info["aspect_ratio"] > 2.0:
            result["warnings"].append("Unusual aspect ratio")
        
        result["valid"] = True
        
    except Exception as e:
        result["errors"].append(f"Image validation error: {e}")
    
    return result 