"""
Утиліти для роботи з файлами та директоріями.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime


def ensure_directory(path: Union[str, Path]) -> Path:
    """Створює директорію якщо вона не існує."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_directory(path: Union[str, Path], keep_files: Optional[List[str]] = None) -> None:
    """Очищає директорію, зберігаючи вказані файли."""
    path = Path(path)
    if not path.exists():
        return
    
    keep_files = keep_files or []
    
    for item in path.iterdir():
        if item.name not in keep_files:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Повертає розмір файлу в мегабайтах."""
    try:
        size_bytes = Path(file_path).stat().st_size
        return round(size_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0


def create_backup(source_dir: Union[str, Path], backup_dir: Union[str, Path]) -> str:
    """Створює резервну копію директорії."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}"
    backup_path = Path(backup_dir) / backup_name
    
    shutil.copytree(source_dir, backup_path)
    return str(backup_path)


def create_archive(source_dir: Union[str, Path], archive_path: Union[str, Path]) -> str:
    """Створює ZIP архів з директорії."""
    source_dir = Path(source_dir)
    archive_path = Path(archive_path)
    
    ensure_directory(archive_path.parent)
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(source_dir)
                zipf.write(file_path, relative_path)
    
    return str(archive_path) 