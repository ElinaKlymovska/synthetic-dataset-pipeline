"""
Трекер прогресу для моніторингу генерації зображень.
"""

import time
from typing import Dict, Any, List, Callable
from ..core import GenerationResult


class ProgressTracker:
    """Трекер прогресу генерації зображень."""
    
    def __init__(self):
        self.total: int = 0
        self.completed: int = 0
        self.successful: int = 0
        self.failed: int = 0
        self.start_time: float = 0
        self.results: List[GenerationResult] = []
        self.callbacks: List[Callable] = []
    
    def start(self, total: int):
        """Ініціалізує трекер з загальною кількістю завдань."""
        self.total = total
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time.time()
        self.results = []
        
        print(f"🚀 Розпочато генерацію {total} зображень...")
    
    def update(self, result: GenerationResult):
        """Оновлює прогрес на основі результату генерації."""
        self.completed += 1
        self.results.append(result)
        
        if result.success:
            self.successful += 1
            print(f"✅ {self.completed}/{self.total} - Успішно: {result.image_path}")
        else:
            self.failed += 1
            print(f"❌ {self.completed}/{self.total} - Помилка: {result.error_message}")
        
        # Показуємо прогрес
        progress_percent = (self.completed / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.completed < self.total:
            # Оцінюємо час до завершення
            avg_time_per_item = elapsed / self.completed
            remaining_time = avg_time_per_item * (self.total - self.completed)
            print(f"📊 Прогрес: {progress_percent:.1f}% | ETA: {remaining_time:.0f}с")
        
        # Викликаємо callbacks
        for callback in self.callbacks:
            callback(self)
    
    def add_callback(self, callback: Callable):
        """Додає callback для отримання оновлень прогресу."""
        self.callbacks.append(callback)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Повертає детальну інформацію про прогрес."""
        elapsed_time = time.time() - self.start_time if self.start_time > 0 else 0
        
        progress_info = {
            "total": self.total,
            "completed": self.completed,
            "successful": self.successful,
            "failed": self.failed,
            "progress_percent": (self.completed / self.total * 100) if self.total > 0 else 0,
            "elapsed_time_seconds": elapsed_time,
            "estimated_remaining_seconds": 0,
            "average_time_per_item": elapsed_time / max(1, self.completed),
            "success_rate": (self.successful / max(1, self.completed)) * 100,
            "is_completed": self.completed >= self.total,
            "total_cost": sum(r.cost or 0 for r in self.results if r.cost),
            "errors": [r.error_message for r in self.results if not r.success and r.error_message]
        }
        
        # Оцінка часу до завершення
        if self.completed > 0 and self.completed < self.total:
            avg_time = elapsed_time / self.completed
            remaining_items = self.total - self.completed
            progress_info["estimated_remaining_seconds"] = avg_time * remaining_items
        
        return progress_info
    
    def __str__(self) -> str:
        """Повертає строкове представлення прогресу."""
        info = self.get_progress_info()
        
        if info["is_completed"]:
            return (f"✅ Завершено: {info['successful']}/{info['total']} успішно "
                   f"({info['success_rate']:.1f}%) за {info['elapsed_time_seconds']:.1f}с")
        else:
            return (f"🔄 Прогрес: {info['completed']}/{info['total']} "
                   f"({info['progress_percent']:.1f}%) | "
                   f"ETA: {info['estimated_remaining_seconds']:.0f}с") 