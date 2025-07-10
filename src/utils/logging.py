"""
Утиліти для логування та діагностики системи.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[str] = None, 
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """Налаштовує систему логування."""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


def log_system_info(logger: logging.Logger) -> Dict[str, Any]:
    """Логує та повертає інформацію про систему."""
    import platform
    import psutil
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "disk_space_gb": round(psutil.disk_usage('.').free / (1024**3), 1),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"System info: {system_info}")
    return system_info 