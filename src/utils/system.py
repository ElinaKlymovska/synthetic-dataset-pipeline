"""
Утиліти для системних перевірок та валідації.
"""

import os
import sys
from typing import Dict, Any


def check_system_requirements() -> Dict[str, Any]:
    """Перевіряє системні вимоги."""
    requirements = {
        "python_version": (3, 8),
        "min_memory_gb": 4,
        "min_disk_space_gb": 2
    }
    
    result = {
        "meets_requirements": True,
        "checks": {},
        "recommendations": []
    }
    
    # Перевіряємо версію Python
    python_version = sys.version_info[:2]
    result["checks"]["python_version"] = {
        "current": f"{python_version[0]}.{python_version[1]}",
        "required": f"{requirements['python_version'][0]}.{requirements['python_version'][1]}",
        "passed": python_version >= requirements["python_version"]
    }
    
    if not result["checks"]["python_version"]["passed"]:
        result["meets_requirements"] = False
        result["recommendations"].append("Upgrade Python to version 3.8 or higher")
    
    # Перевіряємо пам'ять
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)
        result["checks"]["memory"] = {
            "current_gb": round(available_memory, 1),
            "required_gb": requirements["min_memory_gb"],
            "passed": available_memory >= requirements["min_memory_gb"]
        }
        
        if not result["checks"]["memory"]["passed"]:
            result["meets_requirements"] = False
            result["recommendations"].append("Insufficient memory available")
    except ImportError:
        result["checks"]["memory"] = {"error": "psutil not available"}
    
    # Перевіряємо дисковий простір
    try:
        import psutil
        disk_space = psutil.disk_usage('.').free / (1024**3)
        result["checks"]["disk_space"] = {
            "current_gb": round(disk_space, 1),
            "required_gb": requirements["min_disk_space_gb"],
            "passed": disk_space >= requirements["min_disk_space_gb"]
        }
        
        if not result["checks"]["disk_space"]["passed"]:
            result["meets_requirements"] = False
            result["recommendations"].append("Insufficient disk space")
    except ImportError:
        result["checks"]["disk_space"] = {"error": "psutil not available"}
    
    return result


def validate_api_credentials() -> Dict[str, bool]:
    """Перевіряє API креденціали."""
    checks = {}
    
    # Перевіряємо Replicate API токен
    replicate_token = os.getenv('REPLICATE_API_TOKEN')
    checks["replicate_token"] = replicate_token is not None and len(replicate_token) > 10
    
    return checks 