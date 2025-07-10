"""
GenImg - AI Character Dataset Generator

A modular system for generating diverse character image datasets 
for LoRA training using AI image generation.

Architecture:
- core/: Core interfaces, configuration, and services
- generators/: Image generation implementations
- cli/: Command-line interface
- utils/: Utility functions
- app.py: Main application orchestrator
"""

from .app import GenImgApp, create_app, create_app_with_dependencies
from .cli import CLIManager

__version__ = "2.0.0"
__author__ = "GenImg Team"

__all__ = [
    "GenImgApp",
    "create_app", 
    "create_app_with_dependencies",
    "CLIManager"
]
