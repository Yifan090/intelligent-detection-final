"""
配置包
提供配置管理功能
"""

from .config import TrainingConfig, get_config, save_config, print_config

__all__ = ['TrainingConfig', 'get_config', 'save_config', 'print_config']