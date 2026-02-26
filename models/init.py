"""
模型包
提供统一的模型导入接口
"""

from .simple_cnn import SimpleCNN
from .resnet_model import ResNet18, BasicBlock

# 模型注册表
MODEL_REGISTRY = {
    'SimpleCNN': SimpleCNN,
    'ResNet18': ResNet18,
}

def get_model(model_name, num_classes=10, **kwargs):
    """根据名称获取模型实例"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}")
    return MODEL_REGISTRY[model_name](num_classes=num_classes, **kwargs)

def list_models():
    """列出所有可用模型"""
    return list(MODEL_REGISTRY.keys())

__all__ = ['SimpleCNN', 'ResNet18', 'BasicBlock', 'get_model', 'list_models']