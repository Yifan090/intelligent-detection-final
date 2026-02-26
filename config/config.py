"""
配置文件管理
"""
import os
import yaml
import json
import argparse
from dataclasses import dataclass, asdict

@dataclass
class TrainingConfig:
    # 数据配置
    data_dir: str = './data'
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 4
    
    # 模型配置
    model_name: str = 'SimpleCNN'
    use_pretrained: bool = False
    
    # 训练配置
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # 学习率调度
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    
    # 设备配置
    use_gpu: bool = True
    seed: int = 42
    
    # 保存配置
    output_dir: str = './outputs'
    save_interval: int = 10
    log_interval: int = 10
    
    # 可视化配置
    visualize: bool = True
    plot_style: str = 'seaborn'

def get_config(args=None):
    """获取配置"""
    # 创建默认配置
    config = TrainingConfig()
    
    # 从命令行参数更新
    if args:
        if hasattr(args, 'data_dir') and args.data_dir:
            config.data_dir = args.data_dir
        if hasattr(args, 'output_dir') and args.output_dir:
            config.output_dir = args.output_dir
        if hasattr(args, 'model') and args.model:
            config.model_name = args.model
        if hasattr(args, 'epochs') and args.epochs:
            config.num_epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            config.batch_size = args.batch_size
        if hasattr(args, 'lr') and args.lr:
            config.learning_rate = args.lr
        if hasattr(args, 'gpu'):
            config.use_gpu = args.gpu
    
    # 从YAML文件加载（如果存在）
    if hasattr(args, 'config') and args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # 更新配置
            for key, value in yaml_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    return config

def save_config(config, path):
    """保存配置到文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存为JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        
        # 保存为YAML
        yaml_path = path.replace('.json', '.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, allow_unicode=True)
        
        print(f"配置已保存到: {path}")
        print(f"配置已保存到: {yaml_path}")
    except Exception as e:
        print(f"保存配置失败: {e}")

def print_config(config):
    """打印配置"""
    print("\n训练配置:")
    print("-" * 40)
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print("-" * 40)

if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print_config(config)