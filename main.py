#!/usr/bin/env python3
"""
智能检测项目主入口文件
从零开始的图像分类项目 - 符合RI实验室考核要求
"""
import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from config.config import get_config, save_config
from data.data_loader import get_cifar10_loaders
from models.simple_cnn import SimpleCNN
from models.resnet_model import ResNet18
from utils.logger import TrainingLogger
from utils.visualize import Visualizer
from utils.metrics import compute_metrics

def setup_environment(config):
    """设置运行环境"""
    print("=" * 60)
    print("智能检测项目 - 图像分类系统")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'visualizations'), exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(config.output_dir, 'config.json')
    save_config(config, config_path)
    print(f"配置已保存到: {config_path}")
    
    return device

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 打印进度
        if batch_idx % 50 == 0:
            print(f'Epoch [{epoch}/{total_epochs}] '
                  f'Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = total_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_targets

def main(config):
    """主训练函数"""
    # 1. 设置环境
    device = setup_environment(config)
    
    # 2. 加载数据
    print("\n[1/5] 加载数据集...")
    train_loader, val_loader, test_loader, class_names = get_cifar10_loaders(
        batch_size=config.batch_size,
        data_dir=config.data_dir,
        num_workers=config.num_workers
    )
    print(f"训练集: {len(train_loader.dataset)} 张图片")
    print(f"验证集: {len(val_loader.dataset)} 张图片")
    print(f"测试集: {len(test_loader.dataset)} 张图片")
    print(f"类别: {class_names}")
    
    # 3. 创建模型
    print("\n[2/5] 创建模型...")
    if config.model_name == 'SimpleCNN':
        model = SimpleCNN(num_classes=config.num_classes)
    elif config.model_name == 'ResNet18':
        model = ResNet18(num_classes=config.num_classes)
    else:
        raise ValueError(f"未知模型: {config.model_name}")
    
    model = model.to(device)
    print(f"模型: {model.__class__.__name__}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate,
                          weight_decay=config.weight_decay)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=config.lr_step_size,
                                         gamma=config.lr_gamma)
    
    # 5. 创建日志记录器
    logger = TrainingLogger(config.output_dir)
    visualizer = Visualizer(config.output_dir)
    writer = SummaryWriter(os.path.join(config.output_dir, 'tensorboard'))
    
    # 6. 训练循环
    print("\n[3/5] 开始训练...")
    print("-" * 60)
    
    best_val_acc = 0.0
    train_history = []
    val_history = []
    
    for epoch in range(1, config.num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch, config.num_epochs
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )
        
        # 学习率调整
        scheduler.step()
        
        # 记录历史
        train_history.append({
            'epoch': epoch,
            'loss': train_loss,
            'acc': train_acc
        })
        
        val_history.append({
            'epoch': epoch,
            'loss': val_loss,
            'acc': val_acc
        })
        
        # 打印结果
        print(f"\nEpoch {epoch:3d}/{config.num_epochs}:")
        print(f"  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"  验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            torch.save(checkpoint, 
                      os.path.join(config.output_dir, 'checkpoints', 'best_model.pth'))
            print(f"  ✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        # 定期保存检查点
        if epoch % config.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            torch.save(checkpoint, 
                      os.path.join(config.output_dir, 'checkpoints', f'model_epoch_{epoch}.pth'))
    
    writer.close()
    
    # 7. 最终测试
    print("\n[4/5] 在测试集上评估最佳模型...")
    best_checkpoint = torch.load(
        os.path.join(config.output_dir, 'checkpoints', 'best_model.pth'),
        map_location=device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_targets = validate(
        model, test_loader, criterion, device
    )
    
    print(f"测试集结果:")
    print(f"  损失: {test_loss:.4f}")
    print(f"  准确率: {test_acc:.2f}%")
    
    # 8. 计算详细指标
    metrics = compute_metrics(test_targets, test_preds, class_names)
    print("\n详细指标:")
    print(f"  精确率 (平均): {metrics['precision_avg']:.2f}%")
    print(f"  召回率 (平均): {metrics['recall_avg']:.2f}%")
    print(f"  F1分数 (平均): {metrics['f1_avg']:.2f}%")
    
    # 9. 可视化结果
    print("\n[5/5] 生成可视化图表...")
    
    # 训练曲线
    train_losses = [h['loss'] for h in train_history]
    train_accs = [h['acc'] for h in train_history]
    val_losses = [h['loss'] for h in val_history]
    val_accs = [h['acc'] for h in val_history]
    
    visualizer.plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        title='训练和验证曲线'
    )
    
    # 混淆矩阵
    visualizer.plot_confusion_matrix(
        test_targets, test_preds, class_names,
        title='测试集混淆矩阵'
    )
    
    # 类别报告
    visualizer.plot_classification_report(
        test_targets, test_preds, class_names,
        title='分类报告'
    )
    
    # 错误分析
    visualizer.plot_error_analysis(
        test_loader, model, device, class_names,
        num_samples=10,
        title='错误样本分析'
    )
    
    # 10. 保存最终结果
    print("\n保存最终结果...")
    results = {
        'config': vars(config),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'metrics': metrics,
        'train_history': train_history,
        'val_history': val_history,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = os.path.join(config.output_dir, 'final_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {results_path}")
    
    # 11. 输出总结
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"训练轮数: {config.num_epochs}")
    print(f"输出目录: {config.output_dir}")
    print("=" * 60)
    
    return model, results

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='智能检测项目 - 图像分类')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--model', type=str, default='SimpleCNN',
                       choices=['SimpleCNN', 'ResNet18'],
                       help='模型选择')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--gpu', action='store_true',
                       help='使用GPU')
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args)
    
    try:
        model, results = main(config)
        print("\n🎉 项目运行成功！")
        print("所有文件已保存在输出目录中。")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)