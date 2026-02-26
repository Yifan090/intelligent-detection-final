import torch
from models.simple_cnn import SimpleCNN
from data.data_loader import get_cifar10_loaders
from utils.visualize import Visualizer
import os

def main():
    """主函数，用于生成图表"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = SimpleCNN(num_classes=10)
    checkpoint_path = 'outputs/checkpoints/best_model.pth'

    if not os.path.exists(checkpoint_path):
        print("错误：未找到最佳模型，请先完成训练。")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("模型加载成功！")

    # 加载测试数据
    print("正在加载测试数据...")
    _, _, test_loader, class_names = get_cifar10_loaders(batch_size=64, data_dir='./data')

    # 收集预测结果
    print("正在收集预测结果...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # 生成所有图表
    print("正在生成图表...")
    vis = Visualizer('outputs')

    # 逐个生成
    vis.plot_confusion_matrix(all_labels, all_preds, class_names)
    vis.plot_classification_report(all_labels, all_preds, class_names)
    vis.plot_error_analysis(test_loader, model, device, class_names, num_samples=10)

    # 单独获取一个批次用于样本图像
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    vis.plot_sample_images(images, labels, class_names)

    print("\n✅ 所有图表生成完成！请查看 outputs/visualizations/ 目录")

if __name__ == '__main__':
    # 这个保护块是解决多进程问题的关键
    # 参考：https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    main()