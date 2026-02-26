"""
评估指标计算
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_metrics(y_true, y_pred, class_names):
    """计算评估指标"""
    metrics = {}
    
    # 整体指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    
    # 每个类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # 存储每个类别的指标
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': precision_per_class[i] * 100,
            'recall': recall_per_class[i] * 100,
            'f1_score': f1_per_class[i] * 100,
            'support': np.sum(np.array(y_true) == i)
        }
    
    metrics['class_metrics'] = class_metrics
    
    # 平均指标
    metrics['precision_macro'] = np.mean(precision_per_class) * 100
    metrics['recall_macro'] = np.mean(recall_per_class) * 100
    metrics['f1_macro'] = np.mean(f1_per_class) * 100
    
    # 加权平均指标
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted') * 100
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted') * 100
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted') * 100
    
    # 为了方便打印，添加一些别名
    metrics['accuracy_avg'] = metrics['accuracy']
    metrics['precision_avg'] = metrics['precision_macro']
    metrics['recall_avg'] = metrics['recall_macro']
    metrics['f1_avg'] = metrics['f1_macro']
    
    return metrics

def print_detailed_metrics(metrics, class_names):
    """打印详细指标"""
    print("\n详细评估指标:")
    print("-" * 80)
    print(f"{'类别':<10} {'精确率 (%)':<12} {'召回率 (%)':<12} {'F1分数 (%)':<12} {'样本数':<10}")
    print("-" * 80)
    
    for class_name in class_names:
        class_metric = metrics['class_metrics'][class_name]
        print(f"{class_name:<10} "
              f"{class_metric['precision']:<12.2f} "
              f"{class_metric['recall']:<12.2f} "
              f"{class_metric['f1_score']:<12.2f} "
              f"{class_metric['support']:<10}")
    
    print("-" * 80)
    print(f"{'宏平均':<10} "
          f"{metrics['precision_macro']:<12.2f} "
          f"{metrics['recall_macro']:<12.2f} "
          f"{metrics['f1_macro']:<12.2f} ")
    
    print(f"{'加权平均':<10} "
          f"{metrics['precision_weighted']:<12.2f} "
          f"{metrics['recall_weighted']:<12.2f} "
          f"{metrics['f1_weighted']:<12.2f} ")
    
    print(f"{'准确率':<10} {'':<12} {'':<12} {metrics['accuracy']:<12.2f} ")
    print("-" * 80)

def analyze_confusions(y_true, y_pred, class_names, top_n=5):
    """分析最常见的混淆对"""
    from collections import Counter
    
    confusions = []
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            confusions.append((true, pred))
    
    # 统计最常见的混淆
    confusion_counter = Counter(confusions)
    most_common = confusion_counter.most_common(top_n)
    
    print(f"\n最常见的 {top_n} 个混淆:")
    print("-" * 50)
    print(f"{'真实类别':<10} {'预测类别':<10} {'次数':<10} {'百分比 (%)':<10}")
    print("-" * 50)
    
    total_confusions = len(confusions)
    for (true_idx, pred_idx), count in most_common:
        percentage = count / total_confusions * 100 if total_confusions > 0 else 0
        print(f"{class_names[true_idx]:<10} "
              f"{class_names[pred_idx]:<10} "
              f"{count:<10} "
              f"{percentage:<10.2f}")
    
    print("-" * 50)
    print(f"总错误数: {total_confusions}")
    print(f"总样本数: {len(y_true)}")
    print(f"错误率: {total_confusions/len(y_true)*100:.2f}%")
    
    return most_common

def test_metrics():
    """测试评估指标"""
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2, 0, 1, 2]
    class_names = ['A', 'B', 'C']
    
    metrics = compute_metrics(y_true, y_pred, class_names)
    print_detailed_metrics(metrics, class_names)
    
    print("\n混淆分析:")
    analyze_confusions(y_true, y_pred, class_names)
    
    return metrics

if __name__ == "__main__":
    test_metrics()