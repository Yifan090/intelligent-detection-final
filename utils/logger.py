"""
训练日志记录
"""
import os
import json
import csv
from datetime import datetime
import pandas as pd

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, output_dir='./outputs'):
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # 日志文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.logs_dir, f'training_log_{timestamp}.csv')
        self.json_path = os.path.join(self.logs_dir, f'training_log_{timestamp}.json')
        
        # 初始化CSV文件
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 
                           'val_loss', 'val_acc', 'lr', 'timestamp'])
        
        # 初始化JSON日志
        self.json_log = {
            'start_time': datetime.now().isoformat(),
            'config': {},
            'logs': []
        }
        
        print(f"日志文件已创建: {self.csv_path}")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """记录一个epoch的训练结果"""
        timestamp = datetime.now().isoformat()
        
        # CSV格式
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr, timestamp])
        
        # JSON格式
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr,
            'timestamp': timestamp
        }
        self.json_log['logs'].append(log_entry)
        
        # 定期保存JSON
        if epoch % 10 == 0:
            self.save_json()
    
    def log_config(self, config):
        """记录配置"""
        self.json_log['config'] = vars(config) if hasattr(config, '__dict__') else config
        self.save_json()
    
    def log_metrics(self, metrics):
        """记录额外指标"""
        if 'metrics' not in self.json_log:
            self.json_log['metrics'] = {}
        self.json_log['metrics'].update(metrics)
        self.save_json()
    
    def save_json(self):
        """保存JSON日志"""
        self.json_log['end_time'] = datetime.now().isoformat()
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_log, f, indent=2, ensure_ascii=False)
    
    def get_summary(self):
        """获取训练摘要"""
        if not self.json_log['logs']:
            return {}
        
        logs = self.json_log['logs']
        last_log = logs[-1]
        best_log = max(logs, key=lambda x: x['val_acc'])
        
        summary = {
            'total_epochs': len(logs),
            'best_epoch': best_log['epoch'],
            'best_val_acc': best_log['val_acc'],
            'best_val_loss': best_log['val_loss'],
            'final_train_acc': last_log['train_acc'],
            'final_val_acc': last_log['val_acc'],
            'start_time': self.json_log['start_time'],
            'end_time': self.json_log.get('end_time', datetime.now().isoformat())
        }
        
        return summary
    
    def save_summary(self):
        """保存训练摘要"""
        summary = self.get_summary()
        
        # 保存为TXT
        txt_path = os.path.join(self.logs_dir, 'training_summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("训练摘要\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"开始时间: {summary['start_time']}\n")
            f.write(f"结束时间: {summary['end_time']}\n")
            f.write(f"训练轮数: {summary['total_epochs']}\n\n")
            
            f.write(f"最佳验证准确率: {summary['best_val_acc']:.2f}% (第{summary['best_epoch']}轮)\n")
            f.write(f"最终训练准确率: {summary['final_train_acc']:.2f}%\n")
            f.write(f"最终验证准确率: {summary['final_val_acc']:.2f}%\n")
        
        # 保存为JSON
        json_path = os.path.join(self.logs_dir, 'training_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"训练摘要已保存到: {txt_path}")
        return txt_path

def test_logger():
    """测试日志记录器"""
    logger = TrainingLogger('./test_logs')
    
    # 记录配置
    config = {'batch_size': 64, 'learning_rate': 0.001}
    logger.log_config(config)
    
    # 记录几个epoch
    for epoch in range(1, 4):
        logger.log_epoch(
            epoch=epoch,
            train_loss=1.5/epoch,
            train_acc=50 + epoch*10,
            val_loss=1.6/epoch,
            val_acc=48 + epoch*9,
            lr=0.001
        )
    
    # 保存摘要
    logger.save_summary()
    print("日志记录器测试完成!")

if __name__ == "__main__":
    test_logger()