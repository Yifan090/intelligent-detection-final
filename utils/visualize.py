import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 SimHei
matplotlib.rcParams['axes.unicode_minus'] = False
"""
Visualization utilities with English labels (to avoid font issues)
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch

class Visualizer:
    """Visualizer class for training plots and evaluation"""

    def __init__(self, output_dir='./outputs'):
        self.output_dir = output_dir
        self.vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_training_curves(self, train_losses, train_accs, val_losses, val_accs,
                           title='Training and Validation Curves'):
        """Plot training loss and accuracy curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(train_losses) + 1)

        # Loss curve
        axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o')
        axes[0].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy curve
        axes[1].plot(epochs, train_accs, 'g-', linewidth=2, label='Train Acc', marker='o')
        axes[1].plot(epochs, val_accs, 'orange', linewidth=2, label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved to {save_path}")
        return save_path

    def plot_confusion_matrix(self, y_true, y_pred, class_names, title='Confusion Matrix'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to {save_path}")
        return save_path

    def plot_classification_report(self, y_true, y_pred, class_names, title='Classification Report'):
        """Plot classification report as heatmap"""
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        for cls in class_names:
            row = [report[cls][metric] for metric in metrics]
            data.append(row)
        data = np.array(data)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(['Precision', 'Recall', 'F1-score'], fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)

        for i in range(len(class_names)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)

        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, 'classification_report.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Classification report saved to {save_path}")
        return save_path

    def plot_error_analysis(self, dataloader, model, device, class_names,
                          num_samples=10, title='Error Analysis'):
        """Plot misclassified examples"""
        model.eval()
        errors = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)

                wrong_mask = preds != targets
                wrong_inputs = inputs[wrong_mask]
                wrong_preds = preds[wrong_mask]
                wrong_targets = targets[wrong_mask]

                for i in range(len(wrong_inputs)):
                    if len(errors) >= num_samples:
                        break
                    errors.append({
                        'image': wrong_inputs[i].cpu(),
                        'pred': wrong_preds[i].cpu().item(),
                        'target': wrong_targets[i].cpu().item()
                    })
                if len(errors) >= num_samples:
                    break

        if not errors:
            print("No misclassified samples found.")
            return None

        n_cols = 5
        n_rows = min(2, (len(errors) + n_cols - 1) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, error in enumerate(errors[:n_rows*n_cols]):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            img = error['image'].numpy().transpose(1, 2, 0)
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            ax.set_title(f'Pred: {class_names[error["pred"]]}\nTrue: {class_names[error["target"]]}')
            ax.axis('off')

        for idx in range(len(errors), n_rows*n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        fig.suptitle(title)
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, 'error_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Error analysis saved to {save_path}")
        return save_path

    def plot_sample_images(self, images, labels, class_names, title='Sample Images'):
        """Plot sample images from a batch"""
        n_cols = 5
        n_rows = min(2, (len(images) + n_cols - 1) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx in range(min(10, len(images))):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            img = images[idx].numpy().transpose(1, 2, 0)
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            label = labels[idx].item()
            ax.set_title(f'{class_names[label]}')
            ax.axis('off')

        fig.suptitle(title)
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, 'sample_images.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Sample images saved to {save_path}")
        return save_path