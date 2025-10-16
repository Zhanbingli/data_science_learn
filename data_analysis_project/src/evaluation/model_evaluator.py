"""
模型评估模块
提供完整的模型评估和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器类"""

    def __init__(self, task_type='classification'):
        """
        初始化评估器

        Parameters:
        -----------
        task_type : str
            任务类型：'classification' 或 'regression'
        """
        self.task_type = task_type
        self.setup_style()

    def setup_style(self):
        """设置绘图样式"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['figure.titlesize'] = 18

    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """
        评估分类模型

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        y_pred_proba : array-like, optional
            预测概率

        Returns:
        --------
        dict
            评估指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # 如果是二分类且提供了概率
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            if y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def evaluate_regression(self, y_true, y_pred):
        """
        评估回归模型

        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值

        Returns:
        --------
        dict
            评估指标字典
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
        }

        # 避免除零错误
        if not np.all(y_true == 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100

        return metrics

    def print_metrics(self, metrics):
        """
        打印评估指标

        Parameters:
        -----------
        metrics : dict
            评估指标字典
        """
        print("\n" + "="*60)
        print("模型评估指标")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric.upper():15s}: {value:.4f}")
        print("="*60)

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        """
        绘制混淆矩阵

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        labels : list, optional
            类别标签
        save_path : str, optional
            保存路径
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix', fontweight='bold', pad=20)
        plt.ylabel('True Label', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        绘制ROC曲线

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred_proba : array-like
            预测概率
        save_path : str, optional
            保存路径
        """
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curve', fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        绘制Precision-Recall曲线

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred_proba : array-like
            预测概率
        save_path : str, optional
            保存路径
        """
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=3, label='PR Curve')

        plt.xlabel('Recall', fontweight='bold')
        plt.ylabel('Precision', fontweight='bold')
        plt.title('Precision-Recall Curve', fontweight='bold', pad=20)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_residuals(self, y_true, y_pred, save_path=None):
        """
        绘制残差图（回归任务）

        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        save_path : str, optional
            保存路径
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 残差散点图
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontweight='bold')
        axes[0].set_ylabel('Residuals', fontweight='bold')
        axes[0].set_title('Residual Plot', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 残差直方图
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Residual Distribution', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_prediction_vs_actual(self, y_true, y_pred, save_path=None):
        """
        绘制预测值vs真实值图（回归任务）

        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        save_path : str, optional
            保存路径
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')

        # 绘制理想线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect Prediction')

        plt.xlabel('True Values', fontweight='bold')
        plt.ylabel('Predicted Values', fontweight='bold')
        plt.title('Prediction vs Actual', fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加R²值
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}',
                transform=plt.gca().transAxes,
                fontsize=14, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_learning_curve(self, estimator, X, y, cv=5, save_path=None):
        """
        绘制学习曲线

        Parameters:
        -----------
        estimator : object
            模型对象
        X : array-like
            特征
        y : array-like
            目标
        cv : int
            交叉验证折数
        save_path : str, optional
            保存路径
        """
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy' if self.task_type == 'classification' else 'r2'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 8))
        plt.plot(train_sizes, train_mean, 'o-', linewidth=2, label='Training Score')
        plt.plot(train_sizes, val_mean, 'o-', linewidth=2, label='Validation Score')

        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

        plt.xlabel('Training Set Size', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.title('Learning Curve', fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_classification_report(self, y_true, y_pred, y_pred_proba=None,
                                      labels=None, output_dir=None):
        """
        生成完整的分类报告

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        y_pred_proba : array-like, optional
            预测概率
        labels : list, optional
            类别标签
        output_dir : str, optional
            输出目录
        """
        print("\n" + "="*80)
        print("分类模型评估报告")
        print("="*80)

        # 计算指标
        metrics = self.evaluate_classification(y_true, y_pred, y_pred_proba)
        self.print_metrics(metrics)

        # 详细分类报告
        print("\n详细分类报告:")
        print(classification_report(y_true, y_pred, target_names=labels))

        # 绘图
        if output_dir:
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            self.plot_confusion_matrix(y_true, y_pred, labels,
                                      save_path=f"{output_dir}/confusion_matrix.png")

            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                self.plot_roc_curve(y_true, y_pred_proba,
                                   save_path=f"{output_dir}/roc_curve.png")
                self.plot_precision_recall_curve(y_true, y_pred_proba,
                                                 save_path=f"{output_dir}/pr_curve.png")

    def generate_regression_report(self, y_true, y_pred, output_dir=None):
        """
        生成完整的回归报告

        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        output_dir : str, optional
            输出目录
        """
        print("\n" + "="*80)
        print("回归模型评估报告")
        print("="*80)

        # 计算指标
        metrics = self.evaluate_regression(y_true, y_pred)
        self.print_metrics(metrics)

        # 绘图
        if output_dir:
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            self.plot_prediction_vs_actual(y_true, y_pred,
                                          save_path=f"{output_dir}/prediction_vs_actual.png")
            self.plot_residuals(y_true, y_pred,
                               save_path=f"{output_dir}/residuals.png")


if __name__ == "__main__":
    # 示例用法
    from sklearn.datasets import load_breast_cancer, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # 分类任务示例
    print("分类任务示例")
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    evaluator = ModelEvaluator(task_type='classification')
    evaluator.generate_classification_report(y_test, y_pred, y_pred_proba)

    # 回归任务示例
    print("\n\n回归任务示例")
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    evaluator = ModelEvaluator(task_type='regression')
    evaluator.generate_regression_report(y_test, y_pred)
