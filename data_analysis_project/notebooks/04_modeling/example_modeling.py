"""
示例建模脚本
演示如何使用项目的模型训练和评估工具
"""

import sys
sys.path.append('../..')

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.models.model_trainer import ModelTrainer
from src.evaluation.model_evaluator import ModelEvaluator

# 设置随机种子
np.random.seed(42)

# 加载数据
print("="*80)
print("加载数据集")
print("="*80)

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"类别分布: {np.bincount(y)}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 初始化训练器
print("\n" + "="*80)
print("初始化模型训练器")
print("="*80)

trainer = ModelTrainer(task_type='classification', random_state=42)

# 交叉验证所有模型
print("\n" + "="*80)
print("交叉验证模型对比")
print("="*80)

cv_results = trainer.cross_validate_all(X_train, y_train, cv=5, scoring='accuracy')

# 训练最佳模型
print("\n" + "="*80)
print("训练最佳模型")
print("="*80)

best_model_name = cv_results.iloc[0]['Model']
print(f"选择最佳模型: {best_model_name}")

trainer.train_model(best_model_name, X_train, y_train, X_test, y_test)

# 预测
print("\n" + "="*80)
print("模型预测")
print("="*80)

y_pred = trainer.predict(best_model_name, X_test)
y_pred_proba = trainer.predict_proba(best_model_name, X_test)

print(f"预测完成，预测样本数: {len(y_pred)}")

# 评估模型
print("\n" + "="*80)
print("模型评估")
print("="*80)

evaluator = ModelEvaluator(task_type='classification')
evaluator.generate_classification_report(
    y_test, y_pred, y_pred_proba,
    labels=['Malignant', 'Benign']
)

# 特征重要性
print("\n" + "="*80)
print("特征重要性")
print("="*80)

importance_df = trainer.get_feature_importance(
    best_model_name,
    feature_names=feature_names,
    top_n=10
)

print("\nTop 10 最重要特征:")
print(importance_df.to_string(index=False))

# 保存模型
print("\n" + "="*80)
print("保存模型")
print("="*80)

model_path = f"../../models/{best_model_name}_breast_cancer.pkl"
trainer.save_model(best_model_name, model_path)

print("\n✅ 建模示例完成!")
