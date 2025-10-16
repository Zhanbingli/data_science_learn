"""
示例EDA脚本
演示如何使用项目的EDA工具进行探索性数据分析
"""

import sys
sys.path.append('../..')

import pandas as pd
import numpy as np
from src.utils.data_loader import DataLoader
from src.visualization.plot_templates import EDAPlotter
from src.utils.statistical_tests import StatisticalTester

# 设置随机种子
np.random.seed(42)

# 创建示例数据
print("创建示例数据...")
df = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'city': np.random.choice(['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'], 1000),
    'score': np.random.normal(75, 15, 1000),
    'target': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
})

print(f"数据形状: {df.shape}")
print("\n前5行数据:")
print(df.head())

# 初始化工具
plotter = EDAPlotter()
tester = StatisticalTester()

# 1. 数值特征分析
print("\n" + "="*80)
print("1. 数值特征分析")
print("="*80)

plotter.plot_numeric_distribution(df, 'age', target='target')
plotter.plot_numeric_distribution(df, 'income', target='target')
plotter.plot_numeric_distribution(df, 'score', target='target')

# 2. 类别特征分析
print("\n" + "="*80)
print("2. 类别特征分析")
print("="*80)

plotter.plot_categorical_distribution(df, 'education')
plotter.plot_categorical_distribution(df, 'city')

# 3. 目标变量分析
print("\n" + "="*80)
print("3. 目标变量分析")
print("="*80)

plotter.plot_target_analysis(df, 'target')

# 4. 相关性分析
print("\n" + "="*80)
print("4. 相关性分析")
print("="*80)

numeric_df = df.select_dtypes(include=[np.number])
plotter.plot_correlation_heatmap(numeric_df)

# 5. 统计检验
print("\n" + "="*80)
print("5. 统计显著性检验")
print("="*80)

# 数值 vs 类别
result = tester.comprehensive_analysis(df, 'age', 'target')
tester.print_test_results(result)

# 类别 vs 类别
result = tester.comprehensive_analysis(df, 'education', 'target')
tester.print_test_results(result)

print("\n✅ EDA示例完成!")
