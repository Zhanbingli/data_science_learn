# 项目结构说明

## 目录树

```
data_analysis_project/
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据（不可修改）
│   ├── processed/                 # 处理后的数据
│   └── external/                  # 外部数据源
│
├── notebooks/                     # Jupyter笔记本
│   ├── 01_eda/                    # 探索性数据分析
│   ├── 02_preprocessing/          # 数据预处理
│   ├── 03_feature_engineering/    # 特征工程
│   ├── 04_modeling/               # 模型训练
│   └── 05_evaluation/             # 模型评估与优化
│
├── src/                           # 源代码模块
│   ├── __init__.py
│   ├── data/                      # 数据处理模块
│   │   └── __init__.py
│   ├── features/                  # 特征工程模块
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/                    # 模型相关模块
│   │   ├── __init__.py
│   │   └── model_trainer.py
│   ├── evaluation/                # 评估模块
│   │   ├── __init__.py
│   │   └── model_evaluator.py
│   ├── utils/                     # 工具模块
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── statistical_tests.py
│   └── visualization/             # 可视化模块
│       ├── __init__.py
│       └── plot_templates.py
│
├── models/                        # 保存的模型文件
│
├── reports/                       # 报告和可视化结果
│   ├── figures/                   # 图表
│   └── final/                     # 最终报告
│
├── config/                        # 配置文件
│   └── config.yaml
│
├── tests/                         # 单元测试
│
├── docs/                          # 文档和学习笔记
│   ├── getting_started.md
│   └── knowledge_checklist.md
│
├── .gitignore                     # Git忽略文件
├── requirements.txt               # Python依赖包
├── setup.sh                       # 环境设置脚本
├── auto_analysis.py               # 自动化分析脚本
├── README.md                      # 项目说明
└── PROJECT_STRUCTURE.md           # 本文件
```

## 核心模块说明

### 1. src/utils/ - 工具模块

#### data_loader.py
- **DataLoader类**: 统一的数据加载接口
  - `load_train_data()`: 加载训练数据
  - `load_test_data()`: 加载测试数据
  - `get_data_info()`: 获取数据基本信息
  - `print_data_summary()`: 打印数据摘要

- **reduce_mem_usage()**: 内存优化函数

#### statistical_tests.py
- **StatisticalTester类**: 统计检验工具
  - `test_normality()`: 正态性检验
  - `test_two_groups()`: 两组比较检验
  - `test_multiple_groups()`: 多组比较检验
  - `test_categorical_association()`: 卡方检验
  - `test_correlation()`: 相关性检验
  - `comprehensive_analysis()`: 综合统计分析

### 2. src/visualization/ - 可视化模块

#### plot_templates.py
- **EDAPlotter类**: EDA可视化工具
  - `plot_numeric_distribution()`: 数值特征分布
  - `plot_categorical_distribution()`: 类别特征分布
  - `plot_correlation_heatmap()`: 相关性热力图
  - `plot_target_analysis()`: 目标变量分析
  - `plot_feature_importance()`: 特征重要性图

### 3. src/features/ - 特征工程模块

#### feature_engineering.py
- **FeatureEngineer类**: 特征工程工具
  - `handle_missing_values()`: 缺失值处理
  - `remove_outliers()`: 异常值处理
  - `encode_categorical()`: 类别编码
  - `scale_features()`: 特征缩放
  - `create_polynomial_features()`: 多项式特征
  - `create_interaction_features()`: 交互特征
  - `bin_numeric_features()`: 数值分箱
  - `select_features_by_variance()`: 方差选择
  - `select_features_by_correlation()`: 相关性选择

### 4. src/models/ - 模型训练模块

#### model_trainer.py
- **ModelTrainer类**: 模型训练器
  - `add_model()`: 添加模型
  - `get_default_models()`: 获取默认模型集
  - `train_model()`: 训练单个模型
  - `train_all_models()`: 训练所有模型
  - `cross_validate()`: 交叉验证
  - `cross_validate_all()`: 批量交叉验证
  - `predict()`: 模型预测
  - `predict_proba()`: 概率预测
  - `save_model()`: 保存模型
  - `load_model()`: 加载模型
  - `get_feature_importance()`: 获取特征重要性

### 5. src/evaluation/ - 评估模块

#### model_evaluator.py
- **ModelEvaluator类**: 模型评估器
  - `evaluate_classification()`: 分类评估
  - `evaluate_regression()`: 回归评估
  - `plot_confusion_matrix()`: 混淆矩阵
  - `plot_roc_curve()`: ROC曲线
  - `plot_precision_recall_curve()`: PR曲线
  - `plot_residuals()`: 残差图
  - `plot_prediction_vs_actual()`: 预测vs实际
  - `plot_learning_curve()`: 学习曲线
  - `generate_classification_report()`: 分类报告
  - `generate_regression_report()`: 回归报告

## 工作流程

### 1. 数据探索 (notebooks/01_eda/)
```python
from src.utils.data_loader import DataLoader
from src.visualization.plot_templates import EDAPlotter

loader = DataLoader()
df = loader.load_train_data()

plotter = EDAPlotter()
plotter.plot_numeric_distribution(df, 'feature_name')
```

### 2. 特征工程 (notebooks/03_feature_engineering/)
```python
from src.features.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
df = fe.handle_missing_values(df)
df = fe.encode_categorical(df)
df = fe.create_interaction_features(df, [('col1', 'col2')])
```

### 3. 模型训练 (notebooks/04_modeling/)
```python
from src.models.model_trainer import ModelTrainer

trainer = ModelTrainer(task_type='classification')
trainer.cross_validate_all(X_train, y_train, cv=5)
trainer.train_model('LightGBM', X_train, y_train, X_val, y_val)
```

### 4. 模型评估 (notebooks/05_evaluation/)
```python
from src.evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(task_type='classification')
evaluator.generate_classification_report(y_test, y_pred, y_pred_proba)
```

## 配置文件说明

### config/config.yaml
包含项目的所有配置：
- 数据路径配置
- 特征工程参数
- 模型参数
- 超参数搜索空间
- 输出路径配置

## 自动化工具

### auto_analysis.py
一键自动分析脚本，支持：
- 数据概览
- 数据质量检查
- 单变量分析
- 双变量分析
- 多变量分析
- 统计检验
- 自动生成报告

使用方法：
```bash
python auto_analysis.py --train data/raw/train.csv --target target_column
```

## 最佳实践

1. **模块化开发**: 将可复用代码放入src/目录
2. **版本控制**: 使用git管理代码变更
3. **文档记录**: 在notebook中记录思考过程
4. **代码复用**: 优先使用项目提供的工具类
5. **参数配置**: 在config.yaml中管理参数
6. **结果保存**: 统一保存到reports/目录

## 扩展建议

1. 在`src/models/`下添加自定义模型类
2. 在`src/features/`下添加领域特定的特征工程函数
3. 在`tests/`目录下添加单元测试
4. 在`docs/`目录下记录学习笔记和项目文档
