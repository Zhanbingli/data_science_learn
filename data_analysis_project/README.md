# 数据分析项目

> 完整的数据分析与机器学习项目框架

## ✨ 特性

- 完整的数据分析工具链（数据加载、EDA、特征工程、建模、评估）
- 模块化、可复用的代码结构
- 论文级别的可视化模板
- 自动化分析脚本
- 支持多种机器学习模型（LR, RF, XGBoost, LightGBM, CatBoost）
- 详细的文档和示例代码

## 🚀 快速开始

### 1. 环境设置

```bash
# 运行自动设置脚本（推荐）
bash setup.sh

# 或手动设置
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 准备数据

将数据文件放入 `data/raw/` 目录：
```
data/raw/
├── train.csv
└── test.csv
```

### 3. 开始分析

#### 方式一：自动化分析（最快）
```bash
python auto_analysis.py --train data/raw/train.csv --target target_column
```
一键生成完整的EDA报告和可视化图表！

#### 方式二：使用Python脚本
```python
from src.utils.data_loader import DataLoader
from src.visualization.plot_templates import EDAPlotter
from src.models.model_trainer import ModelTrainer

# 数据加载
loader = DataLoader()
df = loader.load_train_data()

# 数据可视化
plotter = EDAPlotter()
plotter.plot_numeric_distribution(df, 'age', target='target')

# 模型训练
trainer = ModelTrainer(task_type='classification')
trainer.cross_validate_all(X_train, y_train, cv=5)
```

#### 方式三：Jupyter Notebook
```bash
jupyter lab
# 查看示例: notebooks/01_eda/example_eda.py
#         notebooks/04_modeling/example_modeling.py
```

---

## 📁 项目结构

```
data_analysis_project/
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后的数据
│   └── external/                  # 外部数据源
│
├── notebooks/                     # Jupyter笔记本
│   ├── 01_eda/                    # 探索性数据分析
│   ├── 02_preprocessing/          # 数据预处理
│   ├── 03_feature_engineering/    # 特征工程
│   ├── 04_modeling/               # 模型训练
│   └── 05_evaluation/             # 模型评估
│
├── src/                           # 源代码模块
│   ├── utils/                     # 工具模块
│   │   ├── data_loader.py         # 数据加载
│   │   └── statistical_tests.py   # 统计检验
│   ├── visualization/             # 可视化模块
│   │   └── plot_templates.py      # 绘图模板
│   ├── features/                  # 特征工程模块
│   │   └── feature_engineering.py # 特征工程工具
│   ├── models/                    # 模型模块
│   │   └── model_trainer.py       # 模型训练器
│   └── evaluation/                # 评估模块
│       └── model_evaluator.py     # 模型评估器
│
├── models/                        # 保存的模型
├── reports/                       # 报告和图表
│   ├── figures/
│   └── final/
├── config/                        # 配置文件
│   └── config.yaml
├── docs/                          # 文档
│
├── auto_analysis.py               # 自动化分析脚本
├── setup.sh                       # 环境设置脚本
├── requirements.txt               # 依赖包
├── .gitignore                     # Git忽略文件
├── README.md                      # 本文件
└── PROJECT_STRUCTURE.md           # 详细结构说明
```

详细说明请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 🛠️ 核心模块

### 1. 数据加载 (`src/utils/data_loader.py`)
```python
from src.utils.data_loader import DataLoader

loader = DataLoader()
train_df = loader.load_train_data()
loader.print_data_summary(train_df, '训练集')
```

### 2. 数据可视化 (`src/visualization/plot_templates.py`)
```python
from src.visualization.plot_templates import EDAPlotter

plotter = EDAPlotter()
plotter.plot_numeric_distribution(df, 'age', target='target')
plotter.plot_categorical_distribution(df, 'category')
plotter.plot_correlation_heatmap(df)
```

### 3. 统计检验 (`src/utils/statistical_tests.py`)
```python
from src.utils.statistical_tests import StatisticalTester

tester = StatisticalTester()
result = tester.comprehensive_analysis(df, 'feature', 'target')
tester.print_test_results(result)
```

### 4. 特征工程 (`src/features/feature_engineering.py`)
```python
from src.features.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
df = fe.handle_missing_values(df, strategy='auto')
df = fe.encode_categorical(df, method='auto')
df = fe.create_interaction_features(df, [('col1', 'col2')])
df, info = fe.remove_outliers(df, method='iqr')
```

### 5. 模型训练 (`src/models/model_trainer.py`)
```python
from src.models.model_trainer import ModelTrainer

trainer = ModelTrainer(task_type='classification')

# 交叉验证对比
results = trainer.cross_validate_all(X_train, y_train, cv=5)

# 训练模型
trainer.train_model('LightGBM', X_train, y_train, X_val, y_val)

# 预测
y_pred = trainer.predict('LightGBM', X_test)
y_pred_proba = trainer.predict_proba('LightGBM', X_test)

# 保存模型
trainer.save_model('LightGBM', 'models/lgb_model.pkl')
```

### 6. 模型评估 (`src/evaluation/model_evaluator.py`)
```python
from src.evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(task_type='classification')

# 生成完整评估报告
evaluator.generate_classification_report(
    y_test, y_pred, y_pred_proba,
    labels=['Class 0', 'Class 1'],
    output_dir='reports/evaluation'
)

# 或单独绘制图表
evaluator.plot_confusion_matrix(y_test, y_pred)
evaluator.plot_roc_curve(y_test, y_pred_proba)
evaluator.plot_learning_curve(model, X, y, cv=5)
```

---

## 📊 完整工作流程示例

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 数据加载
from src.utils.data_loader import DataLoader
loader = DataLoader()
df = loader.load_train_data()

# 2. 探索性分析
from src.visualization.plot_templates import EDAPlotter
plotter = EDAPlotter()
plotter.plot_correlation_heatmap(df)

# 3. 特征工程
from src.features.feature_engineering import FeatureEngineer
fe = FeatureEngineer()
df = fe.handle_missing_values(df)
df = fe.encode_categorical(df, method='auto')

# 4. 准备训练数据
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 模型训练
from src.models.model_trainer import ModelTrainer
trainer = ModelTrainer(task_type='classification')
trainer.cross_validate_all(X_train, y_train, cv=5)
trainer.train_model('LightGBM', X_train, y_train, X_test, y_test)

# 6. 模型评估
from src.evaluation.model_evaluator import ModelEvaluator
evaluator = ModelEvaluator(task_type='classification')
y_pred = trainer.predict('LightGBM', X_test)
y_pred_proba = trainer.predict_proba('LightGBM', X_test)
evaluator.generate_classification_report(y_test, y_pred, y_pred_proba)
```

---

## 📝 配置文件

所有参数都可以在 [config/config.yaml](config/config.yaml) 中配置：

- 数据路径
- 特征工程参数（缺失值处理、异常值阈值、编码方式）
- 模型参数
- 交叉验证策略
- 超参数搜索空间

---

## 🎓 学习资源

- [快速入门指南](docs/getting_started.md)
- [知识点清单](docs/knowledge_checklist.md)
- [项目结构详解](PROJECT_STRUCTURE.md)
- [示例代码](notebooks/)

---

## 💡 使用技巧

1. **使用配置文件**: 在 `config/config.yaml` 中管理所有参数
2. **模块化开发**: 将可复用代码放入 `src/` 目录
3. **版本控制**: 使用 git 跟踪代码变更
4. **文档记录**: 在 notebook 中记录实验过程和思考
5. **自动化**: 优先使用 `auto_analysis.py` 进行初步分析

---

## 🔧 依赖库

核心依赖：
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- lightgbm >= 3.3.0
- xgboost >= 1.7.0
- catboost >= 1.1.0

完整列表见 [requirements.txt](requirements.txt)

---

## 📜 License

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题或建议，欢迎联系。

---

**Happy Coding! 🎉**
