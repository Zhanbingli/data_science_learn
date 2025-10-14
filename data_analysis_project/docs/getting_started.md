# 快速开始指南

本指南将帮助你快速启动数据分析学习项目。

---

## 🚀 第一步：环境配置

### 1. 创建虚拟环境

```bash
# 使用conda（推荐）
conda create -n data_analysis python=3.10
conda activate data_analysis

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
cd data_analysis_project
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import pandas, numpy, sklearn, lightgbm; print('✅ 所有库安装成功')"
```

---

## 📁 第二步：准备数据

### 1. 获取竞赛数据

- 从Kaggle或其他平台下载数据
- 将数据文件放入 `data/raw/` 目录

```bash
data/raw/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 2. 快速查看数据

```bash
# 启动Jupyter Lab
jupyter lab

# 打开第一个notebook
# notebooks/01_eda/01_data_loading_and_overview.ipynb
```

---

## 📚 第三步：开始学习

### 学习路径

按照以下顺序进行学习，每个阶段大约需要1周时间：

#### Week 1: 探索性数据分析 (EDA)
```
notebooks/01_eda/
├── 01_data_loading_and_overview.ipynb  ← 从这里开始
├── 02_univariate_analysis.ipynb
├── 03_bivariate_analysis.ipynb
└── 04_multivariate_analysis.ipynb
```

**本周目标**：
- [ ] 熟悉数据结构
- [ ] 理解每个特征的含义和分布
- [ ] 发现数据质量问题
- [ ] 初步理解特征与目标的关系

**学习重点**：
- Pandas基础操作
- 描述性统计
- 数据可视化
- 相关性分析

---

#### Week 2: 数据预处理
```
notebooks/02_preprocessing/
├── 01_missing_values.ipynb
├── 02_outliers.ipynb
├── 03_data_transformation.ipynb
└── 04_data_splitting.ipynb
```

**本周目标**：
- [ ] 处理缺失值
- [ ] 处理异常值
- [ ] 数据标准化/归一化
- [ ] 正确划分训练集和验证集

**学习重点**：
- 缺失值填充策略
- 异常值检测方法
- StandardScaler vs MinMaxScaler
- 交叉验证原理

---

#### Week 3: 特征工程
```
notebooks/03_feature_engineering/
├── 01_categorical_encoding.ipynb
├── 02_numeric_features.ipynb
├── 03_feature_interaction.ipynb
└── 04_feature_selection.ipynb
```

**本周目标**：
- [ ] 编码类别特征
- [ ] 构造新特征
- [ ] 特征选择
- [ ] 降维

**学习重点**：
- Target Encoding原理
- 特征交互的重要性
- 树模型特征重要性
- PCA降维

---

#### Week 4: 模型训练
```
notebooks/04_modeling/
├── 01_baseline_models.ipynb
├── 02_tree_models.ipynb
├── 03_boosting_models.ipynb
└── 04_hyperparameter_tuning.ipynb
```

**本周目标**：
- [ ] 建立baseline模型
- [ ] 尝试多种算法
- [ ] 理解模型原理
- [ ] 超参数调优

**学习重点**：
- LightGBM/XGBoost使用
- K-Fold交叉验证实践
- Optuna调参
- 避免过拟合

---

#### Week 5: 模型评估与优化
```
notebooks/05_evaluation/
├── 01_model_evaluation.ipynb
├── 02_model_interpretation.ipynb
├── 03_model_ensemble.ipynb
└── 04_final_submission.ipynb
```

**本周目标**：
- [ ] 深入理解评估指标
- [ ] 模型诊断
- [ ] 模型解释
- [ ] 模型融合
- [ ] 生成提交文件

**学习重点**：
- SHAP值解释
- 学习曲线分析
- Stacking/Blending
- 竞赛提交

---

## 💡 学习建议

### 1. 每日工作流程

```
08:00-09:00  复习前一天内容
09:00-11:00  学习新知识点 + 编写代码
11:00-12:00  阅读相关资料/博客
14:00-16:00  完成练习和实践
16:00-17:00  记录笔记和总结
17:00-18:00  参考Kaggle优秀notebook
```

### 2. 如何使用Notebook

每个notebook都包含：
- **🎯 学习目标**：明确本节要学什么
- **📚 知识点**：理论知识讲解
- **💡 学习要点**：重点标注
- **代码框架**：可直接运行的代码
- **📝 总结**：反思和记录

**使用步骤**：
1. 先阅读学习目标和知识点
2. 运行代码，观察结果
3. 修改参数，实验不同方法
4. 在总结部分记录你的理解
5. 尝试在自己的数据上应用

### 3. 代码规范

```python
# ✅ 好的实践
# 1. 添加注释说明代码目的
# 2. 函数命名清晰
# 3. 关键步骤输出中间结果

def calculate_feature_importance(model, X, y):
    """
    计算特征重要性

    Parameters
    ----------
    model : 模型对象
    X : 特征矩阵
    y : 目标变量

    Returns
    -------
    pd.DataFrame : 特征重要性表
    """
    # 实现代码
    pass
```

### 4. 版本控制

```bash
# 初始化git仓库
git init
git add .
git commit -m "Initial commit: project structure"

# 每天提交
git add .
git commit -m "feat: 完成单变量分析"
git push
```

### 5. 记录实验

创建 `experiments.md` 记录你的实验：

```markdown
## 实验 001
**日期**: 2024-01-15
**目标**: 尝试不同的缺失值填充方法
**方法**:
- 均值填充
- 中位数填充
- KNN填充
**结果**: KNN填充效果最好，验证集AUC提升0.02
**结论**: 对于有规律的缺失，KNN填充更合适
```

---

## 🆘 遇到问题怎么办？

### 1. Debug流程

```python
# 1. 打印中间结果
print(f"数据形状: {df.shape}")
print(f"缺失值: {df.isnull().sum().sum()}")

# 2. 查看数据样本
display(df.head())

# 3. 检查数据类型
print(df.dtypes)

# 4. 使用断言验证假设
assert df.shape[0] > 0, "数据为空"
assert 'target' in df.columns, "目标列不存在"
```

### 2. 常见错误

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| KeyError | 列名不存在 | 检查列名拼写 |
| ValueError | 数据类型不匹配 | 转换数据类型 |
| MemoryError | 内存不足 | 使用内存优化函数 |
| SettingWithCopyWarning | Pandas切片问题 | 使用 `.copy()` |

### 3. 学习资源

- **官方文档**
  - Pandas: https://pandas.pydata.org/docs/
  - Scikit-learn: https://scikit-learn.org/
  - LightGBM: https://lightgbm.readthedocs.io/

- **优秀博客**
  - Kaggle Learn: https://www.kaggle.com/learn
  - Towards Data Science
  - Machine Learning Mastery

- **视频课程**
  - Coursera: Applied Data Science with Python
  - Fast.ai

---

## 📊 进度追踪

使用 `docs/knowledge_checklist.md` 追踪你的学习进度：

```markdown
### 1.1 数据加载与理解
- [x] pd.read_csv() 各种参数的使用
- [x] DataFrame基本属性
- [ ] 数据类型转换
- [ ] 内存优化技巧
```

---

## 🎯 最终目标

完成这个项目后，你应该能够：

1. ✅ **独立完成**一个完整的数据分析竞赛
2. ✅ **理解**数据分析的完整流程
3. ✅ **掌握**主流的机器学习算法
4. ✅ **建立**自己的代码库和方法论
5. ✅ **具备**解决实际业务问题的能力

---

## 下一步

👉 打开 `notebooks/01_eda/01_data_loading_and_overview.ipynb` 开始你的学习之旅！

有任何问题，记得：
1. 先查看文档
2. 搜索相关资料
3. 检查代码逻辑
4. 记录问题和解决方案

**祝你学习愉快！** 🎉
