# 数据分析与建模学习项目

> 一个系统的、学习导向的数据分析竞赛项目框架

## 📚 项目目标

1. **完成竞赛任务**：获得良好的模型表现
2. **系统学习**：掌握数据分析完整流程
3. **知识沉淀**：形成可复用的代码库和方法论

---

## 🗂️ 项目结构

```
data_analysis_project/
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据（不可修改）
│   ├── processed/                 # 处理后的数据
│   └── external/                  # 外部数据源
│
├── notebooks/                     # Jupyter笔记本（按流程组织）
│   ├── 01_eda/                    # 探索性数据分析
│   ├── 02_preprocessing/          # 数据预处理
│   ├── 03_feature_engineering/    # 特征工程
│   ├── 04_modeling/               # 模型训练
│   └── 05_evaluation/             # 模型评估与优化
│
├── src/                           # 源代码（模块化）
│   ├── data/                      # 数据处理模块
│   ├── features/                  # 特征工程模块
│   ├── models/                    # 模型相关模块
│   └── visualization/             # 可视化模块
│
├── models/                        # 保存的模型文件
│
├── reports/                       # 报告和可视化结果
│   ├── figures/                   # 图表
│   └── final/                     # 最终报告
│
├── config/                        # 配置文件
│
├── tests/                         # 单元测试
│
├── docs/                          # 文档和学习笔记
│
├── requirements.txt               # 依赖包
├── environment.yml                # Conda环境配置
└── README.md                      # 项目说明
```

---

## 🎯 学习路线图

### Phase 1: 数据探索与理解 (Week 1)
**学习目标**：掌握EDA方法，理解数据特征

- [ ] **1.1 数据加载与初步观察**
  - 知识点：Pandas基础、数据结构
  - 实践：读取数据、查看形状、类型、缺失情况

- [ ] **1.2 单变量分析**
  - 知识点：描述性统计、分布分析
  - 实践：数值型（直方图、箱线图）、类别型（频数统计）

- [ ] **1.3 双变量分析**
  - 知识点：相关性分析、卡方检验
  - 实践：特征与目标的关系分析

- [ ] **1.4 多变量分析**
  - 知识点：相关矩阵、PairPlot
  - 实践：特征间交互关系

**产出**：EDA报告、数据质量报告

---

### Phase 2: 数据预处理 (Week 2)
**学习目标**：掌握数据清洗和转换技术

- [ ] **2.1 缺失值处理**
  - 知识点：缺失机制（MCAR/MAR/MNAR）
  - 方法：删除、均值/中位数填充、KNN填充、预测填充

- [ ] **2.2 异常值处理**
  - 知识点：IQR方法、Z-score、Isolation Forest
  - 实践：检测与处理策略选择

- [ ] **2.3 数据类型转换**
  - 知识点：类型转换、日期处理
  - 实践：优化数据类型降低内存

- [ ] **2.4 数据标准化/归一化**
  - 知识点：StandardScaler、MinMaxScaler、RobustScaler
  - 实践：选择合适的标准化方法

**产出**：清洗后的数据集、处理流程文档

---

### Phase 3: 特征工程 (Week 3)
**学习目标**：掌握特征构造和选择方法

- [ ] **3.1 类别特征编码**
  - 知识点：LabelEncoding、OneHotEncoding、TargetEncoding
  - 实践：处理高基数类别特征

- [ ] **3.2 数值特征构造**
  - 知识点：多项式特征、分箱、对数变换
  - 实践：构造业务相关特征

- [ ] **3.3 特征交互**
  - 知识点：特征组合、特征比例
  - 实践：自动特征交互生成

- [ ] **3.4 特征选择**
  - 知识点：过滤法（相关性、卡方）、包装法（RFE）、嵌入法（树模型重要性）
  - 实践：降低特征维度，避免过拟合

**产出**：特征工程pipeline、特征重要性报告

---

### Phase 4: 模型训练与选择 (Week 4)
**学习目标**：掌握常用机器学习算法和调参方法

- [ ] **4.1 建立Baseline**
  - 知识点：简单模型（逻辑回归、决策树）
  - 实践：快速建立性能基准

- [ ] **4.2 集成学习模型**
  - 知识点：
    - Bagging（RandomForest）
    - Boosting（XGBoost、LightGBM、CatBoost）
    - Stacking
  - 实践：对比不同模型表现

- [ ] **4.3 交叉验证策略**
  - 知识点：K-Fold、StratifiedKFold、GroupKFold、TimeSeriesSplit
  - 实践：选择合适的验证策略

- [ ] **4.4 超参数优化**
  - 知识点：GridSearch、RandomSearch、Bayesian Optimization、Optuna
  - 实践：自动化调参

**产出**：训练好的模型、模型对比报告

---

### Phase 5: 模型评估与优化 (Week 5)
**学习目标**：掌握模型评估和调优技术

- [ ] **5.1 评估指标深入理解**
  - 分类：Accuracy、Precision、Recall、F1、AUC-ROC
  - 回归：MAE、MSE、RMSE、R²、MAPE
  - 实践：根据业务选择合适指标

- [ ] **5.2 模型诊断**
  - 知识点：学习曲线、验证曲线、残差分析
  - 实践：识别过拟合/欠拟合

- [ ] **5.3 模型解释**
  - 知识点：特征重要性、SHAP、LIME、PDP
  - 实践：理解模型决策逻辑

- [ ] **5.4 模型融合**
  - 知识点：加权平均、Stacking、Blending
  - 实践：提升最终表现

**产出**：优化后的模型、模型解释报告、竞赛提交文件

---

## 🛠️ 技术栈

### 核心库
```python
# 数据处理
pandas >= 1.5.0
numpy >= 1.23.0

# 可视化
matplotlib >= 3.6.0
seaborn >= 0.12.0
plotly >= 5.11.0

# 机器学习
scikit-learn >= 1.2.0
xgboost >= 1.7.0
lightgbm >= 3.3.0
catboost >= 1.1.0

# 模型解释
shap >= 0.41.0
eli5 >= 0.13.0

# 超参数优化
optuna >= 3.0.0

# 其他
jupyter >= 1.0.0
tqdm >= 4.64.0
```

---

## 📝 每日工作流程

### 1. 开始工作
```bash
# 激活环境
conda activate data_analysis

# 启动Jupyter
jupyter lab

# 打开对应阶段的notebook
```

### 2. 编码规范
- 每个notebook对应一个具体任务
- 关键函数写入src/目录，便于复用
- 每段代码添加注释说明学习要点
- 记录实验结果和思考

### 3. 版本控制
```bash
git add .
git commit -m "feat: 完成特征工程的类别编码部分"
git push
```

---

## 📊 学习成果检验

### 每个阶段结束后自查：

✅ **理解**：能否用自己的话解释核心概念？
✅ **实践**：能否独立复现代码？
✅ **应用**：能否在新数据集上应用？
✅ **优化**：能否提出改进方案？

---

## 🎓 学习资源推荐

### 书籍
- 《Python数据分析实战》
- 《机器学习实战》
- 《特征工程入门与实践》

### 在线课程
- Kaggle Learn（免费，实践性强）
- Coursera: Applied Data Science with Python

### 社区
- Kaggle Discussions & Notebooks
- GitHub优秀项目学习
- 数据科学社区交流

---

## 💡 学习建议

1. **边学边做**：不要只看理论，一定要动手实践
2. **记录思考**：在notebook中记录为什么这样做
3. **对比实验**：同一问题尝试多种方法，对比效果
4. **代码复用**：将通用函数模块化，形成自己的工具库
5. **定期复盘**：每周总结学到的知识点和踩过的坑
6. **参考学习**：看Kaggle优秀notebook，学习他人思路

---

## 🚀 快速开始

```bash
# 1. 创建环境
conda create -n data_analysis python=3.10
conda activate data_analysis

# 2. 安装依赖
pip install -r requirements.txt

# 3. 放置数据
# 将竞赛数据放入 data/raw/ 目录

# 4. 开始第一阶段
jupyter lab notebooks/01_eda/
```

---

## 📈 进度追踪

| 阶段 | 开始日期 | 完成日期 | 状态 | 关键收获 |
|------|---------|---------|------|---------|
| Phase 1: EDA | | | ⬜ | |
| Phase 2: 预处理 | | | ⬜ | |
| Phase 3: 特征工程 | | | ⬜ | |
| Phase 4: 模型训练 | | | ⬜ | |
| Phase 5: 模型优化 | | | ⬜ | |

---

祝学习顺利！🎉
