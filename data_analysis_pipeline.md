# 数据分析与建模完整管线

## 📊 数据集概览
- **问题类型**: 回归问题（预测医疗保险费用 charges）
- **数据规模**: 60,000 行 × 8 列
- **特征类型**:
  - 数值型: age, bmi, children
  - 类别型: sex, smoker, region
  - 目标变量: charges（保险费用）

---

## 🎯 完整数据分析管线

### 阶段 1: 探索性数据分析（EDA）✅

#### 1.1 数据加载与初步观察 ✅ 已完成
- [x] 读取数据
- [x] 查看数据形状、类型
- [x] 缺失值检查
- [x] 内存优化

文件: `01_data_loading_and_overview.ipynb`

#### 1.2 单变量分析（下一步）
**目标**: 深入理解每个特征的分布特征

**数值型特征分析**:
- [ ] 分布图（直方图 + KDE）
- [ ] 箱线图（识别异常值）
- [ ] 统计量分析（偏度、峰度）
- [ ] 正态性检验

**类别型特征分析**:
- [ ] 频数统计
- [ ] 条形图可视化
- [ ] 类别不平衡检查

**目标变量分析**:
- [ ] charges 的分布形态
- [ ] 是否需要对数转换
- [ ] 异常值检测

文件: `02_univariate_analysis.ipynb`

#### 1.3 双变量分析
**目标**: 探索特征与目标变量的关系

- [ ] 数值型特征 vs 目标变量
  - 散点图
  - 相关系数矩阵
  - 回归线拟合

- [ ] 类别型特征 vs 目标变量
  - 箱线图对比
  - 小提琴图
  - 统计检验（t检验/方差分析）

- [ ] 特征交互探索
  - smoker × age
  - bmi × age
  - region × smoker

文件: `03_bivariate_analysis.ipynb`

#### 1.4 多变量分析
**目标**: 理解特征之间的相互关系

- [ ] 相关性热力图
- [ ] 配对图（pairplot）
- [ ] 特征聚类
- [ ] PCA降维可视化

文件: `04_multivariate_analysis.ipynb`

---

### 阶段 2: 数据预处理

#### 2.1 缺失值处理
- [ ] 数值型: 均值/中位数填充
- [ ] 类别型: 众数填充或新类别
- [ ] 高缺失率特征: 考虑删除

#### 2.2 异常值处理
- [ ] 识别异常值（IQR法/Z-score法）
- [ ] 决策: 删除/上下限截断/保留

#### 2.3 特征编码
- [ ] 类别型特征编码
  - 二分类: Label Encoding（sex, smoker）
  - 多分类: One-Hot Encoding（region）

#### 2.4 特征缩放
- [ ] 标准化（StandardScaler）: 适用于线性模型
- [ ] 归一化（MinMaxScaler）: 适用于神经网络

#### 2.5 目标变量转换
- [ ] 对数转换: log1p(charges)
- [ ] Box-Cox转换
- [ ] 评估转换效果

文件: `05_data_preprocessing.ipynb`

---

### 阶段 3: 特征工程

#### 3.1 特征构造
- [ ] BMI分类: 正常/超重/肥胖
- [ ] 年龄分组: 青年/中年/老年
- [ ] 交互特征:
  - smoker × age（吸烟者年龄越大风险越高）
  - smoker × bmi（吸烟且肥胖风险更高）
  - age × bmi

#### 3.2 多项式特征
- [ ] age², bmi²
- [ ] age × bmi

#### 3.3 特征选择
- [ ] 相关性筛选
- [ ] 特征重要性（树模型）
- [ ] 递归特征消除（RFE）
- [ ] 正则化特征选择（Lasso）

文件: `06_feature_engineering.ipynb`

---

### 阶段 4: 模型构建与评估

#### 4.1 数据划分
- [ ] 训练集/验证集/测试集（70%/15%/15%）
- [ ] 交叉验证策略（K-Fold）

#### 4.2 基准模型
**目标**: 建立性能基线

- [ ] 简单均值模型
- [ ] 线性回归
- [ ] 评估指标:
  - MAE（平均绝对误差）
  - RMSE（均方根误差）
  - R²（决定系数）
  - MAPE（平均绝对百分比误差）

#### 4.3 进阶模型
- [ ] 岭回归（Ridge）
- [ ] Lasso回归
- [ ] ElasticNet
- [ ] 决策树
- [ ] 随机森林
- [ ] 梯度提升树（GBDT）
- [ ] XGBoost
- [ ] LightGBM
- [ ] CatBoost

#### 4.4 模型调优
- [ ] 网格搜索（GridSearchCV）
- [ ] 随机搜索（RandomizedSearchCV）
- [ ] 贝叶斯优化（Optuna）

#### 4.5 模型集成
- [ ] Voting回归
- [ ] Stacking
- [ ] Blending

文件: `07_baseline_models.ipynb`, `08_advanced_models.ipynb`, `09_model_tuning.ipynb`

---

### 阶段 5: 模型解释与验证

#### 5.1 模型解释
- [ ] 特征重要性分析
- [ ] SHAP值分析
- [ ] 部分依赖图（PDP）
- [ ] LIME局部解释

#### 5.2 误差分析
- [ ] 残差分析
- [ ] 预测值 vs 真实值散点图
- [ ] 误差分布
- [ ] 识别预测差的样本

#### 5.3 模型诊断
- [ ] 过拟合/欠拟合检查
- [ ] 学习曲线
- [ ] 验证曲线

文件: `10_model_interpretation.ipynb`

---

### 阶段 6: 模型部署准备

#### 6.1 最终模型选择
- [ ] 综合性能对比
- [ ] 模型复杂度 vs 性能权衡
- [ ] 推理速度测试

#### 6.2 模型保存
- [ ] 保存最佳模型（joblib/pickle）
- [ ] 保存预处理管道
- [ ] 版本管理

#### 6.3 预测测试集
- [ ] 加载测试集
- [ ] 应用预处理
- [ ] 生成预测结果
- [ ] 创建提交文件

文件: `11_final_prediction.ipynb`

---

## 📁 推荐的项目结构

```
data_analysis_project/
├── data/
│   ├── raw/                # 原始数据
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/          # 处理后的数据
│   │   ├── train_clean.csv
│   │   ├── train_engineered.csv
│   │   └── test_processed.csv
│   └── submissions/        # 提交文件
│       └── submission.csv
│
├── notebooks/
│   ├── 01_eda/            # 探索性分析
│   │   ├── 01_data_loading_and_overview.ipynb ✅
│   │   ├── 02_univariate_analysis.ipynb
│   │   ├── 03_bivariate_analysis.ipynb
│   │   └── 04_multivariate_analysis.ipynb
│   │
│   ├── 02_preprocessing/   # 数据预处理
│   │   └── 05_data_preprocessing.ipynb
│   │
│   ├── 03_feature_engineering/  # 特征工程
│   │   └── 06_feature_engineering.ipynb
│   │
│   └── 04_modeling/        # 建模
│       ├── 07_baseline_models.ipynb
│       ├── 08_advanced_models.ipynb
│       ├── 09_model_tuning.ipynb
│       ├── 10_model_interpretation.ipynb
│       └── 11_final_prediction.ipynb
│
├── src/                    # 可复用的代码
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model_utils.py
│
├── models/                 # 保存的模型
│   └── best_model.pkl
│
└── reports/                # 分析报告
    └── final_report.md
```

---

## 🎓 学习建议

### 1. 循序渐进
- 不要跳步，按顺序完成每个阶段
- 每个notebook都要深入理解，而不是只运行代码

### 2. 多问为什么
- 为什么要做这个处理？
- 这个方法适用于什么场景？
- 有没有其他替代方案？

### 3. 动手实践
- 尝试不同的方法
- 对比不同方法的效果
- 记录你的发现和思考

### 4. 查阅文档
- Pandas官方文档
- Scikit-learn用户指南
- Kaggle优秀Kernel

### 5. 关键知识点

#### EDA阶段重点
- 数据分布理解
- 异常值识别
- 特征与目标关系
- 特征之间的相关性

#### 建模阶段重点
- 数据泄露防范
- 交叉验证重要性
- 过拟合与欠拟合
- 模型评估指标选择
- 超参数调优策略

---

## 📊 本项目的关键洞察（待发现）

根据数据特点，你可能会发现：

1. **smoker是最重要的特征**: 吸烟者费用显著更高
2. **age和bmi的非线性关系**: 可能需要多项式特征
3. **交互作用**: smoker × bmi, smoker × age
4. **目标变量偏态**: charges需要对数转换
5. **区域差异**: region可能影响不大

---

## ✅ 下一步行动

1. **立即开始**: 创建 `02_univariate_analysis.ipynb`
2. **分析每个特征**: 理解数据分布
3. **识别问题**: 找出需要处理的数据质量问题
4. **建立假设**: 哪些特征可能重要？为什么？

---

## 📚 推荐学习资源

- **书籍**: 《Python数据分析》(McKinney)
- **课程**: Kaggle Learn - Intermediate ML
- **实战**: Kaggle竞赛和Kernel
- **文档**: Scikit-learn User Guide

---

**记住**: 数据分析是一个迭代的过程，不要追求完美，先跑通整个流程，再回头优化！

Good luck! 🚀
