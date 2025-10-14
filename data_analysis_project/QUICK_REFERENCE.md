# 数据分析快速参考手册

> 常用代码片段和命令速查表

---

## 🔧 环境管理

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 启动Jupyter
jupyter lab

# 查看已安装的包
pip list
```

---

## 📊 Pandas常用操作

### 数据加载
```python
import pandas as pd

# CSV
df = pd.read_csv('data.csv')

# 指定编码
df = pd.read_csv('data.csv', encoding='utf-8')

# 解析日期
df = pd.read_csv('data.csv', parse_dates=['date_col'])

# 只读取部分列
df = pd.read_csv('data.csv', usecols=['col1', 'col2'])
```

### 数据查看
```python
# 基本信息
df.shape              # (行数, 列数)
df.info()             # 数据类型和缺失值
df.describe()         # 统计摘要
df.head(10)           # 前10行
df.tail(10)           # 后10行
df.columns            # 列名
df.dtypes             # 数据类型

# 查看唯一值
df['col'].nunique()
df['col'].value_counts()
```

### 缺失值处理
```python
# 检查缺失
df.isnull().sum()
df.isnull().sum() / len(df) * 100  # 缺失率

# 删除缺失
df.dropna()           # 删除包含缺失的行
df.dropna(axis=1)     # 删除包含缺失的列
df.dropna(thresh=5)   # 至少有5个非空值才保留

# 填充缺失
df.fillna(0)                        # 填充为0
df.fillna(df.mean())                # 填充为均值
df.fillna(df.median())              # 填充为中位数
df.fillna(method='ffill')           # 前向填充
df.fillna(method='bfill')           # 后向填充
df['col'].fillna(df['col'].mode()[0])  # 填充为众数
```

### 数据筛选
```python
# 条件筛选
df[df['age'] > 30]
df[(df['age'] > 30) & (df['gender'] == 'M')]
df[df['name'].isin(['Alice', 'Bob'])]

# 选择列
df['col']                    # 单列
df[['col1', 'col2']]        # 多列

# 位置选择
df.iloc[0]                   # 第一行
df.iloc[0:5]                 # 前5行
df.iloc[:, 0:3]              # 前3列

# 标签选择
df.loc[0]                    # 索引为0的行
df.loc[:, 'col1':'col3']     # 列名范围
```

### 数据转换
```python
# 类型转换
df['col'] = df['col'].astype(int)
df['col'] = df['col'].astype('category')
df['date'] = pd.to_datetime(df['date'])

# 应用函数
df['col'].apply(lambda x: x * 2)
df.apply(lambda row: row['A'] + row['B'], axis=1)

# 映射
df['col'].map({'A': 1, 'B': 2})

# 替换
df['col'].replace({'old': 'new'})
```

### 分组聚合
```python
# 基础分组
df.groupby('category')['value'].mean()
df.groupby('category')['value'].agg(['mean', 'sum', 'count'])

# 多列分组
df.groupby(['cat1', 'cat2'])['value'].mean()

# 自定义聚合
df.groupby('category').agg({
    'col1': 'mean',
    'col2': 'sum',
    'col3': ['min', 'max']
})
```

### 合并数据
```python
# 拼接
pd.concat([df1, df2], axis=0)      # 纵向拼接
pd.concat([df1, df2], axis=1)      # 横向拼接

# 合并
pd.merge(df1, df2, on='key')                    # 内连接
pd.merge(df1, df2, on='key', how='left')        # 左连接
pd.merge(df1, df2, on='key', how='outer')       # 外连接
pd.merge(df1, df2, left_on='a', right_on='b')   # 不同列名
```

---

## 📈 数据可视化

### Matplotlib
```python
import matplotlib.pyplot as plt

# 折线图
plt.plot(x, y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()

# 散点图
plt.scatter(x, y)

# 柱状图
plt.bar(categories, values)

# 直方图
plt.hist(data, bins=30)

# 子图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(x, y)
```

### Seaborn
```python
import seaborn as sns

# 分布图
sns.histplot(data=df, x='col')
sns.kdeplot(data=df, x='col')

# 箱线图
sns.boxplot(data=df, x='category', y='value')

# 小提琴图
sns.violinplot(data=df, x='category', y='value')

# 散点图
sns.scatterplot(data=df, x='col1', y='col2', hue='category')

# 相关性热力图
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# 成对关系图
sns.pairplot(df, hue='target')
```

---

## 🧹 数据预处理

### 异常值处理
```python
from scipy import stats

# IQR方法
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_clean = df[(df['col'] >= lower) & (df['col'] <= upper)]

# Z-score方法
z_scores = np.abs(stats.zscore(df['col']))
df_clean = df[z_scores < 3]

# 截断（Winsorization）
df['col'] = df['col'].clip(lower, upper)
```

### 数据标准化
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 标准化（Z-score）
scaler = StandardScaler()
df['col_scaled'] = scaler.fit_transform(df[['col']])

# 归一化（0-1）
scaler = MinMaxScaler()
df['col_normalized'] = scaler.fit_transform(df[['col']])

# 鲁棒标准化
scaler = RobustScaler()
df['col_robust'] = scaler.fit_transform(df[['col']])
```

### 类别编码
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

# Label Encoding
le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'])

# One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['col'])

# Target Encoding
te = TargetEncoder()
df['col_target'] = te.fit_transform(df['col'], df['target'])

# Frequency Encoding
freq = df['col'].value_counts().to_dict()
df['col_freq'] = df['col'].map(freq)
```

---

## 🤖 机器学习

### 数据分割
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 交叉验证
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Stratified K-Fold（分层）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

### LightGBM
```python
import lightgbm as lgb

# 参数设置
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# 训练
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 预测
y_pred = model.predict(X_test)

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)
```

### XGBoost
```python
import xgboost as xgb

# 参数设置
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# 训练
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# 预测
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)
```

### 超参数优化（Optuna）
```python
import optuna

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15)
    }

    model = lgb.LGBMClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best params:', study.best_params)
print('Best score:', study.best_value)
```

---

## 📊 模型评估

### 分类指标
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# 基础指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# AUC
auc = roc_auc_score(y_true, y_pred_proba)

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 分类报告
report = classification_report(y_true, y_pred)
```

### 回归指标
```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

# MAE
mae = mean_absolute_error(y_true, y_pred)

# MSE
mse = mean_squared_error(y_true, y_pred)

# RMSE
rmse = np.sqrt(mse)

# R²
r2 = r2_score(y_true, y_pred)
```

---

## 🔍 模型解释

### SHAP
```python
import shap

# 创建explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Force plot（单样本）
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Dependence plot
shap.dependence_plot('feature_name', shap_values, X_test)
```

---

## 💾 模型保存与加载

```python
import joblib
import pickle

# Joblib（推荐）
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')

# Pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# LightGBM原生格式
model.save_model('model.txt')
model = lgb.Booster(model_file='model.txt')
```

---

## 🎯 常用技巧

### 内存优化
```python
# 降低数值类型精度
df['int_col'] = df['int_col'].astype('int8')
df['float_col'] = df['float_col'].astype('float32')

# 类别类型
df['cat_col'] = df['cat_col'].astype('category')

# 查看内存使用
df.memory_usage(deep=True)
```

### 时间特征提取
```python
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
```

### 进度条
```python
from tqdm import tqdm

# 循环
for i in tqdm(range(100)):
    # do something
    pass

# Pandas apply
tqdm.pandas()
df['new_col'] = df['col'].progress_apply(lambda x: x * 2)
```

---

## 🔨 调试技巧

```python
# 断点调试
import pdb; pdb.set_trace()

# 打印变量类型和值
print(f"Type: {type(var)}, Value: {var}")

# 查看DataFrame信息
print(df.info())
print(df.describe())
print(df.head())

# 检查空值
assert df.isnull().sum().sum() == 0, "存在空值"

# 检查形状
print(f"Shape: {df.shape}")
```

---

## 📝 Jupyter技巧

```python
# 显示所有列
pd.set_option('display.max_columns', None)

# 显示更多行
pd.set_option('display.max_rows', 100)

# 显示小数位数
pd.set_option('display.float_format', '{:.3f}'.format)

# 查看变量
%whos

# 计时
%%time
# code here

# 内存分析
%load_ext memory_profiler
%memit df.groupby('col').mean()

# 自动重载模块
%load_ext autoreload
%autoreload 2
```

---

## 🎯 快速测试流程

```python
# 1. 加载数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. 简单特征工程
X = train.drop('target', axis=1)
y = train['target']

# 3. 快速训练
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. 预测
pred = model.predict(test)

# 5. 提交
submission = pd.DataFrame({'id': test_id, 'target': pred})
submission.to_csv('submission.csv', index=False)
```

---

保存这个文件，在需要时快速查阅！🚀
