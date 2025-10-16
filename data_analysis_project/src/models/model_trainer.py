"""
模型训练模块
提供完整的模型训练、验证和优化功能
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, task_type='classification', random_state=42):
        """
        初始化模型训练器

        Parameters:
        -----------
        task_type : str
            任务类型：'classification' 或 'regression'
        random_state : int
            随机种子
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.cv_results = {}

    def add_model(self, name, model, params=None):
        """
        添加模型

        Parameters:
        -----------
        name : str
            模型名称
        model : object
            模型对象
        params : dict
            模型参数
        """
        if params:
            model.set_params(**params)
        self.models[name] = model
        print(f"✅ 模型已添加: {name}")

    def get_default_models(self):
        """
        获取默认模型集合

        Returns:
        --------
        dict
            模型字典
        """
        if self.task_type == 'classification':
            return {
                'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'RandomForest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
                'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
            }
        else:
            return {
                'Ridge': Ridge(random_state=self.random_state),
                'Lasso': Lasso(random_state=self.random_state),
                'RandomForest': RandomForestRegressor(random_state=self.random_state, n_estimators=100),
                'XGBoost': xgb.XGBRegressor(random_state=self.random_state),
                'LightGBM': lgb.LGBMRegressor(random_state=self.random_state, verbose=-1),
            }

    def train_model(self, name, X_train, y_train, X_val=None, y_val=None):
        """
        训练单个模型

        Parameters:
        -----------
        name : str
            模型名称
        X_train : pd.DataFrame or np.array
            训练特征
        y_train : pd.Series or np.array
            训练目标
        X_val : pd.DataFrame or np.array, optional
            验证特征
        y_val : pd.Series or np.array, optional
            验证目标

        Returns:
        --------
        object
            训练好的模型
        """
        if name not in self.models:
            raise ValueError(f"模型 '{name}' 未找到，请先使用 add_model() 添加")

        model = self.models[name]
        print(f"\n训练模型: {name}")
        print(f"训练集大小: {X_train.shape}")

        # 训练模型
        if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)) and X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        elif isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)) and X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50, verbose=False)
        else:
            model.fit(X_train, y_train)

        self.trained_models[name] = model

        # 评估模型
        train_pred = model.predict(X_train)
        if self.task_type == 'classification':
            train_score = accuracy_score(y_train, train_pred)
            metric_name = 'Accuracy'
        else:
            train_score = r2_score(y_train, train_pred)
            metric_name = 'R²'

        print(f"训练集 {metric_name}: {train_score:.4f}")

        if X_val is not None:
            val_pred = model.predict(X_val)
            if self.task_type == 'classification':
                val_score = accuracy_score(y_val, val_pred)
            else:
                val_score = r2_score(y_val, val_pred)
            print(f"验证集 {metric_name}: {val_score:.4f}")

        print(f"✅ 模型训练完成: {name}")
        return model

    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练所有已添加的模型

        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            训练特征
        y_train : pd.Series or np.array
            训练目标
        X_val : pd.DataFrame or np.array, optional
            验证特征
        y_val : pd.Series or np.array, optional
            验证目标
        """
        if not self.models:
            print("添加默认模型...")
            self.models = self.get_default_models()

        print("="*80)
        print(f"开始训练 {len(self.models)} 个模型")
        print("="*80)

        for name in self.models:
            try:
                self.train_model(name, X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"❌ 模型 {name} 训练失败: {e}")

        print("\n" + "="*80)
        print("所有模型训练完成!")
        print("="*80)

    def cross_validate(self, name, X, y, cv=5, scoring=None):
        """
        交叉验证模型

        Parameters:
        -----------
        name : str
            模型名称
        X : pd.DataFrame or np.array
            特征
        y : pd.Series or np.array
            目标
        cv : int or cross-validator
            交叉验证策略
        scoring : str
            评分指标

        Returns:
        --------
        dict
            交叉验证结果
        """
        if name not in self.models:
            raise ValueError(f"模型 '{name}' 未找到")

        model = self.models[name]

        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'

        if isinstance(cv, int):
            if self.task_type == 'classification':
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = cv

        print(f"\n交叉验证: {name}")
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)

        result = {
            'model_name': name,
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scoring': scoring
        }

        self.cv_results[name] = result

        print(f"{scoring}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        print(f"所有折得分: {scores}")

        return result

    def cross_validate_all(self, X, y, cv=5, scoring=None):
        """
        对所有模型进行交叉验证

        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征
        y : pd.Series or np.array
            目标
        cv : int
            交叉验证折数
        scoring : str
            评分指标

        Returns:
        --------
        pd.DataFrame
            交叉验证结果汇总
        """
        if not self.models:
            self.models = self.get_default_models()

        print("="*80)
        print(f"开始对 {len(self.models)} 个模型进行 {cv} 折交叉验证")
        print("="*80)

        results = []
        for name in self.models:
            try:
                result = self.cross_validate(name, X, y, cv, scoring)
                results.append({
                    'Model': name,
                    'Mean Score': result['mean_score'],
                    'Std Score': result['std_score'],
                    'Scoring': result['scoring']
                })
            except Exception as e:
                print(f"❌ 模型 {name} 交叉验证失败: {e}")

        results_df = pd.DataFrame(results).sort_values('Mean Score', ascending=False)

        print("\n" + "="*80)
        print("交叉验证结果汇总:")
        print("="*80)
        print(results_df.to_string(index=False))

        return results_df

    def predict(self, name, X):
        """
        使用训练好的模型进行预测

        Parameters:
        -----------
        name : str
            模型名称
        X : pd.DataFrame or np.array
            特征

        Returns:
        --------
        np.array
            预测结果
        """
        if name not in self.trained_models:
            raise ValueError(f"模型 '{name}' 未训练，请先训练模型")

        return self.trained_models[name].predict(X)

    def predict_proba(self, name, X):
        """
        预测概率（仅分类任务）

        Parameters:
        -----------
        name : str
            模型名称
        X : pd.DataFrame or np.array
            特征

        Returns:
        --------
        np.array
            预测概率
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba 仅支持分类任务")

        if name not in self.trained_models:
            raise ValueError(f"模型 '{name}' 未训练")

        model = self.trained_models[name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError(f"模型 '{name}' 不支持 predict_proba")

    def save_model(self, name, filepath):
        """
        保存模型

        Parameters:
        -----------
        name : str
            模型名称
        filepath : str
            保存路径
        """
        if name not in self.trained_models:
            raise ValueError(f"模型 '{name}' 未训练")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_models[name], filepath)
        print(f"✅ 模型已保存: {filepath}")

    def load_model(self, name, filepath):
        """
        加载模型

        Parameters:
        -----------
        name : str
            模型名称
        filepath : str
            模型文件路径
        """
        model = joblib.load(filepath)
        self.trained_models[name] = model
        print(f"✅ 模型已加载: {filepath}")
        return model

    def get_feature_importance(self, name, feature_names=None, top_n=20):
        """
        获取特征重要性

        Parameters:
        -----------
        name : str
            模型名称
        feature_names : list
            特征名列表
        top_n : int
            返回前N个重要特征

        Returns:
        --------
        pd.DataFrame
            特征重要性
        """
        if name not in self.trained_models:
            raise ValueError(f"模型 '{name}' 未训练")

        model = self.trained_models[name]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError(f"模型 '{name}' 不支持特征重要性")

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df


if __name__ == "__main__":
    # 示例用法
    from sklearn.datasets import load_breast_cancer, load_diabetes
    from sklearn.model_selection import train_test_split

    # 分类任务示例
    print("="*80)
    print("分类任务示例")
    print("="*80)

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trainer = ModelTrainer(task_type='classification')
    trainer.cross_validate_all(X_train, y_train, cv=5)

    # 回归任务示例
    print("\n" + "="*80)
    print("回归任务示例")
    print("="*80)

    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trainer = ModelTrainer(task_type='regression')
    trainer.cross_validate_all(X_train, y_train, cv=5)
