"""
特征工程模块
提供完整的特征工程功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程类"""

    def __init__(self, config=None):
        """
        初始化特征工程器

        Parameters:
        -----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}

    def handle_missing_values(self, df, strategy='auto'):
        """
        处理缺失值

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        strategy : str or dict
            填充策略：'auto', 'mean', 'median', 'mode', 'constant', 或字典映射

        Returns:
        --------
        pd.DataFrame
            处理后的数据框
        """
        df_filled = df.copy()

        if strategy == 'auto':
            # 自动策略：数值型用中位数，类别型用众数
            for col in df_filled.columns:
                if df_filled[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_filled[col]):
                        df_filled[col].fillna(df_filled[col].median(), inplace=True)
                    else:
                        df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

        elif isinstance(strategy, dict):
            # 按列指定策略
            for col, strat in strategy.items():
                if col in df_filled.columns and df_filled[col].isnull().any():
                    if strat == 'mean':
                        df_filled[col].fillna(df_filled[col].mean(), inplace=True)
                    elif strat == 'median':
                        df_filled[col].fillna(df_filled[col].median(), inplace=True)
                    elif strat == 'mode':
                        df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
                    else:
                        df_filled[col].fillna(strat, inplace=True)

        else:
            # 统一策略
            for col in df_filled.columns:
                if df_filled[col].isnull().any():
                    if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_filled[col]):
                        df_filled[col].fillna(df_filled[col].mean(), inplace=True)
                    elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_filled[col]):
                        df_filled[col].fillna(df_filled[col].median(), inplace=True)
                    elif strategy == 'mode':
                        df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

        print(f"✅ 缺失值处理完成")
        return df_filled

    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        检测并移除异常值

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        columns : list
            要处理的列，None表示所有数值列
        method : str
            检测方法：'iqr', 'zscore'
        threshold : float
            阈值（IQR倍数或Z-score值）

        Returns:
        --------
        pd.DataFrame
            处理后的数据框
        dict
            异常值统计信息
        """
        df_clean = df.copy()
        outlier_info = {}

        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                n_outliers = outlier_mask.sum()

                # 用边界值替换异常值
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound

                outlier_info[col] = {
                    'n_outliers': n_outliers,
                    'pct_outliers': n_outliers / len(df_clean) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outlier_mask = z_scores > threshold
                n_outliers = outlier_mask.sum()

                # 用均值替换异常值
                df_clean.loc[outlier_mask, col] = df_clean[col].mean()

                outlier_info[col] = {
                    'n_outliers': n_outliers,
                    'pct_outliers': n_outliers / len(df_clean) * 100
                }

        print(f"✅ 异常值处理完成，共处理 {len(outlier_info)} 列")
        return df_clean, outlier_info

    def encode_categorical(self, df, columns=None, method='auto', target=None):
        """
        编码类别特征

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        columns : list
            要编码的列，None表示所有类别列
        method : str
            编码方法：'auto', 'label', 'onehot', 'target'
        target : pd.Series
            目标变量（target编码时需要）

        Returns:
        --------
        pd.DataFrame
            编码后的数据框
        """
        df_encoded = df.copy()

        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            if method == 'auto':
                # 自动选择：类别数<10用onehot，否则用label
                n_unique = df_encoded[col].nunique()
                if n_unique < 10:
                    # One-hot编码
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(col, axis=1, inplace=True)
                else:
                    # Label编码
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le

            elif method == 'label':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le

            elif method == 'onehot':
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)

            elif method == 'target' and target is not None:
                te = TargetEncoder()
                df_encoded[col] = te.fit_transform(df_encoded[col], target)
                self.encoders[col] = te

        print(f"✅ 类别编码完成，共处理 {len(columns)} 列")
        return df_encoded

    def scale_features(self, df, columns=None, method='standard'):
        """
        特征缩放

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        columns : list
            要缩放的列，None表示所有数值列
        method : str
            缩放方法：'standard', 'minmax', 'robust'

        Returns:
        --------
        pd.DataFrame
            缩放后的数据框
        """
        df_scaled = df.copy()

        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"未知的缩放方法: {method}")

        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.scalers[method] = scaler

        print(f"✅ 特征缩放完成 ({method})，共处理 {len(columns)} 列")
        return df_scaled

    def create_polynomial_features(self, df, columns, degree=2):
        """
        创建多项式特征

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        columns : list
            要创建多项式的列
        degree : int
            多项式阶数

        Returns:
        --------
        pd.DataFrame
            包含多项式特征的数据框
        """
        df_poly = df.copy()

        for col in columns:
            for d in range(2, degree + 1):
                df_poly[f'{col}_pow{d}'] = df_poly[col] ** d

        print(f"✅ 创建多项式特征完成，度数={degree}")
        return df_poly

    def create_interaction_features(self, df, column_pairs):
        """
        创建交互特征

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        column_pairs : list of tuples
            列对列表，例如 [('col1', 'col2'), ('col3', 'col4')]

        Returns:
        --------
        pd.DataFrame
            包含交互特征的数据框
        """
        df_interact = df.copy()

        for col1, col2 in column_pairs:
            # 乘积
            df_interact[f'{col1}_x_{col2}'] = df_interact[col1] * df_interact[col2]

            # 比例（避免除零）
            if (df_interact[col2] != 0).all():
                df_interact[f'{col1}_div_{col2}'] = df_interact[col1] / df_interact[col2]

            # 加法
            df_interact[f'{col1}_plus_{col2}'] = df_interact[col1] + df_interact[col2]

            # 减法
            df_interact[f'{col1}_minus_{col2}'] = df_interact[col1] - df_interact[col2]

        print(f"✅ 创建交互特征完成，共 {len(column_pairs)} 对")
        return df_interact

    def bin_numeric_features(self, df, columns, n_bins=5, strategy='quantile'):
        """
        数值特征分箱

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        columns : list or dict
            要分箱的列，或列到分箱数的字典
        n_bins : int
            默认分箱数
        strategy : str
            分箱策略：'quantile', 'uniform', 'kmeans'

        Returns:
        --------
        pd.DataFrame
            包含分箱特征的数据框
        """
        df_binned = df.copy()

        if isinstance(columns, dict):
            for col, bins in columns.items():
                df_binned[f'{col}_binned'] = pd.cut(df_binned[col], bins=bins, labels=False)
        else:
            for col in columns:
                if strategy == 'quantile':
                    df_binned[f'{col}_binned'] = pd.qcut(df_binned[col], q=n_bins,
                                                          labels=False, duplicates='drop')
                else:
                    df_binned[f'{col}_binned'] = pd.cut(df_binned[col], bins=n_bins,
                                                         labels=False)

        print(f"✅ 数值分箱完成")
        return df_binned

    def select_features_by_variance(self, df, threshold=0.01):
        """
        基于方差的特征选择

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        threshold : float
            方差阈值

        Returns:
        --------
        pd.DataFrame
            选择后的数据框
        list
            被移除的特征列表
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()

        low_var_cols = variances[variances < threshold].index.tolist()
        df_selected = df.drop(columns=low_var_cols)

        print(f"✅ 方差选择完成，移除 {len(low_var_cols)} 个低方差特征")
        return df_selected, low_var_cols

    def select_features_by_correlation(self, df, threshold=0.95):
        """
        基于相关性的特征选择

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        threshold : float
            相关性阈值

        Returns:
        --------
        pd.DataFrame
            选择后的数据框
        list
            被移除的特征列表
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()

        # 获取上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # 找出高相关特征
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        df_selected = df.drop(columns=to_drop)

        print(f"✅ 相关性选择完成，移除 {len(to_drop)} 个高相关特征")
        return df_selected, to_drop


if __name__ == "__main__":
    # 示例用法
    fe = FeatureEngineer()

    # 生成示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'education': np.random.choice(['High', 'Medium', 'Low'], 1000),
        'city': np.random.choice(['Beijing', 'Shanghai', 'Guangzhou'], 1000),
        'score': np.random.normal(75, 15, 1000)
    })

    # 添加一些缺失值
    df.loc[np.random.choice(df.index, 50), 'income'] = np.nan

    print("原始数据:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")

    # 处理缺失值
    df = fe.handle_missing_values(df)

    # 编码类别特征
    df = fe.encode_categorical(df, method='label')

    # 创建交互特征
    df = fe.create_interaction_features(df, [('age', 'income')])

    print(f"\n处理后数据形状: {df.shape}")
    print("\n处理后数据:")
    print(df.head())
