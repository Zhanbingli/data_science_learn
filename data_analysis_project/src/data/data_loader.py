"""
数据加载模块
包含数据读取、内存优化等功能
"""

import pandas as pd
import numpy as np
from pathlib import Path


def reduce_mem_usage(df, verbose=True):
    """
    降低DataFrame的内存占用

    Parameters
    ----------
    df : pd.DataFrame
        需要优化的数据框
    verbose : bool
        是否打印优化信息

    Returns
    -------
    pd.DataFrame
        优化后的数据框
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f'内存使用: {start_mem:.2f} MB -> {end_mem:.2f} MB '
              f'(减少 {100 * (start_mem - end_mem) / start_mem:.1f}%)')

    return df


def load_data(train_path, test_path, optimize_memory=True):
    """
    加载训练集和测试集

    Parameters
    ----------
    train_path : str
        训练集路径
    test_path : str
        测试集路径
    optimize_memory : bool
        是否进行内存优化

    Returns
    -------
    tuple
        (train, test) 数据框元组
    """
    print("加载数据...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"训练集形状: {train.shape}")
    print(f"测试集形状: {test.shape}")

    if optimize_memory:
        print("\n优化内存...")
        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

    return train, test


def save_data(df, path, format='csv'):
    """
    保存数据

    Parameters
    ----------
    df : pd.DataFrame
        要保存的数据框
    path : str
        保存路径
    format : str
        保存格式 ('csv', 'pickle', 'parquet')
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if format == 'csv':
        df.to_csv(path, index=False)
    elif format == 'pickle':
        df.to_pickle(path)
    elif format == 'parquet':
        df.to_parquet(path)
    else:
        raise ValueError(f"不支持的格式: {format}")

    print(f"✅ 数据已保存至: {path}")


def get_feature_types(df, exclude_cols=None):
    """
    自动识别特征类型

    Parameters
    ----------
    df : pd.DataFrame
        数据框
    exclude_cols : list
        要排除的列名

    Returns
    -------
    dict
        包含 'numeric' 和 'categorical' 键的字典
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 排除指定列
    numeric_features = [col for col in numeric_features if col not in exclude_cols]
    categorical_features = [col for col in categorical_features if col not in exclude_cols]

    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }
