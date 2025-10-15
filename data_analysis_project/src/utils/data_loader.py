"""
数据加载工具模块
提供标准化的数据读取和初步处理功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类"""

    def __init__(self, config_path='config/config.yaml'):
        """
        初始化数据加载器

        Parameters:
        -----------
        config_path : str
            配置文件路径
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data']['raw_dir'])

    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_train_data(self, **kwargs):
        """
        加载训练数据

        Parameters:
        -----------
        **kwargs : dict
            pd.read_csv的额外参数

        Returns:
        --------
        pd.DataFrame
            训练数据
        """
        train_file = self.data_dir / self.config['data']['train_file']
        logger.info(f"加载训练数据: {train_file}")

        df = pd.read_csv(train_file, **kwargs)
        logger.info(f"训练数据形状: {df.shape}")

        return df

    def load_test_data(self, **kwargs):
        """
        加载测试数据

        Parameters:
        -----------
        **kwargs : dict
            pd.read_csv的额外参数

        Returns:
        --------
        pd.DataFrame
            测试数据
        """
        test_file = self.data_dir / self.config['data']['test_file']
        logger.info(f"加载测试数据: {test_file}")

        df = pd.read_csv(test_file, **kwargs)
        logger.info(f"测试数据形状: {df.shape}")

        return df

    def load_data(self, file_path, **kwargs):
        """
        加载指定路径的数据

        Parameters:
        -----------
        file_path : str or Path
            数据文件路径
        **kwargs : dict
            pd.read_csv的额外参数

        Returns:
        --------
        pd.DataFrame
            数据
        """
        logger.info(f"加载数据: {file_path}")
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"数据形状: {df.shape}")
        return df

    def get_data_info(self, df):
        """
        获取数据基本信息

        Parameters:
        -----------
        df : pd.DataFrame
            数据框

        Returns:
        --------
        dict
            数据信息字典
        """
        info = {
            'shape': df.shape,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
        }

        # 统计不同类型的特征数量
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()

        info['n_numeric_features'] = len(numeric_features)
        info['n_categorical_features'] = len(categorical_features)
        info['numeric_features'] = numeric_features
        info['categorical_features'] = categorical_features

        return info

    def print_data_summary(self, df, name='数据集'):
        """
        打印数据摘要信息

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        name : str
            数据集名称
        """
        info = self.get_data_info(df)

        print("=" * 80)
        print(f"{name} - 数据摘要")
        print("=" * 80)
        print(f"\n📊 基本信息:")
        print(f"   形状: {info['shape']} (行 × 列)")
        print(f"   内存占用: {info['memory_usage']:.2f} MB")
        print(f"   重复行: {info['duplicate_rows']} 行")

        print(f"\n📈 特征类型:")
        print(f"   数值型特征: {info['n_numeric_features']} 个")
        print(f"   类别型特征: {info['n_categorical_features']} 个")

        print(f"\n❓ 缺失值统计:")
        missing_cols = {k: v for k, v in info['missing_percentage'].items() if v > 0}
        if missing_cols:
            for col, pct in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True):
                print(f"   {col}: {info['missing_values'][col]} ({pct:.2f}%)")
        else:
            print("   ✅ 无缺失值")

        print("=" * 80)


def reduce_mem_usage(df, verbose=True):
    """
    减少DataFrame的内存占用

    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    verbose : bool
        是否打印信息

    Returns:
    --------
    pd.DataFrame
        优化后的数据框
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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
        logger.info(f'内存使用从 {start_mem:.2f} MB 降至 {end_mem:.2f} MB '
                   f'(减少 {100 * (start_mem - end_mem) / start_mem:.1f}%)')

    return df


if __name__ == "__main__":
    # 示例用法
    loader = DataLoader()

    # 加载数据
    train = loader.load_train_data()
    test = loader.load_test_data()

    # 打印摘要
    loader.print_data_summary(train, '训练集')
    loader.print_data_summary(test, '测试集')

    # 内存优化
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
