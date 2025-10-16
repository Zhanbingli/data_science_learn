"""
数据分析项目核心模块
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .utils.data_loader import DataLoader, reduce_mem_usage
from .visualization.plot_templates import EDAPlotter
from .utils.statistical_tests import StatisticalTester

__all__ = [
    'DataLoader',
    'reduce_mem_usage',
    'EDAPlotter',
    'StatisticalTester',
]
