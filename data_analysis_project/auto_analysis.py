"""
自动化数据分析脚本
只需读入数据，一键生成完整的论文级别分析报告

使用方法：
python auto_analysis.py --train data/raw/train.csv --target charges

或者在Jupyter中：
from auto_analysis import AutoAnalyzer
analyzer = AutoAnalyzer('data/raw/train.csv', target='charges')
analyzer.run_full_analysis()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.append('src')

from utils.data_loader import DataLoader, reduce_mem_usage
from visualization.plot_templates import EDAPlotter
from utils.statistical_tests import StatisticalTester


class AutoAnalyzer:
    """自动化分析器"""

    def __init__(self, data_path, target=None, output_dir='reports/auto_analysis'):
        """
        初始化自动分析器

        Parameters:
        -----------
        data_path : str
            数据文件路径
        target : str, optional
            目标变量名
        output_dir : str
            输出目录
        """
        print("="*80)
        print("🚀 论文级别自动化数据分析系统")
        print("="*80)

        self.data_path = data_path
        self.target = target
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化工具
        self.plotter = EDAPlotter()
        self.tester = StatisticalTester()

        # 加载数据
        print(f"\n📂 加载数据: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"✅ 数据加载成功! 形状: {self.df.shape}")

        # 内存优化
        self.df = reduce_mem_usage(self.df, verbose=True)

        # 识别特征类型
        self._identify_features()

        # 如果没有指定target，尝试自动识别
        if self.target is None:
            self.target = self._auto_detect_target()

        print(f"\n🎯 目标变量: {self.target}")
        print(f"📊 数值型特征: {len(self.numeric_features)} 个")
        print(f"📊 类别型特征: {len(self.categorical_features)} 个")

    def _identify_features(self):
        """识别特征类型"""
        self.numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

        # ID列检测（通常命名为id, ID, index等）
        id_patterns = ['id', 'index', 'key']
        self.id_columns = [col for col in self.df.columns
                          if any(pattern in col.lower() for pattern in id_patterns)]

        if self.id_columns:
            print(f"\n🔑 检测到ID列: {self.id_columns}")
            # 从特征列表中移除ID列
            self.numeric_features = [f for f in self.numeric_features if f not in self.id_columns]

    def _auto_detect_target(self):
        """自动检测目标变量"""
        # 常见的目标变量名称
        target_patterns = ['target', 'label', 'y', 'class', 'outcome', 'result']

        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in target_patterns):
                print(f"\n💡 自动检测到目标变量: {col}")
                return col

        # 如果没有找到，返回最后一列
        print(f"\n⚠️ 未找到明确的目标变量，使用最后一列: {self.df.columns[-1]}")
        return self.df.columns[-1]

    def run_full_analysis(self, save_report=True):
        """运行完整分析"""
        print("\n" + "="*80)
        print("📊 开始完整分析...")
        print("="*80)

        # 1. 数据概览
        print("\n[1/6] 数据概览分析...")
        self.data_overview()

        # 2. 数据质量检查
        print("\n[2/6] 数据质量检查...")
        self.quality_check()

        # 3. 单变量分析
        print("\n[3/6] 单变量分析...")
        self.univariate_analysis()

        # 4. 双变量分析
        print("\n[4/6] 双变量分析（特征vs目标）...")
        self.bivariate_analysis()

        # 5. 多变量分析
        print("\n[5/6] 多变量分析（相关性）...")
        self.multivariate_analysis()

        # 6. 统计检验
        print("\n[6/6] 统计显著性检验...")
        self.statistical_tests()

        # 生成报告
        if save_report:
            print("\n📝 生成分析报告...")
            self.generate_report()

        print("\n" + "="*80)
        print("✅ 分析完成!")
        print(f"📁 结果保存在: {self.output_dir}")
        print("="*80)

    def data_overview(self):
        """数据概览"""
        print("\n" + "-"*80)
        print("📊 数据基本信息")
        print("-"*80)

        print(f"\n数据形状: {self.df.shape[0]:,} 行 × {self.df.shape[1]} 列")
        print(f"内存占用: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print(f"\n特征类型分布:")
        print(f"  - 数值型: {len(self.numeric_features)} 个")
        print(f"  - 类别型: {len(self.categorical_features)} 个")

        print(f"\n前5行数据:")
        print(self.df.head())

        print(f"\n数据类型:")
        print(self.df.dtypes)

        print(f"\n基本统计:")
        print(self.df.describe())

    def quality_check(self):
        """数据质量检查"""
        print("\n" + "-"*80)
        print("🔍 数据质量检查")
        print("-"*80)

        # 缺失值
        missing = self.df.isnull().sum()
        missing_pct = 100 * missing / len(self.df)

        if missing.sum() > 0:
            print("\n❌ 缺失值检测:")
            missing_df = pd.DataFrame({
                '缺失数量': missing[missing > 0],
                '缺失率(%)': missing_pct[missing > 0]
            }).sort_values('缺失率(%)', ascending=False)
            print(missing_df)

            # 可视化缺失值
            if len(missing_df) > 0:
                fig, ax = plt.subplots(figsize=(12, max(6, len(missing_df) * 0.3)))
                missing_df['缺失率(%)'].plot(kind='barh', ax=ax,
                                           color=self.plotter.config.colors['warning'])
                ax.set_xlabel('缺失率 (%)', fontweight='bold')
                ax.set_title('特征缺失率分布', fontsize=16, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / '01_missing_values.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✅ 缺失值图表已保存")
        else:
            print("\n✅ 无缺失值")

        # 重复行
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"\n⚠️ 检测到 {duplicates} 行重复数据 ({duplicates/len(self.df)*100:.2f}%)")
        else:
            print("\n✅ 无重复行")

        # 常数特征
        constant_features = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_features:
            print(f"\n⚠️ 常数特征（无信息量）: {constant_features}")
        else:
            print("\n✅ 无常数特征")

        # 唯一值统计
        print(f"\n📊 唯一值统计:")
        unique_counts = self.df.nunique().sort_values()
        print(unique_counts.head(10))

    def univariate_analysis(self):
        """单变量分析"""
        print("\n正在生成单变量分析图表...")

        # 分析数值型特征
        for feature in self.numeric_features[:10]:  # 只分析前10个特征，避免太多
            if feature != self.target:
                try:
                    save_path = self.output_dir / f'univariate_numeric_{feature}.png'
                    self.plotter.plot_numeric_distribution(
                        self.df, feature,
                        target=self.target if self.target in self.df.columns else None,
                        save_path=save_path
                    )
                    print(f"  ✅ {feature} 分析完成")
                except Exception as e:
                    print(f"  ❌ {feature} 分析失败: {e}")

        # 分析类别型特征
        for feature in self.categorical_features[:10]:
            if feature != self.target:
                try:
                    save_path = self.output_dir / f'univariate_categorical_{feature}.png'
                    self.plotter.plot_categorical_distribution(
                        self.df, feature, save_path=save_path
                    )
                    print(f"  ✅ {feature} 分析完成")
                except Exception as e:
                    print(f"  ❌ {feature} 分析失败: {e}")

        # 目标变量分析
        if self.target in self.df.columns:
            try:
                save_path = self.output_dir / f'target_analysis_{self.target}.png'
                self.plotter.plot_target_analysis(self.df, self.target, save_path=save_path)
                print(f"  ✅ 目标变量 {self.target} 分析完成")
            except Exception as e:
                print(f"  ❌ 目标变量分析失败: {e}")

    def bivariate_analysis(self):
        """双变量分析"""
        if self.target not in self.df.columns:
            print("⚠️ 未指定目标变量，跳过双变量分析")
            return

        print("\n正在进行双变量分析...")

        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 分析前4个最相关的特征
        if pd.api.types.is_numeric_dtype(self.df[self.target]):
            # 回归问题：计算相关性
            correlations = []
            for feature in self.numeric_features:
                if feature != self.target:
                    corr = self.df[feature].corr(self.df[self.target])
                    correlations.append((feature, abs(corr)))

            top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:4]

            for idx, (feature, corr) in enumerate(top_features):
                ax = axes[idx]
                ax.scatter(self.df[feature], self.df[self.target], alpha=0.5,
                          color=self.plotter.config.colors['primary'])
                ax.set_xlabel(feature, fontweight='bold')
                ax.set_ylabel(self.target, fontweight='bold')
                ax.set_title(f'{feature} vs {self.target}\n相关系数: {corr:.3f}',
                           fontweight='bold')
                ax.grid(True, alpha=0.3)

        else:
            # 分类问题：箱线图
            top_features = self.numeric_features[:4]
            for idx, feature in enumerate(top_features):
                ax = axes[idx]
                self.df.boxplot(column=feature, by=self.target, ax=ax)
                ax.set_title(f'{feature} 按 {self.target} 分组', fontweight='bold')
                ax.set_ylabel(feature, fontweight='bold')

        plt.suptitle('特征与目标变量关系分析', fontsize=18, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_bivariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ 双变量分析图表已保存")

    def multivariate_analysis(self):
        """多变量分析"""
        print("\n正在进行多变量分析...")

        # 相关性热力图
        numeric_df = self.df[self.numeric_features].select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 1:
            save_path = self.output_dir / '03_correlation_heatmap.png'
            self.plotter.plot_correlation_heatmap(numeric_df, save_path=save_path)
            print("  ✅ 相关性热力图已保存")

        # 配对图（如果特征不太多）
        if len(numeric_df.columns) <= 5 and len(self.df) <= 5000:
            print("  正在生成配对图（可能需要一些时间）...")
            try:
                important_features = numeric_df.columns[:5].tolist()
                if self.target in self.df.columns and self.target not in important_features:
                    important_features.append(self.target)

                pairplot_data = self.df[important_features]
                sns.pairplot(pairplot_data, diag_kind='kde', corner=True)
                plt.savefig(self.output_dir / '04_pairplot.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ 配对图已保存")
            except Exception as e:
                print(f"  ⚠️ 配对图生成失败: {e}")

    def statistical_tests(self):
        """统计显著性检验"""
        if self.target not in self.df.columns:
            print("⚠️ 未指定目标变量，跳过统计检验")
            return

        print("\n正在进行统计检验...")

        results = []

        # 对每个特征进行检验
        for feature in (self.numeric_features + self.categorical_features):
            if feature != self.target and feature not in self.id_columns:
                try:
                    test_result = self.tester.comprehensive_analysis(
                        self.df, feature, self.target
                    )
                    results.append({
                        'feature': feature,
                        'test_type': test_result['test_type'],
                        'result': test_result
                    })
                except Exception as e:
                    print(f"  ⚠️ {feature} 检验失败: {e}")

        # 保存结果
        self.test_results = results
        print(f"  ✅ 完成 {len(results)} 个特征的统计检验")

    def generate_report(self):
        """生成分析报告"""
        report_path = self.output_dir / 'analysis_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("论文级别数据分析报告\n")
            f.write("="*80 + "\n\n")

            # 1. 数据概览
            f.write("1. 数据概览\n")
            f.write("-"*80 + "\n")
            f.write(f"数据形状: {self.df.shape}\n")
            f.write(f"数值型特征: {len(self.numeric_features)} 个\n")
            f.write(f"类别型特征: {len(self.categorical_features)} 个\n")
            f.write(f"目标变量: {self.target}\n\n")

            # 2. 数据质量
            f.write("2. 数据质量\n")
            f.write("-"*80 + "\n")
            missing = self.df.isnull().sum()
            if missing.sum() > 0:
                f.write("缺失值:\n")
                for col in missing[missing > 0].index:
                    f.write(f"  - {col}: {missing[col]} ({missing[col]/len(self.df)*100:.2f}%)\n")
            else:
                f.write("✅ 无缺失值\n")

            f.write(f"\n重复行: {self.df.duplicated().sum()}\n\n")

            # 3. 统计检验结果
            if hasattr(self, 'test_results'):
                f.write("3. 统计显著性检验\n")
                f.write("-"*80 + "\n")
                for result in self.test_results:
                    f.write(f"\n特征: {result['feature']}\n")
                    f.write(f"检验类型: {result['test_type']}\n")

                    # 根据检验类型提取关键信息
                    if 'correlation' in result['result']:
                        corr_result = result['result'].get('pearson', result['result'].get('spearman'))
                        if corr_result:
                            f.write(f"相关系数: {corr_result['correlation']:.3f}\n")
                            f.write(f"p值: {corr_result['p_value']:.4f}\n")
                            f.write(f"解释: {corr_result['interpretation']}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("分析完成\n")
            f.write("="*80 + "\n")

        print(f"  ✅ 分析报告已保存: {report_path}")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='自动化数据分析')
    parser.add_argument('--train', type=str, required=True, help='训练数据路径')
    parser.add_argument('--target', type=str, help='目标变量名')
    parser.add_argument('--output', type=str, default='reports/auto_analysis', help='输出目录')

    args = parser.parse_args()

    # 运行分析
    analyzer = AutoAnalyzer(args.train, target=args.target, output_dir=args.output)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
