"""
论文级别可视化模板
提供美观、专业的图表绘制功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PlotConfig:
    """绘图配置类"""

    def __init__(self):
        self.setup_style()

    def setup_style(self):
        """设置全局绘图样式"""
        # 样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')

        # 字体
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 18

        # 中文支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 线条和标记
        plt.rcParams['lines.linewidth'] = 2.5
        plt.rcParams['lines.markersize'] = 8

        # 颜色
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#2ECC71',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'neutral': '#95A5A6'
        }


class EDAPlotter:
    """探索性数据分析绘图类"""

    def __init__(self):
        self.config = PlotConfig()

    def plot_numeric_distribution(self, df, feature, target=None, figsize=(16, 10), save_path=None):
        """
        绘制数值型特征的完整分布分析图

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        feature : str
            特征名
        target : str, optional
            目标变量名（用于分组对比）
        figsize : tuple
            图表尺寸
        save_path : str, optional
            保存路径
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        data = df[feature].dropna()

        # 1. 直方图 + KDE
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(data, bins=50, edgecolor='black', alpha=0.7,
                color=self.config.colors['primary'])
        ax1.axvline(data.mean(), color=self.config.colors['danger'],
                   linestyle='--', linewidth=2, label=f'均值: {data.mean():.2f}')
        ax1.axvline(data.median(), color=self.config.colors['success'],
                   linestyle='--', linewidth=2, label=f'中位数: {data.median():.2f}')
        ax1.set_title(f'{feature} - 分布直方图', fontweight='bold', pad=15)
        ax1.set_xlabel(feature, fontweight='bold')
        ax1.set_ylabel('频数', fontweight='bold')
        ax1.legend(frameon=True, shadow=True)
        ax1.grid(axis='y', alpha=0.3)

        # 2. 核密度图
        ax2 = fig.add_subplot(gs[1, 0])
        data.plot(kind='kde', ax=ax2, linewidth=3,
                 color=self.config.colors['secondary'])
        ax2.set_title(f'{feature} - 核密度估计', fontweight='bold')
        ax2.set_xlabel(feature, fontweight='bold')
        ax2.set_ylabel('密度', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. 箱线图
        ax3 = fig.add_subplot(gs[1, 1])
        if target and target in df.columns:
            df.boxplot(column=feature, by=target, ax=ax3)
            ax3.set_title(f'{feature} 按 {target} 分组', fontweight='bold')
            plt.sca(ax3)
            plt.xticks(rotation=0)
        else:
            ax3.boxplot(data, vert=True)
            ax3.set_title(f'{feature} - 箱线图', fontweight='bold')
        ax3.set_ylabel(feature, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        # 4. QQ图
        ax4 = fig.add_subplot(gs[2, 0])
        stats.probplot(data, dist="norm", plot=ax4)
        ax4.set_title(f'{feature} - QQ图（正态性检验）', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. 统计摘要
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        stats_text = f"""
        📊 统计摘要
        ━━━━━━━━━━━━━━━━━━━━
        样本量:    {len(data):,}
        缺失值:    {df[feature].isnull().sum()} ({df[feature].isnull().sum()/len(df)*100:.2f}%)

        均值:      {data.mean():.4f}
        中位数:    {data.median():.4f}
        标准差:    {data.std():.4f}

        最小值:    {data.min():.4f}
        25%分位:   {data.quantile(0.25):.4f}
        75%分位:   {data.quantile(0.75):.4f}
        最大值:    {data.max():.4f}

        偏度:      {data.skew():.4f}
        峰度:      {data.kurtosis():.4f}

        异常值:    {self._count_outliers(data)} ({self._count_outliers(data)/len(data)*100:.2f}%)
        """

        ax5.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center')

        fig.suptitle(f'特征分析: {feature}', fontsize=20, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

    def _count_outliers(self, data):
        """使用IQR方法统计异常值数量"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return ((data < lower) | (data > upper)).sum()

    def plot_categorical_distribution(self, df, feature, top_n=15, figsize=(14, 6), save_path=None):
        """
        绘制类别型特征分布

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        feature : str
            特征名
        top_n : int
            显示前N个类别
        figsize : tuple
            图表尺寸
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        value_counts = df[feature].value_counts().head(top_n)
        value_pct = df[feature].value_counts(normalize=True).head(top_n) * 100

        # 1. 频数柱状图
        axes[0].barh(range(len(value_counts)), value_counts.values,
                    color=self.config.colors['primary'], edgecolor='black')
        axes[0].set_yticks(range(len(value_counts)))
        axes[0].set_yticklabels(value_counts.index)
        axes[0].set_xlabel('频数', fontweight='bold')
        axes[0].set_title(f'{feature} - Top {top_n} 类别频数', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(value_counts.values):
            axes[0].text(v + max(value_counts)*0.01, i, f'{v:,}',
                        va='center', fontweight='bold')

        # 2. 占比柱状图
        axes[1].barh(range(len(value_pct)), value_pct.values,
                    color=self.config.colors['accent'], edgecolor='black')
        axes[1].set_yticks(range(len(value_pct)))
        axes[1].set_yticklabels(value_pct.index)
        axes[1].set_xlabel('占比 (%)', fontweight='bold')
        axes[1].set_title(f'{feature} - Top {top_n} 类别占比', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)

        # 添加百分比标签
        for i, v in enumerate(value_pct.values):
            axes[1].text(v + max(value_pct)*0.01, i, f'{v:.1f}%',
                        va='center', fontweight='bold')

        fig.suptitle(f'类别特征分析: {feature}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

    def plot_correlation_heatmap(self, df, figsize=(14, 12), save_path=None):
        """
        绘制相关性热力图

        Parameters:
        -----------
        df : pd.DataFrame
            数据框（只包含数值型特征）
        figsize : tuple
            图表尺寸
        save_path : str, optional
            保存路径
        """
        # 计算相关系数
        corr = df.select_dtypes(include=[np.number]).corr()

        # 创建mask（只显示下三角）
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热力图
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)

        ax.set_title('特征相关性热力图', fontsize=18, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

    def plot_target_analysis(self, df, target, figsize=(16, 6), save_path=None):
        """
        绘制目标变量分析图

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        target : str
            目标变量名
        figsize : tuple
            图表尺寸
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 判断是分类还是回归
        if df[target].dtype == 'object' or df[target].nunique() < 20:
            # 分类问题
            value_counts = df[target].value_counts()

            # 柱状图
            axes[0].bar(range(len(value_counts)), value_counts.values,
                       color=self.config.colors['primary'], edgecolor='black')
            axes[0].set_xticks(range(len(value_counts)))
            axes[0].set_xticklabels(value_counts.index, rotation=45)
            axes[0].set_ylabel('频数', fontweight='bold')
            axes[0].set_title('类别频数分布', fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)

            # 饼图
            axes[1].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                       startangle=90, colors=sns.color_palette('husl', len(value_counts)))
            axes[1].set_title('类别占比分布', fontweight='bold')

            # 统计表
            axes[2].axis('off')
            stats_df = pd.DataFrame({
                '类别': value_counts.index,
                '数量': value_counts.values,
                '占比': [f"{v:.2%}" for v in value_counts / value_counts.sum()]
            })

            table = axes[2].table(cellText=stats_df.values,
                                colLabels=stats_df.columns,
                                cellLoc='center', loc='center',
                                bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)

            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_facecolor(self.config.colors['primary'])
                    cell.set_text_props(weight='bold', color='white')

        else:
            # 回归问题
            data = df[target].dropna()

            # 直方图
            axes[0].hist(data, bins=50, edgecolor='black', alpha=0.7,
                        color=self.config.colors['primary'])
            axes[0].axvline(data.mean(), color=self.config.colors['danger'],
                          linestyle='--', linewidth=2, label=f'均值: {data.mean():.2f}')
            axes[0].set_xlabel(target, fontweight='bold')
            axes[0].set_ylabel('频数', fontweight='bold')
            axes[0].set_title('目标变量分布', fontweight='bold')
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)

            # KDE图
            data.plot(kind='kde', ax=axes[1], linewidth=3,
                     color=self.config.colors['secondary'])
            axes[1].set_xlabel(target, fontweight='bold')
            axes[1].set_ylabel('密度', fontweight='bold')
            axes[1].set_title('核密度估计', fontweight='bold')
            axes[1].grid(True, alpha=0.3)

            # 箱线图
            axes[2].boxplot(data, vert=True)
            axes[2].set_ylabel(target, fontweight='bold')
            axes[2].set_title('箱线图', fontweight='bold')
            axes[2].grid(axis='y', alpha=0.3)

        fig.suptitle(f'目标变量分析: {target}', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

    def plot_feature_importance(self, feature_names, importances, top_n=20,
                               figsize=(12, 8), save_path=None):
        """
        绘制特征重要性图

        Parameters:
        -----------
        feature_names : list
            特征名列表
        importances : array-like
            重要性分数
        top_n : int
            显示前N个特征
        figsize : tuple
            图表尺寸
        save_path : str, optional
            保存路径
        """
        # 排序
        indices = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor='black')

        # 设置标签
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('重要性分数', fontweight='bold', fontsize=14)
        ax.set_title(f'Top {top_n} 特征重要性', fontsize=18, fontweight='bold', pad=20)

        # 添加数值标签
        for i, v in enumerate(importances[indices]):
            ax.text(v + max(importances)*0.01, i, f'{v:.4f}',
                   va='center', fontweight='bold')

        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()


if __name__ == "__main__":
    # 示例用法
    plotter = EDAPlotter()

    # 生成示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.normal(45, 15, 1000),
        'income': np.random.lognormal(10, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })

    # 测试绘图
    plotter.plot_numeric_distribution(df, 'age', target='target')
    plotter.plot_categorical_distribution(df, 'category')
    plotter.plot_correlation_heatmap(df)
    plotter.plot_target_analysis(df, 'target')
