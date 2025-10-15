"""
统计检验工具模块
提供常用的统计检验方法
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')


class StatisticalTester:
    """统计检验类"""

    def __init__(self, alpha=0.05):
        """
        初始化统计检验器

        Parameters:
        -----------
        alpha : float
            显著性水平，默认0.05
        """
        self.alpha = alpha

    def test_normality(self, data, method='shapiro'):
        """
        正态性检验

        Parameters:
        -----------
        data : array-like
            数据
        method : str
            检验方法：'shapiro' 或 'ks'

        Returns:
        --------
        dict
            检验结果
        """
        data = data.dropna() if isinstance(data, pd.Series) else data

        if method == 'shapiro':
            statistic, p_value = stats.shapiro(data)
            test_name = 'Shapiro-Wilk检验'
        elif method == 'ks':
            statistic, p_value = stats.kstest(data, 'norm')
            test_name = 'Kolmogorov-Smirnov检验'
        else:
            raise ValueError("method必须是'shapiro'或'ks'")

        result = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > self.alpha,
            'interpretation': f"{'服从' if p_value > self.alpha else '不服从'}正态分布 (p={p_value:.4f})"
        }

        return result

    def test_two_groups(self, group1, group2, test_type='auto'):
        """
        两组数据比较检验

        Parameters:
        -----------
        group1, group2 : array-like
            两组数据
        test_type : str
            检验类型：'auto', 't-test', 'mannwhitney'
            auto会根据正态性自动选择

        Returns:
        --------
        dict
            检验结果
        """
        group1 = group1.dropna() if isinstance(group1, pd.Series) else group1
        group2 = group2.dropna() if isinstance(group2, pd.Series) else group2

        if test_type == 'auto':
            # 检验正态性
            normal1 = self.test_normality(group1)['is_normal']
            normal2 = self.test_normality(group2)['is_normal']

            if normal1 and normal2:
                test_type = 't-test'
            else:
                test_type = 'mannwhitney'

        if test_type == 't-test':
            statistic, p_value = ttest_ind(group1, group2)
            test_name = 't检验'
        elif test_type == 'mannwhitney':
            statistic, p_value = mannwhitneyu(group1, group2)
            test_name = 'Mann-Whitney U检验'
        else:
            raise ValueError("test_type必须是'auto', 't-test'或'mannwhitney'")

        # 计算效果量（Cohen's d）
        cohens_d = (np.mean(group1) - np.mean(group2)) / \
                   np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)

        result = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_cohens_d(cohens_d),
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'mean_difference': np.mean(group1) - np.mean(group2),
            'interpretation': f"{'有' if p_value < self.alpha else '无'}显著差异 (p={p_value:.4f})"
        }

        return result

    def test_multiple_groups(self, *groups, test_type='auto'):
        """
        多组数据比较检验

        Parameters:
        -----------
        *groups : array-like
            多个数据组
        test_type : str
            检验类型：'auto', 'anova', 'kruskal'

        Returns:
        --------
        dict
            检验结果
        """
        groups = [g.dropna() if isinstance(g, pd.Series) else g for g in groups]

        if test_type == 'auto':
            # 检验所有组的正态性
            all_normal = all(self.test_normality(g)['is_normal'] for g in groups)
            test_type = 'anova' if all_normal else 'kruskal'

        if test_type == 'anova':
            statistic, p_value = f_oneway(*groups)
            test_name = '方差分析(ANOVA)'
        elif test_type == 'kruskal':
            statistic, p_value = kruskal(*groups)
            test_name = 'Kruskal-Wallis检验'
        else:
            raise ValueError("test_type必须是'auto', 'anova'或'kruskal'")

        result = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'n_groups': len(groups),
            'group_means': [np.mean(g) for g in groups],
            'interpretation': f"组间{'有' if p_value < self.alpha else '无'}显著差异 (p={p_value:.4f})"
        }

        return result

    def test_categorical_association(self, cat1, cat2):
        """
        类别变量关联性检验（卡方检验）

        Parameters:
        -----------
        cat1, cat2 : array-like
            两个类别变量

        Returns:
        --------
        dict
            检验结果
        """
        # 创建交叉表
        crosstab = pd.crosstab(cat1, cat2)

        # 卡方检验
        chi2, p_value, dof, expected = chi2_contingency(crosstab)

        # 计算Cramér's V（效果量）
        n = crosstab.sum().sum()
        min_dim = min(crosstab.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        result = {
            'test_name': '卡方检验',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'is_significant': p_value < self.alpha,
            'cramers_v': cramers_v,
            'effect_size': self._interpret_cramers_v(cramers_v),
            'crosstab': crosstab,
            'interpretation': f"变量间{'有' if p_value < self.alpha else '无'}显著关联 (p={p_value:.4f})"
        }

        return result

    def test_correlation(self, var1, var2, method='pearson'):
        """
        相关性检验

        Parameters:
        -----------
        var1, var2 : array-like
            两个变量
        method : str
            相关系数类型：'pearson', 'spearman', 'kendall'

        Returns:
        --------
        dict
            检验结果
        """
        var1 = var1.dropna() if isinstance(var1, pd.Series) else var1
        var2 = var2.dropna() if isinstance(var2, pd.Series) else var2

        # 确保两个变量长度一致（去除配对的缺失值）
        mask = ~(pd.isna(var1) | pd.isna(var2))
        var1, var2 = var1[mask], var2[mask]

        if method == 'pearson':
            corr, p_value = stats.pearsonr(var1, var2)
            test_name = 'Pearson相关系数'
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(var1, var2)
            test_name = 'Spearman相关系数'
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(var1, var2)
            test_name = "Kendall's Tau"
        else:
            raise ValueError("method必须是'pearson', 'spearman'或'kendall'")

        result = {
            'test_name': test_name,
            'correlation': corr,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'strength': self._interpret_correlation(abs(corr)),
            'direction': 'positive' if corr > 0 else 'negative',
            'interpretation': f"{self._interpret_correlation(abs(corr))}"
                            f"{'显著' if p_value < self.alpha else '不显著'}"
                            f"{'正' if corr > 0 else '负'}相关 (r={corr:.3f}, p={p_value:.4f})"
        }

        return result

    def _interpret_cohens_d(self, d):
        """解释Cohen's d效果量"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return '微小效应'
        elif abs_d < 0.5:
            return '小效应'
        elif abs_d < 0.8:
            return '中等效应'
        else:
            return '大效应'

    def _interpret_cramers_v(self, v):
        """解释Cramér's V效果量"""
        if v < 0.1:
            return '微小关联'
        elif v < 0.3:
            return '小关联'
        elif v < 0.5:
            return '中等关联'
        else:
            return '强关联'

    def _interpret_correlation(self, r):
        """解释相关系数"""
        if r < 0.3:
            return '弱'
        elif r < 0.7:
            return '中等'
        else:
            return '强'

    def comprehensive_analysis(self, df, feature, target):
        """
        对特征和目标变量进行综合统计分析

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        feature : str
            特征名
        target : str
            目标变量名

        Returns:
        --------
        dict
            分析结果
        """
        results = {}

        # 判断特征和目标的类型
        feature_is_numeric = pd.api.types.is_numeric_dtype(df[feature])
        target_is_numeric = pd.api.types.is_numeric_dtype(df[target])

        if feature_is_numeric and target_is_numeric:
            # 数值 vs 数值：相关性分析
            results['test_type'] = '相关性分析'
            results['pearson'] = self.test_correlation(df[feature], df[target], 'pearson')
            results['spearman'] = self.test_correlation(df[feature], df[target], 'spearman')

        elif not feature_is_numeric and target_is_numeric:
            # 类别 vs 数值：组间比较
            groups = [df[df[feature] == cat][target] for cat in df[feature].unique()]

            if len(groups) == 2:
                results['test_type'] = '两组比较'
                results['comparison'] = self.test_two_groups(groups[0], groups[1])
            else:
                results['test_type'] = '多组比较'
                results['comparison'] = self.test_multiple_groups(*groups)

        elif feature_is_numeric and not target_is_numeric:
            # 数值 vs 类别：组间比较（反过来）
            groups = [df[df[target] == cat][feature] for cat in df[target].unique()]

            if len(groups) == 2:
                results['test_type'] = '两组比较'
                results['comparison'] = self.test_two_groups(groups[0], groups[1])
            else:
                results['test_type'] = '多组比较'
                results['comparison'] = self.test_multiple_groups(*groups)

        else:
            # 类别 vs 类别：卡方检验
            results['test_type'] = '类别关联分析'
            results['chi2'] = self.test_categorical_association(df[feature], df[target])

        return results

    def print_test_results(self, results):
        """打印检验结果"""
        print("=" * 80)
        print("统计检验结果")
        print("=" * 80)

        for key, value in results.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    if k != 'crosstab':  # 跳过交叉表
                        print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

        print("=" * 80)


if __name__ == "__main__":
    # 示例用法
    tester = StatisticalTester()

    # 生成示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.normal(45, 15, 1000),
        'income': np.random.lognormal(10, 1, 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'education': np.random.choice(['High', 'Medium', 'Low'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })

    # 测试
    print("正态性检验:")
    print(tester.test_normality(df['age']))

    print("\n两组比较:")
    male = df[df['gender'] == 'M']['income']
    female = df[df['gender'] == 'F']['income']
    print(tester.test_two_groups(male, female))

    print("\n综合分析:")
    results = tester.comprehensive_analysis(df, 'age', 'target')
    tester.print_test_results(results)
