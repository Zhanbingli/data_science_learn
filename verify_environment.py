#!/usr/bin/env python
"""
验证 Python 环境中机器学习相关包的安装情况

使用方法:
    python verify_environment.py
    或
    conda run -n <环境名> python verify_environment.py
"""

import sys
from typing import List, Tuple, Optional

def check_packages(packages: List[Tuple[str, Optional[str]]]) -> Tuple[int, int]:
    """
    检查包列表的安装情况

    Args:
        packages: 包列表，格式为 (包名, 别名) 的元组列表

    Returns:
        (成功数量, 总数量)
    """
    success_count = 0

    for pkg, alias in packages:
        try:
            if alias:
                exec(f'import {pkg} as {alias}')
                if pkg == 'matplotlib.pyplot':
                    version = eval('plt.matplotlib.__version__')
                else:
                    version = eval(f'{alias}.__version__')
            else:
                exec(f'import {pkg}')
                version = eval(f'{pkg}.__version__')

            print(f'  ✅ {pkg:30s} v{version}')
            success_count += 1

        except ImportError as e:
            print(f'  ❌ {pkg:30s} 未安装')

        except Exception as e:
            error_msg = str(e)[:45] + '...' if len(str(e)) > 45 else str(e)
            print(f'  ⚠️  {pkg:30s} {type(e).__name__}: {error_msg}')

    return success_count, len(packages)


def main():
    """主函数"""
    print('='*70)
    print('🎯 Python 环境包验证工具')
    print('='*70)

    # 显示环境信息
    print(f'\n📍 环境信息:')
    print(f'  Python 版本: {sys.version.split()[0]}')
    print(f'  Python 路径: {sys.executable}')

    # 必需的包
    required_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('matplotlib', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('sklearn', None),
    ]

    # 机器学习相关包
    ml_packages = [
        ('lightgbm', 'lgb'),
        ('xgboost', 'xgb'),
        ('catboost', 'cb'),
        ('optuna', None),
    ]

    # 其他有用的包
    optional_packages = [
        ('shap', None),
        ('scipy', None),
        ('joblib', None),
    ]

    # 检查必需包
    print(f'\n📦 必需包 (数据科学基础):')
    print('-'*70)
    req_success, req_total = check_packages(required_packages)

    # 检查机器学习包
    print(f'\n📦 机器学习包:')
    print('-'*70)
    ml_success, ml_total = check_packages(ml_packages)

    # 检查可选包
    print(f'\n📦 可选包:')
    print('-'*70)
    opt_success, opt_total = check_packages(optional_packages)

    # 总结
    print('='*70)
    total_success = req_success + ml_success + opt_success
    total_packages = req_total + ml_total + opt_total

    print(f'\n📊 总结:')
    print(f'  必需包: {req_success}/{req_total} ✓')
    print(f'  ML 包:  {ml_success}/{ml_total} ✓')
    print(f'  可选包: {opt_success}/{opt_total} ✓')
    print(f'  总计:   {total_success}/{total_packages} ✓')

    # 最终判断
    print('\n' + '='*70)
    if req_success == req_total and ml_success >= ml_total - 1:
        print('🎉 环境配置完成！可以开始机器学习项目了！')
        return 0
    elif req_success == req_total:
        print('⚠️  基础包已安装，但缺少一些 ML 包')
        print('💡 建议运行: conda install -c conda-forge lightgbm xgboost catboost optuna')
        return 1
    else:
        print('❌ 环境配置不完整，缺少必需的包')
        print('💡 请查看上方的错误信息并安装缺失的包')
        return 2


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
