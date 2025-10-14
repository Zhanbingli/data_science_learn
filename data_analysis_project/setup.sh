#!/bin/bash
# 数据分析项目快速启动脚本

echo "=========================================="
echo "  数据分析学习项目 - 环境配置"
echo "=========================================="
echo ""

# 1. 检查Python版本
echo "1. 检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   当前Python版本: $python_version"

# 2. 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo ""
    echo "2. 创建虚拟环境..."
    python3 -m venv venv
    echo "   ✅ 虚拟环境创建成功"
else
    echo ""
    echo "2. 虚拟环境已存在，跳过创建"
fi

# 3. 激活虚拟环境
echo ""
echo "3. 激活虚拟环境..."
source venv/bin/activate
echo "   ✅ 虚拟环境已激活"

# 4. 升级pip
echo ""
echo "4. 升级pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "   ✅ pip已升级到最新版本"

# 5. 安装依赖
echo ""
echo "5. 安装项目依赖（这可能需要几分钟）..."
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ 所有依赖安装成功"
else
    echo "   ⚠️  依赖安装过程中有警告，请检查"
fi

# 6. 验证关键库
echo ""
echo "6. 验证关键库..."
python3 -c "import pandas, numpy, sklearn, lightgbm, xgboost" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ 核心库验证通过"
else
    echo "   ❌ 核心库验证失败，请手动检查"
    exit 1
fi

# 7. 创建必要的目录
echo ""
echo "7. 确保所有目录存在..."
mkdir -p data/raw data/processed data/external models reports/figures
echo "   ✅ 目录结构完整"

# 8. 初始化git（如果尚未初始化）
if [ ! -d ".git" ]; then
    echo ""
    echo "8. 初始化Git仓库..."
    git init > /dev/null 2>&1
    echo "   ✅ Git仓库已初始化"
else
    echo ""
    echo "8. Git仓库已存在，跳过初始化"
fi

# 完成
echo ""
echo "=========================================="
echo "  🎉 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 将竞赛数据放入 data/raw/ 目录"
echo "  2. 运行: jupyter lab"
echo "  3. 打开: notebooks/01_eda/01_data_loading_and_overview.ipynb"
echo ""
echo "查看快速开始指南: docs/getting_started.md"
echo ""
