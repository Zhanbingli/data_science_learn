# 🚀 Python 学习与实战项目合集

> 从零开始的 Python 学习之旅 - 涵盖基础语法、API 调用、数据分析和 AI 应用开发

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

## 📖 项目简介

这是一个完整的 Python 学习实战仓库，记录了从 Python 小白到能够独立开发完整项目的成长过程。这里记录了我学习数据分析、参与 Kaggle 竞赛、开发 AI 应用的一些痕迹。

通过这个仓库的学习，希望能将数据分析和 AI 技术应用到未来的生活和工作中。这是一个挺难，但充满趣味的旅程。我相信通过掌握这些技能，可以更好地融入到技术的世界。

**如果你也想通过自学掌握 Python 和数据分析，不妨克隆这个仓库去学习！**

### 项目包含：

- 🐍 **Python 基础学习**：面向对象、函数、数据结构等核心概念
- 🌐 **API 调用实战**：GitHub API、DeepSeek API、OpenAI API 的完整使用
- 📊 **数据分析项目**：完整的数据分析工作流程和机器学习实践
- 🤖 **AI 应用开发**：基于 DeepSeek 的智能聊天助手
- 🏆 **Kaggle 竞赛**：实战数据建模和模型优化

---

## 🎯 适合人群

- ✅ Python 初学者，想要系统学习编程
- ✅ 希望掌握 API 调用和数据处理的开发者
- ✅ 对 AI 应用开发感兴趣的学习者
- ✅ 参加 Kaggle 竞赛的实践者
- ✅ 需要完整项目参考的学习者

---

## 📂 项目结构

```
data/
├── 📚 学习教程
│   ├── test.py                          # 面向对象编程完全指南
│   ├── learn_api.py                     # API 调用完整教程
│   ├── learn_openai.py                  # DeepSeek API 使用教程
│   ├── learn_pandas.ipynb               # Pandas 数据处理实战
│   └── github_api_helper.py             # GitHub API 封装工具
│
├── 🤖 AI 应用项目
│   └── hand_write_code.py               # DeepSeek 智能聊天助手（核心项目）
│       ├── 交互式聊天界面
│       ├── 对话历史管理
│       ├── 流式输出支持
│       └── 思考动画效果
│
├── 📊 数据分析项目
│   └── data_analysis_project/           # 完整的数据分析工作流
│       ├── notebooks/                   # Jupyter 笔记本
│       │   ├── 01_eda/                 # 探索性数据分析
│       │   ├── 02_feature_engineering/ # 特征工程
│       │   └── 03_modeling/            # 模型训练与优化
│       ├── src/                        # 源代码
│       │   ├── data/                   # 数据处理模块
│       │   ├── features/               # 特征工程模块
│       │   ├── models/                 # 模型训练模块
│       │   └── evaluation/             # 模型评估模块
│       └── reports/                    # 分析报告
│
├── 📝 学习笔记
│   ├── Untitled.ipynb                   # API 调用练习
│   ├── Untitled1.ipynb                  # 代码实践笔记
│   └── model_optimization_tutorial.ipynb # 模型优化教程
│
├── 📖 文档
│   ├── TUTORIAL_GUIDE.md                # 教程指南
│   ├── TUTORIAL_COMPLETE.md             # 完整教程
│   ├── OUTLIER_HANDLING_GUIDE.md        # 异常值处理指南
│   └── data_analysis_pipeline.md        # 数据分析流程
│
└── ⚙️ 配置文件
    ├── .env                             # 环境变量（API Keys）
    ├── .gitignore                       # Git 忽略规则
    └── requirements.txt                 # 依赖包列表
```

---

## 🌟 核心项目亮点

### 1️⃣ DeepSeek 智能聊天助手 `hand_write_code.py`

一个功能完整的 AI 聊天应用，具备以下特性：

#### ✨ 核心功能
- 🎭 **交互式聊天界面**：命令行实时对话
- 💾 **对话历史管理**：保存/加载/重置对话记录
- 🌊 **流式输出**：像 ChatGPT 一样逐字显示回复
- 💭 **思考动画**：DeepSeek Reasoner 模型专属旋转动画
- 🎨 **自定义角色**：通过系统提示词定制 AI 行为

#### 📋 使用示例

```python
from hand_write_code import DeepSeekChatbot

# 创建聊天机器人
bot = DeepSeekChatbot(
    system_prompt="你是一个 Python 编程专家"
)

# 普通对话
response = bot.chat("什么是装饰器？")
print(response)

# 流式对话（带动画）
bot.chat("请解释多线程", stream=True)

# 查看历史
bot.show_history()

# 保存对话
bot.save_history("my_chat.json")
```

#### 🎮 交互命令

| 命令 | 功能 |
|------|------|
| 直接输入 | 与 AI 对话 |
| `quit` / `exit` / `退出` | 退出程序 |
| `reset` / `重置` | 清空对话历史 |
| `history` / `历史` | 查看对话记录 |
| `save` / `保存` | 保存到 JSON 文件 |

---

### 2️⃣ API 调用教程合集

#### GitHub API `github_api_helper.py`
```python
from github_api_helper import GitHubAPI

github = GitHubAPI()

# 测试连接
github.test_connection()

# 获取用户信息
user = github.get_user('torvalds')

# 搜索仓库
repos = github.search_repositories('python machine learning')
```

#### DeepSeek API `learn_openai.py`
完整的 DeepSeek API 使用教程，包含：
- ✅ 基础对话
- ✅ 参数调整（temperature、max_tokens）
- ✅ 流式输出
- ✅ 代码生成
- ✅ 文本分析
- ✅ 翻译功能
- ✅ 智能问答系统

---

### 3️⃣ 数据分析完整工作流

位于 `data_analysis_project/` 目录，包含完整的 Kaggle 竞赛实战经验：

#### 📊 完整流程
1. **数据加载与概览**
   - 数据读取
   - 基本统计
   - 缺失值分析

2. **探索性数据分析（EDA）**
   - 分布分析
   - 相关性分析
   - 异常值检测

3. **特征工程**
   - 特征创建
   - 特征选择
   - 目标编码
   - 领域特征构建

4. **模型训练**
   - LightGBM
   - XGBoost
   - 随机森林

5. **模型优化**
   - 超参数调优
   - 交叉验证
   - 模型融合

6. **模型评估**
   - 性能指标
   - 混淆矩阵
   - ROC 曲线

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
# 克隆仓库
git clone <your-repo-url>
cd data

# 创建虚拟环境
conda create -n api_env python=3.11 -y
conda activate api_env

# 安装依赖
pip install openai python-dotenv requests pandas numpy matplotlib seaborn scikit-learn lightgbm
```

### 2️⃣ 配置 API Keys

创建 `.env` 文件：

```bash
# 复制示例配置
# 如果没有 .env.example，手动创建 .env 文件
```

`.env` 文件内容：
```env
# DeepSeek API（推荐）
DEEPSEEK_API_KEY=sk-your_deepseek_key_here

# GitHub API
GITHUB_TOKEN=ghp_your_github_token_here

# OpenAI API（可选）
OPENAI_API_KEY=sk-your_openai_key_here
```

### 3️⃣ 运行项目

#### 运行 DeepSeek 聊天助手
```bash
python hand_write_code.py
```

选择使用方式：
- `1` - 交互式聊天（推荐）⭐
- `2` - 单次对话测试
- `3` - 多轮对话测试

#### 运行 API 教程
```bash
# GitHub API 示例
python github_api_helper.py

# DeepSeek API 完整教程
python learn_openai.py

# API 调用基础教程
python learn_api.py
```

#### 运行数据分析项目
```bash
cd data_analysis_project
jupyter notebook notebooks/01_eda/01_data_loading_and_overview.ipynb
```

---

## 📚 学习路径

### 🌱 第一阶段：Python 基础（建议 1-2 周）

#### Step 1: 面向对象编程
**文件：** `test.py`

**学习内容：**
- ✅ 类（Class）和对象（Object）
- ✅ 构造函数 `__init__`
- ✅ self 的理解与使用
- ✅ 实例属性和方法
- ✅ 继承（Inheritance）
- ✅ 多态（Polymorphism）

**练习：**
```python
# 创建自己的类
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"我是 {self.name}，{self.age} 岁"

# 创建对象
student = Student("小明", 18)
print(student.introduce())
```

#### Step 2: API 调用入门
**文件：** `learn_api.py`

**学习内容：**
- ✅ HTTP 请求方法（GET/POST）
- ✅ requests 库使用
- ✅ JSON 数据处理
- ✅ 错误处理与重试
- ✅ 环境变量管理

**练习：**
- 调用 GitHub API 获取用户信息
- 搜索仓库
- 处理 API 响应

### 🚀 第二阶段：AI API 应用（建议 2-3 周）

#### Step 3: DeepSeek API 使用
**文件：** `learn_openai.py`

**学习内容：**
- ✅ OpenAI 兼容 API 调用
- ✅ Chat Completion 接口
- ✅ 流式响应处理
- ✅ 参数调优
- ✅ 系统提示词设计

#### Step 4: 完整项目开发
**文件：** `hand_write_code.py`

**学习内容：**
- ✅ 项目架构设计
- ✅ 类的设计与封装
- ✅ 用户交互设计
- ✅ 多线程编程（思考动画）
- ✅ 文件 I/O 操作
- ✅ 异常处理

**核心技能：**
- 独立开发完整应用
- 代码组织与模块化
- 用户体验优化

### 📊 第三阶段：数据分析实战（建议 4-6 周）

#### Step 5: 数据处理基础
**文件：** `learn_pandas.ipynb`

**学习内容：**
- ✅ Pandas 数据结构
- ✅ 数据读取与保存
- ✅ 数据清洗
- ✅ 数据转换
- ✅ 数据聚合

#### Step 6: 完整数据分析项目
**目录：** `data_analysis_project/`

**学习内容：**
- ✅ 探索性数据分析
- ✅ 特征工程技巧
- ✅ 机器学习建模
- ✅ 模型调优
- ✅ 结果评估与可视化

---

## 💡 核心知识点总结

### Python 基础
- ✅ 变量、数据类型、字符串操作
- ✅ 条件判断（if/elif/else）
- ✅ 循环（for/while）
- ✅ 函数定义与调用
- ✅ 列表、字典、集合
- ✅ 异常处理（try/except）

### 面向对象编程
- ✅ 类的定义（class）
- ✅ 构造函数（`__init__`，两个下划线！）
- ✅ 实例属性（self.xxx）
- ✅ 实例方法（def method(self)）
- ✅ self 的理解
- ✅ 继承与多态

### API 调用
- ✅ REST API 原理
- ✅ requests 库使用
- ✅ GET/POST 请求
- ✅ 请求参数与请求头
- ✅ JSON 数据处理
- ✅ 流式响应处理
- ✅ 错误处理与重试

### 进阶技能
- ✅ 多线程编程（threading）
- ✅ 文件 I/O 操作
- ✅ 环境变量管理（dotenv）
- ✅ 模块导入与使用
- ✅ 类型注解（typing）
- ✅ 上下文管理器（with）

### 数据分析
- ✅ Pandas 数据处理
- ✅ NumPy 数值计算
- ✅ Matplotlib/Seaborn 可视化
- ✅ 特征工程
- ✅ 机器学习模型（LightGBM/XGBoost）
- ✅ 模型评估与优化

---

## 🎓 学习成果

通过完成这个仓库的学习，你将能够：

### 1. 编程能力
- ✅ 独立编写 Python 程序
- ✅ 理解面向对象编程
- ✅ 设计类和对象
- ✅ 处理复杂的数据结构

### 2. API 调用能力
- ✅ 调用各种 REST API
- ✅ 处理 API 响应
- ✅ 实现错误处理
- ✅ 管理 API 密钥

### 3. AI 应用开发
- ✅ 集成 AI 模型 API
- ✅ 构建聊天应用
- ✅ 优化用户体验
- ✅ 实现高级功能（流式输出、动画）

### 4. 数据分析能力
- ✅ 完整的数据分析流程
- ✅ 特征工程技巧
- ✅ 机器学习建模
- ✅ 模型优化与评估

### 5. 项目开发能力
- ✅ 项目架构设计
- ✅ 代码组织与模块化
- ✅ 版本控制（Git）
- ✅ 文档编写

---

## 🔧 常见问题

### Q1: 如何获取 API Keys？

**DeepSeek API:**
1. 访问 https://platform.deepseek.com
2. 注册账号并登录
3. 进入 API Keys 页面生成密钥
4. 💰 新用户有免费额度！

**GitHub Token:**
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 选择权限（repo, user）并生成

### Q2: 程序运行时显示 "未找到 API Key"

**解决方法：**
1. ✅ 确认已创建 `.env` 文件
2. ✅ 检查 API Key 格式（无多余空格）
3. ✅ 确认 `.env` 文件在项目根目录
4. ✅ 检查是否运行了 `load_dotenv()`

### Q3: DeepSeek 聊天助手没有"正在思考"动画

**原因：**
- 思考动画仅在使用 `deepseek-reasoner` 模型时显示
- 如果使用 `deepseek-chat` 模型，会直接显示回复

**解决：**
- 将 `model='deepseek-chat'` 改为 `model='deepseek-reasoner'`

### Q4: 如何保存对话历史？

在交互式聊天中：
- 输入 `save` 或 `保存`
- 对话将自动保存到 `chat_history_YYYYMMDD_HHMMSS.json`

### Q5: 导入错误 "No module named 'xxx'"

**解决：**
```bash
# 激活虚拟环境
conda activate api_env

# 安装缺失的包
pip install xxx

# 或安装所有依赖
pip install -r requirements.txt
```

### Q6: `__init__` 拼写错误

**错误示例：**
```python
def ___init__(self):  # ❌ 三个下划线
```

**正确写法：**
```python
def __init__(self):   # ✅ 两个下划线
```

---

## 📖 学习资源

### 官方文档
- [Python 官方文档](https://docs.python.org/3/)
- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [GitHub API 文档](https://docs.github.com/en/rest)
- [Pandas 文档](https://pandas.pydata.org/docs/)
- [Scikit-learn 文档](https://scikit-learn.org/)

### 本仓库教程
- `TUTORIAL_GUIDE.md` - 教程指南
- `TUTORIAL_COMPLETE.md` - 完整教程
- `OUTLIER_HANDLING_GUIDE.md` - 异常值处理
- `data_analysis_pipeline.md` - 数据分析流程

### 推荐学习网站
- [Kaggle](https://www.kaggle.com/) - 数据科学竞赛平台
- [GitHub](https://github.com/) - 开源代码学习
- [Stack Overflow](https://stackoverflow.com/) - 问题解答

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 如何贡献
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献方向
- 💡 改进文档
- 🐛 修复 Bug
- ✨ 添加新功能
- 📝 完善示例代码
- 🌍 翻译文档

---

## 📝 更新日志

### v1.0.0 (2024-10-19)
- ✅ 完成 DeepSeek 智能聊天助手
- ✅ 添加思考动画效果（多线程实现）
- ✅ 实现对话历史管理（保存/加载）
- ✅ 完成 API 调用教程合集
- ✅ 完成数据分析项目结构
- ✅ 完成 Python OOP 完整教程
- ✅ 完成 Kaggle 竞赛实战项目

---

## 📄 License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 💬 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 📧 Email: zhanbing2025@gmail.com
- 🐙 GitHub Issues: 在本仓库提交 Issue

---

## ⭐ Star History

如果这个项目对你有帮助，请给个 Star ⭐️ 支持一下！

你的 Star 是我持续更新的动力！

---

## 🙏 致谢

感谢以下项目和服务的支持：

- [DeepSeek](https://www.deepseek.com/) - 提供强大且经济的 AI 能力
- [OpenAI](https://openai.com/) - API 设计参考
- [GitHub](https://github.com/) - 代码托管平台
- [Python](https://www.python.org/) - 优秀的编程语言
- [Kaggle](https://www.kaggle.com/) - 数据科学学习平台
- 所有开源贡献者 - 感谢你们的无私奉献

---

## 🎯 下一步计划

- [ ] 添加更多 AI 应用示例
- [ ] 完善数据可视化教程
- [ ] 添加 Web 应用开发教程（Flask/FastAPI）
- [ ] 增加更多 Kaggle 竞赛案例
- [ ] 制作视频教程
- [ ] 翻译英文文档

---

<div align="center">

**🚀 Happy Coding! 让我们一起成长！ 🚀**

Made with ❤️ by a Python Learner

*"学习是一场永无止境的旅程，每一行代码都是进步的印记。"*

</div>
