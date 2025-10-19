"""
API 调用完全指南 - 从入门到精通
重点：OpenAI API 使用教程
作者：Python 专家
目标：让小白轻松掌握 API 调用技巧
"""

# ============================================================================
# 第一部分：什么是 API？
# ============================================================================

"""
【API 是什么？】
API = Application Programming Interface（应用程序编程接口）

【通俗理解】
API 就像餐厅的菜单：
- 你（客户端）：看菜单点菜
- 服务员（API）：把你的需求传递给厨房
- 厨房（服务器）：做好菜
- 服务员（API）：把菜端给你

【为什么需要 API？】
1. 数据获取：获取天气、新闻、股票数据等
2. 服务调用：翻译、图像识别、AI对话
3. 系统集成：连接不同的软件系统
4. 功能扩展：使用别人的功能而不用自己开发

【常见的 API 类型】
1. REST API（最常用）：通过 HTTP 请求获取数据
2. GraphQL API：更灵活的数据查询
3. WebSocket API：实时双向通信
"""


# ============================================================================
# 第二部分：HTTP 基础 - API 调用的基石
# ============================================================================

"""
【HTTP 请求方法】
- GET：获取数据（查询）
- POST：提交数据（创建）
- PUT：更新数据（完整更新）
- PATCH：部分更新数据
- DELETE：删除数据

【HTTP 状态码】
- 200：成功
- 201：创建成功
- 400：请求错误
- 401：未授权（需要登录）
- 403：禁止访问
- 404：资源不存在
- 429：请求过多（限流）
- 500：服务器错误
"""

import requests
import json
import os
from typing import Dict, List, Optional
import time


# ============================================================================
# 第三部分：基础 API 调用示例
# ============================================================================

def example_1_simple_get():
    """示例1：最简单的 GET 请求"""
    print("\n" + "=" * 70)
    print("【示例1：简单的 GET 请求】")
    print("=" * 70)

    # 调用一个公开的 API（无需认证）
    url = "https://api.github.com/users/torvalds"

    # 发送 GET 请求
    response = requests.get(url)

    # 检查状态码
    print(f"状态码: {response.status_code}")

    # 解析 JSON 响应
    data = response.json()

    # 打印结果
    print(f"用户名: {data['login']}")
    print(f"真实姓名: {data['name']}")
    print(f"粉丝数: {data['followers']}")
    print(f"仓库数: {data['public_repos']}")


def example_2_with_parameters():
    """示例2：带参数的 GET 请求"""
    print("\n" + "=" * 70)
    print("【示例2：带参数的 GET 请求】")
    print("=" * 70)

    # GitHub 搜索 API
    url = "https://api.github.com/search/repositories"

    # 查询参数
    params = {
        'q': 'python machine learning',  # 搜索关键词
        'sort': 'stars',                  # 按星标排序
        'order': 'desc',                  # 降序
        'per_page': 5                     # 每页5个结果
    }

    # 发送请求
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print(f"找到 {data['total_count']} 个仓库")
        print("\n前5名：")
        for i, repo in enumerate(data['items'], 1):
            print(f"{i}. {repo['name']} - ⭐ {repo['stargazers_count']}")
    else:
        print(f"请求失败：{response.status_code}")


def example_3_with_headers():
    """示例3：带请求头的 API 调用"""
    print("\n" + "=" * 70)
    print("【示例3：带请求头的 API 调用】")
    print("=" * 70)

    url = "https://api.github.com/user"

    # 请求头（Headers）- 通常用于认证和指定格式
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Python-Tutorial'
    }

    response = requests.get(url, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应头: {dict(response.headers)}")


def example_4_post_request():
    """示例4：POST 请求 - 提交数据"""
    print("\n" + "=" * 70)
    print("【示例4：POST 请求示例】")
    print("=" * 70)

    # 使用测试 API
    url = "https://httpbin.org/post"

    # 要提交的数据
    data = {
        'name': '张三',
        'age': 25,
        'skills': ['Python', 'Machine Learning', 'API']
    }

    # 发送 POST 请求
    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        print("服务器收到的数据：")
        print(json.dumps(result['json'], indent=2, ensure_ascii=False))


# ============================================================================
# 第四部分：错误处理和最佳实践
# ============================================================================

def safe_api_call(url: str, max_retries: int = 3) -> Optional[Dict]:
    """安全的 API 调用（带重试和错误处理）"""

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)

            # 检查状态码
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # 限流，等待后重试
                wait_time = 2 ** attempt  # 指数退避
                print(f"请求过多，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            elif response.status_code == 404:
                print("资源不存在")
                return None
            else:
                print(f"请求失败：{response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print(f"请求超时，第 {attempt + 1} 次重试...")
        except requests.exceptions.ConnectionError:
            print(f"连接错误，第 {attempt + 1} 次重试...")
        except Exception as e:
            print(f"未知错误：{e}")
            return None

    print("达到最大重试次数")
    return None


# ============================================================================
# 第五部分：OpenAI API 详细教程 ⭐⭐⭐
# ============================================================================

"""
【OpenAI API 简介】
OpenAI 提供强大的 AI 模型接口，包括：
1. ChatGPT（GPT-4, GPT-3.5）：对话和文本生成
2. DALL-E：图像生成
3. Whisper：语音识别
4. Embeddings：文本向量化

【准备工作】
1. 注册 OpenAI 账号：https://platform.openai.com
2. 获取 API Key：https://platform.openai.com/api-keys
3. 安装库：pip install openai
4. 设置环境变量：export OPENAI_API_KEY="你的密钥"
"""


class OpenAIHelper:
    """OpenAI API 辅助类"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 OpenAI 客户端

        Parameters:
        -----------
        api_key : str, optional
            API密钥，如果不提供则从环境变量读取
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')

        if not self.api_key:
            print("⚠️  警告：未设置 API Key")
            print("请设置环境变量：export OPENAI_API_KEY='你的密钥'")

        # 导入 OpenAI 库
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            print("⚠️  请先安装 OpenAI 库：pip install openai")
            self.client = None

    def chat_completion(self,
                       messages: List[Dict],
                       model: str = "gpt-3.5-turbo",
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> Optional[str]:
        """
        调用 ChatGPT API

        Parameters:
        -----------
        messages : List[Dict]
            对话消息列表
        model : str
            模型名称（gpt-4, gpt-3.5-turbo等）
        temperature : float
            温度参数（0-2），越高越随机
        max_tokens : int, optional
            最大令牌数

        Returns:
        --------
        str
            模型的回复
        """
        if not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"API 调用失败：{e}")
            return None

    def simple_chat(self, user_message: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        简化的聊天接口

        Parameters:
        -----------
        user_message : str
            用户消息
        system_prompt : str, optional
            系统提示词

        Returns:
        --------
        str
            AI 回复
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_message})

        return self.chat_completion(messages)

    def generate_image(self,
                      prompt: str,
                      size: str = "1024x1024",
                      n: int = 1) -> Optional[List[str]]:
        """
        使用 DALL-E 生成图像

        Parameters:
        -----------
        prompt : str
            图像描述
        size : str
            图像尺寸（256x256, 512x512, 1024x1024）
        n : int
            生成图像数量

        Returns:
        --------
        List[str]
            图像 URL 列表
        """
        if not self.client:
            return None

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                n=n
            )

            return [img.url for img in response.data]

        except Exception as e:
            print(f"图像生成失败：{e}")
            return None

    def create_embedding(self, text: str, model: str = "text-embedding-ada-002") -> Optional[List[float]]:
        """
        创建文本嵌入向量

        Parameters:
        -----------
        text : str
            要嵌入的文本
        model : str
            嵌入模型名称

        Returns:
        --------
        List[float]
            嵌入向量
        """
        if not self.client:
            return None

        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            print(f"嵌入创建失败：{e}")
            return None


def demo_openai_basic():
    """OpenAI API 基础示例"""
    print("\n" + "=" * 70)
    print("【OpenAI API 基础使用】")
    print("=" * 70)

    # 创建辅助对象
    ai = OpenAIHelper()

    if not ai.client:
        print("请先设置 API Key 再运行此示例")
        return

    # 示例1：简单对话
    print("\n1️⃣  简单对话示例")
    print("-" * 70)
    response = ai.simple_chat("用一句话解释什么是机器学习")
    print(f"AI: {response}")

    # 示例2：带系统提示词的对话
    print("\n2️⃣  角色扮演示例")
    print("-" * 70)
    response = ai.simple_chat(
        user_message="你好！",
        system_prompt="你是一个友好的Python编程助手，擅长用简单的语言解释复杂概念。"
    )
    print(f"AI: {response}")

    # 示例3：多轮对话
    print("\n3️⃣  多轮对话示例")
    print("-" * 70)
    messages = [
        {"role": "system", "content": "你是一个数学老师"},
        {"role": "user", "content": "什么是质数？"},
        {"role": "assistant", "content": "质数是只能被1和自己整除的大于1的自然数。"},
        {"role": "user", "content": "请给我5个质数的例子"}
    ]
    response = ai.chat_completion(messages)
    print(f"AI: {response}")


def demo_openai_advanced():
    """OpenAI API 高级示例"""
    print("\n" + "=" * 70)
    print("【OpenAI API 高级用法】")
    print("=" * 70)

    ai = OpenAIHelper()

    if not ai.client:
        print("请先设置 API Key 再运行此示例")
        return

    # 示例1：代码生成
    print("\n1️⃣  代码生成")
    print("-" * 70)
    response = ai.simple_chat(
        user_message="用Python写一个函数，判断一个数是否是质数",
        system_prompt="你是一个Python专家，请提供简洁、高效的代码，并添加注释。"
    )
    print(f"生成的代码：\n{response}")

    # 示例2：文本总结
    print("\n2️⃣  文本总结")
    print("-" * 70)
    long_text = """
    机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进，
    而无需明确编程。机器学习算法通过识别模式和做出决策，
    在各种应用中发挥作用，从推荐系统到自动驾驶汽车。
    主要的机器学习类型包括监督学习、无监督学习和强化学习。
    """
    response = ai.simple_chat(f"请用一句话总结以下内容：{long_text}")
    print(f"总结：{response}")

    # 示例3：数据分析
    print("\n3️⃣  数据分析建议")
    print("-" * 70)
    response = ai.simple_chat(
        "我有一个包含用户年龄、性别、购买金额的数据集，我想预测用户的购买倾向，应该用什么机器学习模型？",
        system_prompt="你是一个数据科学专家，请提供专业的建议。"
    )
    print(f"建议：{response}")


# ============================================================================
# 第六部分：流式响应（Stream）
# ============================================================================

def demo_streaming():
    """流式响应示例 - 实时显示 AI 回复"""
    print("\n" + "=" * 70)
    print("【流式响应示例】")
    print("=" * 70)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        print("\n🤖 AI 正在思考...\n")

        # 流式调用
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "用50字介绍Python"}],
            stream=True  # 启用流式输出
        )

        print("AI: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"流式调用失败：{e}")


# ============================================================================
# 第七部分：实战案例 - 智能问答系统
# ============================================================================

class SmartQASystem:
    """智能问答系统"""

    def __init__(self):
        self.ai = OpenAIHelper()
        self.conversation_history = []

    def ask(self, question: str) -> str:
        """
        提问并获得答案

        Parameters:
        -----------
        question : str
            用户的问题

        Returns:
        --------
        str
            AI 的回答
        """
        # 添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        # 调用 API
        response = self.ai.chat_completion(self.conversation_history)

        # 保存 AI 回复
        if response:
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

        return response

    def reset(self):
        """重置对话历史"""
        self.conversation_history = []
        print("对话历史已清空")

    def show_history(self):
        """显示对话历史"""
        print("\n" + "=" * 70)
        print("对话历史")
        print("=" * 70)
        for msg in self.conversation_history:
            role = "用户" if msg["role"] == "user" else "AI"
            print(f"{role}: {msg['content']}\n")


def demo_qa_system():
    """问答系统演示"""
    print("\n" + "=" * 70)
    print("【智能问答系统演示】")
    print("=" * 70)

    qa = SmartQASystem()

    if not qa.ai.client:
        print("请先设置 API Key")
        return

    # 进行多轮对话
    questions = [
        "什么是深度学习？",
        "它和机器学习有什么区别？",
        "能举个实际应用的例子吗？"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n问题 {i}: {q}")
        answer = qa.ask(q)
        print(f"回答: {answer}")
        print("-" * 70)

    # 显示完整历史
    qa.show_history()


# ============================================================================
# 第八部分：API 调用最佳实践
# ============================================================================

"""
【API 调用最佳实践】

1. 安全性 🔒
   - 永远不要在代码中硬编码 API Key
   - 使用环境变量：os.environ.get('API_KEY')
   - 不要提交 .env 文件到 Git

2. 错误处理 ⚠️
   - 始终检查状态码
   - 使用 try-except 捕获异常
   - 实现重试机制

3. 性能优化 ⚡
   - 实现请求缓存
   - 使用连接池
   - 批量请求而非单个请求

4. 速率限制 ⏱️
   - 遵守 API 的速率限制
   - 实现指数退避策略
   - 使用队列管理请求

5. 成本控制 💰
   - 监控 API 使用量
   - 设置最大令牌数
   - 使用更便宜的模型（如 gpt-3.5-turbo）

6. 日志记录 📝
   - 记录所有 API 调用
   - 保存请求和响应
   - 便于调试和审计
"""


class RateLimiter:
    """速率限制器"""

    def __init__(self, max_calls: int, time_window: int):
        """
        初始化速率限制器

        Parameters:
        -----------
        max_calls : int
            时间窗口内最大调用次数
        time_window : int
            时间窗口（秒）
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def wait_if_needed(self):
        """如果需要，等待直到可以发送请求"""
        now = time.time()

        # 清理过期的调用记录
        self.calls = [call_time for call_time in self.calls
                     if now - call_time < self.time_window]

        # 检查是否超过限制
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                print(f"速率限制，等待 {sleep_time:.2f} 秒...")
                time.sleep(sleep_time)
                self.calls = []

        # 记录本次调用
        self.calls.append(time.time())


# ============================================================================
# 第九部分：完整实战项目 - AI 文章生成器
# ============================================================================

class AIArticleGenerator:
    """AI 文章生成器"""

    def __init__(self):
        self.ai = OpenAIHelper()
        self.rate_limiter = RateLimiter(max_calls=3, time_window=60)

    def generate_outline(self, topic: str) -> str:
        """生成文章大纲"""
        self.rate_limiter.wait_if_needed()

        prompt = f"请为主题「{topic}」生成一个详细的文章大纲，包括引言、3-5个主要部分和结论。"
        return self.ai.simple_chat(prompt, system_prompt="你是一个专业的内容策划师。")

    def generate_section(self, section_title: str) -> str:
        """生成文章段落"""
        self.rate_limiter.wait_if_needed()

        prompt = f"请为文章小节「{section_title}」写一段200字左右的内容。"
        return self.ai.simple_chat(prompt, system_prompt="你是一个专业的文章撰稿人。")

    def generate_full_article(self, topic: str) -> Dict:
        """生成完整文章"""
        print(f"\n📝 开始生成文章：{topic}")
        print("=" * 70)

        # 1. 生成大纲
        print("\n1️⃣  生成大纲...")
        outline = self.generate_outline(topic)
        print(f"大纲：\n{outline}\n")

        # 2. 生成引言
        print("2️⃣  生成引言...")
        intro = self.generate_section(f"{topic} - 引言")

        # 3. 生成结论
        print("3️⃣  生成结论...")
        conclusion = self.generate_section(f"{topic} - 结论")

        article = {
            'topic': topic,
            'outline': outline,
            'introduction': intro,
            'conclusion': conclusion
        }

        print("\n✅ 文章生成完成！")
        return article


# ============================================================================
# 第十部分：总结和练习
# ============================================================================

def print_summary():
    """打印教程总结"""
    print("\n" + "=" * 70)
    print("【API 调用总结】")
    print("=" * 70)

    summary = """
📚 你已经学会了：

1. API 基础概念
   ✓ 什么是 API
   ✓ HTTP 请求方法
   ✓ 状态码含义

2. 使用 requests 库
   ✓ GET/POST 请求
   ✓ 请求参数和请求头
   ✓ JSON 数据处理

3. OpenAI API 完整使用
   ✓ Chat Completion（对话）
   ✓ 流式响应
   ✓ 图像生成
   ✓ 文本嵌入

4. 最佳实践
   ✓ 错误处理和重试
   ✓ 速率限制
   ✓ 安全性考虑
   ✓ 成本控制

5. 实战项目
   ✓ 智能问答系统
   ✓ AI 文章生成器

🎯 练习建议：

1. 初级练习
   - 调用公开的 API（如 GitHub、豆瓣）获取数据
   - 实现一个天气查询程序
   - 创建一个简单的翻译工具

2. 中级练习
   - 构建一个新闻聚合器
   - 实现一个智能客服机器人
   - 创建一个代码审查助手

3. 高级练习
   - 构建 RAG（检索增强生成）系统
   - 实现多模态 AI 应用（文本+图像）
   - 创建一个 AI 驱动的数据分析平台

💡 重要提醒：
- 妥善保管 API Key
- 监控使用量和成本
- 遵守服务条款
- 实现合理的错误处理
    """

    print(summary)


def print_useful_apis():
    """打印常用 API 列表"""
    print("\n" + "=" * 70)
    print("【常用免费 API 推荐】")
    print("=" * 70)

    apis = """
🌐 公开数据 API：

1. GitHub API
   - 地址：https://api.github.com
   - 功能：仓库、用户、代码搜索
   - 无需认证（有限流）

2. JSONPlaceholder
   - 地址：https://jsonplaceholder.typicode.com
   - 功能：测试用假数据
   - 完全免费

3. OpenWeather
   - 地址：https://openweathermap.org/api
   - 功能：天气数据
   - 需要注册（免费额度）

4. NewsAPI
   - 地址：https://newsapi.org
   - 功能：新闻聚合
   - 需要 API Key（免费额度）

5. REST Countries
   - 地址：https://restcountries.com
   - 功能：国家信息
   - 完全免费

🤖 AI 相关 API：

1. OpenAI API
   - ChatGPT、DALL-E、Whisper
   - 按使用量付费

2. Hugging Face API
   - 各种开源 AI 模型
   - 免费额度

3. Google Cloud AI
   - 翻译、视觉、语音
   - 免费额度

📚 学术 API：

1. PubMed/NCBI
   - 地址：https://www.ncbi.nlm.nih.gov/home/develop/api/
   - 功能：医学文献检索
   - 免费

2. arXiv API
   - 地址：https://arxiv.org/help/api
   - 功能：学术论文检索
   - 免费
    """

    print(apis)


# ============================================================================
# 主程序 - 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("🚀 API 调用完全教程")
    print("=" * 70)

    # 基础示例
    print("\n【第一部分：基础 API 调用】")
    example_1_simple_get()
    example_2_with_parameters()
    example_3_with_headers()
    example_4_post_request()

    # OpenAI 示例（需要 API Key）
    print("\n\n【第二部分：OpenAI API 使用】")
    print("⚠️  以下示例需要设置 OPENAI_API_KEY 环境变量")
    print("如果没有 API Key，请访问：https://platform.openai.com/api-keys")

    if os.environ.get('OPENAI_API_KEY'):
        demo_openai_basic()
        demo_openai_advanced()
        demo_streaming()
        demo_qa_system()

        # 文章生成器示例
        print("\n\n【第三部分：实战项目】")
        generator = AIArticleGenerator()
        article = generator.generate_full_article("Python 机器学习入门")
        print("\n生成的文章：")
        print(json.dumps(article, indent=2, ensure_ascii=False))
    else:
        print("\n💡 提示：设置 API Key 后可运行完整示例")
        print("   export OPENAI_API_KEY='your-api-key-here'")

    # 打印总结
    print_summary()
    print_useful_apis()

    print("\n" + "=" * 70)
    print("🎉 恭喜！你已经掌握了 API 调用的核心技能！")
    print("=" * 70)
