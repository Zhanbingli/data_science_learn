"""
DeepSeek API 完全教程 - 打造智能聊天助手
作者：Python 专家
目标：从零开始学习 AI API 调用和 Python 编程
"""

# ============================================================================
# 第一部分：DeepSeek API 简介
# ============================================================================

"""
【DeepSeek 是什么？】
DeepSeek 是一个强大的中文 AI 模型，特点：
1. 性价比高：比 OpenAI 便宜很多
2. 中文友好：对中文理解更好
3. 功能强大：支持对话、代码生成、文本分析等

【API 对比】
┌─────────────┬──────────────┬──────────────┬─────────────┐
│   模型      │   价格       │   中文能力   │   推荐度    │
├─────────────┼──────────────┼──────────────┼─────────────┤
│ DeepSeek    │   💰 便宜    │   ⭐⭐⭐⭐⭐   │   ✅ 推荐   │
│ GPT-4       │   💰💰💰 贵  │   ⭐⭐⭐⭐    │   ⚠️ 贵    │
│ GPT-3.5     │   💰💰 中等  │   ⭐⭐⭐     │   ✅ 推荐   │
└─────────────┴──────────────┴──────────────┴─────────────┘

【准备工作】
1. 注册 DeepSeek：https://platform.deepseek.com
2. 获取 API Key：https://platform.deepseek.com/api_keys
3. 安装依赖：pip install openai python-dotenv
4. 设置环境变量：在 .env 文件中添加 DEEPSEEK_API_KEY
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

# 加载环境变量
load_dotenv()


# ============================================================================
# 第二部分：基础 API 调用
# ============================================================================

def example_1_simple_chat():
    """示例1：最简单的对话"""
    print("\n" + "=" * 70)
    print("【示例1：简单对话】")
    print("=" * 70)

    # 获取 API Key
    api_key = os.environ.get('DEEPSEEK_API_KEY')

    if not api_key:
        print("❌ 未找到 DEEPSEEK_API_KEY")
        print("请在 .env 文件中设置：DEEPSEEK_API_KEY=你的密钥")
        return None

    # 创建客户端（注意：使用 DeepSeek 的 base_url）
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # 发送请求
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个友好的AI助手"},
            {"role": "user", "content": "你好！请介绍一下你自己"}
        ],
        stream=False  # 非流式输出
    )

    # 获取回复
    reply = response.choices[0].message.content

    print(f"用户：你好！请介绍一下你自己")
    print(f"AI：{reply}")

    return reply


def example_2_with_parameters():
    """示例2：使用不同的参数"""
    print("\n" + "=" * 70)
    print("【示例2：调整参数】")
    print("=" * 70)

    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ 请先设置 DEEPSEEK_API_KEY")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 测试不同的 temperature 参数
    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "用一句话描述春天"}
            ],
            temperature=temp,  # 温度：0=确定性，2=随机性
            max_tokens=50      # 最大令牌数
        )

        print(f"回复：{response.choices[0].message.content}")


def example_3_streaming():
    """示例3：流式输出（像 ChatGPT 一样逐字显示）"""
    print("\n" + "=" * 70)
    print("【示例3：流式输出】")
    print("=" * 70)

    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ 请先设置 DEEPSEEK_API_KEY")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("用户：请讲一个关于AI的笑话")
    print("AI：", end="", flush=True)

    # 流式请求
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "请讲一个关于AI的笑话"}
        ],
        stream=True  # 启用流式输出
    )

    # 逐字打印
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


# ============================================================================
# 第三部分：打造完整的聊天助手
# ============================================================================

class DeepSeekChatbot:
    """DeepSeek 聊天助手类"""

    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        初始化聊天机器人

        Parameters:
        -----------
        api_key : str, optional
            DeepSeek API Key，如果不提供则从环境变量读取
        system_prompt : str, optional
            系统提示词，定义 AI 的角色和行为
        """
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')

        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        # 创建客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        # 对话历史
        self.conversation_history = []

        # 系统提示词
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })

    def chat(self, user_message: str, stream: bool = False) -> str:
        """
        发送消息并获取回复

        Parameters:
        -----------
        user_message : str
            用户的消息
        stream : bool
            是否使用流式输出

        Returns:
        --------
        str
            AI 的回复
        """
        # 添加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            if stream:
                # 流式输出
                return self._chat_stream()
            else:
                # 普通输出
                return self._chat_normal()

        except Exception as e:
            error_msg = f"❌ 调用 API 失败：{e}"
            print(error_msg)
            return error_msg

    def _chat_normal(self) -> str:
        """普通对话模式"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.conversation_history,
            temperature=0.7
        )

        # 获取回复
        reply = response.choices[0].message.content

        # 添加到历史
        self.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def _chat_stream(self) -> str:
        """流式对话模式"""
        stream = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.conversation_history,
            temperature=0.7,
            stream=True
        )

        # 收集完整回复
        full_reply = ""

        print("AI: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_reply += content

        print()  # 换行

        # 添加到历史
        self.conversation_history.append({
            "role": "assistant",
            "content": full_reply
        })

        return full_reply

    def reset(self):
        """重置对话历史"""
        # 保留系统提示词
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            system_msg = self.conversation_history[0]
            self.conversation_history = [system_msg]
        else:
            self.conversation_history = []

        print("✅ 对话历史已重置")

    def show_history(self):
        """显示对话历史"""
        print("\n" + "=" * 70)
        print("【对话历史】")
        print("=" * 70)

        for i, msg in enumerate(self.conversation_history, 1):
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                print(f"{i}. [系统] {content}\n")
            elif role == "user":
                print(f"{i}. [用户] {content}\n")
            elif role == "assistant":
                print(f"{i}. [AI] {content}\n")

    def save_history(self, filename: str = None):
        """保存对话历史到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)

        print(f"✅ 对话历史已保存到：{filename}")

    def load_history(self, filename: str):
        """从文件加载对话历史"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)

        print(f"✅ 已加载对话历史：{filename}")


# ============================================================================
# 第四部分：交互式聊天界面
# ============================================================================

def interactive_chat():
    """交互式聊天界面"""
    print("\n" + "=" * 70)
    print("🤖 DeepSeek 智能聊天助手")
    print("=" * 70)
    print("\n命令说明：")
    print("  - 直接输入文字进行对话")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'reset' 重置对话")
    print("  - 输入 'history' 查看历史")
    print("  - 输入 'save' 保存对话")
    print("=" * 70)

    # 创建聊天机器人
    try:
        bot = DeepSeekChatbot(
            system_prompt="你是一个友好、专业的 AI 助手，擅长用简单的语言解释复杂的概念。"
        )
    except ValueError as e:
        print(f"❌ {e}")
        return

    # 开始对话循环
    while True:
        # 获取用户输入
        user_input = input("\n你: ").strip()

        # 处理命令
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\n👋 再见！感谢使用 DeepSeek 聊天助手！")
            break

        elif user_input.lower() in ['reset', '重置']:
            bot.reset()
            continue

        elif user_input.lower() in ['history', '历史']:
            bot.show_history()
            continue

        elif user_input.lower() in ['save', '保存']:
            bot.save_history()
            continue

        elif not user_input:
            print("⚠️ 请输入内容")
            continue

        # 发送消息并获取回复（使用流式输出）
        bot.chat(user_input, stream=True)


# ============================================================================
# 第五部分：实用功能示例
# ============================================================================

def demo_code_generation():
    """示例：代码生成"""
    print("\n" + "=" * 70)
    print("【功能演示：代码生成】")
    print("=" * 70)

    bot = DeepSeekChatbot(
        system_prompt="你是一个 Python 专家，擅长编写清晰、高效的代码。"
    )

    # 请求生成代码
    question = "写一个 Python 函数，计算斐波那契数列的第 n 项"
    print(f"\n用户：{question}\n")

    reply = bot.chat(question, stream=True)


def demo_text_analysis():
    """示例：文本分析"""
    print("\n" + "=" * 70)
    print("【功能演示：文本分析】")
    print("=" * 70)

    bot = DeepSeekChatbot(
        system_prompt="你是一个文本分析专家。"
    )

    text = """
    人工智能技术正在快速发展，深度学习、自然语言处理、计算机视觉等领域
    取得了突破性进展。AI 已经广泛应用于医疗、金融、教育等多个领域，
    为人类社会带来了巨大变革。
    """

    question = f"请分析以下文本的主题和情感：\n{text}"
    print(f"用户：{question}\n")

    reply = bot.chat(question, stream=True)


def demo_translation():
    """示例：翻译"""
    print("\n" + "=" * 70)
    print("【功能演示：翻译】")
    print("=" * 70)

    bot = DeepSeekChatbot(
        system_prompt="你是一个专业的翻译专家。"
    )

    text = "Machine learning is a subset of artificial intelligence."
    question = f"请将以下英文翻译成中文：{text}"

    print(f"用户：{question}\n")
    reply = bot.chat(question, stream=True)


def demo_learning_assistant():
    """示例：学习助手"""
    print("\n" + "=" * 70)
    print("【功能演示：Python 学习助手】")
    print("=" * 70)

    bot = DeepSeekChatbot(
        system_prompt="你是一个耐心的 Python 编程导师，擅长用简单的例子解释概念。"
    )

    questions = [
        "什么是列表推导式？",
        "能给我一个例子吗？",
        "它和普通的 for 循环有什么区别？"
    ]

    for q in questions:
        print(f"\n用户：{q}\n")
        bot.chat(q, stream=True)
        print("\n" + "-" * 70)


# ============================================================================
# 第六部分：高级功能 - 自定义角色
# ============================================================================

class CustomAssistant:
    """自定义助手工厂"""

    @staticmethod
    def create_coding_tutor():
        """创建编程导师"""
        return DeepSeekChatbot(
            system_prompt="""你是一个资深的 Python 编程导师，特点：
1. 用简单的语言解释复杂概念
2. 提供实际的代码示例
3. 指出常见错误和最佳实践
4. 鼓励学生动手实践"""
        )

    @staticmethod
    def create_data_analyst():
        """创建数据分析师"""
        return DeepSeekChatbot(
            system_prompt="""你是一个专业的数据分析师，擅长：
1. 数据清洗和预处理
2. 统计分析和可视化
3. 机器学习模型选择
4. 结果解读和建议"""
        )

    @staticmethod
    def create_translator():
        """创建翻译助手"""
        return DeepSeekChatbot(
            system_prompt="""你是一个专业翻译，特点：
1. 准确传达原文含义
2. 保持语言流畅自然
3. 考虑文化差异
4. 提供多种翻译选项"""
        )

    @staticmethod
    def create_writer():
        """创建写作助手"""
        return DeepSeekChatbot(
            system_prompt="""你是一个专业的写作助手，擅长：
1. 文章结构规划
2. 内容创作和润色
3. 语言优化
4. SEO 优化建议"""
        )


def demo_custom_assistants():
    """演示自定义助手"""
    print("\n" + "=" * 70)
    print("【自定义助手演示】")
    print("=" * 70)

    # 1. 编程导师
    print("\n1️⃣  编程导师")
    print("-" * 70)
    tutor = CustomAssistant.create_coding_tutor()
    tutor.chat("请解释 Python 的装饰器", stream=True)

    # 2. 数据分析师
    print("\n2️⃣  数据分析师")
    print("-" * 70)
    analyst = CustomAssistant.create_data_analyst()
    analyst.chat("我有一个用户行为数据集，应该如何分析？", stream=True)


# ============================================================================
# 第七部分：错误处理和最佳实践
# ============================================================================

def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 70)
    print("【错误处理示例】")
    print("=" * 70)

    # 测试无效的 API Key
    try:
        bot = DeepSeekChatbot(api_key="invalid_key")
        bot.chat("你好")
    except Exception as e:
        print(f"❌ 捕获错误：{e}")

    # 测试网络超时
    print("\n提示：生产环境中应该添加重试机制和超时设置")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("🚀 DeepSeek API 完全教程")
    print("=" * 70)

    # 检查 API Key
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("\n⚠️  未检测到 DEEPSEEK_API_KEY")
        print("\n请按以下步骤设置：")
        print("1. 在 .env 文件中添加：DEEPSEEK_API_KEY=你的密钥")
        print("2. 或运行：export DEEPSEEK_API_KEY='你的密钥'")
        print("3. 获取密钥：https://platform.deepseek.com/api_keys")
        print("\n" + "=" * 70)
    else:
        print("✅ 已检测到 DEEPSEEK_API_KEY\n")

        # 运行示例（可以注释掉不需要的）

        # 基础示例
        example_1_simple_chat()
        example_2_with_parameters()
        example_3_streaming()

        # 功能演示
        demo_code_generation()
        demo_text_analysis()
        demo_translation()
        demo_learning_assistant()

        # 自定义助手
        demo_custom_assistants()

        # 交互式聊天（注释掉，因为会阻塞程序）
        # print("\n" + "=" * 70)
        # print("准备启动交互式聊天...")
        # print("=" * 70)
        # interactive_chat()

    print("\n" + "=" * 70)
    print("🎉 教程完成！")
    print("=" * 70)
    print("\n💡 提示：")
    print("1. 取消注释 interactive_chat() 来启动交互式聊天")
    print("2. 修改系统提示词来定制 AI 的行为")
    print("3. 查看保存的对话历史文件")
    print("=" * 70)
