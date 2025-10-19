import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Optional

load_dotenv()

# def example1():
#     api_key = os.environ.get('DEEPSEEK_API_KEY')

#     if not api_key:
#         print('no api_key')
#         return

#     client = OpenAI(
#         api_key=api_key,
#         base_url="https://api.deepseek.com"
#     )
#     temperatures = [0.5, 1, 1.5]

#     for temp in temperatures:
#         print(f"\n--- Temperature = {temp} ---")
#         response = client.chat.completions.create(
#             model= 'deepseek-chat',
#             messages=[
#                 {"role":"system", "content":"You are a helpful assistant"},
#                 {"role":"user", "content":"hello! please you help me improve my English"}
#             ],
#             temperature=temp,
#             stream=True,
#             max_tokens=50
#         )
#     for chunk in response:
#         if chunk.choices[0].delta.content is not None:
#             print(chunk.choices[0].delta.content, end='', flush=True)



class DeepSeekChatbot:
    """deepseek chatbot"""
    def __init__(self, api_key:Optional[str]=None, system_prompt:Optional[str]=None):
        """init chatbot"""
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError('please set your api-key')
        # create client
        self.client= OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        # conversation history
        self.conversation_history = []

        # system prompt
        if system_prompt:
            self.conversation_history.append({
                'role': 'system',
                'content': system_prompt
            })

    def chat(self, user_message: str, stream: bool = False) -> str:
        """message and feedback"""
        # add conversation history
        self.conversation_history.append({
            "role":"user",
            "content": user_message
        })

        try:
            if stream:
                return self._chat_stream()
            else:
                return self._chat_normal()

        except Exception as e:
            erro_msg = f"call api faire {e}"
            print(erro_msg)
            return erro_msg

    def _chat_normal(self) -> str:
        """chat normal"""
        response = self.client.chat.completions.create(
            model='deepseek-reasoner',
            messages=self.conversation_history,
            temperature= 0.7
        )

        # reply
        reply = response.choices[0].message.content

        # add history
        self.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def _chat_stream(self) -> str:
        """chat stream with thinking animation"""
        import threading
        import time

        # 思考动画标志
        thinking = {'active': True}

        def show_thinking_animation():
            """显示思考动画"""
            animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            idx = 0
            while thinking['active']:
                print(f"\r💭 AI 正在思考 {animation[idx % len(animation)]}", end="", flush=True)
                idx += 1
                time.sleep(0.1)
            print("\r" + " " * 50 + "\r", end="", flush=True)  # 清除动画

        # 启动思考动画线程
        animation_thread = threading.Thread(target=show_thinking_animation)
        animation_thread.daemon = True
        animation_thread.start()

        try:
            stream = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=self.conversation_history,
                temperature=0.7,
                stream=True
            )

            full_reply = ""
            first_content = True

            for chunk in stream:
                # 处理回复内容
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content

                    # 第一次收到回复时停止动画
                    if first_content:
                        thinking['active'] = False
                        time.sleep(0.2)  # 等待动画线程结束
                        print("✨ AI 回复:", flush=True)
                        first_content = False

                    print(content, end="", flush=True)
                    full_reply += content

            # 停止动画（以防万一）
            thinking['active'] = False

            print("\n")

        except Exception as e:
            thinking['active'] = False
            raise e

        # add history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_reply
        })

        return full_reply

    def reset(self):
        if self.conversation_history and self.conversation_history[0]['role'] == "system":
            system_msg = self.conversation_history[0]
            self.conversation_history = [system_msg]
        else:
            self.conversation_history = []

        print("reset conversation")

    def show_history(self):
        print("conversation history")
        print("=" * 70)

        for i, msg in enumerate(self.conversation_history, 1):
            role = msg['role']
            content = msg["content"]

            if role == "system":
                print(f"{i}. [system] {content}\n")
            elif role == "user":
                print(f"{i}. [user] {content}\n")
            elif role == "assistant":
                print(f"{i}. [ai] {content}\n")

    def save_history(self, filename: str = None):
        """save history"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)

        print(f"load history{filename}")


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

    try:
        bot = DeepSeekChatbot(
            system_prompt="你是一个友好、专业的AI助手，擅长用简单的语言解释复杂的概念。"
        )
    except ValueError as e:
        print(f'❌ {e}')
        return

    # 对话循环
    while True:
        user_input = input("\n你: ").strip()

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\n👋 再见！感谢使用 DeepSeek 聊天助手！")
            break
        elif user_input.lower() in ['reset', '重置']:
            bot.reset()
            continue
        elif user_input.lower() in ["history", '历史']:
            bot.show_history()
            continue
        elif user_input.lower() in ['save', '保存']:
            bot.save_history()
            continue
        elif not user_input:
            print("⚠️ 请输入内容")
            continue

        # 发送消息（流式输出）
        bot.chat(user_input, stream=True)


# ============================================================================
# 主程序 - 选择不同的使用方式
# ============================================================================

if __name__ == "__main__":
    import sys

    print("🚀 DeepSeek 聊天机器人")
    print("=" * 70)

    # 检查 API Key
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("\n⚠️ 未检测到 DEEPSEEK_API_KEY")
        print("\n请按以下步骤设置：")
        print("1. 在 .env 文件中添加：DEEPSEEK_API_KEY=你的密钥")
        print("2. 或运行：export DEEPSEEK_API_KEY='你的密钥'")
        print("3. 获取密钥：https://platform.deepseek.com/api_keys")
        print("\n" + "=" * 70)
        sys.exit(1)

    print("\n选择使用方式：")
    print("1. 交互式聊天（推荐）")
    print("2. 单次对话测试")
    print("3. 多轮对话测试")

    choice = input("\n请选择 (1/2/3，直接回车默认选1): ").strip()

    if choice == '' or choice == '1':
        # 方式1：交互式聊天
        interactive_chat()

    elif choice == '2':
        # 方式2：单次对话测试
        print("\n【单次对话测试】")
        print("-" * 70)

        bot = DeepSeekChatbot()
        response = bot.chat('你好！请介绍一下你自己')
        print(f"\nAI 回复：{response}")

    elif choice == '3':
        # 方式3：多轮对话测试
        print("\n【多轮对话测试】")
        print("-" * 70)

        bot = DeepSeekChatbot(
            system_prompt="你是一个 Python 编程专家。"
        )

        questions = [
            "什么是 Python？",
            "它有什么优点？",
            "请给我一个简单的代码示例"
        ]

        for i, q in enumerate(questions, 1):
            print(f"\n问题 {i}: {q}")
            print("AI: ", end="")
            bot.chat(q, stream=True)
            print("-" * 70)

        # 显示历史
        print("\n【对话历史】")
        bot.show_history()

    else:
        print("无效选择！")

    print("\n" + "=" * 70)
    print("✅ 程序结束")
    print("=" * 70)

