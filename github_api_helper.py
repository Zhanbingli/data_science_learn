"""
GitHub API 辅助工具
提供简单易用的 GitHub API 调用接口
"""

from dotenv import load_dotenv
import os
import requests
from typing import Optional, Dict, List

# 加载环境变量
load_dotenv()


class GitHubAPI:
    """GitHub API 封装类"""

    def __init__(self, token: Optional[str] = None):
        """
        初始化 GitHub API 客户端

        Parameters:
        -----------
        token : str, optional
            GitHub Personal Access Token
            如果不提供，则从环境变量 GITHUB_TOKEN 读取
        """
        self.token = token or os.environ.get('GITHUB_TOKEN')

        if not self.token:
            print("⚠️ 警告：未设置 GitHub Token")
            print("方法1：设置环境变量")
            print("  export GITHUB_TOKEN='你的token'")
            print("\n方法2：创建 .env 文件")
            print("  复制 .env.example 为 .env，然后填写 token")
            print("\n获取 Token：https://github.com/settings/tokens")

        self.base_url = 'https://api.github.com'
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Python-GitHub-Client'
        }

        if self.token:
            self.headers['Authorization'] = f'Bearer {self.token}'

    def test_connection(self) -> bool:
        """测试 API 连接和认证"""
        print("=" * 70)
        print("【GitHub API 连接测试】")
        print("=" * 70)

        if not self.token:
            print("❌ 未设置 Token，无法测试认证")
            return False

        try:
            response = requests.get(f'{self.base_url}/user', headers=self.headers)

            if response.status_code == 200:
                user = response.json()
                print("✅ 连接成功！认证通过！")
                print(f"用户名：{user['login']}")
                print(f"姓名：{user.get('name', 'N/A')}")
                print(f"邮箱：{user.get('email', 'N/A')}")

                # 检查 token 权限
                scopes = response.headers.get('X-OAuth-Scopes', '')
                print(f"\nToken 权限：{scopes}")

                # 检查速率限制
                self.check_rate_limit()

                return True
            elif response.status_code == 401:
                print("❌ 认证失败：Token 无效或已过期")
                print("请检查 .env 文件中的 GITHUB_TOKEN")
                return False
            else:
                print(f"❌ 请求失败：{response.status_code}")
                return False

        except Exception as e:
            print(f"❌ 连接错误：{e}")
            return False

    def check_rate_limit(self):
        """检查 API 速率限制"""
        try:
            response = requests.get(f'{self.base_url}/rate_limit', headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                core = data['resources']['core']

                print("\nAPI 速率限制：")
                print(f"  总限制：{core['limit']} 次/小时")
                print(f"  已使用：{core['limit'] - core['remaining']} 次")
                print(f"  剩余：{core['remaining']} 次")

                # 如果有认证，限制是 5000/小时，否则是 60/小时
                if core['limit'] == 60:
                    print("\n⚠️ 当前使用未认证限制（60次/小时）")
                    print("   建议使用 Token 以获得 5000次/小时 的限制")

        except Exception as e:
            print(f"检查速率限制失败：{e}")

    def get_current_user(self) -> Optional[Dict]:
        """获取当前认证用户信息"""
        if not self.token:
            print("❌ 需要 Token 才能获取当前用户信息")
            return None

        try:
            response = requests.get(f'{self.base_url}/user', headers=self.headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取用户信息失败：{response.status_code}")
                return None

        except Exception as e:
            print(f"错误：{e}")
            return None

    def get_user(self, username: str) -> Optional[Dict]:
        """获取指定用户信息（无需认证）"""
        try:
            response = requests.get(
                f'{self.base_url}/users/{username}',
                headers=self.headers
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取用户 {username} 信息失败：{response.status_code}")
                return None

        except Exception as e:
            print(f"错误：{e}")
            return None

    def search_repositories(self, query: str, sort: str = 'stars',
                          per_page: int = 5) -> Optional[Dict]:
        """搜索仓库"""
        params = {
            'q': query,
            'sort': sort,
            'order': 'desc',
            'per_page': per_page
        }

        try:
            response = requests.get(
                f'{self.base_url}/search/repositories',
                headers=self.headers,
                params=params
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"搜索失败：{response.status_code}")
                return None

        except Exception as e:
            print(f"错误：{e}")
            return None


def quick_test():
    """快速测试函数"""
    print("🚀 GitHub API 快速测试\n")

    # 创建 API 客户端
    github = GitHubAPI()

    # 测试连接
    if github.test_connection():
        print("\n" + "=" * 70)
        print("【获取用户信息】")
        print("=" * 70)

        # 获取 Linus Torvalds 的信息
        user = github.get_user('torvalds')
        if user:
            print(f"用户名：{user['login']}")
            print(f"姓名：{user['name']}")
            print(f"粉丝数：{user['followers']}")
            print(f"仓库数：{user['public_repos']}")

        print("\n" + "=" * 70)
        print("【搜索仓库】")
        print("=" * 70)

        # 搜索 Python 机器学习相关仓库
        result = github.search_repositories('python machine learning')
        if result:
            print(f"找到 {result['total_count']} 个仓库\n")
            for i, repo in enumerate(result['items'], 1):
                print(f"{i}. {repo['name']}")
                print(f"   ⭐ {repo['stargazers_count']} | 🍴 {repo['forks_count']}")
                print(f"   {repo['html_url']}\n")


if __name__ == "__main__":
    quick_test()
