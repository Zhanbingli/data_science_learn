"""
数据结构与算法完全指南
作者：Python 专家
目标：从零开始掌握核心数据结构和算法
适合：Python 初学者到进阶开发者
"""

# ============================================================================
# 第一部分：为什么要学习数据结构和算法？
# ============================================================================

"""
【为什么重要？】

1. 技术面试必考 📝
   - 几乎所有大厂都会考察
   - LeetCode 刷题的基础
   - 评估编程能力的标准

2. 提高代码效率 ⚡
   - 选择正确的数据结构能提升 10-1000 倍性能
   - 理解时间复杂度和空间复杂度
   - 写出更优雅的代码

3. 解决实际问题 🔧
   - 推荐系统（图算法）
   - 搜索引擎（哈希表、排序）
   - GPS 导航（最短路径算法）
   - 文件压缩（树、堆）

【学习路线】
数组 → 链表 → 栈 → 队列 → 哈希表 → 树 → 图 → 高级算法
"""

import time
from typing import List, Optional, Any
from collections import deque
import heapq


# ============================================================================
# 第二部分：时间复杂度和空间复杂度（Big O）
# ============================================================================

"""
【Big O 表示法】

时间复杂度：算法执行时间随输入规模增长的趋势

常见复杂度（从快到慢）：
O(1)         < O(log n)    < O(n)     < O(n log n) < O(n²)     < O(2^n)    < O(n!)
常数时间       对数时间      线性时间     线性对数       平方时间      指数时间      阶乘时间
访问数组       二分查找      遍历数组     快速排序       冒泡排序      递归斐波那契   全排列

【实际例子】
"""


def compare_time_complexity():
    """比较不同时间复杂度的实际运行时间"""
    print("\n" + "=" * 70)
    print("【时间复杂度实际对比】")
    print("=" * 70)

    n = 10000

    # O(1) - 常数时间
    start = time.time()
    arr = list(range(n))
    result = arr[5000]  # 直接访问
    time_o1 = time.time() - start
    print(f"O(1) - 数组访问：{time_o1:.6f} 秒")

    # O(n) - 线性时间
    start = time.time()
    total = sum(arr)  # 遍历所有元素
    time_on = time.time() - start
    print(f"O(n) - 遍历数组：{time_on:.6f} 秒")

    # O(n²) - 平方时间
    start = time.time()
    count = 0
    for i in range(100):  # 使用较小的 n
        for j in range(100):
            count += 1
    time_on2 = time.time() - start
    print(f"O(n²) - 双重循环（n=100）：{time_on2:.6f} 秒")

    print(f"\n性能对比：O(1) : O(n) : O(n²) = 1 : {time_on/time_o1:.0f} : {time_on2/time_o1:.0f}")


# ============================================================================
# 第三部分：数组（Array）和列表（List）
# ============================================================================

"""
【数组/列表】
Python 的 list 是动态数组

特点：
✅ 通过索引快速访问：O(1)
✅ 在末尾添加元素：O(1) 平均
❌ 在开头插入/删除：O(n)
❌ 查找元素：O(n)
"""


class ArrayExamples:
    """数组操作示例"""

    @staticmethod
    def basic_operations():
        """基础操作"""
        print("\n" + "=" * 70)
        print("【数组基础操作】")
        print("=" * 70)

        # 创建数组
        arr = [1, 2, 3, 4, 5]
        print(f"初始数组：{arr}")

        # 访问元素 - O(1)
        print(f"访问索引 2：{arr[2]}")

        # 修改元素 - O(1)
        arr[2] = 30
        print(f"修改后：{arr}")

        # 添加元素 - O(1) 平均
        arr.append(6)
        print(f"append(6)：{arr}")

        # 插入元素 - O(n)
        arr.insert(0, 0)
        print(f"insert(0, 0)：{arr}")

        # 删除元素 - O(n)
        arr.remove(30)
        print(f"remove(30)：{arr}")

        # 切片 - O(k) k是切片长度
        sliced = arr[2:5]
        print(f"切片 [2:5]：{sliced}")

    @staticmethod
    def two_sum(nums: List[int], target: int) -> List[int]:
        """
        经典题目：两数之和
        给定一个整数数组和目标值，找出数组中和为目标值的两个数的索引

        Example:
            >>> two_sum([2, 7, 11, 15], 9)
            [0, 1]  # 因为 nums[0] + nums[1] = 2 + 7 = 9
        """
        # 方法1：暴力解法 - O(n²)
        print("\n方法1：暴力解法")
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    print(f"找到：nums[{i}] + nums[{j}] = {nums[i]} + {nums[j]} = {target}")
                    return [i, j]

        # 方法2：哈希表 - O(n)（稍后讲解）
        print("\n方法2：哈希表（更快）")
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                print(f"找到：nums[{seen[complement]}] + nums[{i}] = {complement} + {num} = {target}")
                return [seen[complement], i]
            seen[num] = i

        return []

    @staticmethod
    def move_zeros(nums: List[int]) -> None:
        """
        移动零：将所有 0 移到数组末尾，保持非零元素的相对顺序

        Example:
            >>> nums = [0, 1, 0, 3, 12]
            >>> move_zeros(nums)
            >>> nums
            [1, 3, 12, 0, 0]
        """
        print("\n【移动零】")
        print(f"原数组：{nums}")

        # 双指针法 - O(n)
        left = 0  # 指向下一个非零元素应该放的位置

        # 第一遍：把所有非零元素移到前面
        for right in range(len(nums)):
            if nums[right] != 0:
                nums[left] = nums[right]
                left += 1

        # 第二遍：剩余位置填充 0
        while left < len(nums):
            nums[left] = 0
            left += 1

        print(f"结果：{nums}")


# ============================================================================
# 第四部分：链表（Linked List）
# ============================================================================

"""
【链表】
由节点组成，每个节点包含数据和指向下一个节点的指针

特点：
✅ 插入/删除节点：O(1)（如果已知位置）
❌ 访问元素：O(n)（需要遍历）
❌ 占用额外空间（存储指针）

应用场景：
- LRU 缓存
- 浏览器的前进后退
- 音乐播放器的播放列表
"""


class ListNode:
    """链表节点"""

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class LinkedListExamples:
    """链表操作示例"""

    @staticmethod
    def create_linked_list(values: List[int]) -> Optional[ListNode]:
        """从数组创建链表"""
        if not values:
            return None

        head = ListNode(values[0])
        current = head

        for val in values[1:]:
            current.next = ListNode(val)
            current = current.next

        return head

    @staticmethod
    def print_linked_list(head: Optional[ListNode]):
        """打印链表"""
        values = []
        current = head
        while current:
            values.append(str(current.val))
            current = current.next
        print(" -> ".join(values) + " -> None")

    @staticmethod
    def reverse_linked_list(head: Optional[ListNode]) -> Optional[ListNode]:
        """
        反转链表
        1 -> 2 -> 3 -> 4 -> None
        变成
        4 -> 3 -> 2 -> 1 -> None
        """
        print("\n【反转链表】")

        # 打印原链表
        print("原链表：", end="")
        LinkedListExamples.print_linked_list(head)

        # 迭代法 - O(n)
        prev = None
        current = head

        while current:
            # 保存下一个节点
            next_node = current.next
            # 反转指针
            current.next = prev
            # 移动指针
            prev = current
            current = next_node

        print("反转后：", end="")
        LinkedListExamples.print_linked_list(prev)

        return prev

    @staticmethod
    def has_cycle(head: Optional[ListNode]) -> bool:
        """
        检测链表是否有环（快慢指针）

        思路：
        - 慢指针每次走 1 步
        - 快指针每次走 2 步
        - 如果有环，快慢指针一定会相遇
        """
        if not head:
            return False

        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False


# ============================================================================
# 第五部分：栈（Stack）
# ============================================================================

"""
【栈】
后进先出（LIFO - Last In First Out）

特点：
✅ push（压栈）：O(1)
✅ pop（出栈）：O(1)
✅ peek（查看栈顶）：O(1)

应用场景：
- 函数调用栈
- 浏览器的后退功能
- 括号匹配
- 表达式求值
"""


class Stack:
    """栈的实现"""

    def __init__(self):
        self.items = []

    def push(self, item):
        """入栈"""
        self.items.append(item)

    def pop(self):
        """出栈"""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")

    def peek(self):
        """查看栈顶元素"""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")

    def is_empty(self):
        """判断栈是否为空"""
        return len(self.items) == 0

    def size(self):
        """返回栈的大小"""
        return len(self.items)

    def __repr__(self):
        return f"Stack({self.items})"


class StackExamples:
    """栈的应用示例"""

    @staticmethod
    def valid_parentheses(s: str) -> bool:
        """
        有效的括号
        判断字符串中的括号是否有效配对

        Example:
            >>> valid_parentheses("()")
            True
            >>> valid_parentheses("()[]{}")
            True
            >>> valid_parentheses("(]")
            False
        """
        print(f"\n检查括号：{s}")

        stack = []
        mapping = {')': '(', '}': '{', ']': '['}

        for char in s:
            if char in mapping:
                # 遇到右括号，检查栈顶
                top = stack.pop() if stack else '#'
                if mapping[char] != top:
                    print(f"❌ 不匹配")
                    return False
            else:
                # 遇到左括号，入栈
                stack.append(char)

        result = len(stack) == 0
        print(f"{'✅ 有效' if result else '❌ 无效'}")
        return result

    @staticmethod
    def daily_temperatures(temperatures: List[int]) -> List[int]:
        """
        每日温度
        给定温度列表，返回需要等多少天才能等到更暖和的天气

        Example:
            >>> daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])
            [1, 1, 4, 2, 1, 1, 0, 0]
        """
        print(f"\n温度列表：{temperatures}")

        n = len(temperatures)
        result = [0] * n
        stack = []  # 存储索引

        for i in range(n):
            # 当前温度比栈顶温度高
            while stack and temperatures[i] > temperatures[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = i - prev_index

            stack.append(i)

        print(f"等待天数：{result}")
        return result


# ============================================================================
# 第六部分：队列（Queue）
# ============================================================================

"""
【队列】
先进先出（FIFO - First In First Out）

特点：
✅ enqueue（入队）：O(1)
✅ dequeue（出队）：O(1)

应用场景：
- 任务调度
- 广度优先搜索（BFS）
- 消息队列
- 打印队列
"""


class Queue:
    """队列的实现"""

    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        """入队"""
        self.items.append(item)

    def dequeue(self):
        """出队"""
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Queue is empty")

    def front(self):
        """查看队首元素"""
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Queue is empty")

    def is_empty(self):
        """判断队列是否为空"""
        return len(self.items) == 0

    def size(self):
        """返回队列大小"""
        return len(self.items)

    def __repr__(self):
        return f"Queue({list(self.items)})"


class QueueExamples:
    """队列应用示例"""

    @staticmethod
    def level_order_traversal(root):
        """
        二叉树的层序遍历（使用队列）
        稍后在树的部分详细讲解
        """
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            level = []

            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level)

        return result


# ============================================================================
# 第七部分：哈希表（Hash Table / Dictionary）
# ============================================================================

"""
【哈希表】
Python 的 dict 就是哈希表

特点：
✅ 查找元素：O(1) 平均
✅ 插入元素：O(1) 平均
✅ 删除元素：O(1) 平均

应用场景：
- 快速查找
- 去重
- 统计频率
- 缓存
"""


class HashTableExamples:
    """哈希表应用示例"""

    @staticmethod
    def find_duplicates(nums: List[int]) -> List[int]:
        """
        查找数组中的重复元素

        Example:
            >>> find_duplicates([4,3,2,7,8,2,3,1])
            [2, 3]
        """
        print(f"\n查找重复：{nums}")

        seen = set()
        duplicates = set()

        for num in nums:
            if num in seen:
                duplicates.add(num)
            else:
                seen.add(num)

        result = list(duplicates)
        print(f"重复元素：{result}")
        return result

    @staticmethod
    def group_anagrams(strs: List[str]) -> List[List[str]]:
        """
        字母异位词分组

        Example:
            >>> group_anagrams(["eat","tea","tan","ate","nat","bat"])
            [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
        """
        print(f"\n字母异位词分组：{strs}")

        groups = {}

        for s in strs:
            # 排序后的字符串作为 key
            key = ''.join(sorted(s))
            if key not in groups:
                groups[key] = []
            groups[key].append(s)

        result = list(groups.values())
        print(f"分组结果：{result}")
        return result

    @staticmethod
    def top_k_frequent(nums: List[int], k: int) -> List[int]:
        """
        前 K 个高频元素

        Example:
            >>> top_k_frequent([1,1,1,2,2,3], 2)
            [1, 2]
        """
        print(f"\n找出前 {k} 个高频元素：{nums}")

        # 统计频率
        from collections import Counter
        counter = Counter(nums)

        # 使用堆（稍后讲解）
        return [num for num, _ in counter.most_common(k)]


# ============================================================================
# 第八部分：树（Tree）
# ============================================================================

"""
【二叉树】
每个节点最多有两个子节点

特点：
- 前序遍历：根 -> 左 -> 右
- 中序遍历：左 -> 根 -> 右
- 后序遍历：左 -> 右 -> 根
- 层序遍历：一层一层遍历

应用场景：
- 文件系统
- 数据库索引（B树）
- 表达式树
- 决策树
"""


class TreeNode:
    """二叉树节点"""

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeExamples:
    """树的操作示例"""

    @staticmethod
    def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
        """
        中序遍历（递归）
        左 -> 根 -> 右
        """
        result = []

        def traverse(node):
            if not node:
                return

            traverse(node.left)     # 左
            result.append(node.val)  # 根
            traverse(node.right)    # 右

        traverse(root)
        return result

    @staticmethod
    def max_depth(root: Optional[TreeNode]) -> int:
        """
        二叉树的最大深度

        思路：递归
        - 如果节点为空，深度为 0
        - 否则，深度 = max(左子树深度, 右子树深度) + 1
        """
        if not root:
            return 0

        left_depth = TreeExamples.max_depth(root.left)
        right_depth = TreeExamples.max_depth(root.right)

        return max(left_depth, right_depth) + 1

    @staticmethod
    def is_valid_bst(root: Optional[TreeNode]) -> bool:
        """
        验证二叉搜索树

        二叉搜索树（BST）：
        - 左子树的所有节点 < 根节点
        - 右子树的所有节点 > 根节点
        """

        def validate(node, min_val, max_val):
            if not node:
                return True

            if node.val <= min_val or node.val >= max_val:
                return False

            return (validate(node.left, min_val, node.val) and
                    validate(node.right, node.val, max_val))

        return validate(root, float('-inf'), float('inf'))


# ============================================================================
# 第九部分：堆（Heap）
# ============================================================================

"""
【堆】
完全二叉树，分为最大堆和最小堆

最小堆特点：
- 父节点 ≤ 子节点
- 根节点是最小值

操作：
✅ 插入元素：O(log n)
✅ 删除最小值：O(log n)
✅ 查看最小值：O(1)

应用场景：
- 优先队列
- Top K 问题
- 中位数维护
"""


class HeapExamples:
    """堆的应用示例"""

    @staticmethod
    def kth_largest(nums: List[int], k: int) -> int:
        """
        数组中第 K 个最大元素

        Example:
            >>> kth_largest([3,2,1,5,6,4], 2)
            5
        """
        print(f"\n找第 {k} 大的元素：{nums}")

        # 使用最小堆，保持堆大小为 k
        heap = []

        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)

        result = heap[0]
        print(f"第 {k} 大的元素：{result}")
        return result

    @staticmethod
    def merge_k_sorted_lists(lists: List[List[int]]) -> List[int]:
        """
        合并 K 个有序列表

        Example:
            >>> merge_k_sorted_lists([[1,4,5],[1,3,4],[2,6]])
            [1,1,2,3,4,4,5,6]
        """
        print(f"\n合并 K 个有序列表：{lists}")

        heap = []
        result = []

        # 把每个列表的第一个元素放入堆
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))

        while heap:
            val, list_idx, elem_idx = heapq.heappop(heap)
            result.append(val)

            # 如果该列表还有元素，继续加入堆
            if elem_idx + 1 < len(lists[list_idx]):
                next_val = lists[list_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

        print(f"合并结果：{result}")
        return result


# ============================================================================
# 第十部分：排序算法
# ============================================================================

"""
【常见排序算法】

算法          时间复杂度(平均)  空间复杂度  稳定性
冒泡排序      O(n²)            O(1)        稳定
选择排序      O(n²)            O(1)        不稳定
插入排序      O(n²)            O(1)        稳定
快速排序      O(n log n)       O(log n)    不稳定
归并排序      O(n log n)       O(n)        稳定
堆排序        O(n log n)       O(1)        不稳定
"""


class SortingAlgorithms:
    """排序算法实现"""

    @staticmethod
    def bubble_sort(arr: List[int]) -> List[int]:
        """
        冒泡排序
        重复遍历，比较相邻元素，交换顺序
        """
        n = len(arr)
        arr = arr.copy()

        for i in range(n):
            swapped = False
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True

            if not swapped:
                break

        return arr

    @staticmethod
    def quick_sort(arr: List[int]) -> List[int]:
        """
        快速排序
        选择基准值，分区，递归排序
        """
        if len(arr) <= 1:
            return arr

        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]

        return SortingAlgorithms.quick_sort(left) + middle + SortingAlgorithms.quick_sort(right)

    @staticmethod
    def merge_sort(arr: List[int]) -> List[int]:
        """
        归并排序
        分而治之，递归排序后合并
        """
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid])
        right = SortingAlgorithms.merge_sort(arr[mid:])

        return SortingAlgorithms.merge(left, right)

    @staticmethod
    def merge(left: List[int], right: List[int]) -> List[int]:
        """合并两个有序数组"""
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result


# ============================================================================
# 第十一部分：搜索算法
# ============================================================================

"""
【搜索算法】

1. 线性搜索：O(n)
2. 二分搜索：O(log n) - 仅用于有序数组
3. 深度优先搜索（DFS）：用于树、图
4. 广度优先搜索（BFS）：用于树、图
"""


class SearchAlgorithms:
    """搜索算法实现"""

    @staticmethod
    def binary_search(arr: List[int], target: int) -> int:
        """
        二分搜索
        在有序数组中查找目标值

        Returns:
            目标值的索引，如果不存在返回 -1
        """
        left, right = 0, len(arr) - 1

        while left <= right:
            mid = (left + right) // 2

            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1

    @staticmethod
    def search_insert_position(arr: List[int], target: int) -> int:
        """
        搜索插入位置
        在有序数组中找到目标值，如果不存在则返回应该插入的位置

        Example:
            >>> search_insert_position([1,3,5,6], 5)
            2
            >>> search_insert_position([1,3,5,6], 2)
            1
        """
        left, right = 0, len(arr)

        while left < right:
            mid = (left + right) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid

        return left


# ============================================================================
# 主程序 - 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 数据结构与算法完全教程")
    print("=" * 70)

    # 时间复杂度对比
    compare_time_complexity()

    # 数组
    print("\n" + "=" * 70)
    print("第一章：数组（Array）")
    print("=" * 70)
    ArrayExamples.basic_operations()
    ArrayExamples.two_sum([2, 7, 11, 15], 9)
    nums = [0, 1, 0, 3, 12]
    ArrayExamples.move_zeros(nums)

    # 链表
    print("\n" + "=" * 70)
    print("第二章：链表（Linked List）")
    print("=" * 70)
    head = LinkedListExamples.create_linked_list([1, 2, 3, 4, 5])
    LinkedListExamples.reverse_linked_list(head)

    # 栈
    print("\n" + "=" * 70)
    print("第三章：栈（Stack）")
    print("=" * 70)
    stack = Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print(f"栈：{stack}")
    print(f"栈顶：{stack.peek()}")
    print(f"出栈：{stack.pop()}")
    print(f"栈：{stack}")

    StackExamples.valid_parentheses("()[]{}")
    StackExamples.valid_parentheses("(]")
    StackExamples.daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])

    # 队列
    print("\n" + "=" * 70)
    print("第四章：队列（Queue）")
    print("=" * 70)
    queue = Queue()
    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)
    print(f"队列：{queue}")
    print(f"队首：{queue.front()}")
    print(f"出队：{queue.dequeue()}")
    print(f"队列：{queue}")

    # 哈希表
    print("\n" + "=" * 70)
    print("第五章：哈希表（Hash Table）")
    print("=" * 70)
    HashTableExamples.find_duplicates([4, 3, 2, 7, 8, 2, 3, 1])
    HashTableExamples.group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])

    # 堆
    print("\n" + "=" * 70)
    print("第六章：堆（Heap）")
    print("=" * 70)
    HeapExamples.kth_largest([3, 2, 1, 5, 6, 4], 2)
    HeapExamples.merge_k_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]])

    # 排序
    print("\n" + "=" * 70)
    print("第七章：排序算法")
    print("=" * 70)
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"原数组：{arr}")
    print(f"冒泡排序：{SortingAlgorithms.bubble_sort(arr)}")
    print(f"快速排序：{SortingAlgorithms.quick_sort(arr)}")
    print(f"归并排序：{SortingAlgorithms.merge_sort(arr)}")

    # 搜索
    print("\n" + "=" * 70)
    print("第八章：搜索算法")
    print("=" * 70)
    sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"有序数组：{sorted_arr}")
    print(f"二分搜索 7：索引 {SearchAlgorithms.binary_search(sorted_arr, 7)}")
    print(f"搜索插入位置 6：索引 {SearchAlgorithms.search_insert_position(sorted_arr, 6)}")

    print("\n" + "=" * 70)
    print("🎉 教程完成！")
    print("=" * 70)
    print("\n💡 下一步：")
    print("1. 在 LeetCode 刷题巩固")
    print("2. 实现更多算法")
    print("3. 参加算法竞赛")
    print("=" * 70)
