import time
from typing import List, Optional, Any
from collections import deque
import heapq

# def compare_time_complexity():
#     n = 1000000
#     # o(1)
#     arr = list(range(n))

#     start = time.time()
#     result = arr[50000]
#     time_o1 = time.time() - start
#     print(f'o(1) - 数组访问：{time_o1:.6f}seconde')

#     # o(n)
#     start = time.time()
#     total =sum(arr)
#     time_on = time.time() - start
#     print(f"0(n) - total:{time_on:.6f}seconde")

#     # o(n^2)
#     start = time.time()
#     count = 0
#     for i in range(100):
#         for j in range(100):
#             count += 1
#     time_on2 = time.time() - start
#     print(f"o(n^2) - loop(n=100):{time_on2:.6f}seconde")
#     print(f"\n performance: o(1) : o(n) : o(n^2) = 1 : {time_on/time_o1:.0f} : {time_on2/time_o1:.0f}")
#     return
# print(compare_time_complexity()) #1:1:3


class ArrayExamples:
    @staticmethod
    def basic_operation():
        arr = [1, 2, 3, 4, 5]
        print(f"1 :{arr}")
        print(f"index2 :{arr[2]}")
        arr[2] = 30
        print(f"modify :{arr}")
        arr.append(6)
        print(f"append(6):{arr}")
        arr.insert(0, 0)
        print(f"insert(0,0):{arr}")
        arr.remove(30)
        print(f"remove(30):{arr}")
        sliced = arr[2:5]
        print(f"sliced:{sliced}")

    @staticmethod
    def two_sum(nums:List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    print(f"find:nums[{i}] + nums[{j}] = {nums[i]} + {nums[j]} = {target}")
                    return[i, j]

        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                print(f"find:nums[{seen[complement]}] + nums[{i}] = {complement} + {num} = {target}")
                return[seen[complement], i]
            seen[num] = i

        return []

    @staticmethod
    def move_zeros(nums: List[int]) -> None:
        left = 0
        for right in range(len(nums)):
            if nums[right] != 0:
                nums[left] = nums[right]
                left += 1

        while left < len(nums):
            nums[left] = 0
            left += 1

        print(f"result:{nums}")


ArrayExamples.two_sum([2, 7, 11, 15], 13)
ArrayExamples.move_zeros([0, 0, 1, 0, 3])


