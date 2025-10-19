"""
Python 面向对象编程（OOP）完全指南
作者：Python 专家
目标：让小白轻松理解类、对象、继承和多态
"""

# ============================================================================
# 第一部分：什么是面向对象编程？
# ============================================================================

"""
【比喻理解】
- 面向过程编程：像做菜的步骤（切菜 → 炒菜 → 装盘）
- 面向对象编程：像制造汽车（把引擎、轮胎、车身组装起来，每个部件都是独立的对象）

【核心思想】
把现实世界的事物抽象成"对象"，每个对象有：
1. 属性（特征）：汽车的颜色、品牌、价格
2. 方法（行为）：汽车能加速、刹车、鸣笛
"""


# ============================================================================
# 第二部分：函数（Function）- 基础构建块
# ============================================================================

def greet(name):
    """
    函数：完成特定任务的代码块
    - 可以接收参数（name）
    - 可以返回结果
    """
    return f"你好，{name}！"


def add_numbers(a, b):
    """简单的加法函数"""
    return a + b


# 使用函数
print("=" * 60)
print("【函数示例】")
print(greet("小明"))
print(f"5 + 3 = {add_numbers(5, 3)}")


# ============================================================================
# 第三部分：类（Class）和对象（Object）
# ============================================================================

"""
【类】= 蓝图/模板
【对象】= 根据蓝图制造出来的实体

比喻：
- 类：汽车设计图纸
- 对象：根据图纸生产的一辆辆真实的汽车
"""


class Dog:
    """狗类 - 定义狗的通用特征和行为"""

    # 类属性：所有狗共享的属性
    species = "犬科动物"

    def __init__(self, name, age, breed):
        """
        构造函数（初始化方法）
        当创建新对象时自动调用

        self：代表对象自己
        name, age, breed：实例属性，每只狗都不同
        """
        self.name = name      # 实例属性：狗的名字
        self.age = age        # 实例属性：狗的年龄
        self.breed = breed    # 实例属性：狗的品种

    def bark(self):
        """狗叫的方法"""
        return f"{self.name}：汪汪汪！"

    def get_info(self):
        """获取狗的信息"""
        return f"{self.name}是一只{self.age}岁的{self.breed}"

    def celebrate_birthday(self):
        """过生日，年龄+1"""
        self.age += 1
        return f"{self.name}过生日了！现在{self.age}岁啦！"


# 创建对象（实例化）
print("\n" + "=" * 60)
print("【类和对象示例】")

dog1 = Dog("旺财", 3, "金毛")
dog2 = Dog("小黑", 2, "拉布拉多")

# 访问属性
print(f"狗1的名字：{dog1.name}")
print(f"狗2的品种：{dog2.breed}")
print(f"类属性：{Dog.species}")

# 调用方法
print(dog1.bark())
print(dog1.get_info())
print(dog2.get_info())
print(dog1.celebrate_birthday())


# ============================================================================
# 第四部分：继承（Inheritance）
# ============================================================================

"""
【继承】：子类继承父类的属性和方法

比喻：
- 父类（Animal）：所有动物的共同特征（吃饭、睡觉）
- 子类（Dog、Cat）：继承动物的特征，还有自己的特殊行为

优点：
1. 代码复用：不用重复写相同的代码
2. 层次结构：清晰的组织关系
3. 扩展性：方便添加新功能
"""


class Animal:
    """动物父类"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        """所有动物都会吃"""
        return f"{self.name}正在吃东西"

    def sleep(self):
        """所有动物都会睡觉"""
        return f"{self.name}正在睡觉 💤"

    def make_sound(self):
        """动物叫声（父类中的通用方法）"""
        return "动物发出声音"


class Cat(Animal):
    """猫类 - 继承自 Animal"""

    def __init__(self, name, age, color):
        # super() 调用父类的构造函数
        super().__init__(name, age)
        self.color = color  # 猫特有的属性

    def make_sound(self):
        """重写父类方法（方法覆盖）"""
        return f"{self.name}：喵喵喵~"

    def climb_tree(self):
        """猫特有的方法"""
        return f"{self.color}色的{self.name}爬上了树！"


class Bird(Animal):
    """鸟类 - 继承自 Animal"""

    def __init__(self, name, age, can_fly=True):
        super().__init__(name, age)
        self.can_fly = can_fly

    def make_sound(self):
        """重写父类方法"""
        return f"{self.name}：叽叽喳喳！"

    def fly(self):
        """鸟特有的方法"""
        if self.can_fly:
            return f"{self.name}飞向了天空 🦅"
        else:
            return f"{self.name}不会飞（比如企鹅）"


# 使用继承
print("\n" + "=" * 60)
print("【继承示例】")

cat = Cat("咪咪", 2, "橘")
bird = Bird("小鸟", 1)
penguin = Bird("企鹅", 3, can_fly=False)

# 调用继承的方法
print(cat.eat())
print(bird.sleep())

# 调用重写的方法
print(cat.make_sound())
print(bird.make_sound())

# 调用子类特有的方法
print(cat.climb_tree())
print(bird.fly())
print(penguin.fly())


# ============================================================================
# 第五部分：多态（Polymorphism）
# ============================================================================

"""
【多态】：同一个方法，不同的对象调用时表现不同

关键特征：
1. 不同的子类对同一消息做出不同的响应
2. 父类引用指向子类对象
3. 方法重写（覆盖）是实现多态的基础

比喻：
"叫一声" 这个指令：
- 对狗说 → 汪汪汪
- 对猫说 → 喵喵喵
- 对鸟说 → 叽叽喳喳

同一个指令，不同的结果！
"""


def animal_concert(animals):
    """动物音乐会 - 展示多态"""
    print("\n🎵 动物音乐会开始！")
    print("-" * 60)
    for animal in animals:
        # 同样调用 make_sound()，但每个动物的表现不同
        print(animal.make_sound())
        print(animal.eat())
        print("-" * 60)


# 多态示例
print("\n" + "=" * 60)
print("【多态示例】")

# 创建不同类型的动物
animals = [
    Cat("小花", 1, "白"),
    Bird("鹦鹉", 2),
    Cat("大橘", 3, "橘"),
    Bird("麻雀", 1)
]

# 统一处理，但表现各异
animal_concert(animals)


# ============================================================================
# 第六部分：实战案例 - 机器学习模型评估器
# ============================================================================

"""
结合之前看到的 ModelEvaluator，理解 OOP 的实际应用
"""


class BaseEvaluator:
    """评估器基类"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics = {}

    def evaluate(self, y_true, y_pred):
        """评估方法（父类中定义接口）"""
        raise NotImplementedError("子类必须实现此方法")

    def print_report(self):
        """打印报告"""
        print(f"\n{'=' * 50}")
        print(f"{self.model_name} 评估报告")
        print(f"{'=' * 50}")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")


class ClassificationEvaluator(BaseEvaluator):
    """分类评估器 - 继承自 BaseEvaluator"""

    def __init__(self, model_name):
        super().__init__(model_name)

    def evaluate(self, y_true, y_pred):
        """重写评估方法 - 计算分类指标"""
        # 简化版准确率计算
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct / len(y_true)

        self.metrics = {
            '准确率 (Accuracy)': accuracy,
            '样本数量': len(y_true),
            '正确预测': correct
        }
        return self.metrics


class RegressionEvaluator(BaseEvaluator):
    """回归评估器 - 继承自 BaseEvaluator"""

    def __init__(self, model_name):
        super().__init__(model_name)

    def evaluate(self, y_true, y_pred):
        """重写评估方法 - 计算回归指标"""
        import numpy as np

        # 计算 MAE 和 MSE
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

        self.metrics = {
            '平均绝对误差 (MAE)': mae,
            '均方误差 (MSE)': mse,
            '样本数量': len(y_true)
        }
        return self.metrics


# 使用评估器
print("\n" + "=" * 60)
print("【实战案例：模型评估器】")

# 分类任务
clf_eval = ClassificationEvaluator("随机森林分类器")
y_true_clf = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred_clf = [1, 0, 1, 0, 0, 1, 0, 1]
clf_eval.evaluate(y_true_clf, y_pred_clf)
clf_eval.print_report()

# 回归任务
reg_eval = RegressionEvaluator("线性回归器")
y_true_reg = [10.5, 20.3, 15.7, 30.2]
y_pred_reg = [11.0, 19.8, 16.2, 29.5]
reg_eval.evaluate(y_true_reg, y_pred_reg)
reg_eval.print_report()


# ============================================================================
# 第七部分：OOP 四大特性总结
# ============================================================================

print("\n" + "=" * 60)
print("【面向对象四大特性总结】")
print("=" * 60)

summary = """
1. 封装（Encapsulation）
   - 把数据和方法打包在一起
   - 隐藏内部实现细节
   - 示例：Dog 类把 name、age 和 bark() 封装在一起

2. 继承（Inheritance）
   - 子类继承父类的属性和方法
   - 实现代码复用
   - 示例：Cat 和 Bird 继承自 Animal

3. 多态（Polymorphism）
   - 同一方法，不同对象有不同表现
   - 提高代码灵活性
   - 示例：make_sound() 在不同动物中的表现

4. 抽象（Abstraction）
   - 只关注对象的必要特征
   - 忽略不相关的细节
   - 示例：BaseEvaluator 定义评估接口，具体实现由子类完成
"""

print(summary)


# ============================================================================
# 第八部分：关键概念对比表
# ============================================================================

print("\n" + "=" * 60)
print("【关键概念对比】")
print("=" * 60)

comparison = """
┌─────────────┬──────────────────────┬────────────────────────┐
│   概念      │       定义           │         比喻            │
├─────────────┼──────────────────────┼────────────────────────┤
│   类 (Class)│  对象的蓝图/模板     │  汽车设计图纸          │
│   对象 (Object)│ 类的实例          │  一辆具体的汽车        │
│   属性 (Attribute)│ 对象的特征    │  汽车的颜色、品牌      │
│   方法 (Method)│ 对象的行为       │  汽车的加速、刹车      │
│   self      │  对象自己的引用      │  "我自己"              │
│   继承 (Inheritance)│ 子类继承父类 │  孩子继承父母的特征    │
│   多态 (Polymorphism)│ 同方法不同表现│ 同一指令不同反应     │
└─────────────┴──────────────────────┴────────────────────────┘
"""

print(comparison)


# ============================================================================
# 第九部分：常见错误和注意事项
# ============================================================================

print("\n" + "=" * 60)
print("【常见错误】")
print("=" * 60)

# 错误1：忘记 self
"""
class WrongDog:
    def __init__(name, age):  # ❌ 缺少 self
        name = name           # ❌ 不会保存到对象
        age = age

# 正确写法
class CorrectDog:
    def __init__(self, name, age):  # ✅ 有 self
        self.name = name              # ✅ 保存到对象
        self.age = age
"""

# 错误2：忘记调用父类构造函数
"""
class WrongCat(Animal):
    def __init__(self, name, age, color):
        # ❌ 忘记调用父类 __init__
        self.color = color

# 正确写法
class CorrectCat(Animal):
    def __init__(self, name, age, color):
        super().__init__(name, age)  # ✅ 调用父类
        self.color = color
"""


# ============================================================================
# 第十部分：练习题
# ============================================================================

print("\n" + "=" * 60)
print("【练习题：尝试自己实现】")
print("=" * 60)

practice = """
练习1：创建一个 Student 类
- 属性：name（姓名）、age（年龄）、grade（年级）
- 方法：introduce()（自我介绍）、study(subject)（学习某科目）

练习2：创建 Teacher 类继承 Person 类
- Person 有 name 和 age
- Teacher 额外有 subject（教授科目）
- Teacher 有 teach() 方法

练习3：实现多态
- 创建多个不同的形状类（Circle、Rectangle、Triangle）
- 都有 calculate_area() 方法
- 创建一个函数统一处理不同形状的面积计算

提示：参考上面的示例代码，动手敲一遍就会理解了！
"""

print(practice)

print("\n" + "=" * 60)
print("🎉 恭喜！你已经掌握了 Python 面向对象编程的基础！")
print("=" * 60)
