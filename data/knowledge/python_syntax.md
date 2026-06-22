# Python 语法基础知识库

## Python 程序与缩进规则
Python 是解释型、动态类型语言。程序通常从上到下执行，代码块不用花括号，而是依靠缩进表示层级。`if`、`for`、`while`、`def`、`class` 等语句后面通常以冒号结尾，下一行缩进形成代码块。缩进不一致会导致 `IndentationError`。

示例：
```python
if score >= 60:
    print("pass")
else:
    print("fail")
```

## 变量、对象与引用
Python 中变量是对象的引用，赋值语句会让变量名绑定到对象。变量不需要提前声明类型，类型属于对象本身，而不是变量名本身。`type()` 可以查看对象类型，`id()` 可以查看对象身份标识。

示例：
```python
name = "Alice"
age = 18
print(type(name))
```

## 可变对象与不可变对象
Python 对象可以分为可变对象和不可变对象。不可变对象创建后内容不能原地修改，例如 `int`、`float`、`bool`、`str`、`tuple`。可变对象可以原地修改，例如 `list`、`dict`、`set`。理解可变性有助于分析参数传递、浅拷贝、默认参数等问题。

## 数字与布尔类型
`int` 表示整数，`float` 表示浮点数，`bool` 表示布尔值。布尔值只有 `True` 和 `False`。比较运算会返回布尔值，例如 `3 > 2` 的结果是 `True`。浮点数存在精度误差，比较浮点数时不应简单依赖完全相等。

## 字符串 str
字符串 `str` 是不可变序列，用来保存文本。常见操作包括索引、切片、拼接、查找、替换和格式化。常用方法有 `strip()`、`split()`、`join()`、`replace()`、`lower()`、`upper()`。字符串格式化推荐使用 f-string。

示例：
```python
name = "Tom"
message = f"hello, {name}"
```

## 列表 list
列表 `list` 是可变序列，可以保存多个元素，元素类型可以不同。列表支持索引、切片、遍历和原地修改。常用方法包括 `append()`、`extend()`、`insert()`、`remove()`、`pop()`、`sort()`。列表适合保存需要增删改的数据集合。

示例：
```python
numbers = [1, 2, 3]
numbers.append(4)
numbers[0] = 100
```

## 元组 tuple
元组 `tuple` 是不可变序列，创建后不能修改其中元素。元组适合保存固定结构的数据，例如坐标、日期、函数返回的多个值。只有一个元素的元组必须写成 `(value,)`，逗号是关键。

示例：
```python
point = (10, 20)
single = (1,)
```

## list 和 tuple 的区别 difference
`list` 和 `tuple` 都是有序序列，都支持索引、切片和遍历。核心区别是：`list` 可变，适合保存需要增删改的数据；`tuple` 不可变，适合保存固定结构、不可随意修改的数据。由于 tuple 不可变，它更适合表达“这组值不应该被改”的语义，也可以在满足元素可哈希时作为字典键。

简要对比：
- `list`: mutable, supports append/remove/item assignment.
- `tuple`: immutable, does not support item assignment.
- list 用方括号 `[]`，tuple 通常用圆括号 `()`。

## 字典 dict
字典 `dict` 使用键值对保存数据，适合按照键快速查找值。键必须是可哈希对象，常见键类型包括字符串、数字和元组。常用操作包括新增键值对、修改值、删除键、遍历 `items()`。

示例：
```python
scores = {"Tom": 90, "Jerry": 85}
scores["Tom"] = 95
```

## 集合 set
集合 `set` 是无序且元素不重复的数据结构，适合去重和集合运算。常见操作包括 `add()`、`remove()`、交集 `&`、并集 `|`、差集 `-`。集合元素也必须是可哈希对象。

## 条件语句 if elif else
条件语句用于根据条件选择不同分支。`if` 判断第一个条件，`elif` 表示其他条件，`else` 表示所有条件都不满足时执行。Python 中空字符串、空列表、空字典、数字 0、`None` 通常会被视为假值。

## for 循环
`for` 循环用于遍历可迭代对象，例如列表、字符串、字典、range 对象。`range(start, stop, step)` 常用于生成整数序列。遍历字典时可以使用 `items()` 同时获得键和值。

示例：
```python
for name, score in scores.items():
    print(name, score)
```

## while 循环
`while` 循环会在条件为真时反复执行代码块，适合循环次数不确定的场景。循环体中可以使用 `break` 提前结束循环，使用 `continue` 跳过本次循环进入下一轮。

## 函数 def function
Python 使用 `def` 关键字定义函数。函数可以接收参数，可以通过 `return` 返回结果。函数的作用是封装一段可复用逻辑，让程序结构更清晰。

语法格式：
```python
def add(a, b):
    return a + b
```

函数调用时，位置参数按照顺序传入，关键字参数按照参数名传入。默认参数可以提供默认值，但不要把可变对象作为默认参数。

## 函数参数
常见参数形式包括位置参数、关键字参数、默认参数、可变位置参数 `*args` 和可变关键字参数 `**kwargs`。默认参数在函数定义时计算一次，因此 `def f(items=[])` 这种写法可能导致多次调用共享同一个列表。

推荐写法：
```python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

## 作用域 LEGB
Python 变量查找遵循 LEGB 规则：Local 本地作用域、Enclosing 外层函数作用域、Global 全局作用域、Built-in 内置作用域。在函数内部给变量赋值时，默认会创建本地变量。需要修改全局变量时可以使用 `global`，但应谨慎使用。

## 异常处理
Python 使用 `try`、`except`、`else`、`finally` 处理异常。`try` 中放可能出错的代码，`except` 捕获指定异常，`else` 在没有异常时执行，`finally` 无论是否异常都会执行。异常处理不应过度宽泛，避免用裸 `except` 吞掉真实错误。

示例：
```python
try:
    number = int(text)
except ValueError:
    print("input is not an integer")
```

## 文件读写
文件读写推荐使用 `with open(...) as f`，这样文件会在代码块结束后自动关闭。读取文本文件时应指定编码，例如 `encoding="utf-8"`。常用方法包括 `read()`、`readline()`、`readlines()` 和迭代文件对象。

示例：
```python
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

## 模块导入
Python 使用 `import` 导入模块。可以导入标准库、第三方库或自己写的模块。`from module import name` 可以导入模块中的指定对象。脚本中常用 `if __name__ == "__main__":` 判断当前文件是否作为主程序运行。

## 类与对象
类 `class` 用于定义对象的属性和方法。`__init__` 是构造方法，在创建对象时初始化属性。实例方法第一个参数通常命名为 `self`，表示当前对象。

示例：
```python
class Student:
    def __init__(self, name):
        self.name = name

    def say_hi(self):
        return f"hi, {self.name}"
```

## 列表推导式
列表推导式用于简洁地生成列表，格式通常为 `[表达式 for 变量 in 可迭代对象 if 条件]`。它适合简单转换和过滤，但复杂逻辑应使用普通循环提升可读性。

示例：
```python
squares = [x * x for x in range(5)]
```

## 常见易错点
1. 不要把可变对象作为默认参数。
2. 遍历列表时直接删除元素容易跳过数据，建议创建新列表或倒序处理。
3. 浮点数比较要考虑精度误差。
4. `is` 比较对象身份，`==` 比较值是否相等。
5. 变量赋值不是复制对象，多个变量可能引用同一个可变对象。
