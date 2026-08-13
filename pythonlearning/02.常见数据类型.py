print(type("Hello, World!"))  # <class 'str'>
print("Hello, World!")  # Hello, World!
print("======================================")

print(type(42))  # <class 'int'>
print(42)  # 42
print("======================================")

print(type(3.14))  # <class 'float'>
print(3.14)  # 3.14
print("======================================")

print(type(True))  # <class 'bool'>
print(True)  # True
print("======================================")

print(type(False))  # <class 'bool'>
print(False)  # False
print("======================================")

num = 100
print(type(num))  # <class 'int'>
print(num)  # 100
print("======================================")

# isinstance() 函数用于判断一个对象是否是一个已知的类型，类似 type()。
print(isinstance(num, int))  # True
print(isinstance(num, bool))  # False
