# 字面量的写法
print(100)  # 整数
print(3.14)
print(True)
print(False)
print("Hello, World!")
print(None)
print(True+1)
print("==============================================================================")


# 变量
num = 1114.1
print(num)

num = num + 1
print(num)

num = "ok"
print(num)

num = True
print(num)
print("==============================================================================")

# 案例
base, new = 20.7, 50

month = 1
sum = base + new * month
print("第一个月的收入是", sum)

month = 2
sum = base + new * month
print("第二个月的收入是", sum)
print("==============================================================================")

# 标识符命名规则
# 1. 只能包含字母、数字和下划线，不能以数字开头
# 2. 不能使用关键字作为标识符
# 3. 关键字有: and, as, assert, break, class, continue, def, del, elif, else, except, exec, finally, for, from, global, if, import, in, is, lambda, not, or, pass, print, raise, return, try, while, with, yield
true = 1
print(true)
