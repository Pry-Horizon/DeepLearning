# 算数运算符

print("10 +20 =", 10 + 20)  # 加法
print("10 - 20 =", 10 - 20)  # 减法
print("10 * 20 =", 10 * 20)  # 乘法
print("10 / 20 =", 10 / 20)  # 除法
print("10 // 20 =", 10 // 20)  # 整除
print("10**2 =", 10 ** 2)  # 幂运算


# 算数运算符的优先级
print("10 + 20 * 5 =", 10 + 20 * 5)  # 乘法优先于加法
print("(10 + 20) * 5 =", (10 + 20) * 5)  # 括号优先于其他运算符


# 赋值运算符(= +=, -=, *=, /=, //=, **=)
num = 85
num += 10  # num = num + 10
print("num += 10 =", num)

num -= 5  # num = num - 5
print("num -= 5 =", num)

num *= 2  # num = num * 2
print("num *= 2 =", num)

num /= 4  # num = num / 4
print("num /= 4 =", num)

num //= 2  # num = num // 2
print("num //= 2 =", num)

num **= 3  # num = num ** 3
print("num **= 3 =", num)


# 比较运算符(==, !=, >, <, >=, <=)

print("10 == 20:", 10 == 20)  # 等于
print("10 != 20:", 10 != 20)  # 不等于
print("10 > 20:", 10 > 20)   # 大于
print("10 < 20:", 10 < 20)   # 小于
print("10 >= 20:", 10 >= 20)  # 大于等于
print("10 <= 20:", 10 <= 20)  # 小于等于

# 逻辑运算符(and, or, not)
print("True and False:", True and False)  # 与
print("True or False:", True or False)   # 或
print("not True:", not True)             # 非

# 案例
num = input("请输入一个整数：")
num = int(num)
if 10 <= num <= 20:
    print(f"{num}在10到20之间:")
else:
    print(f"{num}不在10到20之间:")
