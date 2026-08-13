name = input("请输入你的名字：")
print(f"欢迎您,{name}")
age = input("请输入你的年龄：")
print(f"您的年龄是：{age}")

# 案例，小智取款
total = 10000
password = input("请输入取款密码：")
print(f"密码正确，{password}")
num = input("请输入取款金额：")
print(f"取款成功，您的余额是：{total - int(num)}")

# 案例，加法计算器
num1 = input("请输入第一个数字：")
num2 = input("请输入第二个数字：")
print(f"计算结果是：{num1}+{num2} = {int(num1) + int(num2)}")
