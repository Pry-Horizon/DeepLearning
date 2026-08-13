# if~~else语句的使用
score = input("请输入你的高考分数：")
score = int(score)
if score >= 680:
    print("我要去读清华！")
else:
    print("我不去清华了！")


# 案例(B站登录 正确的账号和密码是18888888888/666888)

admin = int(input("请输入账号："))
password = int(input("请输入密码："))
bool = (admin == 18888888888) and (password == 666888)
if bool:
    print("登录成功！")
else:
    print("账号或密码错误！")


# 案例(判断闰年)

year = int(input("请输入年份："))
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print(f"{year}是闰年")
else:
    print(f"{year}是平年")


# if~~elif~~else语句的使用
num = float(input("请输入一个数："))
if num > 0:
    print(f"{num}是正数")
elif num < 0:
    print(f"{num}是负数")
else:
    print(f"{num}是零")


# 案例（根据输入用户名/密码登录）
# 用户名/密码为 admin/123456 root/456123 QHH/111222
A_username, B_username, C_username = "admin", "root", "QHH"
A_password, B_password, C_password = 123456, 456123, 111222
username = input("请输入用户名：")
password = int(input("请输入密码："))
if (username == A_username and password == A_password):
    print("A登录成功！")
elif (username == B_username and password == B_password):
    print("B登录成功！")
elif (username == C_username and password == C_password):
    print("C登录成功！")
else:
    print("用户名或密码错误！")


# 案例（识别三角形）
a = float(input("请输入三角形的第一条边："))
b = float(input("请输入三角形的第二条边："))
c = float(input("请输入三角形的第三条边："))
if (a + b > c and a + c > b and b + c > a):
    if a == b and b == c:
        print("这是一个等边三角形")
    elif a == b or a == c or b == c:
        print("这是一个等腰三角形")
    else:
        print("这是一个普通三角形")
else:
    print("这不是一个三角形的边长！")
