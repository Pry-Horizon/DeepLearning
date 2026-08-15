"""
1. while循环：
    ===================
    while 条件表达式
        循环体语句1
        循环体语句2
        ....
    ===================
"""
#案例：打印十遍人生苦短，我用python！
i = 0
while i<10:
    print("人生苦短，我用python！")
    i = i + 1
else:
    print("循环结束")

#案例：计算1-100之间所有偶数之和
i = 0
sum = 0
while i<100:
    i += 2
    sum = sum + i
else:
    print(f"1-100所有偶数之和为{sum}")
