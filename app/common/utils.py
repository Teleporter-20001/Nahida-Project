

def to_ints(iterable):
    t = type(iterable)  # 记住原类型
    return t(int(x) for x in iterable)

def printred(*args, **kwargs):
    print("\033[91m", end="")  # 红色
    print(*args, **kwargs)
    print("\033[0m", end="")  # 结束颜色
    
def printyellow(*args, **kwargs):
    print("\033[93m", end="")  # 黄色
    print(*args, **kwargs)
    print("\033[0m", end="")  # 结束颜色
    
def printgreen(*args, **kwargs):
    print("\033[92m", end="")  # 绿色
    print(*args, **kwargs)
    print("\033[0m", end="")  # 结束颜色
    
def printblue(*args, **kwargs):
    print("\033[94m", end="")  # 蓝色
    print(*args, **kwargs)
    print("\033[0m", end="")  # 结束颜色
    
def printorange(*args, **kwargs):
    print("\033[38;5;208m", end="")  # 橙色
    print(*args, **kwargs)
    print("\033[0m", end="")  # 结束颜色
    
def printpurple(*args, **kwargs):
    print("\033[95m", end="")  # 紫色
    print(*args, **kwargs)
    print("\033[0m", end="")  # 结束颜色
    
def printcyan(*args, **kwargs):
    print("\033[96m", end="")  # 青色
    print(*args, **kwargs)
    print("\033[0m", end="")  # 结束颜色