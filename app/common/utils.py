

def to_ints(iterable):
    t = type(iterable)  # 记住原类型
    return t(int(x) for x in iterable)
