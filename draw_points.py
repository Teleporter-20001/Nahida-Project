import tkinter as tk
import re

# 假设你有如下的点列表
points = []

# 创建tkinter窗口和画布
root = tk.Tk()
canvas = tk.Canvas(root, width=600, height=800)
canvas.pack()

with open("AI_data/behit_pos.txt", 'r') as file:
    content = file.read()
    matches = re.findall(r'\((.*?),\s(.*?)\)', content)

    for pt in matches:
        points.append((float(pt[0]), float(pt[1])))

    # 对于点列表中的每个点，创建一个小的圆作为散点
    for point in points:
        x, y = point
        canvas.create_oval(x-2, y-2, x+2, y+2, fill='black')
    canvas.create_oval(300-2, 560-2, 300+2, 560+2, fill='red')

root.mainloop()
