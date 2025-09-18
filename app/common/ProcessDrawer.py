import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class ProcessDrawer:
    def __init__(self, title="Dynamic Curve", xlabel="epoch", ylabel="y", color="b-"):

        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = "1000,200"

        plt.ion()  # 打开交互模式
        self.fig, self.ax = plt.subplots()
        self.x_data, self.y_data = [], []
        (self.line,) = self.ax.plot([], [], color, label="reward")

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()

        os.environ.pop('SDL_VIDEO_WINDOW_POS', None)

    def add_data(self, x, y):
        """追加数据点"""
        self.x_data.append(x)
        self.y_data.append(y)

    def clear(self):
        self.x_data.clear()
        self.y_data.clear()

    def update(self, pause_time=0.01):
        """更新图像"""
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()            # 重新计算数据范围
        self.ax.autoscale_view()   # 自动缩放坐标
        self.fig.canvas.draw()     # 重绘
        plt.pause(pause_time)      # 短暂暂停以刷新窗口
        # gc.collect()               # 手动进行垃圾回收