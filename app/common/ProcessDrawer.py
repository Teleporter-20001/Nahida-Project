import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os

class ProcessDrawer:
    def __init__(self, title="Dynamic Curve", xlabel="epoch", ylabel="y", color="b-"):

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

    def update(self):
        """更新图像，但不抢占焦点"""
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()   # 标记需要重绘，不立即抢焦点
        self.fig.canvas.flush_events()  # 刷新事件队列，但不 raise 窗口
