import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os

class ProcessDrawer:
    def __init__(self, title="Dynamic Curve", xlabel="epoch", ylabel="y", color="b-", labels=None):

        os.environ['SDL_VIDEO_WINDOW_POS'] = "1000,200"

        plt.ion()  # 打开交互模式
        self.fig, self.ax = plt.subplots()

    # 支持多条曲线。内部使用 lists 存储每条曲线的 y 数据。
        self.x_data = []
        # y_series 是一个列表，里面每个元素都是对应曲线的 y 值列表
        self.y_series = []
        # lines 存放 matplotlib Line2D 对象
        self.lines = []

        # 预定义常见颜色调色板，会按索引分配给每条线
        self.palette = [
            'b',  # blue
            'g',  # green
            'r',  # red
            'c',  # cyan
            'm',  # magenta
            'y',  # yellow
            'k',  # black
            'orange',
            'purple',
        ]

        # 支持用户传入单个颜色/样式字符串，也支持默认自动生成颜色
        # 如果传入的是单个样式字符串，我们把它当作第一条曲线的样式
        self.default_style = color
        # labels 可以是 None 或者一个字符串列表，用于每条曲线的图例标签
        # 如果用户提供 labels，则在创建曲线时使用对应的标签
        self.labels = list(labels) if labels is not None else None

        # 默认只创建一条空线，后续在第一次 add_data 时按实际曲线数目调整
        first_label = self.labels[0] if self.labels and len(self.labels) > 0 else "y0"
        first_color = self.palette[0] if self.palette else self.default_style
        (line,) = self.ax.plot([], [], color=first_color, label=first_label)
        self.lines.append(line)
        self.y_series.append([])

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()

        os.environ.pop('SDL_VIDEO_WINDOW_POS', None)

    def add_data(self, x, y, *args):
        """追加数据点。

        接收形式为 add_data(x, y, *args)，其中 y 对应第 0 条曲线，args 对应之后的曲线。
        假设每次调用时 args 的长度保持一致（用户保证）。
        """
        # 首次调用时，如果 args 比当前已存在曲线更多，需要创建额外曲线对象
        total_series = 1 + len(args)

        # 如果当前 y_series 少于 total_series，则创建新的空序列和 line
        while len(self.y_series) < total_series:
            self.y_series.append([])
            # 创建新的 Line2D，自动选择样式
            style = None
            # 如果是第一附加线，尝试使用不同颜色/样式，否则让 matplotlib 自动分配
            # 我们这里不强制样式，保持默认
            # 选择对应的 label（如果提供了 labels 列表）
            idx = len(self.y_series) - 1
            lbl = self.labels[idx] if self.labels and idx < len(self.labels) else f"y{idx}"
            color = self.palette[idx % len(self.palette)] if self.palette else None
            if color:
                (new_line,) = self.ax.plot([], [], color=color, label=lbl)
            else:
                (new_line,) = self.ax.plot([], [], label=lbl)
            self.lines.append(new_line)

        # 如果已有更多系列，则保持它们

        # 追加 x
        self.x_data.append(x)

        # 将每一条曲线的 y 值追加到对应的序列
        self.y_series[0].append(y)
        for idx, val in enumerate(args, start=1):
            self.y_series[idx].append(val)

    def clear(self):
        self.x_data.clear()
        for ys in self.y_series:
            ys.clear()

    def update(self):
        """更新图像，但不抢占焦点"""
        # 如果曲线数量和 line 数量不一致，修正图例和 lines
        # 确保每条 y_series 都有对应的 Line2D
        if len(self.y_series) != len(self.lines):
            # 移除所有旧的 lines 并重建（简单粗暴但可靠）
            for ln in self.lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            self.lines = []
            for i in range(len(self.y_series)):
                lbl = self.labels[i] if self.labels and i < len(self.labels) else f"y{i}"
                color = self.palette[i % len(self.palette)] if self.palette else None
                if color:
                    (ln,) = self.ax.plot([], [], color=color, label=lbl)
                else:
                    (ln,) = self.ax.plot([], [], label=lbl)
                self.lines.append(ln)

        # 更新每条线的数据
        for ln, ys in zip(self.lines, self.y_series):
            ln.set_data(self.x_data, ys)

        # 始终重建图例，确保标签与当前 lines 对应
        if self.labels and len(self.labels) >= len(self.lines):
            legend_labels = self.labels[: len(self.lines)]
        else:
            legend_labels = [f"y{i}" for i in range(len(self.lines))]
        # 使用当前 lines 与 legend_labels 重建图例
        self.ax.legend(self.lines, legend_labels)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()   # 标记需要重绘，不立即抢焦点
        self.fig.canvas.flush_events()  # 刷新事件队列，但不 raise 窗口
