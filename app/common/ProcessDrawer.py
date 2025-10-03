import matplotlib

from app.common.Settings import Settings

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os


class ProcessDrawer:
    def __init__(
        self,
        title="Dynamic Curve",
        xlabel="epoch",
        ylabel="y",
        color="b-",
        labels=None,
        # 下面为“单个 episode 内 step 曲线子图”的可选参数
        step_title="Episode Steps",
        step_xlabel="step",
        step_ylabel="y",
        step_labels=None,
        sharey=False,
    ):

        os.environ['SDL_VIDEO_WINDOW_POS'] = "2560,1200"

        plt.ion()  # 打开交互模式
        # 创建两个子图：左边按 episode，右边按 step
        self.fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=sharey)
        self.ax_episode, self.ax_step = axes

        # -------- 按 episode 曲线的数据结构 --------
        # 支持多条曲线。内部使用 lists 存储每条曲线的 y 数据。
        self.episode_x = []
        # y_series 是一个列表，里面每个元素都是对应曲线的 y 值列表
        self.episode_y_series = []
        # lines 存放 matplotlib Line2D 对象
        self.episode_lines = []

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
        (line0,) = self.ax_episode.plot([], [], color=first_color, label=first_label)
        self.episode_lines.append(line0)
        self.episode_y_series.append([])

        self.ax_episode.set_title(title)
        self.ax_episode.set_xlabel(xlabel)
        self.ax_episode.set_ylabel(ylabel)
        self.ax_episode.legend()

        # -------- 按 step 曲线的数据结构（单个 episode 内） --------
        self.step_x = []
        self.step_y_series = []
        self.step_lines = []
        self.step_labels = list(step_labels) if step_labels is not None else None

        step_first_label = self.step_labels[0] if self.step_labels and len(self.step_labels) > 0 else "y0"
        step_first_color = self.palette[0] if self.palette else self.default_style
        (sline0,) = self.ax_step.plot([], [], color=step_first_color, label=step_first_label)
        self.step_lines.append(sline0)
        self.step_y_series.append([])

        self.ax_step.set_title(step_title)
        self.ax_step.set_xlabel(step_xlabel)
        self.ax_step.set_ylabel(step_ylabel)
        self.ax_step.legend()

        os.environ.pop('SDL_VIDEO_WINDOW_POS', None)

    # -------- 兼容旧属性名（保持向后兼容）--------
    @property
    def x_data(self):
        return self.episode_x

    @property
    def y_series(self):
        return self.episode_y_series

    @property
    def lines(self):
        return self.episode_lines

    @property
    def ax(self):
        return self.ax_episode

    def add_1_episode_data(self, x, y, *args):
        """追加数据点。

        接收形式为 add_data(x, y, *args)，其中 y 对应第 0 条曲线，args 对应之后的曲线。
        假设每次调用时 args 的长度保持一致（用户保证）。
        """

        # 首次调用时，如果 args 比当前已存在曲线更多，需要创建额外曲线对象
        total_series = 1 + len(args)

        # 如果当前 y_series 少于 total_series，则创建新的空序列和 line
        while len(self.episode_y_series) < total_series:
            self.episode_y_series.append([])
            # 创建新的 Line2D，自动选择样式
            style = None
            # 如果是第一附加线，尝试使用不同颜色/样式，否则让 matplotlib 自动分配
            # 我们这里不强制样式，保持默认
            # 选择对应的 label（如果提供了 labels 列表）
            idx = len(self.episode_y_series) - 1
            lbl = self.labels[idx] if self.labels and idx < len(self.labels) else f"y{idx}"
            color = self.palette[idx % len(self.palette)] if self.palette else None
            if color:
                (new_line,) = self.ax_episode.plot([], [], color=color, label=lbl)
            else:
                (new_line,) = self.ax_episode.plot([], [], label=lbl)
            self.episode_lines.append(new_line)

        # 如果已有更多系列，则保持它们

        # 追加 x
        self.episode_x.append(x)

        # 将每一条曲线的 y 值追加到对应的序列
        self.episode_y_series[0].append(y)
        for idx, val in enumerate(args, start=1):
            self.episode_y_series[idx].append(val)

    def add_1_step_data(self, step_x, y, *args):
        """追加单个 episode 内的 step 级别数据点。

        形式与 add_1_episode_data 一致：add_1_step_data(step_x, y, *args)。
        注意：建议在每个新 episode 开始时调用 clear_steps() 清空上一个 episode 的 step 曲线。
        """
        total_series = 1 + len(args)
        while len(self.step_y_series) < total_series:
            self.step_y_series.append([])
            idx = len(self.step_y_series) - 1
            lbl = self.step_labels[idx] if self.step_labels and idx < len(self.step_labels) else f"y{idx}"
            color = self.palette[idx % len(self.palette)] if self.palette else None
            if color:
                (new_line,) = self.ax_step.plot([], [], color=color, label=lbl)
            else:
                (new_line,) = self.ax_step.plot([], [], label=lbl)
            self.step_lines.append(new_line)

        self.step_x.append(step_x)
        self.step_y_series[0].append(y)
        for idx, val in enumerate(args, start=1):
            self.step_y_series[idx].append(val)

    def clear(self):
        # 清空按 episode 的数据
        self.episode_x.clear()
        for ys in self.episode_y_series:
            ys.clear()
        # 清空按 step 的数据
        self.clear_steps()

    def clear_steps(self):
        """仅清空当前 episode 的 step 曲线数据。"""
        self.step_x.clear()
        for ys in self.step_y_series:
            ys.clear()

    def update(self):
        """更新图像，但不抢占焦点。会同时更新 episode 子图与 step 子图。"""
        # -------- 同步按 episode 的 lines --------
        if len(self.episode_y_series) != len(self.episode_lines):
            for ln in self.episode_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            self.episode_lines = []
            for i in range(len(self.episode_y_series)):
                lbl = self.labels[i] if self.labels and i < len(self.labels) else f"y{i}"
                color = self.palette[i % len(self.palette)] if self.palette else None
                if color:
                    (ln,) = self.ax_episode.plot([], [], color=color, label=lbl)
                else:
                    (ln,) = self.ax_episode.plot([], [], label=lbl)
                self.episode_lines.append(ln)

        for ln, ys in zip(self.episode_lines, self.episode_y_series):
            ln.set_data(self.episode_x, ys)

        if self.labels and len(self.labels) >= len(self.episode_lines):
            legend_labels_ep = self.labels[: len(self.episode_lines)]
        else:
            legend_labels_ep = [f"y{i}" for i in range(len(self.episode_lines))]
        self.ax_episode.legend(self.episode_lines, legend_labels_ep)
        self.ax_episode.relim()
        self.ax_episode.autoscale_view()

        # -------- 同步按 step 的 lines --------
        if len(self.step_y_series) != len(self.step_lines):
            for ln in self.step_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            self.step_lines = []
            for i in range(len(self.step_y_series)):
                lbl = self.step_labels[i] if self.step_labels and i < len(self.step_labels) else f"y{i}"
                color = self.palette[i % len(self.palette)] if self.palette else None
                if color:
                    (ln,) = self.ax_step.plot([], [], color=color, label=lbl)
                else:
                    (ln,) = self.ax_step.plot([], [], label=lbl)
                self.step_lines.append(ln)

        for ln, ys in zip(self.step_lines, self.step_y_series):
            ln.set_data(self.step_x, ys)

        if self.step_labels and len(self.step_labels) >= len(self.step_lines):
            legend_labels_step = self.step_labels[: len(self.step_lines)]
        else:
            legend_labels_step = [f"y{i}" for i in range(len(self.step_lines))]
        self.ax_step.legend(self.step_lines, legend_labels_step)
        self.ax_step.relim()
        self.ax_step.autoscale_view()

        # 刷新但不抢焦点
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # save image
        if len(self.x_data) % Settings.save_training_curve_period == 0:
            plt.savefig(os.path.join('Agent', 'models', 'trainCurves', f'train_curve_{len(self.x_data):05d}.png'))
