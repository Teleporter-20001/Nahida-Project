from collections.abc import Callable
from collections import deque
from multiprocessing import Process, Manager

import numpy as np

from app.Agent.Brains.BaseBrain import BaseBrain
from app.Agent.Brains.OptUtils.OptDrawer import OptDrawer
from app.Agent.DataStructure import State, Action
from app.common.Settings import Settings
from app.common.utils import printred, printyellow


def _predict_trajectory(xs: np.ndarray, ys: np.ndarray, vxs: np.ndarray, vys: np.ndarray) -> tuple[float, float, float, float, float, float, Callable[[int], tuple[float, float]]]:
    """
    输入历史N帧的观测数据（左边的数据较早），计算现在的加速度，返回所有轨迹参数。
    ---
    Returns:
        (x, y, vx, vy, ax, ay): 轨迹参数
    """
    assert len(xs) == len(ys) == len(vxs) == len(vys), f'invalid predict input length - xs: {len(xs)}, ys: {len(ys)}, vxs: {len(vxs)}, vys: {len(vys)}'
    # avoid none bullet
    if len(xs) > 0 and (np.abs(xs[-1]) < 10 or np.abs(ys[-1]) < 10):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lambda step: (0.0, 0.0)
    
    datalen = len(xs)
    # assert datalen >= 2, f'not enough data to predict acc: {datalen} < 2'
    if datalen <= 2:
        printyellow(f'not enough data to predict acc: {datalen} < 2')
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lambda step: (0.0, 0.0)
    
    # 尝试对子弹突变进行检查并截断
    for i in reversed(range(1, datalen)):
        if np.abs(xs[i] - xs[i-1]) > 10 or np.abs(ys[i] - ys[i-1]) > 10:
            xs = xs[i:]
            ys = ys[i:]
            vxs = vxs[i:]
            vys = vys[i:]
            datalen = len(xs)  # 修复：直接使用截断后数组的长度
            break
    if datalen <= 2:
        # 刚刚发生突变，干脆用上一次的结果，反正延迟一帧也无所谓
        printyellow(f'not enough data to predict acc after truncation: {datalen} < 2')
        datalen = len(xs)
            

    idxs = np.arange(-datalen, -1+1, dtype=int)
    assert len(idxs) == datalen, f'error in calculating idxs when predicting traj: {datalen} != {len(idxs)}'

    ax, ay = None, None
    coeff_vx = np.polyfit(idxs, vxs, 2)
    coeff_vy = np.polyfit(idxs, vys, 2)
    coeff_ax = np.polyder(coeff_vx)
    coeff_ay = np.polyder(coeff_vy)
    ax = coeff_ax[-1]    # 我觉得应该是-1才对
    ay = coeff_ay[-1]

    x_now, y_now, vx_now, vy_now = xs[-1], ys[-1], vxs[-1], vys[-1]
    def traj(step_idx: int):
        """Calculate the trajectory of the object. To be more accurate, the position after `step_idx` steps.

        Args:
            step_idx (int): The index of the step in the trajectory.
        Returns:
            (x, y): The position after `step_idx` steps.
        """
        nonlocal x_now, y_now, vx_now, vy_now, ax, ay
        delta_t = 1.0 / Settings.FPS
        if ax is None or ay is None:
            if datalen == 2:
                ax = 0
                ay = 0
            else:
                ax = (vxs[-1] - vxs[-2]) / (datalen - 2)
                ay = (vys[-1] - vys[-2]) / (datalen - 2)
        # 使用正确的运动学公式：x = x0 + v0*t + 0.5*a*t²
        t = delta_t * step_idx  # 总时间
        x_new = x_now + vx_now * t + 0.5 * ax * t * t
        y_new = y_now + vy_now * t + 0.5 * ay * t * t
        return x_new, y_new
    return x_now, y_now, vx_now, vy_now, ax, ay, traj


class OptBrain(BaseBrain):

    def __init__(self, memory_len: int = 10, predict_len: int = 10, consider_bullets_num: int = 10):
        super().__init__()
        
        self.memory_len = memory_len
        self.predict_len = predict_len
        self.consider_bullets_num = consider_bullets_num
        self._mem_boss_xs: deque[float] = deque(maxlen=memory_len)
        self._mem_boss_ys: deque[float] = deque(maxlen=memory_len)
        self._mem_boss_vxs: deque[float] = deque(maxlen=memory_len)
        self._mem_boss_vys: deque[float] = deque(maxlen=memory_len)
        self._mem_bullets_xs: list[deque[float]] = [deque(maxlen=memory_len) for _ in range(consider_bullets_num)]
        self._mem_bullets_ys: list[deque[float]] = [deque(maxlen=memory_len) for _ in range(consider_bullets_num)]
        self._mem_bullets_vxs: list[deque[float]] = [deque(maxlen=memory_len) for _ in range(consider_bullets_num)]
        self._mem_bullets_vys: list[deque[float]] = [deque(maxlen=memory_len) for _ in range(consider_bullets_num)]
        self._mem_bullets_rads: list[deque[int]] = [deque(maxlen=memory_len) for _ in range(consider_bullets_num)]
        
        self.debug_drawer: OptDrawer = OptDrawer.instance()
        self.debug_drawer.set_predict_len(self.predict_len)
        self.debug_drawer.start_window()

    def decide_action(self, state: State) -> Action:

        obs = state.observation
        if obs is None:
            printred('error: empty observation')
            return Action.NOMOVE
        
        # step 0: maintain memory
        player_x = obs['player_x']
        player_y = obs['player_y']
        boss_from_player_x = obs['boss_from_player_x']
        boss_from_player_y = obs['boss_from_player_y']
        boss_velx = obs['boss_velx']
        boss_vely = obs['boss_vely']
        nearest_bullets = obs['nearest_bullets']  # shape (k,5) array of nearest enemy bullets, each row is (x, y(relative to player), vx, vy, radius)
        
        self._mem_boss_xs.append(boss_from_player_x)
        self._mem_boss_ys.append(boss_from_player_y)
        self._mem_boss_vxs.append(boss_velx)
        self._mem_boss_vys.append(boss_vely)
        for i in range(self.consider_bullets_num):
            relx, rely, vx, vy, _r = nearest_bullets[i]
            self._mem_bullets_xs[i].append(relx)
            self._mem_bullets_ys[i].append(rely)
            self._mem_bullets_vxs[i].append(vx)
            self._mem_bullets_vys[i].append(vy)
            self._mem_bullets_rads[i].append(_r)

        # opt step 1: predict boss and bullets' trajectories
        boss_traj_params = _predict_trajectory(
            np.array(self._mem_boss_xs),
            np.array(self._mem_boss_ys),
            np.array(self._mem_boss_vxs),
            np.array(self._mem_boss_vys)
        )
        boss_accx, boss_accy = boss_traj_params[4], boss_traj_params[5]
        # boss_traj = boss_traj_params[6]
        bullet_trajs_params = []
        for i in range(self.consider_bullets_num):
            bullet_traj_param = _predict_trajectory(
                np.array(self._mem_bullets_xs[i]),
                np.array(self._mem_bullets_ys[i]),
                np.array(self._mem_bullets_vxs[i]),
                np.array(self._mem_bullets_vys[i])
            )   # each row: (x, y, vx, vy, ax, ay, traj)
            bullet_trajs_params.append(bullet_traj_param)
        
            
        self.debug_drawer.write_player_data(player_x, player_y)
        self.debug_drawer.write_boss_data(boss_from_player_x, boss_from_player_y, boss_velx, boss_vely, 0.0, 0.0)
        for i in range(self.consider_bullets_num):
            relx = self._mem_bullets_xs[i][-1] if len(self._mem_bullets_xs[i]) > 0 else 0.0
            rely = self._mem_bullets_ys[i][-1] if len(self._mem_bullets_ys[i]) > 0 else 0.0
            vx = self._mem_bullets_vxs[i][-1] if len(self._mem_bullets_vxs[i]) > 0 else 0.0
            vy = self._mem_bullets_vys[i][-1] if len(self._mem_bullets_vys[i]) > 0 else 0.0
            ax = (bullet_trajs_params[i][4] if len(bullet_trajs_params[i]) > 1 else 0.0)
            ay = (bullet_trajs_params[i][5] if len(bullet_trajs_params[i]) > 1 else 0.0)
            r = self._mem_bullets_rads[i][-1] if len(self._mem_bullets_rads[i]) > 0 else 10
            self.debug_drawer.write_bullet_data(i, relx, rely, vx, vy, ax, ay, r)
        self.debug_drawer.draw()
            
            
        return Action.NOMOVE

    def __del__(self):
        self.debug_drawer.stop_window()
