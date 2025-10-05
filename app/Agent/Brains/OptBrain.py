from collections.abc import Callable
from collections import deque
from multiprocessing import Process, Manager

import numpy as np

from app.characters.Nahida import Nahida

# Try to import scipy optimization functions
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from app.Agent.Brains.BaseBrain import BaseBrain
from app.Agent.Brains.OptUtils.OptDrawer import OptDrawer
from app.Agent.DataStructure import State, Action
from app.common.Settings import Settings
from app.common.utils import printgreen, printred, printyellow


def _predict_trajectory(xs: np.ndarray, ys: np.ndarray, vxs: np.ndarray, vys: np.ndarray) -> tuple[float, float, float, float, float, float, Callable[[int], tuple[float, float]]]:
    """
    输入历史N帧的观测数据（左边的数据较早），计算现在的加速度，返回所有轨迹参数。
    ---
    Returns:
        (x, y, vx, vy, ax, ay): 轨迹参数
    """
    assert len(xs) == len(ys) == len(vxs) == len(vys), f'invalid predict input length - xs: {len(xs)}, ys: {len(ys)}, vxs: {len(vxs)}, vys: {len(vys)}'
    # avoid none bullet
    if len(xs) > 0 and np.hypot(xs[-1], ys[-1]) < 9:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lambda step: (0.0, 0.0)
    
    datalen = len(xs)
    # assert datalen >= 2, f'not enough data to predict acc: {datalen} < 2'
    if datalen < 2:
        # printyellow(f'not enough data to predict acc: {datalen} < 2')
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lambda step: (0.0, 0.0)
    
    # 尝试对子弹突变进行检查并截断
    MAX_SPEED = 500.0  # 每帧最大速度，超过这个速度就认为是突变
    MAX_MOVEMENT_PER_FRAME = MAX_SPEED / Settings.FPS  # 每帧最大移动距离
    for i in reversed(range(1, datalen)):
        if np.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]) > MAX_MOVEMENT_PER_FRAME:
            xs = xs[i:]
            ys = ys[i:]
            vxs = vxs[i:]
            vys = vys[i:]
            datalen = len(xs)  # 修复：直接使用截断后数组的长度
            break
    
    if datalen == 2:
        dt = 1.0 / Settings.FPS
        ax = (vxs[-1] - vxs[-2]) / dt
        ay = (vys[-1] - vys[-2]) / dt
    elif datalen < 2:
        # 刚刚发生突变，干脆用上一次的结果，反正延迟一帧也无所谓
        # printyellow(f'not enough data to predict acc after truncation: {datalen} < 2')

        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lambda step: (0.0, 0.0)
    else:
        dt = 1.0 / Settings.FPS
        times = np.arange(-datalen * dt, (-1+1) * dt, step=dt, dtype=np.float32)
        assert len(times) == datalen, f'error in calculating times when predicting traj: {datalen} != {len(times)}'

        ax, ay = None, None
        coeff_vx = np.polyfit(times, vxs, 2)
        coeff_vy = np.polyfit(times, vys, 2)
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

    def __init__(self, memory_len: int = 10, predict_len: int = 10, action_predict_len: int = 10, consider_bullets_num: int = 10):
        """Initialize the OptBrain.

        Args:
            memory_len (int, optional): The length of the memory. Defaults to 10.
            predict_len (int, optional): The length of the prediction for boss and enemy bullets. Defaults to 10.
            action_predict_len (int, optional): The length of player action prediction. Defaults to 10.
            consider_bullets_num (int, optional): The number of bullets to consider. Defaults to 10.
        """
        assert action_predict_len <= predict_len, f'action_predict_len {action_predict_len} should be less than or equal to predict_len {predict_len}'
        super().__init__()
        
        self.memory_len = memory_len
        self.predict_len = predict_len
        self.action_predict_len = action_predict_len
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
        
        # for step 2
        self._bullets_waypoints: np.ndarray = np.zeros((consider_bullets_num, predict_len, 2), dtype=np.float32)  # 记录每个子弹在未来predict_len帧内的轨迹点
        self._bullets_waypoints_calced: bool = False  # 标记子弹轨迹点是否已经计算过，避免重复计算
        self._boss_waypoints: np.ndarray = np.zeros((predict_len, 2), dtype=np.float32)  # 记录boss在未来predict_len帧内的轨迹点
        self._boss_waypoints_calced: bool = False  # 标记boss轨迹点是否已经计算过，避免重复计算
        self._rough_collision_weights: np.ndarray = np.array(list(reversed(np.exp(-np.linspace(0, predict_len-1, predict_len) / (predict_len / 1)))))  # 指数衰减的权重，用于粗略碰撞检测的加权
        self._bullet_danger_zone_radius = Settings.player_radius + 15 + 15  # 粗略碰撞检测的危险区域半径
        self._last_target_pos: tuple[float, float] = (Settings.window_width / 2, Settings.window_height * 3 / 4)  # 上一帧的目标位置，初始为屏幕中央偏下
        self._target_pos: tuple[float, float] = self._last_target_pos  # 当前帧的目标位置，初始为上一帧的位置
        
        # for step 3
        self.beam_width = 40  # Beam search的beam宽度
        self.early_stop_threshold = 1.0 / np.Infinity  # 提前停止的阈值，如果找到成本很低的解就提前停止
        

    def decide_action(self, state: State) -> Action:

        obs = state.observation
        if obs is None:
            printred('error: empty observation')
            return Action.NOMOVE

        # ----------------------------------------------------------------------------
        # step 0: maintain memory
        player_x = obs['player_x']
        player_y = obs['player_y']
        boss_from_player_x = obs['boss_from_player_x']
        boss_from_player_y = obs['boss_from_player_y']
        boss_velx = obs['boss_velx']
        boss_vely = obs['boss_vely']
        nearest_bullets = obs['nearest_bullets']  # shape (k,5) array of nearest enemy bullets, each row is (x, y(relative to player), vx, vy, radius)
        
        # Store current player position for coordinate conversion
        self._current_player_x = player_x
        self._current_player_y = player_y
        
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
            
        self._bullets_waypoints_calced = False  # 新的一帧，子弹轨迹点需要重新计算
        self._boss_waypoints_calced = False  # 新的一帧，boss轨迹点需要重新计算

        # ----------------------------------------------------------------------------
        # opt step 1: predict boss and bullets' trajectories
        boss_traj_params = _predict_trajectory(
            np.array(self._mem_boss_xs),
            np.array(self._mem_boss_ys),
            np.array(self._mem_boss_vxs),
            np.array(self._mem_boss_vys)
        )
        boss_accx, boss_accy = boss_traj_params[4], boss_traj_params[5]
        boss_traj = boss_traj_params[6]
        bullet_trajs_params = []
        for i in range(self.consider_bullets_num):
            bullet_traj_param = _predict_trajectory(
                np.array(self._mem_bullets_xs[i]),
                np.array(self._mem_bullets_ys[i]),
                np.array(self._mem_bullets_vxs[i]),
                np.array(self._mem_bullets_vys[i])
            )   # each row: (x, y, vx, vy, ax, ay, traj)
            bullet_trajs_params.append(bullet_traj_param)
        bullet_trajs = [params[6] for params in bullet_trajs_params]
            
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
        
        # -------------------------------------------------------------------------
        # opt step 2: decide the best target position to go to
        weight_prefer, weight_collision, weight_smooth = 12, 0.04, 0.5
        
        # Define the combined objective function
        def objective_function_step2(pos):
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                target_pos = (float(pos[0]), float(pos[1]))
            else:
                target_pos = (float(pos[0]), float(pos[1]))
            
            # Calculate individual cost components
            cost_prefer = self._J_goal(target_pos, boss_traj, bullet_trajs)
            cost_collision = self._J_collision_rough(target_pos, boss_traj, bullet_trajs)
            cost_smooth = self._J_smooth(target_pos, self._last_target_pos)
            
            # Combined cost with weights
            total_cost = (weight_prefer * cost_prefer + 
                         weight_collision * cost_collision + 
                         weight_smooth * cost_smooth)
            
            return total_cost
        
        # Set bounds for optimization (player must stay within game window)
        x_min, x_max = Settings.player_img_width / 2, Settings.window_width - Settings.player_img_width / 2
        y_min, y_max = Settings.player_img_height / 2, Settings.window_height - Settings.player_img_height / 2
        bounds = [(x_min, x_max), (y_min, y_max)]
        
        # Initialize best position and cost
        # best_pos = self._last_target_pos
        # best_cost = float('inf')
        
        if SCIPY_AVAILABLE:
            # Use scipy optimization methods
            self._target_pos = self._optimize_with_scipy(objective_function_step2, bounds, player_x, player_y)
        else:
            # Use fallback grid search method
            self._target_pos = self._optimize_with_grid_search(objective_function_step2, bounds)
            
        self.debug_drawer.write_target_data(self._target_pos[0], self._target_pos[1])
        
        # Debug information
        # boss_abs_x = player_x + boss_from_player_x
        # boss_abs_y = player_y + boss_from_player_y
        # print(f"Player: ({player_x:.1f}, {player_y:.1f}), Boss: ({boss_abs_x:.1f}, {boss_abs_y:.1f}), Target: ({self._target_pos[0]:.1f}, {self._target_pos[1]:.1f})")
        
        # ----------------------------------------------------------------------------
        # step 3: decide action to move towards target position with MPC
        if self._target_pos is None:
            printred("error: target_pos is None, using NOMOVE")
            return Action.NOMOVE
        
        # 使用Beam Search找到最优行动序列
        try:
            optimal_action_sequence = self._beam_search_action_sequence(
                weight_collision=5.0,    # 避免碰撞的权重
                weight_togoal=1.0,       # 接近目标的权重
                weight_smooth=0.1        # 平滑性权重
            )
            
            # 从最优序列中获取第一个动作
            next_action = self._get_best_action_from_sequence(optimal_action_sequence)
            
            # 调试信息（可选）
            printgreen(
                f"Optimal action sequence cost - \
                    collision: {self._J_collision_rough(self._target_pos, boss_traj, bullet_trajs):.2f}, \
                    togoal: {self._J_goal(self._target_pos, boss_traj, bullet_trajs):.2f}, \
                    smooth: {self._J_smooth(self._target_pos, self._last_target_pos):.2f}")
            printgreen(f"Beam search found sequence: {optimal_action_sequence[:4]}..., first action: {next_action}")
            player_traj = [(player_x, player_y)]
            for idx in optimal_action_sequence:
                action = self._get_action(idx)
                curr_x, curr_y = player_traj[-1][0] + action.xfactor * Nahida.ORIGIN_SPEED / Settings.FPS, player_traj[-1][1] + action.yfactor * Nahida.ORIGIN_SPEED / Settings.FPS
                player_traj.append((curr_x, curr_y))
            self.debug_drawer.write_player_traj(player_traj)

        except Exception as e:
            printyellow(f"Beam search failed, using NOMOVE: {e}")
            next_action = Action.NOMOVE
        
        # step final: maintain memories, such as last target pos
        self.debug_drawer.draw()
        self._last_target_pos = self._target_pos
            
        return next_action
    
    # step 2 helper functions
    def _J_goal(self, target_pos: tuple[float, float], boss_traj: Callable[[int], tuple[float, float]], bullet_trajs: list[Callable[[int], tuple[float, float]]]) -> float:
        """Calculate the goal cost function J for a given target position.

        Args:
            target_pos (tuple[float, float]): The target position (x, y) to evaluate (absolute coordinates).
            boss_traj (Callable[[int], tuple[float, float]]): The predicted trajectory function of the boss (returns relative coordinates).
            bullet_trajs (list[Callable[[int], tuple[float, float]]]): The list of predicted trajectory functions of the bullets.

        Returns:
            float: The calculated goal cost.
        """
        is_out_of_bound = not (Settings.player_img_width / 2 <= target_pos[0] <= Settings.window_width - Settings.player_img_width / 2 and Settings.player_img_height / 2 <= target_pos[1] <= Settings.window_height - Settings.player_img_height / 2)
        if is_out_of_bound:
            return float('inf')  # out of bound is not allowed

        # Calculate the boss target after action_predict_len frames
        boss_target_x, boss_target_y = boss_traj(self.action_predict_len) if boss_traj is not None else (0.0, 0.0)
        boss_target_x += self._current_player_x  # convert to absolute coordinates
        boss_target_y += self._current_player_y  # convert to absolute coordinates

        prefer_target_x = boss_target_x
        prefer_target_y = max(Settings.window_height * 3. / 4, min(boss_target_y + Settings.window_height * 0.4, Settings.window_height - Settings.boss_radius))  # prefer to stay below the boss
        
        weight_x, weight_y = 2.2, 2.2  # x方向的权重较大，y方向的权重较小，表示希望靠近boss的水平位置但保持一定的垂直距离
        cost = weight_x * np.abs(prefer_target_x - target_pos[0]) + weight_y * np.abs(prefer_target_y - target_pos[1])
        return cost
    
    def _J_collision_rough(self, target_pos: tuple[float, float], boss_traj: Callable[[int], tuple[float, float]], bullet_trajs: list[Callable[[int], tuple[float, float]]]) -> float:
        """Calculate the rough collision cost J for a given target position.

        Args:
            target_pos (tuple[float, float]): The target position (x, y) to evaluate.
            boss_traj (Callable[[int], tuple[float, float]]): The predicted trajectory function of the boss.
            bullet_trajs (list[Callable[[int], tuple[float, float]]]): The list of predicted trajectory functions of the bullets.

        Returns:
            float: The calculated collision cost.
        """
        def phi(z: float) -> float:
            """Punishment function for collision cost.

            Args:
                z (float): The distance to the danger zone.

            Returns:
                punishment: The calculated punishment cost.
            """
            gamma = 1.0
            return gamma * max(0, -z) ** 2  # quadratic punishment for being inside the danger zone
            
        # calc bullets and boss trajectories if not calculated
        if not self._bullets_waypoints_calced:
            for i in range(self.consider_bullets_num):
                if i < len(bullet_trajs) and bullet_trajs[i] is not None:
                    for step in range(self.predict_len):
                        self._bullets_waypoints[i, step, :] = bullet_trajs[i](step)
                else:
                    self._bullets_waypoints[i, :, :] = 0.0
            self._bullets_waypoints_calced = True
        if not self._boss_waypoints_calced:
            if boss_traj is not None:
                for step in range(self.predict_len):
                    self._boss_waypoints[step, :] = boss_traj(step)
            else:
                self._boss_waypoints[:, :] = 0.0
            self._boss_waypoints_calced = True
        
        # Rough collision detection: check if the target position is within the danger zone of any bullets
        cost = 0.0
        for i in range(self.consider_bullets_num):
            for step in range(self.predict_len):
                # bullet_waypoints contains relative positions, convert to absolute
                bullet_rel_x, bullet_rel_y = self._bullets_waypoints[i, step, :]
                bullet_abs_x = self._current_player_x + bullet_rel_x
                bullet_abs_y = self._current_player_y + bullet_rel_y
                
                dist_to_bullet = np.hypot(bullet_abs_x - target_pos[0], bullet_abs_y - target_pos[1])
                z = dist_to_bullet - self._bullet_danger_zone_radius
                cost += self._rough_collision_weights[step] * phi(z)
                
        for step in range(self.predict_len):
            # boss_waypoints contains relative positions, convert to absolute
            boss_rel_x, boss_rel_y = self._boss_waypoints[step, :]
            boss_abs_x = self._current_player_x + boss_rel_x
            boss_abs_y = self._current_player_y + boss_rel_y
            
            dist_to_boss = np.hypot(boss_abs_x - target_pos[0], boss_abs_y - target_pos[1])
            z = dist_to_boss - (Settings.player_radius + Settings.boss_radius + 30)  # 留一点余量
            cost += self._rough_collision_weights[step] * phi(z)

        return cost
    
    def _J_smooth(self, target_pos: tuple[float, float], last_target_pos: tuple[float, float]) -> float:
        """Calculate the smoothness cost J for a given target position.

        Args:
            target_pos (tuple[float, float]): The current target position (x, y) to evaluate.
            last_target_pos (tuple[float, float]): The last target position (x, y) from the previous frame.

        Returns:
            float: The calculated smoothness cost.
        """
        def phi(z: float) -> float:
            """Punishment function for smoothness cost.

            Args:
                z (float): The distance between current and last target positions.
            Returns:
                punishment: The calculated punishment cost.
            """
            gamma = 1.0
            return gamma * max(0, z) ** 2  # quadratic punishment for large changes in target position
        THRESHOLD = 600 / Settings.FPS
        return phi(np.hypot(target_pos[0] - last_target_pos[0], target_pos[1] - last_target_pos[1]) - THRESHOLD)

    def _optimize_with_scipy(self, objective_function, bounds, player_x: float, player_y: float) -> tuple[float, float]:
        """Use scipy optimization methods to find the best target position.

        Args:
            objective_function: The objective function to minimize
            bounds: The bounds for optimization [(x_min, x_max), (y_min, y_max)]
            player_x: Current player x position
            player_y: Current player y position

        Returns:
            tuple[float, float]: The optimal target position (x, y)
        """
        best_pos = self._last_target_pos
        best_cost = float('inf')
        
        # Method 1: Local optimization starting from last target position
        try:
            result = minimize(objective_function, 
                            x0=np.array(self._last_target_pos),
                            bounds=bounds,
                            method='L-BFGS-B',
                            options={'maxiter': 20, 'ftol': 1.0})
            if result.success and result.fun < best_cost:
                best_cost = result.fun
                best_pos = (result.x[0], result.x[1])
        except Exception as e:
            printyellow(f"Local optimization failed: {e}")
        
        # Method 2: Try optimization starting from current player position
        try:
            result = minimize(objective_function,
                            x0=np.array([player_x, player_y]),
                            bounds=bounds,
                            method='L-BFGS-B',
                            options={'maxiter': 20, 'ftol': 1.0})
            if result.success and result.fun < best_cost:
                best_cost = result.fun
                best_pos = (result.x[0], result.x[1])
        except Exception as e:
            printyellow(f"Player position optimization failed: {e}")
        
        # Method 3: Global optimization (fallback for complex landscapes)
        if best_cost == float('inf'):
            try:
                result = differential_evolution(objective_function,
                                              bounds=bounds,
                                              maxiter=20,
                                              popsize=10)
                if result.success:
                    best_cost = result.fun
                    best_pos = (result.x[0], result.x[1])
            except Exception as e:
                printred(f"Global optimization failed: {e}")
                # Use safe fallback position
                best_pos = (Settings.window_width / 2, Settings.window_height * 3 / 4)
        
        return best_pos

    def _optimize_with_grid_search(self, objective_function, bounds) -> tuple[float, float]:
        """Use grid search as a fallback optimization method when scipy is not available.

        Args:
            objective_function: The objective function to minimize
            bounds: The bounds for optimization [(x_min, x_max), (y_min, y_max)]

        Returns:
            tuple[float, float]: The optimal target position (x, y)
        """
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        
        # Create a coarse grid for searching
        grid_size = 15  # 15x15 grid for balance between speed and accuracy
        x_candidates = np.linspace(x_min, x_max, grid_size)
        y_candidates = np.linspace(y_min, y_max, grid_size)
        
        best_pos = self._last_target_pos
        best_cost = float('inf')
        
        # Grid search
        for x in x_candidates:
            for y in y_candidates:
                try:
                    cost = objective_function([x, y])
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y)
                except Exception:
                    continue  # Skip positions that cause errors
        
        # If grid search fails, do a refined search around the best candidate
        if best_cost < float('inf'):
            # Refine search around the best grid point
            refinement_range = min((x_max - x_min) / grid_size, (y_max - y_min) / grid_size) / 2
            refined_x = np.linspace(max(x_min, best_pos[0] - refinement_range),
                                  min(x_max, best_pos[0] + refinement_range), 5)
            refined_y = np.linspace(max(y_min, best_pos[1] - refinement_range),
                                  min(y_max, best_pos[1] + refinement_range), 5)
            
            for x in refined_x:
                for y in refined_y:
                    try:
                        cost = objective_function([x, y])
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y)
                    except Exception:
                        continue
        
        # Final fallback if everything fails
        if best_cost == float('inf'):
            best_pos = (Settings.window_width / 2, Settings.window_height * 3 / 4)
        
        return best_pos

    # step 3 helper functions
    def _get_action(self, idx: int) -> Action:
        return list(Action)[idx]
    
    def _J_collision(self, action_seq: list[int]) -> float:
        """Calculate the precise collision cost J for a given player position.

        Args:
            action_seq (list[int]): The sequence of actions to evaluate.
        Returns:
            cost: The calculated collision cost.
        """
        # We think that bullets' waypoints have been calculated in step 2
        def phi(z: float) -> float:
            gamma = 1.0
            return gamma * max(0, -z) ** 2  # quadratic punishment for being inside the danger zone
        cost = 0.0
        player_pos = (self._current_player_x, self._current_player_y)
        SAFETY_THRESH = Settings.player_radius + 15 + 15
        for step in range(len(action_seq)):
            _action = self._get_action(action_seq[step])
            player_pos = tuple(np.array(player_pos) + (np.array([_action.xfactor, _action.yfactor]) * Nahida.ORIGIN_SPEED / Settings.FPS))
            for i in range(self.consider_bullets_num):
                bullet_rel_x, bullet_rel_y = self._bullets_waypoints[i, step, :]
                bullet_abs_x, bullet_abs_y = self._current_player_x + bullet_rel_x, self._current_player_y + bullet_rel_y
                z = np.hypot(bullet_abs_x - player_pos[0], bullet_abs_y - player_pos[1]) - SAFETY_THRESH
                cost += phi(z)
            boss_rel_x, boss_rel_y = self._boss_waypoints[step, :]
            boss_abs_x, boss_abs_y = self._current_player_x + boss_rel_x, self._current_player_y + boss_rel_y
            z = np.hypot(boss_abs_x - player_pos[0], boss_abs_y - player_pos[1]) - (Settings.player_radius + Settings.boss_radius + 30)
            cost += phi(z)
        return cost
    
    def _J_action_togoal(self, action_seq: list[int]) -> float:
        """Calculate the precise cost J to encourage reaching the goal.

        Args:
            action_seq (list[int]): The sequence of actions to evaluate.

        Returns:
            float: The calculated cost to reach the goal.
        """
        cost = 0.0
        player_pos = (self._current_player_x, self._current_player_y)
        # player_to_target_vec = np.array(self._target_pos) - np.array(player_pos)
        # player_to_target_vec_unit = player_to_target_vec / (np.linalg.norm(player_to_target_vec) + 1e-3)
        for step in range(len(action_seq)):
            _action = self._get_action(action_seq[step])
            player_pos = tuple(np.array(player_pos) + (np.array([_action.xfactor, _action.yfactor]) * Nahida.ORIGIN_SPEED / Settings.FPS))
            cost += np.hypot(self._target_pos[0] - player_pos[0], self._target_pos[1] - player_pos[1]) ** 2  # 希望每一步都接近目标位置
        #     action_vec = np.array([_action.xfactor, _action.yfactor])
        #     action_vec_unit = action_vec / (np.linalg.norm(action_vec) + 1e-3) if _action != Action.NOMOVE else np.dot(np.array([[np.cos(np.pi/2), np.sin(np.pi/2)], [-np.sin(np.pi/2), np.cos(np.pi/2)]]), player_to_target_vec_unit)
        #     cost += -np.cos(np.dot(player_to_target_vec_unit, action_vec_unit)) + 1  # 希望action_vec和player_to_target_vec方向一致
        # cost += 0.03 * np.hypot(player_pos[0] - self._target_pos[0], player_pos[1] - self._target_pos[1])  # 希望最终位置接近目标位置
        return cost
    
    def _J_action_smooth(self, action_seq: list[int]) -> float:
        """Calculate the smoothness cost J for a given action sequence.

        Args:
            action_seq (list[int]): The sequence of actions to evaluate.
        Returns:
            cost: The calculated smoothness cost.
        """
        cost = 0.0
        if len(action_seq) < 2:
            return cost
        for step in range(1, len(action_seq)):
            prev_action = self._get_action(action_seq[step - 1])
            curr_action = self._get_action(action_seq[step])
            if prev_action == Action.NOMOVE or curr_action == Action.NOMOVE:
                continue
            cost += (np.arctan2(curr_action.yfactor, curr_action.xfactor) - np.arctan2(prev_action.yfactor, prev_action.xfactor)) ** 2
        return cost

    def _beam_search_action_sequence(self, weight_collision: float = 1.0, weight_togoal: float = 0.5, weight_smooth: float = 0.1) -> list[int]:
        """Use Beam Search to find the optimal action sequence for step 3.

        Args:
            weight_collision (float): Weight for collision cost
            weight_togoal (float): Weight for reaching target goal cost
            weight_smooth (float): Weight for smoothness cost

        Returns:
            list[int]: The optimal action sequence as a list of action indices
        """
        # 定义组合目标函数
        def combined_cost(action_seq: list[int]) -> float:
            """Calculate the combined cost for an action sequence."""
            try:
                cost_collision = self._J_collision(action_seq)
                cost_togoal = self._J_action_togoal(action_seq)
                cost_smooth = self._J_action_smooth(action_seq)
                
                total_cost = (weight_collision * cost_collision + 
                            weight_togoal * cost_togoal + 
                            weight_smooth * cost_smooth)
                return total_cost
            except Exception as e:
                # 如果计算失败，返回一个很大的惩罚值
                return float('inf')

        # Beam Search实现
        num_actions = len(Action)  # 9个可能的动作
        
        # 初始化：第一层的所有可能动作
        # 每个beam元素是一个tuple: (action_sequence, cumulative_cost)
        current_beams = []
        for action_idx in range(num_actions):
            try:
                cost = combined_cost([action_idx])
                if cost < float('inf'):
                    current_beams.append(([action_idx], cost))
            except Exception:
                continue
        
        # 如果第一层就没有有效的动作，返回默认动作序列
        if not current_beams:
            printred("Beam search found no valid initial actions, returning NOMOVE sequence.")
            return [list(Action).index(Action.NOMOVE) for _ in range(self.action_predict_len)]

        # 按成本排序并保留前beam_width个
        current_beams.sort(key=lambda x: x[1])
        current_beams = current_beams[:self.beam_width]
        
        # 逐层扩展beam search
        for depth in range(1, self.action_predict_len):
            next_beams = []
            
            # 对当前层的每个beam进行扩展
            for beam_seq, beam_cost in current_beams:
                # 尝试添加每个可能的下一个动作
                for action_idx in range(num_actions):
                    new_seq = beam_seq + [action_idx]
                    try:
                        new_cost = combined_cost(new_seq)
                        if new_cost < float('inf'):
                            next_beams.append((new_seq, new_cost))
                    except Exception:
                        continue
            
            # 如果没有有效的扩展，使用当前最优序列并用NOMOVE填充
            if not next_beams:
                best_seq = current_beams[0][0]
                while len(best_seq) < self.action_predict_len:
                    printyellow(f"Beam search terminated early at depth {depth} due to no valid expansions.")
                    best_seq.append(list(Action).index(Action.NOMOVE))
                return best_seq
            
            # 按成本排序并保留前beam_width个
            next_beams.sort(key=lambda x: x[1])
            current_beams = next_beams[:self.beam_width]
            
            # 早期停止检查：如果找到了非常好的解，可以提前结束
            if len(current_beams) > 0 and current_beams[0][1] < self.early_stop_threshold:
                best_seq = current_beams[0][0]
                while len(best_seq) < self.action_predict_len:
                    printgreen(f"Beam search early stopping at depth {depth} with cost {current_beams[0][1]:.6f}")
                    best_seq.append(list(Action).index(Action.NOMOVE))
                return best_seq
        
        # 返回最优序列
        if current_beams:
            best_sequence = current_beams[0][0]
            return best_sequence
        else:
            # 备用方案：返回全NOMOVE序列
            printred("Beam search found no valid action sequences, returning NOMOVE sequence.")
            return [list(Action).index(Action.NOMOVE) for _ in range(self.action_predict_len)]

    def _get_best_action_from_sequence(self, action_sequence: list[int]) -> Action:
        """Get the first action from the optimal action sequence.

        Args:
            action_sequence (list[int]): The optimal action sequence

        Returns:
            Action: The first action to take
        """
        if not action_sequence:
            return Action.NOMOVE
        
        try:
            action_idx = action_sequence[0]
            actions_list = list(Action)
            if 0 <= action_idx < len(actions_list):
                return actions_list[action_idx]
            else:
                return Action.NOMOVE
        except Exception:
            return Action.NOMOVE

    def __del__(self):
        self.debug_drawer.stop_window()
