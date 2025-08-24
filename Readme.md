# Nahida Project

> **NOTE：本游戏尚在开发中。部分素材还没有替换，可能涉及版权问题，如有请留言。**

## 项目简介

Nahida Project 是一个基于 Pygame 的射击游戏项目，玩家需要控制角色躲避敌人的攻击并进行反击。游戏包含多个模块，如子弹管理、角色管理、碰撞检测和事件管理等。

## 文件说明

- `Bullets.py`：定义了各种子弹类及其行为。
- `Characters.py`：定义了玩家和敌人角色类及其行为。
- `Collision_manager.py`：负责碰撞检测和处理。
- `draw_points.py`：用于绘制玩家被击中的位置。
- `Event_manager.py`：管理游戏中的各种事件。
- `Game_class_test_AI.py`：用于测试 AI 角色的游戏逻辑。
- `Game_class_test_boss.py`：用于测试 Boss 角色的游戏逻辑。
- `Game_class_test_bullet.py`：用于测试子弹的游戏逻辑。
- `main.py`：游戏打包时的入口文件。我们通常不从这里启动游戏，而是运行`Game_class`类。
- `Scene_updater.py`：负责更新游戏场景。
- `Settings.py`：包含游戏的配置和常量。

## 安装和运行

1. 克隆项目到本地：
    ```sh
    git clone git@github.com:Teleporter-20001/Nahida-Project.git
    cd Nahida-Project
    ```

2. 安装依赖：
    ```sh
    pip install pygame
    ```

3. 运行游戏：
    ```sh
    python Game_class.py
    ```

## 游戏玩法

- 使用方向键控制角色移动。
- 按住 `Z` 键以持续进行射击。
- 按住 `Shift` 键以从高速移动切换到低速移动，增加操作精细度。
- 躲避敌人的攻击，并击败敌人，获取更高的分数吧！

## 许可证

该项目使用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。**不允许商用。**
