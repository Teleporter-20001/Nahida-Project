from Game_class_test_collision import *
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


if __name__ == "__main__":

    gameobj = Game()

    while gameobj.event_manager.running:
        gameobj.event_manager.event_handle()
        if not gameobj.event_manager.is_pause:
            gameobj.collision_manager.collision_handle()
            gameobj.scene_updater.game_update()
    
    gameobj.gameover()
