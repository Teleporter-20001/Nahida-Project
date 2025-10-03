from dataclasses import dataclass

import pygame
import typing
import json
import atexit
import os


class Settings:
    """Singleton Settings with a Qt-based editor dialog on initialization.

    On creation, this class attempts to open a Qt window that allows editing
    the configurable attributes. If Qt is unavailable, it falls back to a
    console prompt-based editor. Save operations display a success or
    failure message to the user.

    Usage:
        s = Settings()  # always returns the singleton instance
    """

    # default values
    FPS: int = 60
    window_width: int = 600
    window_height: int = 800
    window_background_color: pygame.color.Color = pygame.Color(80, 160, 150)

    # related to reward
    alive_reward: float = 0.055
    BORDER_BUFFER: int = 60
    BORDER_PUNISH: float = -0.4
    hit_reward: float = 0.6
    kill_boss_reward: float = 80.0
    behit_reward: float = -100.0
    avoid_reward: float = 0.2

    # related to training
    batch_size: int = 256
    gamma: float = 0.991
    lr: float = 1e-5
    target_update: int = 1500
    net_name = 'RecurrentQNet'

    # related to training control
    render: bool = True
    begin_episode: int = 0
    epsilon_begin: float = 1.0
    epsilon_end: float = 0.001
    epsilon_decay: float = 0.9999
    repeat_period: int = 30000
    save_model_period: int = 100
    save_training_curve_period: int = 50
    teach_mode: bool = False

    # singleton instance
    _instance: typing.Optional['Settings'] = None
    # references to Qt objects to keep windows/app alive
    _qt_app: typing.Optional[object] = None
    _qt_dialogs: typing.List[object] = []
    _qt_thread: typing.Optional[object] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            # initialize instance attributes from class defaults
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v) and not isinstance(v, classmethod):
                    try:
                        setattr(cls._instance, k, v)
                    except Exception:
                        pass
            # try to load persisted settings from JSON (app/common/settings.json)
            load_settings: bool = False
            try:
                # settings file is located in the same directory as this file
                if load_settings:
                    base_dir = os.path.dirname(__file__)
                    cls._settings_path = os.path.join(base_dir, 'settings.json')
                    if os.path.exists(cls._settings_path):
                        cls._instance._load_from_json(cls._settings_path)
            except Exception as e:
                print(f"Failed to load settings.json: {e}")

            # after attribute population and load, attempt to show editor
            show_editor: bool = False
            try:
                if show_editor:
                    cls._instance._open_editor()
                pass
            except Exception as e:
                # editor failure should not crash; print for visibility
                print(f"Settings editor failed to open: {e}")

            # register atexit save handler to persist settings on interpreter exit
            try:
                atexit.register(cls._instance._save_on_exit)
            except Exception:
                pass
        return cls._instance

    def _to_primitive(self, v):
        """Convert a setting value to a JSON-serializable primitive."""
        if isinstance(v, pygame.color.Color):
            return {'__type__': 'Color', 'r': v.r, 'g': v.g, 'b': v.b, 'a': v.a}
        if isinstance(v, (int, float, bool, str)):
            return v
        # fallback to string
        return str(v)

    def _from_primitive(self, v):
        """Reverse of _to_primitive: convert stored JSON value back to python object."""
        if isinstance(v, dict) and v.get('__type__') == 'Color':
            return pygame.Color(v.get('r', 0), v.get('g', 0), v.get('b', 0), v.get('a', 255))
        return v

    def _load_from_json(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # data is expected to be a dict of name -> primitive
            for k, raw in data.items():
                if hasattr(self, k):
                    try:
                        val = self._from_primitive(raw)
                        setattr(self, k, val)
                        try:
                            setattr(type(self), k, val)
                        except Exception:
                            pass
                    except Exception:
                        # ignore individual failures
                        pass
        except Exception as e:
            raise

    def _save_to_json(self, path: str):
        data = {}
        # collect current editable attributes
        keys = [
            'FPS', 'window_width', 'window_height', 'window_background_color',
            'alive_reward', 'BORDER_BUFFER', 'BORDER_PUNISH', 'hit_reward',
            'kill_boss_reward', 'behit_reward', 'avoid_reward', 'render', 'begin_episode',
            'epsilon_begin', 'epsilon_end', 'epsilon_decay', 'repeat_period', 'teach_mode',
            'batch_size', 'lr', 'gamma', 'target_update'
        ]
        for k in keys:
            if hasattr(self, k):
                try:
                    data[k] = self._to_primitive(getattr(self, k))
                except Exception:
                    data[k] = None
        # ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_on_exit(self):
        try:
            path = getattr(self, '_settings_path', None)
            if path is None:
                base_dir = os.path.dirname(__file__)
                path = os.path.join(base_dir, 'settings.json')
            self._save_to_json(path)
        except Exception as e:
            print(f"Failed to save settings.json on exit: {e}")

    def _open_editor(self):
        """Try to open a Qt dialog to edit settings; fallback to console."""
        # prefer PyQt5, fall back to other bindings
        qt = None
        for mod in ("PyQt5", "PyQt6", "PySide2", "PySide6"):
            try:
                qt = __import__(mod)
                binding = mod
                break
            except Exception:
                qt = None
        if qt is None:
            # no Qt available, fallback to console editor
            self._console_editor()
            return

        # import the required Qt widgets in a binding-agnostic way
        try:
            if binding == 'PyQt5':
                from PyQt5 import QtWidgets, QtCore
            elif binding == 'PyQt6':
                from PyQt6 import QtWidgets, QtCore
            elif binding == 'PySide2':
                from PySide2 import QtWidgets, QtCore
            else:
                from PySide6 import QtWidgets, QtCore
        except Exception as e:
            print(f"Failed to import Qt widgets: {e}")
            self._console_editor()
            return

        # Build the dialog
        class SettingsDialog(QtWidgets.QDialog):
            def __init__(self, settings_obj: 'Settings'):
                super().__init__()
                self.setWindowTitle('Edit Settings')
                self.settings = settings_obj
                self.layout = QtWidgets.QFormLayout(self)

                self._fields = {}

                # map of attribute -> (widget, type)
                for name, val in self._iter_editable():
                    w = QtWidgets.QLineEdit(self)
                    w.setText(self._to_string(val))
                    self.layout.addRow(name, w)
                    self._fields[name] = (w, type(val))

                btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
                # get the actual button objects so we can wire Save to not close the dialog
                save_btn = btn_box.button(QtWidgets.QDialogButtonBox.Save)
                cancel_btn = btn_box.button(QtWidgets.QDialogButtonBox.Cancel)
                if save_btn is not None:
                    save_btn.clicked.connect(self.save)
                if cancel_btn is not None:
                    cancel_btn.clicked.connect(self.reject)
                self.layout.addRow(btn_box)

            def _iter_editable(self):
                # yield attributes we allow editing
                keys = [
                    'FPS', 'window_width', 'window_height', 'window_background_color',
                    'alive_reward', 'BORDER_BUFFER', 'BORDER_PUNISH', 'hit_reward',
                    'kill_boss_reward', 'behit_reward', 'avoid_reward', 'render', 'begin_episode',
                    'epsilon_begin', 'epsilon_end', 'epsilon_decay', 'repeat_period',
                    'teach_mode', 'batch_size', 'lr', 'gamma', 'target_update'
                ]
                for k in keys:
                    if hasattr(self.settings, k):
                        yield k, getattr(self.settings, k)

            def _to_string(self, v):
                if isinstance(v, pygame.color.Color):
                    return f"{v.r},{v.g},{v.b}"
                if isinstance(v, bool):
                    return 'True' if v else 'False'
                return str(v)

            def save(self):
                # try to parse and assign all fields
                try:
                    for name, (widget, typ) in self._fields.items():
                        text = widget.text().strip()
                        val = self._parse_value(text, typ)
                        setattr(self.settings, name, val)
                        try:
                            setattr(type(self.settings), name, val)
                        except Exception:
                            pass
                    # keep dialog open after saving; show non-blocking message
                    try:
                        QtWidgets.QMessageBox.information(self, 'Success', 'Settings saved successfully.')
                    except Exception:
                        # some bindings/environments may not allow message boxes here; ignore
                        pass
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to save settings:\n{e}')

            def _parse_value(self, text: str, typ: type):
                if typ is bool:
                    return text.lower() in ('1', 'true', 'yes', 'y')
                if typ is int:
                    return int(text)
                if typ is float:
                    return float(text)
                if typ is pygame.color.Color:
                    parts = [p.strip() for p in text.split(',')]
                    if len(parts) < 3:
                        raise ValueError('Color must be R,G,B')
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    return pygame.Color(r, g, b)
                # fallback
                return typ(text)

        # start or reuse a QApplication
        app = QtCore.QCoreApplication.instance()
        owns_app = False
        started_thread = None

        # If there is no running QApplication, create one and run its event loop in a background thread
        if app is None:
            # create application in a background thread to avoid blocking the caller
            def _qt_thread():
                nonlocal app
                try:
                    app = QtWidgets.QApplication([])
                    # store reference on class to prevent GC
                    Settings._qt_app = app
                    dlg = SettingsDialog(self)
                    # ensure dialog isn't deleted when closed unless user explicitly closes
                    try:
                        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
                    except Exception:
                        pass
                    dlg.show()
                    Settings._qt_dialogs.append(dlg)
                    # start event loop (will block this thread only)
                    try:
                        if hasattr(app, 'exec_'):
                            app.exec_()
                        else:
                            app.exec()
                    except Exception:
                        pass
                except Exception:
                    pass

            t = __import__('threading').Thread(target=_qt_thread, daemon=False)
            t.start()
            Settings._qt_thread = t
            started_thread = t
        else:
            # we already have QApplication running in main thread - create dialog non-blocking
            dlg = SettingsDialog(self)
            try:
                dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            except Exception:
                pass
            dlg.show()
            Settings._qt_dialogs.append(dlg)

        # Note: if we started a background QApplication thread, we don't join it here; it's daemonized so it won't block exit

    def _console_editor(self):
        """A very small console fallback to edit settings when Qt is not available."""
        print('Qt not available. Falling back to console input for Settings.')
        editable = [
            'FPS', 'window_width', 'window_height', 'window_background_color',
            'alive_reward', 'BORDER_BUFFER', 'BORDER_PUNISH', 'hit_reward',
            'kill_boss_reward', 'behit_reward', 'render', 'begin_episode',
            'epsilon_begin', 'epsilon_end', 'epsilon_decay', 'repeat_period'
        ]
        for k in editable:
            if not hasattr(self, k):
                continue
            cur = getattr(self, k)
            try:
                prompt = f"{k} (current: {cur}) - enter new value or leave empty to keep: "
                new = input(prompt)
            except Exception:
                # non-interactive environment; skip
                print('Non-interactive environment, skipping console editing.')
                return
            if new is None or new.strip() == '':
                continue
            try:
                if isinstance(cur, bool):
                    val = new.lower() in ('1', 'true', 'yes', 'y')
                elif isinstance(cur, int):
                    val = int(new)
                elif isinstance(cur, float):
                    val = float(new)
                elif isinstance(cur, pygame.color.Color):
                    parts = [p.strip() for p in new.split(',')]
                    if len(parts) < 3:
                        print('Invalid color, skipping')
                        continue
                    val = pygame.Color(int(parts[0]), int(parts[1]), int(parts[2]))
                else:
                    val = type(cur)(new)
                setattr(self, k, val)
                try:
                    setattr(type(self), k, val)
                except Exception:
                    pass
                print(f'Set {k} = {val}')
            except Exception as e:
                print(f'Failed to parse input for {k}: {e}')
