import winsound
import os
from pathlib import Path
from threading import Thread

_ENABLED = True
_SOUND_DIR = Path("assets/sounds")
_LAST_PLAY = 0.0
_MIN_INTERVAL = 1.0

def set_enabled(val: bool):
    global _ENABLED
    _ENABLED = val

def _play_async(func):
    import time as _time
    global _LAST_PLAY
    now = _time.time()
    if now - _LAST_PLAY < _MIN_INTERVAL:
        return
    _LAST_PLAY = now
    t = Thread(target=func, daemon=True)
    t.start()

def entry():
    if not _ENABLED:
        return
    _play_async(lambda: winsound.Beep(800, 100))

def win():
    if not _ENABLED:
        return
    _play_async(lambda: winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS))

def loss():
    if not _ENABLED:
        return
    _play_async(lambda: winsound.PlaySound("SystemHand", winsound.SND_ALIAS))

def alert():
    if not _ENABLED:
        return
    _play_async(lambda: winsound.Beep(1200, 200))
