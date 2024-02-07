from playsound import playsound 
from pathlib import Path
import sys
import os

target_obj_verb = 'apple'

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

file = ROOT / f'sound/{target_obj_verb}.mp3'
playsound(file)