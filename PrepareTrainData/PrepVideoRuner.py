from pathlib import Path
from .VideoCutter import VideoSceneSplitter
from .TrainingVideoValidator import TrainingVideoValidator

base_path = Path('VideoCutter.py').resolve().parent.parent
data_path = base_path / 'Dirty & Brutal Fouls in Football.mp4'
output_path = base_path / '!Studia' / 'ProjektOutputData'

video_splitter = VideoSceneSplitter(Path('Dirty & Brutal Fouls in Football.mp4'), output_path)
video_splitter.scene_based_splitting()