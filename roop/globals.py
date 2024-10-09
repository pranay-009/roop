from typing import List, Optional
import onnxruntime

def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1

source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None
headless: Optional[bool] = True
frame_processors: List[str] = ['face_swapper','face_enhancer']
keep_fps: Optional[bool] = True
keep_frames: Optional[bool] = True
skip_audio: Optional[bool] = True
many_faces: Optional[bool] = True
reference_face_position: Optional[int] = 0
reference_frame_number: Optional[int] = 0
similar_face_distance: Optional[float] =  0.85
temp_frame_format: Optional[str] = "png"
temp_frame_quality: Optional[int] = 0
output_video_encoder: Optional[str] = 'libx264'
output_video_quality: Optional[int] = 0
max_memory: Optional[int] = None
execution_providers: List[str] = ['CPUExecutionProvider']
execution_threads: Optional[int] = suggest_execution_threads()
log_level: str = 'error'
