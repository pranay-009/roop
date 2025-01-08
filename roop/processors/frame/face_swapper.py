from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

def add_border(image_path:str, border_size:int =100):

    image = cv2.imread(image_path)
    white_border_image = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    cv2.imwrite(image_path, white_border_image)

def remove_border( image_path:str, border_size:int = 100):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read the image from {image_path}")

    height, width, _ = image.shape
    cropped_image = image[
        border_size:height-border_size,  # Crop top and bottom
        border_size:width-border_size   # Crop left and right
    ]

    
    cv2.imwrite(image_path, cropped_image)


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def get_face(path,border):
    try:
        face = get_one_face(cv2.imread(path))
        if face is None:
            add_border(path)
            border = True
            face = get_one_face(cv2.imread(path))

            if face is None:
                print("No Face was Found")
                return None, False
        print("Border added or not:",border)
            
        
        return face,border
    except Exception as e:
        print("Failed to get Face:",e)
        return None, False
      

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def multiface_process_frame(source_face, reference_face, temp_frame):
    print("length of faces",len(source_face))
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        face= source_face[-1]
        if many_faces:
            for target_face in many_faces:
                if source_face:
                    face = source_face.pop(0)
                temp_frame = swap_face(face, target_face, temp_frame)

        return temp_frame
    else:
        print("Roop global many faces is false")


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def multi_face_swap(source_path, target_path, output_path):
    lst_src_face = []
    for path in source_path:
        print(path)
        face,border = get_face(path,False)
        lst_src_face.append(face)
    
    target_frame = cv2.imread(target_path)
    result = multiface_process_frame(lst_src_face, None, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
