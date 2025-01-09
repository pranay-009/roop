import threading
from typing import Any, Optional, List
import insightface
import numpy
import cv2

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        return get_face_analyser().get(frame)
    except ValueError:
        return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None

def draw_bbox(img,faces):
    dimg = img.copy()
    dictionary = {}
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(int)
        color = (0, 0, 255)
        dictionary[i] = face
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(dimg,str(i), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

    return dimg,dictionary

def get_faces_with_boxes(img):
    faces = get_many_faces(img)
    if len(faces) == 0:
        print("No face found")
        return img,None,None
    draw_bbox_img,face_pairs = draw_bbox(img,faces)
    return draw_bbox_img,faces,face_pairs