import shutil
import os
from RealESRGAN import RealESRGAN
from PIL import Image
import cv2
import numpy as np
import roop.globals
import roop.metadata
from roop.processors.frame.core import get_frame_processors_modules
from roop.core import  update_status
from roop.utilities import is_image
from roop.processors.frame.face_enhancer import process_image
from roop.processors.frame.face_swapper import process_image as swap_image
from roop.processors.frame.face_swapper import multi_face_swap as multi_face_swap



class FaceSwapper:

    def __init__(self,ckpt_dir:str='weights/RealESRGAN_x4.pth',use_sr:bool = True,use_gfpgan :bool= True):
        self.ckpt_dir = ckpt_dir
        self.use_sr = use_sr

        if self.use_sr:
            self.srmodel = RealESRGAN('cuda', scale=2)
            self.srmodel.load_weights(self.ckpt_dir, download=True)
        self.use_gfp = use_gfpgan
        
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            if not frame_processor.pre_check():
                pass

    def preprocessing(self,image_path:str):
        image = cv2.imread(image_path)
        h,w,c = image.shape
        if h< 512 or w<512:
            image = cv2.resize(image, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_path, image)
        return image_path
    def add_border(self,image_path:str, border_size:int =100):

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

    def remove_border(self,image_path:str, border_size:int = 100):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read the image from {image_path}")

        height, width, _ = image.shape
        cropped_image = image[
            border_size:height-border_size,  # Crop top and bottom
            border_size:width-border_size   # Crop left and right
        ]

     
        cv2.imwrite(image_path, cropped_image)


    def swap(self,source_path:str,target_path:str,output_path:str,postprocess:bool = True):

        swap_image(source_path = source_path,target_path = target_path,output_path = output_path)
        if postprocess:
            self.post_processing(cv2.imread(output_path),path=output_path)
    
        return output_path 

    def super_resolution(self,image):

        sr_image = self.srmodel.predict(image)
        sr_image = np.array(sr_image, dtype=np.uint8)
        return sr_image
    def gfpgan(self,image_path:str, output_path:str):
        return process_image(source_path = image_path,target_path = image_path,output_path = output_path)
    def save(self,image,path:str):
        cv2.imwrite(path, image)

    def post_processing(self,image,scale=0.5,blurr_kernel=7,path = '/content/output.png',):
        #image = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)
        #image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h,w,c = image.shape
        if h>2048 or w>2048:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        blurred_image = cv2.GaussianBlur(image, (blurr_kernel, blurr_kernel), 0)
        denoised_pil = Image.fromarray(blurred_image)
        denoised_array = np.array(denoised_pil)
        if self.use_sr:
            denoised_array = self.super_resolution(denoised_array)

        if path:
            self.save(denoised_array,path)
        if self.use_gfp:
            print("GFPGAN USED")
            self.gfpgan(path,path)

    def multi_faceswap(self,list_image_path,target_path,output_path) -> None:

        shutil.copy2(target_path, output_path)

        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            print("Curent Frame Processor is:",frame_processor)
            if frame_processor.NAME == 'ROOP.FACE-SWAPPER':
                multi_face_swap(list_image_path,target_path, output_path)
            else:
                self.post_processing(cv2.imread(output_path),path=output_path)
            frame_processor.post_process()
        # validate image
        if is_image(target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return

