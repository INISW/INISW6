from flask import Blueprint, redirect
from database.db_handler import DBHandler
import json
import os
from werkzeug.utils import secure_filename

#---------main caption import---------#
import cv2
import pandas as pd
from PIL import Image
#-------------------------------------#

#---------inference import---------#
import torch
from transformers import AutoProcessor
#----------------------------------#

#---------inference(민재) import---------#
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import sys
sys.path.append("C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\blip")

from models.blip import blip_decoder
import ruamel.yaml as yaml
#----------------------------------------#

caption = Blueprint("caption", __name__)

@caption.route('/video_captioning')
def video_captioning(video_name, video_id, input_keyword):

    print("-" * 40)
    print(video_name)
    print("-" * 40)

    db_handler = DBHandler()

    video = os.path.join('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\INISW_proj\\web\\www\\static\\videos', str(video_name))

    data = data_from_mysql() # db 불러와서 object_id별로 filtering
    result = getcaptions_from_video(video, data) # object_id 별 captioning 생성 
    db_handler.savecaption(data, result) # 각 db 행에 대하여 object_id matching후 caption 넣고 저장하기''' 

    sql_rs = db_handler.get_caption(video_id, input_keyword)

    return json.dumps({'status':'200', 'caption_info':sql_rs})

##------data section------
def data_from_mysql(): # db를 받아 dict 형태로 받기 
    db_handler = DBHandler()
    sql_rs = db_handler.db_to_dataframe()
    result = pd.DataFrame(sql_rs)
    # table의 모든 것을 dataframe 형식으로 가져오고 이걸 object_id형식으로 
    db = result.drop_duplicates(subset='object_id') 

    print(db)
    return db

##------result section------
def cropping(cap, frame_id, box):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id) # 이미지 불러오기 
    T, image = cap.read()
    image = image[box[1]:box[3],box[0]:box[2]] if T else print("error")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if T else None

def getcaptions_from_video(video, data): #data와 video를 받아와서 images들을 보내준다.
    cap =cv2.VideoCapture(video)
    print(cap)
    images = []
    for _, info in data.iterrows(): # 이미지 전체 다 가져오기
        images.append(Image.fromarray(cropping(cap, info['frame_id'], [info['x1'],info['y1'],info['x2'],info['y2']])))
    
        
    return multi_image_caption(images)

#--------------inference---------------#
# def multi_image_caption(images):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = torch.load('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\git\\git.pt', map_location=device)
#     model.eval()
    
#     vis_processors = AutoProcessor.from_pretrained('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\git\\preprocessor').image_processor # 훨씬 더 빠르다. 
#     decode = AutoProcessor.from_pretrained('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\git\\preprocessor').batch_decode
#     return [single_image_caption(image, model, vis_processors, decode, device) for image in images]


# def single_image_caption(image, model, vis_processors, decode, device):
#     generated_ids = model.generate(pixel_values= vis_processors(images=image, return_tensors="pt").to(device).pixel_values,max_length=500)
#     caption= decode(generated_ids, skip_special_tokens=True)[0]
#     return caption

@torch.no_grad()
def multi_image_caption(images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = blip_decoder(pretrained='C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\blip\\checkpoint_last_epoch5_CA.pth')
    model.eval()
    model = model.to(device)
    
    return [single_image_caption(image, model, device) for image in images]

@torch.no_grad()
def single_image_caption(image, model, device):
    config = yaml.load(open('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\blip\\configs\\caption_cctv.yaml', 'r', encoding="UTF-8"), Loader=yaml.Loader)
    transform = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    caption = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], min_length=config['min_length'])[0]

    return caption
#---------------------------------------#

@caption.route('/test_page/caption')
def test_page():
    return redirect('/#keyword')
