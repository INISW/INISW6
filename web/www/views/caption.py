from flask import Blueprint, redirect
from database.db_handler import DBHandler
import json, os
#---------main caption import---------#
import cv2
import pandas as pd
from PIL import Image
import numpy as np
#-------------------------------------#

#---------inference import------------#
import torch
from transformers import AutoProcessor, Swin2SRImageProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
#-------------------------------------#

caption = Blueprint("caption", __name__)

@caption.route('/video_captioning')
def video_captioning(video_name, video_id, input_keyword):

    print("-" * 70)
    print(video_name, " Captioning 시작")
    print("-" * 70)

    db_handler = DBHandler()

    video = os.path.join('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\INISW_proj\\web\\www\\static\\videos', str(video_name))

    data = data_from_mysql() # db 불러와서 object_id별로 filtering
    result = getcaptions_from_video(video, data) # object_id 별 captioning 생성 
    db_handler.savecaption(data, result) # 각 db 행에 대하여 object_id matching후 caption 넣고 저장하기''' 

    return json.dumps({'status':'200'})

##------data section------
def data_from_mysql(): # db를 받아 dict 형태로 받기 
    db_handler = DBHandler()
    sql_rs = db_handler.db_to_dataframe()
    result = pd.DataFrame(sql_rs)
    # table의 모든 것을 dataframe 형식으로 가져오고 이걸 object_id형식으로 
    db = result.drop_duplicates(subset='object_id',keep='first') 
    db = db.reset_index(drop=True)
    return db

##------result section------
def cropping(cap, frame_id, box):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id) # 이미지 불러오기 
    T, image = cap.read()
    image = image[box[1]:box[3],box[0]:box[2]] if T else print("error")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if T else None

def getcaptions_from_video(video, data): #data와 video를 받아와서 images들을 보내준다.
    cap =cv2.VideoCapture(video)
    images = []
    for _, info in data.iterrows(): # 이미지 전체 다 가져오기
        images.append(Image.fromarray(cropping(cap, info['frame_id'], [info['x1'],info['y1'],info['x2'],info['y2']])))
        
    return multi_image_caption(images)

#--------------BLIP inference---------------#
def super_resolution(image,model_sr,pro_sr,device):
    inputs = pro_sr(image, return_tensors="pt").to(device)

    # forward pass
    with torch.no_grad():
        outputs = model_sr(**inputs)

    output = outputs.reconstruction.data.squeeze().cpu().float().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)

def multi_image_caption(images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load("C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\blip\\blip_all_1e6_final.pt", map_location=device)
    model.eval()
    vis_processors = BlipProcessor.from_pretrained('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\blip\\preprocessor').image_processor # 훨씬 더 빠르다. 
    decode = BlipProcessor.from_pretrained('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\blip\\preprocessor').batch_decode
   
    pro_sr = Swin2SRImageProcessor.from_pretrained('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\git\\sr\\preprocessor')
    model_sr = torch.load('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\caption\\git\\sr\\swinsr.pt',map_location=device)

    caption = []
    for image in images: 
        image = super_resolution(image,model_sr,pro_sr,device) if image.size[0]<50 or image.size[1]<100 else image # 이미지 전처리 과정 
        
        caption.append(single_image_caption(image,model,vis_processors,decode,device)) # 이미지 captioning
    return caption

def single_image_caption(image, model, vis_processors, decode, device):
    generated_ids = model.generate(pixel_values= vis_processors(images=image, return_tensors="pt").to(device).pixel_values,max_length=300)
    caption= decode(generated_ids, skip_special_tokens=True)[0]
    return caption
#--------------------------------------------#
