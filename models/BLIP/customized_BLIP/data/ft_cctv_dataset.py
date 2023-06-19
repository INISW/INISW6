import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset
#from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption, define_model, ImageSuperResolution

class cctv_caption_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        #url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        # filename = 'coco_karpathy_train.json'
        filename = 'cctv_captions_ft.json'

        #download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root,ann['image'])
        image = Image.open(image_path).convert('RGB')

        if image.size[0] < 100 or image.size[1] < 100:
            img_lq = cv2.imread(f'{image_path}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
            image = ImageSuperResolution(img_lq)
            
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    
class cctv_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        #urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
        #        'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'cctv_captions_val_ft.json','test':'cctv_100_caption_test.json'}
        
        #download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')

        if image.size[0] < 100 or image.size[1] < 100:
            img_lq = cv2.imread(f'{image_path}', cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.
            image = ImageSuperResolution(img_lq)

        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('e')[-1]
        
        return image, int(img_id)   