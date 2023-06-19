import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import ruamel.yaml as yaml

@torch.no_grad()
def multi_image_caption(images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = blip_decoder(pretrained='checkpoint_best.pth')
    model.eval()
    model = model.to(device)
    
    return [single_image_caption(image, model, device) for image in images]

@torch.no_grad()
def single_image_caption(image, model, device):
    config = yaml.load(open('configs/caption_cctv.yaml', 'r'), Loader=yaml.Loader)
    transform = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    caption = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], min_length=config['min_length'])[0]

    return caption