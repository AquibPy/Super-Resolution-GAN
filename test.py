import torch
import config
import torch.optim as optim
from model import Generator
from torchvision.utils import save_image
from utils import load_checkpoint
from PIL import Image
import numpy as np

image_path = "0002.png"
img = np.array(Image.open(image_path))
img = config.lowres_transform(image=img)['image']
img = torch.unsqueeze(img,0).to(config.DEVICE)
save_image(img,'input.png')
gen = Generator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
load_checkpoint('gen.pth.tar',gen,opt_gen,config.LEARNING_RATE)

gen.eval()
with torch.no_grad():
    upscaled_img = gen(img)
    save_image(upscaled_img*0.5+0.5,"output.png")
gen.train()