import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# def plot_examples(low_res_folder, gen):
#     files = os.listdir(low_res_folder)

#     gen.eval()
#     for file in files:
#         image = Image.open("test_images/" + file)
#         with torch.no_grad():
#             upscaled_img = gen(
#                 config.test_transform(image=np.asarray(image))["image"]
#                 .unsqueeze(0)
#                 .to(config.DEVICE)
#             )
#         save_image(upscaled_img * 0.5 + 0.5, f"saved/{file}")
#     gen.train()