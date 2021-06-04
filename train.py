import torch
import config
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import VGGLoss
from dataset import ImageData
from model import Generator,Discriminator
from utils import save_checkpoint,load_checkpoint

def train_fn(loader,disc,gen,opt_disc,opt_gen,mse,bce,vgg_loss):
    loop = tqdm(loader,leave=True)
    for idx, (low_res,high_res) in enumerate(loop):
        low_res = low_res.to(config.DEVICE)
        high_res = high_res.to(config.DEVICE)

        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_real_loss = bce(disc_real,torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real))
        disc_fake_loss = bce(disc_fake,torch.zeros_like(disc_fake))
        loss_disc = disc_fake_loss + disc_real_loss

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()
        
        disc_fake = disc(fake)
        #l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake,torch.ones_like(disc_fake))
        loss_vgg = 0.006 * vgg_loss(fake,high_res)
        loss_gen = adversarial_loss + loss_vgg

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

def main():
    dataset = ImageData(root_dir=config.ROOT_DIR)
    loader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True)
    gen = Generator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    disc = Discriminator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader,disc,gen,opt_disc,opt_gen,mse,bce,vgg_loss)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

if __name__=='__main__':
    main()