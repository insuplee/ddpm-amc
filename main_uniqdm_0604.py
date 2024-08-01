# 240305_Conditional_Diffusion_MNIST

import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from model_diffusion import DDPM, ContextUnet
from utility import get_time_info, fix_seed

import argparse
import time


#################################################################################
#                                  Training Loop                                #
#################################################################################

def get_class_info(dataset):
    _cti = dataset.class_to_idx  # {'00_BPSK' : 0, ...}  # cf. data_rfml.class = {'00_BPSK', ...}
    snr_mod_maps = [key for key, val in _cti.items()]  # {0: '00_BPSK', ...}
    return snr_mod_maps


def main(args):
    # hardcoding these here
    n_T = 400  # 500
    device = "cuda:0"
    lrate = 1e-4
    ws_test = [2.0]  # [0.0, 0.5, 2.0], strength of generative guidance
    img_shape = (1, args.img_size, args.img_size)

    day, hms = get_time_info()
    save_dir = f"./results/{day}_{hms}"  # './data/diffusion_outputs10/'

    print(">> MODEL LOAD")
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=128, n_classes=args.n_classes),
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))
    ddpm.to(device)
    fix_seed(seed=42)

    print(f">> DATA LOAD from {args.root}")
    # ImageFolder automatically converts to RGB
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.ImageFolder(root=args.root, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)

    print(">> START Training ")
    img_dir = "{}/generated_data/".format(save_dir)
    model_dir = "{}/trained_models/".format(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    snr_mod_maps = get_class_info(dataset)  # ['-2_128QAM', '-2_32PSK', '-2_BPSK', '-2_GMSK', '-2_OQPSK',], 

    for epoch in range(1, args.n_epoch+1):
        print(f'.. epoch {epoch}')
        ddpm.train()
        optim.param_groups[0]['lr'] = lrate * (1 - epoch / args.n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        ddpm.eval()

        if epoch % args.gen_period == 0:
            with torch.no_grad():
                n_sample = args.n_classes
                # [*] generate images - total categories
                print(f"[*] Epoch {epoch} : gen-images -> total categories")
                for w_i, w in enumerate(ws_test):
                    x_gen, _ = ddpm.sample(n_sample, img_shape, device, guide_w=w, n_c=args.n_classes)
                    grid = make_grid(x_gen, nrow=args.n_mod)  # n_mod
                    save_image(grid, img_dir + f"image_ep{epoch}_w{w}.png")

                    # [*] generate images - detailed categories
                    print(f"[*] Epoch {epoch} : gen-images -> detailed categories ({args.n_classes})")
                    for i_mod in range(args.n_classes):
                        x_gen2, _ = ddpm.sample_2(args.gen_sample_num, img_shape, device, guide_w=w, class_idx=i_mod)
                        mod = snr_mod_maps[i_mod]
                        img2_dir = "{}/details/{}/".format(img_dir, mod)
                        os.makedirs(img2_dir, exist_ok=True)
                        for i in range(args.gen_sample_num):
                            save_image(x_gen2[i], img2_dir + f"ep{epoch}_w{w}_i{i}.png")

        if epoch % args.ckpt_every == 0 or epoch == args.n_epoch :
            torch.save(ddpm.state_dict(), model_dir + f"model_{epoch}.pth")
            print('saved model at ' + model_dir + f"model_{epoch}.pth")


if __name__ == "__main__":
    print(f">> test in 240604, cuda availability : {torch.cuda.is_available()}")

    _start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epoch", type=int, default=1000)
    parser.add_argument("--n-mod", type=int, default=6)
    parser.add_argument("--n-classes", type=int, default=42)
    parser.add_argument("--root", type=str, default="../datasets/rfml/train")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--gen-period", type=int, default=10)  # epoch; during training
    parser.add_argument("--ckpt-every", type=int, default=50)  # epoch; during training
    parser.add_argument("--gen-sample-num", type=int, default=3)  # during training
    parser.add_argument("--lrate", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
    print(">> END in {:.2f} sec".format(time.time() - _start_time))
