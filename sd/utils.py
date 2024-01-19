import sys
import os
import glob
import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    model.cuda()
    model.eval()
    return model
    
def get_sample(prompt):
    config = OmegaConf.load('./sd/sd-config.yaml')
    model = load_model_from_config(config, './ckpt/mask.ckpt')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            for n in trange(1, desc="Sampling"):
                c = model.get_learned_conditioning(1 * [prompt])
                samples_ddim, _ = sampler.sample(S=100, conditioning=c, batch_size=1, shape=[3, 64, 64] , eta=1.0)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                all_samples.append(x_samples_ddim)

    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid)

    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    grid = grid.astype(np.uint8)[:,:,0]
    return grid

