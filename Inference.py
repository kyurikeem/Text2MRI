import os
# import sys
import einops
import numpy as np
import torch
import random
import cv2
from PIL import Image
from pytorch_lightning import seed_everything
import argparse

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from transformers import logging
logging.set_verbosity_error()
from sd.utils import get_sample
from ldm.util import remove_non_empty_directory


def inference(args):
    for idx in range(args.sample_num):
        print("Inference.. %d/%d"%(idx, args.sample_num))
        condition = get_sample("a T2 modality MR image, type of %s glioma"%(args.tumor_type))
        model = create_model('config.yaml').cpu()
        model.load_state_dict(load_state_dict('./ckpt/last.ckpt'))
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)
        apply_canny = CannyDetector()
        
        os.makedirs(args.save_dir, exist_ok=True) 
        
        with torch.no_grad():
            num_samples = 1
            condition_map = apply_canny(condition.astype(np.uint8), 100, 200)        
            condition_map = resize_image(HWC3(condition_map), 512)
            control = torch.from_numpy(condition_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            seed = random.randint(0, 65535) 
            seed_everything(seed)
            
            prompt = 'MR image, type of %s glioma'%(args.tumor_type)
            flair_prompt = 'a FLAIR modality ' + prompt
            t1_prompt = 'a T1 modality ' + prompt
            t1ce_prompt = 'a T1CE modality ' + prompt
            t2_prompt = 'a T2 modality ' + prompt

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([flair_prompt, t1_prompt, t1ce_prompt, t2_prompt] * num_samples)]}
            samples, intermediates = ddim_sampler.sample(100, num_samples*4, [3, 64, 64], cond, verbose=False, eta=1.0)
    
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [x_samples[i] for i in range(num_samples*4)]
            saver = np.concatenate(results, axis=1) 
            Image.fromarray(saver.astype(np.uint8)).save(os.path.join(args.save_dir, '%s.png'%(idx)), compress_level=9)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor_type", type=str, default="mutant")
    parser.add_argument("--save_dir", type=str, default='./results')
    parser.add_argument("--sample_num", type=int, default=3)
    args = parser.parse_args()

    inference(args)