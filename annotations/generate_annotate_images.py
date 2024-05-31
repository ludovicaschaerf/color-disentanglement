# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from transformers import PretrainedConfig
import click
import numpy as np
import PIL.Image
import torch
import pickle
import cv2
from tqdm import tqdm
import re
import pandas as pd
DATA_DIR = '../data/'

from annotate_images import *

sys.path.append('../utils')
from utils import *

sys.path.insert(0, '/shares/weddigen.ki.phf.uzh/ludosc/color-disentanglement/stylegan')
import dnnlib 
import legacy
from networks_stylegan3 import *

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--k', help='Numbers of colors', type=int, required=True, metavar='DIR', default=8)
@click.option('--save', help='Store the generated file', type=bool, required=True, metavar='SAVE')
def generate_annotate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    save: bool,
    class_idx: Optional[int],
    k: Optional[int]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Saving', save)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    w_vals = []
    z_vals = []
    fnames = []
    colours = []
    # Generate images.
    for seed_idx, seed in enumerate(tqdm(seeds)):
        # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))
        z_vals.append(z.numpy())
        z = z.to(device)
        fnames.append(f'{outdir}/seed{seed:05d}.png')
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        W = G.mapping(z, label, truncation_psi=truncation_psi)
        w_vals.append(W[:,0,:].cpu().numpy())
        img = G.synthesis(W, noise_mode=noise_mode)

        # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        if save:
            img.save(f'{outdir}/seed{seed:05d}.png')


        color = annotate_textile_image(img, k)
        colours.append(color)
        
        
    info = {'fname': fnames, 'seeds':seeds, 'z_vectors': z_vals, 'w_vectors': w_vals, 'color':colours}
    with open(f'{DATA_DIR}seeds{seeds[0]:05d}-{seeds[-1]:05d}.pkl', 'wb') as f:
        pickle.dump(info, f)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_annotate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
