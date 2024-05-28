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
from typing import List, Optional, Tuple, Union
from transformers import PretrainedConfig
import click
import numpy as np
import PIL.Image
import torch
import pickle
import cv2
from tqdm import tqdm
import pandas as pd
DATA_DIR = '../data/'

from color_annotations import *

sys.path.append('../utils')
from utils import *

sys.path.insert(0, '/shares/weddigen.ki.phf.uzh/ludosc/color-disentanglement/stylegan')
import dnnlib 
import legacy
from networks_stylegan3 import *

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

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
    class_idx: Optional[int]
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


        # colors_x = extcolors.extract_from_path(im, tolerance=8, limit=13)
        K = 8
        colors_x = adaptive_clustering_2(img, K=K)
        df_color = color_to_df(colors_x)
        print(df_color.head())
        top_cols = extract_color(df_color)
        top_cols_filtered = [cc[0] for cc in top_cols if (cc[1] != 0) and (cc[2] != 0)]
        harmonies = extract_harmonies(top_cols_filtered)
        hsvs_names = [[f'H{str(i)}', f'S{str(i)}', f'V{str(i)}'] for i in range(1,K+1)]
        hsvs_names = [x for xs in hsvs_names for x in xs]
        colours.append([f'{outdir}/seed{seed:05d}.png'] +[c for cc in top_cols for c in cc]+harmonies)
        if seed_idx % 10 == 0:
            df = pd.DataFrame(colours, columns=['fname', *hsvs_names, 'Monochromatic', 
                                                'Analogous', 'Complementary', 'Triadic', 'Split Complementary',
                                                'Double Complementary'])
            df['Color'] = cat_from_hue(np.array(df['H1'].values), df['S1'], df['V1'])
            print(df.head())
            df.to_csv(DATA_DIR + f'color_palette{seeds[0]:05d}-{seeds[-1]:05d}.csv', index=False)
        
    info = {'fname': fnames, 'seeds':seeds, 'z_vectors': z_vals, 'w_vectors': w_vals}
    with open(f'{DATA_DIR}seeds{seeds[0]:05d}-{seeds[-1]:05d}.pkl', 'wb') as f:
        pickle.dump(info, f)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_annotate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
