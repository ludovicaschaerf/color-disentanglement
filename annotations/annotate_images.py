#!/usr/bin/env python
"""Extract color features from the generated textile images."""

import sys
import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
#from transformers import pipeline
import numpy as np
import pandas as pd
import time

import click
import math 
import pickle
from glob import glob 

from sklearn.cluster import KMeans

import cv2
import extcolors
DATA_DIR = '../data/'

with open(DATA_DIR + 'quantized_colors_and_names.pkl', 'rb') as infile:
    colors_dict = pickle.load(infile) 

from color_utils import *

sys.path.append('../utils/')
from utils import *

# @click.command()
# @click.option('--images_dir', help='Where the generated images are saved', type=str, required=True, metavar='DIR')
# @click.option('--pkl_file_name', help='File with latents and fnames', type=str, required=True, metavar='DIR')
# @click.option('--k', help='Numbers of colors', type=int, default=8)
# @click.option('--save_palette', help='Whether to save the color palette', type=bool, default=False)

def annotate_textile_image(image, k):
    """Produce annotations for the generated images.
    \b
    # 
    python annotate_textiles.py --images_dir generated_images_directory
    """
    try:
        colors, counts = extract_palette(image, k)
        sorted_pairs = sorted(zip(colors, counts), key=lambda x: x[1], reverse=True)
        sorted_colors, sorted_counts = zip(*sorted_pairs)

        # Convert the sorted tuples back to lists
        colors = np.array(sorted_colors)*255
        counts = list(sorted_counts)
            
        colour = get_color(colors[0], colors_dict['RGB'], colors_dict['names'])
        return colour    
    except Exception as e:
        print(e)
        return 'None'
            

#----------------------------------------------------------------------------

if __name__ == "__main__":
    with open(pkl_file_name, 'rb') as infile:
        pkl_file = pickle.load(infile)
        
    colours = []
    
    images = [images_dir + x.split('/')[-1] for x in pkl_file['fname']]
    
    print('Extracting main color from', len(images), 'images')
    for i,im in enumerate(tqdm(images)):
        colours.append(annotate_textile_image(image, k)) # pylint: disable=no-value-for-parameter
    
    pkl_file['color'] = colours
    
    with open(pkl_file_name.split('.pkl')[0] + '_color.pkl', 'wb') as outfile:
        pickle.dump(pkl_file, outfile)
    

#----------------------------------------------------------------------------

