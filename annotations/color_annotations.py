#!/usr/bin/env python
"""Extract color features from the generated textile images."""

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

import cv2
import extcolors
from colormap import rgb2hex
DATA_DIR = '../data/'

from color_harmony import extract_harmonies
sys.path.append('../utils')
from utils import *

def color_to_df(input_, hex_=False):
    colors_pre_list = str(input_).replace('([(','').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    if hex_:
        #convert RGB to HEX code
        df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                              int(i.split(", ")[1]),
                              int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    else:
        df_color_up = [rgb2hsv(int(i.split(", ")[0].replace("(","")),
                              int(i.split(", ")[1]),
                              int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
    return df

    
def extract_color(df_color, zoom=None, outpath=None, save=None):
    #annotate text
    list_color = list(df_color['c_code'])
    list_precent = [int(i) for i in list(df_color['occurence'])]
    text_c = [str(c) + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
    if save:
        plot_color_palette(outpath, zoom, list_precent, text_c, list_color, input_image,)
    colors = list(df_color['c_code'])
    return colors[:5]


@click.command()
@click.option('--images_dir', help='Where the output images are saved', type=str, required=True, metavar='DIR')

def annotate_textile_images(
    images_dir: str,
    
):
    """Produce annotations for the generated images.
    \b
    # 
    python annotate_textiles.py --images_dir generated_images_directory
    """
    colours = []
    images = glob(images_dir + '/*.jpg')
    
    for i,im in enumerate(tqdm(images)):
        colors_x = extcolors.extract_from_path(im, tolerance=8, limit=13)
        df_color = color_to_df(colors_x)
        top_cols = extract_color(df_color)
        top_cols_no_black = [cc[0] for cc in top_cols if (cc[1] != 0) and (cc[2] != 0) and (cc[0] != 0)]
        top_cols_filtered = [cc[0] for cc in top_cols if (cc[1] != 0) and (cc[2] != 0)]
        harmonies = extract_harmonies(top_cols_filtered)
        colours.append([im.split('/')[-1]]+[c for cc in top_cols_no_black for c in cc]+harmonies)
        if i % 1000 == 0:
            df = pd.DataFrame(colours, columns=['fname', 'H1', 'S1', 'V1', 'H2', 'S2', 'V2', 'H3', 'S3', 'V3',
                                                'H4', 'S4', 'V4', 'H5', 'S5', 'V5', 
                                                'Monochromatic', 'Analogous', 'Complementary', 'Triadic',
                                                'Split Complementary', 'Double Complementary',
                                                ])
            df['Color'] = cat_from_hue(df['H1'], df['S1'], df['V1'])
            df = df.sort_values('fname')
            seeds_min = int(df['fname'][0].replace('.jpg'))
            seeds_max = int(df['fname'][-1].replace('.jpg'))
            print(df.head())
            print(df['Triadic'].value_counts())
            print(df['Double Complementary'].value_counts())
            df.to_csv(DATA_DIR + f'color_palette{seeds_min:05d}-{seeds_max:05d}.csv', index=False)
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    annotate_textile_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

