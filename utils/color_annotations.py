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
from PIL import Image
import math 
import pickle
from glob import glob 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2
import extcolors

from colormap import rgb2hex
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def color2range(color, colors_list, color_bins):
    if color == 'BW':
        range_col = {'h': None, 's':[0, 10], 'v':[0, 10]}
    elif color == 'Red':
        range_col = {'h': [340, 10], 's':[10, 100], 'v':None}
    else:
        try:
            idx = colors_list.index(color)
            x = color_bins[idx]
            y = color_bins[idx + 1]
            range_col = {'h': [max(x,10), min(y, 340)], 's':[10, 100], 'v':None}
        except Exception as e:
            print(e)
            range_col = {'h':None, 's':None, 'v':None}
    print(color, range_col)
    return range_col
    
def range2color(h, s, v, colors_list, color_bins):
    if (s >= 0 and s < 10) or (v >= 0 and v < 10):
        color = 'BW'
    elif (h >= 340 or h < 10):
        color = 'Red'
    else:
        for idx, col in enumerate(colors_list[:-1]):
            x = color_bins[idx]
            y = color_bins[idx + 1]
            if (h >= x and h < y):
                color = col
    return color
    
def range2continuous(val, range_col_h):
    if range_col_h is None:
        print('Value is None, not implemented yet')
        return 
    if val >= range_col_h[0] and val <= range_col_h[1]:
        cont_value = 180
    elif range_col_h[0] == 340:
        if (val >= range_col_h[0] or val <= range_col_h[1]):
            cont_value = 180
        elif val < range_col_h[0] and val >= range_col_h[0] - 180:
            cont_value = (180 - np.abs(range_col_h[0] - val)) 
        elif val > range_col_h[1] and val <= range_col_h[1] + 180:
            cont_value = (180 - np.abs(range_col_h[1] - val)) 
        else:
            print('Not sure what this case is', val, range_col_h)
            return    
    else:
        if val > range_col_h[1] and val <= range_col_h[1] + 180:
            cont_value = (180 - np.abs(range_col_h[1] - val)) 
        elif val > range_col_h[1] and val >= range_col_h[1] + 180:
            remainder = 360 - val
            cont_value = (180 - np.abs(range_col_h[0] + remainder))
        elif val < range_col_h[0] and val <= np.abs(range_col_h[0] - 180):
            cont_value = (180 - np.abs(val - range_col_h[0])) 
        elif val < range_col_h[0] and val >= np.abs(range_col_h[0] - 180):
            remainder = val
            cont_value = (180 - np.abs((360 - range_col_h[1]) + remainder))
        else:
            print('Not sure what this case is', val, range_col_h)
            return
    #print(range_col_h, val, cont_value)
    return np.abs(cont_value)

def color_to_df(input):
    colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    
    #convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                          int(i.split(", ")[1]),
                          int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    
    df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
    return df

def extract_color(input_image, tolerance, zoom, outpath, save=None):
    colors_x = extcolors.extract_from_image(input_image, tolerance = tolerance, limit = 13)
    df_color = color_to_df(colors_x)
    
    #annotate text
    list_color = list(df_color['c_code'])
    list_precent = [int(i) for i in list(df_color['occurence'])]
    text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
    colors = list(df_color['c_code'])
    if '#000000' in colors:
        colors.remove('#000000')
    return colors[:3]


@click.command()
@click.option('--genimages_dir', help='Where the output images are saved', type=str, required=True, metavar='DIR')

def annotate_textile_images(
    genimages_dir: str,
    
):
    """Produce annotations for the generated images.
    \b
    # 
    python annotate_textiles.py --genimages_dir /home/ludosc/data/stylegan-10000-textile-upscale
    """
    colours = []
    pickle_files = glob(genimages_dir + '/imgs0000*.pkl')
    for pickle_file in pickle_files:
        print('Using pickle file: ', pickle_file)
        with open(pickle_file, 'rb') as f:
            info = pickle.load(f)

        listlen = len(info['fname'])
        os.makedirs('/data/ludosc/colour_palettes/', exist_ok=True)
        for i,im in enumerate(tqdm(info['fname'])):
            try:
                top_cols = exact_color(im, 12, 5, '/data/ludosc/colour_palettes/' + im.split('/')[-1])
                colours.append([im]+top_cols)
            except Exception as e:
                print(e)
            if i % 1000 == 0:
                df = pd.DataFrame(colours, columns=['fname', 'top1col', 'top2col', 'top3col'])
                print(df.head())
                df.to_csv(genimages_dir + f'/top_three_colours.csv', index=False)
                
        df = pd.DataFrame(colours, columns=['fname', 'top1col', 'top2col', 'top3col'])
        print(df.head())
        df.to_csv(genimages_dir + f'/final_sim_{os.path.basename(pickle_file.split(".")[0])}.csv', index=False)
        
#----------------------------------------------------------------------------

if __name__ == "__main__":
    annotate_textile_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
