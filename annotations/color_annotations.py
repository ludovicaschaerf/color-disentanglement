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

from color_harmony import extract_harmonies
sys.path.append('../utils/')
from utils import *
from colormap import rgb2hex, rgb2hsv

def color_to_df(input_, hex_=False, adaptive=True):
    if adaptive:
        df_rgb = input_[0]
        if hex_:
            df_color_up = [rgb2hex(*i, normalised=True) for i in df_rgb]
        else:
            df_color_up = [rgb2hsv(*i, normalised=True) for i in df_rgb]
        
        df_color_up = [[x*360, y*100,z*100] for [x,y,z] in df_color_up]
        df_percent = input_[1]
    else:
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
    df_rgb = [tuple(i) for i in df_rgb]
    df = pd.DataFrame(zip(df_color_up, df_percent, df_rgb), columns = ['c_code','occurence', 'rgb'])
    return df
    
def extract_color(df_color, input_image=None, outpath='color_palettes/', save=None):
    #annotate text
    list_color = list(df_color['rgb'])
    list_precent = [int(i) for i in list(df_color['occurence'])]
    text_c = [str(c) + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
    if save:
        plot_color_palette(outpath, 1, list_precent, text_c, list_color, input_image,)
    colors = list(df_color['c_code'])
    # colors_no_black = [cc for cc in colors if ((cc[1] != 0) and (cc[2] != 0) and (cc[0] != 0))]
    # print(colors_no_black)
        
    return colors#[:5]

def adaptive_clustering_2(image_path, K=8, max_iterations=100, convergence_threshold=0.01):
    # Load the image in RGB
    if type(image_path) == str:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    else:
        image = np.array(image_path)
    
    original_shape = image.shape
    # Flatten the image to a 2D array of pixels (3 columns for R, G, B)
    pixels = image.reshape(-1, 3)
    if np.isnan(pixels).any() or (pixels == 0).all():
        print("Data contains NaN values")
        print(image_path)
        return np.array([(0,0,0)]*K), [len(pixels)]*K
    
    # Initial K-means to segment the image
    kmeans = KMeans(n_clusters=K, random_state=0)
    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_

    old_labels = labels

    # Adaptive iteration
    n_iterations = 0
    while n_iterations < max_iterations:
        # Recalculate centroids from labels
        centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(K)])

        # Re-cluster using updated centroids
        kmeans = KMeans(n_clusters=K, init=centroids, n_init=1)
        labels = kmeans.fit_predict(pixels)

        # Check for convergence
        if np.sum(labels != old_labels) < convergence_threshold * len(pixels):
            break

        old_labels = labels
        n_iterations += 1

    cnts = np.array([len(pixels[labels == i]) for i in range(K)])
    
    return np.array(centroids) / 255, cnts

@click.command()
@click.option('--images_dir', help='Where the generated images are saved', type=str, required=True, metavar='DIR')

def annotate_textile_images(
    images_dir: str,
    
):
    """Produce annotations for the generated images.
    \b
    # 
    python annotate_textiles.py --images_dir generated_images_directory
    """
    colours = []
    with open('/home/ludosc/ludosc/color-disentanglement/data/seeds_asyrp_0000_4319.pkl', 'rb') as infile:
        pkl_file = pickle.load(infile)
        
    images = [images_dir + x.split('/')[-1] for x in pkl_file['fname']] # glob(images_dir + '/*.png')
    print(len(images))
    for i,im in enumerate(tqdm(images)):
        try:
            # colors_x = extcolors.extract_from_path(im, tolerance=8, limit=13)
            K = 8
            colors_x = adaptive_clustering_2(im, K=K)
            df_color = color_to_df(colors_x)
            
            if i < 50:
                top_cols = extract_color(df_color, input_image=im, outpath='color_palettes/', save=True)
            else:
                top_cols = extract_color(df_color, save=None)
                
            top_cols_filtered = [cc[0] for cc in top_cols if (cc[1] != 0) and (cc[2] != 0)]
            harmonies = extract_harmonies(top_cols_filtered)
            colours.append([im.split('/')[-1]] + [c for cc in top_cols for c in cc] + harmonies)
            
        except:
            print(im)
            colours.append([im.split('/')[-1]] + [0]*3*K + [False]*6)
            
        if i % 10 == 0:
            hsvs_names = [[f'H{str(i)}', f'S{str(i)}', f'V{str(i)}'] for i in range(1,K+1)]
            hsvs_names = [x for xs in hsvs_names for x in xs]
            df = pd.DataFrame(colours, columns=['fname', *hsvs_names, 
                                                    'Monochromatic', 'Analogous', 'Complementary', 'Triadic',
                                                    'Split Complementary', 'Double Complementary',
                                                    ])
            df['Color'] = cat_from_hue(np.array(df['H1'].values), df['S1'], df['V1'])
            df = df.sort_values('fname').reset_index()
            print(df.head())
            if 'seed' in df['fname'][0]:
                seeds_min = 0 #int(df['fname'][0].replace('.png', '').replace('seed', ''))
                seeds_max = 4319 #int(df['fname'][len(df)-1].replace('.png', '').replace('seed', ''))
            else:
                seeds_min = 0
                seeds_max = len(df)
            df.to_csv(DATA_DIR + f'color_palette{seeds_min:05d}-{seeds_max:05d}.csv', index=False)
            
    df = pd.DataFrame(colours, columns=['fname', *hsvs_names, 
                                         'Monochromatic', 'Analogous', 'Complementary', 'Triadic',
                                         'Split Complementary', 'Double Complementary',
                                         ])
    df['Color'] = cat_from_hue(np.array(df['H1'].values), df['S1'], df['V1'])
    df = df.sort_values('fname').reset_index()
    print(df.head())
    df.to_csv(DATA_DIR + f'color_palette{seeds_min:05d}-{seeds_max:05d}.csv', index=False)
            
    pkl_file['color'] = df['Color']
    with open('/home/ludosc/ludosc/color-disentanglement/data/seeds_asyrp_0000_4319_c.pkl', 'wb') as outfile:
        pickle.dump(pkl_file, outfile)
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    annotate_textile_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

