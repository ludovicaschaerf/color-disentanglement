#!/usr/bin/env python

DATA_DIR = '../data/'

import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import sys
from scipy import signal
import cv2
import random

sys.path.append('../annotations')
from color_annotations import extract_color, color_to_df
from color_harmony import extract_harmonies
import extcolors

sys.path.append('../disentanglement')
from disentanglement import DisentanglementBase
        
sys.path.append('../utils/')
from utils import *

sys.path.append('../stylegan')
from networks_stylegan3 import *
import dnnlib 
import legacy

class DisentanglementEvaluation(DisentanglementBase):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)
    
    def structural_coherence(self, im1, im2):
        ## struct coherence
        img_orig_norm = cv2.normalize(np.array(im1.convert('L')), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_norm = cv2.normalize(np.array(im2.convert('L')), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cor = cv2.filter2D(img_orig_norm, ddepth=-1, kernel=img_norm)
        cor = np.max(cor)
        return cor
    
    def DCI(self):
        """Still needs to be properly implemented"""
        if not self.categorical:
            factors = self.df[self.variable]
            codes = self.get_encoded_latent()
            disentanglement, completeness, informativeness = dci(factors, codes)
            return disentanglement, completeness, informativeness
        else:
            print('Not implemented for categorical data')
    

    def obtain_changes(self, separation_vector, seeds, lambda_range=15, method='None', feature='None', subfolder='test'):
        variation_scores = []
        for i,seed in enumerate(tqdm(seeds)):
            images, lambdas = self.generate_changes(seed, separation_vector, min_epsilon=0, max_epsilon=lambda_range, 
                                                    count=lambda_range+1, savefig=False, subfolder=subfolder, method=method, feature=feature) 
            orig_image = images[0]
            for j, (img, lmb) in enumerate(zip(images, lambdas)):
                colors_x = extcolors.extract_from_image(img, tolerance=8, limit=13)
                df_color = color_to_df(colors_x)
                top_cols = extract_color(df_color)
                top_cols_filtered = [cc[0] for cc in top_cols if (cc[1] != 0) and (cc[2] != 0)]
                harmonies = extract_harmonies(top_cols_filtered)
                cor = self.structural_coherence(img, orig_image)
                variation_scores.append([seed, lmb, cor] + [c for cc in top_cols for c in cc] + harmonies)
                
        
        df = pd.DataFrame(variation_scores, columns=['seed', 'lambda', 'SSIM', 'H1', 'S1', 'V1', 'H2', 'S2', 'V2', 'H3', 
                                            'S3', 'V3', 'H4', 'S4', 'V4', 'H5', 'S5', 'V5', 
                                            'Monochromatic', 'Analogous', 'Complementary', 'Triadic',
                                            'Split Complementary', 'Double Complementary',
                            ])
        df['Color'] = cat_from_hue(np.array(df['H1'].values), df['S1'], df['V1'])
        print(df)
        return df

if __name__ == '__main__':          
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--df_file', type=str, default='../data/color_palette00000-99999.csv')
    parser.add_argument('--model_file', type=str, default='../data/network-snapshot-005000.pkl')
    parser.add_argument('--df_separation_vectors', type=str, default='../data/interfaceGAN_separation_vector_Color.csv') #
    parser.add_argument('--max_lambda', type=int, default=15)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)

    args = parser.parse_args()
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    df = pd.read_csv(args.df_file).fillna(0)
    df['seed'] = df['fname'].str.replace('.png', '').str.replace('seed', '').astype(int)
    df = df.sort_values('seed').reset_index()
    print(df.head())
    
    with dnnlib.util.open_url(args.model_file) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,10000) for i in range(100)]
       
    df_separation_vectors = pd.read_csv(args.df_separation_vectors)

    disentanglemnet_eval = DisentanglementEvaluation(model, annotations, df, space='w', color_bins=None, colors_list=None)
    
    df_modifications = pd.DataFrame()
    ## save vector in npy and metadata in csv
    print('Now obtaining modifications using directions from', args.df_separation_vectors)
    for i,row in df_separation_vectors.iterrows():
        if 'GANSpace' in row['Method']:
            print('Skipping GANSpace')
            continue
        
        separation_vector = np.array([float(x.strip('[] ')) for x in row['Separation Vector'].replace('\n', ' ').split(' ') if x.strip('[] ') != ''])
        
        variation_scores_df = disentanglemnet_eval.obtain_changes(separation_vector, args.seeds, args.max_lambda)
        df_tmp = pd.concat([row[['Variable', 'Feature', 'Space', 'Method', 'Subfolder']], variation_scores_df], axis=1)
        df_modifications = pd.concat([df_modifications, df_tmp], axis=0)


    df = pd.DataFrame(data, columns=['Feature', 'Variable', 'Space', 'Method', 'Subfolder', 'Classes', 'Bins', 'Separation Vector'])
    df.to_csv(DATA_DIR + 'modifications_'+ args.df_separation_vectors, index=False)
    