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
from annotate_images import annotate_textile_image

sys.path.append('../disentanglement')
from disentanglement import DisentanglementBase
        
sys.path.append('../utils/')
from utils import *

sys.path.append('../stylegan')
from networks_stylegan3 import *
import dnnlib 
import legacy

class DisentanglementEvaluation(DisentanglementBase):
    def __init__(self, model, annotations, space, compute_s=False, variable='color', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, space, compute_s, variable, categorical, repo_folder)
    
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
                clr = annotate_textile_image(img, 8)
                cor = self.structural_coherence(img, orig_image)
                variation_scores.append([seed, lmb, cor, clr])
                
        
        df = pd.DataFrame(variation_scores, columns=['seed', 'lambda', 'SSIM', 'color'])
        print(df)
        return df

if __name__ == '__main__':          
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--model_file', type=str, default='../data/network-snapshot-005000.pkl')
    parser.add_argument('--df_separation_vectors', type=str, default='../data/interfaceGAN_separation_vector_color.csv') #
    parser.add_argument('--max_lambda', type=int, default=20)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)

    args = parser.parse_args()
    
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)
    
    with dnnlib.util.open_url(args.model_file) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,1000) for i in range(50)]
       
    df_separation_vectors = pd.read_csv(args.df_separation_vectors)

    disentanglemnet_eval = DisentanglementEvaluation(model, annotations, space='w')
    
    df_modifications = pd.DataFrame()
    ## save vector in npy and metadata in csv
    print('Now obtaining modifications using directions from', args.df_separation_vectors)
    for i,row in df_separation_vectors.iterrows():
        if 'GANSpace' in row['Method']:
            print('Skipping GANSpace')
            continue
        
        separation_vector = np.array([float(x.strip('[] ')) for x in row['Separation Vector'].replace('\n', ' ').split(' ') if x.strip('[] ') != ''])
        
        variation_scores_df = disentanglemnet_eval.obtain_changes(separation_vector, args.seeds, args.max_lambda)
        for col in ['Variable', 'Feature', 'Space', 'Method', 'Subfolder']:
            variation_scores_df[col] = row[col]
        df_modifications = pd.concat([df_modifications, variation_scores_df], axis=0)
        df_modifications.to_csv(DATA_DIR + 'modifications_'+ args.df_separation_vectors.split('/')[-1], index=False)
    