#!/usr/bin/env python

DATA_DIR = '../data/'

import argparse
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from tqdm import tqdm
import random
from os.path import join
import os
import pickle
import sys

from disentanglement import DisentanglementBase
sys.path.append('../stylegan')
from networks_stylegan3 import *
import dnnlib 
import legacy
sys.path.append('../utils')
from utils import *

class GANSpace(DisentanglementBase):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)

    def GANSpace_separation_vectors(self, num_components):
        """Unsupervised method using PCA to find most important directions"""
        x_train, x_val, y_train, y_val = self.get_train_val()
        if self.space.lower() == 'w':
            pca = PCA(n_components=num_components)

            dims_pca = pca.fit_transform(x_train.T)
            dims_pca /= np.linalg.norm(dims_pca, axis=0)
            
            return dims_pca
        
        else:
            raise("""This space is not allowed in this function, 
                     only W""")
    
def main():
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--df_file', type=str, default='../data/color_palette00000-99999.csv')
    parser.add_argument('--model_file', type=str, default='../data/network-snapshot-005000.pkl')
    parser.add_argument('--subfolder', type=str, default='GANSpace/')
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--max_lambda', type=int, default=15)
    
    args = parser.parse_args()
    
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)
    
    kwargs = {'max_lambda':[args.max_lambda], 'num_factors':10}
    
    df = pd.read_csv(args.df_file)
    df['seed'] = df['fname'].str.replace('.png', '').str.replace('seed', '').astype(int)
    df = df.sort_values('seed').reset_index()
    print(df.head())
    
    with dnnlib.util.open_url(args.model_file) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,10000) for i in range(10)]
       
    disentanglemnet_exp = GANSpace(model, annotations, df, space='w', colors_list=None, color_bins=None, variable='Color')
            
    ## save vector in npy and metadata in csv
    print('Now obtaining separation vector for using GANSpace')
    separation_vectors = disentanglemnet_exp.GANSpace_separation_vectors(kwargs['num_factors'])
    print('Checking shape of vectors', separation_vectors.shape)
    print('Generating images with variations')
    for i in range(10):
        name = 'GANSpace_dimension_' + str(i)
        for seed in args.seeds:
            for eps in kwargs['max_lambda']:
                disentanglemnet_exp.generate_changes(seed, separation_vectors.T[i], min_epsilon=-eps, max_epsilon=eps, 
                                                     savefig=True, feature='dim' + str(i), subfolder=args.subfolder, method=name)
   
   
    np.save(DATA_DIR + 'GANSpace_separation_vectors.npy', separation_vectors)

if __name__ == "__main__":
    main()