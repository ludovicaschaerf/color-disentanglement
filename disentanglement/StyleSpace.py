#!/usr/bin/env python

DATA_DIR = '../data/'

import argparse
import numpy as np
import pandas as pd

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

class StyleSpace(DisentanglementBase):
    def __init__(self, model, annotations, space, compute_s=False, variable='color', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, space, compute_s, variable, categorical, repo_folder)
    
    
    def StyleSpace_separation_vector(self, sign=True, num_factors=20, cutout=0.25):
        """ Formula from StyleSpace Analysis """
        x_train, x_val, y_train, y_val = self.get_train_val()
        
        positive_idxs = []
        negative_idxs = []
        for color in self.colors_list:
            x_col = x_train[np.where(y_train == color)]
            mp = np.mean(x_train, axis=0)
            sp = np.std(x_train, axis=0)
            de = (x_col - mp) / sp
            meu = np.mean(de, axis=0)
            seu = np.std(de, axis=0)
            if sign:
                thetau = meu / seu
                positive_idx = np.argsort(thetau)[-num_factors//2:]
                negative_idx = np.argsort(thetau)[:num_factors//2]
                
            else:
                thetau = np.abs(meu) / seu
                positive_idx = np.argsort(thetau)[-num_factors:]
                negative_idx = []
                

            if cutout:
                beyond_cutout = np.where(np.abs(thetau) > cutout)
                positive_idx = np.intersect1d(positive_idx, beyond_cutout)
                negative_idx = np.intersect1d(negative_idx, beyond_cutout)
                
                if len(positive_idx) == 0 and len(negative_idx) == 0:
                    print('No values found above the current cutout', cutout, 'for color', color, '.\n Disentangled vector will be all zeros.' )
                
            positive_idxs.append(positive_idx)
            negative_idxs.append(negative_idx)
        
        separation_vectors = self.get_original_position_latent(positive_idxs, negative_idxs)
        return separation_vectors

def main():
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--model_file', type=str, default='../data/network-snapshot-005000.pkl')
    parser.add_argument('--subfolder', type=str, default='interfaceGAN/color/')
    parser.add_argument('--variable', type=str, default='color')
    parser.add_argument('--max_lambda', type=int, default=15)
    parser.add_argument('--continuous_experiment', type=bool, default=False)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)

    args = parser.parse_args()
    
    kwargs = {'sign':[True], 'num_factors':[1, 5, 10, 20], 
              'max_lambda':[args.max_lambda], 'cutout':[False]}
    
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)
    
    with dnnlib.util.open_url(args.model_file) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,1000) for i in range(20)]
          
    disentanglemnet_exp = StyleSpace(model, annotations, space='w', 
                                    categorical=True, ##continous experiment not allowed for StyleSpace method
                                    variable=args.variable)
    data = []
    ## save vector in npy and metadata in csv
    print('Now obtaining separation vector for using StyleSpace on task', args.variable)
    for sign in kwargs['sign']:
        for num_factors in kwargs['num_factors']:
            for cutout in kwargs['cutout']:
                separation_vectors = disentanglemnet_exp.StyleSpace_separation_vector(sign=sign, num_factors=num_factors, cutout=cutout)
            
                features = list(set(annotations['color']))
                print('Checking length of outputted vectors', len(separation_vectors), len(features))
                for i in range(len(separation_vectors)):
                    print(separation_vectors[i])
                    print(f'Generating images with variations for {args.variable}, feature: {features[i]}')
                    name = 'StyleSpace_' + str(sign) + '_' + str(num_factors) + '_' + str(len(features)) + '_' + str(args.variable) + '_' + str(cutout)
                    data.append([features[i], args.variable, 'w', name, args.subfolder, ', '.join(features), str(separation_vectors[i])])
                    for seed in args.seeds:
                        for eps in kwargs['max_lambda']:
                            disentanglemnet_exp.generate_changes(seed, separation_vectors[i], min_epsilon=-eps,
                                                                    max_epsilon=eps, savefig=True, 
                                                                    feature=str(features[i]), subfolder=args.subfolder, method=name)

   
    df = pd.DataFrame(data, columns=['Feature', 'Variable', 'Space', 'Method', 'Subfolder', 'Classes', 'Separation Vector'])
    df.to_csv(DATA_DIR + 'stylespace_separation_vector_'+ args.variable +'.csv', index=False)
    np.save(DATA_DIR + 'stylespace_separation_vector_'+ args.variable +'.npy', separation_vectors)

if __name__ == "__main__":
    main()
