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
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import shap

from disentanglement import DisentanglementBase
sys.path.append('../stylegan')
from networks_stylegan3 import *
import dnnlib 
import legacy
sys.path.append('../utils')
from utils import *

class ShapleyVec(DisentanglementBase):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)

    def extract_percent_values(self, values, ratio=0.8):
        # Sort values in descending order
        shap_idxs = np.argsort(abs(values))[::-1]
        shap_vals = np.sort(abs(values))[::-1]
        # Calculate total sum
        total_sum = sum(shap_vals)
    
        # Target sum (80% of total)
        target_sum = ratio * total_sum
        print(total_sum, target_sum)
        # Accumulate values until the sum reaches 80% of the total
        accumulated_sum = 0
        values_percent_idx = []
        for value, idx in zip(shap_vals, shap_idxs):
            if accumulated_sum <= target_sum:
                accumulated_sum += value
                values_percent_idx.append(idx)
            else:
                break
    
        print(len(values_percent_idx), 'subselected')
        return values_percent_idx
        
    def Shapley_separation_vector(self, variation=0.8, method='LR', weighted=True):
        """Test method that uses Shapley value to find most important channels"""
        x_train, x_val, y_train, y_val = self.get_train_val()
        le = LabelEncoder()
        le.fit(y_train)
        
        print(x_train.shape)
        ## use InterfaceGAN 
        
        if method == 'LR':
            model = LogisticRegression(random_state=0, C=0.1)
            model.fit(x_train, le.transform(y_train))
            performance = np.round(model.score(x_val, le.transform(y_val)), 2)
            explainer = shap.LinearExplainer(model, x_train, feature_dependence="independent")
            shap_values = explainer.shap_values(x_train,)
        
        else:
            raise Exception("""This method is not allowed for this technique. Select LR""")
        
        #if self.device == 'cuda':
        #    explainer = shap.GPUTreeExplainer(model)
        #else:
            
        positive_idxs = []
        negative_idxs = []
        
        if weighted:
            positive_vals = []
            negative_vals = []
        for class_ in tqdm(le.classes_):
            label = le.transform(np.array([class_]))[0]
            print(class_, label)
            shap_vals = shap_values[label].mean(axis=0)
            
            shap_idxs = self.extract_percent_values(shap_vals, variation)
             
            if weighted:
                model_masked = LogisticRegression(random_state=0, C=0.1)
                x_train_masked = x_train[:, shap_idxs]
                y_train_class = y_train == class_
                print(x_train_masked.shape, y_train_class.shape)
                model_masked.fit(x_train_masked, y_train_class)
            
                coeff_vals = model_masked.coef_ / np.linalg.norm(model_masked.coef_)
                positive_idx = np.array(shap_idxs)[np.where(coeff_vals[0,:] > 0)[0]]
                positive_idxs.append(positive_idx)
                negative_idx = np.array(shap_idxs)[np.where(coeff_vals[0,:] <= 0)[0]]
                negative_idxs.append(negative_idx)
                
                coeff_vals_pos = [val for val in coeff_vals[0,:] if val > 0]
                coeff_vals_neg = [val for val in coeff_vals[0,:] if val <= 0]
                positive_vals.append(coeff_vals_pos)
                negative_vals.append(coeff_vals_neg)
            
            else:
                positive_idxs.append(shap_idxs)  
        
        #shap.summary_plot(shap_values, x_val)
        #plt.savefig('shap.png')
        if weighted:
            separation_vectors = self.get_original_position_latent(positive_idxs, negative_idxs, positive_vals, negative_vals)
        else:
            separation_vectors = self.get_original_position_latent(positive_idxs, negative_idxs)
        
        if 'H' in self.variable or 'Color' in self.variable:
            idxs = [list(le.classes_).index(col) for col in self.colors_list]
            separation_vectors = np.array(separation_vectors)[idxs]
        
        return separation_vectors, performance

   
def main():
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--df_file', type=str, default='../data/color_palette00000-99999.csv')
    parser.add_argument('--model_file', type=str, default='../data/network-snapshot-005000.pkl')
    parser.add_argument('--colors_list', nargs='+', default=['Brown', 'Yellow', 'Green', 'Cyan', 'Blue', 'Magenta', 'Red', 'BW'])
    parser.add_argument('--color_bins', nargs='+', type=int, default=[0, 35, 70, 150, 200, 260, 345, 360])
    parser.add_argument('--subfolder', type=str, default='StyleSpace/color/')
    parser.add_argument('--variable', type=str, default='Color')
    parser.add_argument('--max_lambda', type=int, default=15)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)

    args = parser.parse_args()
    
    kwargs = {'weighted':[True], 'variation':[0.25, 0.5], 
              'max_lambda':[args.max_lambda], 'method':['LR']}
    
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    df = pd.read_csv(args.df_file).fillna(0)
    df['seed'] = df['fname'].str.replace('.png', '').str.replace('seed', '').astype(int)
    df = df.sort_values('seed').reset_index()
    print(df.head())
    
    if 'Color' in df.columns:
        args.colors_list = df['Color'].unique()
        
    with dnnlib.util.open_url(args.model_file) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,10000) for i in range(10)]
       
    disentanglemnet_exp = ShapleyVec(model, annotations, df, space='w', 
                                       colors_list=args.colors_list, color_bins=args.color_bins,
                                       categorical=True, ##continous experiment not implemented for ShapleyVec method
                                       variable=args.variable)
    data = []
    ## save vector in npy and metadata in csv
    print('Now obtaining separation vector for using ShapleyVec on task', args.variable)
    for weighted in kwargs['weighted']:
        for variation in kwargs['variation']:
            for met in kwargs['method']:
                separation_vectors, performance = disentanglemnet_exp.Shapley_separation_vector(variation=variation, weighted=weighted, method=met)
                if not weighted:
                    if variation == 0.4:
                        continue
                features = df[args.variable].unique()
                print('Checking length of outputted vectors', len(separation_vectors), len(args.colors_list))
                for i in range(len(separation_vectors)):
                    print(separation_vectors[i])
                    print(f'Generating images with variations for {args.variable}, feature: {features[i]}')
                    name = 'ShapleyVec_' + str(weighted) + '_' + str(variation) + '_' + str(len(args.colors_list)) + '_' + str(args.variable) + '_' + str(met)
                    data.append([features[i], args.variable, 'w', name, args.subfolder, ', '.join(args.colors_list), str(args.color_bins), str(separation_vectors[i])])
                    for seed in args.seeds:
                        for eps in kwargs['max_lambda']:
                            disentanglemnet_exp.generate_changes(seed, separation_vectors[i], min_epsilon=-eps,
                                                                    max_epsilon=eps, savefig=True, 
                                                                    feature=str(features[i]), subfolder=args.subfolder, method=name)

   
    df = pd.DataFrame(data, columns=['Feature', 'Variable', 'Space', 'Method', 'Subfolder', 'Classes', 'Bins', 'Separation Vector'])
    df.to_csv(DATA_DIR + 'shapleyvec_separation_vector_'+ args.variable +'.csv', index=False)
    np.save(DATA_DIR + 'shapeyvec_separation_vector_'+ args.variable +'.npy', separation_vectors)

if __name__ == "__main__":
    main()
   