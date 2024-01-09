#!/usr/bin/env python

### srun --pty -n 1 -c 2 --time=01:00:00 --mem=64G bash -l

import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import sys

from evaluation import DisentanglementEvaluation

DATA_DIR = '../data/'
FEATURE2TARGET = {'Brown': 17.5,'Yellow': 52.5, 'Green':110, 'Cyan':175, 'Blue':230, 'Magenta':302.5, 'Red':17.5, 'BW':0}        

class ReScoring(DisentanglementEvaluation):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)
    
    def calculate_color_score_hsv(self, hue, target_hue, hue_range=50):
        """
        Calculate the color score based on how close the hue is to a target hue.
        :param hue: Hue value of the color (0 to 360)
        :param target_hue: The target hue to compare against
        :param hue_range: Acceptable range around the target hue
        :return: Color score (higher is closer to target color)
        """
        adjusted_hue = min(abs(hue - target_hue), 360 - abs(hue - target_hue))
        if adjusted_hue <= hue_range:
            return 1 - (adjusted_hue / hue_range)
        else:
            return 0
        
    def add_original_color(self, variations):
        L0 = variations[variations['lambda'] == 0]
        SEED2COLOR = {seed:orig for orig, seed in zip(L0['Color'], L0['seed'])}
        variations['Color original'] = [SEED2COLOR[seed] for seed in variations['seed']]
        return variations
    
    def filter_variations(self, variations, variable):
        variations = self.add_original_color(variations)
        print('total', variations.shape)
        accepted_seeds = variations[variations['lambda'] == 0][variations['Feature'] != variations[variable]]['seed']
        variations = variations[variations['seed'].isin(accepted_seeds)]
        print('after filtering already correct ones', variations.shape)
        variations = variations[variations['Color original'] != 'BW']
        print('number of non-BW originally', variations.shape)
        return variations
        
    def calculate_optimal_lambda(self, variations):
        L0 = variations[variations['lambda'] == 0]
        SEED2SSIM = {seed:np.round(orig) for orig, seed in zip(L0['SSIM'], L0['seed'])}
        variations['SSIM_change'] = [(SEED2SSIM[seed] - ssim)/ssim for ssim, seed in zip(variations['SSIM'], variations['seed'])]
        print(variations['SSIM_change'])
        optimal_lambda = max([i for i,group in variations.groupby('lambda') if np.mean(np.abs(group['SSIM_change'])) < 0.25])
        return optimal_lambda
        
    def re_scoring_categorical(self, variations, feature, variable='Color', broad=False):
        # Categorical evaluation 
        variations = variations[variations['Feature'] == feature]
        variations.loc[variations[variations['Feature'] == variations[variable]].index, 'score'] = 1
        variations.loc[variations[variations['Feature'] != variations[variable]].index, 'score'] = 0
        print(variations['score'].value_counts())
        if broad:
            tolerance = 100 if feature != 'BW' else 1
            variations['score'] = [self.calculate_color_score_hsv(hue, FEATURE2TARGET[feature], tolerance) if score != 1 else score for hue, score in zip(variations['H1'], variations['score'])]
            print(variations['score'].value_counts())
            
        scores = {}
        scores_all = np.round(variations[variations['lambda'] != 0]['score'].mean(), 3)
        scores_per_lambda = [np.round(variations[variations['lambda'] == l]['score'].mean(), 3)
                                for l in range(1, 16)]
        
        
        return scores_all, scores_per_lambda
            
    def re_scoring_continuous(self, variations, variable='S1'):
        # Continuous evaluation 
        # Re-scoring formula from Semantic Hierarchy emerges
        mean_var = {}
        for i, group in variations.groupby('lambda'):
            mean_var[i] = group[variable].mean()
        scores_per_lambda = [np.round(max(mean_var[i] - mean_var[0], 0), 3)
                                for l in range(1, 16)]
        scores_all = np.round(np.mean(np.array(scores_per_lambda)), 3)
        return scores_all, scores_per_lambda
    

if __name__ == '__main__':          
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--df_file', type=str, default='../data/color_palette00000-99999.csv')
    parser.add_argument('--df_modification_vectors', type=str, default='../data/modifications_shapleyvec_separation_vector_Color.csv') #
    
    args = parser.parse_args()
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    df = pd.read_csv(args.df_file).fillna(0)
    df['seed'] = df['fname'].str.replace('.png', '').str.replace('seed', '').astype(int)
    df = df.sort_values('seed').reset_index()
    print(df.head())
    
    df_modification_vectors = pd.read_csv(args.df_modification_vectors)

    disentanglemnet_eval = ReScoring(None, annotations, df, space='w', color_bins=None, colors_list=None)
    scores = []
    
    for method, group in df_modification_vectors.groupby('Method'):
        variable = list(group['Variable'].unique())[0]
        group = disentanglemnet_eval.filter_variations(group, variable)
        if 'S' in variable or 'V' in variable:
            optimal_lambda = disentanglemnet_eval.calculate_optimal_lambda(group)
            scores_all, scores_per_lambda = disentanglemnet_eval.re_scoring_continuous(group,
                                                                                       variable=variable)
            scores.append([method, variable, None, None, optimal_lambda, scores_all] + scores_per_lambda)
        else:
            for feature in list(group['Feature'].unique()):
                scores_all, scores_per_lambda = disentanglemnet_eval.re_scoring_categorical(group, feature, variable, False)
                optimal_lambda = disentanglemnet_eval.calculate_optimal_lambda(group[group['Feature'] == feature])
                scores.append([method, variable, feature, False, optimal_lambda, scores_all] + scores_per_lambda)
                if variable == 'Color':
                    scores_all_b, scores_per_lambda_b = disentanglemnet_eval.re_scoring_categorical(group, feature, variable, True)
                    scores.append([method, variable, feature, True, optimal_lambda, scores_all_b] + scores_per_lambda_b)
                
        df = pd.DataFrame(scores, columns=['Method', 'Variable', 'Feature', 'Broad', 'Optimal lambda', 'Total Score'] + [f'Score lambda {i}' for i in range(1,16)])
        df['Final Score'] = [max([row[f'Score lambda {j+1}'] for j in range(int(row["Optimal lambda"]))]) for i,row in df.iterrows()]
        df.to_csv(DATA_DIR + 'scores_'+ args.df_modification_vectors.split('/')[-1], index=False)
    
        

