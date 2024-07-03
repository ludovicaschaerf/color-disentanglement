#!/usr/bin/env python

### srun --pty -n 1 -c 2 --time=01:00:00 --mem=64G bash -l

import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, '../annotations')
from color_utils import color_hue_distance

from evaluation import DisentanglementEvaluation

DATA_DIR = '../data/'

with open(DATA_DIR + 'quantized_colors_and_names.pkl', 'rb') as infile:
    colors_dict = pickle.load(infile) 

FEATURE2TARGET = {k:v for k,v in zip(colors_dict['names'], colors_dict['RGB'])}

class ReScoring(DisentanglementEvaluation):
    def __init__(self, model, annotations, space, compute_s=False, variable='color', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, space, compute_s, variable, categorical, repo_folder)
    
    def calculate_color_score_hsv(self, hue, target_hue, hue_range=50):
        """
        Calculate the color score based on how close the hue is to a target hue.
        :param hue: Hue value of the color (0 to 360)
        :param target_hue: The target hue to compare against
        :param hue_range: Acceptable range around the target hue
        :return: Color score (higher is closer to target color)
        """
        if type(hue) != int:
            return max(0, 1 - color_hue_distance(hue, target_hue, lum_strength=0.1, to_hsv=True))
        else:
            adjusted_hue = min(abs(hue - target_hue), 360 - abs(hue - target_hue))
            if adjusted_hue <= hue_range:
                return 1 - (adjusted_hue / hue_range)
            else:
                return 0
        
    def add_original_color(self, variations):
        L0 = variations[variations['lambda'] == 0]
        SEED2COLOR = {seed:orig for orig, seed in zip(L0['color'], L0['seed'])}
        variations['color original'] = [SEED2COLOR[seed] for seed in variations['seed']]
        return variations
    
    def filter_variations(self, variations, variable):
        variations = self.add_original_color(variations)
        print('total', variations.shape)
        accepted_seeds = variations[variations['lambda'] == 0][variations['Feature'] != variations[variable]]['seed']
        variations = variations[variations['seed'].isin(accepted_seeds)]
        print('after filtering already correct ones', variations.shape)
        variations = variations[variations['color original'] != 'BW']
        print('number of non-BW originally', variations.shape)
        return variations
        
    def calculate_optimal_lambda(self, variations):
        L0 = variations[variations['lambda'] == 0]
        SEED2SSIM = {seed:np.round(orig) for orig, seed in zip(L0['SSIM'], L0['seed'])}
        variations['SSIM_change'] = [(SEED2SSIM[seed] - ssim)/ssim for ssim, seed in zip(variations['SSIM'], variations['seed'])]
        print(variations['SSIM_change'])
        optimal_lambda = max([i for i,group in variations.groupby('lambda') if np.mean(np.abs(group['SSIM_change'])) < 0.25])
        return optimal_lambda
        
    def re_scoring_categorical(self, variations, feature, variable='color', broad=False, max_lambda=15):
        # Categorical evaluation 
        variations = variations[variations['Feature'] == feature]
        variations.loc[variations[variations['Feature'] == variations[variable]].index, 'score'] = 1
        variations.loc[variations[variations['Feature'] != variations[variable]].index, 'score'] = 0
        print(variations['score'].value_counts())
        if broad:
            tolerance = 100 if feature != 'BW' else 1
            variations['score'] = [self.calculate_color_score_hsv(FEATURE2TARGET[hue], FEATURE2TARGET[feature], tolerance) if score != 1 else score for hue, score in zip(variations['color'], variations['score'])]
            print(variations['score'].value_counts())
            
        scores = {}
        scores_all = np.round(variations[variations['lambda'] != 0]['score'].mean(), 3)
        scores_per_lambda = [np.round(variations[variations['lambda'] == l]['score'].mean(), 3)
                                for l in range(1, max_lambda+1)]
        
        
        return scores_all, scores_per_lambda
            
    def re_scoring_continuous(self, variations, variable='S1'):
        # Continuous evaluation 
        # Re-scoring formula from Semantic Hierarchy emerges
        mean_var = {}
        for i, group in variations.groupby('lambda'):
            mean_var[i] = group[variable].mean()
        scores_per_lambda = [np.round(max(mean_var[l] - mean_var[0], 0) / 100, 3)
                                for l in range(1, 16)]
        scores_all = np.round(np.mean(np.array(scores_per_lambda)), 3)
        return scores_all, scores_per_lambda
    

if __name__ == '__main__':          
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--df_modification_vectors', type=str, default='../data/modifications_shapleyvec_separation_vector_Color.csv') #
    parser.add_argument('--max_lambda', type=int, default=25)
    
    args = parser.parse_args()
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    df_modification_vectors = pd.read_csv(args.df_modification_vectors)

    disentanglemnet_eval = ReScoring(None, annotations, space='w')
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
                scores_all, scores_per_lambda = disentanglemnet_eval.re_scoring_categorical(group, feature, variable, False, max_lambda=args.max_lambda)
                optimal_lambda = disentanglemnet_eval.calculate_optimal_lambda(group[group['Feature'] == feature])
                scores.append([method, variable, feature, False, optimal_lambda, scores_all] + scores_per_lambda)
                if variable == 'color':
                    scores_all_b, scores_per_lambda_b = disentanglemnet_eval.re_scoring_categorical(group, feature, variable, True)
                    scores.append([method, variable, feature, True, optimal_lambda, scores_all_b] + scores_per_lambda_b)
                
        df = pd.DataFrame(scores, columns=['Method', 'Variable', 'Feature', 'Broad', 'Optimal lambda', 'Total Score'] + [f'Score lambda {i}' for i in range(1,args.max_lambda + 1)])
        df['Final Score'] = [max([row[f'Score lambda {j}'] for j in range(1, int(row["Optimal lambda"]) + 1)]) for i,row in df.iterrows()]
        df.to_csv(DATA_DIR + 'scores_'+ args.df_modification_vectors.split('/')[-1], index=False)
    
        

