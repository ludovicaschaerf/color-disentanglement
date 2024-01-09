#!/usr/bin/env python

### srun --pty -n 1 -c 2 --time=01:00:00 --mem=64G bash -l

import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import sys

from evaluation import DisentanglementEvaluation

FEATURE2TARGET = {'Brown': 17.5,'Yellow': 52.5, 'Green':110, 'Cyan':175, 'Blue':230, 'Magenta':302.5, 'Red':17.5, 'BW':0}        

class ReScoring(DisentanglementEvaluation):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)
    
    def calculate_color_score_hsv(hue, target_hue, hue_range=50):
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
    
    def plot_hues_per_lambda(variations, feature, variable='Color'):
        variations = variations[variations[variable] == feature]
        for lambd, group in variations.groupy('Lambda'):
            color_hues = [col[0] for col in group['H1']]  
            print(color_hues)
            hue_wheel_image = plt.imread(DATA_DIR + 'Linear_RGB_color_wheel.png')
            hue_wheel_image = resize(hue_wheel_image, (256,256))
            # Display the hue wheel image
            fig, ax = plt.subplots(dpi=80)
            ax.imshow(hue_wheel_image)
            ax.axis('off')  # Turn off axis
            # Assuming the center of the hue wheel and the radius are known
            center_x, center_y, radius = 128, 128, 126
            # Define your color hues in degrees
            
            # Convert degrees to radians and plot the radii
            for i, hue in enumerate(color_hues):
                # Calculate the end point of the radius
                end_x = center_x + radius * np.cos(np.radians(hue - 90))
                end_y = center_y + radius * np.sin(np.radians(hue - 90))

                # Plot a line from the center to the edge of the hue wheel
                ax.plot([center_x, end_x], [center_y, end_y], 'w-', markersize=4)  # 'w-' specifies a white line
                ax.plot([end_x], [end_y], color=colors[i], marker='o', markerfacecolor=colors[i], markersize=15)  # 'w-' specifies a white line
            
            os.makedirs(join(self.repo_folder, 'figures'), exist_ok=True)
            plt.savefig(join(self.repo_folder, 'figures', f'{feature}_at_lambda_{lambd}.png'))
            plt.close() 
    
    def re_scoring_categorical(self, variations, feature, variable='Color', broad=False):
        # Categorical evaluation 
        variations = variations[variations['Feature'] == feature]
            
        if not broad:
            variations.loc[variations[variations['Feature'] == variations[variable]].index, 'score'] = 1
            variations.loc[variations[variations['Feature'] != variations[variable]].index, 'score'] = 0
            print(variations['score'].value_counts())
        else:
            tolerance = 50 if features != 'BW' else 1
            variations['score'] = [calculate_color_score_hsv(hue, FEATURE2TARGET[feature], tolerance) for hue in variations[variable]]
            
        scores = {}
        scores_all = np.round(variations[variations['lambda'] != 0]['score'].mean(), 3)
        scores_per_lambda = [np.round(variations[variations['lambda'] == l]['score'].mean() - variations[variations['lambda'] == 0]['score'].mean(), 3)
                                for l in range(1, 16)]
        
        
        return scores_all, scores_per_lambda
            
    def re_scoring_continuous(self, variations, variable='S1'):
        # Continuous evaluation 
        # Re-scoring formula from Semantic Hierarchy emerges
        mean_var = {}
        for i, group in variations.groupby('Lambda'):
            mean_var[i] = group[variable].mean()
        scores_per_lambda = [np.round(max(mean_var[i] - mean_var[0], 0), 3)
                                for l in range(1, 16)]
        scores_all = np.round(np.mean(np.array(scores_per_lambda)), 3)
        return scores_all, scores_per_lambda
    

if __name__ == '__main__':          
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--df_file', type=str, default='../data/color_palette00000-99999.csv')
    parser.add_argument('--df_separation_vectors', type=str, default='../data/shapleyvec_separation_vector_Color.csv') #
    
    args = parser.parse_args()
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    df = pd.read_csv(args.df_file).fillna(0)
    df['seed'] = df['fname'].str.replace('.png', '').str.replace('seed', '').astype(int)
    df = df.sort_values('seed').reset_index()
    print(df.head())
    
    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,10000) for i in range(100)]
       
    df_modification_vectors = pd.read_csv('modifications_' + args.df_separation_vectors)

    disentanglemnet_eval = ReScoring(None, annotations, df, space='w', color_bins=None, colors_list=None)
    scores = []
    
    for method, group in df_modification_vectors.groupby('Method'):
        variable = list(group['Variable'].unique())[0]
        if 'S' in variable or 'V' in variable:
            scores_all, scores_per_lambda = disentanglemnet_eval.re_scoring_continuous(group,
                                                                                       variable=variable)
            scores.append([method, variable, None, None, scores_all] + scores_per_lambda)
        else:
            for feature in list(group['Feature'].unique()):
                scores_all, scores_per_lambda = disentanglemnet_eval.re_scoring_categorical(group, feature, variable, False)
                scores.append([method, variable, feature, False, scores_all] + scores_per_lambda)
                if variable == 'Color':
                    scores_all_b, scores_per_lambda_b = disentanglemnet_eval.re_scoring_categorical(group, feature, variable, True)
                    plot_hues_per_lambda(group, feature)
                    scores.append([method, variable, feature, True, scores_all_b] + scores_per_lambda_b)
                
        df = pd.DataFrame(scores, columns=['Method', 'Variable', 'Feature', 'Broad', 'Total Score'] + [f'Score Lambda {i}' for i in range(1,16)])
        df.to_csv(DATA_DIR + 'scores_'+ args.df_separation_vectors, index=False)
    
        

