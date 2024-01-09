#!/usr/bin/env python

### srun --pty -n 1 -c 2 --time=01:00:00 --mem=64G bash -l

import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
from os.path import join
import seaborn as sns

DATA_DIR = '../data/'
from rescoring import ReScoring

# Creating a color map
COLOR_MAP = {
        "BW": "gray",
        "Brown": "brown",
        "Yellow": "yellow",
        "Green": "green",
        "Cyan": "cyan",
        "Blue": "blue",
        "Magenta": "magenta",
        "Red": "red"
}

class AttributeDependency(ReScoring):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)
    
    def plot_hues_per_lambda(self, variations, feature, method, variable='Color'):
        variations = variations[variations['Feature'] == feature]
        for lambd, group in variations.groupby('lambda'):
            color_hues = list(group['H1'])
            hue_wheel_image = plt.imread(DATA_DIR + 'Linear_RGB_color_wheel.png')
            hue_wheel_image = resize(hue_wheel_image, (256,256))
            # Display the hue wheel image
            fig, ax = plt.subplots(dpi=80)
            ax.set_title(f'Color distribution of {feature} at lambda {lambd}')
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
                #ax.plot([end_x], [end_y], color=colors[i], marker='o', markerfacecolor=colors[i], markersize=15)  # 'w-' specifies a white line
            
            os.makedirs(join(self.repo_folder, 'figures', 'color_wheels'), exist_ok=True)
            plt.savefig(join(self.repo_folder, 'figures', 'color_wheels', f'{method}_feature_{feature}_at_lambda_{lambd}.png'))
            plt.close() 
    
        
    def heatmap_per_colors(self, variations, lambd, method, variable1='Color', variable2='Feature'):
        #options: colors heatmap to show confusion between original color, final color, direction color
        # Plotting the confusion matrix
        colors1 = list(variations[variable1].unique())
        colors2 = list(variations[variable2].unique())
        colors = [color for color in colors1 if color in colors2]
        # Creating a confusion matrix
        confusion_matrix = np.zeros((len(colors), len(colors)))

        # Filling the confusion matrix
        for i, color in enumerate(colors):
            for j, target_color in enumerate(colors):
                confusion_matrix[i, j] = variations[variations[variable2] == color][variations[variable1] == target_color].shape[0]

        plt.figure(figsize=(10, 8))
        sns.heatmap(np.array(confusion_matrix).astype(int), annot=True, fmt="d", cmap='viridis',
                    yticklabels=colors, 
                    xticklabels=colors)
        plt.title(f"Confusion Matrix ({method})")
        plt.xlabel("Final Color" if variable1 == 'Color' else variable1)
        plt.ylabel("Direction Color" if variable2 == 'Feature' else variable2)
        plt.tight_layout()
        os.makedirs(join(self.repo_folder, 'figures', 'confusion_matrices'), exist_ok=True)
        plt.savefig(join(self.repo_folder, 'figures', 'confusion_matrices', f'{method}_features_{variable1},{variable2}_at_lambda_{lambd}.png'))
        plt.close() 
    
    def launch_gui(self, matrix_col2col, which='originalxfinal'):
        #select which matches to print
        return ''
    
    

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

    disentanglemnet_eval = AttributeDependency(None, annotations, df, space='w', color_bins=None, colors_list=None)
    
    for method, group in df_modification_vectors.groupby('Method'):
        variable = list(group['Variable'].unique())[0]
        group = disentanglemnet_eval.filter_variations(group, variable)
        if 'S' in variable or 'V' in variable:
            optimal_lambda = disentanglemnet_eval.calculate_optimal_lambda(group)
        elif variable == 'Color':
            for feature in list(group['Feature'].unique()):
                optimal_lambda = disentanglemnet_eval.calculate_optimal_lambda(group[group['Feature'] == feature])
                # disentanglemnet_eval.plot_hues_per_lambda(group, feature, method)
            disentanglemnet_eval.heatmap_per_colors(group, optimal_lambda, method, variable1='Color', variable2='Feature')
            disentanglemnet_eval.heatmap_per_colors(group, optimal_lambda, method, variable1='Color', variable2='Color original')
            
       