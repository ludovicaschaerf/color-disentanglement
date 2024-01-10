import numpy as np
import pandas as pd
import random
from os.path import join
import os
import pickle
from glob import glob
import json
import sys
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)

EXPORT_DIR = 'LatentSpaceNavigator/ui/public/'
DATA_DIR = '../data/'

sys.path.append('../disentanglement')
from disentanglement import DisentanglementBase
        
sys.path.append('../utils/')
from utils import *

sys.path.append('../stylegan')
from networks_stylegan3 import *
import dnnlib 
import legacy

annotations_file = DATA_DIR + 'seeds0000-100000.pkl'
df_file = DATA_DIR + 'color_palette00000-99999.csv'
df = pd.read_csv(df_file)

df_scores_vectors = glob(DATA_DIR + 'scores_modifications*')
df_scores = pd.DataFrame()
for df_score in df_scores_vectors:
    df_score_l = pd.read_csv(df_score)
    df_scores = pd.concat([df_scores, df_score_l], axis=0)

df_scores = df_scores.sort_values('Final Score', ascending=False).reset_index()
df_scores.loc[df_scores['Variable'] == 'V1', 'Feature'] = 'V1'
df_scores.loc[df_scores['Variable'] == 'S1', 'Feature'] = 'S1'
df_scores = df_scores.groupby('Feature').first()


separation_vectors = glob(DATA_DIR + '*_separation_vector*.csv')
df_sep_vecs = pd.DataFrame()
for df_sep_vec in separation_vectors:
    if 'modifications' not in df_sep_vec:
        df_sep_vec_l = pd.read_csv(df_sep_vec)
        df_sep_vecs = pd.concat([df_sep_vecs, df_sep_vec_l], axis=0)
df_sep_vecs.loc[df_sep_vecs['Variable'] == 'V1', 'Feature'] = 'V1'
df_sep_vecs.loc[df_sep_vecs['Variable'] == 'S1', 'Feature'] = 'S1'

df_scores = df_scores.merge(df_sep_vecs, left_on=['Feature','Variable','Method'], right_on=['Feature','Variable','Method'], how='left')
print(df_scores[['Feature', 'Variable', 'Method', 'Final Score']].head(10), df_scores.shape)


space = 'w'
colors_list = ['Brown', 'Yellow', 'Green', 'Cyan', 'Blue', 'Magenta', 'Red', 'BW']
color_bins = None
variable = 'Color'

with open(annotations_file, 'rb') as f:
    annotations = pickle.load(f)

disentanglemnet_exp = DisentanglementBase(None, annotations, df, space=space, 
                                          colors_list=colors_list, color_bins=color_bins, 
                                          variable=variable)

x_train, x_val, y_train, y_val = disentanglemnet_exp.get_train_val()
# pca.fit(X)

pca = pickle.load(open("pca.pkl","rb"))
X_3d = pca.transform(x_train)

points_per_color = {}
for color in colors_list:
    y_color_where = np.where(y_train == color)
    X_color = X_3d[y_color_where]
    x_col = X_color[:100, :]
    points_per_color[color] = [[float(x) for x in xx] for xx in x_col]

# with open(EXPORT_DIR + "3d_points.json", "w") as outfile: 
#     json.dump(points_per_color, outfile)
    

points_per_color = {}
for color in colors_list:
    y_color_where = np.where(y_train == color)
    X_color = x_train[y_color_where]
    x_col = X_color[:100, :]
    points_per_color[color] = [[float(x) for x in xx] for xx in x_col]

# with open(EXPORT_DIR + "512d_points.json", "w") as outfile: 
#     json.dump(points_per_color, outfile)
    
Xx = np.array([np.array([float(x.strip('[] ')) for x in row['Separation Vector'].replace('\n', ' ').split(' ') if x.strip('[] ') != '']) for i, row in df_scores.iterrows()])

sampled_points_3d = {}
color_orig_vec = {}
for sep_vec, col in zip(Xx, df_scores['Feature']):
    if col == 'S1' or col == 'V1':
        color_orig_vec['-'+col] = list(-sep_vec)
    elif col == 'BW':
        color_orig_vec['grey'] = list(sep_vec)
    else:
        color_orig_vec[col.lower()] = list(sep_vec)
    
    # Define the direction vector (unit vector)
    direction_vector = sep_vec  # Normalize to ensure it's a unit vector

    # Starting point (you can choose any starting point)
    starting_point = np.zeros(512)  # Replace this with your desired starting point

    # Number of points to sample
    num_points = 100  # Adjust as needed

    # Sample points along the direction vector
    sampled_points = []
    for i in range(num_points):
        alpha = i  # You can adjust this factor to control the spacing between points
        sampled_point = starting_point + alpha * direction_vector
        sampled_points.append(sampled_point)
        sampled_point = starting_point - alpha * direction_vector
        sampled_points.append(sampled_point)
        
    # Convert the sampled points to a NumPy array
    sampled_points = np.array(sampled_points)
    sampled_points_3d_vec = pca.transform(sampled_points)
    sampled_points_3d[col] = sampled_points_3d_vec
    # Now, 'sampled_points' contains the sampled points along the direction vector
    

sampled_points_3d_unit = {}

for sep_vec, col in zip(Xx, df_scores['Feature']):
    # Define the direction vector (unit vector)
    direction_vector = sep_vec  # Normalize to ensure it's a unit vector

    # Starting point (you can choose any starting point)
    starting_point = np.zeros(512)  # Replace this with your desired starting point

    # Number of points to sample
    num_points = 1  # Adjust as needed

    # Sample points along the direction vector
    sampled_points = []
    for i in range(num_points):
        alpha = i+1  # You can adjust this factor to control the spacing between points
        sampled_point = starting_point + alpha * direction_vector  
    # Convert the sampled points to a NumPy array
    sampled_points = np.array(sampled_point)
    starting_point_3d = pca.transform(starting_point.reshape(1,-1))[0]
    sampled_points_3d_vec = pca.transform(sampled_points.reshape(1,-1))
    sampled_points_3d_unit[col] = list(sampled_points_3d_vec[0])
    # Now, 'sampled_points' contains the sampled points along the direction vector

with open(EXPORT_DIR + "3d_directions.json", "r") as infile: 
    direction_points_3d = json.load(infile)
    
for k,v in sampled_points_3d_unit.items():
    if k == 'S1' or k == 'V1':
        direction_points_3d['-'+k]['direction'] = list(-(np.array(v) - np.array(starting_point_3d)))
    direction_points_3d[k]["direction"] = list(np.array(v) - np.array(starting_point_3d))


with open(EXPORT_DIR + "3d_directions.json", "w") as outfile: 
    json.dump(direction_points_3d, outfile)
    
with open(EXPORT_DIR + "512d_directions.json", "w") as outfile: 
    json.dump(color_orig_vec, outfile)