import numpy as np
import pandas as pd
import random
from os.path import join
import os
import pickle
import json
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)

EXPORT_DIR = 'LatentSpaceNavigator/src/public/'
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
df_separation_vectors = DATA_DIR  + 'best_separation_vectors.csv'
space = 'w'
colors_list = None
color_bins = None
variable = 'Color'

with open(annotations_file, 'rb') as f:
    annotations = pickle.load(f)


df = pd.read_csv(df_file).fillna(0)
df['seed'] = df['fname'].str.replace('.png', '').str.replace('seed', '').astype(int)
df = df.sort_values('seed').reset_index()
print(df.head())
      
df_separation_vectors = pd.read_csv(df_separation_vectors)

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
    print(X_color.shape)
    x_col = X_color[:100, :]
    print(x_col.shape)
    points_per_color[color] = [[float(x) for x in xx] for xx in x_col]

with open(EXPORT_DIR + "3d_points.json", "w") as outfile: 
    json.dump(points_per_color, outfile)
    

points_per_color = {}
for color in colors_list:
    y_color_where = np.where(y_train == color)
    X_color = X[y_color_where]
    print(X_color.shape)
    x_col = X_color[:100, :]
    print(x_col.shape)
    points_per_color[color] = [[float(x) for x in xx] for xx in x_col]

with open(EXPORT_DIR + "512d_points.json", "w") as outfile: 
    json.dump(points_per_color, outfile)
    
Xx = np.array([np.array([float(x.strip('[] ')) for x in row['Separation Vector'].replace('\n', ' ').split(' ') if x.strip('[] ') != '']) for i, row in df_separation_vectors.iterrows()])

sampled_points_3d = {}
color_orig_vec = {}
for sep_vec, col in zip(Xx, df_separation_vectors['Feature']):
    color_orig_vec[col] = list(sep_vec)
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
    
direction_points_3d = {k:list(np.array(v) - np.array(starting_point_3d)) for k,v in sampled_points_3d_unit.items()}

with open(EXPORT_DIR + "3d_directions.json", "w") as outfile: 
    json.dump(direction_points_3d, outfile)
    
with open(EXPORT_DIR + "512d_directions.json", "w") as outfile: 
    json.dump(color_orig_vec, outfile)