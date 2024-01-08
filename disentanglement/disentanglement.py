#!/usr/bin/env python

import torch
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import random
from os.path import join
import os
import pickle

import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageColor

import sys
sys.path.append('../stylegan')
from networks_stylegan3 import *
import dnnlib 
import legacy

sys.path.append('../utils')
from utils import *

class DisentanglementBase:
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device', self.device)
        self.repo_folder = repo_folder
        self.model = model.to(self.device)
        self.annotations = annotations
        self.df = df
        self.space = space
        self.categorical = categorical
        self.variable = variable
        
        self.layers = ['input', 'L0_36_512', 'L1_36_512', 'L2_36_512', 'L3_52_512',
                       'L4_52_512', 'L5_84_512', 'L6_84_512', 'L7_148_512', 'L8_148_512', 
                       'L9_148_362', 'L10_276_256', 'L11_276_181', 'L12_276_128', 
                       'L13_256_128', 'L14_256_3']
        self.layers_shapes = [4, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 362, 256, 181, 128, 128]
        self.decoding_layers = 16
        
        self.color_bins = color_bins
        self.colors_list = colors_list
        
        if 'top1col' in self.df.columns:
            self.to_hsv()
        if compute_s:
            self.get_s_space()
        
    def to_hsv(self):
        """
        The tohsv function takes the top 3 colors of each image and converts them to HSV values.
        It then adds these values as new columns in the dataframe.
        
        :param self: Allow the function to access the dataframe
        :return: The dataframe with the new columns added
        :doc-author: Trelent
        """
        print('Adding HSV encoding')
        self.df['H1'] = self.df['top1col'].map(lambda x: rgb2hsv(*hex2rgb(x))[0])
        self.df['H2'] = self.df['top2col'].map(lambda x: rgb2hsv(*hex2rgb(x))[0])
        self.df['H3'] = self.df['top3col'].map(lambda x: rgb2hsv(*hex2rgb(x))[0])
        
        self.df['S1'] = self.df['top1col'].map(lambda x: rgb2hsv(*hex2rgb(x))[1])
        self.df['S2'] = self.df['top2col'].map(lambda x: rgb2hsv(*hex2rgb(x))[1])
        self.df['S3'] = self.df['top3col'].map(lambda x: rgb2hsv(*hex2rgb(x))[1])
        
        self.df['V1'] = self.df['top1col'].map(lambda x: rgb2hsv(*hex2rgb(x))[2])
        self.df['V2'] = self.df['top2col'].map(lambda x: rgb2hsv(*hex2rgb(x))[2])
        self.df['V3'] = self.df['top3col'].map(lambda x: rgb2hsv(*hex2rgb(x))[2])
        
        print('Adding RGB encoding')
        self.df['R1'] = self.df['top1col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[0])
        self.df['R2'] = self.df['top2col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[0])
        self.df['R3'] = self.df['top3col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[0])
        
        self.df['G1'] = self.df['top1col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[1])
        self.df['G2'] = self.df['top2col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[1])
        self.df['G3'] = self.df['top3col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[1])
        
        self.df['B1'] = self.df['top1col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[2])
        self.df['B2'] = self.df['top2col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[2])
        self.df['B3'] = self.df['top3col'].map(lambda x: ImageColor.getcolor(x, 'RGB')[2])
    
    def get_s_space(self):
        """
        The get_s_space function takes the w_vectors from the annotations dictionary and uses them to generate s_vectors.
        The s_space is a space of vectors that are generated by passing each w vector through each layer of the model.
        This allows us to see how much information about a particular class is contained in different layers.
        
        :param self: Bind the method to a class
        :return: A list of lists of s vectors
        :doc-author: Trelent
        """
        print('Getting S space from W')
        ss = []
        for w in tqdm(self.annotations['w_vectors']):
            w_torch = torch.from_numpy(w).to(self.device)
            W = w_torch.expand((16, -1)).unsqueeze(0)
            s = []
            for i,layer in enumerate(self.layers):
                s.append(getattr(self.model.synthesis, layer).affine(W[0, i].unsqueeze(0)).cpu().numpy())

            ss.append(s)
        self.annotations['s_vectors'] = ss
        annotations_file = join(self.repo_folder, 'data/textile_annotated_files/seeds0000-100000_S.pkl')
        print('Storing s for future use here:', annotations_file)
        with open(annotations_file, 'wb') as f:
            pickle.dump(self.annotations, f)

    def get_encoded_latent(self):
        # ... (existing code for getX)
        if self.space.lower() == 'w':
            X = np.array(self.annotations['w_vectors']).reshape((len(self.annotations['w_vectors']), 512))
        elif self.space.lower() == 'z':
            X = np.array(self.annotations['z_vectors']).reshape((len(self.annotations['z_vectors']), 512))
        elif self.space.lower() == 's':
            concat_v = []
            for i in range(len(self.annotations['w_vectors'])):
                concat_v.append(np.concatenate(self.annotations['s_vectors'][i], axis=1))
            X = np.array(concat_v)
            X = X[:, 0, :]
        else:
            Exception("Sorry, option not available, select among Z, W, S")
            
        print('Shape embedding:', X.shape)
        return X
    
    def get_train_val(self, extremes=False):
        y = np.array(self.df[self.variable].values)
        print(y.shape, 'y')
        y_v = np.array(self.df['V1'].values)
        y_s = np.array(self.df['S1'].values)
        X = self.get_encoded_latent()[:y.shape[0], :]
        print(X.shape, 'X')
        if self.categorical:
            if 'H' in self.variable:
                y_cat = cat_from_hue(y, y_s, y_v, colors_list=self.colors_list, colors_bin=self.color_bins)   
            else:
                y_cat = y
                print('already existing')
            print('Training color distributions', pd.Series(y_cat).value_counts())
            x_train, x_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2)
        else:
            if extremes:
                # Calculate the number of elements to consider (20% of array size)
                num_elements = int(0.2 * len(y))
                # Get indices of the top num_elements maximum values
                top_indices = np.argpartition(y, -num_elements)[-num_elements:]
                bottom_indices = np.argpartition(y, -num_elements)[:num_elements]
                y_ext = y[list(top_indices) + list(bottom_indices)]
                X_ext = X[list(top_indices) + list(bottom_indices), :]
                x_train, x_val, y_train, y_val = train_test_split(X_ext, y_ext, test_size=0.2)
            else:
                x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        return x_train, x_val, y_train, y_val
        
    def get_original_position_latent(self, positive_idxs, negative_idxs, positive_vals=None, negative_vals=None):
        # Reconstruct the latent direction
        separation_vectors = []
        for i in range(len(self.colors_list)):
            if self.space.lower() == 's':
                current_idx = 0
                vectors = []
                for j, (leng, layer) in enumerate(zip(self.layers_shapes, self.layers)):
                    arr = np.zeros(leng)
                    for positive_idx in positive_idxs[i]:
                        if positive_idx >= current_idx and positive_idx < current_idx + leng:
                            arr[positive_idx - current_idx] = 1
                    for negative_idx in negative_idxs[i]:
                        if negative_idx >= current_idx and negative_idx < current_idx + leng:
                            arr[negative_idx - current_idx] = 1
                        arr = np.round(arr / (np.linalg.norm(arr) + 0.000001), 4)
                    vectors.append(arr)
                    current_idx += leng
            elif self.space.lower() == 'z' or self.space.lower() == 'w':
                vectors = np.zeros(512)
                if positive_vals:
                    vectors[positive_idxs[i]] = positive_vals[i]
                else:
                    vectors[positive_idxs[i]] = 1
                if negative_vals:
                    vectors[negative_idxs[i]] = negative_vals[i]
                else:
                    vectors[negative_idxs[i]] = -1
                vectors = np.round(vectors / (np.linalg.norm(vectors) + 0.000001), 4)
            else:
                raise Exception("""This space is not allowed in this function, 
                                    select among Z, W, S""")
            separation_vectors.append(vectors)
            
        return separation_vectors    
    
    def generate_images(self, seed, separation_vector=None, lambd=0):
        """
        The generate_original_image function takes in a latent vector and the model,
        and returns an image generated from that latent vector.
        
        
        :param z: Generate the image
        :param model: Generate the image
        :return: A pil image
        :doc-author: Trelent
        """
        G = self.model.to(self.device) # type: ignore
        # Labels.
        label = torch.zeros([1, G.c_dim], device=self.device)
        if self.space.lower() == 'z':
            vec = self.annotations['z_vectors'][seed]
            Z = torch.from_numpy(vec.copy()).to(self.device)
            if separation_vector is not None:
                change = torch.from_numpy(separation_vector.copy()).unsqueeze(0).to(self.device)
                Z = torch.add(Z, change, alpha=lambd)
            img = G(Z, label, truncation_psi=1, noise_mode='const')
        elif self.space.lower() == 'w':
            vec = self.annotations['w_vectors'][seed]
            W = torch.from_numpy(np.repeat(vec, self.decoding_layers, axis=0)
                                 .reshape(1, self.decoding_layers, vec.shape[1]).copy()).to(self.device)
            if separation_vector is not None:
                change = torch.from_numpy(separation_vector.copy()).unsqueeze(0).to(self.device)
                W = torch.add(W, change, alpha=lambd)
            img = G.synthesis(W, noise_mode='const')
        else:
            raise Exception("""This space is not allowed in this function, 
                            select either W or Z or use generate_flexible_images""")
            
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

    def forward_from_style(self, x, styles, layer):
        """Custom image generation using style layers of the network"""
        dtype = torch.float16 if (getattr(self.model.synthesis, layer).use_fp16 and self.device=='cuda') else torch.float32
        
        if getattr(self.model.synthesis, layer).is_torgb:
            weight_gain = 1 / np.sqrt(getattr(self.model.synthesis, layer).in_channels * (getattr(self.model.synthesis, layer).conv_kernel ** 2))
            styles = styles * weight_gain
        
        input_gain = getattr(self.model.synthesis, layer).magnitude_ema.rsqrt().to(dtype)
        
        # Execute modulated conv2d.
        x = modulated_conv2d(x=x.to(dtype), w=getattr(self.model.synthesis, layer).weight.to(dtype), s=styles.to(dtype),
        padding=getattr(self.model.synthesis, layer).conv_kernel-1, 
                        demodulate=(not getattr(self.model.synthesis, layer).is_torgb), 
                        input_gain=input_gain.to(dtype))
        
        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if getattr(self.model.synthesis, layer).is_torgb else np.sqrt(2)
        slope = 1 if getattr(self.model.synthesis, layer).is_torgb else 0.2
        
        x = filtered_lrelu.filtered_lrelu(x=x, fu=getattr(self.model.synthesis, layer).up_filter, fd=getattr(self.model.synthesis, layer).down_filter, 
                                            b=getattr(self.model.synthesis, layer).bias.to(x.dtype),
                                            up=getattr(self.model.synthesis, layer).up_factor, down=getattr(self.model.synthesis, layer).down_factor, 
                                            padding=getattr(self.model.synthesis, layer).padding,
                                            gain=gain, slope=slope, clamp=getattr(self.model.synthesis, layer).conv_clamp)
        return x
    
    def generate_flexible_images(self, seed, separation_vector=None, lambd=0):
        if self.space.lower() != 's':
            raise Exception("""This space is not allowed in this function, 
                            select S or use generate_images""")
            
        vec = self.annotations['w_vectors'][seed]
        w_torch = torch.from_numpy(vec).to(self.device)
        W = w_torch.expand((self.decoding_layers, -1)).unsqueeze(0)
        x = self.model.synthesis.input(W[0,0].unsqueeze(0))
        for i, layer in enumerate(self.layers[1:]):
            style = getattr(self.model.synthesis, layer).affine(W[0, i].unsqueeze(0))
            if separation_vector is not None:
                change = torch.from_numpy(separation_vector[i+1].copy()).unsqueeze(0).to(self.device)
                style = torch.add(style, change, alpha=lambd)
            x = self.forward_from_style(x, style, layer)
        
        if self.model.synthesis.output_scale != 1:
                x = x * self.model.synthesis.output_scale

        img = (x.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
            
        return img

    def generate_changes(self, seed, separation_vector, min_epsilon=-3, max_epsilon=3, count=7, savefig=True, subfolder='baseline', feature=None, method=None, save_separately=False):
        """
        The regenerate_images function takes a model, z, and decision_boundary as input.  It then
        constructs an inverse rotation/translation matrix and passes it to the generator.  The generator
        expects this matrix as an inverse to avoid potentially failing numerical operations in the network.
        The function then generates images using G(z_0, label) where z_0 is a linear combination of z and the decision boundary.
        
        :param model: Pass in the model to be used for image generation
        :param z: Generate the starting point of the line
        :param decision_boundary: Generate images along the direction of the decision boundary
        :param min_epsilon: Set the minimum value of lambda
        :param max_epsilon: Set the maximum distance from the original image to generate
        :param count: Determine the number of images that are generated
        :return: A list of images and a list of lambdas
        :doc-author: Trelent
        """
            
        lambdas = np.linspace(min_epsilon, max_epsilon, count)
        images = []
        # Generate images.
        for _, lambd in enumerate(lambdas):
            if self.space.lower() == 's':
                images.append(self.generate_flexible_images(seed, separation_vector=separation_vector, lambd=lambd))
            elif self.space.lower() in ['z', 'w']:
                images.append(self.generate_images(seed, separation_vector=separation_vector, lambd=lambd))
        
        if savefig:
            os.makedirs(join(self.repo_folder, 'figures', subfolder), exist_ok=True)
            fig, axs = plt.subplots(1, len(images), figsize=(110,20))
            title = 'Disentanglement method: '+ method + ', on feature: ' + feature + ' on space: ' + self.space + ', image seed: ' + str(seed)
            name = '_'.join([method, feature, self.space, str(seed), str(lambdas[-1])])
            fig.suptitle(title, fontsize=20)
                
            for i, (image, lambd) in enumerate(zip(images, lambdas)):
                axs[i].imshow(image)
                axs[i].set_title(np.round(lambd, 2))
            plt.tight_layout()
            plt.savefig(join(self.repo_folder, 'figures', subfolder, name+'.jpg'))
            plt.close()
            
            if save_separately:
                for i, (image, lambd) in enumerate(zip(images, lambdas)):
                    plt.imshow(image)
                    plt.tight_layout()
                    plt.savefig(join(self.repo_folder, 'figures', subfolder, name + '_' + str(lambd) + '.jpg'))
                    plt.close()
            
        return images, lambdas