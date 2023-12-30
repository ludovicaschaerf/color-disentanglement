
#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import random
from os.path import join
import os
import pickle

from disentanglement import DisentanglementBase
DATA_DIR = '../data/'
import sys
sys.path.append('../stylegan')
from networks_stylegan3 import *
import dnnlib 
import legacy

sys.path.append('../utils')
from utils import *


class InterfaceGAN(DisentanglementBase):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)
    
    def InterFaceGAN_separation_vector(self, method='LR', C=0.1, extremes=False):
        """
        Method from InterfaceGAN
        The get_separation_space function takes in a type_bin, annotations, and df.
        It then samples 100 of the most representative abstracts for that type_bin and 100 of the least representative abstracts for that type_bin.
        It then trains an SVM or logistic regression model on these 200 samples to find a separation space between them. 
        The function returns this separation space as well as how many nodes are important in this separation space.
        
        :param type_bin: Select the type of abstracts to be used for training
        :param annotations: Access the z_vectors
        :param df: Get the abstracts that are used for training
        :param samples: Determine how many samples to take from the top and bottom of the distribution
        :param method: Specify the classifier to use
        :param C: Control the regularization strength
        :return: The weights of the linear classifier
        :doc-author: Trelent
        """
        x_train, x_val, y_train, y_val = self.get_train_val(extremes=extremes)
        
        if self.categorical:
            
            if method == 'SVM':
                svc = SVC(gamma='auto', kernel='linear', random_state=0, C=C)
                svc.fit(x_train, y_train)
                idxs = [list(svc.classes_).index(col) for col in self.colors_list]
                performance = np.round(svc.score(x_val, y_val), 2)
                print('Val performance SVM', performance)
                svc_coef = svc.coef_[idxs]
                return svc_coef / np.linalg.norm(svc_coef), performance
            
            elif method == 'LR':
                clf = LogisticRegression(random_state=0, C=C)
                clf.fit(x_train, y_train)
                idxs = [list(clf.classes_).index(col) for col in self.colors_list]
                print(list(clf.classes_), self.colors_list, idxs)
                performance = np.round(clf.score(x_val, y_val), 2)
                print('Val performance logistic regression', performance)
                clf_coef = clf.coef_[idxs]
                return clf_coef / np.linalg.norm(clf_coef), performance
            else:
                raise Exception("""This method is not allowed for this technique. Select SVM or LR""")
        
        else:
            clf = LinearRegression()
            clf.fit(x_train, y_train)
            performance = np.round(clf.score(x_val, y_val), 2)
            print('Val performance linear regression', performance)
            return clf.coef_ / np.linalg.norm(clf.coef_), performance
    
## store npy per direction

def main():
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='../data/seeds0000-100000.pkl')
    parser.add_argument('--df_file', type=str, default='../data/color_palette00000-99999.csv')
    parser.add_argument('--model_file', type=str, default='../data/network-snapshot-005000.pkl')
    parser.add_argument('--colors_list', nargs='+', default=['Brown', 'Yellow', 'Green', 'Cyan', 'Blue', 'Magenta', 'Red', 'BW'])
    parser.add_argument('--color_bins', nargs='+', type=int, default=[0, 35, 70, 150, 200, 260, 345, 360])
    parser.add_argument('--subfolder', type=str, default='figures/textiles/interfaceGAN/')
    parser.add_argument('--variable', type=str, default='Color')
    parser.add_argument('--continuous_experiment', type=bool, default=False)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)

    args = parser.parse_args()
    
    kwargs = {'CL method':['LR', 'SVM'], 'C':[0.1, 0.01, 1], 
              'max_lambda':[15], 'extremes':[True, False]}
    
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    df = pd.read_csv(args.df_file)
    if 'Color' in df.columns:
        args.colors_list = df['Color'].unique()
        
    with dnnlib.util.open_url(args.model_file) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,100) for i in range(10)]
       
    disentanglemnet_exp = InterfaceGAN(model, annotations, df, space='w', 
                                       colors_list=args.colors_list, color_bins=args.color_bins,
                                       variable=args.variable)
    data = []
    ## save vector in npy and metadata in csv
    print('Now obtaining separation vector for using InterfaceGAN')
    for met in kwargs['CL method']:
        for c in kwargs['C']:
            for extr in kwargs['extremes']: 
                separation_vectors, performance = disentanglemnet_exp.InterFaceGAN_separation_vector(method=met, C=c, extremes=extr)
                if not args.continuous_experiment:
                    if extr:
                        continue
            
                print('Checking length of outputted vectors', len(separation_vectors), len(args.colors_list))
                print('Performance:', performance)
                for i, color in enumerate(args.colors_list[:len(separation_vectors)]):
                    print(f'Generating images with variations for color {color}')
                    name = 'InterfaceGAN_' + str(met) + '_' + str(c) + '_' + str(len(args.colors_list)) + '_' + str(args.variable)
                    data.append([color, args.variable, 'w', name, args.subfolder, ', '.join(args.colors_list), str(args.color_bins), str(separation_vectors[i])])
                    for seed in args.seeds:
                        for eps in kwargs['max_lambda']:
                            disentanglemnet_exp.generate_changes(seed, separation_vectors[i], min_epsilon=-eps,
                                                                 max_epsilon=eps, savefig=True, 
                                                                 feature=color, subfolder=args.subfolder, method=name)

   
    df = pd.DataFrame(data, columns=['Feature', 'Variable', 'Space', 'Method', 'Subfolder', 'Classes', 'Bins', 'Separation Vector'])
    df.to_csv(DATA_DIR + 'interfaceGAN_separation_vector_'+ args.variable +'.csv', index=False)
    np.save(DATA_DIR + 'interfaceGAN_separation_vector_'+ args.variable +'.npy', separation_vectors)

if __name__ == "__main__":
    main()