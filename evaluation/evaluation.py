import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import sys
from scipy import signal
import cv2
 
sys.path.append('../annotations')
from color_annotations import extract_color

sys.path.append('../stylegan')
import dnnlib 
import legacy
import random

sys.path.append('../disentanglement')
from disentanglement import DisentanglementBase
        
class DisentanglementEvaluation(DisentanglementBase):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)
    
    
    def obtain_changes(self, separation_vector, seeds, lambda_range=15, method='None', feature='None', subfolder='test'):
        variation_scores = [[]]*len(seeds)*(lambda_range+1)
        idx = 0
        for i,seed in enumerate(tqdm(seeds)):
            images, lambdas = self.generate_changes(seed, separation_vector, min_epsilon=0, max_epsilon=lambda_range, 
                                                    count=lambda_range+1, savefig=True, subfolder=subfolder, method=method, feature=feature) 
            for j, (img, lmb) in enumerate(zip(images, lambdas)):
                idx += 1
                if lmb == 0:
                    try:
                        img_orig = img
                        colors_orig = extract_color(img_orig, 5, 1, None)
                        hsv_orig = list(rgb2hsv(*hex2rgb(colors_orig[0])))
                        top_three_colors = colors_orig[:3]
                        color_orig = range2color(*hsv_orig, self.colors_list, self.color_bins)
                        img_orig_norm = cv2.normalize(np.array(img_orig.convert('L')), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        img_norm = cv2.normalize(np.array(img.convert('L')), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        cor = cv2.filter2D(img_orig_norm, ddepth=-1, kernel=img_norm)
                        cor = np.max(cor)
                        print('seed ', seed, 'orig HSV', hsv_orig, ', color:', color_orig, ', moving in dir:', feature, ', self corr:', cor)
                        variation_scores[idx] = [seed, lmb, hsv_orig, color_orig, cor, top_three_colors]
                    except Exception as e:
                        print('No colors could be found for the original image', e)
                else:
                    try:
                        colors = extract_color(img, 5, 1, None)
                        hsv = list(rgb2hsv(*hex2rgb(colors[0])))
                        top_three_colors = colors[:3]
                        color = range2color(*hsv, self.colors_list, self.color_bins)
                        #cor = signal.correlate2d(np.array(img_orig.convert('L')), np.array(img.convert('L')), mode='full', boundary='fill')
                        # convert to float32
                        img_orig_norm = cv2.normalize(np.array(img_orig.convert('L')), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        img_norm = cv2.normalize(np.array(img.convert('L')), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        cor = cv2.filter2D(img_orig_norm, ddepth=-1, kernel=img_norm)
                        cor = np.max(cor)
                        print('seed ', seed, 'img HSV', hsv, ', orig color:', color_orig, ', color now:', color, ', moving in dir:', feature, ', corr to orig img:', cor)
                        variation_scores[idx] = [seed, lmb, hsv, color, cor, top_three_colors]
                    except Exception as e:
                        print('Failed other image extraction', e)
        return variation_scores
            
    def structural_coherence(self, im1, im2):
        # Implement your custom metric 3 evaluation logic here
        # You can use specific techniques for this metric and call generate_changes if needed.
        ## struct coherence
        img_orig_norm = cv2.normalize(np.array(im1.convert('L')), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_norm = cv2.normalize(np.array(im2.convert('L')), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cor = cv2.filter2D(img_orig_norm, ddepth=-1, kernel=img_norm)
        cor = np.max(cor)
                        
        return cor

            
if __name__ == '__main__':
    print('Disentanglement evaluation starting')
    
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='data/textile_annotated_files/seeds0000-100000.pkl')
    parser.add_argument('--df_file', type=str, default='data/textile_annotated_files/final_sim_seeds0000-100000.csv')
    parser.add_argument('--model_file', type=str, default='data/textile_model_files/network-snapshot-005000.pkl')
    parser.add_argument('--df_separation_vectors', type=str, default='data/separation_vector_textile.csv') #
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--obtain_changes', type=bool, default=False)
    parser.add_argument('--obtain_dci', type=bool, default=False)
    parser.add_argument('--lambdas', type=int, default=15)

    args = parser.parse_args()
    
    print(args.obtain_changes, 'obtain changes?')
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    df = pd.read_csv(args.df_file).fillna('#000000')
    df.columns = [df.columns[0], 'top1col', 'top2col', 'top3col']
    
    with dnnlib.util.open_url(args.model_file) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,100000) for i in range(200)]
       
    df_separation_vectors = pd.read_csv(args.df_separation_vectors)
    
    categorical_accuracies = []
    categorical_accuracies_per_lambda = []
    all_variations = pd.DataFrame([], columns=['seed', 'lambda', 'hsv', 'color', '2dcorr', 'color_vector', 'method',
                                                'subfolder', 'colors_list', 'color_bins', 'variable', 'space'])
    for i,row in df_separation_vectors.iterrows():
        if i == 0:
            colors_list = row['Classes'].split(', ')
            color_bins = [int(x.strip('[]')) for x in row['Bins'].split(', ')]
            variable = row['Variable']
            space = row['Space']
            disentanglemnet_eval = DisentanglementEvaluation(model, annotations, df, space=space, colors_list=colors_list, color_bins=color_bins, variable=variable) 
            if args.obtain_dci:
                disentanglement, completeness, informativeness = disentanglemnet_eval.DCI()
                print(disentanglement, completeness, informativeness)
        
        color = row['Feature']
        method = row['Method']
        subfolder = row['Subfolder']
        
        if 'GANSpace' in method:
            print('Skipping GANSpace')
            continue
        if 'XGB' in method:
            print('Skipping XGB')
            continue
        
        separation_vector = np.array([float(x.strip('[] ')) for x in row['Separation Vector'].replace('\n', ' ').split(' ') if x.strip('[] ') != ''])
        
        color_range = color2range(color, colors_list, color_bins)
        
        if args.obtain_changes:
            variation_scores = disentanglemnet_eval.obtain_changes(separation_vector, args.seeds, args.lambdas, method, color)
            var_scores_df = pd.DataFrame(variation_scores, columns=['seed', 'lambda', 'hsv', 'color', '2dcorr', 'top_three_colors'])
            var_scores_df['color_vector'] = color
            var_scores_df['method'] = method
            var_scores_df['subfolder'] = subfolder
            var_scores_df['colors_list'] = str(colors_list)
            var_scores_df['color_bins'] = str(color_bins)
            var_scores_df['variable'] = variable
            var_scores_df['space'] = space
            all_variations = pd.concat([all_variations, var_scores_df], axis=0)
            print(all_variations.head())
            all_variations.to_csv('data/variations_new_'+ color + '_' + method + '_' + subfolder + '.csv', index=False)
        else:
            try:
                all_variations = pd.read_csv('data/variations_new_'+ color + '_' + method + '_' + subfolder + '.csv')
            except:
                break
            
        try:
            categorical_accuracy, categorical_accuracy_per_lambda = disentanglemnet_eval.re_scoring_categorical(all_variations, color, method, subfolder)
            print(categorical_accuracy, len(categorical_accuracy_per_lambda))
            categorical_accuracies.append(categorical_accuracy)
            categorical_accuracies_per_lambda.append(categorical_accuracy_per_lambda)
        except Exception as e:
            print(e)
    
    print(len(categorical_accuracies), len(categorical_accuracies_per_lambda))
    df_separation_vectors.loc[:len(categorical_accuracies), 'categorical_accuracy'] = categorical_accuracies
    df_separation_vectors.loc[:len(categorical_accuracies), [f'categorical_accuracy_lambda_{i}' for i in range(1,16)]] = categorical_accuracies_per_lambda
    if args.obtain_dci:
        df_separation_vectors['disentanglement'] = disentanglement
        df_separation_vectors['completeness'] = completeness
        df_separation_vectors['informativeness'] = informativeness
        
    df_separation_vectors.to_csv('data/scores_new_'+subfolder+'.csv', index=False)
    

    
        
        