from disentanglement import DisentanglementBase

class GANSpace(DisentanglementBase):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)

    def GANSpace_separation_vectors(self, num_components):
        """Unsupervised method using PCA to find most important directions"""
        x_train, x_val, y_train, y_val = self.get_train_val()
        if self.space.lower() == 'w':
            pca = PCA(n_components=num_components)

            dims_pca = pca.fit_transform(x_train.T)
            dims_pca /= np.linalg.norm(dims_pca, axis=0)
            
            return dims_pca
        
        else:
            raise("""This space is not allowed in this function, 
                     only W""")
    
def main():
    parser = argparse.ArgumentParser(description='Process input arguments')
    
    parser.add_argument('--annotations_file', type=str, default='data/textile_annotated_files/seeds0000-100000.pkl')
    parser.add_argument('--df_file', type=str, default='data/textile_annotated_files/final_sim_seeds0000-100000.csv')
    parser.add_argument('--model_file', type=str, default='data/textile_model_files/network-snapshot-005000.pkl')
    parser.add_argument('--colors_list', nargs='+', default=['Brown', 'Yellow', 'Green', 'Cyan', 'Blue', 'Magenta', 'Red', 'BW'])
    parser.add_argument('--color_bins', nargs='+', type=int, default=[0, 45, 75, 140, 195, 260, 340, 360])
    parser.add_argument('--subfolder', type=str, default='textiles')
    parser.add_argument('--variable', type=str, default='H1')
    parser.add_argument('--continuous_experiment', type=bool, default=False)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)

    args = parser.parse_args()
    
    kwargs = {'CL method':['LR',],  'S method': ['LR', 'XGB'], 'C':[0.1], 'sign':[True], #'SVM',
              'num_factors':[10, 20], 'cutout': [None], 'max_lambda':[15], 
              'extremes':[True, False], 'weighted':[True]}
    
    with open(args.annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    df = pd.read_csv(args.df_file).fillna('#000000')
    df.columns = [df.columns[0], 'top1col', 'top2col', 'top3col']
    
    with dnnlib.util.open_url(args.model_file) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [random.randint(0,100) for i in range(10)]
       
    disentanglemnet_exp = GANSpace(model, annotations, df, space=space, colors_list=args.colors_list, color_bins=args.color_bins, variable=args.variable)
            
    ## save vector in npy and metadata in csv
    print('Now obtaining separation vector for using GANSpace')
    separation_vectors = disentanglemnet_exp.GANSpace_separation_vectors(10)
    print('Checking shape of vectors', separation_vectors.shape)
    print('Generating images with variations')
    for i in range(10):
        name = 'GANSpace_dimension_' + str(i)
        data.append(['Dim'+str(i), args.variable, 'w', name, args.subfolder, '', '', str(separation_vectors.T[i])])
        for seed in args.seeds:
            for eps in kwargs['max_lambda']:
                disentanglemnet_exp.generate_changes(seed, separation_vectors.T[i], min_epsilon=-eps, max_epsilon=eps, 
                                                    savefig=True, feature='dim' + str(i), subfolder=args.subfolder, method=name)
   
    df = pd.DataFrame(data, columns=['Feature', 'Variable', 'Space', 'Method', 'Subfolder', 'Classes', 'Bins', 'Separation Vector'])
    df.to_csv('data/separation_vector_'+ args.subfolder +'.csv', index=False)

if __name__ == "__main__":
    main()