from disentanglement import DisentanglementBase

class StyleSpace(DisentanglementBase):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)
    
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
    
    def StyleSpace_separation_vector(self, sign=True, num_factors=20, cutout=0.25):
        """ Formula from StyleSpace Analysis """
        x_train, x_val, y_train, y_val = self.get_train_val()
        
        positive_idxs = []
        negative_idxs = []
        for color in self.colors_list:
            x_col = x_train[np.where(y_train == color)]
            mp = np.mean(x_train, axis=0)
            sp = np.std(x_train, axis=0)
            de = (x_col - mp) / sp
            meu = np.mean(de, axis=0)
            seu = np.std(de, axis=0)
            if sign:
                thetau = meu / seu
                positive_idx = np.argsort(thetau)[-num_factors//2:]
                negative_idx = np.argsort(thetau)[:num_factors//2]
                
            else:
                thetau = np.abs(meu) / seu
                positive_idx = np.argsort(thetau)[-num_factors:]
                negative_idx = []
                

            if cutout:
                beyond_cutout = np.where(np.abs(thetau) > cutout)
                positive_idx = np.intersect1d(positive_idx, beyond_cutout)
                negative_idx = np.intersect1d(negative_idx, beyond_cutout)
                
                if len(positive_idx) == 0 and len(negative_idx) == 0:
                    print('No values found above the current cutout', cutout, 'for color', color, '.\n Disentangled vector will be all zeros.' )
                
            positive_idxs.append(positive_idx)
            negative_idxs.append(negative_idx)
        
        separation_vectors = self.get_original_position_latent(positive_idxs, negative_idxs)
        return separation_vectors

