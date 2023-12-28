from disentanglement import DisentanglementBase

class ShapleyVec(DisentanglementBase):
    def __init__(self, model, annotations, df, space, colors_list, color_bins, compute_s=False, variable='H1', categorical=True, repo_folder='.'):
        super().__init__(model, annotations, df, space, colors_list, color_bins, compute_s, variable, categorical, repo_folder)

    def Shapley_separation_vector(self, num_factors=10, method='XGB', weighted=False):
        """Test method that uses Shapley value to find most important channels"""
        x_train, x_val, y_train, y_val = self.get_train_val()
        le = LabelEncoder()
        le.fit(y_train)
        
        print(x_train.shape)
        ## use InterfaceGAN 
        
        if method == 'XGB':
            # Use "hist" for constructing the trees, with early stopping enabled.
            model = xgb.XGBClassifier(device=self.device)
            # Fit the model, test sets are used for early stopping.
            model.fit(x_train, le.transform(y_train), eval_set=[(x_val, le.transform(y_val))])
            performance = np.round(model.score(x_val, le.transform(y_val)), 2)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_train, y=le.transform(y_train))
        
        
            ## select highest shapley values
        elif method == 'LR':
            model = LogisticRegression(random_state=0, C=0.1)
            model.fit(x_train, le.transform(y_train))
            performance = np.round(model.score(x_val, le.transform(y_val)), 2)
            explainer = shap.LinearExplainer(model, x_train, feature_dependence="independent")
            shap_values = explainer.shap_values(x_train,)
        
        else:
            raise Exception("""This method is not allowed for this technique. Select XGB or LR""")
        #if self.device == 'cuda':
        #    explainer = shap.GPUTreeExplainer(model)
        #else:
            
        positive_idxs = []
        negative_idxs = []
        
        if weighted:
            positive_vals = []
            negative_vals = []
        for color in tqdm(self.colors_list):
            label = le.transform(np.array([color]))[0]
            print(color, label, le.get_params())
            shap_vals = shap_values[label].mean(axis=0)
            shap_idxs = np.argsort(shap_vals)
            shap_vals = np.sort(shap_vals)
            
            
            ## get vector
            positive_idx = list(shap_idxs[-num_factors//2:])
            negative_idx = list(shap_idxs[:num_factors//2])
            
                
            positive_idxs.append(positive_idx)
            negative_idxs.append(negative_idx)
            
            if weighted:
                if method == 'LR':
                    coeff_vals = model.coef_[label] / np.linalg.norm(model.coef_[label])
                    coeff_vals_pos = coeff_vals[positive_idx]
                    coeff_vals_neg = coeff_vals[negative_idx]
                    positive_vals.append(list(coeff_vals_pos))
                    negative_vals.append(list(coeff_vals_neg))
            
                else:
                    positive_vals.append(list(shap_vals[-num_factors//2:]))
                    negative_vals.append(list(shap_vals[:num_factors//2]))
            
        #shap.summary_plot(shap_values, x_val)
        #plt.savefig('shap.png')
        if weighted:
            separation_vectors = self.get_original_position_latent(positive_idxs, negative_idxs, positive_vals, negative_vals)
        else:
            separation_vectors = self.get_original_position_latent(positive_idxs, negative_idxs)
        return separation_vectors, performance

    def Ngram_separation_vector(self):
        """Method to refine the chosen vectors that takes into account which ones interact successfully"""
        x_train, x_val, y_train, y_val = self.get_train_val()
        
        ## use shapley values or StyleSpace found channels
        
        ## find interaction channels using interaction terms in classifier
        
        ## get vector using the interacting terms
        separation_vectors = []
        #for color in self.colors_list:
            #...    
            
        return separation_vectors

