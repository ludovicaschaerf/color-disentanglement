
from disentanglement import DisentanglementBase

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
    