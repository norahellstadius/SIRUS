import random 
import numpy as np
from quantiles import cutpoints
from tree import DecisionTreeClassifier, DecisionTreeRegression

PARTIAL_SAMPLING_DEFAULT = 0.7
N_TREES_DEFAULT = 1_000
MAX_DEPTH_DEFAULT = 2

# TODO: GENERALISE IF COLUMNS INPUT IS NONE

class StableForest: 
    def __init__(self, trees, classes) -> None:
        self.trees = [] #either a node or a leaf 

class RandomForest: 
    def __init__(self, type, max_depth = 2, min_data_in_leaf = 5, random_state = 1) -> None:

        if type not in ["Classification", "Regression"]:
            raise ValueError("Invalid value for self.type. Expected 'Classification' or 'Regression', but got '{}'.".format(self.type))

        self.type = type
        self.max_depth = max_depth
        self.seed = random_state
        self.min_data_in_leaf = min_data_in_leaf


    def forest(self, X, y, colnms, max_split_candidates = None, partial_sampling=0.5, n_trees=100, max_depth=2, q=10, quantiles = None, min_data_in_leaf=5):   
            
            if max_depth > 2:
                raise ValueError("""
                    Tree depth is too high. Rule filtering for a depth above 2 is not implemented.
                    In the original paper, the authors also advise using a depth of no more than 2.
                    """)
            if max_depth < 1:
                raise ValueError(f"Minimum tree depth is 1; got {max_depth}")

            n_samples = int(partial_sampling * len(y))

            trees = [None] * n_trees
            seeds = list(range(n_trees))

            # TODO: implement with threads to make this parrallel
            for i in range(n_trees):
                seed = seeds[i]

                # TODO: check if this should be with or without replacement
                #sample with replacement 
                random.seed(seed)
                row_idxs = np.random.choice(range(len(y)), size=n_samples, replace=True)
                X_samp = X[row_idxs, :]
                y_samp = y[row_idxs] #Y NEEDS TO BE ARRAY FOR THIS TO WORK 

                if self.type == "Classification":
                    tree_model = DecisionTreeClassifier(max_depth, min_data_in_leaf, self.seed)
                elif self.type == "Regression": 
                    tree_model = DecisionTreeRegression(max_depth, min_data_in_leaf, self.seed)
                else:
                    raise ValueError("Invalid value for self.type. Expected 'Classification' or 'Regression', but got '{}'.".format(self.type))

                tree = tree_model.fit(
                    X = X_samp,
                    y = y_samp,
                    colnms = colnms,  
                    max_split_candidates = max_split_candidates,
                    q = q,
                    quantiles = quantiles,
                )

                trees[i] = tree 

            return StableForest(trees)
    
    def fit(self, X, y, colnms = None,  n_trees=10, max_split_candidates = None, partial_sampling = 0.5, q = 10, quantiles = None):
        self.X = X 
        self.y = y 
        self.n_trees = n_trees 
        
        if quantiles is None:
            quantiles = cutpoints(X, q)

        stable_forest = self.forest(X = self.X, 
                    y = self.y, 
                    colnms = colnms, 
                    max_split_candidates = max_split_candidates, 
                    partial_sampling = partial_sampling, 
                    n_trees= self.n_trees,
                    max_depth= self.max_depth, 
                    q=10, 
                    quantiles = quantiles, 
                    min_data_in_leaf= self.min_data_in_leaf)

        return stable_forest



   



    


