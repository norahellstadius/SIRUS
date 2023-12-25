import numpy as np 
from tree_solvers import RandomForest, DecisionTree, TreeNode

class SIRUS_Solver:
    def __init__(self, X_train, X_test, y_train, y_test, num_subsampled_points, frequency_threshold, max_tree_depth, q, number_trees):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.q = q
        self.p0 = frequency_threshold 
        self.a_n = num_subsampled_points #number of samples bootraped for each tree in RF
        self.max_features = int(np.floor(X_train.shape[0] / X_train.shape[1]))  # number of features considered in each node of RF
        self.num_trees = number_trees
        self.max_tree_depth = max_tree_depth 
        self.quantiles = self.empirical_quantiles() #valids splits for each node 
        self.trees = {}
        self.paths = {}
        self.rules = {}
        self.rf_model = None

    def empirical_quantiles(self):
        return np.percentile(self.X_train, np.arange(0, 100, self.q), axis = 0) 

    def store_paths(self):
        for key, tree in self.trees.items():
            self.paths[key] = tree.paths

    def store_rules(self):
        for key, tree in self.trees.items():
            self.rules[key] = tree.rules

    def fit_trees(self):
        self.rf_model = RandomForest(self.num_trees, self.a_n, self.max_tree_depth, self.max_features, self.quantiles)
        self.rf_model.fit(self.X_train, self.y_train)
        self.trees = self.rf_model.trees
        self.store_paths()
        self.store_rules()

    def get_accuracy(self):
        y_pred = self.rf_model.predict(self.X_test)
        test_acc = np.mean(y_pred == self.y_test)
        print(f"test accuracy: {test_acc}")


if __name__ == "__main__":

    from preprocess.get_data import get_BW_data
    from sklearn.model_selection import train_test_split

    X, y = get_BW_data("/Users/norahallqvist/Code/SIRUS/data/BreastWisconsin.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Hyperparameters
    num_subsampled_points = int((X_train.shape[0]) * 0.95)
    frequency_threshold = 0.5
    tree_depth = 1
    q = 10
    num_trees = 2

    solver = SIRUS_Solver(X_train, X_test, y_train, y_test, num_subsampled_points, frequency_threshold, tree_depth, q, num_trees)
    solver.fit_trees()
    solver.get_accuracy()

    #print paths
    for i, tree_paths in enumerate(solver.paths.values()): 
        print(f"tree num: {i + 1}")
        for path in tree_paths:
            print(path)
    
    for i, tree_rules in enumerate(solver.rules.values()): 
        print(f"tree num: {i + 1}")
        for path, rule in tree_rules.items():
            print(path, "then Y = ", rule)

    print(solver.rules.values())

    


    