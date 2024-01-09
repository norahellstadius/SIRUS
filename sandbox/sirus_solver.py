import re
import numpy as np 
from sandbox.tree_solvers import RandomForest
from sandbox.rule_filter import Condition, Path, Rule, filter_linearly_dependent_rules

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
        self.rf_model = None

        self.raw_all_rules = {}
        self.all_rules = []
        self.rules_high_freq = {}
        self.independent_rules = []
        
    def empirical_quantiles(self):
        return np.percentile(self.X_train, np.arange(0, 100, self.q), axis = 0) 

    def _store_raw_rules(self):
        for key, tree in self.rf_model.trees.items():
            self.raw_all_rules[key] = tree.rules

    """
    Simplify the rules that contain a single split by only retaining rules that point left and
    removing duplicates.
    """
    def _store_formated_rules(self):
        rules_flatten  = {}
        for _, tree_rules in self.raw_all_rules.items():
            for path, rule in tree_rules.items():
                rules_flatten[path] = rule

        if len(rules_flatten) % 2 != 0:
            raise AssertionError("The length of rules_flatten is not even.")

        iter_rules_items = iter(rules_flatten.items())
        all_rules = []

        for (path_right, node_pred_right), (path_left, node_pred_left) in zip(iter_rules_items, iter_rules_items):

            #double check that the same node is being compared
            numeric_values_right = [int(float(value)) for value in re.findall(r'\d+\.\d+', path_right)]
            numeric_sum_right = sum(numeric_values_right)

            numeric_values_left = [int(float(value)) for value in re.findall(r'\d+\.\d+', path_left)]
            numeric_sum_left = sum(numeric_values_left)

            if numeric_sum_right != numeric_sum_left:
                raise ValueError("Error: Feature mismatch (i.e the same node is not being processed)")

            conditions = path_right.split("&")
            path_conditions = []
            for condition in conditions:
                feature, operation, split = (condition.strip()).split(" ")[1:]
                path_conditions.append(Condition(int(feature), operation, float(split)))
            path_to_node = Path(path_conditions)
            node_rule = Rule(path_to_node, [float(node_pred_right), float(node_pred_left)])
            all_rules.append(node_rule)
        
        self.all_rules = all_rules

    def fit_trees(self):
        self.rf_model = RandomForest(self.num_trees, self.a_n, self.max_tree_depth, self.max_features, self.quantiles)
        self.rf_model.fit(self.X_train, self.y_train)
        self._store_raw_rules()
        self._store_formated_rules()

    def get_independent_rules(self):
        self.independent_rules = filter_linearly_dependent_rules(self.all_rules)

    def print_rules(self):
        for rule in self.independent_rules:
            rule.print_rule()

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
    num_trees = 100

    solver = SIRUS_Solver(X_train, X_test, y_train, y_test, num_subsampled_points, frequency_threshold, tree_depth, q, num_trees)
    solver.fit_trees()
    solver.get_accuracy()


    # for i, tree_rules in enumerate(solver.raw_all_rules.values()): 
    #     print(f"tree num: {i + 1}")
    #     for path, rule in tree_rules.items():
    #         print(path, "then Y = ", rule)

    print("-------- formated rules -------")
    
    # for rule in solver.all_rules: 
    #     rule.print_rule()

    print("-------- indepndent rules -------")

    solver.get_independent_rules()
    solver.print_rules()
    
    # import json 
    # with open('/Users/norahallqvist/Code/SIRUS/results/example_rules_dict.json', 'w') as json_file:
    #     json.dump(solver.raw_all_rules, json_file)


    
