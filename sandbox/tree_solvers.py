import numpy as np
from collections import Counter
from sklearn.metrics import  mean_absolute_error,mean_squared_error


class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.left = None
        self.right = None

class DecisionTree():
    def __init__(self, max_depth: int = 1, min_samples_leaf: int = 1, min_information_gain: float = 0.0) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.paths = []
        self.rules = {}
    
    #the lower the better
    def _entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def _class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _partition_entropy(self, subsets: list) -> float:
        total_count = sum([len(subset) for subset in subsets])
        return sum([self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def _data_entropy(self, labels: list) -> float:
        return self._entropy(self._class_probabilities(labels))

    def _find_label_probs(self, data: np.array) -> np.array:

        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities
    
    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        
        mask_below_threshold = data[:, feature_idx] < feature_val
        left_group = data[mask_below_threshold]
        right_group = data[~mask_below_threshold]
        return left_group, right_group

    def _find_best_split(self, data: np.array, selected_features, valid_splits) -> tuple: 
        min_part_entropy = 1e6
        min_entropy_feature_idx = None
        min_entropy_feature_val = None
        groupLeft_min, groupRight_min = None, None

        for feature_idx in selected_features:
            for q_value in valid_splits[:, feature_idx]:
                left_group, right_group = self._split(data, feature_idx, q_value)
                part_entropy = self._partition_entropy([left_group[:, -1], right_group[:, -1]])
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = feature_idx
                    min_entropy_feature_val = q_value
                    groupLeft_min, groupRight_min = left_group, right_group
        
        return groupLeft_min, groupRight_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _create_tree(self, data: np.array, current_depth: int, num_features:int, valid_splits:np.array, path=None) -> TreeNode:

        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None

        #minus 1 since last column in the response varibale 
        selected_features = np.random.choice(data.shape[1] - 1, size=num_features, replace=False)

        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data, selected_features, valid_splits)
        
        label_probabilities = self._find_label_probs(data)

        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        
        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1

        #TODO: make it more general for multiclass and regression task
        node_pred_left = int(np.mean(split_1_data[:,-1]) >= 0.5)
        node_pred_right = int(np.mean(split_2_data[:,-1]) >= 0.5)


        left_path = path + [f"Feature {split_feature_idx} < {split_feature_val}"]
        right_path = path + [f"Feature {split_feature_idx} >= {split_feature_val}"]

        self.paths.append(left_path)
        self.paths.append(right_path)

        self.rules[' & '.join(left_path)] = node_pred_left
        self.rules[' & '.join(right_path)] = node_pred_right

        node.left = self._create_tree(split_1_data, current_depth, num_features, valid_splits, left_path)
        node.right = self._create_tree(split_2_data, current_depth, num_features, valid_splits, right_path)
       

        return node

    def fit(self, X_train: np.array, Y_train: np.array, num_features:int, valid_splits:np.array) -> None:
        
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        self.tree = self._create_tree(train_data, 0, num_features, valid_splits, [])

    def predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds   

    

class RandomForest():
    def __init__(self, n_trees:int, num_samples:int, max_tree_depth:int, max_features:int, valid_splits:np.array):
        self.n_trees = n_trees 
        self.a_n = num_samples
        self.max_depth = max_tree_depth
        self.max_features = max_features
        self.splits = valid_splits
        self.trees = {}
    
    def __make_boostraps(self, X_train, y_train):

        n, _ = X_train.shape
        train_data = np.concatenate((X_train, np.reshape(y_train, (-1, 1))), axis=1)

        bootstraps = {}

        all_idxs = list(range(n))

        for idx in range(self.n_trees):
            sampled_idxs = np.random.choice(n, size=self.a_n, replace=True)
            in_sample_data = train_data[sampled_idxs, :]

            out_of_sample_idxs   = list(set(all_idxs) - set(sampled_idxs))

            if out_of_sample_idxs:
                out_of_sample = train_data[out_of_sample_idxs,:]

            bootstraps[idx] = {"in_sample" : in_sample_data, "out_sample": out_of_sample}
        
        return bootstraps
    
    def _train(self, X_train, y_train):
        
        bootstraps_dict = self.__make_boostraps(X_train, y_train)
        out_of_sample_data = {}

        for key, data_sample in bootstraps_dict.items():

            X_train_boot, y_train_boot = data_sample["in_sample"][:,:-1], data_sample["in_sample"][:,-1]

            tree_model = DecisionTree(self.max_depth, 1, 0.0)
            tree_model.fit(X_train_boot, y_train_boot, self.max_features, self.splits)
            self.trees[key] = tree_model

            if data_sample["out_sample"].size:
                out_of_sample_data[key] = data_sample["out_sample"]
            else:
                out_of_sample_data[key] = np.array([])
        
        return out_of_sample_data


    def fit(self, X_train, y_train, print_metrics=False):
        out_of_sample_data = self._train(X_train, y_train)

        if print_metrics:
            #initialise metric arrays
            maes = np.array([])
            mses = np.array([])

            #loop through each bootstrap sample
            for data_idx, single_tree in zip(out_of_sample_data, self.trees):
                #compute the predictions on the out-of-bag test set & compute metrics
                if out_of_sample_data[data_idx].size:
                    y_pred  = single_tree.predict(out_of_sample_data[data_idx][:,:-1])
                    mae = mean_absolute_error(out_of_sample_data[data_idx][:,-1],y_pred)
                    mse = mean_squared_error(out_of_sample_data[data_idx][:,-1],y_pred)   
                    #store the error metrics
                    maes = np.concatenate((maes,mae.flatten()))
                    mses = np.concatenate((mses,mse.flatten()))

            #print standard errors
            print("Standard error in mean absolute error: %.2f" % np.std(maes))
            print("Standard error in mean squred error: %.2f" % np.std(mses))
            

    def predict(self, X_test):

        if not self.trees:
            print('You must train the ensemble before making predictions!')
            return(None)

        predictions = []
        for single_tree in self.trees.values():
            y_pred = single_tree.predict(X_test)
            predictions.append(y_pred.reshape(-1,1))

        ypred = np.mean(np.concatenate(predictions,axis=1),axis=1)

        return np.round(ypred).astype(int)





if __name__ == "__main__":

    from preprocess.get_data import get_BW_data
    from sklearn.model_selection import train_test_split

    def compute_empirical_quantiles(data, quantiles = np.arange(0, 100, 10)):
        result = np.percentile(data, quantiles, axis=0)
        return result

    X, y = get_BW_data("/Users/norahallqvist/Code/SIRUS/data/BreastWisconsin.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    splits = compute_empirical_quantiles(X_train)

    rf_model = RandomForest(100, int(X.shape[0] * 0.95), 1, 6, splits)
    rf_model.fit(X_train, y_train)  

    # print(rf_model.trees.items())
    # for tree in rf_model.trees.values():
    #     for path in tree.paths:
    #             print(path)

    y_pred = rf_model.predict(X_test)
    print("test accuracy: ", np.mean(y_pred == y_test))

    




