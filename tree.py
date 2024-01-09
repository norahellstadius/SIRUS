import numpy as np 
from quantiles import cutpoints

# ------ Tree -----------

class Leaf:
    def __init__(self, value: list) -> None:
        self.value = value
    """
    For classification, this is a vector of probabilities of each class.
    For regression, this is a vector of one element.    
    """

    def predict(self):
        return self.value


class SplitPoint: 
    def __init__(self, feature, value, feature_name) -> None:
        """
        A location where the tree splits.

        Attributes:
        - feature: Feature index.
        - split_value: Value of split.
        - feature_name: Name of the feature which is used for pretty printing.
        """

        self.feature = feature #feature index 
        self.split_value = value 
        self.feature_name = feature_name

class Node():
    def __init__(self, split_point: SplitPoint, left, right) -> None:
        self.splitpoint = split_point
        self.right = left #can either be a node or a leaf 
        self.left = right #can either be a node or a leaf 
    
    def children(self):
        return [self.right, self.left]
    
    def node_value(self):
        return self.splitpoint

class DecisionTree:
    def __init__(self, max_depth = 2, min_data_in_leaf = 5, random_state = 1) -> None:
        self.max_depth = max_depth
        self.seed = random_state
        self.min_data_in_leaf = min_data_in_leaf

    def rand_subset(self, points: list, n: int, seed:int) -> list:
        """
        Generate a random subset of size n from the given list of points without replacement.

        Parameters:
        - points (list): A list of elements to create a random subset from.
        - n (int): The size of the desired random subset.

        Returns:
        - list: A random subset of size n from the input list of points.
        """
        random.seed(self.seed)
        random_sample_n = random.sample(points, n)
        return random_sample_n

    @staticmethod
    def get_max_split_candidates(X):
        """
        Calculate the maximum split candidates for a given input.

        Parameters:
        - X: int
            The input value (representing the number of features).

        Returns:
        - int
            The rounded square root of the input value.
        """
        n, p = X.shape
        return round(np.sqrt(p))
    
    @staticmethod
    def seperate_y(data, y, comparison, cutpoint):
        """
        Returns a `y` for which the `comparison` holds in `data`.

        Parameters:
        - y_new (array): Mutable array to store `y` for which the `comparison` holds in `data`.
        - data (iterable): Array for comparison.
        - y (iterable): Original array.
        - comparison (function): A function to determine if a comparison holds.
        - cutpoint: A value against which the comparison is made.

        Returns:
        - y_new (array): `y_new` contains the valid elements.
        """
        
        y_new = []

        for i, value in enumerate(data):
            result = comparison(value, cutpoint)
            if result:
                y_new.append(y[i])

        return y_new
    
    @staticmethod
    def seperate_X_y(X, y, splitpoint:SplitPoint, comparison): 
        """
        Split the feature matrix and target vector based on a given split value and comparison.

        Parameters:
        - X: numpy.ndarray
            The feature matrix.
        - y: numpy.ndarray
            The target vector.
        - splitpoint: SplitPoint
            The SplitPoint object specifying the feature and split value.
        - comparison: function
            A comparison function that takes a value and the split value,
            returning a boolean result.

        Returns:
        - Tuple[numpy.ndarray, numpy.ndarray]
            A tuple containing the split feature matrix (X_split) and the split target vector (y_split).

        Raises:
        - AssertionError: If the length of the data and target vector does not match.
        """
        
        data = X[:, splitpoint.feature]
        assert len(data) == len(y), "Length mismatch between data and target vector."
        mask = []
        for i in range(len(y)):
            data_val = data[i]
            result = comparison(data_val, splitpoint.split_value)
            mask.append(result)

        mask_array = np.array(mask)
        X_split = X[mask_array, :]
        y_split = y[mask_array] #THIS WILL ONLY WORK IF Y IS AN ARRAY

        return X_split, y_split
    

    def split(self, X, y, classes, colnms, quantiles, max_split_candidates):
        """
        Find the best split point for a decision tree node based on maximising Gini index.

        Parameters:
        - X: numpy.ndarray
            The feature matrix.
        - y: numpy.ndarray
            The target vector.
        - classes: list
            List of unique classes in the target vector.
        - colnms: list
            List of column names corresponding to features in X.
        - quantiles: list of lists
            List of quantiles for each feature, defining potential cutpoints.
        - max_split_candidates: int, optional
            Maximum number of features to consider for splitting.
            If not provided, it defaults to the total number of features in X.

        Returns:
        - Union[None, SplitPoint]
            If a split point is found that improves the Gini index, returns a SplitPoint object.
            Otherwise, returns None.
        """

        if max_split_candidates is None:
            max_split_candidates = self.get_max_split_candidates(X)

        score_improved_bool = False
        best_score = self.start_score() #gini the larger the better
        best_score_feature = 0
        best_score_cutpoint = 0.0

        p = X.shape[1]
        #get feature indicies to used in the split
        possible_features = list(range(0, p)) if max_split_candidates == p else self.rand_subset(list(range(0, p)), max_split_candidates)
        reused_data = self.index_score(y, classes) #Data to be re-used in the loop on features and splitpoints

        for feature in possible_features:
            #get column of datapoints for that feature
            feature_data = X[:, feature] 

            #go through all q quantiles of feature (splitting options)
            for cutpoint in quantiles[feature]:
                
                #get the left and right y based on the cutpoint
                y_left = self.seperate_y(feature_data, y, np.less, cutpoint)
                if len(y_left) == 0:
                    continue

                y_right = self.seperate_y(feature_data, y, np.greater_equal, cutpoint)
                if len(y_right) == 0:
                    continue
                
                #get weighted gini score for the split 
                current_score = self.current_score(y, y_left, y_right, classes, reused_data)

                if self.score_improved(best_score, current_score):
                    score_improved_bool = True
                    best_score = current_score
                    best_score_feature = feature
                    best_score_cutpoint = cutpoint

        if score_improved_bool:
            feature_name = colnms[best_score_feature]
            return SplitPoint(best_score_feature, best_score_cutpoint, feature_name)
        else:
            return None
    

    def tree(self, X, y, classes, colnms=None, max_split_candidates=None, depth=0, max_depth=2, quantiles=None, min_data_in_leaf=5):
        """
        Build a decision tree recursively.

        Parameters:
        - X (array-like): Input features.
        - y (array-like): Target values.
        - classes (list): List of unique class labels.
        - colnms (list, optional): List of column names for the input features.
        - max_split_candidates (int, optional): Maximum number of split candidates to consider.
        - depth (int, optional): Current depth of the tree.
        - max_depth (int, optional): Maximum depth of the tree.
        - q (int, optional): Number of quantiles to use for cutpoints.
        - cps (array-like, optional): Precomputed cutpoints for splitting.
        - min_data_in_leaf (int, optional): Minimum number of data points in a leaf node.

        Returns:
        - Node: The root node of the decision tree.
        """

        if depth == max_depth:
            return self.create_leaf(y, classes) 

        sp = self.split(X, y, classes, colnms, quantiles, max_split_candidates=max_split_candidates)

        if sp is None or len(y) <= min_data_in_leaf:
            return self.create_leaf(y, classes)

        depth += 1

        left = self.tree(*self.seperate_X_y(X, y, sp, np.less), 
                        classes = classes, 
                        colnms = colnms, 
                        quantiles=quantiles, 
                        depth=depth)

        right = self.tree(*self.seperate_X_y(X, y, sp, np.greater_equal), 
                        classes = classes, 
                        colnms = colnms, 
                        quantiles=quantiles, 
                        depth=depth)

        node = Node(sp, left, right)
        return node

    def fit(self, X, y, colnms = None, max_split_candidates = None, q = 10, quantiles = None):

        assert len(X) == len(y), "Length mismatch between data and target vector."

        self.X = y 
        self.y = y 
        self.classes = self.get_classes(y)

        if quantiles is None:
            quantiles = cutpoints(X, q)

        if max_split_candidates is None:
            max_split_candidates = self.get_max_split_candidates(X)

        root_node = self.tree(X = self.X, 
                                y = self.y, 
                                classes = self.classes, 
                                columns = colnms, 
                                max_split_candidates = max_split_candidates, 
                                depth = 0, 
                                max_depth = self.max_depth, 
                                quantiles = quantiles,
                                min_data_in_leaf = self.min_data_in_leaf)
        return root_node



# -------- Classification ---------------
def count_equal(y: list, label_type: list) -> int:
    """
    Count occurrences of a specific class label in a list.

    Parameters:
    - y (List): List of class labels.
    - label_type (Any): Specific class label to count.

    Returns:
    - int: Count of occurrences of the specified class label in the list.
    """
    count = sum(1 for label in y if label == label_type)
    return count

def gini_index(y: list, classes: list) -> float:
    """
    Calculate Gini index for multiclass classification.

    Parameters:
    - y: List of class labels.
    - classes: List of unique classes.

    Returns:
    - Gini index.
    """
    total_samples = len(y)
    
    # Calculate the Gini index
    gini_impurity = 1.0
    for c in classes:
        p_c = count_equal(y, c) / total_samples
        gini_impurity -= p_c**2

    return gini_impurity

def weighted_gini(y: list, yl: list, yr: list, classes: list) -> float:
    """
    Calculate the weighted Gini index for a binary split.

    Parameters:
    - y: List of class labels.
    - yl: List of class labels for the left split.
    - yr: List of class labels for the right split.
    - classes: List of unique classes.

    Returns:
    - Weighted Gini index.
    """
    # Check if proportions add up to 1
    if abs(len(yl) / len(y) + len(yr) / len(y) - 1) > 1e-10:
        raise ValueError("Proportions of yl and yr must add up to 1.")

    p = len(yl) / len(y)
    weighted_gini = p * gini_index(yl, classes) + (1 - p) * gini_index(yr, classes)
    return weighted_gini

def information_gain(y: list, yl: list, yr: list, classes: list, starting_impurity: float) -> float:
    """
    Calculate information gain for a binary split.

    Parameters:
    - y: List of class labels.
    - yl: List of class labels for the left split.
    - yr: List of class labels for the right split.
    - classes: List of unique classes.
    - starting_impurity: Initial impurity measure (e.g., Gini index).

    Returns:
    - Information gain.
    """
    impurity_change = weighted_gini(y, yl, yr, classes)
    return starting_impurity - impurity_change



class DecisionTreeClassifier(DecisionTree):

    def __init__(self, max_depth = 2, min_data_in_leaf = 5, random_state = 1):
        super().__init__(max_depth, min_data_in_leaf, random_state)

    @staticmethod
    def score_improved(best_score: float, current_score: float) -> bool:
        """
        Check if the current score is an improvement over the best score.
        The bigger the better

        Parameters:
        - best_score: The best (lowest or highest, depending on the context) score achieved so far.
        - current_score: The current score to compare with the best score.

        Returns:
        - True if the current score is an improvement, False otherwise.
        """
        return best_score <= current_score

    "Data to be re-used in the loop on features and splitpoints. In julia library they call it _reused_data " 
    def reused_data(self, y: list, classes: list) ->float:
        return gini_index(y, classes) 
    
    def current_score(y: list, yl: list, yr: list, classes: list, starting_impurity: float) -> float:
        return information_gain(y, yl, yr, classes, starting_impurity)

    def create_leaf(y, classes):
        num_points = len(y)
        probabilities = [count_equal(y, c) / num_points for c in classes]
        return Leaf(probabilities) 

    @staticmethod
    def get_classes(y):
        unique_sorted_y = sorted(set(y))
        return unique_sorted_y
    
    @staticmethod
    def start_score():
        """Return the start score for the maximization problem.
           The larger the better for gini bounded by[0, 1]
        """
        return 0.0
    
    def predict_one_sample(self, x):
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node == Node:
            
            feature = node.splitpoint.feature
            value = node.splitpoint.split_value

            if x[feature] < value:
                node = node.left
            else:
                node = node.right

        #it exits when its a leaf 
        return node.value

    def predict_proba(self, X_set):
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set):
        """Returns the predicted probs for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds   


# ----------- Regression -----------
def rss(y):
    """
    Calculate the Residual Sum of Squares (RSS) for a given vector `y`.

    Parameters:
    - y (array-like): Input vector.

    Returns:
    float: Residual Sum of Squares.
    """
    m = np.mean(y)
    out = 0.0
    for x in y:
        out += (x - m) ** 2
    return out



class DecisionTreeRegression(DecisionTree):

    def __init__(self):
        super().__init__()

    @staticmethod
    def score_improved(best_score: float, current_score: float) -> bool:
        """
        Check if the current score is an improvement over the best score.
        The smaller the better 

        Parameters:
        - best_score: The best (lowest or highest, depending on the context) score achieved so far.
        - current_score: The current score to compare with the best score.

        Returns:
        - True if the current score is an improvement, False otherwise.
        """
        return best_score >= current_score

    "Data to be re-used in the loop on features and splitpoints. In julia library they call it _reused_data " 
    def reused_data(self, y: list = [], classes: list = []) ->float:
        return 0

    def current_score(y: list, yl: list, yr: list, classes: list = [], starting_impurity: float = 0) -> float:
        return rss(yl) + rss(yr)

    def create_leaf(y, classes = []): 
        return Leaf([np.mean(y)]) 

    @staticmethod
    def start_score():
        """Return the start score for the minimisation problem.
        """
        return np.inf
