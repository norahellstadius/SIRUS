import random
import numpy as np
from typing import Callable, Tuple, Union
from quantiles import cutpoints
from sklearn.metrics import mean_absolute_error

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

        self.feature = feature  # feature index
        self.split_value = value
        self.feature_name = feature_name


class Node:
    def __init__(self, split_point: SplitPoint, left, right) -> None:
        self.splitpoint = split_point
        self.right = left  # can either be a node or a leaf
        self.left = right  # can either be a node or a leaf

    def children(self):
        return [self.right, self.left]

    def node_value(self):
        return self.splitpoint


class DecisionTree:
    def __init__(self, max_depth:int=2, min_data_in_leaf:int=5, random_state:int=1) -> None:
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.seed = random_state
        self.root_node = None #to save the fitted tree

    def rand_subset(self, points: list, n: int) -> list:
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
        _, p = X.shape
        return round(np.sqrt(p))

    @staticmethod
    def seperate_y(data: np.ndarray, y:np.ndarray, comparison: Callable[[float, float], bool], cutpoint:float) -> np.ndarray:
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

        return np.array(y_new)

    @staticmethod
    def seperate_X_y(X: np.ndarray, y: np.ndarray, splitpoint: SplitPoint, comparison: Callable[[float, float], bool]) -> Tuple[np.ndarray, np.ndarray]:
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
        y_split = y[mask_array]  # THIS WILL ONLY WORK IF Y IS AN ARRAY

        return X_split, y_split

    def split(self, X:np.ndarray, y:np.ndarray, classes:np.ndarray, colnms:np.ndarray, quantiles:np.ndarray, max_split_candidates:int) -> Union[SplitPoint, None]:
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

        score_improved_bool = False
        best_score = self.start_score()  # gini the larger the better
        best_score_feature = 0
        best_score_cutpoint = 0.0

        _, p = X.shape
        # get feature indicies to used in the split
        possible_features = (
            list(range(0, p))
            if max_split_candidates == p
            else self.rand_subset(list(range(0, p)), max_split_candidates)
        )
        reused_data = self.reused_data(
            y, classes
        )  # Data to be re-used in the loop on features and splitpoints

        for feature in possible_features:
            # get column of datapoints for that feature
            feature_data = X[:, feature]

            # go through all q quantiles of feature (splitting options)
            for cutpoint in quantiles[feature]:
                # get the left and right y based on the cutpoint
             
                y_left = self.seperate_y(feature_data, y, np.less, cutpoint)
                if len(y_left) == 0:
                    continue
            
                y_right = self.seperate_y(feature_data, y, np.greater_equal, cutpoint)
                if len(y_right) == 0:
                    continue

                assert len(y) == (len(y_left) + len(y_right)), "Splittig of y into y_left and y_right incorrect shape."

                # get weighted gini score for the split
                current_score = self.current_score(
                    y, y_left, y_right, classes, reused_data
                )

                if self.score_improved(best_score, current_score):
                    score_improved_bool = True
                    best_score = current_score
                    best_score_feature = feature
                    best_score_cutpoint = cutpoint

        if score_improved_bool:
            feature_name = colnms[best_score_feature] if colnms is not None else None
            return SplitPoint(best_score_feature, best_score_cutpoint, feature_name)
        else:
            return None

    def tree(
        self,
        X:np.ndarray,
        y:np.ndarray,
        classes:np.ndarray,
        max_split_candidates:int,
        colnms=None,
        depth:int=0,
        max_depth:int=2,
        quantiles=None,
        min_data_in_leaf:int=5,
    ) -> Node:
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

        sp = self.split(
            X = X, y = y, classes = classes, colnms = colnms, quantiles = quantiles, max_split_candidates=max_split_candidates
        )

        if sp is None or len(y) <= min_data_in_leaf:
            return self.create_leaf(y, classes)

        depth += 1

        left = self.tree(
            *self.seperate_X_y(X, y, sp, np.less),
            classes=classes,
            max_split_candidates = max_split_candidates,
            colnms=colnms,
            max_depth = max_depth,
            quantiles=quantiles,
            depth=depth
        )

        right = self.tree(
            *self.seperate_X_y(X, y, sp, np.greater_equal),
            classes=classes,
            max_split_candidates = max_split_candidates,
            colnms=colnms,
            max_depth =max_depth,
            quantiles=quantiles,
            depth=depth
        )

        node = Node(sp, left, right)
        return node

    def fit(self, X:np.ndarray, y:np.ndarray, colnms=None, max_split_candidates=None, q:int=10, quantiles=None) -> Node:
        assert len(X) == len(y), "Length mismatch between data and target vector."

        self.X = X
        self.y = y
        self.classes = self.get_classes(y)

        if quantiles is None:
            quantiles = cutpoints(X, q)

        if max_split_candidates is None:
            max_split_candidates = self.get_max_split_candidates(X)

        self.root_node = self.tree(
            X=self.X,
            y=self.y,
            classes=self.classes,
            colnms=colnms,
            max_split_candidates=max_split_candidates,
            depth=0,
            max_depth=self.max_depth,
            quantiles=quantiles,
            min_data_in_leaf=self.min_data_in_leaf,
        )

        return self.root_node


# -------- Classification ---------------
def count_equal(y: np.ndarray, label_type: np.ndarray) -> int:
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


def gini_index(y: np.ndarray, classes: np.ndarray) -> float:
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


def weighted_gini(y: np.ndarray, yl: np.ndarray, yr: np.ndarray, classes: np.ndarray) -> float:
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


def information_gain(
    y: np.ndarray, yl: np.ndarray, yr: np.ndarray, classes: np.ndarray, starting_impurity: float
) -> float:
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
    def __init__(self, max_depth=2, min_data_in_leaf=5, random_state=1):
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

    def reused_data(self, y: np.ndarray, classes: np.ndarray) -> float:
        return gini_index(y, classes)

    @staticmethod
    def current_score(
        y: np.ndarray, yl: np.ndarray, yr: np.ndarray, classes: np.ndarray, starting_impurity: float
    ) -> float:
        return information_gain(y, yl, yr, classes, starting_impurity)

    @staticmethod
    def create_leaf(y:np.ndarray, classes:np.ndarray):
        num_points = len(y)
        probabilities = [count_equal(y, c) / num_points for c in classes]
        return Leaf(probabilities)

    @staticmethod
    def get_classes(y:np.ndarray):
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
        node = self.root_node

        # Finds the leaf which X belongs
        while isinstance(node, Node):
            feature = node.splitpoint.feature
            value = node.splitpoint.split_value

            if x[feature] < value:
                node = node.left
            else:
                node = node.right

        # Check if it's a leaf node (instance of Leaf class)
        if isinstance(node, Leaf):
            return node.value
        else:
            # Handle the case where the loop exits for reasons other than reaching a leaf
            raise ValueError("Invalid tree structure encountered.")


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

    def reused_data(self, y: list = [], classes: list = []) -> float:
        return 0

    def current_score(
        y: list, yl: list, yr: list, classes: list = [], starting_impurity: float = 0
    ) -> float:
        return rss(yl) + rss(yr)

    def create_leaf(y, classes=[]):
        return Leaf([np.mean(y)])

    @staticmethod
    def start_score():
        """Return the start score for the minimisation problem."""
        return np.inf
    
    @staticmethod
    def get_classes(y:np.ndarray):
        """Return an empty array since there are no classes in regression"""
        return []

    def predict_one_sample(self, x):
        """Returns prediction for 1 dim array"""
        node = self.root_node

        # Finds the leaf which X belongs
        while isinstance(node, Node):
            feature = node.splitpoint.feature
            value = node.splitpoint.split_value

            if x[feature] < value:
                node = node.left
            else:
                node = node.right

        # Check if it's a leaf node (instance of Leaf class)
        if isinstance(node, Leaf):
            return node.value 
        else:
            # Handle the case where the loop exits for reasons other than reaching a leaf
            raise ValueError("Invalid tree structure encountered.")


    def predict_avg(self, X_set):
        """Returns the predicted avg for a given data set"""

        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        return pred_probs

    def predict(self, X_set):
        """Returns the predicted probs for a given data set"""

        pred_avg = self.predict_avg(X_set)
        return pred_avg



if __name__ == "__main__":
    from src.preprocess.get_data import get_BW_data
    from sklearn.model_selection import train_test_split

    X, y = get_BW_data("/Users/norahallqvist/Code/SIRUS/data/BreastWisconsin.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    splits = cutpoints(X=X_train, q=10)
    tree_model = DecisionTreeClassifier(max_depth=2, min_data_in_leaf=5, random_state=10)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    
    print("mean absolute error: ", mean_absolute_error(y_test, y_pred))

    def print_tree(node, indent="", prefix="Root"):
        if isinstance(node, Node):
            print(indent + f"{prefix} - Feature: {node.splitpoint.feature}, Value: {node.splitpoint.split_value}")
            print_tree(node.left, indent + "  ", "Left")
            print_tree(node.right, indent + "  ", "Right")
        elif isinstance(node, Leaf):
            print(indent + f"{prefix} - Probabilites: {node.value}")

  
    node = tree_model.root_node
    print_tree(node)


