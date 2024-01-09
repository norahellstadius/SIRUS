import random
import numpy as np
from quantiles import cutpoints
from tree import DecisionTreeClassifier, DecisionTreeRegression

PARTIAL_SAMPLING_DEFAULT = 0.7
N_TREES_DEFAULT = 1_000
MAX_DEPTH_DEFAULT = 2

# TODO: GENERALISE IF COLUMNS INPUT IS NONE

# class StableForest:
#     def __init__(self, trees) -> None:
#         self.trees = trees  # either a node or a leaf


class RandomForest:
    def __init__(self, type, max_depth=2, min_data_in_leaf=5, n_trees=10, random_state=1) -> None:
        if type not in ["Classification", "Regression"]:
            raise ValueError(
                "Invalid value for self.type. Expected 'Classification' or 'Regression', but got '{}'.".format(
                    self.type
                )
            )

        self.type = type
        self.max_depth = max_depth
        self.seed = random_state
        self.min_data_in_leaf = min_data_in_leaf
        self.n_trees = n_trees
        self.trees = None

    def forest(
        self,
        X,
        y,
        colnms,
        max_split_candidates=None,
        partial_sampling=0.5,
        n_trees=100,
        max_depth=2,
        q=10,
        quantiles=None,
        min_data_in_leaf=5,
    ):
        if max_depth > 2:
            raise ValueError(
                """
                    Tree depth is too high. Rule filtering for a depth above 2 is not implemented.
                    In the original paper, the authors also advise using a depth of no more than 2.
                    """
            )
        if max_depth < 1:
            raise ValueError(f"Minimum tree depth is 1; got {max_depth}")

        n_samples = int(partial_sampling * len(y))

        trees = [None] * n_trees
        seeds = list(range(n_trees))

        # TODO: implement with threads to make this parrallel
        for i in range(n_trees):
            seed = seeds[i]

            # TODO: check if this should be with or without replacement
            # sample with replacement
            random.seed(seed)
            row_idxs = np.random.choice(range(len(y)), size=n_samples, replace=True)
            X_samp = X[row_idxs, :]
            y_samp = y[row_idxs]  # Y NEEDS TO BE ARRAY FOR THIS TO WORK

            if self.type == "Classification":
                tree_model = DecisionTreeClassifier(
                    max_depth, min_data_in_leaf, self.seed
                )
            elif self.type == "Regression":
                tree_model = DecisionTreeRegression(
                    max_depth, min_data_in_leaf, self.seed
                )
            else:
                raise ValueError(
                    "Invalid value for self.type. Expected 'Classification' or 'Regression', but got '{}'.".format(
                        self.type
                    )
                )

            tree_model.fit(
                X=X_samp,
                y=y_samp,
                colnms=colnms,
                max_split_candidates=max_split_candidates,
                q=q,
                quantiles=quantiles,
            )

            trees[i] = tree_model

        self.trees = trees

        return self.trees

    def fit(
        self,
        X,
        y,
        colnms=None,
        max_split_candidates=None,
        partial_sampling=0.5,
        q=10,
        quantiles=None,
    ):
        self.X = X
        self.y = y

        if quantiles is None:
            quantiles = cutpoints(X, q)

        stable_forest = self.forest(
            X=self.X,
            y=self.y,
            colnms=colnms,
            max_split_candidates=max_split_candidates,
            partial_sampling=partial_sampling,
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            q=10,
            quantiles=quantiles,
            min_data_in_leaf=self.min_data_in_leaf,
        )

        return stable_forest

    def predict(self, X_test):
        #get preds for all trees 
        all_preds = np.array([tree.predict(X_test) for tree in self.trees])
        #aggregate pred over the trees
        if self.type == "Classification":
            return np.median(all_preds, axis = 0) # TODO: check paper if this should be mean or median 
        elif self.type == "Regression":
            return np.mean(all_preds, axis = 0)
        else:
            raise ValueError(
                    "Invalid value for self.type. Expected 'Classification' or 'Regression', but got '{}'.".format(
                        self.type
                    ))


if __name__ == "__main__":
    from preprocess.get_data import get_BW_data, get_boston_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error


    # X, y = get_BW_data("/Users/norahallqvist/Code/SIRUS/data/BreastWisconsin.csv")
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1
    # )
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)

    # splits = cutpoints(X=X_train, q=10)
    # tree_model = RandomForest(type = "Classification", max_depth=2, min_data_in_leaf=5, random_state=1)
    # tree_model.fit(X_train, y_train)
    # y_pred = tree_model.predict(X_test)
    # print("mean absolute error: ", np.mean(y_pred == y_test))



    X, y = get_boston_housing("/Users/norahallqvist/Code/SIRUS/data/boston_housing.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    splits = cutpoints(X=X_train, q=10)
    tree_model = RandomForest(type = "Regression", max_depth=2, min_data_in_leaf=5, n_trees = 10, random_state=1)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    print("mean absolute error: ", mean_absolute_error(y_test, y_pred))




    