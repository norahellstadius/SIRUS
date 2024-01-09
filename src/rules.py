from typing import Callable, Union, List 
import numpy as np 
from tree import Node, Leaf, SplitPoint
from forest import RandomForest

# NOTE: np.less == Left  np.greater_equal == Right 

class SubClause:
    """
    A SubClause is equivalent to a split in a decision tree.
    Each rule contains a clause with one or more SubClause.
    For example, the rule `if X[i, 1] > 3 & X[i, 2] < 4, then ...` contains two subclauses.

    Attributes:
        feature_idx (int): Index of the feature associated with the SubClause.
        feature_name (str): Name of the feature associated with the SubClause.
        direction (Callable[[float, float], bool]): A callable representing the conditional operation.
            This operation takes two float values and returns a boolean result.
        split_value (float): The value used for comparison in the condition.
    """
    def __init__(self, feature_idx: int, feature_name: str,  split_value: float, direction: Callable[[float, float], bool]):
        
        if direction not in [np.greater_equal, np.less]:
            raise AssertionError("Invalid direction: {}".format(self.direction))
        
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.direction = direction
        self.split_value = split_value

    def print_condition(self):
        print(f"Feature {self.feature_idx} {self.direction} {self.split_value}")

    def reverse(self):
        new_direction = np.greater_equal if self.direction == np.less else np.less
        return SubClause(self.feature_idx, self.feature_name, self.split_value, new_direction)

    def __eq__(self, other):
        if isinstance(other, SubClause):
            return (
                self.feature_idx == other.feature_idx and
                self.direction == other.direction and
                self.split_value == other.split_value
            )
        return False

    def string(self):
        sign = "<" if self.direction == np.less else ">="
        return f"X[i, {self.feature_idx}] {sign} {self.split_value}"
    
    def __str__(self):
        return self.string()


def create_subclause(sp:SplitPoint, direction):
    return SubClause(sp.feature, sp.feature_name, sp.split_value, direction)


class Clause:
    """
    A path denotes a conditional on one or more features.
    Each rule contains a path with one or more conditions.
    
    A Path is equivalent to a path in a decision tree.
    For example, the path `X[i, 1] > 3 & X[i, 2] < 4` can be interpreted as a path
    going through two nodes.

    Note that a path can also be a path to a node; not necessarily a leaf.
    """
    def __init__(self, conditions:list):
        self.subclauses = conditions

    @staticmethod
    def parse_str_clause(path_str:str):
        try: 
            comparisons = [c.strip() for c in path_str.split('&')]
            conditions = []

            for c in comparisons:
                direction = np.less if '<' in c else np.greater_equal
                feature_text_end = c.find(']') # Returns index where "]" is located
                if feature_text_end == -1:
                    raise ValueError(f"Couldn't find feature number such as 'X[i, 3]' in '{path_str}'")
                
                feature_text = c[5:feature_text_end] 
                if feature_text.startswith(':'):
                    raise ValueError(f"Can only parse feature numbers such as 'X[i, 3]', "
                                    f"but got 'X[i, {feature_text}]'")
                feature_idx = int(feature_text) #get feature_idx
                start = c.find('<') + 2 if direction == np.less else c.find('>=') + 3
                split_value = float(c[start:])
                feature_name = str(feature_idx)
                conditions.append(SubClause(feature_idx, feature_name, split_value, direction))

        except ValueError as e:
            raise e
        except Exception as e:
            msg = f"Couldn't parse '{path_str}'\n" \
                "Is the syntax correct? Valid examples are:\n" \
                "- parse_clause('X[i, 1] < 1.0 ')\n" \
                "- parse_clause('X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 ')"
            raise ValueError(msg) from e
        
        return conditions

    def __eq__(self, other):
        if isinstance(other, Clause):
            for cond1, cond2 in zip(self.subclauses, other):
                if not (cond1 == cond2):
                    return False
            return True
        return False

    def string(self):
        if not self.subclauses:
            return ""
        
        subclause_strings = [sc.string() for sc in self.subclauses[::-1]]
        return " & ".join(subclause_strings)

    def __str__(self):
        return self.string()



class Rule: 
    """
    A rule is a Path with a then and otherwise predictions. 
    For example, the rule
    `if X[i, 1] > 3 & X[i, 2] < 4, then 5 else 4` is a rule with two
    conditions. The name `otherwise` is used internally instead of `else` since
    `else` is a reserved keyword.
    """

    def __init__(self, path:Clause, then:list, otherwise:list):
        self.clause = path
        self.then = then #in julia its LeafContent = Vector{Float64}
        self.otherwise = otherwise #in julia its LeafContent = Vector{Float64}

    def subclauses(self):
        return self.clause.subclauses

    def reverse(self):
        """
        Return a reversed version of the `rule`.
        Assumes that the rule has only one split (conditions) since two conditions
        cannot be reversed.
        """
        conditions = self.subclauses()
        assert len(conditions) == 1, "Can only reverse a rule with one condition"
        condition = conditions[0]
        path = Clause([condition.reverse()])
        return Rule(path, self.otherwise, self.then)
    
    def left_rule(self):
        conditions = self.subclauses()
        assert len(conditions) == 1, "Can only make a rule left that has one condition"
        condition = conditions[0]
        return self if condition.direction == np.less else self.reverse()

    def __eq__(self, other):
        if isinstance(other, Rule):
            if other.clause == self.clause and other.then == self.then and other.otherwise == self.otherwise:
                return True
            return False 
        
        return False
    
    def __str__(self):
        path_str = self.clause.string()
        return path_str + f" then {self.then} else {self.otherwise} \n"


def then_output(node: Union[Leaf, Node]) -> list:
    """
    Add the leaf contents for the training points which satisfy the
    rule to the `contents` vector.
    """
    stack = [node]
    contents = []

    while stack:
        current_node = stack.pop()

        if isinstance(current_node, Leaf):
            contents.append(current_node.value)
        elif isinstance(current_node, Node):
            stack.append(current_node.right)
            stack.append(current_node.left)

    return contents

def else_output(not_node: Union[Node, Leaf], node: Node) -> list:
    """
    Add the leaf contents for the training points which do not satisfy
    the rule to the `contents` vector.
    """
    stack = [node]
    contents = []

    while stack:
        current_node = stack.pop()

        if current_node == not_node:
            continue

        if isinstance(current_node, Leaf):
            contents.append(current_node.value)
        elif isinstance(current_node, Node):
            stack.append(current_node.right)
            stack.append(current_node.left)

    return contents

def create_rule(root: Node, node: Union[Node, Leaf], subclauses: List[SubClause]) -> Rule:
    clause = Clause(subclauses)
    
    then_output_values = then_output(node)
    then = np.mean(then_output_values)
    
    else_output_values = else_output(node, root)
    otherwise = np.mean(else_output_values)
    
    return Rule(clause, then, otherwise)

def get_tree_rules(node, subclauses=None, rules=None, root=None):
    """"
    Return a all the rules for all paths in a tree.
    This is the rule generation step of SIRUS.
    There will be a path for each node and leaf in the tree.
    In the paper, for a random free Θ, the list of extracted paths is defined as T(Θ, Dn).
    Note that the rules are also created for internal nodes as can be seen from supplement Table 3.
    """
    if rules is None:
        rules = []
    if root is None:
        root = node
    if subclauses is None:
        subclauses = []

    if isinstance(node, Leaf):
        rule = create_rule(root, node, subclauses)
        rules.append(rule)
    else:
        if subclauses:
            rule = create_rule(root, node, subclauses)
            rules.append(rule)

        subclause_L = create_subclause(node.splitpoint, np.less)
        new_subclauses_L = [subclause_L] + subclauses
        get_tree_rules(node.left, new_subclauses_L, rules, root)

        subclause_R = create_subclause(node.splitpoint, np.greater_equal)
        new_subclauses_R = [subclause_R] + subclauses
        get_tree_rules(node.right, new_subclauses_R, rules, root)

    return rules

def get_rules(rf_model: RandomForest):
    rules = []
    for tree in rf_model.trees: 
        tree_rules = get_tree_rules(tree.root_node)
        for rule in tree_rules:
            rules.append(rule)
    return rules



if __name__ == "__main__":
    # path_str = "X[i, 1] < 1.0 & X[i, 2] >= 4.0"
    # result = Clause(path_str)
    # print(result.subclauses)

    from preprocess.get_data import get_BW_data, get_boston_housing
    from sklearn.model_selection import train_test_split
    from quantiles import cutpoints
    from forest import RandomForest, DecisionTreeRegression


    # X, y = get_BW_data("/Users/norahallqvist/Code/SIRUS/data/BreastWisconsin.csv")
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1
    # )
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)

    # splits = cutpoints(X=X_train, q=10)
    # tree_model = RandomForest(type = "Classification", max_depth=2, min_data_in_leaf=5, n_trees = 1, random_state=1)
    # tree_model.fit(X_train, y_train)

    X, y = get_boston_housing("/Users/norahallqvist/Code/SIRUS/data/boston_housing.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    splits = cutpoints(X=X_train, q=10)
    tree_model = RandomForest(type = "Regression", max_depth=2, min_data_in_leaf=5, n_trees = 1, random_state=10)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    rules = get_rules(tree_model)

    for r in rules: 
        print(r)

    

