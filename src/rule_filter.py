import re
import copy 
import numpy as np 

#---------- Classes for Parts--------------------

class Condition:
    def __init__(self, feature_idx, operation, split_value):
        self.feature_idx = feature_idx
        self.operation = operation
        self.split_value = split_value

    def print_condition(self):
        print(f"Feature {self.feature_idx} {self.operation} {self.split_value}")

    def reverse_condition(self):
        if self.operation == ">=":
            new_operation = "<"
        elif self.operation == '<':
            new_operation = ">="
        else:
            # Handle the case where self.operation is neither ">=" nor "<"
            raise AssertionError("Invalid operation: {}".format(self.operation))
        return Condition(self.feature_idx, new_operation, self.split_value)

    def __eq__(self, other):
        if isinstance(other, Condition):
            return (
                self.feature_idx == other.feature_idx and
                self.operation == other.operation and
                self.split_value == other.split_value
            )
        return False

class Path:
    def __init__(self, node_path:list):
        self.node_conditions = node_path
    
    def print_path(self):
        path = []
        for condition in self.node_conditions:
            path += [f"Feature {condition.feature_idx} {condition.operation} {condition.split_value}"]
        print(' & '.join(path))

    def __eq__(self, other):
        if isinstance(other, Path):
            for cond1, cond2 in zip(self.node_conditions, other):
                if not (cond1 == cond2):
                    return False
            return True
        return False


class Rule: 
    def __init__(self, node_path:Path, node_pred:list):
        self.node_path = node_path
        self.node_pred_then = node_pred[0]
        self.node_pred_else = node_pred[1]

    
    def print_rule(self):
        path = []
        for condition in self.node_path.node_conditions:
            path += [f"Feature {condition.feature_idx} {condition.operation} {condition.split_value}"]
        print(' & '.join(path) + f" then Y = {self.node_pred_then} else Y = {self.node_pred_else}")


# ---------- Functions -------------

#check if condition 1 -> condition 2
def _implies(condition1, condition2):
    if condition1.feature_idx == condition1.feature_idx:
        if condition1.operation == "<":
            if condition2.operation == "<":
                return condition1.split_value <= condition2.split_value 
            else: 
                return False 
        else:
            if condition2.feature_idx == ">=":
                return condition1.split_value >= condition2.split_value
            else: 
                return False 
    else:
        return False

#check if ondition 1 & condition 2 --> Rule 
def _implies_condition(condition: tuple[Condition, Condition], rule: Rule) -> bool:
    cond1, cond2 = condition
    implied = [any(_implies(cond1, condition) or _implies(cond2, condition) for condition in rule.node_path.node_conditions)]
    return all(implied)

def _feature_space(rules:list, cond1:Condition, cond2:Condition):
    num_rules = len(rules)
    data_matrix = np.empty((4, num_rules + 1), dtype=bool)
    data_matrix[:,0] = np.ones((4))

    reverse_cond1 = cond1.reverse_condition()
    reverse_cond2 = cond2.reverse_condition()

    for col_idx in range(1, (num_rules + 1)):
        cur_rule = rules[col_idx - 1]
        data_matrix[0, col_idx] = _implies_condition((cond1, cond2), cur_rule)
        data_matrix[1, col_idx] = _implies_condition((cond1, reverse_cond2), cur_rule)
        data_matrix[2, col_idx] = _implies_condition((reverse_cond1, cond2), cur_rule)
        data_matrix[3, col_idx] = _implies_condition((reverse_cond1, reverse_cond2), cur_rule)
    
    return data_matrix


def get_left_condition(condition:Condition):
    if condition.operation != "<":
        return condition.reverse_condition()
    return condition

def is_unqiue_condition(condition:Condition, set_of_condition:list):
    for cond_in_set in set_of_condition:
        if cond_in_set.split_value == condition.split_value and cond_in_set.feature_idx == condition.feature_idx:
            return False
    return True 

"""
Return a vector of unique left splits for `rules`.
These splits will be used to form `(A, B)` pairs and generate the feature space.
For example, the pair `x[i, 1] < 32000` (A) and `x[i, 3] < 64` (B) will be used to generate
the feature space `A & B`, `A & !B`, `!A & B`, `!A & !B`.
"""

def _unique_left_conditions(rules:list):
    set_of_conditions = []
    for rule in rules:
        for condition in rule.node_path.node_conditions:
            #get the left version of the condition (if already left remain left)
            condition = get_left_condition(condition)
            if is_unqiue_condition(condition, set_of_conditions): 
                set_of_conditions.append(condition)
    return set_of_conditions


"""
Return whether some rule is either related to `A` or `B` or both.
Here, it is very important to get rid of rules which are about the same feature but different thresholds.
Otherwise, rules will be wrongly classified as linearly dependent in the next step.
"""
def _related_rule(rule:Rule, cond1:Condition, cond2:Condition):
    assert cond1.operation == '<', "Assertion failed: cond1 should be < (something is wrong with _unique_left_conditions)"
    assert cond2.operation == '<', "Assertion failed: cond 2 should be < (something is wrong with _unique_left_conditions)"
    conditions_in_path = rule.node_path.node_conditions
    if len(conditions_in_path) == 1:
        single_condtion = conditions_in_path[0]
        left_condition = get_left_condition(single_condtion)
        return left_condition == cond1 or left_condition == cond2 
    elif len(conditions_in_path) == 2:
        single_condtion_1, single_condtion_2 = conditions_in_path
        left_condtion_1, left_condtion_2 = get_left_condition(single_condtion_1), get_left_condition(single_condtion_2)
        return (left_condtion_1 == cond1 and left_condtion_2 == cond2) or (left_condtion_1 == cond2 and left_condtion_2 == cond1)
    else:
        raise ValueError("Unexpected number of conditions in the path. Expected 1 or 2 conditions, but got {} conditions.".format(len(conditions_in_path)))


"""
Return a vector of booleans with a true for every rule in `rules` that is linearly dependent on a combination of the previous rules.
To find rules for this method, collect all rules containing some feature for each pair of features.
That should be a fairly quick way to find subsets that are easy to process.
"""

def _linearly_dependent(rules: list, cond1:Condition, cond2:Condition):
    data_matrix = _feature_space(rules, cond1, cond2)
    num_rules = len(rules)
    dependent = np.empty(num_rules, dtype=bool)
    atol = 1e-6
    current_rank = np.linalg.matrix_rank(data_matrix[:, 0], tol=atol)

    for i in range(num_rules):

        #adding an additional rank and checking if rank increases or decreases
        new_rank = np.linalg.matrix_rank(data_matrix[:, 0:i+2], tol=atol)
        if current_rank < new_rank:
            dependent[i] = False
            current_rank = new_rank
        else:
            dependent[i] = True

    return dependent
    
"""
Return all unique pairs of elements in `V`.
More formally, return all pairs (v_i, v_j) where i < j.
"""
def _create_unique_pairs(unique_conditions:list):
    num_conditions = len(unique_conditions)
    unique_condition_pairs = []

    for i in range(num_conditions):
        left = unique_conditions[i]
        for j in range(num_conditions):
            if i < j: 
                right = unique_conditions[j]
                unique_condition_pairs.append((left, right))

    return unique_condition_pairs

"""
Return a vector of rules that are not linearly dependent on any other rule.

This is done by considering each pair of splits.
For example, considers the pair `x[i, 1] < 32000` (A) and `x[i, 3] < 64` (B).
Then, for each rule, it checks whether the rule is linearly dependent on the pair.
As soon as a dependent rule is found, it is removed from the set to avoid considering it again.
If we don't do this, we might remove some rule `r` that causes another rule to be linearly
dependent in one related set, but then is removed in another related set.
"""
def filter_linearly_dependent_rules(rules:list):
    unique_conditions = _unique_left_conditions(rules)
    # TODO: need to sort the rules by gap size??
    condition_pairs = _create_unique_pairs(unique_conditions) #get the lower traingle of all combinations
    independent_rules = copy.deepcopy(rules)


    for (cond1, cond2) in condition_pairs:
        independent_rules_idxs = [rule_idx for rule_idx, rule in enumerate(independent_rules) if _related_rule(rule, cond1, cond2)]
        independent_rules_subset = [independent_rules[i] for i in independent_rules_idxs]
        dependent_subset = _linearly_dependent(independent_rules_subset, cond1, cond2) #a list indicating if rule is dependent or not

        assert len(independent_rules_idxs) == len(independent_rules_subset)
        assert len(dependent_subset) == len(independent_rules_subset)

        dependent_indexes = [independent_rules_idxs[i] for i, is_dependent in enumerate(dependent_subset) if is_dependent]
        dependent_indexes.sort() #is this needed?? #TODO: CHECK if this is needed 
        for index in reversed(dependent_indexes):
            independent_rules.pop(index)

    return independent_rules



if __name__ == "__main__":
    
    import json

    def return_formated_rules(rules_dict):
        rules_flatten  = {}
        for tree_idx, tree_rules in rules_dict.items():
            for path, rule in tree_rules.items():
                rules_flatten[path] = rule

        #TODO: check that list is even (assumed that there are 6 rules)

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
        
        return all_rules


    with open('/Users/norahallqvist/Code/SIRUS/results/example_rules_dict.json', 'r') as json_file:
        rules_dict = json.load(json_file)

    all_rules =return_formated_rules(rules_dict)
    for r in all_rules:
        r.print_rule()

    independent_rules = filter_linearly_dependent_rules(all_rules)

    print("results -----")
    for r in independent_rules:
        r.print_rule()


    

    


    

    