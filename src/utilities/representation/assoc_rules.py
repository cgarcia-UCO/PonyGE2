import sys
import pandas as pd
import numpy as np
from algorithm.parameters import params, set_params
from utilities.metrics.assoc_rules_metrics import AssocRules_Stats


class Rule:

    def __init__(self):
        self.conditions = []
        self.consequent = None

    def add_condition(self, condition):
        self.conditions.insert(0, condition)

    def set_consequent(self, consequent):
        self.consequent = consequent

    def get_consequent(self):
        return self.consequent

    def __str__(self):
        return '&'.join(self.conditions) + ' => ' + self.consequent

    def get_antecedent(self):
        if len(self.conditions) <= 0:
            return '(np.array([True for i in range(x.shape[0])]))'
        else:
            return '&'.join(self.conditions)

    def __eq__(self, other):
        if isinstance(other, Rule):
            if self.consequent != other.consequent:
                return False
            else:
                # I have not considered comparing the lengths becase a rule might have the same condition
                # multiple times. I think that simplifying the rule at construction could be interesting
                for i in self.conditions:
                    found = len([j for j in other.conditions if j == i])

                    if found <= 0:
                        return False

                for i in other.conditions:
                    found = len([j for j in self.conditions if j == i])

                    if found <= 0:
                        return False

                return True

        else:
            return NotImplemented

class RuleSet:

    def __init__(self):
        self.rules = []

    def filter_duplicates(self):

        removing_indexes = set()

        for index_i, i in enumerate(self.rules):
            for index_j, j in enumerate(self.rules[index_i+1:]):
                if j == i:
                    removing_indexes.add(index_i + index_j)

        init_num_rules = len(self.rules)
        self.rules = [i for index, i in enumerate(self.rules) if index not in removing_indexes]
        assert len(self.rules) == init_num_rules - len(removing_indexes)

    def filter_by_confidence(self, x, y, threshold):
        stats_evaluator = AssocRules_Stats()
        confidence_values = stats_evaluator.compute_confidence(self.rules, x, y)

        self.rules = [i for (index, i), confidence in zip(enumerate(self.rules), confidence_values)
                      if confidence >= threshold]

    def filter_by_consequent(self, target):
        self.rules = [i for i in self.rules if i.get_consequent() == target]

    def fill_rules(self, recursive_list_rules):

        if isinstance(recursive_list_rules, Rule):
            self.rules.append(recursive_list_rules)
        else:
            for i in recursive_list_rules:
                self.fill_rules(i)

    def r_add_conditions(self, set_of_rules, condition):

        if isinstance(set_of_rules, Rule):
            set_of_rules.add_condition(condition)
        else:
            for i in set_of_rules:
                self.r_add_conditions(i, condition)

    def read_tree(self, tree, depth = 0):
        """
        This function reads a tree produced with the if_else grammar (np.where(...,np.where(...,...,...)))
        """

        # If this is not a split node, it should be a leaf
        if tree.children[0].root != 'np.where(':
            current_consequent = self.get_consequent(tree.children[0])
            current_rule = Rule()
            current_rule.set_consequent(current_consequent)

            if depth == 0:
                self.fill_rules([current_rule])

            return [current_rule]

        else: #Main branch of the recursion
            set_conditions = self.get_conditions(tree.children[1])

            # Left branch recursion
            output_left = self.read_tree(tree.children[3], depth + 1)

            for i in set_conditions:
                self.r_add_conditions(output_left, i)

            # Right branch recursion
            output_right = self.read_tree(tree.children[5], depth + 1)

            neg_set_conditions = '(~ (' + '&'.join(set_conditions) + '))'

            self.r_add_conditions(output_right, neg_set_conditions)

            if depth == 0:
                self.fill_rules([output_left, output_right])

            return [output_left, output_right]


    def get_consequent(self, tree):
        if len(tree.children) <= 0:
            return tree.root
        elif len(tree.children) == 1:
            return self.get_consequent(tree.children[0])
        else:
            raise Exception('Weird consequent: ' + str(tree))

    def get_conditions(self, tree):
        set_conds = []
        self.r_get_conditions(tree, set_conds)
        return set_conds


    def r_get_conditions(self, tree, set_conds = None, reading_cond = -1, depth = 0):
        """
        This function produces a list with the conditions in a <cond> subtree.
        Basically, it just outputs the leaves in preorder
        """
        current_cond = ''

        if set_conds == None:
            set_conds = []

        if len(tree.children) > 0:

            for index, i in enumerate(tree.children):
                current_output, reading_cond = self.r_get_conditions(i, set_conds, reading_cond, depth + 1)

                if reading_cond >= 0:
                    current_cond += current_output

            if reading_cond > depth:
                set_conds.append(current_cond)
                current_cond = ''
                reading_cond = -1

        else:
            current_cond = tree.root

            if current_cond != ' & ':
                if reading_cond == -1:
                    reading_cond = depth

        return (current_cond, reading_cond)

if __name__ == "__main__":
    """ Run program """
    set_params(sys.argv[1:])  # exclude the ponyge.py arg itself

    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Obtein and evaluate the rules
    rules = RuleSet()

    for ind in individuals:
        if ind.invalid == False:
            rules.read_tree(ind.tree)

    # Filter the rules
    if 'ASSOC_CONF_FILTER' not in params:
        conf_filtering = 0.7
    else:
        conf_filtering = params['ASSOC_CONF_FILTER']

    if 'FILTEROUT_NOTTARGET_RULES' in params and params['FILTEROUT_NOTTARGET_RULES']:
        rules.filter_by_consequent("'" + params['CLASS_ASSOC_RULES_TARGET'] + "'")

    rules.filter_duplicates()
    rules.filter_by_confidence(params['FITNESS_FUNCTION'].training_in,
                                   params['FITNESS_FUNCTION'].training_exp,conf_filtering)

    # Evaluación
    AssocRules_Stats().print_stats(rules.rules,
                                   params['FITNESS_FUNCTION'].training_in,
                                   params['FITNESS_FUNCTION'].training_exp)
