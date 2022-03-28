from algorithm.parameters import params
import numpy as np
from collections import Counter

from representation.individual import Individual
from representation.tree import Tree

def prune_if_else_tree(*args):
    """
    This function simplifies individuals with the if_else grammar (np.where(....,np.where(...,...,...),...)
    The idea is to remove the splits where less than min_patterns are considered, and those with the same output.

    :param args: You can pass just one An individual
    :param min_patterns: The minimal number of patterns that should be considered in the split
    :return: The new individual.
    """
    result = []
    for ind in args:
        if not ind.invalid:
            r_prune_if_else_tree(ind.tree)
            ind.tree.parent = None
            #ind.__init__(genome=None,ind_tree=ind.tree)
            #result.append(ind)
            result.append(Individual(genome=None,ind_tree=ind.tree,map_ind=True))
        else:
            result.append(ind)

    if len(result) == 1:
        return result[0]
    else:
        return result

def generate_leaf(tree, decision):
    productions = params['BNF_GRAMMAR'].rules[tree.root]['choices']
    # no_choices = params['BNF_GRAMMAR'].rules[tree.root]['no_choices']

    # if decision is among productions, chose it
    chosen_prod = [i for i in productions if i['choice'][0]['symbol'] == decision]

    # else, chose the <GE_GENERATE:dataset_target_labels>
    if len(chosen_prod) == 0:
        chosen_prod = [i for i in productions if i['choice'][0]['symbol'] == '<GE_GENERATE:dataset_target_labels>']

    chosen_prod = chosen_prod[0]

    symbol = chosen_prod['choice'][0]
    tree.children.append(Tree(symbol["symbol"], tree))

    if symbol["type"] == "NT":
        generate_leaf(tree.children[-1], decision)



def r_prune_if_else_tree(tree, current_condition=''):
    """
    Recursive function that
    """

    y = params['FITNESS_FUNCTION'].training_exp
    x = params['FITNESS_FUNCTION'].training_in
    min_gini_reduction = params['MIN_GINI_REDUCTION']

    # If the condition leading to this node is not empty, check if there are enough patterns
    if current_condition != '':
        eval_current_condition = eval(current_condition)
        count_current_condition = np.sum(eval_current_condition)

        # If there are not patterns reaching this node, return the sibling node, which produces subsequentially the remove of the parent
        if count_current_condition == 0:
            try:
                if id(tree) == id(tree.parent.children[3]):
                    return tree.parent.children[5], 0, None
                else:
                    return tree.parent.children[3], 0, None
            except:
                print("CURRENT_CONDITION:", current_condition, flush=True)
                raise
    else: # if the condiiton is empty, fill it with True for compatibilty with recursive calls
        current_condition = '(np.array([True for i in range(x.shape[0])]))'
        eval_current_condition = eval(current_condition)
        count_current_condition = len(y)

    try:
        count_classes = Counter(y[eval_current_condition])  # This produces a dict with the frequency for every value
        most_frecuent = max(count_classes, key=count_classes.get)  # This gets the key with the maximal value
    except:
        print('CURRENT_CONDITION (Counter):', current_condition, flush=True)
        raise

    try:
        gini = 1. - np.sum((np.asarray(list(count_classes.values())) / count_current_condition)**2)
    except:
        gini = np.nan

    # If this is not a split node, it is a leaf. Then, check that the decision is the right one
    if tree.children[0].root != 'np.where(':
        # Get the output of the leaf node
        _, output, _, _, \
        _ = tree.children[0].get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(),
                                           [], [])

        try: # The try-except is neccesary for individuals with just a decision (do nothing for them)
            # If the decision is not the most frequent one in the training set, change it
            if output[0] != ('\'' + most_frecuent + '\''):
                tree.children = []
                generate_leaf(tree, '\'' + most_frecuent + '\'')
        except:
            pass

        return tree, count_current_condition, gini

    else: # Main branch of the recursion.
        min_patterns = params['SIMPLIFY_MIN_PATTERNS']
        # Get the condition of this node
        _, this_condition, _, _, \
            _ = tree.children[1].get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(),[],[])
        this_condition = "".join(this_condition)

        # Void condition
        if 'x' not in this_condition:
            if eval(this_condition):
                return tree.children[3],count_current_condition,gini
            else:
                return tree.children[5],count_current_condition, gini

        # If this is not the root node of the tree
        if tree.parent is not None:
            # If there are not enough patterns reaching this node, remove the split and generate
            # a leaf with the most frequent decision
            if count_current_condition < min_patterns and count_current_condition > 0:
                tree.children = []
                generate_leaf(tree, '\'' + most_frecuent + '\'')
                return tree, count_current_condition, gini
            # If there are not any pattern reaching this node, return the sibling, whih
            # produces the subsequent remove of the parent
            elif count_current_condition == 0 or count_current_condition == x.shape[0]:
                try: # No sé bien por qué a veces falla el siguiente tree.parent.children[3] (keyError)
                    if id(tree) == id(tree.parent.children[3]):
                        return tree.parent.children[5], 0, None
                    else:
                        return tree.parent.children[3], 0, None
                except:
                    return tree, count_current_condition, None

        # Recursive call on the left branch
        so_far_condition = current_condition + ' & ' + this_condition
        tree.children[3].parent = tree # En algunas llamadas recursivas, esto no era cierto y daba problemas. No sé por qué
        tree.children[3], count_left, gini_left = r_prune_if_else_tree(tree.children[3], so_far_condition)
        tree.children[3].parent = tree

        # Get the output of the left branch
        _, output_left, _, _, \
        _ = tree.children[3].get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(),
                                           [], [])

        # Recursive call on the right branch
        so_far_condition = current_condition + ' & (~(' + this_condition + '))'
        tree.children[5].parent = tree # En algunas llamadas recursivas, esto no era cierto y daba problemas. No sé por qué
        tree.children[5], count_right, gini_right = r_prune_if_else_tree(tree.children[5], so_far_condition)
        tree.children[5].parent = tree

        # Get the output of the right branch
        _, output_right, _, _, \
        _ = tree.children[5].get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(),
                                           [], [])

        if count_left > 0 and count_right > 0:
            gini_split = (count_left / count_current_condition) * gini_left + \
                         (count_right / count_current_condition) * gini_right
        else:
            gini_split = np.nan

        # If this is not the root node of the tree and the gini reduction is below the threshold
        # replace the node by a decision leaf
        if tree.parent is not None and \
                (np.isnan(gini_split) or ((gini - gini_split) < min_gini_reduction)):
            tree.children = []
            generate_leaf(tree, '\'' + most_frecuent + '\'')
            return tree, count_current_condition, gini

        # In case that the outputs of the left and the right branches are the same,
        # remove this split and return the left node
        if output_left == output_right and tree.parent is not None:
            return tree.children[3], count_current_condition, gini
        else:
            return tree, count_current_condition, gini
