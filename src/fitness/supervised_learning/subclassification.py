import numpy as np
import pandas as pd

from algorithm.parameters import params
from fitness.supervised_learning.classification import classification
from utilities.fitness.assoc_rules_measures import get_metrics
from utilities.fitness.error_metric import f1_score

from utilities.misc.get_labels_probabilities import get_labels_prob
from utilities.misc.get_gini import *
from utilities.misc.nested_conds_2_rules_list import nested_conds_2_rules_list


class subclassification(classification):
    """Fitness function for subclassification."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Set error metric if it's not set already.
        if params['ERROR_METRIC'] is None:
            params['ERROR_METRIC'] = f1_score

        #self.maximise = params['ERROR_METRIC'].maximise
        self.maximise = False

    def evaluate(self, ind, **kwargs):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :param kwargs: An optional parameter for problems with training/test
        data. Specifies the distribution (i.e. training or test) upon which
        evaluation is to be performed.
        :return: The fitness of the evaluated individual.
        """

        dist = kwargs.get('dist', 'training')

        if dist == "training":
            # Set training datasets.
            x = self.training_in
            y = self.training_exp

        elif dist == "test":
            # Set test datasets.
            x = self.test_in
            y = self.test_exp

        else:
            raise ValueError("Unknown dist: " + dist)

        rules, consecuents = nested_conds_2_rules_list(ind.phenotype)

        if ind.invalid == True or rules == []:
            # Return 'np.nan' if the individual is invalid or there are no rules to
            # analyse.
            return np.nan

        rules_length = len(rules)

        assert rules_length == len(
            consecuents), "Length of 'rules' list and 'consecuents' list must be the same."

        weighted_gini_list = []

        for index in range(rules_length):
            aux = pd.DataFrame()
            labels = []

            try: # If we try to get rows with (-1.0 <= (-3.0 + -0.01)), which can be generated, that would produce an error
                aux = x[eval(rules[index])]
                labels = y[eval(rules[index])]
            except:
                pass

            # Get the list of metrics.
            metrics = get_metrics(
                eval(rules[index]), y, consecuents[index], visualize=False)

            # Get labels probabilities.
            probabilities, n_labels = get_labels_prob(labels)

            # Gini.
            # Note: 'metrics[0]' contains 'antec_support' which will be used as weight.
            weighted_gini_list.append(
                get_weighted_gini_impurity(probabilities.values(), metrics[0]))

        fitness = sum(weighted_gini_list)

        # Get Gini index.
        return fitness
