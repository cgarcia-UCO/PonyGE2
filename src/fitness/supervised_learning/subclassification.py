import numpy as np
from algorithm.parameters import params
from fitness.supervised_learning.classification import classification
from utilities.fitness.error_metric import f1_score

from utilities.misc.get_labels_probabilities import get_labels_prob
from utilities.misc.get_gini import get_gini
from utilities.misc.nested_conds_2_rules_list import nested_conds_2_rules_list


class subclassification(classification):
    """Fitness function for subclassification."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Set error metric if it's not set already.
        if params['ERROR_METRIC'] is None:
            params['ERROR_METRIC'] = f1_score

        self.maximise = params['ERROR_METRIC'].maximise

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

        rules = nested_conds_2_rules_list(ind.phenotype)

        aux = x[eval(rules[0])]
        labels = y[eval(rules[0])]

        # Get labels probabilities.
        probabilities = get_labels_prob(labels)

        # TODO: Maybe compute Gini-Simpson index instead of Gini index.
        # Get Gini index.
        G = get_gini(probabilities.values())

        return G
