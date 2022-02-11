from re import L
import numpy as np
import pandas as pd
from algorithm.parameters import params
from fitness.supervised_learning.classification import classification
from utilities.fitness.error_metric import f1_score

from utilities.misc.get_labels_probabilities import get_labels_prob
from utilities.fitness.assoc_rules_measures import get_metrics
from utilities.misc.get_gini import get_weighted_gini_index
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

        rules, consecuents = nested_conds_2_rules_list(ind.phenotype)

        if ind.invalid == True or rules == []:
            # Return 'np.nan' if the individual is invalid or there are no rules to analyse.
            return np.nan

        # FIXME: Hacerlo genérico for rule in rules:...
        aux = x[eval(rules[0])]
        labels = y[eval(rules[0])]

        X = eval(rules[0]).tolist()

        df = pd.DataFrame(y.tolist())
        # Encoder: Para calcular tanto precision como recall.
        y_encoded = df[0].map({'Si': 1, 'No': 0}).tolist()

        get_metrics(X, y_encoded)

        # Get labels probabilities.
        probabilities, n_labels = get_labels_prob(labels)

        # Get Gini index.
        return get_weighted_gini_index(probabilities.values(), n_labels.values())
