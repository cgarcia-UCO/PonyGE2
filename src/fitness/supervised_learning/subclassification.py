from re import L
import numpy as np
from algorithm.parameters import params
from fitness.supervised_learning.classification import classification
#from utilities.fitness.assoc_rules_measures import compute_precision, compute_recall, compute_lift, compute_leverage, compute_conviction
from utilities.fitness.error_metric import f1_score

from utilities.misc.get_labels_probabilities import get_labels_prob
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

        n_records = len(x)

        # FIXME: Hacerlo genérico for rule in rules:...
        aux = x[eval(rules[0])]
        labels = y[eval(rules[0])]

        # Antecedent: Number of True elementes in 'x':
        # FIXME: De vez en cuando se obtiene un error en la siguiente línea.
        # FIXME: Hacerlo genérico for rule in rules:...
        true_values_indexes = [index for index, value in enumerate(
            eval(rules[0]).values) if value == True]
        #true_values = np.count_nonzero(eval(rules[0]).values)

        # Consecuent that satisfies antecedent (intersection A and C):
        # FIXME: Hacerlo genérico for rule in rules:...
        intersec_antec_consec = [index for index, value in enumerate(
            true_values_indexes) if y.values[value] == consecuents[0]]

        # Get labels probabilities.
        probabilities, n_labels = get_labels_prob(labels)

        #rule_support = len(aux) / n_records
        rule_support = len(intersec_antec_consec) / n_records

        # Get Gini index.
        return get_weighted_gini_index(probabilities.values(), n_labels.values())
