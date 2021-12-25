def get_gini_index(probabilities):
    """
    Get Gini index from a probabilities list using the following expression:
        G = 1 - sum_{0}^{n - 1}(P_i)^2

    :param probabilities: List containing the probabilities.
    :return Gini index.
    """

    summation = 0
    for p_i in probabilities:
        summation += (p_i ** 2)

    return 1 - summation


def get_weighted_gini_index(probabilities, weights):
    """
    Get weighted Gini index from a probabilities list using the following expression:
        G = 1 - sum_{0}^{n - 1}(P_i)^2 * (w_i)
    Beeing [w_i = (n_elements of label i / total labels)]

    :param probabilities: Dictionary values containing the probabilities.
    :param weights: Dictionary values containing the amount of labels obtained.
    :return weighted Gini index.
    """

    assert len(probabilities) == len(
        weights), 'Length of probabilities and the amount of labels must be the same.'

    summation = 0
    p = list(probabilities)
    w = list(weights)
    total = sum(w)

    for i in range(len(probabilities)):
        summation += (p[i] ** 2) * (w[i] / total)

    return 1 - summation
