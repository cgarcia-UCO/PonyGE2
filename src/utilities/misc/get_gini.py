def get_gini(probabilities):
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
