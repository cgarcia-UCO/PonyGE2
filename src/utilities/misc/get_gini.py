def get_gini(probabilities):
    """
    Get Gini index from a probabilities list using the following expression:
        G = 1 - sum_{0}^{n - 1}(P_i)^2

        Parameters
        ----------
        probabilities: list
            List containing the probabilities.

        Returns
        -------
        Gini index
    """

    assert sum(probabilities) == 1.0, 'The sum of probabilities must be 1.'

    summation = 0
    for p_i in probabilities:
        summation += (p_i ** 2)

    return 1 - summation
