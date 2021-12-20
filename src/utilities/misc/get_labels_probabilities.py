def get_labels_prob(labels):
    """
    Get the probabilities of each label in labels set.

    :param labels: Series composed by the elements of the dataset that satisfy a set of rules.
    :return prob: Dictionary with the following structure:
        - Key --> Label[i]
        - Value --> Probability associated to the label[i]
    """

    probs = {}

    increment = 1. / len(labels)

    for i in labels:
        probs[i] = probs.get(i, 0) + increment

    return probs
