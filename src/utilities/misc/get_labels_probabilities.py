def get_labels_prob(labels):
    """
    Get the probabilities of each label in labels set.

    :param labels: Series composed by the elements of the dataset that satisfy a set of rules.
    :return probs: Dictionary with the following structure:
        - Key --> Label[i]
        - Value --> Probability associated to the label[i]
    :return n_labels: Dictionary with the following structure:
        - Key --> Label[i]
        - Value --> Amount of i labels
    """

    probs = {}
    n_labels = {}

    increment = 1. / len(labels)

    for i in labels:
        probs[i] = probs.get(i, 0) + increment
        n_labels[i] = n_labels.get(i, 0) + 1

    return probs, n_labels
