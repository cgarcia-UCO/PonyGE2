from itertools import chain, combinations

def get_subsets(collection, min_size=2, max_size=8):
    """
    Return all the subsets of values in collection with sizes between min_size and max_size

    In case len(collection) > 20, max_size is reduced to 3

    :param collection: Iterable, for instance, a list of values
    :param min_size: Minimum size of the generated subsets
    :param max_size: Maximum size of the generated subsets
    :return: The set of subsets of values with sizes between min_size and max_size
    """

    if len(collection) > 20:
        max_size = 3

    return chain.from_iterable(combinations(collection, r) for r in range(min_size,max_size)) # range(len(collection) + 1))
