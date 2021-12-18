def print_yes_no_prob_stats(stats):
    """
    Get 'yes' and 'no' probabilities of a subset of individuals.

        Parameters
        ----------
            stats: list
                List with the stats of the get_yes_no_prob() function.
                    -stats[0] = Total(Y).
                    -stats[1] = Total(N).
                    -stats[2] = Total.
                    -stats[3] = P(Y).
                    -stats[4] = P(N).

        Returns
        -------
        Nothing.
    """
    print('Yes/No Stats:')
    print(f'\tTotal(Y) = {stats[0]}')
    print(f'\tTotal(N) = {stats[1]}')
    print(f'\tTotal = {stats[2]}')
    print(f'\tP(Y) = {stats[3]}')
    print(f'\tP(N) = {stats[4]}')


def get_yes_no_prob(subset, size):
    """
    Get 'yes' and 'no' probabilities of a subset of individuals.

        Parameters
        ----------
        subset: series
            Series composed by the elements of the dataset that satisfy a set of rules.
        size: integer
            Size of the series.

        Returns
        -------
        prob_yes: float
            Probability of 'Yes' in the subset.
        prob_no: float
            Probability of 'No' in the subset.
    """

    total_yes = subset.values.tolist().count('Si')
    total_no = subset.values.tolist().count('No')

    prob_yes = total_yes / size
    prob_no = total_no / size

    stats = [total_yes, total_no, size, prob_yes, prob_no]
    print_yes_no_prob_stats(stats)

    return [prob_yes, prob_no]
